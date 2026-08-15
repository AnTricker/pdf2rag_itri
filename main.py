from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

from local_rag.adapters import (
    OllamaChatBackend,
    OllamaCorpusProfileBackend,
    OllamaImageReviewBackend,
    PypdfTextExtractor,
    SentenceTransformerEmbeddingBackend,
    UltralyticsImagePipeline,
)
from local_rag.application import ImageReviewIncompleteError, PdfRagApplication
from local_rag.config import AppConfig
from local_rag.doctor import run_doctor
from local_rag.serve_web import create_app


BASE_DIR = Path(__file__).resolve().parent


def read_prompt(name: str) -> str:
    return (BASE_DIR / "prompts" / name).read_text(encoding="utf-8")


def embedding_backend(config: AppConfig) -> SentenceTransformerEmbeddingBackend:
    return SentenceTransformerEmbeddingBackend(
        config.embedding_model,
        batch_size=config.embedding_batch_size,
        reserved_tokens=config.embedding_reserved_tokens,
    )


def image_review_backend(config: AppConfig) -> OllamaImageReviewBackend:
    return OllamaImageReviewBackend(
        config,
        classifier_prompt=read_prompt("image_classifier_v1.txt"),
        table_extractor_prompt=read_prompt("table_extractor_v1.txt"),
        table_summary_prompt=read_prompt("table_summary_v1.txt"),
        figure_extractor_prompt=read_prompt("figure_extractor_v1.txt"),
    )


def report_image_review_failure(error: ImageReviewIncompleteError) -> None:
    for failure in error.failures:
        detail = " ".join(str(failure.error).split())
        print(
            f"[VLM ERROR] {failure.artifact_path} | {failure.stage} | "
            f"{type(failure.error).__name__}: {detail}",
            file=sys.stderr,
        )
    print(
        f"[VLM SUMMARY] operation={error.operation} "
        f"complete={error.image_complete} failed={len(error.failures)} "
        f"pending={error.image_pending} total={error.image_total}",
        file=sys.stderr,
    )
    print(f"[VLM OUTPUT] {error.output_root}", file=sys.stderr)
    print(f"[VLM REVIEW] {error.review_path}", file=sys.stderr)


def pipeline_application(
    config: AppConfig,
    *,
    include_image_pipeline: bool,
    include_image_review: bool,
    include_profile: bool,
) -> PdfRagApplication:
    return PdfRagApplication(
        config=config,
        text_extractor=PypdfTextExtractor(),
        embedding_backend=embedding_backend(config),
        image_pipeline=(
            UltralyticsImagePipeline(config) if include_image_pipeline else None
        ),
        image_review_backend=(
            image_review_backend(config) if include_image_review else None
        ),
        profile_backend=(
            OllamaCorpusProfileBackend(
                config,
                read_prompt("corpus_profile_v1.txt"),
            )
            if include_profile
            else None
        ),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reviewable local PDF RAG pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--input", type=Path)
    ingest.add_argument("--mode", choices=("text", "image", "multi"))

    review = subparsers.add_parser("review")
    review.add_argument("--output", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--output", required=True)
    build.add_argument("--review", required=True, type=int)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    config = AppConfig.from_env(BASE_DIR)

    if args.command == "doctor":
        ready, checks = run_doctor(config)
        print(
            json.dumps(
                {"ready": ready, "checks": checks},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if ready else 1

    if args.command == "ingest":
        source = args.input or config.input_pdf
        if source is None:
            parser.error("ingest requires --input or LOCAL_RAG_INPUT_PDF")
        mode = args.mode or config.ingest_mode
        app = pipeline_application(
            config,
            include_image_pipeline=mode in {"image", "multi"},
            include_image_review=mode in {"image", "multi"},
            include_profile=False,
        )
        try:
            output_root = app.ingest(source, mode=mode)
        except ImageReviewIncompleteError as error:
            report_image_review_failure(error)
            return 1
        print(output_root.name)
        return 0

    if args.command == "review":
        app = pipeline_application(
            config,
            include_image_pipeline=False,
            include_image_review=True,
            include_profile=False,
        )
        try:
            review_path = app.review(args.output)
        except ImageReviewIncompleteError as error:
            report_image_review_failure(error)
            return 1
        print(review_path)
        return 0

    if args.command == "build":
        app = pipeline_application(
            config,
            include_image_pipeline=False,
            include_image_review=False,
            include_profile=True,
        )
        build_path = app.build(args.output, args.review)
        print(build_path)
        return 0

    if args.command == "serve":
        if not config.session_secret:
            raise ValueError("LOCAL_RAG_SESSION_SECRET is required for serve")
        embedding = embedding_backend(config)
        chat = OllamaChatBackend(
            config,
            splitter_prompt=read_prompt("query_planner_v1.txt"),
            admission_prompt=read_prompt("query_admission_v2.txt"),
            query_builder_prompt=read_prompt("retrieval_query_builder_v2.txt"),
            evidence_draft_prompt=read_prompt("evidence_draft_v1.txt"),
            final_answer_prompt=read_prompt("grounded_answer_v1.txt"),
        )
        resolver = PdfRagApplication(
            config=config,
            text_extractor=PypdfTextExtractor(),
            embedding_backend=embedding,
        )
        output_root = resolver.resolve_output(args.output)
        build_root = resolver.latest_build(output_root)
        warm_metrics = chat.warm()
        print(f"[LLM WARM] model={chat.model} metrics={warm_metrics}", file=sys.stderr)
        app = None
        try:
            app = create_app(
                config=config,
                embedding_backend=embedding,
                chat_backend=chat,
                build_root=build_root,
                output_root=output_root,
            )
            from waitress import serve

            serve(app, host=config.web_host, port=config.web_port,
                  threads=config.web_threads)
        finally:
            if app is not None:
                app.extensions["chat_job_manager"].shutdown()
            try:
                chat.unload()
            except Exception as error:
                print(f"[LLM UNLOAD WARNING] {error}", file=sys.stderr)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
