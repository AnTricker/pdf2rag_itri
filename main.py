from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Optional

from local_rag.adapters import (
    OllamaChatBackend,
    OllamaCorpusProfileBackend,
    OllamaImageReviewBackend,
    PypdfTextExtractor,
    SentenceTransformerEmbeddingBackend,
    Q3QueryEmbeddingBackend,
    UltralyticsImagePipeline,
)
from local_rag.application import ImageReviewIncompleteError, PdfRagApplication
from local_rag.config import AppConfig
from local_rag.doctor import run_doctor
from local_rag.q3_importer import Q3Importer
from local_rag.q3_index import Q3KnowledgeIndex
from local_rag.serve_web import create_app


BASE_DIR = Path(__file__).resolve().parent


def read_prompt(name: str) -> str:
    return (BASE_DIR / "prompts" / name).read_text(encoding="utf-8")


def embedding_backend(config: AppConfig, *, q3: bool = False) -> SentenceTransformerEmbeddingBackend:
    backend = Q3QueryEmbeddingBackend if q3 else SentenceTransformerEmbeddingBackend
    return backend(
        config.embedding_model,
        batch_size=config.embedding_batch_size,
        reserved_tokens=config.embedding_reserved_tokens,
        device=config.embedding_device,
        dtype=config.embedding_dtype,
        attention=config.embedding_attention,
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
    parser.add_argument("--serveq3", metavar="KB_NAME")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor")

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--input", type=Path)
    ingest.add_argument("--mode", choices=("text", "image", "multi"))

    review = subparsers.add_parser("review")
    review.add_argument("--output", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--output", required=True)
    build.add_argument("--review", required=True, type=int)

    qwen_import = subparsers.add_parser("q3Importer")
    qwen_import.add_argument("--input", required=True, action="append", type=Path)
    qwen_import.add_argument("--output", required=True)
    qwen_import.add_argument("--name")
    qwen_import.add_argument("--profile", type=Path)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.serveq3 and args.command:
        parser.error("--serveq3 cannot be combined with a subcommand")
    if not args.serveq3 and not args.command:
        parser.error("a subcommand or --serveq3 KB_NAME is required")
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

    if args.command == "q3Importer":
        build_path = Q3Importer(config).import_collections(
            args.input,
            output_id=args.output,
            collection_name=args.name or args.output,
            profile_path=args.profile,
        )
        print(build_path)
        return 0

    if args.command == "serve" or args.serveq3:
        if not config.session_secret:
            raise ValueError("LOCAL_RAG_SESSION_SECRET is required for serve")
        q3_index = None
        if args.serveq3:
            logging.basicConfig(level=logging.INFO)
            q3_index = Q3KnowledgeIndex.load(
                config.q3_root or config.base_dir / "runtime", args.serveq3,
            )
            output_root = q3_index.output_root
            build_root = None
        else:
            output_root = PdfRagApplication.resolve_configured_output(
                config.index_root, args.output
            )
            build_root = PdfRagApplication.latest_build(output_root)
        embedding = embedding_backend(config, q3=q3_index is not None)
        if q3_index is not None:
            q3_index.validate_query_model(embedding)
        chat = OllamaChatBackend(
            config,
            splitter_prompt=read_prompt("q3_query_planner_v1.txt" if q3_index is not None else "query_planner_v1.txt"),
            admission_prompt=read_prompt("query_admission_v2.txt"),
            query_builder_prompt=read_prompt("retrieval_query_builder_v2.txt"),
            evidence_draft_prompt=read_prompt("evidence_draft_v1.txt"),
            final_answer_prompt=read_prompt("grounded_answer_v1.txt"),
        )
        app = None
        try:
            app = create_app(
                config=config,
                embedding_backend=embedding,
                chat_backend=chat,
                build_root=build_root,
                output_root=output_root,
                index=q3_index,
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
