from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from local_rag.adapters import (
    OllamaCaptionBackend,
    OllamaChatBackend,
    OllamaCorpusProfileBackend,
    PypdfTextExtractor,
    SentenceTransformerEmbeddingBackend,
    UltralyticsImagePipeline,
)
from local_rag.application import PdfRagApplication
from local_rag.cli import run_cli
from local_rag.config import AppConfig
from local_rag.doctor import run_doctor
from local_rag.web import create_app


BASE_DIR = Path(__file__).resolve().parent


def read_prompt(name: str) -> str:
    return (BASE_DIR / "prompts" / name).read_text(encoding="utf-8")


def embedding_backend(config: AppConfig) -> SentenceTransformerEmbeddingBackend:
    return SentenceTransformerEmbeddingBackend(
        config.embedding_model,
        batch_size=config.embedding_batch_size,
        reserved_tokens=config.embedding_reserved_tokens,
    )


def ingestion_application(config: AppConfig, mode: str) -> PdfRagApplication:
    embedding = embedding_backend(config)
    image_enabled = mode in {"image", "multi"}
    image_pipeline = (
        UltralyticsImagePipeline(config)
        if image_enabled and config.detector_model
        else None
    )
    caption = None
    if image_enabled and image_pipeline is not None and config.vlm_model:
        caption = OllamaCaptionBackend(config, read_prompt("image_caption_v1.txt"))
    profile = None
    if not (
        config.corpus_profile_override
        and config.corpus_profile_override.is_file()
    ):
        profile = OllamaCorpusProfileBackend(
            config,
            read_prompt("corpus_profile_v1.txt"),
        )
    return PdfRagApplication(
        config=config,
        text_extractor=PypdfTextExtractor(),
        embedding_backend=embedding,
        image_pipeline=image_pipeline,
        caption_backend=caption,
        profile_backend=profile,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Portable local PDF RAG MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--input", type=Path)
    ingest.add_argument("--mode", choices=("text", "image", "multi"))
    subparsers.add_parser("serve")
    args = parser.parse_args(argv)
    config = AppConfig.from_env(BASE_DIR)

    if args.command == "doctor":
        ready, checks = run_doctor(config)
        print(json.dumps({"ready": ready, "checks": checks}, ensure_ascii=False, indent=2))
        return 0 if ready else 1
    if args.command == "ingest":
        source = args.input or config.input_pdf
        if source is None:
            parser.error("ingest requires --input or LOCAL_RAG_INPUT_PDF")
        selected_mode = args.mode or config.ingest_mode
        ingest_args = [
            "ingest",
            "--input",
            str(source),
            "--mode",
            selected_mode,
        ]
        return run_cli(
            ingest_args,
            application_factory=lambda: ingestion_application(
                config, selected_mode
            ),
        )
    if args.command == "serve":
        embedding = embedding_backend(config)
        chat = OllamaChatBackend(
            config,
            splitter_prompt=read_prompt("question_splitter_v2.txt"),
            admission_prompt=read_prompt("query_admission_v2.txt"),
            query_builder_prompt=read_prompt("retrieval_query_builder_v2.txt"),
            evidence_draft_prompt=read_prompt("evidence_draft_v1.txt"),
            final_answer_prompt=read_prompt("final_answer_v2.txt"),
        )
        app = create_app(config=config, embedding_backend=embedding, chat_backend=chat)
        app.run(host=config.web_host, port=config.web_port, debug=False)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
