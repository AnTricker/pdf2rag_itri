from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _relative_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean setting: {value}")


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    index_root: Path
    ingest_mode: str = "multi"
    profile_max_records: int = 40
    input_pdf: Optional[Path] = None
    log_root: Optional[Path] = None
    chunk_max_tokens: int = 384
    chunk_overlap_tokens: int = 48
    poppler_path: Optional[Path] = None
    render_dpi: int = 180
    detector_backend: str = "rtdetr"
    detector_model: Optional[Path] = None
    detector_iou: float = 0.7
    min_confidence: float = 0.6
    vlm_backend: str = "ollama"
    vlm_url: str = "http://localhost:11434"
    vlm_model: str = ""
    vlm_timeout_seconds: int = 120
    embedding_model: str = ""
    embedding_backend: str = "sentence_transformers_local"
    embedding_batch_size: int = 16
    embedding_reserved_tokens: int = 8
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.30
    retrieval_context_max_records: int = 8
    llm_url: str = "http://localhost:11434"
    llm_backend: str = "ollama"
    llm_model: str = ""
    llm_timeout_seconds: int = 120
    llm_context_tokens: int = 32768
    answer_input_budget_tokens: int = 24576
    web_host: str = "127.0.0.1"
    web_port: int = 8000
    web_threads: int = 8
    session_secret: str = ''
    session_ttl_seconds: int = 3600
    job_retention_seconds: int = 3600
    upload_max_files: int = 3
    upload_max_file_bytes: int = 5 * 1024 * 1024
    upload_max_total_bytes: int = 10 * 1024 * 1024
    upload_max_pixels: int = 20_000_000
    history_max_turns: int = 12
    history_max_tokens: int = 4096
    preprocessor_prompt_version: str = "question-preprocessor-v1"
    preprocessor_max_retries: int = 1
    answer_max_retries: int = 1
    session_log_enabled: bool = False
    session_log_root: Optional[Path] = None
    markdown_enabled: bool = True

    @classmethod
    def from_env(cls, base_dir: Path) -> "AppConfig":
        base = base_dir.resolve()
        load_dotenv(base / ".env", override=False)
        get = os.environ.get
        input_value = get("LOCAL_RAG_INPUT_PDF", "").strip()
        detector_value = get("LOCAL_RAG_DETECTOR_MODEL", "").strip()
        embedding_value = get("LOCAL_RAG_EMBEDDING_MODEL", "").strip()
        llm_model_value = get("LOCAL_RAG_LLM_MODEL", "").strip()
        vlm_model_value = get("LOCAL_RAG_VLM_MODEL", "").strip() or llm_model_value
        poppler_value = get("LOCAL_RAG_POPPLER_PATH", "").strip()

        config = cls(
            llm_context_tokens=int(get('LOCAL_RAG_LLM_CONTEXT_TOKENS', '32768')),
            answer_input_budget_tokens=int(get('LOCAL_RAG_ANSWER_INPUT_BUDGET_TOKENS', '24576')),
            web_threads=int(get('LOCAL_RAG_WEB_THREADS', '8')),
            session_secret=get('LOCAL_RAG_SESSION_SECRET', ''),
            job_retention_seconds=int(get('LOCAL_RAG_JOB_RETENTION_SECONDS', '3600')),
            upload_max_files=int(get('LOCAL_RAG_UPLOAD_MAX_FILES', '3')),
            upload_max_file_bytes=int(get('LOCAL_RAG_UPLOAD_MAX_FILE_BYTES', str(5 * 1024 * 1024))),
            upload_max_total_bytes=int(get('LOCAL_RAG_UPLOAD_MAX_TOTAL_BYTES', str(10 * 1024 * 1024))),
            upload_max_pixels=int(get('LOCAL_RAG_UPLOAD_MAX_PIXELS', '20000000')),
            base_dir=base,
            index_root=_relative_path(base, get("LOCAL_RAG_INDEX_ROOT", "runtime/outputs")),
            ingest_mode=get("LOCAL_RAG_INGEST_MODE", "multi").strip().lower(),

            profile_max_records=int(get("LOCAL_RAG_PROFILE_MAX_RECORDS", "40")),
            input_pdf=_relative_path(base, input_value) if input_value else None,
            log_root=_relative_path(base, get("LOCAL_RAG_LOG_ROOT", "runtime/logs")),
            chunk_max_tokens=int(get("LOCAL_RAG_CHUNK_SIZE_TOKENS", "384")),
            chunk_overlap_tokens=int(get("LOCAL_RAG_CHUNK_OVERLAP_TOKENS", "48")),
            poppler_path=_relative_path(base, poppler_value) if poppler_value else None,
            render_dpi=int(get("LOCAL_RAG_RENDER_DPI", "180")),
            detector_backend=get("LOCAL_RAG_DETECTOR_BACKEND", "rtdetr").strip().lower(),
            detector_model=_relative_path(base, detector_value) if detector_value else None,
            detector_iou=float(get("LOCAL_RAG_DETECTOR_IOU", "0.7")),
            min_confidence=float(get("LOCAL_RAG_MIN_CONFIDENCE", "0.6")),
            vlm_backend=get("LOCAL_RAG_VLM_BACKEND", "ollama").strip().lower(),
            vlm_url=get("LOCAL_RAG_VLM_URL", "http://localhost:11434").rstrip("/"),
            vlm_model=vlm_model_value,
            vlm_timeout_seconds=int(get("LOCAL_RAG_VLM_TIMEOUT_SECONDS", "120")),
            embedding_model=str(_relative_path(base, embedding_value)) if embedding_value else "",
            embedding_backend=get("LOCAL_RAG_EMBEDDING_BACKEND", "sentence_transformers_local").strip().lower(),
            embedding_batch_size=int(get("LOCAL_RAG_EMBEDDING_BATCH_SIZE", "16")),
            embedding_reserved_tokens=int(get("LOCAL_RAG_EMBEDDING_RESERVED_TOKENS", "8")),
            retrieval_top_k=int(get("LOCAL_RAG_RETRIEVAL_TOP_K", "5")),
            retrieval_min_score=float(get("LOCAL_RAG_RETRIEVAL_MIN_SCORE", "0.30")),
            retrieval_context_max_records=int(
                get("LOCAL_RAG_RETRIEVAL_CONTEXT_MAX_RECORDS", "8")
            ),
            llm_url=get("LOCAL_RAG_LLM_URL", "http://localhost:11434").rstrip("/"),
            llm_backend=get("LOCAL_RAG_LLM_BACKEND", "ollama").strip().lower(),
            llm_model=llm_model_value,
            llm_timeout_seconds=int(get("LOCAL_RAG_LLM_TIMEOUT_SECONDS", "120")),
            web_host=get("LOCAL_RAG_WEB_HOST", "127.0.0.1"),
            web_port=int(get("LOCAL_RAG_WEB_PORT", "8000")),
            session_ttl_seconds=int(get("LOCAL_RAG_SESSION_TTL_SECONDS", "3600")),
            history_max_turns=int(get("LOCAL_RAG_HISTORY_MAX_TURNS", "12")),
            history_max_tokens=int(get("LOCAL_RAG_HISTORY_MAX_TOKENS", "4096")),
            preprocessor_prompt_version=get(
                "LOCAL_RAG_PREPROCESSOR_PROMPT_VERSION",
                "question-preprocessor-v1",
            ).strip(),
            preprocessor_max_retries=int(
                get("LOCAL_RAG_PREPROCESSOR_MAX_RETRIES", "1")
            ),
            answer_max_retries=int(get("LOCAL_RAG_ANSWER_MAX_RETRIES", "1")),
            session_log_enabled=_boolean(
                get("LOCAL_RAG_SESSION_LOG_ENABLED", "true")
            ),
            session_log_root=_relative_path(
                base,
                get(
                    "LOCAL_RAG_SESSION_LOG_ROOT",
                    "runtime/logs/chat_sessions",
                ),
            ),
            markdown_enabled=_boolean(get("LOCAL_RAG_MARKDOWN_ENABLED", "true")),
        )
        config.validate()
        return config

    @classmethod
    def for_test(cls, *, index_root: Path) -> "AppConfig":
        return cls(base_dir=index_root.parent.resolve(), index_root=index_root.resolve())

    def validate(self) -> None:
        if self.llm_context_tokens <= 0:
            raise ValueError('llm_context_tokens must be positive')
        if not 0 < self.answer_input_budget_tokens < self.llm_context_tokens:
            raise ValueError('answer input budget must be smaller than LLM context')
        if self.web_threads <= 0 or self.job_retention_seconds <= 0:
            raise ValueError('web threads and job retention must be positive')
        if (self.upload_max_files <= 0 or self.upload_max_file_bytes <= 0 or self.upload_max_total_bytes < self.upload_max_file_bytes or self.upload_max_pixels <= 0):
            raise ValueError('invalid upload limits')
        if self.chunk_max_tokens <= 0:
            raise ValueError("chunk size must be positive")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_max_tokens:
            raise ValueError("chunk overlap must be smaller than chunk size")
        if self.detector_backend not in {"yolo", "rtdetr"}:
            raise ValueError("detector backend must be yolo or rtdetr")
        if self.ingest_mode not in {"text", "image", "multi"}:
            raise ValueError("ingest mode must be text, image or multi")
        if self.profile_max_records <= 0:
            raise ValueError("profile_max_records must be positive")
        if self.web_host not in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}:
            raise ValueError("unsupported web host")
        if self.retrieval_top_k <= 0 or not -1 <= self.retrieval_min_score <= 1:
            raise ValueError("invalid retrieval settings")
        if self.retrieval_context_max_records <= 0:
            raise ValueError("retrieval_context_max_records must be positive")
        if self.history_max_turns <= 0:
            raise ValueError("history_max_turns must be positive")
        if self.history_max_tokens <= 0:
            raise ValueError("history_max_tokens must be positive")
        if self.preprocessor_max_retries < 0:
            raise ValueError("preprocessor_max_retries cannot be negative")
        if self.answer_max_retries < 0:
            raise ValueError("answer_max_retries cannot be negative")
        if not self.preprocessor_prompt_version:
            raise ValueError("preprocessor_prompt_version cannot be blank")
        if self.embedding_backend != "sentence_transformers_local":
            raise ValueError("unsupported embedding backend")
        if self.llm_backend != "ollama" or self.vlm_backend != "ollama":
            raise ValueError("unsupported local model backend")

    def with_chat(self, *, retrieval_top_k: int, retrieval_min_score: float) -> "AppConfig":
        return replace(self, retrieval_top_k=retrieval_top_k, retrieval_min_score=retrieval_min_score)

    def with_text(self, *, chunk_max_tokens: int, chunk_overlap_tokens: int) -> "AppConfig":
        candidate = replace(
            self,
            chunk_max_tokens=chunk_max_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )
        candidate.validate()
        return candidate
