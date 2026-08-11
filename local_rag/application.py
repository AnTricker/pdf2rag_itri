from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Optional, Protocol
from uuid import uuid4

import numpy as np

from .config import AppConfig
from .models import (
    BuildReport,
    CorpusProfile,
    EmbeddingMetadata,
    ExtractedPage,
    ImageCandidate,
    ImageCaption,
    ImageMetadata,
    KnowledgeRecord,
    ProcessingMetadata,
    SnapshotManifest,
    SourceMetadata,
)
from .snapshot import SnapshotWriter


class TextExtractor(Protocol):
    name: str

    def extract(self, pdf_path: Path) -> list[ExtractedPage]: ...


class EmbeddingBackend(Protocol):
    name: str
    model_name: str
    dimension: int

    def count(self, text: str) -> int: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


class ImagePipeline(Protocol):
    name: str

    def extract(self, pdf_path: Path, artifact_root: Path) -> list[ImageCandidate]: ...


class CaptionBackend(Protocol):
    name: str
    model_name: str
    prompt_version: str

    def describe(self, crop_path: Path) -> ImageCaption: ...


class CorpusProfileBackend(Protocol):
    def build(
        self,
        *,
        document_name: str,
        document_id: str,
        records: list[KnowledgeRecord],
    ) -> CorpusProfile: ...


class PdfRagApplication:
    def __init__(
        self,
        *,
        config: AppConfig,
        text_extractor: TextExtractor,
        embedding_backend: EmbeddingBackend,
        image_pipeline: Optional[ImagePipeline] = None,
        caption_backend: Optional[CaptionBackend] = None,
        profile_backend: Optional[CorpusProfileBackend] = None,
    ) -> None:
        self.config = config
        self.text_extractor = text_extractor
        self.embedding_backend = embedding_backend
        self.image_pipeline = image_pipeline
        self.caption_backend = caption_backend
        self.profile_backend = profile_backend

    def ingest(self, input_pdf: Path, *, mode: Optional[str] = None) -> Path:
        source = input_pdf.resolve(strict=True)
        selected_mode = (mode or self.config.ingest_mode).strip().lower()
        if selected_mode not in {"text", "image", "multi"}:
            raise ValueError("ingest mode must be text, image or multi")
        document_hash = self._sha256(source)
        records: list[KnowledgeRecord] = []
        warnings: list[str] = []
        stage_statuses = {
            "text_extraction": "disabled",
            "image_processing": "disabled",
        }
        if selected_mode in {"text", "multi"}:
            try:
                for page in self.text_extractor.extract(source):
                    for ordinal, content in enumerate(
                        self._chunk_text(page.text), start=1
                    ):
                        records.append(
                            self._text_record(
                                source,
                                document_hash,
                                page.page_number,
                                ordinal,
                                content,
                            )
                        )
                stage_statuses["text_extraction"] = "complete"
            except Exception:
                stage_statuses["text_extraction"] = "failed"
                if selected_mode == "text":
                    raise
                warnings.append("text pipeline failed")
        crop_artifacts: list[Path] = []
        work_root = self.config.index_root / f".ingest-{uuid4().hex}"
        try:
            if selected_mode in {"image", "multi"} and self.image_pipeline is None:
                if selected_mode == "image":
                    raise ValueError("image mode requires an image pipeline")
                warnings.append("image pipeline disabled")
            if selected_mode in {"image", "multi"} and self.image_pipeline is not None:
                try:
                    candidates = self.image_pipeline.extract(source, work_root)
                    crop_artifacts.extend(
                        candidate.crop_path for candidate in candidates
                    )
                    for candidate in candidates:
                        if self.caption_backend is None:
                            warnings.append(
                                f"caption unavailable: {candidate.artifact_name}"
                            )
                            continue
                        try:
                            caption = self.caption_backend.describe(
                                candidate.crop_path
                            )
                        except Exception:
                            warnings.append(
                                f"caption failed: {candidate.artifact_name}"
                            )
                            continue
                        records.extend(
                            self._image_records(
                                source, document_hash, candidate, caption
                            )
                        )
                    stage_statuses["image_processing"] = "complete"
                except Exception:
                    stage_statuses["image_processing"] = "failed"
                    if selected_mode == "image":
                        raise
                    warnings.append("image pipeline failed")
            if not records:
                raise ValueError("ingestion did not yield any searchable records")
            profile = self._build_corpus_profile(source, document_hash, records)
            return self._write_snapshot(
                source=source,
                records=records,
                crop_artifacts=crop_artifacts,
                warnings=warnings,
                mode=selected_mode,
                corpus_profile=profile,
                stage_statuses=stage_statuses,
            )
        finally:
            shutil.rmtree(work_root, ignore_errors=True)

    def _write_snapshot(
        self,
        *,
        source: Path,
        records: list[KnowledgeRecord],
        crop_artifacts: list[Path],
        warnings: list[str],
        mode: str,
        corpus_profile: CorpusProfile,
        stage_statuses: dict[str, str],
    ) -> Path:
        vectors = np.asarray(
            self.embedding_backend.embed([record.content for record in records]),
            dtype=np.float32,
        )
        manifest = SnapshotManifest(
            document_sha256=records[0].document_id,
            record_count=len(records),
            vector_count=len(vectors),
            vector_dimension=self.embedding_backend.dimension,
            embedding=EmbeddingMetadata(
                backend=self.embedding_backend.name,
                model=self.embedding_backend.model_name,
            ),
            chunk_max_tokens=self.config.chunk_max_tokens,
            chunk_overlap_tokens=self.config.chunk_overlap_tokens,
            prompt_versions=(
                {"image_caption": self.caption_backend.prompt_version}
                if self.caption_backend is not None
                else {}
            ),
        )
        report = BuildReport(
            source_document=source.name,
            document_id=records[0].document_id,
            text_record_count=sum(record.modality == "text" for record in records),
            image_record_count=sum(record.modality == "image" for record in records),
            embedded_count=len(records),
            warnings=warnings,
            stage_statuses={
                **stage_statuses,
                "embedding": "complete",
                "snapshot_validation": "complete",
            },
            artifact_locations={
                "records": "records.jsonl",
                "pretty_records": "records.pretty.json",
                "corpus_profile": "corpus_profile.json",
                "embeddings": "embeddings.npy",
                "source_document": f"document/{source.name}",
            },
        )
        return SnapshotWriter(self.config.index_root).write(
            source_pdf=source,
            records=records,
            embeddings=vectors,
            manifest=manifest,
            report=report,
            crop_artifacts=crop_artifacts,
            corpus_profile=corpus_profile,
        )

    def _build_corpus_profile(
        self,
        source: Path,
        document_hash: str,
        records: list[KnowledgeRecord],
    ) -> CorpusProfile:
        override = self.config.corpus_profile_override
        if override is not None and override.is_file():
            profile = CorpusProfile.model_validate_json(
                override.read_text(encoding="utf-8")
            )
        else:
            if self.profile_backend is None:
                raise ValueError("corpus profile backend or override is required")
            maximum = self.config.profile_max_records
            if len(records) <= maximum:
                samples = records
            elif maximum == 1:
                samples = [records[0]]
            else:
                indices = {
                    round(position * (len(records) - 1) / (maximum - 1))
                    for position in range(maximum)
                }
                samples = [records[index] for index in sorted(indices)]
            profile = self.profile_backend.build(
                document_name=source.name,
                document_id=document_hash,
                records=samples,
            )
        if profile.document_id != document_hash or profile.document_name != source.name:
            raise ValueError("corpus profile does not match the source document")
        return profile

    def _text_record(
        self,
        source: Path,
        document_hash: str,
        page_number: int,
        chunk_ordinal: int,
        content: str,
    ) -> KnowledgeRecord:
        identity = f"{document_hash}:{page_number}:text:{chunk_ordinal}:{content}"
        return KnowledgeRecord(
            record_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            document_id=document_hash,
            modality="text",
            content=content,
            language=self._language(content),
            source=SourceMetadata(
                document_name=source.name,
                document_sha256=document_hash,
                page_start=page_number,
                page_end=page_number,
            ),
            processing=ProcessingMetadata(
                extractor=self.text_extractor.name,
                token_count=self.embedding_backend.count(content),
                content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            ),
        )

    def _chunk_text(self, text: str) -> list[str]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return []
        limit = min(
            self.config.chunk_max_tokens,
            int(self.embedding_backend.max_input_tokens),
        )
        if self.embedding_backend.count(cleaned) <= limit:
            return [cleaned]
        units = re.findall(r"\S+", cleaned) if " " in cleaned else list(cleaned)
        chunks: list[str] = []
        start = 0
        while start < len(units):
            end = start
            best = ""
            while end < len(units):
                candidate = " ".join(units[start : end + 1]) if " " in cleaned else "".join(units[start : end + 1])
                if self.embedding_backend.count(candidate) > limit:
                    break
                best = candidate
                end += 1
            if not best:
                raise ValueError("a text unit exceeds the embedding token limit")
            chunks.append(best)
            if end >= len(units):
                break
            overlap_start = end
            while overlap_start > start:
                candidate = " ".join(units[overlap_start - 1 : end]) if " " in cleaned else "".join(units[overlap_start - 1 : end])
                if self.embedding_backend.count(candidate) > self.config.chunk_overlap_tokens:
                    break
                overlap_start -= 1
            start = overlap_start if overlap_start < end else end
        return chunks

    def _image_records(
        self,
        source: Path,
        document_hash: str,
        candidate: ImageCandidate,
        caption: ImageCaption,
    ) -> list[KnowledgeRecord]:
        assert self.caption_backend is not None
        records = []
        for chunk_ordinal, content in enumerate(
            self._chunk_text(caption.searchable_text()), start=1
        ):
            identity = (
                f"{document_hash}:{candidate.page_number}:image:"
                f"{candidate.artifact_name}:{chunk_ordinal}:{content}"
            )
            records.append(
                KnowledgeRecord(
                    record_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                    document_id=document_hash,
                    modality="image",
                    content=content,
                    language=self._language(content),
                    source=SourceMetadata(
                        document_name=source.name,
                        document_sha256=document_hash,
                        page_start=candidate.page_number,
                        page_end=candidate.page_number,
                        bbox_normalized=candidate.bbox_normalized,
                        artifact_path=f"crops/{candidate.artifact_name}",
                    ),
                    processing=ProcessingMetadata(
                        extractor=(
                            self.image_pipeline.name
                            if self.image_pipeline
                            else "image"
                        ),
                        token_count=self.embedding_backend.count(content),
                        content_sha256=hashlib.sha256(
                            content.encode("utf-8")
                        ).hexdigest(),
                    ),
                    image=ImageMetadata(
                        detector_label=candidate.detector_label,
                        confidence=candidate.confidence,
                        caption_model=self.caption_backend.model_name,
                        prompt_version=self.caption_backend.prompt_version,
                    ),
                )
            )
        return records

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _language(text: str) -> str:
        has_zh = any("\u3400" <= character <= "\u9fff" for character in text)
        has_en = any(character.isascii() and character.isalpha() for character in text)
        if has_zh and has_en:
            return "mixed"
        if has_zh:
            return "zh"
        if has_en:
            return "en"
        return "und"
