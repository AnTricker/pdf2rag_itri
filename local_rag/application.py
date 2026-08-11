from __future__ import annotations

from datetime import datetime
import hashlib
import json
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
    FigureExtraction,
    ImageCandidate,
    ImageClassification,
    ImageMetadata,
    KnowledgeRecord,
    ProcessingMetadata,
    ReviewEnvelope,
    ReviewImageRecord,
    ReviewSource,
    ReviewTextRecord,
    SnapshotManifest,
    SourceMetadata,
    TableCell,
    TableData,
    TableExtraction,
    TableExtractionDraft,
    TableSummary,
)
from .snapshot import BuildWriter


OUTPUT_ID_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}$")
REVIEW_FILE_PATTERN = re.compile(r"^(\d+)_records\.pretty\.json$")


class TextExtractor(Protocol):
    name: str

    def extract(self, pdf_path: Path) -> list[ExtractedPage]: ...


class EmbeddingBackend(Protocol):
    name: str
    model_name: str
    dimension: int
    max_input_tokens: int

    def count(self, text: str) -> int: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


class ImagePipeline(Protocol):
    name: str

    def extract(self, pdf_path: Path, artifact_root: Path) -> list[ImageCandidate]: ...


class ImageReviewBackend(Protocol):
    name: str
    model_name: str
    prompt_versions: dict[str, str]

    def classify(self, crop_path: Path, detector_label: str) -> ImageClassification: ...
    def extract_table(self, crop_path: Path) -> TableExtractionDraft: ...
    def summarize_table(self, extraction: TableExtraction) -> TableSummary: ...
    def extract_figure(self, crop_path: Path) -> FigureExtraction: ...


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
        image_review_backend: Optional[ImageReviewBackend] = None,
        profile_backend: Optional[CorpusProfileBackend] = None,
    ) -> None:
        self.config = config
        self.text_extractor = text_extractor
        self.embedding_backend = embedding_backend
        self.image_pipeline = image_pipeline
        self.image_review_backend = image_review_backend
        self.profile_backend = profile_backend

    def ingest(self, input_pdf: Path, *, mode: Optional[str] = None) -> Path:
        source = input_pdf.resolve(strict=True)
        selected_mode = (mode or self.config.ingest_mode).strip().lower()
        if selected_mode not in {"text", "image", "multi"}:
            raise ValueError("ingest mode must be text, image or multi")

        output_root = self._new_output_root()
        output_root.mkdir(parents=True)
        pending_root = output_root / "pending"
        pending_root.mkdir()
        document_hash = self._sha256(source)
        records: list[ReviewTextRecord | ReviewImageRecord] = []
        try:
            if selected_mode in {"text", "multi"}:
                for page in self.text_extractor.extract(source):
                    for content in self._chunk_text(page.text):
                        records.append(
                            ReviewTextRecord(
                                content=content,
                                language=self._language(content),
                                source=ReviewSource(
                                    document_name=source.name,
                                    document_sha256=document_hash,
                                    page_number=page.page_number,
                                ),
                                extractor=self.text_extractor.name,
                            )
                        )

            if selected_mode in {"image", "multi"}:
                if self.image_pipeline is None:
                    raise ValueError("image mode requires an image pipeline")
                candidates = self.image_pipeline.extract(source, output_root)
                shutil.rmtree(output_root / "rendered", ignore_errors=True)
                for candidate in candidates:
                    record = ReviewImageRecord(
                        source=ReviewSource(
                            document_name=source.name,
                            document_sha256=document_hash,
                            page_number=candidate.page_number,
                            bbox_normalized=candidate.bbox_normalized,
                            artifact_path=f"crops/{candidate.artifact_name}",
                        ),
                        detector_label=candidate.detector_label,
                        detector_confidence=candidate.confidence,
                        model_name=(
                            self.image_review_backend.model_name
                            if self.image_review_backend is not None
                            else ""
                        ),
                    )
                    self._complete_image(record, output_root)
                    records.append(record)

            if not records:
                raise ValueError("ingestion did not yield any review records")
            envelope = ReviewEnvelope(
                document_name=source.name,
                document_sha256=document_hash,
                ingest_mode=selected_mode,
                records=records,
            )
            review_path = pending_root / "001_records.pretty.json"
            self._write_review(review_path, envelope)
            return output_root
        except Exception:
            shutil.rmtree(output_root, ignore_errors=True)
            raise

    def review(self, output_id: str) -> Path:
        output_root = self.resolve_output(output_id)
        latest_version, latest_path = self.latest_review(output_root)
        envelope = ReviewEnvelope.model_validate_json(
            latest_path.read_text(encoding="utf-8")
        )
        before = envelope.model_dump_json()
        for record in envelope.records:
            if isinstance(record, ReviewImageRecord):
                self._complete_image(record, output_root)
        if envelope.model_dump_json() == before:
            return latest_path
        next_path = (
            output_root
            / "pending"
            / f"{latest_version + 1:03d}_records.pretty.json"
        )
        self._write_review(next_path, envelope)
        return next_path

    def build(self, output_id: str, review_version: int) -> Path:
        output_root = self.resolve_output(output_id)
        review_path = (
            output_root / "pending" / f"{review_version:03d}_records.pretty.json"
        )
        if not review_path.is_file():
            raise FileNotFoundError(f"review revision not found: {review_path}")
        envelope = ReviewEnvelope.model_validate_json(
            review_path.read_text(encoding="utf-8")
        )

        records: list[KnowledgeRecord] = []
        excluded_unusable = 0
        excluded_incomplete = 0
        for review_record in envelope.records:
            if isinstance(review_record, ReviewTextRecord):
                records.append(self._knowledge_text_record(review_record))
                continue
            converted = self._knowledge_image_record(review_record)
            if converted is not None:
                records.append(converted)
            elif review_record.content_type == "unusable":
                excluded_unusable += 1
            else:
                excluded_incomplete += 1

        if not records:
            raise ValueError("selected review did not yield any searchable records")
        if self.profile_backend is None:
            raise ValueError("build requires a corpus profile backend")
        profile = self._build_corpus_profile(envelope, records)
        vectors = np.asarray(
            self.embedding_backend.embed([record.content for record in records]),
            dtype=np.float32,
        )
        prompt_versions: dict[str, str] = {}
        for review_record in envelope.records:
            if isinstance(review_record, ReviewImageRecord):
                prompt_versions.update(review_record.prompt_versions)
        manifest = SnapshotManifest(
            schema_version="2.0",
            document_sha256=envelope.document_sha256,
            record_count=len(records),
            vector_count=len(vectors),
            vector_dimension=self.embedding_backend.dimension,
            embedding=EmbeddingMetadata(
                backend=self.embedding_backend.name,
                model=self.embedding_backend.model_name,
            ),
            chunk_max_tokens=self.config.chunk_max_tokens,
            chunk_overlap_tokens=self.config.chunk_overlap_tokens,
            prompt_versions=prompt_versions,
        )
        report = BuildReport(
            source_document=envelope.document_name,
            document_id=envelope.document_sha256,
            text_record_count=sum(record.modality == "text" for record in records),
            image_record_count=sum(record.modality == "image" for record in records),
            excluded_unusable_count=excluded_unusable,
            excluded_incomplete_count=excluded_incomplete,
            embedded_count=len(records),
            warnings=(
                ["build contains no image records"]
                if not any(record.modality == "image" for record in records)
                else []
            ),
            stage_statuses={
                "review_validation": "complete",
                "corpus_profile": "complete",
                "embedding": "complete",
                "build_validation": "complete",
            },
            artifact_locations={
                "records": "records.jsonl",
                "corpus_profile": "corpus_profile.json",
                "embeddings": "embeddings.npy",
            },
        )
        return BuildWriter(output_root, review_version).write(
            records=records,
            embeddings=vectors,
            manifest=manifest,
            report=report,
            corpus_profile=profile,
        )

    def _complete_image(
        self, record: ReviewImageRecord, output_root: Path
    ) -> None:
        if self.image_review_backend is None:
            return
        crop_path = self._resolve_crop(output_root, record.source.artifact_path)
        if record.content_type is None:
            try:
                classification = self.image_review_backend.classify(
                    crop_path, record.detector_label
                )
            except Exception:
                return
            record.content_type = classification.content_type
            record.classification_reason = classification.reason
            record.model_name = self.image_review_backend.model_name
            record.prompt_versions.update(
                self.image_review_backend.prompt_versions
            )

        if record.content_type == "unusable":
            return
        if record.content_type == "table":
            if record.table_extraction is None:
                try:
                    draft = self.image_review_backend.extract_table(crop_path)
                    record.table_extraction = self._normalize_table(draft)
                except Exception:
                    return
            if record.table_summary is None:
                try:
                    summary = self.image_review_backend.summarize_table(
                        record.table_extraction
                    )
                    self._validate_summary(record.table_extraction, summary)
                    record.table_summary = summary
                except Exception:
                    return
            return
        if record.content_type == "figure" and record.figure_extraction is None:
            try:
                record.figure_extraction = self.image_review_backend.extract_figure(
                    crop_path
                )
            except Exception:
                return

    @staticmethod
    def _normalize_table(draft: TableExtractionDraft) -> TableExtraction:
        cells = [
            TableCell(id=f"r{cell.row_index}c{cell.column_index}", **cell.model_dump())
            for cell in draft.cells
        ]
        row_count = max(cell.row_index + cell.row_span - 1 for cell in cells)
        column_count = max(
            cell.column_index + cell.column_span - 1 for cell in cells
        )
        return TableExtraction(
            title=draft.title,
            notes=draft.notes,
            row_count=row_count,
            column_count=column_count,
            cells=cells,
        )

    @staticmethod
    def _validate_summary(
        extraction: TableExtraction, summary: TableSummary
    ) -> None:
        allowed = {cell.id for cell in extraction.cells}
        unknown = set(summary.evidence_cell_ids) - allowed
        if unknown:
            raise ValueError(f"summary references unknown cells: {sorted(unknown)}")

    def _knowledge_text_record(
        self, review_record: ReviewTextRecord
    ) -> KnowledgeRecord:
        source = review_record.source
        identity = (
            f"{source.document_sha256}:{source.page_number}:text:"
            f"{review_record.content}"
        )
        return KnowledgeRecord(
            record_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            document_id=source.document_sha256,
            modality="text",
            content=review_record.content,
            language=review_record.language,
            source=self._source_metadata(source),
            processing=self._processing_metadata(
                review_record.extractor, review_record.content
            ),
        )

    def _knowledge_image_record(
        self, review_record: ReviewImageRecord
    ) -> Optional[KnowledgeRecord]:
        if review_record.content_type == "table":
            if (
                review_record.table_extraction is None
                or review_record.table_summary is None
            ):
                return None
            self._validate_summary(
                review_record.table_extraction, review_record.table_summary
            )
            table = TableData(
                **review_record.table_extraction.model_dump(),
                summary=review_record.table_summary,
            )
            content = self._serialize_table(table)
            self._ensure_image_content_fits(content)
            figure = None
        elif review_record.content_type == "figure":
            if review_record.figure_extraction is None:
                return None
            content = self._serialize_figure(review_record.figure_extraction)
            self._ensure_image_content_fits(content)
            table = None
            figure = review_record.figure_extraction
        else:
            return None

        source = review_record.source
        identity = (
            f"{source.document_sha256}:{source.page_number}:image:"
            f"{source.artifact_path}:{content}"
        )
        return KnowledgeRecord(
            record_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            document_id=source.document_sha256,
            modality="image",
            content=content,
            language=self._language(content),
            source=self._source_metadata(source),
            processing=self._processing_metadata("reviewed-image-v2", content),
            image=ImageMetadata(
                content_type=review_record.content_type,
                caption_model=review_record.model_name,
                prompt_versions=review_record.prompt_versions,
            ),
            table=table,
            figure=figure,
        )

    def _ensure_image_content_fits(self, content: str) -> None:
        limit = min(
            self.config.chunk_max_tokens,
            int(self.embedding_backend.max_input_tokens),
        )
        if self.embedding_backend.count(content) > limit:
            raise ValueError(
                "complete image record exceeds embedding token limit; "
                "content was not truncated"
            )

    @staticmethod
    def _serialize_table(table: TableData) -> str:
        lines = [
            f"表格標題：{table.title or '[EMPTY]'}",
            f"摘要：{table.summary.text}",
            "",
            "儲存格：",
        ]
        for cell in sorted(
            table.cells, key=lambda item: (item.row_index, item.column_index)
        ):
            text = cell.text if cell.text else "[EMPTY]"
            lines.append(
                f"[{cell.id}; {cell.cell_type}; row={cell.row_index}; "
                f"column={cell.column_index}; row_span={cell.row_span}; "
                f"column_span={cell.column_span}] {text}"
            )
        lines.append("")
        lines.append(
            "註解：" + ("；".join(table.notes) if table.notes else "無")
        )
        return "\n".join(lines)

    @staticmethod
    def _serialize_figure(figure: FigureExtraction) -> str:
        lines = [f"圖片描述：{figure.visual_description}"]
        if figure.text_blocks:
            lines.append("可見文字：")
            for block in figure.text_blocks:
                lines.append(f"[{block.location}] {block.text}")
        else:
            lines.append("可見文字：無")
        return "\n".join(lines)

    def _build_corpus_profile(
        self, envelope: ReviewEnvelope, records: list[KnowledgeRecord]
    ) -> CorpusProfile:
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
            document_name=envelope.document_name,
            document_id=envelope.document_sha256,
            records=samples,
        )
        if (
            profile.document_id != envelope.document_sha256
            or profile.document_name != envelope.document_name
        ):
            raise ValueError("corpus profile does not match selected review")
        return profile

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
                candidate = (
                    " ".join(units[start : end + 1])
                    if " " in cleaned
                    else "".join(units[start : end + 1])
                )
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
                candidate = (
                    " ".join(units[overlap_start - 1 : end])
                    if " " in cleaned
                    else "".join(units[overlap_start - 1 : end])
                )
                if (
                    self.embedding_backend.count(candidate)
                    > self.config.chunk_overlap_tokens
                ):
                    break
                overlap_start -= 1
            start = overlap_start if overlap_start < end else end
        return chunks

    def _processing_metadata(
        self, extractor: str, content: str
    ) -> ProcessingMetadata:
        return ProcessingMetadata(
            extractor=extractor,
            token_count=self.embedding_backend.count(content),
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )

    @staticmethod
    def _source_metadata(source: ReviewSource) -> SourceMetadata:
        return SourceMetadata(
            document_name=source.document_name,
            document_sha256=source.document_sha256,
            page_start=source.page_number,
            page_end=source.page_number,
            bbox_normalized=source.bbox_normalized,
            artifact_path=source.artifact_path,
            section_path=source.section_path,
        )

    def _new_output_root(self) -> Path:
        self.config.index_root.mkdir(parents=True, exist_ok=True)
        while True:
            output_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            candidate = self.config.index_root / output_id
            if not candidate.exists():
                return candidate

    def resolve_output(self, output_id: str) -> Path:
        if not OUTPUT_ID_PATTERN.fullmatch(output_id):
            raise ValueError("output must be a timestamp identifier")
        root = self.config.index_root.resolve()
        candidate = (root / output_id).resolve(strict=True)
        if candidate.parent != root:
            raise ValueError("output escapes configured output root")
        return candidate

    @staticmethod
    def latest_review(output_root: Path) -> tuple[int, Path]:
        candidates = []
        pending_root = output_root / "pending"
        for path in pending_root.iterdir():
            match = REVIEW_FILE_PATTERN.fullmatch(path.name)
            if match and path.is_file():
                candidates.append((int(match.group(1)), path))
        if not candidates:
            raise FileNotFoundError("output has no review revisions")
        return max(candidates, key=lambda item: item[0])

    @staticmethod
    def latest_build(output_root: Path) -> Path:
        from .index import FileVectorIndex

        builds_root = output_root / "builds"
        candidates = sorted(
            [
                (int(path.name), path)
                for path in builds_root.iterdir()
                if path.is_dir()
                and path.name.isdigit()
                and not path.name.startswith(".")
            ]
            if builds_root.is_dir()
            else [],
            reverse=True,
        )
        for _version, path in candidates:
            try:
                FileVectorIndex.load(path, output_root=output_root)
                return path
            except Exception:
                continue
        raise FileNotFoundError("output has no valid completed builds")

    @staticmethod
    def _write_review(path: Path, envelope: ReviewEnvelope) -> None:
        if path.exists():
            raise FileExistsError(f"review revision already exists: {path}")
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    envelope.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _resolve_crop(output_root: Path, artifact_path: Optional[str]) -> Path:
        if not artifact_path:
            raise ValueError("image record has no crop artifact path")
        relative = Path(artifact_path)
        if relative.is_absolute() or relative.parts[:1] != ("crops",):
            raise ValueError("crop path must be relative to crops/")
        candidate = (output_root / relative).resolve()
        crops_root = (output_root / "crops").resolve()
        if crops_root not in candidate.parents or not candidate.is_file():
            raise FileNotFoundError(f"crop not found: {artifact_path}")
        return candidate

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