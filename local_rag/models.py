from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExtractedPage(BaseModel):
    page_number: int = Field(ge=1)
    text: str


class SourceMetadata(BaseModel):
    document_name: str
    document_sha256: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    bbox_normalized: Optional[tuple[float, float, float, float]] = None
    artifact_path: Optional[str] = None
    section_path: list[str] = Field(default_factory=list)


class ProcessingMetadata(BaseModel):
    extractor: str
    token_count: int = Field(ge=0)
    content_sha256: str


class ImageMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_type: Literal["table", "figure"]
    caption_model: str
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    image_group_id: str = ""
    chunk_index: int = Field(default=1, ge=1)
    chunk_count: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_chunk_position(self) -> "ImageMetadata":
        if self.chunk_index > self.chunk_count:
            raise ValueError("image chunk index exceeds chunk count")
        return self


class ImageClassification(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    content_type: Literal["table", "figure", "unusable"]
    reason: str = Field(min_length=1)


class TableCellDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    row_index: int = Field(ge=1)
    column_index: int = Field(ge=1)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    cell_type: Literal["column_header", "row_header", "corner_header", "content"]


class TableCell(TableCellDraft):
    id: str = Field(pattern=r"^r[1-9][0-9]*c[1-9][0-9]*$")


class TableExtractionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    notes: list[str]
    cells: list[TableCellDraft] = Field(min_length=1)


class TableExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    notes: list[str]
    row_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    cells: list[TableCell] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_grid(self) -> "TableExtraction":
        occupied: dict[tuple[int, int], str] = {}
        for cell in self.cells:
            expected_id = f"r{cell.row_index}c{cell.column_index}"
            if cell.id != expected_id:
                raise ValueError(f"cell id must be deterministic: {expected_id}")
            row_end = cell.row_index + cell.row_span - 1
            column_end = cell.column_index + cell.column_span - 1
            if row_end > self.row_count or column_end > self.column_count:
                raise ValueError("cell span exceeds declared table dimensions")
            for row in range(cell.row_index, row_end + 1):
                for column in range(cell.column_index, column_end + 1):
                    coordinate = (row, column)
                    if coordinate in occupied:
                        raise ValueError(
                            f"cells {occupied[coordinate]} and {cell.id} overlap"
                        )
                    occupied[coordinate] = cell.id
        expected = {
            (row, column)
            for row in range(1, self.row_count + 1)
            for column in range(1, self.column_count + 1)
        }
        missing = sorted(expected - occupied.keys())
        if missing:
            raise ValueError(f"table grid has missing cells: {missing[:8]}")
        return self


class TableSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1)
    evidence_cell_ids: list[str] = Field(min_length=1)


class TableData(TableExtraction):
    summary: TableSummary


class FigureTextBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1)
    location: str = Field(min_length=1)


class FigureExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    visual_description: str = Field(min_length=1)
    text_blocks: list[FigureTextBlock]


class KnowledgeRecord(BaseModel):
    schema_version: str = "2.1"
    record_id: str
    document_id: str
    modality: Literal["text", "image"]
    content: str
    language: Literal["zh", "en", "mixed", "und"] = "und"
    source: SourceMetadata
    processing: ProcessingMetadata
    image: Optional[ImageMetadata] = None
    table: Optional[TableData] = None
    figure: Optional[FigureExtraction] = None


class ImageCandidate(BaseModel):
    page_number: int = Field(ge=1)
    crop_path: Path
    artifact_name: str
    detector_label: str
    confidence: float = Field(ge=0, le=1)
    bbox_normalized: tuple[float, float, float, float]


class ReviewSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_name: str
    document_sha256: str
    page_number: int = Field(ge=1)
    bbox_normalized: Optional[tuple[float, float, float, float]] = None
    artifact_path: Optional[str] = None
    section_path: list[str] = Field(default_factory=list)


class ReviewTextRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modality: Literal["text"] = "text"
    content: str
    language: Literal["zh", "en", "mixed", "und"] = "und"
    source: ReviewSource
    extractor: str


class ReviewImageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modality: Literal["image"] = "image"
    source: ReviewSource
    detector_label: str
    detector_confidence: float = Field(ge=0, le=1)
    content_type: Optional[Literal["table", "figure", "unusable"]] = None
    classification_reason: Optional[str] = None
    table_extraction: Optional[TableExtraction] = None
    table_summary: Optional[TableSummary] = None
    figure_extraction: Optional[FigureExtraction] = None
    model_name: str = ""
    prompt_versions: dict[str, str] = Field(default_factory=dict)


class ReviewInstructions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    editable_fields: list[str] = Field(
        default_factory=lambda: [
            "records[].content_type",
            "records[].table_extraction.title",
            "records[].table_extraction.notes",
            "records[].table_extraction.cells",
            "records[].table_summary",
            "records[].figure_extraction.visual_description",
            "records[].figure_extraction.text_blocks",
        ]
    )
    read_only_fields: list[str] = Field(
        default_factory=lambda: [
            "schema_version",
            "document_name",
            "document_sha256",
            "records[].source",
            "records[].detector_label",
            "records[].detector_confidence",
            "records[].model_name",
            "records[].prompt_versions",
        ]
    )
    rules: list[str] = Field(
        default_factory=lambda: [
            "描述欄位使用繁體中文；擷取文字保持原文，不翻譯或修正。",
            "原始空白 cell 必須以空字串保留。",
            "不得修改 crop 路徑、來源識別資訊或 schema version。",
        ]
    )


class ReviewEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    document_name: str
    document_sha256: str
    ingest_mode: Literal["text", "image", "multi"]
    review_instructions: ReviewInstructions = Field(default_factory=ReviewInstructions)
    records: list[Union[ReviewTextRecord, ReviewImageRecord]]

class CorpusProfileDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    in_scope_topics: list[str] = Field(min_length=1)
    out_of_scope_examples: list[str] = Field(default_factory=list)
    key_sections: list[str] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)


class CorpusProfile(CorpusProfileDraft):
    schema_version: str = "1.0"
    document_id: str
    document_name: str
    profile_source: str
    generation_model: str
    prompt_version: str


class QuestionSplit(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    questions: list[str] = Field(min_length=1)


class AdmissionDecision(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    category: Literal["document_question", "out_of_scope", "security_request"]


class RetrievalQueryPlan(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    queries: list[str] = Field(min_length=1, max_length=3)


class EvidenceNote(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    text: str = Field(min_length=1)


class EvidenceDraft(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    notes: list[EvidenceNote]


class FinalAnswerBlock(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    text: str = Field(min_length=1)
    source_numbers: list[int]


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    blocks: list[FinalAnswerBlock] = Field(min_length=1)


class QAHistoryPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class EmbeddingMetadata(BaseModel):
    backend: str
    model: str


class SnapshotManifest(BaseModel):
    schema_version: str = "2.1"
    snapshot_id: str = ""
    document_sha256: str
    records_sha256: str = ""
    embeddings_sha256: str = ""
    record_count: int = Field(ge=0)
    vector_count: int = Field(ge=0)
    vector_dimension: int = Field(gt=0)
    vector_dtype: Literal["float32"] = "float32"
    embedding: EmbeddingMetadata
    chunk_max_tokens: int = Field(gt=0)
    chunk_overlap_tokens: int = Field(ge=0)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    artifact_sha256: dict[str, str] = Field(default_factory=dict)


class BuildReport(BaseModel):
    status: Literal["complete"] = "complete"
    source_document: str
    snapshot_id: str = ""
    document_id: str
    text_record_count: int = Field(ge=0)
    image_record_count: int = Field(ge=0)
    source_image_count: int = Field(default=0, ge=0)
    image_chunk_count: int = Field(default=0, ge=0)
    excluded_unusable_count: int = Field(default=0, ge=0)
    excluded_incomplete_count: int = Field(default=0, ge=0)
    embedded_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    stage_statuses: dict[str, str] = Field(default_factory=dict)
    artifact_locations: dict[str, str] = Field(default_factory=dict)
