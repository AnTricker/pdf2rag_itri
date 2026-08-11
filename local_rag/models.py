from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


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
    detector_label: str
    confidence: float = Field(ge=0, le=1)
    caption_model: str
    prompt_version: str


class KnowledgeRecord(BaseModel):
    schema_version: str = "1.0"
    record_id: str
    document_id: str
    modality: Literal["text", "image"]
    content: str
    language: Literal["zh", "en", "mixed", "und"] = "und"
    source: SourceMetadata
    processing: ProcessingMetadata
    image: Optional[ImageMetadata] = None


class ImageCandidate(BaseModel):
    page_number: int = Field(ge=1)
    crop_path: Path
    artifact_name: str
    detector_label: str
    confidence: float = Field(ge=0, le=1)
    bbox_normalized: tuple[float, float, float, float]


class ImageCaption(BaseModel):
    summary: str = Field(min_length=1)
    visible_text: str = ""
    content_type: Literal["photo", "diagram", "table", "other"]

    def searchable_text(self) -> str:
        parts = [f"類型：{self.content_type}", f"摘要：{self.summary.strip()}"]
        if self.visible_text.strip():
            parts.append(f"可見文字：{self.visible_text.strip()}")
        return "\n".join(parts)


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
    schema_version: str = "1.0"
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
    embedded_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    stage_statuses: dict[str, str] = Field(default_factory=dict)
    artifact_locations: dict[str, str] = Field(default_factory=dict)
