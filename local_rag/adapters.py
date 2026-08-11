from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image
from pdf2image import convert_from_path
from pypdf import PdfReader
from zhconv import convert

from .config import AppConfig
from .models import (
    AdmissionDecision,
    CorpusProfile,
    CorpusProfileDraft,
    EvidenceDraft,
    ExtractedPage,
    FinalAnswer,
    FigureExtraction,
    ImageCandidate,
    ImageClassification,
    KnowledgeRecord,
    TableExtraction,
    TableExtractionDraft,
    TableSummary,
    QuestionSplit,
    RetrievalQueryPlan,
)


class PypdfTextExtractor:
    name = "pypdf"

    def extract(self, pdf_path: Path) -> list[ExtractedPage]:
        pages = []
        for number, page in enumerate(PdfReader(str(pdf_path)).pages, start=1):
            text = convert(page.extract_text() or "", "zh-tw")
            pages.append(ExtractedPage(page_number=number, text=text))
        return pages


class SentenceTransformerEmbeddingBackend:
    name = "sentence_transformers_local"

    def __init__(self, model_path: str, *, batch_size: int = 16, reserved_tokens: int = 8) -> None:
        if not model_path:
            raise ValueError("LOCAL_RAG_EMBEDDING_MODEL is required")
        from sentence_transformers import SentenceTransformer

        self.model_name = model_path
        self.model = SentenceTransformer(model_path, local_files_only=True)
        self.batch_size = batch_size
        self.tokenizer = self.model.tokenizer
        model_limit = int(getattr(self.model, "max_seq_length", 512))
        self.max_input_tokens = max(1, model_limit - reserved_tokens)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def count(self, text: str) -> int:
        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
                verbose=False,
            )
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        oversized = [self.count(text) for text in texts if self.count(text) > self.max_input_tokens]
        if oversized:
            raise ValueError("embedding input exceeds configured token limit")
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=len(texts) > self.batch_size,
        )
        return np.asarray(vectors, dtype=np.float32)


class UltralyticsImagePipeline:
    """Preserves the current YOLO/RT-DETR construction and predict kwargs."""

    name = "ultralytics-canonical-crop-v1"

    def __init__(self, config: AppConfig) -> None:
        if config.detector_model is None:
            raise ValueError("LOCAL_RAG_DETECTOR_MODEL is required for image ingestion")
        from ultralytics import RTDETR, YOLO

        model_classes = {"yolo": YOLO, "rtdetr": RTDETR}
        self.model = model_classes[config.detector_backend](str(config.detector_model))
        self.config = config

    def extract(self, pdf_path: Path, artifact_root: Path) -> list[ImageCandidate]:
        render_dir = artifact_root / "rendered"
        crop_dir = artifact_root / "crops"
        render_dir.mkdir(parents=True, exist_ok=True)
        crop_dir.mkdir(parents=True, exist_ok=True)
        pages = convert_from_path(
            str(pdf_path),
            dpi=self.config.render_dpi,
            poppler_path=str(self.config.poppler_path) if self.config.poppler_path else None,
        )
        candidates: list[ImageCandidate] = []
        for page_number, page in enumerate(pages, start=1):
            page_path = render_dir / f"page_{page_number:04d}.png"
            page.save(page_path)
            results = self.model.predict(
                source=str(page_path),
                save=False,
                verbose=False,
                iou=self.config.detector_iou,
                agnostic_nms=False,
            )
            detections = json.loads(results[0].to_json())
            candidates.extend(
                self._crop_page(page, page_number, detections, crop_dir)
            )
        return candidates

    def _crop_page(
        self,
        page: Image.Image,
        page_number: int,
        detections: list[dict],
        crop_dir: Path,
    ) -> list[ImageCandidate]:
        width, height = page.size
        candidates = []
        accepted = [item for item in detections if float(item.get("confidence", 0)) >= self.config.min_confidence]
        for ordinal, item in enumerate(accepted, start=1):
            box = item.get("box", {})
            x1 = max(0, min(int(box.get("x1", 0)), width))
            y1 = max(0, min(int(box.get("y1", 0)), height))
            x2 = max(0, min(int(box.get("x2", 0)), width))
            y2 = max(0, min(int(box.get("y2", 0)), height))
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            name = f"page_{page_number:04d}_crop_{ordinal:04d}.jpg"
            crop_path = crop_dir / name
            page.crop((x1, y1, x2, y2)).convert("RGB").save(crop_path, quality=92)
            candidates.append(
                ImageCandidate(
                    page_number=page_number,
                    crop_path=crop_path,
                    artifact_name=name,
                    detector_label=str(item.get("name", "unknown")),
                    confidence=float(item.get("confidence", 0)),
                    bbox_normalized=(x1 / width, y1 / height, x2 / width, y2 / height),
                )
            )
        return candidates


class OllamaImageReviewBackend:
    name = "ollama"

    def __init__(
        self,
        config: AppConfig,
        *,
        classifier_prompt: str,
        table_extractor_prompt: str,
        table_summary_prompt: str,
        figure_extractor_prompt: str,
    ) -> None:
        if not config.vlm_model:
            raise ValueError("LOCAL_RAG_VLM_MODEL is required")
        self.url = config.vlm_url
        self.model_name = config.vlm_model
        self.timeout = config.vlm_timeout_seconds
        self.classifier_prompt = classifier_prompt
        self.table_extractor_prompt = table_extractor_prompt
        self.table_summary_prompt = table_summary_prompt
        self.figure_extractor_prompt = figure_extractor_prompt
        self.prompt_versions = {
            "image_classifier": "image-classifier-v1",
            "table_extractor": "table-extractor-v1",
            "table_summary": "table-summary-v1",
            "figure_extractor": "figure-extractor-v1",
        }

    def _structured(self, prompt: str, schema_type, *, image_path: Optional[Path] = None):
        import base64

        last_error: Optional[Exception] = None
        for _attempt in range(2):
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "format": schema_type.model_json_schema(),
                "stream": False,
            }
            if image_path is not None:
                payload["images"] = [
                    base64.b64encode(image_path.read_bytes()).decode("ascii")
                ]
            try:
                response = requests.post(
                    f"{self.url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return schema_type.model_validate_json(response.json()["response"])
            except Exception as error:
                last_error = error
        assert last_error is not None
        raise last_error

    def classify(
        self, crop_path: Path, detector_label: str
    ) -> ImageClassification:
        return self._structured(
            self.classifier_prompt.format(detector_label=detector_label),
            ImageClassification,
            image_path=crop_path,
        )

    def extract_table(self, crop_path: Path) -> TableExtractionDraft:
        return self._structured(
            self.table_extractor_prompt,
            TableExtractionDraft,
            image_path=crop_path,
        )

    def summarize_table(self, extraction: TableExtraction) -> TableSummary:
        return self._structured(
            self.table_summary_prompt.format(
                table_json=json.dumps(extraction.model_dump(), ensure_ascii=False)
            ),
            TableSummary,
        )

    def extract_figure(self, crop_path: Path) -> FigureExtraction:
        return self._structured(
            self.figure_extractor_prompt,
            FigureExtraction,
            image_path=crop_path,
        )

class OllamaCorpusProfileBackend:
    name = "ollama"
    prompt_version = "corpus-profile-v1"

    def __init__(self, config: AppConfig, prompt: str) -> None:
        if not config.llm_model:
            raise ValueError("LOCAL_RAG_LLM_MODEL is required")
        self.url = config.llm_url
        self.model_name = config.llm_model
        self.timeout = config.llm_timeout_seconds
        self.prompt = prompt

    def build(
        self,
        *,
        document_name: str,
        document_id: str,
        records: list[KnowledgeRecord],
    ) -> CorpusProfile:
        samples = "\n\n".join(
            f"[Record {index}; page {record.source.page_start}; {record.modality}]\n"
            f"{record.content}"
            for index, record in enumerate(records, start=1)
        )
        response = requests.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": self.prompt.format(
                    document_name=document_name,
                    record_samples=samples,
                ),
                "format": CorpusProfileDraft.model_json_schema(),
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        draft = CorpusProfileDraft.model_validate_json(response.json()["response"])
        return CorpusProfile(
            **draft.model_dump(),
            document_id=document_id,
            document_name=document_name,
            profile_source=self.name,
            generation_model=self.model_name,
            prompt_version=self.prompt_version,
        )


class OllamaChatBackend:
    name = "ollama"
    _split_separator_pattern = re.compile(r"[\s,，.。!！?？;；:：、]+")

    def __init__(
        self,
        config: AppConfig,
        *,
        splitter_prompt: str,
        admission_prompt: str,
        query_builder_prompt: str,
        evidence_draft_prompt: str,
        final_answer_prompt: str,
    ) -> None:
        if not config.llm_model:
            raise ValueError("LOCAL_RAG_LLM_MODEL is required")
        self.url = config.llm_url
        self.model = config.llm_model
        self.timeout = config.llm_timeout_seconds
        self.splitter_prompt = splitter_prompt
        self.admission_prompt = admission_prompt
        self.query_builder_prompt = query_builder_prompt
        self.evidence_draft_prompt = evidence_draft_prompt
        self.final_answer_prompt = final_answer_prompt
        self.preprocessor_max_retries = config.preprocessor_max_retries
        self.answer_max_retries = config.answer_max_retries

    def _generate(self, prompt: str, *, schema: Optional[dict] = None) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if schema is not None:
            payload["format"] = schema
        response = requests.post(
            f"{self.url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return str(response.json()["response"]).strip()

    def health(self) -> bool:
        response = requests.get(f"{self.url}/api/tags", timeout=min(self.timeout, 5))
        response.raise_for_status()
        models = [item.get("name", "") for item in response.json().get("models", [])]
        return any(name == self.model or name.startswith(f"{self.model}:") for name in models)

    def _structured(
        self,
        *,
        stage,
        base_prompt,
        schema_type,
        retries,
        stage_input,
        trace,
        validator=None,
    ):
        last_error = None
        previous_output = None
        retry_reasons: list[str] = []
        for attempt in range(retries + 1):
            prompt = base_prompt
            if retry_reasons:
                prompt += (
                    "\n\n前次輸出："
                    + str(previous_output)
                    + "\n驗證錯誤："
                    + retry_reasons[-1]
                    + "\n請只重新輸出完整且符合 schema 的 JSON。"
                )
            try:
                if trace is not None:
                    trace(
                        "model_call_started",
                        stage=stage,
                        attempt=attempt + 1,
                        backend=self.name,
                        model=self.model,
                        input={**stage_input, "validation_feedback": retry_reasons[-1] if retry_reasons else None},
                    )
                raw_output = self._generate(prompt, schema=schema_type.model_json_schema())
                previous_output = raw_output
                if trace is not None:
                    trace(
                        "model_call_completed",
                        stage=stage,
                        attempt=attempt + 1,
                        backend=self.name,
                        model=self.model,
                        output={"raw_output": raw_output},
                    )
                result = schema_type.model_validate_json(raw_output)
                if validator is not None:
                    validator(result)
                if hasattr(result, "_retry_reasons"):
                    result._retry_reasons = retry_reasons
                return result
            except (KeyError, ValueError) as error:
                last_error = error
                retry_reasons.append(str(error))
                if trace is not None:
                    trace(
                        "model_output_rejected",
                        stage=stage,
                        attempt=attempt + 1,
                        backend=self.name,
                        model=self.model,
                        output={
                            "raw_output": previous_output,
                            "validation_error": str(error),
                        },
                    )
        raise ValueError(f"{stage} returned invalid structured output: {last_error}") from last_error

    def split_questions(self, latest_input, trace=None) -> QuestionSplit:
        return self._structured(
            stage="question_splitter",
            base_prompt=self.splitter_prompt.format(
                latest_input=latest_input,
            ),
            schema_type=QuestionSplit,
            retries=self.preprocessor_max_retries,
            stage_input={"latest_input": latest_input},
            trace=trace,
            validator=lambda result: self._validate_extractive_split(
                latest_input,
                result,
            ),
        )

    @classmethod
    def _validate_extractive_split(
        cls,
        latest_input: str,
        result: QuestionSplit,
    ) -> None:
        expected = cls._split_separator_pattern.sub("", latest_input)
        actual = "".join(
            cls._split_separator_pattern.sub("", question)
            for question in result.questions
        )
        if not expected or actual != expected:
            raise ValueError(
                "splitter questions must preserve every non-separator token "
                "from latest_input in the original order without translation, "
                "rewriting, insertion, deletion, or deduplication"
            )

    def classify_question(self, question, corpus_profile, trace=None) -> AdmissionDecision:
        stage_input = {
            "question": question,
            "corpus_profile": corpus_profile.model_dump(),
        }
        return self._structured(
            stage="query_admission",
            base_prompt=self.admission_prompt.format(
                question=question,
                corpus_profile=json.dumps(corpus_profile.model_dump(), ensure_ascii=False),
            ),
            schema_type=AdmissionDecision,
            retries=self.preprocessor_max_retries,
            stage_input=stage_input,
            trace=trace,
        )

    def build_retrieval_queries(
        self,
        question,
        history,
        corpus_profile,
        trace=None,
    ) -> RetrievalQueryPlan:
        return self._structured(
            stage="retrieval_query_builder",
            base_prompt=self.query_builder_prompt.format(
                question=question,
                history=json.dumps(history[-8:], ensure_ascii=False),
                corpus_profile=json.dumps(corpus_profile.model_dump(), ensure_ascii=False),
            ),
            schema_type=RetrievalQueryPlan,
            retries=self.preprocessor_max_retries,
            stage_input={
                "question": question,
                "history": history[-8:],
                "corpus_profile": corpus_profile.model_dump(),
            },
            trace=trace,
        )

    def draft_evidence(self, question, records, trace=None) -> EvidenceDraft:
        return self._structured(
            stage="evidence_draft",
            base_prompt=self.evidence_draft_prompt.format(
                question=question,
                records=json.dumps(records, ensure_ascii=False),
            ),
            schema_type=EvidenceDraft,
            retries=self.answer_max_retries,
            stage_input={"question": question, "records": records},
            trace=trace,
        )

    def compose_answer(
        self,
        question,
        evidence,
        records,
        trace=None,
    ) -> FinalAnswer:
        return self._structured(
            stage="final_answer_composer",
            base_prompt=self.final_answer_prompt.format(
                question=question,
                evidence=json.dumps(evidence.model_dump(), ensure_ascii=False),
                records=json.dumps(records, ensure_ascii=False),
            ),
            schema_type=FinalAnswer,
            retries=self.answer_max_retries,
            stage_input={
                "question": question,
                "evidence": evidence.model_dump(),
                "records": records,
            },
            trace=trace,
        )
