from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

import main as cli
from local_rag.application import (
    ImageReviewIncompleteError,
    ImageTaskFailure,
    PdfRagApplication,
)
from local_rag.config import AppConfig
from local_rag.index import FileVectorIndex
from local_rag.web import create_app
from local_rag.models import (
    CorpusProfile,
    ExtractedPage,
    FigureExtraction,
    ImageCandidate,
    ImageClassification,
    KnowledgeRecord,
    TableCell,
    TableCellDraft,
    TableExtraction,
    TableExtractionDraft,
    TableSummary,
)


class TextExtractor:
    name = "fixture-text"

    def extract(self, _pdf_path: Path) -> list[ExtractedPage]:
        return [ExtractedPage(page_number=1, text="設備重設程序")]


class EmbeddingBackend:
    name = "fixture-embedding"
    model_name = "fixture"
    dimension = 3
    max_input_tokens = 10000

    def count(self, text: str) -> int:
        return len(text)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[float(index + 1), 1.0, 0.5] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )


class ImagePipeline:
    name = "fixture-crop"

    def __init__(self) -> None:
        self.calls = 0

    def extract(
        self, _pdf_path: Path, artifact_root: Path
    ) -> list[ImageCandidate]:
        self.calls += 1
        crops = artifact_root / "crops"
        crops.mkdir(parents=True)
        crop_path = crops / "page_0001_crop_0001.jpg"
        crop_path.write_bytes(b"fixture crop")
        return [
            ImageCandidate(
                page_number=1,
                crop_path=crop_path,
                artifact_name=crop_path.name,
                detector_label="dstable",
                confidence=0.8,
                bbox_normalized=(0.1, 0.2, 0.9, 0.8),
            )
        ]


class TwoImagePipeline(ImagePipeline):
    def extract(
        self, pdf_path: Path, artifact_root: Path
    ) -> list[ImageCandidate]:
        first = super().extract(pdf_path, artifact_root)[0]
        second_path = artifact_root / "crops" / "page_0001_crop_0002.jpg"
        second_path.write_bytes(b"second fixture crop")
        return [
            first,
            ImageCandidate(
                page_number=1,
                crop_path=second_path,
                artifact_name=second_path.name,
                detector_label="dsfig",
                confidence=0.7,
                bbox_normalized=(0.2, 0.3, 0.8, 0.7),
            ),
        ]


class ReviewBackend:
    name = "fixture-review"
    model_name = "fixture-model"
    prompt_versions = {
        "image_classifier": "image-classifier-v1",
        "table_extractor": "table-extractor-v1",
        "table_summary": "table-summary-v1",
        "figure_extractor": "figure-extractor-v1",
    }

    def __init__(self, *, unusable: bool = False) -> None:
        self.unusable = unusable
        self.classify_calls = 0
        self.table_calls = 0
        self.summary_calls = 0
        self.figure_calls = 0

    def classify(
        self, _crop_path: Path, _detector_label: str
    ) -> ImageClassification:
        self.classify_calls += 1
        return ImageClassification(
            content_type="unusable" if self.unusable else "table",
            reason="測試分類",
        )

    def extract_table(self, _crop_path: Path) -> TableExtractionDraft:
        self.table_calls += 1
        return TableExtractionDraft(
            title="測試表",
            notes=["單位：%"],
            cells=[
                TableCellDraft(
                    text="項目",
                    row_index=1,
                    column_index=1,
                    cell_type="column_header",
                ),
                TableCellDraft(
                    text="數值",
                    row_index=1,
                    column_index=2,
                    cell_type="column_header",
                ),
                TableCellDraft(
                    text="Model B",
                    row_index=2,
                    column_index=1,
                    cell_type="row_header",
                ),
                TableCellDraft(
                    text="87.4%",
                    row_index=2,
                    column_index=2,
                    cell_type="content",
                ),
            ],
        )

    def summarize_table(self, _extraction: TableExtraction) -> TableSummary:
        self.summary_calls += 1
        return TableSummary(
            text="Model B 的數值為 87.4%。",
            evidence_cell_ids=["r2c1", "r2c2"],
        )

    def extract_figure(self, _crop_path: Path) -> FigureExtraction:
        self.figure_calls += 1
        return FigureExtraction(
            visual_description="設備圖片。",
            text_blocks=[],
        )


class FigureReviewBackend(ReviewBackend):
    def classify(
        self, _crop_path: Path, _detector_label: str
    ) -> ImageClassification:
        self.classify_calls += 1
        return ImageClassification(
            content_type="figure",
            reason="測試圖片",
        )

class FailingClassifierBackend(ReviewBackend):
    def classify(
        self, _crop_path: Path, _detector_label: str
    ) -> ImageClassification:
        self.classify_calls += 1
        raise ConnectionError("Ollama unavailable")


class SecondClassifierFailsBackend(ReviewBackend):
    def classify(
        self, crop_path: Path, detector_label: str
    ) -> ImageClassification:
        if self.classify_calls == 1:
            self.classify_calls += 1
            raise ConnectionError("Ollama interrupted")
        return super().classify(crop_path, detector_label)


class FailingSummaryBackend(ReviewBackend):
    def summarize_table(self, _extraction: TableExtraction) -> TableSummary:
        self.summary_calls += 1
        raise ValueError("invalid summary schema")


class ProfileBackend:
    def build(
        self,
        *,
        document_name: str,
        document_id: str,
        records: list[KnowledgeRecord],
    ) -> CorpusProfile:
        self.last_records = records
        return CorpusProfile(
            document_id=document_id,
            document_name=document_name,
            document_type="操作文件",
            summary="設備操作程序。",
            in_scope_topics=["設備"],
            out_of_scope_examples=[],
            key_sections=[],
            key_entities=[],
            profile_source="fixture",
            generation_model="fixture-model",
            prompt_version="corpus-profile-v1",
        )


class ReviewablePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pdf = self.root / "manual.pdf"
        self.pdf.write_bytes(b"%PDF fixture")
        self.config = AppConfig.for_test(index_root=self.root / "outputs")
        self.image_pipeline = ImagePipeline()
        self.review_backend = ReviewBackend()
        self.app = PdfRagApplication(
            config=self.config,
            text_extractor=TextExtractor(),
            embedding_backend=EmbeddingBackend(),
            image_pipeline=self.image_pipeline,
            image_review_backend=self.review_backend,
            profile_backend=ProfileBackend(),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_ingest_review_build_are_isolated_and_resumable(self) -> None:
        output = self.app.ingest(self.pdf, mode="multi")
        self.assertRegex(
            output.name,
            r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}$",
        )
        self.assertEqual(self.image_pipeline.calls, 1)
        self.assertFalse((output / "builds").exists())
        self.assertFalse((output / "document").exists())

        review_one = output / "pending" / "001_records.pretty.json"
        payload = json.loads(review_one.read_text(encoding="utf-8"))
        self.assertIn("review_instructions", payload)
        image_records = [
            item for item in payload["records"] if item["modality"] == "image"
        ]
        self.assertEqual(len(image_records), 1)
        self.assertEqual(image_records[0]["content_type"], "table")
        self.assertEqual(len(image_records[0]["table_extraction"]["cells"]), 4)

        image_records[0]["table_extraction"]["title"] = "人工修正標題"
        image_records[0]["table_summary"] = None
        review_one.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        review_two = self.app.review(output.name)
        self.assertEqual(review_two.name, "002_records.pretty.json")
        self.assertEqual(self.image_pipeline.calls, 1)
        self.assertEqual(self.review_backend.classify_calls, 1)
        self.assertEqual(self.review_backend.table_calls, 1)
        self.assertEqual(self.review_backend.summary_calls, 2)

        build = self.app.build(output.name, 2)
        self.assertEqual(build, output / "builds" / "002")
        lines = (build / "records.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        image = next(json.loads(line) for line in lines if '"modality": "image"' in line)
        self.assertEqual(image["table"]["title"], "人工修正標題")
        self.assertIn("[r2c2; content;", image["content"])
        self.assertNotIn("detector_label", image["image"])
        self.assertFalse((build / "records.pretty.json").exists())
        self.assertFalse((build / "document").exists())
        self.assertIsNotNone(
            FileVectorIndex.load(build, output_root=output)
        )
        with self.assertRaises(FileExistsError):
            self.app.build(output.name, 2)

    def test_unusable_image_is_kept_in_review_but_excluded_from_text_only_build(self) -> None:
        app = PdfRagApplication(
            config=self.config,
            text_extractor=TextExtractor(),
            embedding_backend=EmbeddingBackend(),
            image_pipeline=ImagePipeline(),
            image_review_backend=ReviewBackend(unusable=True),
            profile_backend=ProfileBackend(),
        )
        output = app.ingest(self.pdf, mode="multi")
        review = json.loads(
            (output / "pending" / "001_records.pretty.json").read_text(
                encoding="utf-8"
            )
        )
        image = next(
            item for item in review["records"] if item["modality"] == "image"
        )
        self.assertEqual(image["content_type"], "unusable")
        self.assertIsNone(image["figure_extraction"])
        self.assertIsNone(image["table_extraction"])

        build = app.build(output.name, 1)
        records = (build / "records.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertTrue(records)
        self.assertTrue(all('"modality": "text"' in line for line in records))
        report = json.loads(
            (build / "build_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["excluded_unusable_count"], 1)
        self.assertIn("build contains no image records", report["warnings"])

    def test_figure_build_is_served_from_shared_crop(self) -> None:
        app = PdfRagApplication(
            config=self.config,
            text_extractor=TextExtractor(),
            embedding_backend=EmbeddingBackend(),
            image_pipeline=ImagePipeline(),
            image_review_backend=FigureReviewBackend(),
            profile_backend=ProfileBackend(),
        )
        output = app.ingest(self.pdf, mode="image")
        unchanged = app.review(output.name)
        self.assertEqual(unchanged.name, "001_records.pretty.json")
        build = app.build(output.name, 1)
        record = next(
            KnowledgeRecord.model_validate_json(line)
            for line in (build / "records.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        )
        self.assertEqual(record.image.content_type, "figure")
        self.assertIn("圖片描述：設備圖片。", record.content)

        client = create_app(
            config=self.config,
            embedding_backend=EmbeddingBackend(),
            chat_backend=object(),
            build_root=build,
            output_root=output,
        ).test_client()
        crop = client.get(f"/api/crops/{record.record_id}")
        self.assertEqual(crop.status_code, 200)
        self.assertEqual(crop.data, b"fixture crop")
        crop.close()
    def test_classifier_failure_preserves_output_and_reports_incomplete(self) -> None:
        app = PdfRagApplication(
            config=self.config,
            text_extractor=TextExtractor(),
            embedding_backend=EmbeddingBackend(),
            image_pipeline=ImagePipeline(),
            image_review_backend=FailingClassifierBackend(),
            profile_backend=ProfileBackend(),
        )

        with self.assertRaises(ImageReviewIncompleteError) as caught:
            app.ingest(self.pdf, mode="multi")

        error = caught.exception
        self.assertEqual(error.operation, "ingest")
        self.assertEqual(error.image_complete, 0)
        self.assertEqual(error.image_pending, 1)
        self.assertEqual(error.failures[0].stage, "classifier")
        self.assertTrue(error.output_root.is_dir())
        payload = json.loads(error.review_path.read_text(encoding="utf-8"))
        image = next(
            item for item in payload["records"] if item["modality"] == "image"
        )
        self.assertIsNone(image["content_type"])
        self.assertEqual(image["model_name"], "")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            cli.report_image_review_failure(error)
        report = stderr.getvalue()
        self.assertIn("[VLM ERROR]", report)
        self.assertIn("Ollama unavailable", report)
        self.assertIn("complete=0 failed=1 pending=1 total=1", report)

    def test_partial_classifier_failure_preserves_successful_image(self) -> None:
        backend = SecondClassifierFailsBackend()
        app = PdfRagApplication(
            config=self.config,
            text_extractor=TextExtractor(),
            embedding_backend=EmbeddingBackend(),
            image_pipeline=TwoImagePipeline(),
            image_review_backend=backend,
            profile_backend=ProfileBackend(),
        )

        with self.assertRaises(ImageReviewIncompleteError) as caught:
            app.ingest(self.pdf, mode="image")

        error = caught.exception
        self.assertEqual(error.image_complete, 1)
        self.assertEqual(error.image_pending, 1)
        payload = json.loads(error.review_path.read_text(encoding="utf-8"))
        images = [
            item for item in payload["records"] if item["modality"] == "image"
        ]
        self.assertEqual(images[0]["content_type"], "table")
        self.assertIsNotNone(images[0]["table_summary"])
        self.assertIsNone(images[1]["content_type"])

    def test_summary_failure_keeps_completed_fields_for_review_retry(self) -> None:
        backend = FailingSummaryBackend()
        app = PdfRagApplication(
            config=self.config,
            text_extractor=TextExtractor(),
            embedding_backend=EmbeddingBackend(),
            image_pipeline=ImagePipeline(),
            image_review_backend=backend,
            profile_backend=ProfileBackend(),
        )

        with self.assertRaises(ImageReviewIncompleteError) as caught:
            app.ingest(self.pdf, mode="image")

        error = caught.exception
        payload = json.loads(error.review_path.read_text(encoding="utf-8"))
        image = next(
            item for item in payload["records"] if item["modality"] == "image"
        )
        self.assertEqual(error.failures[0].stage, "table summary")
        self.assertEqual(image["content_type"], "table")
        self.assertIsNotNone(image["table_extraction"])
        self.assertIsNone(image["table_summary"])
        self.assertEqual(image["model_name"], "fixture-model")

        with self.assertRaises(ImageReviewIncompleteError) as review_caught:
            app.review(error.output_root.name)
        self.assertEqual(review_caught.exception.operation, "review")
        self.assertEqual(review_caught.exception.review_path.name, "001_records.pretty.json")

        app.image_review_backend = ReviewBackend()
        review_two = app.review(error.output_root.name)
        self.assertEqual(review_two.name, "002_records.pretty.json")

    def test_cli_returns_nonzero_for_retained_vlm_failure(self) -> None:
        output = self.root / "outputs" / "2026-08-12_00-00-00-000001"
        review = output / "pending" / "001_records.pretty.json"
        failure = ImageReviewIncompleteError(
            operation="ingest",
            output_root=output,
            review_path=review,
            failures=[
                ImageTaskFailure(
                    artifact_path="crops/page_0001_crop_0001.jpg",
                    stage="classifier",
                    error=ConnectionError("Ollama unavailable"),
                )
            ],
            image_total=1,
            image_complete=0,
        )

        class RaisingApplication:
            def ingest(self, _source: Path, *, mode: str) -> Path:
                raise failure

        stderr = io.StringIO()
        with (
            patch.object(cli.AppConfig, "from_env", return_value=self.config),
            patch.object(cli, "pipeline_application", return_value=RaisingApplication()),
            redirect_stderr(stderr),
        ):
            exit_code = cli.main(
                ["ingest", "--input", str(self.pdf), "--mode", "image"]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("[VLM SUMMARY]", stderr.getvalue())
        self.assertIn(str(output), stderr.getvalue())

    def test_table_grid_rejects_missing_cells(self) -> None:
        with self.assertRaises(ValidationError):
            TableExtraction(
                title="broken",
                notes=[],
                row_count=2,
                column_count=2,
                cells=[
                    TableCell(
                        id="r1c1",
                        text="header",
                        row_index=1,
                        column_index=1,
                        cell_type="column_header",
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()