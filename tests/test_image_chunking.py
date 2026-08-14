from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from local_rag.application import PdfRagApplication
from local_rag.chat import ChatEngine
from local_rag.config import AppConfig
from local_rag.models import (
    FigureExtraction,
    FigureTextBlock,
    ImageClassification,
    ImageMetadata,
    TableCellDraft,
    TableExtraction,
    TableExtractionDraft,
    TableSummary,
)
from tests.test_reviewable_pipeline import (
    ImagePipeline,
    ProfileBackend,
    ReviewBackend,
    TextExtractor,
)


class LimitedEmbedding:
    name = "fixture-limited"
    model_name = "char-count"
    dimension = 3
    max_input_tokens = 120

    def count(self, text: str) -> int:
        return len(text)

    def embed(self, texts: list[str]) -> np.ndarray:
        if any(self.count(text) > self.max_input_tokens for text in texts):
            raise AssertionError("oversized image chunk reached embedding")
        return np.ones((len(texts), self.dimension), dtype=np.float32)


class SixtyTokenEmbedding(LimitedEmbedding):
    max_input_tokens = 60


class LongCellBackend(ReviewBackend):
    def extract_table(self, _crop_path: Path) -> TableExtractionDraft:
        self.table_calls += 1
        return TableExtractionDraft(
            title="長表格",
            notes=[],
            cells=[
                TableCellDraft(
                    text="資料" * 100,
                    row_index=1,
                    column_index=1,
                    cell_type="content",
                )
            ],
        )

    def summarize_table(self, _extraction: TableExtraction) -> TableSummary:
        self.summary_calls += 1
        return TableSummary(
            text="長儲存格測試。",
            evidence_cell_ids=["r1c1"],
        )


class LongFigureBackend(ReviewBackend):
    def classify(
        self, _crop_path: Path, _detector_label: str
    ) -> ImageClassification:
        self.classify_calls += 1
        return ImageClassification(content_type="figure", reason="長圖片")

    def extract_figure(self, _crop_path: Path) -> FigureExtraction:
        self.figure_calls += 1
        return FigureExtraction(
            visual_description="設備流程" * 60,
            text_blocks=[
                FigureTextBlock(
                    location="圖片中央",
                    text="原始文字" * 55,
                )
            ],
        )


class ImageChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pdf = self.root / "manual.pdf"
        self.pdf.write_bytes(b"%PDF fixture")
        self.config = AppConfig.for_test(
            index_root=self.root / "outputs"
        ).with_text(
            chunk_max_tokens=120,
            chunk_overlap_tokens=20,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_app(self, backend, embedding=None) -> PdfRagApplication:
        return PdfRagApplication(
            config=self.config,
            text_extractor=TextExtractor(),
            embedding_backend=embedding or LimitedEmbedding(),
            image_pipeline=ImagePipeline(),
            image_review_backend=backend,
            profile_backend=ProfileBackend(),
        )

    @staticmethod
    def image_records(build: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in (build / "records.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if '"modality": "image"' in line
        ]

    def test_table_build_creates_bounded_grouped_chunks(self) -> None:
        app = self.make_app(ReviewBackend())
        output = app.ingest(self.pdf, mode="image")
        build = app.build(output.name, 1)
        records = self.image_records(build)

        self.assertGreater(len(records), 1)
        self.assertTrue(
            all(record["processing"]["token_count"] <= 120 for record in records)
        )
        self.assertEqual(
            {record["image"]["image_group_id"] for record in records}.__len__(),
            1,
        )
        self.assertEqual(
            [record["image"]["chunk_index"] for record in records],
            list(range(1, len(records) + 1)),
        )
        self.assertTrue(
            all(record["image"]["chunk_count"] == len(records) for record in records)
        )
        self.assertTrue(all(record["schema_version"] == "2.1" for record in records))
        self.assertTrue(
            all(record["table"]["cells"] == records[0]["table"]["cells"] for record in records)
        )

        report = json.loads(
            (build / "build_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["source_image_count"], 1)
        self.assertEqual(report["image_chunk_count"], len(records))
        self.assertEqual(report["image_record_count"], len(records))
        manifest = json.loads(
            (build / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], "2.1")
        self.assertEqual(manifest["record_count"], len(records))

    def test_oversized_cell_splits_body_and_repeats_cell_id(self) -> None:
        app = self.make_app(LongCellBackend())
        output = app.ingest(self.pdf, mode="image")
        records = self.image_records(app.build(output.name, 1))
        cell_chunks = [
            record for record in records if "[r1c1;" in record["content"]
        ]

        self.assertGreater(len(cell_chunks), 1)
        self.assertTrue(
            all("[r1c1;" in record["content"] for record in cell_chunks)
        )
        self.assertTrue(
            all(record["processing"]["token_count"] <= 120 for record in records)
        )

    def test_figure_description_and_text_blocks_are_chunked(self) -> None:
        app = self.make_app(LongFigureBackend())
        output = app.ingest(self.pdf, mode="image")
        records = self.image_records(app.build(output.name, 1))
        contents = [record["content"] for record in records]

        self.assertGreater(len(records), 2)
        self.assertTrue(any("圖片描述：" in content for content in contents))
        self.assertTrue(
            sum("[text_block=1; text]" in content for content in contents) > 1
        )
        self.assertTrue(
            all(record["processing"]["token_count"] <= 120 for record in records)
        )
        self.assertTrue(
            all(record["figure"] == records[0]["figure"] for record in records)
        )

    def test_effective_limit_rejects_overlap_from_env_when_too_large(self) -> None:
        config = self.config.with_text(
            chunk_max_tokens=120,
            chunk_overlap_tokens=60,
        )
        app = PdfRagApplication(
            config=config,
            text_extractor=TextExtractor(),
            embedding_backend=SixtyTokenEmbedding(),
        )
        with self.assertRaisesRegex(
            ValueError, "chunk overlap must be smaller"
        ):
            app._effective_chunk_settings()

    def test_logical_unit_overlap_repeats_complete_unit(self) -> None:
        config = self.config.with_text(
            chunk_max_tokens=25,
            chunk_overlap_tokens=12,
        )
        app = PdfRagApplication(
            config=config,
            text_extractor=TextExtractor(),
            embedding_backend=LimitedEmbedding(),
        )
        chunks = app._chunk_image_units(
            [
                ("A:", "11111111"),
                ("B:", "22222222"),
                ("C:", "33333333"),
            ]
        )

        self.assertEqual(len(chunks), 2)
        self.assertIn("B:22222222", chunks[0])
        self.assertIn("B:22222222", chunks[1])

    def test_old_image_metadata_uses_single_chunk_defaults(self) -> None:
        metadata = ImageMetadata(
            content_type="figure",
            caption_model="legacy-model",
        )
        self.assertEqual(metadata.image_group_id, "")
        self.assertEqual(metadata.chunk_index, 1)
        self.assertEqual(metadata.chunk_count, 1)
    def test_citations_deduplicate_chunks_with_the_same_crop(self) -> None:
        app = self.make_app(ReviewBackend())
        output = app.ingest(self.pdf, mode="image")
        records = self.image_records(app.build(output.name, 1))
        from local_rag.models import KnowledgeRecord

        hits = [
            SimpleNamespace(
                record=KnowledgeRecord.model_validate(record),
                score=1.0 - index / 10,
            )
            for index, record in enumerate(records[:2])
        ]
        citations = ChatEngine._citations(hits)

        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["record_id"], records[0]["record_id"])
        self.assertEqual(
            citations[0]["crop_url"],
            f'/api/crops/{records[0]["record_id"]}',
        )


if __name__ == "__main__":
    unittest.main()
