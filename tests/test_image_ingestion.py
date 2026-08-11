from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from local_rag.application import PdfRagApplication
from local_rag.config import AppConfig
from local_rag.models import ExtractedPage, ImageCandidate, ImageCaption
from tests.fakes import FakeCorpusProfileBackend


class FakeTextExtractor:
    name = "fake-text"

    def extract(self, pdf_path: Path) -> list[ExtractedPage]:
        return [ExtractedPage(page_number=1, text="Reset procedure.")]


class FakeImagePipeline:
    name = "fake-existing-ultralytics-call"

    def extract(self, pdf_path: Path, artifact_root: Path) -> list[ImageCandidate]:
        artifact_root.mkdir(parents=True, exist_ok=True)
        candidates = []
        for ordinal in (1, 2):
            crop = artifact_root / f"page_0001_crop_{ordinal:04d}.jpg"
            crop.write_bytes(f"crop-{ordinal}".encode("ascii"))
            candidates.append(
                ImageCandidate(
                    page_number=1,
                    crop_path=crop,
                    artifact_name=crop.name,
                    detector_label="diagram",
                    confidence=0.91,
                    bbox_normalized=(0.1, 0.2, 0.8, 0.9),
                )
            )
        return candidates


class FakeCaptionBackend:
    name = "fake-vlm"
    model_name = "fixture-vlm"
    prompt_version = "caption-v1"

    def describe(self, crop_path: Path) -> ImageCaption:
        if crop_path.name.endswith("0002.jpg"):
            raise RuntimeError("fixture caption failure")
        return ImageCaption(
            summary="控制器重設按鈕位置圖",
            visible_text="RESET",
            content_type="diagram",
        )


class FakeEmbeddingBackend:
    name = "fake-embedding"
    model_name = "fixture-2d"
    dimension = 2
    max_input_tokens = 128

    def count(self, text: str) -> int:
        return len(text.split())

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class StrictSmallEmbeddingBackend(FakeEmbeddingBackend):
    max_input_tokens = 4

    def embed(self, texts: list[str]) -> np.ndarray:
        if any(self.count(text) > self.max_input_tokens for text in texts):
            raise ValueError("embedding input exceeds configured token limit")
        return super().embed(texts)


class LongCaptionBackend(FakeCaptionBackend):
    def describe(self, crop_path: Path) -> ImageCaption:
        if crop_path.name.endswith("0002.jpg"):
            raise RuntimeError("fixture caption failure")
        return ImageCaption(
            summary="reset button diagram with numbered steps and warning labels",
            visible_text="RESET HOLD THREE SECONDS GREEN LIGHT",
            content_type="diagram",
        )


class ImageIngestionTests(unittest.TestCase):
    def test_only_successfully_captioned_crops_become_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf = root / "vendor.pdf"
            pdf.write_bytes(b"%PDF fixture")
            app = PdfRagApplication(
                config=AppConfig.for_test(index_root=root / "index"),
                text_extractor=FakeTextExtractor(),
                embedding_backend=FakeEmbeddingBackend(),
                image_pipeline=FakeImagePipeline(),
                caption_backend=FakeCaptionBackend(),
                profile_backend=FakeCorpusProfileBackend(),
            )

            active = app.ingest(pdf)

            records = [
                json.loads(line)
                for line in (active / "records.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            report = json.loads((active / "build_report.json").read_text(encoding="utf-8"))
            crops = sorted((active / "crops").iterdir())
            image_records = [record for record in records if record["modality"] == "image"]
            self.assertEqual(len(crops), 2)
            self.assertEqual(len(image_records), 1)
            self.assertEqual(image_records[0]["source"]["artifact_path"], "crops/page_0001_crop_0001.jpg")
            self.assertEqual(image_records[0]["image"]["caption_model"], "fixture-vlm")
            self.assertEqual(report["image_record_count"], 1)
            self.assertEqual(len(report["warnings"]), 1)
            self.assertIn("page_0001_crop_0002.jpg", report["warnings"][0])

    def test_long_caption_is_chunked_before_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf = root / "vendor.pdf"
            pdf.write_bytes(b"%PDF fixture")
            active = PdfRagApplication(
                config=AppConfig.for_test(index_root=root / "index").with_text(
                    chunk_max_tokens=4,
                    chunk_overlap_tokens=1,
                ),
                text_extractor=FakeTextExtractor(),
                embedding_backend=StrictSmallEmbeddingBackend(),
                image_pipeline=FakeImagePipeline(),
                caption_backend=LongCaptionBackend(),
                profile_backend=FakeCorpusProfileBackend(),
            ).ingest(pdf)

            records = [
                json.loads(line)
                for line in (active / "records.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            image_records = [record for record in records if record["modality"] == "image"]
            self.assertGreater(len(image_records), 1)
            self.assertTrue(
                all(record["processing"]["token_count"] <= 4 for record in image_records)
            )
            self.assertEqual(
                {record["source"]["artifact_path"] for record in image_records},
                {"crops/page_0001_crop_0001.jpg"},
            )


if __name__ == "__main__":
    unittest.main()
