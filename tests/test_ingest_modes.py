from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from local_rag.application import PdfRagApplication
from local_rag.cli import run_cli
from local_rag.config import AppConfig
from local_rag.models import ExtractedPage, ImageCandidate, ImageCaption
from tests.fakes import FakeCorpusProfileBackend


class RecordingTextExtractor:
    name = "fixture-text"

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, pdf_path: Path) -> list[ExtractedPage]:
        self.calls += 1
        return [ExtractedPage(page_number=1, text="Reset procedure")]


class FailingTextExtractor(RecordingTextExtractor):
    def extract(self, pdf_path: Path) -> list[ExtractedPage]:
        self.calls += 1
        raise RuntimeError("fixture text failure")


class RecordingImagePipeline:
    name = "fixture-image"

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, pdf_path: Path, artifact_root: Path) -> list[ImageCandidate]:
        self.calls += 1
        artifact_root.mkdir(parents=True, exist_ok=True)
        crop = artifact_root / "page_0001_crop_0001.jpg"
        crop.write_bytes(b"crop")
        return [
            ImageCandidate(
                page_number=1,
                crop_path=crop,
                artifact_name=crop.name,
                detector_label="diagram",
                confidence=0.9,
                bbox_normalized=(0.1, 0.1, 0.9, 0.9),
            )
        ]


class CaptionBackend:
    name = "fixture-vlm"
    model_name = "fixture-model"
    prompt_version = "caption-v1"

    def describe(self, crop_path: Path) -> ImageCaption:
        return ImageCaption(
            summary="Reset button diagram",
            visible_text="RESET",
            content_type="diagram",
        )


class EmbeddingBackend:
    name = "fixture-embedding"
    model_name = "fixture-2d"
    dimension = 2
    max_input_tokens = 64

    def count(self, text: str) -> int:
        return len(text.split())

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


class IngestModeTests(unittest.TestCase):
    def test_cli_mode_runs_only_selected_record_pipelines(self) -> None:
        expectations = {
            "text": ({"text"}, 1, 0),
            "image": ({"image"}, 0, 1),
            "multi": ({"text", "image"}, 1, 1),
        }
        for mode, (modalities, text_calls, image_calls) in expectations.items():
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                pdf = root / "manual.pdf"
                pdf.write_bytes(b"fixture")
                text = RecordingTextExtractor()
                image = RecordingImagePipeline()
                app = PdfRagApplication(
                    config=AppConfig.for_test(index_root=root / "index"),
                    text_extractor=text,
                    embedding_backend=EmbeddingBackend(),
                    image_pipeline=image,
                    caption_backend=CaptionBackend(),
                    profile_backend=FakeCorpusProfileBackend(),
                )

                exit_code = run_cli(
                    ["ingest", "--input", str(pdf), "--mode", mode],
                    application_factory=lambda: app,
                )

                self.assertEqual(exit_code, 0)
                records = [
                    json.loads(line)
                    for line in (root / "index" / "current" / "records.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                self.assertEqual({record["modality"] for record in records}, modalities)
                self.assertEqual(text.calls, text_calls)
                self.assertEqual(image.calls, image_calls)

    def test_multi_mode_preserves_successful_image_branch_when_text_branch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf = root / "manual.pdf"
            pdf.write_bytes(b"fixture")
            active = PdfRagApplication(
                config=AppConfig.for_test(index_root=root / "index"),
                text_extractor=FailingTextExtractor(),
                embedding_backend=EmbeddingBackend(),
                image_pipeline=RecordingImagePipeline(),
                caption_backend=CaptionBackend(),
                profile_backend=FakeCorpusProfileBackend(),
            ).ingest(pdf, mode="multi")

            records = [
                json.loads(line)
                for line in (active / "records.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            report = json.loads(
                (active / "build_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual({record["modality"] for record in records}, {"image"})
            self.assertEqual(report["stage_statuses"]["text_extraction"], "failed")
            self.assertEqual(report["stage_statuses"]["image_processing"], "complete")
            self.assertEqual(len(report["warnings"]), 1)


if __name__ == "__main__":
    unittest.main()
