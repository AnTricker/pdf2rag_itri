from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from local_rag.application import PdfRagApplication
from local_rag.cli import run_cli
from local_rag.config import AppConfig
from local_rag.models import ExtractedPage
from tests.fakes import FakeCorpusProfileBackend


class FakeTextExtractor:
    name = "fake-pdf-text-v1"

    def extract(self, pdf_path: Path) -> list[ExtractedPage]:
        return [
            ExtractedPage(
                page_number=1,
                text="Power on the controller. Confirm the green status light.",
            )
        ]


class FakeEmbeddingBackend:
    name = "fake-embedding"
    model_name = "fixture-3d"
    dimension = 3
    max_input_tokens = 128

    def count(self, text: str) -> int:
        return len(text.split())

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


class IngestCliTests(unittest.TestCase):
    def test_ingest_builds_a_valid_portable_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_pdf = root / "vendor.pdf"
            input_pdf.write_bytes(b"%PDF-1.4 fixture")
            index_root = root / "index"
            config = AppConfig.for_test(index_root=index_root)
            application = PdfRagApplication(
                config=config,
                text_extractor=FakeTextExtractor(),
                embedding_backend=FakeEmbeddingBackend(),
                profile_backend=FakeCorpusProfileBackend(),
            )

            exit_code = run_cli(
                ["ingest", "--input", str(input_pdf)],
                application_factory=lambda: application,
            )

            self.assertEqual(exit_code, 0)
            active = index_root / "current"
            self.assertTrue((active / "document" / "vendor.pdf").is_file())
            self.assertTrue((active / "records.jsonl").is_file())
            self.assertTrue((active / "embeddings.npy").is_file())
            self.assertTrue((active / "manifest.json").is_file())
            self.assertTrue((active / "build_report.json").is_file())

            records = [
                json.loads(line)
                for line in (active / "records.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            vectors = np.load(active / "embeddings.npy", allow_pickle=False)
            manifest = json.loads(
                (active / "manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["modality"], "text")
            self.assertEqual(records[0]["source"]["page_start"], 1)
            self.assertEqual(vectors.shape, (1, 3))
            self.assertEqual(vectors.tolist(), [[1.0, 0.0, 0.0]])
            self.assertEqual(manifest["record_count"], 1)
            self.assertEqual(manifest["vector_count"], 1)
            self.assertEqual(manifest["vector_dimension"], 3)
            self.assertEqual(manifest["embedding"]["model"], "fixture-3d")
            self.assertNotIn(str(root.resolve()), json.dumps(records))


if __name__ == "__main__":
    unittest.main()
