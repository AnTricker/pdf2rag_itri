from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from local_rag.application import PdfRagApplication
from local_rag.config import AppConfig
from local_rag.index import FileVectorIndex
from local_rag.models import ExtractedPage
from tests.fakes import FakeCorpusProfileBackend


class Extractor:
    name = "fixture"

    def extract(self, pdf_path: Path) -> list[ExtractedPage]:
        return [ExtractedPage(page_number=1, text="安全操作說明")]


class Embedding:
    name = "fixture"
    model_name = "fixture-2d"
    dimension = 2
    max_input_tokens = 32

    def count(self, text: str) -> int:
        return len(text)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


class SnapshotIntegrityTests(unittest.TestCase):
    def test_loader_rejects_same_length_record_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf = root / "manual.pdf"
            pdf.write_bytes(b"fixture")
            active = PdfRagApplication(
                config=AppConfig.for_test(index_root=root / "index"),
                text_extractor=Extractor(),
                embedding_backend=Embedding(),
                profile_backend=FakeCorpusProfileBackend(),
            ).ingest(pdf)
            manifest = json.loads((active / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["records_sha256"]), 64)
            self.assertEqual(len(manifest["embeddings_sha256"]), 64)
            records_path = active / "records.jsonl"
            original = records_path.read_text(encoding="utf-8")
            records_path.write_text(original.replace("安全", "危險"), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "checksum"):
                FileVectorIndex.load(active)


if __name__ == "__main__":
    unittest.main()
