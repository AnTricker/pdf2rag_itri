from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from local_rag.application import PdfRagApplication
from local_rag.config import AppConfig
from local_rag.models import ExtractedPage
from tests.fakes import FakeCorpusProfileBackend


class TextExtractor:
    name = "fixture-text"

    def extract(self, pdf_path: Path) -> list[ExtractedPage]:
        return [ExtractedPage(page_number=2, text="Hold RESET for three seconds.")]


class EmbeddingBackend:
    name = "fixture-embedding"
    model_name = "fixture-2d"
    dimension = 2
    max_input_tokens = 64

    def count(self, text: str) -> int:
        return len(text.split())

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


class SnapshotDerivedArtifactTests(unittest.TestCase):
    def test_snapshot_contains_human_records_and_required_corpus_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf = root / "manual.pdf"
            pdf.write_bytes(b"fixture")
            active = PdfRagApplication(
                config=AppConfig.for_test(index_root=root / "index"),
                text_extractor=TextExtractor(),
                embedding_backend=EmbeddingBackend(),
                profile_backend=FakeCorpusProfileBackend(),
            ).ingest(pdf, mode="text")

            pretty = json.loads(
                (active / "records.pretty.json").read_text(encoding="utf-8")
            )
            profile = json.loads(
                (active / "corpus_profile.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (active / "manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(pretty[0]["content"], "Hold RESET for three seconds.")
            self.assertNotIn("record_id", pretty[0])
            self.assertNotIn("document_id", pretty[0])
            self.assertNotIn("document_sha256", pretty[0]["source"])
            self.assertNotIn("content_sha256", pretty[0]["processing"])
            self.assertEqual(profile["document_name"], "manual.pdf")
            self.assertEqual(profile["in_scope_topics"], ["operation", "reset"])
            self.assertIn("records.pretty.json", manifest["artifact_sha256"])
            self.assertIn("corpus_profile.json", manifest["artifact_sha256"])


if __name__ == "__main__":
    unittest.main()
