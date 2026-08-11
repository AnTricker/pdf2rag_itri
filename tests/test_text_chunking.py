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


class LongPageExtractor:
    name = "fixture-long-page"

    def extract(self, pdf_path: Path) -> list[ExtractedPage]:
        return [ExtractedPage(page_number=2, text="one two three four five six seven")]


class WordEmbedding:
    name = "fixture"
    model_name = "word-count"
    dimension = 2
    max_input_tokens = 4

    def count(self, text: str) -> int:
        return len(text.split())

    def embed(self, texts: list[str]) -> np.ndarray:
        if any(self.count(text) > self.max_input_tokens for text in texts):
            raise AssertionError("oversized input reached embedding backend")
        return np.ones((len(texts), 2), dtype=np.float32)


class TextChunkingTests(unittest.TestCase):
    def test_long_page_is_chunked_before_embedding_without_silent_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf = root / "manual.pdf"
            pdf.write_bytes(b"fixture")
            config = AppConfig.for_test(index_root=root / "index").with_text(
                chunk_max_tokens=4,
                chunk_overlap_tokens=1,
            )
            active = PdfRagApplication(
                config=config,
                text_extractor=LongPageExtractor(),
                embedding_backend=WordEmbedding(),
                profile_backend=FakeCorpusProfileBackend(),
            ).ingest(pdf)

            records = [
                json.loads(line)
                for line in (active / "records.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["content"] for record in records], ["one two three four", "four five six seven"])
            self.assertTrue(all(record["processing"]["token_count"] <= 4 for record in records))


if __name__ == "__main__":
    unittest.main()
