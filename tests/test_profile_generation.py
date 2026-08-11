from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from local_rag.adapters import OllamaCorpusProfileBackend
from local_rag.application import PdfRagApplication
from local_rag.config import AppConfig
from local_rag.models import ExtractedPage


class TextExtractor:
    name = "fixture-text"

    def extract(self, pdf_path: Path) -> list[ExtractedPage]:
        return [ExtractedPage(page_number=1, text="Hold RESET for three seconds.")]


class EmbeddingBackend:
    name = "fixture-embedding"
    model_name = "fixture-2d"
    dimension = 2
    max_input_tokens = 64

    def count(self, text: str) -> int:
        return len(text.split())

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


class Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


class ProfileGenerationTests(unittest.TestCase):
    def test_ingestion_requests_schema_constrained_corpus_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf = root / "manual.pdf"
            pdf.write_bytes(b"fixture")
            config = replace(
                AppConfig.for_test(index_root=root / "index"),
                llm_model="fixture-model",
            )

            def generate(*_args, **kwargs):
                schema = kwargs["json"]["format"]
                required = set(schema.get("required", [])) if isinstance(schema, dict) else set()
                if {"document_type", "summary", "in_scope_topics"} <= required:
                    content = {
                        "document_type": "vendor manual",
                        "summary": "Reset procedure.",
                        "in_scope_topics": ["reset"],
                        "out_of_scope_examples": ["weather"],
                        "key_sections": ["Reset"],
                        "key_entities": ["RESET button"],
                    }
                else:
                    content = {}
                return Response({"response": json.dumps(content)})

            with patch("local_rag.adapters.requests.post", side_effect=generate):
                active = PdfRagApplication(
                    config=config,
                    text_extractor=TextExtractor(),
                    embedding_backend=EmbeddingBackend(),
                    profile_backend=OllamaCorpusProfileBackend(config, "{record_samples}"),
                ).ingest(pdf, mode="text")

            profile = json.loads((active / "corpus_profile.json").read_text(encoding="utf-8"))
            self.assertEqual(profile["in_scope_topics"], ["reset"])


if __name__ == "__main__":
    unittest.main()
