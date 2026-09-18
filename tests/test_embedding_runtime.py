from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from local_rag.adapters import SentenceTransformerEmbeddingBackend
from local_rag.config import AppConfig


class FakeSentenceTransformer:
    calls = []

    def __init__(self, model_path, **kwargs):
        self.calls.append((model_path, kwargs))
        self.tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1])
        self.max_seq_length = 32

    def get_sentence_embedding_dimension(self):
        return 4096


class EmbeddingRuntimeTests(unittest.TestCase):
    def test_qwen_runtime_is_forwarded_to_sentence_transformer(self) -> None:
        module = SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            backend = SentenceTransformerEmbeddingBackend(
                "/models/Qwen3-VL-Embedding-8B",
                batch_size=1,
                device="cuda",
                dtype="float16",
                attention="sdpa",
            )
        self.assertEqual(backend.dimension, 4096)
        _path, kwargs = FakeSentenceTransformer.calls[-1]
        self.assertEqual(kwargs["device"], "cuda")
        self.assertEqual(kwargs["model_kwargs"]["torch_dtype"], "float16")
        self.assertEqual(kwargs["model_kwargs"]["attn_implementation"], "sdpa")

    def test_config_rejects_unknown_embedding_runtime_values(self) -> None:
        config = AppConfig.for_test(index_root=Path("/tmp/index"))
        with self.assertRaisesRegex(ValueError, "dtype"):
            replace(config, embedding_dtype="int8").validate()
        with self.assertRaisesRegex(ValueError, "attention"):
            replace(config, embedding_attention="unknown").validate()


if __name__ == "__main__":
    unittest.main()
