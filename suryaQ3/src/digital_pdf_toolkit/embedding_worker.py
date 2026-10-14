from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .embedding import build_knowledge_base


class SentenceTransformerEmbedder:
    def __init__(self, settings: dict[str, Any], preprocess: dict[str, Any]) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - real embedding environment
            raise RuntimeError("qwen3vl requires torch and sentence-transformers[image]") from exc

        runtime = settings["runtime"]
        self.model_id = str(settings["model_id"])
        self._torch = torch
        self.revision = settings.get("revision")
        self.dimension = int(settings["dimension"])
        self.dtype = str(runtime.get("dtype", "float16"))
        self.normalize_embeddings = bool(runtime.get("normalize_embeddings", True))
        self.batch_size = int(runtime.get("batch_size", 1))
        self.device = str(runtime.get("device", "cuda"))
        image_preprocess = preprocess.get("image_preprocess") or {}
        self.image_patch_size = int(image_preprocess.get("patch_size", 16))
        self.image_min_pixels = int(image_preprocess.get("min_pixels", 4096))
        self.image_max_pixels = int(image_preprocess.get("max_pixels", 1310720))
        dtype = getattr(torch, self.dtype, None)
        if dtype is None:
            raise ValueError(f"Unsupported torch dtype: {self.dtype}")
        model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        attention = runtime.get("attention")
        if attention:
            model_kwargs["attn_implementation"] = attention
        token = os.environ.get("HF_TOKEN")
        self.model = SentenceTransformer(
            self.model_id,
            revision=self.revision,
            device=self.device,
            model_kwargs=model_kwargs,
            token=token,
        )
        self._configure_image_processor()
        configured_max = runtime.get("max_tokens")
        model_max = getattr(self.model, "max_seq_length", None)
        if configured_max is None and model_max is None:
            raise ValueError("Model does not expose max_seq_length; set runtime.max_tokens")
        self.max_tokens = int(configured_max or model_max)
        if self.max_tokens <= 0:
            raise ValueError("runtime.max_tokens must be greater than zero")

    def _encode(self, values: list[Any]) -> Any:
        method = getattr(self.model, "encode_document", self.model.encode)
        return method(
            values,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def encode_texts(self, values: list[str]) -> Any:
        return self._encode(values)

    def encode_image(self, value: Path) -> Any:
        return self._encode([{"image": str(value)}])

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize()

    def _image_processor(self) -> Any:
        candidates = [self.model]
        first_module = getattr(self.model, "_first_module", None)
        if callable(first_module):
            candidates.append(first_module())
        candidates.extend(getattr(self.model, "_modules", {}).values())
        for candidate in candidates:
            processor = getattr(candidate, "processor", None)
            image_processor = getattr(processor, "image_processor", None)
            if image_processor is not None:
                return image_processor
        raise RuntimeError("SentenceTransformer model does not expose its image processor")

    def _configure_image_processor(self) -> None:
        image_processor = self._image_processor()
        if hasattr(image_processor, "min_pixels"):
            image_processor.min_pixels = self.image_min_pixels
        if hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = self.image_max_pixels
        if hasattr(image_processor, "size"):
            image_processor.size = {
                "shortest_edge": self.image_min_pixels,
                "longest_edge": self.image_max_pixels,
            }

    def inspect_image(self, value: Path) -> dict[str, Any]:
        features = self.model.tokenize([{"image": str(value)}])
        grid = features.get("image_grid_thw")
        if grid is None:
            raise RuntimeError("Qwen3-VL processor did not return image_grid_thw")
        if hasattr(grid, "detach"):
            grid = grid.detach().cpu().tolist()
        elif hasattr(grid, "tolist"):
            grid = grid.tolist()
        if isinstance(grid, list) and grid and isinstance(grid[0], list):
            grid = grid[0]
        if not isinstance(grid, list) or len(grid) != 3:
            raise ValueError(f"Unexpected image_grid_thw: {grid!r}")
        temporal, grid_height, grid_width = (int(part) for part in grid)
        return {
            "image_grid_thw": [temporal, grid_height, grid_width],
            "patch_size": self.image_patch_size,
            "effective_width": grid_width * self.image_patch_size,
            "effective_height": grid_height * self.image_patch_size,
        }

    def split_text(self, value: str) -> list[str]:
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("SentenceTransformer model does not expose its tokenizer")
        special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
        budget = self.max_tokens - special_tokens
        if budget <= 0:
            raise ValueError("Model token budget is too small")
        token_ids = tokenizer.encode(value, add_special_tokens=False)
        if len(token_ids) <= budget:
            return [value]
        return [
            tokenizer.decode(token_ids[offset:offset + budget], skip_special_tokens=True)
            for offset in range(0, len(token_ids), budget)
        ]


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Resolved config must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embedding-worker")
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = _load_config(args.config_json)
    embedding = config["embedding"]
    mode = str(embedding["mode"])
    if mode != "qwen3vl":
        raise ValueError(f"Unsupported embedding mode: {mode}")
    settings = embedding["modes"][mode]
    if settings.get("provider") != "sentence_transformers":
        raise ValueError(f"Unsupported embedding provider: {settings.get('provider')}")
    embedder = SentenceTransformerEmbedder(
        settings, config.get("embedding_preprocess", {}),
    )
    build_knowledge_base(args.input, args.output_dir, config, embedder, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
