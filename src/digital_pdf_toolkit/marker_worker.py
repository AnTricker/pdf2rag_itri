from __future__ import annotations

import argparse
import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

from PIL import Image


class MarkerRuntime(NamedTuple):
    PdfConverter: type
    JSONRenderer: type
    MarkdownRenderer: type
    create_model_dict: Any
    save_output: Any
    build_document_stages: Any


def _load_runtime() -> MarkerRuntime:
    try:
        from marker.builders.document import DocumentBuilder
        from marker.builders.line import LineBuilder
        from marker.builders.ocr import OcrBuilder
        from marker.builders.structure import StructureBuilder
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import save_output
        from marker.providers.registry import provider_from_filepath
        from marker.renderers.json import JSONRenderer
        from marker.renderers.markdown import MarkdownRenderer
    except ImportError as exc:  # pragma: no cover - marker environment only
        raise RuntimeError("Marker conversion requires marker-pdf>=2,<3") from exc

    def build_document_stages(converter: Any, filepath: str, capture: Any) -> Any:
        provider_cls = provider_from_filepath(filepath)
        provider = provider_cls(filepath, converter.config)
        layout_builder = converter.resolve_dependencies(converter.layout_builder_class)
        line_builder = converter.resolve_dependencies(LineBuilder)
        ocr_builder = converter.resolve_dependencies(OcrBuilder)
        document = DocumentBuilder(converter.config)(
            provider, layout_builder, line_builder, ocr_builder
        )
        capture("DocumentBuilder", document)

        structure_builder = converter.resolve_dependencies(StructureBuilder)
        structure_builder(document)
        for processor in converter.processor_list:
            processor(document)
        capture("PdfConverter.build_document", document)
        return document

    return MarkerRuntime(
        PdfConverter,
        JSONRenderer,
        MarkdownRenderer,
        create_model_dict,
        save_output,
        build_document_stages,
    )


def _marker_json_key(value: Any) -> str | int | float | bool | None:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Marker tree contains unsupported JSON key: {type(value).__name__}")


def marker_json_value(value: Any) -> Any:
    """Convert a Marker model to JSON values without stringifying unknown objects."""
    if isinstance(value, Image.Image):
        return {"width": value.width, "height": value.height, "mode": value.mode}
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            _marker_json_key(key): marker_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [marker_json_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return marker_json_value(model_dump(mode="python", warnings=False))
    raise TypeError(f"Marker tree contains unsupported value: {type(value).__name__}")


def write_document_tree(document: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = marker_json_value(document)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def collect_block_provenance(document: Any) -> dict[str, Any]:
    def record(block: Any) -> dict[str, Any]:
        method = getattr(block, "text_extraction_method", None)
        source_kind = "ocr" if method == "surya" else method
        return {
            "block_id": str(block.id),
            "block_type": str(block.block_type),
            "page_id": block.page_id,
            "text_extraction_method": method,
            "source_kind": source_kind,
        }

    pages = [record(page) for page in getattr(document, "pages", [])]
    blocks = [record(block) for block in document.contained_blocks()]
    pages.sort(key=lambda item: (item["page_id"], item["block_id"]))
    blocks.sort(key=lambda item: (item["page_id"], item["block_id"]))
    return {"pages": pages, "blocks": blocks}


def convert_pdf(
    input_pdf: Path,
    output_dir: Path,
    runtime: MarkerRuntime | None = None,
    *,
    mode: str = "balanced",
    inference_backend: str = "llamacpp",
) -> None:
    input_pdf = input_pdf.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_pdf.is_file() or input_pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Marker requires one PDF file: {input_pdf}")
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["SURYA_INFERENCE_BACKEND"] = inference_backend
    runtime = runtime or _load_runtime()
    converter = runtime.PdfConverter(artifact_dict=runtime.create_model_dict(), config={"mode": mode})
    provenance: dict[str, Any] = {"schema_version": 2}

    def capture(stage: str, document: Any) -> None:
        write_document_tree(document, output_dir / "result" / f"{stage}.json")
        provenance[stage] = collect_block_provenance(document)

    document = runtime.build_document_stages(converter, str(input_pdf), capture)

    json_renderer = converter.resolve_dependencies(runtime.JSONRenderer)
    markdown_renderer = converter.resolve_dependencies(runtime.MarkdownRenderer)
    runtime.save_output(json_renderer(document), str(output_dir), "result")
    runtime.save_output(markdown_renderer(document), str(output_dir), "result")

    (output_dir / "block_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marker-worker")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", default="balanced", choices=("balanced", "fast"))
    parser.add_argument("--inference-backend", default="llamacpp", choices=("llamacpp", "vllm"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        convert_pdf(args.input, args.output, mode=args.mode, inference_backend=args.inference_backend)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
