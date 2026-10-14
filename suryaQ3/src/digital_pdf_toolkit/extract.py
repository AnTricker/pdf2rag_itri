from __future__ import annotations

from pathlib import Path
from typing import Any

from .coordinates import displayed_size, rotate_bbox
from .events import EventLogger
from .io import relative_to, write_json


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "x") and hasattr(value, "y"):
        return [float(value.x), float(value.y)]
    if hasattr(value, "x0") and hasattr(value, "y0"):
        return [float(value.x0), float(value.y0), float(value.x1), float(value.y1)]
    return str(value)


def _canonical_bbox(raw_bbox: Any, width: float, height: float, rotation: int) -> list[float]:
    return list(rotate_bbox(raw_bbox, width, height, rotation))


def extract_document(pdf_path: Path, run_root: Path, render_dpi: int, logger: EventLogger) -> dict[str, Any]:
    """Extract observable PDF objects without invoking OCR or layout models."""
    try:
        import fitz
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - exercised in the real core env
        raise RuntimeError("Core environment requires PyMuPDF and pdfplumber") from exc

    pages_root = run_root / "pages"
    renders_root = run_root / "page_renders"
    embedded_root = run_root / "embedded_images"
    for directory in (pages_root, renders_root, embedded_root):
        directory.mkdir(parents=True, exist_ok=True)

    logger.emit("extract", "START", f"native extraction: {pdf_path.name}")
    document = fitz.open(pdf_path)
    if document.needs_pass:
        document.close()
        raise PermissionError("PDF requires a password")

    summary: dict[str, Any] = {
        "schema_version": "0.2",
        "source": str(pdf_path.resolve()),
        "metadata": _jsonable(document.metadata),
        "page_count": document.page_count,
        "coordinate_system": {
            "origin": "top_left",
            "x_axis": "right",
            "y_axis": "down",
            "unit": "pt",
            "rotation_applied": True,
        },
        "pages": [],
        "warnings": [],
    }

    with pdfplumber.open(pdf_path) as plumber_document:
        for page_index, page in enumerate(document):
            plumber_page = plumber_document.pages[page_index]
            rotation = int(page.rotation or 0) % 360
            raw_width = float(page.mediabox.width)
            raw_height = float(page.mediabox.height)
            width, height = displayed_size(raw_width, raw_height, rotation)
            page_id = f"p{page_index + 1:04d}"

            pixmap = page.get_pixmap(dpi=render_dpi, alpha=False)
            render_path = renders_root / f"{page_id}.png"
            pixmap.save(render_path)

            raw_text = page.get_text("rawdict")
            native_chars: list[dict[str, Any]] = []
            native_spans: list[dict[str, Any]] = []
            for block_index, block in enumerate(raw_text.get("blocks", [])):
                if block.get("type") != 0:
                    continue
                for line_index, line in enumerate(block.get("lines", [])):
                    for span_index, span in enumerate(line.get("spans", [])):
                        span_id = f"{page_id}-s{len(native_spans) + 1:06d}"
                        span_chars = span.get("chars", [])
                        native_spans.append({
                            "id": span_id,
                            "text": "".join(char.get("c", "") for char in span_chars),
                            "bbox": _canonical_bbox(span["bbox"], raw_width, raw_height, rotation),
                            "font": span.get("font"),
                            "size": span.get("size"),
                            "color": span.get("color"),
                            "flags": span.get("flags"),
                            "source": {
                                "kind": "native",
                                "tool": "pymupdf",
                                "block": block_index,
                                "line": line_index,
                                "span": span_index,
                            },
                        })
                        for char in span_chars:
                            native_chars.append({
                                "id": f"{page_id}-c{len(native_chars) + 1:07d}",
                                "text": char.get("c", ""),
                                "bbox": _canonical_bbox(char["bbox"], raw_width, raw_height, rotation),
                                "origin": _jsonable(char.get("origin")),
                                "span_id": span_id,
                                "source": {"kind": "native", "tool": "pymupdf"},
                            })

            images: list[dict[str, Any]] = []
            extracted_xrefs: set[int] = set()
            for image_index, image_info in enumerate(page.get_images(full=True)):
                xref = int(image_info[0])
                rects = page.get_image_rects(xref)
                asset_path: Path | None = None
                if xref not in extracted_xrefs:
                    try:
                        image = document.extract_image(xref)
                        extension = image.get("ext", "bin")
                        asset_path = embedded_root / f"xref-{xref}.{extension}"
                        asset_path.write_bytes(image["image"])
                        extracted_xrefs.add(xref)
                    except Exception as exc:  # keep observable geometry even if bytes fail
                        summary["warnings"].append({
                            "code": "IMAGE_EXTRACTION_FAILED",
                            "page_index": page_index,
                            "xref": xref,
                            "detail": str(exc),
                        })
                for occurrence, rect in enumerate(rects or [None]):
                    images.append({
                        "id": f"{page_id}-img{image_index + 1:04d}-{occurrence + 1}",
                        "xref": xref,
                        "bbox": (
                            _canonical_bbox(rect, raw_width, raw_height, rotation)
                            if rect is not None else None
                        ),
                        "width": image_info[2],
                        "height": image_info[3],
                        "colorspace": image_info[5],
                        "asset": relative_to(asset_path, run_root) if asset_path else None,
                        "source": {"kind": "native", "tool": "pymupdf"},
                    })

            drawings = []
            for drawing_index, drawing in enumerate(page.get_drawings()):
                serialized = _jsonable(drawing)
                serialized["id"] = f"{page_id}-draw{drawing_index + 1:05d}"
                if drawing.get("rect") is not None:
                    serialized["bbox"] = _canonical_bbox(
                        drawing["rect"], raw_width, raw_height, rotation
                    )
                serialized["source"] = {"kind": "native", "tool": "pymupdf"}
                drawings.append(serialized)

            plumber_geometry = {
                "chars": [
                    {
                        **_jsonable(char),
                        "bbox": _canonical_bbox(
                            (char["x0"], char["top"], char["x1"], char["bottom"]),
                            float(plumber_page.width),
                            float(plumber_page.height),
                            rotation,
                        ),
                    }
                    for char in plumber_page.chars
                ],
                "lines": _jsonable(plumber_page.lines),
                "rects": _jsonable(plumber_page.rects),
                "curves": _jsonable(plumber_page.curves),
                "source": {"kind": "native", "tool": "pdfplumber"},
            }

            page_value = {
                "id": page_id,
                "page_index": page_index,
                "width": width,
                "height": height,
                "raw_width": raw_width,
                "raw_height": raw_height,
                "rotation": rotation,
                "render": {
                    "path": relative_to(render_path, run_root),
                    "dpi": render_dpi,
                    "width_px": pixmap.width,
                    "height_px": pixmap.height,
                },
                "native_chars": native_chars,
                "native_spans": native_spans,
                "images": images,
                "drawings": drawings,
                "pdfplumber_geometry": plumber_geometry,
            }
            page_path = pages_root / f"{page_id}.json"
            write_json(page_path, page_value)
            summary["pages"].append({
                "id": page_id,
                "page_index": page_index,
                "width": width,
                "height": height,
                "path": relative_to(page_path, run_root),
                "render": page_value["render"],
                "counts": {
                    "chars": len(native_chars),
                    "spans": len(native_spans),
                    "images": len(images),
                    "drawings": len(drawings),
                },
            })
            logger.emit("extract", "RUN", f"page={page_index + 1}/{document.page_count}")

    document.close()
    write_json(run_root / "document.json", summary)
    logger.emit("extract", "DONE", f"pages={summary['page_count']}")
    return summary
