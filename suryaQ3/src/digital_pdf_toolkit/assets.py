from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .adapters.surya import SuryaRunReader
from .events import EventLogger
from .io import relative_to, write_json


COLORS = {
    "table": "#e63946", "figure": "#457b9d", "picture": "#457b9d",
    "image": "#457b9d", "formula": "#8338ec", "chart": "#fb8500", "text": "#2a9d8f",
}


def build_surya_assets(
    result_root: Path,
    images: list[dict[str, Any]],
    logger: EventLogger,
) -> dict[str, Any]:
    """Build crops and overlays using strict source-filename to global-page mapping."""
    assets_root = result_root / "assets"
    crops_root = assets_root / "crops"
    overlays_root = assets_root / "overlays"
    crops_root.mkdir(parents=True, exist_ok=True)
    overlays_root.mkdir(parents=True, exist_ok=True)
    logger.emit("assets", "START", "building Surya crops and overlays")
    page_map = {int(page["page_index"]): page for page in images}
    stems: dict[str, int] = {}
    for page_index, page in page_map.items():
        stem = Path(page["path"]).stem.casefold()
        if stem in stems:
            raise ValueError(f"Duplicate source image stem: {stem}")
        stems[stem] = page_index
    reader = SuryaRunReader(result_root / "results.json").with_page_mapping(stems)
    expected = set(page_map)
    if reader.page_indices() != expected:
        missing = sorted(expected - reader.page_indices())
        unexpected = sorted(reader.page_indices() - expected)
        raise ValueError(f"Surya filename mapping mismatch; missing={missing}, unexpected={unexpected}")
    index: dict[str, Any] = {
        "tool": "surya", "mode": "surya2", "pages": {},
        "regions": {}, "overlays": {}, "warnings": [],
    }
    for page_index, page in page_map.items():
        source_path = Path(page["path"])
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")
            overlay = source.copy()
            draw = ImageDraw.Draw(overlay)
            for region in reader.regions(page_index):
                try:
                    pixels = region.coordinates.to_pixels(source.width, source.height, source.width, source.height)
                    if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
                        raise ValueError(f"empty crop bbox: {pixels}")
                    crop_path = crops_root / f"{region.id}.png"
                    source.crop(pixels).save(crop_path)
                    color = COLORS.get(region.type, "#ffbe0b")
                    draw.rectangle(pixels, outline=color, width=3)
                    draw.text((pixels[0] + 2, pixels[1] + 2), region.type, fill=color)
                    index["regions"][region.id] = {
                        "page_index": page_index, "type": region.type,
                        "source_image": str(source_path.resolve()),
                        "tool_bbox": list(region.coordinates.bbox),
                        "coordinate_space": region.coordinates.space,
                        "render_bbox": list(pixels),
                        "exact_crop": relative_to(crop_path, result_root),
                        "text": region.text,
                        "reading_order": region.reading_order,
                        "polygon": region.polygon,
                        "raw_label": region.raw_label,
                        "confidence": region.confidence,
                        "skipped": region.skipped,
                        "error": region.error,
                        "provenance": region.provenance,
                    }
                except Exception as exc:
                    index["warnings"].append({"region_id": region.id, "detail": str(exc)})
            overlay_path = overlays_root / source_path.name
            overlay.save(overlay_path)
            index["pages"][str(page_index)] = {
                "source_image": str(source_path.resolve()),
                "batch": page.get("batch"),
                "batch_position": page.get("batch_position"),
            }
            index["overlays"][str(page_index)] = relative_to(overlay_path, result_root)
    write_json(assets_root / "index.json", index)
    logger.emit("assets", "DONE", f"regions={len(index['regions'])}")
    return index
