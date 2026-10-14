from __future__ import annotations

import html
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .io import read_json, sha256_file, stable_hash, write_json
from .sorting import natural_key


SCHEMA_VERSION = "1.2"
HEADER_TYPES = {"pageheader", "page-header", "header"}
FOOTER_TYPES = {"pagefooter", "page-footer", "footer"}
HEADING_TYPES = {"sectionheader", "section-header", "heading", "title"}


class Embedder(Protocol):
    model_id: str
    revision: str | None
    dimension: int
    dtype: str
    normalize_embeddings: bool

    def encode_texts(self, values: list[str]) -> Any: ...

    def encode_image(self, value: Path) -> Any: ...

    def inspect_image(self, value: Path) -> dict[str, Any]: ...

    def split_text(self, value: str) -> list[str]: ...

    def synchronize(self) -> None: ...


def discover_surya_indexes(input_path: Path) -> list[Path]:
    value = input_path.expanduser().resolve()
    if value.is_file():
        if value.name != "index.json" or value.parent.name != "assets":
            raise ValueError(f"Expected a Surya assets/index.json: {value}")
        return [value]
    if not value.is_dir():
        raise ValueError(f"Surya run or assets index does not exist: {value}")

    direct = [value / "assets" / "index.json", value / "result" / "assets" / "index.json"]
    found = [path for path in direct if path.is_file()]
    numeric = sorted(
        (path for path in value.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    batch_indexes = [path / "assets" / "index.json" for path in numeric]
    if found and batch_indexes:
        raise ValueError(f"Ambiguous Surya run layout: {value}")
    if batch_indexes:
        missing = [path for path in batch_indexes if not path.is_file()]
        if missing:
            raise ValueError(f"Missing batch assets index: {missing[0]}")
        return batch_indexes
    if len(found) == 1:
        return found
    raise ValueError(f"No Surya assets/index.json found under: {value}")


def _collapse_space(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _table_markdown(table: Any) -> str:
    rows: list[list[str]] = []
    for row in table.find_all("tr"):
        cells = [_collapse_space(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    escaped = [[cell.replace("|", "\\|") for cell in row] for row in rows]
    output = ["| " + " | ".join(escaped[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    output.extend("| " + " | ".join(row) + " |" for row in escaped[1:])
    return "\n".join(output)


def parse_surya_text(raw_html: str) -> tuple[str, str]:
    """Return normalized plain text and structure-aware, model-neutral text."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - dependency validation
        raise RuntimeError("Embedding preprocessing requires beautifulsoup4") from exc

    source = raw_html or ""
    soup = BeautifulSoup(source, "html.parser")
    plain_text = _collapse_space(soup.get_text(" ", strip=True))
    tables = [_table_markdown(table) for table in soup.find_all("table")]
    tables = [table for table in tables if table]
    lists = []
    for list_node in soup.find_all(["ul", "ol"]):
        ordered = list_node.name == "ol"
        items = []
        for index, item in enumerate(list_node.find_all("li", recursive=False), start=1):
            prefix = f"{index}." if ordered else "-"
            items.append(f"{prefix} {_collapse_space(item.get_text(' ', strip=True))}")
        if items:
            lists.append("\n".join(items))
    remainder = BeautifulSoup(source, "html.parser")
    for node in remainder.find_all(["table", "ul", "ol"]):
        node.decompose()
    ordinary = _collapse_space(remainder.get_text(" ", strip=True))
    structured = "\n\n".join(value for value in [ordinary, *lists, *tables] if value)
    return plain_text, structured or plain_text


def _normalized_type(region: dict[str, Any]) -> str:
    return str(region.get("type") or region.get("raw_label") or "unknown").strip().lower()


def _reading_order(region: dict[str, Any]) -> int:
    value = region.get("reading_order")
    if value is None:
        value = region.get("provenance", {}).get("block_index", 10**9)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 10**9


def _heading_level(raw_html: str, plain_text: str) -> int:
    match = re.search(r"<h([1-6])\b", raw_html, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    numbered = re.match(r"^\s*(\d+(?:\.\d+)*)[.、]?\s+", plain_text)
    return len(numbered.group(1).split(".")) if numbered else 1


def _update_heading_stack(stack: list[str], text: str, level: int) -> list[str]:
    level = max(1, level)
    stack = stack[: level - 1]
    while len(stack) < level - 1:
        stack.append("")
    stack.append(text)
    return [value for value in stack if value]


def _record_id(payload: dict[str, Any]) -> str:
    identity = {
        "scope": payload["metadata"].get("scope"),
        "region_ids": payload["metadata"].get("region_ids", []),
        "page_index": payload["metadata"].get("page_index"),
        "plain_text": payload.get("plain_text", ""),
        "chunk_index": payload["metadata"].get("chunk_index"),
    }
    return f"qwen3vl-{stable_hash(identity)[:20]}"


def _decorate_embedding_text(body: str, types: list[str], heading_path: list[str]) -> str:
    labels = [f"[type={','.join(types) if types else 'unknown'}]"]
    if heading_path:
        labels.append(f"[section={' > '.join(heading_path)}]")
    labels.append(body)
    return "\n".join(value for value in labels if value)


def _crop_path(index_path: Path, region: dict[str, Any]) -> Path | None:
    value = region.get("exact_crop")
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else index_path.parent.parent / path


def _readable_image(path: Path | None) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _visual_fingerprint(path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency validation
        raise RuntimeError("Visual deduplication requires Pillow and numpy") from exc

    resize = int(settings["resize"])
    hash_size = int(settings["hash_size"])
    if resize <= 0 or hash_size <= 0 or hash_size > resize:
        raise ValueError("visual_dedup requires 0 < hash_size <= resize")
    with Image.open(path) as source:
        width, height = source.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Image has invalid dimensions: {path}")
        gray = source.convert("L")
        hash_image = gray.resize((resize, resize), Image.Resampling.LANCZOS)
        pixels = np.asarray(hash_image, dtype=np.float64) / 255.0

        sharp_image = gray.copy()
        sharp_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        sharp_pixels = np.asarray(sharp_image, dtype=np.float32)

    positions = np.arange(resize, dtype=np.float64)
    frequencies = np.arange(resize, dtype=np.float64)[:, None]
    basis = np.cos(np.pi * (2.0 * positions + 1.0) * frequencies / (2.0 * resize))
    basis[0, :] *= 1.0 / np.sqrt(2.0)
    basis *= np.sqrt(2.0 / resize)
    coefficients = basis @ pixels @ basis.T
    low_frequency = coefficients[:hash_size, :hash_size]
    median_source = low_frequency.reshape(-1)[1:]
    threshold = float(np.median(median_source)) if median_source.size else float(low_frequency[0, 0])
    bits = (low_frequency > threshold).reshape(-1)
    hash_value = 0
    for bit in bits:
        hash_value = (hash_value << 1) | int(bit)

    if sharp_pixels.shape[0] >= 3 and sharp_pixels.shape[1] >= 3:
        center = sharp_pixels[1:-1, 1:-1]
        laplacian = (
            4.0 * center
            - sharp_pixels[:-2, 1:-1]
            - sharp_pixels[2:, 1:-1]
            - sharp_pixels[1:-1, :-2]
            - sharp_pixels[1:-1, 2:]
        )
        sharpness = float(np.var(laplacian))
    else:
        sharpness = 0.0
    bit_count = hash_size * hash_size
    return {
        "phash": hash_value,
        "phash_hex": f"{hash_value:0{(bit_count + 3) // 4}x}",
        "width": width,
        "height": height,
        "area": width * height,
        "aspect_ratio": width / height,
        "sharpness": sharpness,
    }


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _visual_match(left: dict[str, Any], right: dict[str, Any], settings: dict[str, Any]) -> bool:
    ratio_delta = abs(left["aspect_ratio"] - right["aspect_ratio"]) / max(
        left["aspect_ratio"], right["aspect_ratio"],
    )
    return (
        ratio_delta <= float(settings["aspect_ratio_tolerance"])
        and _hamming_distance(left["phash"], right["phash"])
        <= int(settings["hamming_threshold"])
    )


def _cluster_visual_candidates(
    candidates: list[dict[str, Any]], settings: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    ordered = sorted(candidates, key=lambda candidate: (
        int(candidate["region"].get("page_index", 0)),
        _reading_order(candidate["region"]),
        natural_key(Path(candidate["id"])),
    ))
    if not bool(settings.get("enabled", True)):
        return [[candidate] for candidate in ordered]
    clusters: list[list[dict[str, Any]]] = []
    for candidate in ordered:
        for cluster in clusters:
            if all(_visual_match(candidate["visual"], member["visual"], settings) for member in cluster):
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])
    return clusters


def _representative(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(entries, key=lambda entry: (
        -int(entry["visual"]["area"]),
        -float(entry["visual"]["sharpness"]),
        int(entry["region"].get("page_index", 0)),
        _reading_order(entry["region"]),
        natural_key(Path(entry["id"])),
    ))[0]


def _copy_crop_as_png(source: Path, destination: Path) -> None:
    if source.suffix.lower() == ".png":
        shutil.copy2(source, destination)
        return
    from PIL import Image
    with Image.open(source) as image:
        image.save(destination, format="PNG")


def _image_details(path: Path) -> dict[str, Any]:
    from PIL import Image
    with Image.open(path) as image:
        width, height = image.size
        return {
            "width": width,
            "height": height,
            "pixel_count": width * height,
            "mode": image.mode,
            "format": image.format,
        }


def _smart_resize(width: int, height: int, settings: dict[str, Any]) -> tuple[int, int]:
    factor = int(settings["factor"])
    min_pixels = int(settings["min_pixels"])
    max_pixels = int(settings["max_pixels"])
    if factor <= 0 or min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("image_preprocess requires factor > 0 and max_pixels >= min_pixels > 0")
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")

    resized_width = max(factor, round(width / factor) * factor)
    resized_height = max(factor, round(height / factor) * factor)
    if resized_width * resized_height > max_pixels:
        scale = math.sqrt((width * height) / max_pixels)
        resized_width = max(factor, math.floor(width / scale / factor) * factor)
        resized_height = max(factor, math.floor(height / scale / factor) * factor)
    elif resized_width * resized_height < min_pixels:
        scale = math.sqrt(min_pixels / (width * height))
        resized_width = max(factor, math.ceil(width * scale / factor) * factor)
        resized_height = max(factor, math.ceil(height * scale / factor) * factor)
    return resized_width, resized_height


def _materialize_embedding_input(
    source: Path, destination: Path, settings: dict[str, Any],
) -> dict[str, Any]:
    from PIL import Image, ImageOps
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        source_width, source_height = image.size
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        width, height = _smart_resize(source_width, source_height, settings)
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        image.save(destination, format="PNG")
    return {
        "width": width,
        "height": height,
        "pixel_count": width * height,
        "mode": "RGB",
        "format": "PNG",
    }


def _image_preprocess_settings(preprocess: dict[str, Any]) -> dict[str, Any]:
    settings = {
        "factor": 32,
        "patch_size": 16,
        "min_pixels": 4096,
        "max_pixels": 1310720,
    }
    settings.update(preprocess.get("image_preprocess") or {})
    _smart_resize(1, 1, settings)
    if int(settings["patch_size"]) <= 0:
        raise ValueError("image_preprocess.patch_size must be positive")
    return settings


def _base_metadata(
    scope: str,
    entries: list[dict[str, Any]],
    heading_path: list[str],
) -> dict[str, Any]:
    orders = [_reading_order(entry["region"]) for entry in entries]
    pages = sorted({int(entry["region"].get("page_index", 0)) for entry in entries})
    metadata: dict[str, Any] = {
        "scope": scope,
        "region_ids": [entry["id"] for entry in entries],
        "types": list(dict.fromkeys(_normalized_type(entry["region"]) for entry in entries)),
        "page_index": pages[0] if len(pages) == 1 else None,
        "reading_order_start": min(orders) if orders else None,
        "reading_order_end": max(orders) if orders else None,
        "heading_path": list(heading_path),
        "source_indexes": list(dict.fromkeys(str(entry["index_path"]) for entry in entries)),
        "skipped": any(bool(entry["region"].get("skipped")) for entry in entries),
        "error": next(
            (entry["region"].get("error") for entry in entries if entry["region"].get("error")),
            False,
        ),
    }
    if len(pages) > 1:
        metadata["page_indexes"] = pages
    return metadata


def _text_record(scope: str, entries: list[dict[str, Any]], heading_path: list[str]) -> dict[str, Any]:
    raw_html = "\n".join(str(entry["region"].get("text") or "") for entry in entries)
    parsed = [parse_surya_text(str(entry["region"].get("text") or "")) for entry in entries]
    plain_text = "\n".join(value[0] for value in parsed if value[0])
    body = "\n\n".join(value[1] for value in parsed if value[1])
    metadata = _base_metadata(scope, entries, heading_path)
    metadata["route"] = "text_vector"
    record = {
        "id": "",
        "embedding_text": _decorate_embedding_text(body, metadata["types"], heading_path),
        "plain_text": plain_text,
        "raw_html": raw_html,
        "metadata": metadata,
        "vector_ref": None,
        "_embedding_body": body,
    }
    record["id"] = _record_id(record)
    return record


def _provenance_record(entry: dict[str, Any], warning: str | None = None) -> dict[str, Any]:
    region = entry["region"]
    raw_html = str(region.get("text") or "")
    plain_text, structured = parse_surya_text(raw_html)
    metadata = _base_metadata("region", [entry], [])
    metadata["skipped"] = bool(region.get("skipped"))
    metadata["error"] = region.get("error")
    metadata["source_provenance"] = region.get("provenance")
    metadata["route"] = "provenance_only"
    if warning:
        metadata["warning"] = warning
    record = {
        "id": "",
        "embedding_text": _decorate_embedding_text(structured, metadata["types"], []),
        "plain_text": plain_text,
        "raw_html": raw_html,
        "metadata": metadata,
        "vector_ref": None,
    }
    record["id"] = _record_id(record)
    return record


def _image_record(
    entries: list[dict[str, Any]],
    representative: dict[str, Any],
    settings: dict[str, Any],
    aggregate: bool,
) -> dict[str, Any]:
    ordered = sorted(entries, key=lambda entry: (
        int(entry["region"].get("page_index", 0)),
        _reading_order(entry["region"]),
        natural_key(Path(entry["id"])),
    ))
    metadata = _base_metadata("document_root" if aggregate else "region", ordered, [])
    metadata["route"] = "image_vector"
    metadata["raw_labels"] = list(dict.fromkeys(
        str(entry["region"].get("raw_label"))
        for entry in ordered
        if entry["region"].get("raw_label") is not None
    ))
    representative_hash = int(representative["visual"]["phash"])
    metadata["source_regions"] = [
        {
            "region_id": entry["id"],
            "page_index": int(entry["region"].get("page_index", 0)),
            "reading_order": _reading_order(entry["region"]),
            "type": entry["region"].get("type"),
            "raw_label": entry["region"].get("raw_label"),
            "confidence": entry["region"].get("confidence"),
            "skipped": bool(entry["region"].get("skipped")),
            "error": entry["region"].get("error"),
            "crop": entry["region"].get("exact_crop"),
            "source_provenance": entry["region"].get("provenance"),
            "phash": entry["visual"]["phash_hex"],
            "hamming_distance_to_representative": _hamming_distance(
                int(entry["visual"]["phash"]), representative_hash,
            ),
        }
        for entry in ordered
    ]
    metadata["representative_region_id"] = representative["id"]
    metadata["representative_selection"] = (
        "largest_area_then_highest_laplacian_variance_then_earliest_page_"
        "reading_order_region_id"
    )
    metadata["visual_dedup"] = {
        "algorithm": f"phash-{int(settings['hash_size']) ** 2}",
        "linkage": "complete",
        "resize": int(settings["resize"]),
        "hash_size": int(settings["hash_size"]),
        "hamming_threshold": int(settings["hamming_threshold"]),
        "aspect_ratio_tolerance": float(settings["aspect_ratio_tolerance"]),
        "min_distinct_pages": int(settings["min_distinct_pages"]),
        "cluster_size": len(ordered),
        "distinct_page_count": len({int(entry["region"].get("page_index", 0)) for entry in ordered}),
    }
    record = {
        "id": "",
        "embedding_text": _decorate_embedding_text("", metadata["types"], []),
        "plain_text": "",
        "raw_html": "\n".join(str(entry["region"].get("text") or "") for entry in ordered),
        "metadata": metadata,
        "vector_ref": None,
        "_image_source": representative["_image_source"],
    }
    record["id"] = _record_id(record)
    return record


def _visual_settings(preprocess: dict[str, Any]) -> dict[str, Any]:
    settings = {
        "enabled": True,
        "hash_size": 8,
        "resize": 32,
        "hamming_threshold": 6,
        "aspect_ratio_tolerance": 0.05,
        "min_distinct_pages": 3,
    }
    settings.update(preprocess.get("visual_dedup") or {})
    if int(settings["hamming_threshold"]) < 0:
        raise ValueError("visual_dedup.hamming_threshold must be non-negative")
    if float(settings["aspect_ratio_tolerance"]) < 0:
        raise ValueError("visual_dedup.aspect_ratio_tolerance must be non-negative")
    if int(settings["min_distinct_pages"]) < 1:
        raise ValueError("visual_dedup.min_distinct_pages must be at least 1")
    return settings


def prepare_records(
    index_paths: list[Path], preprocess: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index_path in index_paths:
        payload = read_json(index_path)
        regions = payload.get("regions")
        if not isinstance(regions, dict):
            raise ValueError(f"Surya index regions must be an object: {index_path}")
        for region_id, region in regions.items():
            if region_id in seen_ids:
                raise ValueError(f"Duplicate Surya region ID: {region_id}")
            if not isinstance(region, dict):
                raise ValueError(f"Invalid Surya region {region_id}: {index_path}")
            seen_ids.add(region_id)
            entries.append({"id": region_id, "region": region, "index_path": index_path})

    entries.sort(key=lambda entry: (
        int(entry["region"].get("page_index", 0)),
        _reading_order(entry["region"]),
        natural_key(Path(entry["id"])),
    ))
    independent = {str(value).lower() for value in preprocess.get("independent_text_types", ["table", "form"])}
    aggregate_repeated = bool(preprocess.get("aggregate_repeated_regions", True))
    persist_headings = bool(preprocess.get("persist_headings_across_pages", True))
    text_enabled = bool(preprocess.get("text_when_nonempty", True))
    image_enabled = bool(preprocess.get("image_when_text_empty", True))
    visual_settings = _visual_settings(preprocess)
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    visual_candidates: list[dict[str, Any]] = []
    page_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    repeated: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    heading_stack: list[str] = []
    current_page: int | None = None

    for entry in entries:
        region = entry["region"]
        page_index = int(region.get("page_index", 0))
        if current_page is not None and page_index != current_page and not persist_headings:
            heading_stack = []
        current_page = page_index
        region_type = _normalized_type(region)
        raw_html = str(region.get("text") or "")
        plain_text, _ = parse_surya_text(raw_html)
        if region.get("error"):
            records.append(_provenance_record(entry))
            continue
        if plain_text and not text_enabled:
            warning = f"Region {entry['id']} text embedding is disabled"
            warnings.append(warning)
            records.append(_provenance_record(entry, warning))
            continue
        if region_type in HEADER_TYPES | FOOTER_TYPES and plain_text and aggregate_repeated:
            repeated[(region_type, _collapse_space(plain_text).casefold())].append(entry)
            continue
        if region_type in HEADING_TYPES and plain_text:
            level = _heading_level(raw_html, plain_text)
            heading_stack = _update_heading_stack(heading_stack, plain_text, level)
            continue
        if plain_text:
            if region_type in independent:
                records.append(_text_record("region", [entry], heading_stack))
            else:
                page_groups[int(region.get("page_index", 0))].append({**entry, "heading_path": list(heading_stack)})
            continue

        crop = _crop_path(entry["index_path"], region)
        if image_enabled and _readable_image(crop):
            candidate = {**entry, "_image_source": crop}
            candidate["visual"] = _visual_fingerprint(crop, visual_settings)
            visual_candidates.append(candidate)
        else:
            warning = (
                f"Region {entry['id']} image embedding is disabled"
                if not image_enabled
                else f"Region {entry['id']} has no usable text or crop"
            )
            warnings.append(warning)
            records.append(_provenance_record(entry, warning))

    clusters = _cluster_visual_candidates(visual_candidates, visual_settings)
    aggregate_cluster_count = 0
    deduplicated_region_count = 0
    for cluster in clusters:
        distinct_pages = {int(entry["region"].get("page_index", 0)) for entry in cluster}
        aggregate = (
            bool(visual_settings["enabled"])
            and len(distinct_pages) >= int(visual_settings["min_distinct_pages"])
        )
        if aggregate:
            aggregate_cluster_count += 1
            deduplicated_region_count += len(cluster) - 1
            records.append(_image_record(
                cluster, _representative(cluster), visual_settings, aggregate=True,
            ))
        else:
            records.extend(
                _image_record([entry], entry, visual_settings, aggregate=False)
                for entry in cluster
            )

    for page_index in sorted(page_groups):
        group = page_groups[page_index]
        if not group:
            continue
        heading_path = group[-1].get("heading_path", [])
        records.append(_text_record("page", group, heading_path))

    for (_, _), group in sorted(repeated.items(), key=lambda item: (_normalized_type(item[1][0]["region"]), item[0][1])):
        heading_path: list[str] = []
        records.append(_text_record("document_root", group, heading_path))

    records.sort(key=lambda record: (
        record["metadata"].get("page_index") if record["metadata"].get("page_index") is not None else -1,
        record["metadata"].get("reading_order_start") if record["metadata"].get("reading_order_start") is not None else -1,
        record["id"],
    ))
    visual_stats = {
        "visual_candidates": len(visual_candidates),
        "visual_clusters": len(clusters),
        "aggregate_visual_clusters": aggregate_cluster_count,
        "deduplicated_image_regions": deduplicated_region_count,
    }
    return records, warnings, visual_stats


def _as_matrix(value: Any, expected_rows: int, dimension: int, label: str) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency validation
        raise RuntimeError("Embedding output requires numpy") from exc
    matrix = np.asarray(value)
    if matrix.shape != (expected_rows, dimension):
        raise ValueError(f"{label} embeddings have shape {matrix.shape}; expected {(expected_rows, dimension)}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} embeddings contain non-finite values")
    return matrix


def _finalize_matrix(matrix: Any, dtype: str, normalize: bool, label: str) -> Any:
    import numpy as np
    matrix = matrix.astype(np.dtype(dtype), copy=False)
    if normalize and matrix.shape[0]:
        work = matrix.astype(np.float32, copy=False)
        norms = np.linalg.norm(work, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError(f"{label} embeddings contain a zero vector")
        matrix = (work / norms).astype(np.dtype(dtype), copy=False)
    return matrix


def _split_long_text_records(records: list[dict[str, Any]], embedder: Embedder) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        body = record.pop("_embedding_body", None)
        if record["metadata"].get("route") != "text_vector" or body is None:
            output.append(record)
            continue
        parts = [part.strip() for part in embedder.split_text(body) if part.strip()]
        if not parts:
            raise ValueError(f"Tokenizer produced no content for record {record['id']}")
        if len(parts) == 1:
            output.append(record)
            continue
        for index, part in enumerate(parts):
            chunk = deepcopy(record)
            chunk["metadata"]["chunk_index"] = index
            chunk["metadata"]["chunk_count"] = len(parts)
            chunk["embedding_text"] = _decorate_embedding_text(
                part, chunk["metadata"]["types"], chunk["metadata"]["heading_path"],
            )
            chunk["id"] = _record_id(chunk)
            output.append(chunk)
    return output


def _save_matrix(path: Path, matrix: Any) -> None:
    """Atomically replace a NumPy checkpoint."""
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            import numpy as np

            np.save(handle, matrix)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _model_checkpoint(embedder: Embedder) -> dict[str, Any]:
    return {
        "model_id": embedder.model_id,
        "revision": embedder.revision,
        "dimension": embedder.dimension,
        "dtype": embedder.dtype,
        "normalize_embeddings": embedder.normalize_embeddings,
    }


def _checkpoint_payload(
    indexes: list[Path],
    text_records: list[dict[str, Any]],
    image_records: list[dict[str, Any]],
    image_preprocessing: dict[str, Any],
    embedder: Embedder,
) -> dict[str, Any]:
    return {
        "sources": [
            {"index": str(path), "sha256": sha256_file(path)}
            for path in indexes
        ],
        "model": _model_checkpoint(embedder),
        "image_preprocessing": image_preprocessing,
        "text_record_ids": [record["id"] for record in text_records],
        "image_record_ids": [record["id"] for record in image_records],
    }


def _source_image_id(record: dict[str, Any]) -> str:
    value = record["metadata"].get(
        "source_image_id",
        record["metadata"].get(
            "representative_region_id", record["metadata"]["region_ids"][0],
        ),
    )
    if Path(value).name != value:
        raise ValueError(f"Unsafe Surya region ID for image filename: {value}")
    return value


def _expected_image_metadata(record: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    metadata = record["metadata"]
    return deepcopy(metadata.get("source_regions", [])), {
        "representative_selection": deepcopy(metadata.get("representative_selection")),
        "visual_dedup": deepcopy(metadata.get("visual_dedup")),
        "raw_labels": deepcopy(metadata.get("raw_labels", [])),
    }


def _bind_image_record(record: dict[str, Any], source_image_id: str) -> None:
    record["metadata"].pop("source_regions", None)
    record["metadata"].pop("representative_selection", None)
    record["metadata"].pop("visual_dedup", None)
    record["metadata"].pop("raw_labels", None)
    record["metadata"].pop("representative_region_id", None)
    record["metadata"]["source_image_id"] = source_image_id
    record["metadata"]["image_metadata_ref"] = (
        f"embedding_inputs/metadata.json#{source_image_id}"
    )


def _artifact_path(root: Path, relative: str, expected: str) -> Path:
    if relative != expected:
        raise ValueError(f"Resume artifact path mismatch: expected {expected}, got {relative}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Resume artifact escapes knowledge base: {relative}") from exc
    return path


def _validate_artifact(path: Path, descriptor: dict[str, Any], label: str) -> None:
    expected_hash = descriptor.get("sha256")
    if not path.is_file() or not expected_hash:
        raise ValueError(f"Resume {label} is missing or has no hash: {path}")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"Resume {label} hash mismatch: {path}")


def _load_resume_checkpoint(
    temporary: Path,
    image_metadata: dict[str, Any],
    checkpoint: dict[str, Any],
    text_records: list[dict[str, Any]],
    image_records: list[dict[str, Any]],
    embedder: Embedder,
) -> tuple[Any, Any, int]:
    import numpy as np

    stored_checkpoint = image_metadata.get("checkpoint")
    if stored_checkpoint is not None and stored_checkpoint != checkpoint:
        raise ValueError("Resume checkpoint does not match sources, model, or preprocessing")
    if image_metadata.get("image_preprocessing") != checkpoint["image_preprocessing"]:
        raise ValueError("Resume image preprocessing does not match resolved config")

    text_path = temporary / "vectors" / "text.npy"
    image_path = temporary / "vectors" / "image.npy"
    if not text_path.is_file() or not image_path.is_file():
        raise ValueError("Resume vectors are incomplete")
    text_matrix = np.load(text_path, allow_pickle=False)
    image_matrix = np.load(image_path, allow_pickle=False)
    expected_dtype = np.dtype(embedder.dtype)
    if text_matrix.shape != (len(text_records), embedder.dimension):
        raise ValueError(f"Resume text vector shape mismatch: {text_matrix.shape}")
    if text_matrix.dtype != expected_dtype:
        raise ValueError(f"Resume text vector dtype mismatch: {text_matrix.dtype}")
    if image_matrix.ndim != 2 or image_matrix.shape[1] != embedder.dimension:
        raise ValueError(f"Resume image vector shape mismatch: {image_matrix.shape}")
    if image_matrix.dtype != expected_dtype:
        raise ValueError(f"Resume image vector dtype mismatch: {image_matrix.dtype}")
    for row, record in enumerate(text_records):
        record["vector_ref"] = {"kind": "text_vector", "row": row}

    items = image_metadata.get("items")
    if not isinstance(items, list) or len(items) > len(image_records):
        raise ValueError("Resume image metadata item count is invalid")
    succeeded = 0
    found_incomplete = False
    for image_index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Resume image metadata item {image_index} is invalid")
        record = image_records[image_index]
        source = Path(record["_image_source"])
        source_image_id = _source_image_id(record)
        expected_regions, expected_dedup = _expected_image_metadata(record)
        if item.get("source_image_id") != source_image_id:
            raise ValueError(f"Resume image order mismatch at index {image_index}")
        if item.get("source_regions", []) != expected_regions:
            raise ValueError(f"Resume source regions mismatch for {source_image_id}")
        if item.get("deduplication") != expected_dedup:
            raise ValueError(f"Resume deduplication mismatch for {source_image_id}")
        source_crop = item.get("source_crop") or {}
        if source_crop.get("original_path") != str(source):
            raise ValueError(f"Resume source crop mismatch for {source_image_id}")
        crop_path = _artifact_path(
            temporary, str(source_crop.get("path", "")), f"crops/{source_image_id}.png",
        )
        inputs = item.get("embedding_inputs")
        if not isinstance(inputs, list) or len(inputs) != 1:
            raise ValueError(f"Resume embedding inputs are invalid for {source_image_id}")
        model_input = inputs[0]
        input_path = _artifact_path(
            temporary,
            str(model_input.get("path", "")),
            f"embedding_inputs/{source_image_id}-overview.png",
        )
        if model_input.get("embedding_index") != image_index:
            raise ValueError(f"Resume embedding index mismatch for {source_image_id}")
        status = model_input.get("status")
        if status == "succeeded":
            if found_incomplete:
                raise ValueError("Resume successful image appears after an incomplete image")
            _validate_artifact(crop_path, source_crop, "crop")
            _validate_artifact(input_path, model_input, "embedding input")
            if source.suffix.lower() == ".png" and sha256_file(source) != source_crop["sha256"]:
                raise ValueError(f"Resume source image changed for {source_image_id}")
            expected_ref = {"kind": "image_vector", "row": succeeded}
            if model_input.get("vector_ref") != expected_ref:
                raise ValueError(f"Resume vector row mismatch for {source_image_id}")
            record["vector_ref"] = expected_ref
            succeeded += 1
        elif status in {"failed", "pending"}:
            if found_incomplete or image_index != len(items) - 1:
                raise ValueError("Resume contains multiple incomplete images")
            found_incomplete = True
            if crop_path.exists() or source_crop.get("sha256"):
                _validate_artifact(crop_path, source_crop, "crop")
                if source.suffix.lower() == ".png" and sha256_file(source) != source_crop["sha256"]:
                    raise ValueError(f"Resume source image changed for {source_image_id}")
            if input_path.exists() or model_input.get("sha256"):
                _validate_artifact(input_path, model_input, "embedding input")
        else:
            raise ValueError(f"Resume image status is invalid for {source_image_id}: {status}")
        _bind_image_record(record, source_image_id)

    if image_matrix.shape[0] == succeeded + 1 and found_incomplete:
        image_matrix = image_matrix[:succeeded]
        _save_matrix(image_path, image_matrix)
    elif image_matrix.shape[0] != succeeded:
        raise ValueError(
            f"Resume image vector count mismatch: {image_matrix.shape[0]} != {succeeded}"
        )
    image_metadata["checkpoint"] = checkpoint
    image_metadata["schema_version"] = "1.1"
    image_metadata["status"] = "building"
    image_metadata.pop("failure", None)
    return text_matrix, image_matrix, succeeded


def build_knowledge_base(
    input_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    embedder: Embedder,
    *,
    resume: bool = False,
) -> Path:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency validation
        raise RuntimeError("Embedding output requires numpy") from exc

    indexes = discover_surya_indexes(input_path)
    preprocess = config.get("embedding_preprocess", {})
    image_preprocess = _image_preprocess_settings(preprocess)
    records, warnings, visual_stats = prepare_records(indexes, preprocess)
    records = _split_long_text_records(records, embedder)
    text_records = [record for record in records if record["metadata"].get("route") == "text_vector"]
    image_records = [record for record in records if record["metadata"].get("route") == "image_vector"]

    text_values = [record["embedding_text"] for record in text_records]
    vector_dtype = np.dtype(embedder.dtype)
    target = output_dir / "knowledge_base"
    temporary = output_dir / ".knowledge_base.tmp"
    failed = output_dir / "knowledge_base.failed"
    if target.exists():
        raise FileExistsError(target)
    image_preprocessing = {
        **image_preprocess,
        "maintain_aspect_ratio": True,
        "convert_rgb": True,
        "exif_transpose": True,
        "resample": "bicubic",
    }
    checkpoint = _checkpoint_payload(
        indexes, text_records, image_records, image_preprocessing, embedder,
    )
    if resume:
        if failed.exists() and temporary.exists():
            raise ValueError("Resume is ambiguous: both knowledge_base.failed and temporary exist")
        resume_root = failed if failed.exists() else temporary
        if not resume_root.exists():
            raise FileNotFoundError(f"No failed knowledge base to resume under: {output_dir}")
        resume_metadata_path = resume_root / "embedding_inputs" / "metadata.json"
        if not resume_metadata_path.is_file():
            raise ValueError("Resume image metadata is missing")
        image_metadata = read_json(resume_metadata_path)
        if not isinstance(image_metadata, dict):
            raise ValueError("Resume image metadata must be a JSON object")
        if resume_root == failed:
            failed.replace(temporary)
    else:
        if failed.exists():
            raise FileExistsError(failed)
        if temporary.exists():
            shutil.rmtree(temporary)
        (temporary / "vectors").mkdir(parents=True)
        (temporary / "crops").mkdir()
        (temporary / "embedding_inputs").mkdir()
        image_metadata = {
            "schema_version": "1.1",
            "status": "building",
            "image_preprocessing": image_preprocessing,
            "checkpoint": checkpoint,
            "items": [],
        }
    metadata_path = temporary / "embedding_inputs" / "metadata.json"
    current_input: dict[str, Any] | None = None
    try:
        if resume:
            text_matrix, image_matrix, completed_images = _load_resume_checkpoint(
                temporary, image_metadata, checkpoint, text_records, image_records, embedder,
            )
            write_json(metadata_path, image_metadata)
        else:
            text_matrix = (
                _as_matrix(
                    embedder.encode_texts(text_values), len(text_values), embedder.dimension, "text",
                )
                if text_values
                else np.empty((0, embedder.dimension), dtype=vector_dtype)
            )
            text_matrix = _finalize_matrix(
                text_matrix, embedder.dtype, embedder.normalize_embeddings, "text",
            )
            image_matrix = np.empty((0, embedder.dimension), dtype=vector_dtype)
            completed_images = 0
            for row, record in enumerate(text_records):
                record["vector_ref"] = {"kind": "text_vector", "row": row}
            _save_matrix(temporary / "vectors" / "text.npy", text_matrix)
            _save_matrix(temporary / "vectors" / "image.npy", image_matrix)
            write_json(metadata_path, image_metadata)

        for image_index, record in enumerate(image_records):
            source = Path(record["_image_source"])
            source_image_id = _source_image_id(record)
            crop_path = temporary / "crops" / f"{source_image_id}.png"
            model_path = temporary / "embedding_inputs" / f"{source_image_id}-overview.png"
            if image_index < len(image_metadata["items"]):
                item = image_metadata["items"][image_index]
                current_input = item["embedding_inputs"][0]
                if image_index < completed_images:
                    continue
                current_input["status"] = "pending"
                current_input["error"] = None
                current_input["vector_ref"] = None
            else:
                source_regions, deduplication = _expected_image_metadata(record)
                current_input = {
                    "kind": "overview",
                    "path": model_path.relative_to(temporary).as_posix(),
                    "width": None,
                    "height": None,
                    "pixel_count": None,
                    "mode": None,
                    "format": "PNG",
                    "sha256": None,
                    "embedding_index": image_index,
                    "model_input": None,
                    "vector_ref": None,
                    "status": "pending",
                    "error": None,
                }
                item = {
                    "source_image_id": source_image_id,
                    "source_crop": {
                        "path": crop_path.relative_to(temporary).as_posix(),
                        "original_path": str(source),
                        "width": None,
                        "height": None,
                        "pixel_count": None,
                        "mode": None,
                        "format": None,
                        "sha256": None,
                    },
                    "source_regions": source_regions,
                    "deduplication": deduplication,
                    "embedding_inputs": [current_input],
                }
                image_metadata["items"].append(item)
            _bind_image_record(record, source_image_id)
            write_json(metadata_path, image_metadata)

            try:
                if not crop_path.is_file() or not item["source_crop"].get("sha256"):
                    _copy_crop_as_png(source, crop_path)
                    item["source_crop"].update(_image_details(crop_path))
                    item["source_crop"]["sha256"] = sha256_file(crop_path)
                if not model_path.is_file() or not current_input.get("sha256"):
                    input_details = _materialize_embedding_input(
                        source, model_path, image_preprocess,
                    )
                    current_input.update(input_details)
                    current_input["sha256"] = sha256_file(model_path)
                else:
                    input_details = {
                        key: current_input[key]
                        for key in ("width", "height", "pixel_count", "mode", "format")
                    }
                write_json(metadata_path, image_metadata)
                model_input = embedder.inspect_image(model_path)
                expected_size = (input_details["width"], input_details["height"])
                effective_size = (
                    int(model_input["effective_width"]),
                    int(model_input["effective_height"]),
                )
                current_input["model_input"] = model_input
                if effective_size != expected_size:
                    raise ValueError(
                        f"Processor resized {source_image_id} from {expected_size} "
                        f"to {effective_size}"
                    )
                encoded_image = embedder.encode_image(model_path)
                embedder.synchronize()
                row_matrix = _as_matrix(
                    encoded_image, 1, embedder.dimension, "image",
                )
                row_matrix = _finalize_matrix(
                    row_matrix, embedder.dtype, embedder.normalize_embeddings, "image",
                )
                vector_row = image_matrix.shape[0]
                image_matrix = np.concatenate((image_matrix, row_matrix), axis=0)
                record["vector_ref"] = {"kind": "image_vector", "row": vector_row}
                current_input["vector_ref"] = record["vector_ref"]
                current_input["status"] = "succeeded"
                _save_matrix(temporary / "vectors" / "image.npy", image_matrix)
                write_json(metadata_path, image_metadata)
            except BaseException as exc:
                current_input["status"] = "failed"
                current_input["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                write_json(metadata_path, image_metadata)
                raise

        for record in records:
            record.pop("_image_source", None)
            record.pop("_embedding_body", None)

        records_path = temporary / "records.jsonl"
        with records_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

        image_metadata["status"] = "completed"
        write_json(metadata_path, image_metadata)
        file_hashes = {
            "records.jsonl": sha256_file(records_path),
            "vectors/text.npy": sha256_file(temporary / "vectors" / "text.npy"),
            "vectors/image.npy": sha256_file(temporary / "vectors" / "image.npy"),
        }
        for directory in (temporary / "crops", temporary / "embedding_inputs"):
            for artifact in sorted(path for path in directory.rglob("*") if path.is_file()):
                file_hashes[artifact.relative_to(temporary).as_posix()] = sha256_file(artifact)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "qwen3vl",
            "model": {
                "provider": "sentence_transformers",
                "model_id": embedder.model_id,
                "revision": embedder.revision,
                "dimension": embedder.dimension,
                "dtype": embedder.dtype,
                "normalize_embeddings": embedder.normalize_embeddings,
            },
            "sources": [
                {"index": str(path), "sha256": sha256_file(path)}
                for path in indexes
            ],
            "counts": {
                "records": len(records),
                "text_vectors": len(text_records),
                "image_vectors": len(image_records),
                "provenance_only": len(records) - len(text_records) - len(image_records),
                "warnings": len(warnings),
                **visual_stats,
            },
            "image_preprocessing": image_metadata["image_preprocessing"],
            "warnings": warnings,
            "files": file_hashes,
        }
        write_json(temporary / "manifest.json", manifest)
        temporary.replace(target)
    except BaseException as exc:
        if temporary.exists():
            image_metadata["status"] = "failed"
            image_metadata["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            if current_input is not None and current_input["status"] == "pending":
                current_input["status"] = "failed"
                current_input["error"] = image_metadata["failure"]
            write_json(metadata_path, image_metadata)
            temporary.replace(failed)
        raise
    return target
