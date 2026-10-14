from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from digital_pdf_toolkit.io import read_json


def json_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def load_first(root: Path, preferred_patterns: list[str]) -> tuple[Path, Any]:
    files = json_files(root)
    for pattern in preferred_patterns:
        for path in files:
            if path.match(pattern) or path.name == pattern:
                return path, read_json(path)
    if not files:
        raise FileNotFoundError(f"No JSON result found under {root}")
    return files[0], read_json(files[0])


def walk(value: Any) -> Iterator[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def box_from(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        return [float(item) for item in value]
    if len(value) == 4 and all(isinstance(item, (list, tuple)) and len(item) >= 2 for item in value):
        xs = [float(item[0]) for item in value]
        ys = [float(item[1]) for item in value]
        return [min(xs), min(ys), max(xs), max(ys)]
    if len(value) == 8 and all(isinstance(item, (int, float)) for item in value):
        xs = [float(value[index]) for index in range(0, 8, 2)]
        ys = [float(value[index]) for index in range(1, 8, 2)]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None

