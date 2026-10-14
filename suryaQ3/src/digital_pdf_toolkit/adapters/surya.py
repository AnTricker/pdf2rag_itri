from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from typing import Any, Iterable

from digital_pdf_toolkit.coordinates import RegionCoordinates, normalize_bbox

from .base import AnalysisRegion, AnalysisRunReader
from .helpers import box_from
from digital_pdf_toolkit.io import read_json


class SuryaRunReader(AnalysisRunReader):
    tool_name = "surya"

    def __init__(self, results_path: Path):
        super().__init__(results_path.parent)
        if results_path.name != "results.json" or not results_path.is_file():
            raise FileNotFoundError(f"Expected normalized Surya result: {results_path}")
        self.path = results_path
        self.document = read_json(results_path)
        self._entries = list(self._page_values())
        self._entry_indices = {
            index: inferred for index, (_, inferred, _, _) in enumerate(self._entries)
        }
        self._regions = list(self._parse())

    def _page_values(self) -> Iterable[tuple[str, int, int, dict[str, Any]]]:
        if isinstance(self.document, dict):
            fallback = 0
            for key, value in self.document.items():
                raw_page_key = str(key)
                match = re.search(r"p(\d{4})", raw_page_key)
                inferred = int(match.group(1)) - 1 if match else fallback
                if isinstance(value, list):
                    for offset, page in enumerate(value):
                        if isinstance(page, dict):
                            yield raw_page_key, inferred + offset, offset, page
                            fallback += 1
        elif isinstance(self.document, list):
            yield from (
                (str(index), index, index, page)
                for index, page in enumerate(self.document)
                if isinstance(page, dict)
            )

    def _parse(self) -> Iterable[AnalysisRegion]:
        sequence = 0
        for entry_index, (raw_page_key, _, page_entry_index, page) in enumerate(self._entries):
            page_index = self._entry_indices[entry_index]
            image_bbox = box_from(page.get("image_bbox")) or [0.0, 0.0, 1.0, 1.0]
            source_width = image_bbox[2] - image_bbox[0]
            source_height = image_bbox[3] - image_bbox[1]
            for block_index, box in enumerate(page.get("bboxes", page.get("blocks", []))):
                if not isinstance(box, dict):
                    continue
                bbox = box_from(box.get("bbox") or box.get("polygon"))
                if bbox is None:
                    continue
                sequence += 1
                yield AnalysisRegion(
                    id=f"surya-p{page_index}-r{sequence}",
                    page_index=page_index,
                    type=str(box.get("label", "unknown")).lower(),
                    coordinates=RegionCoordinates(
                        normalize_bbox(bbox), "image_px", source_width, source_height
                    ),
                    text=box.get("text") or box.get("html"),
                    polygon=box.get("polygon"),
                    reading_order=box.get("reading_order", box.get("position")),
                    raw_label=box.get("raw_label"),
                    confidence=box.get("confidence"),
                    skipped=box.get("skipped"),
                    error=box.get("error"),
                    provenance={
                        "tool": "surya",
                        "raw_file": str(self.path),
                        "raw_page_key": raw_page_key,
                        "entry_index": entry_index,
                        "page_entry_index": page_entry_index,
                        "block_index": block_index,
                    },
                )

    def with_page_mapping(self, mapping: dict[str, int]) -> "SuryaRunReader":
        normalized = {str(key).casefold(): value for key, value in mapping.items()}
        resolved: dict[int, int] = {}
        seen: set[str] = set()
        for entry_index, (raw_page_key, _, _, _) in enumerate(self._entries):
            stem = Path(raw_page_key).stem.casefold()
            if stem in seen:
                raise ValueError(f"Duplicate Surya result stem: {stem}")
            seen.add(stem)
            if stem not in normalized:
                raise ValueError(f"Surya result has no source image mapping: {raw_page_key}")
            resolved[entry_index] = normalized[stem]
        self._entry_indices = resolved
        self._regions = [
            replace(
                region,
                id=f"surya-p{resolved[int(region.provenance['entry_index'])]}-r{sequence}",
                page_index=resolved[int(region.provenance["entry_index"])],
            )
            for sequence, region in enumerate(self._regions, start=1)
        ]
        return self

    def page_indices(self) -> set[int]:
        return set(self._entry_indices.values())

    def pages(self) -> list[int]:
        return sorted({region.page_index for region in self._regions})

    def regions(self, page_index: int | None = None) -> Iterable[AnalysisRegion]:
        return [region for region in self._regions if page_index is None or region.page_index == page_index]
