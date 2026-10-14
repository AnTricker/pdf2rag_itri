from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from digital_pdf_toolkit.coordinates import RegionCoordinates


@dataclass(frozen=True)
class AnalysisRegion:
    id: str
    page_index: int
    type: str
    coordinates: RegionCoordinates
    text: str | None = None
    polygon: list[list[float]] | None = None
    reading_order: int | None = None
    raw_label: str | None = None
    confidence: float | None = None
    skipped: bool | None = None
    error: bool | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


class AnalysisRunReader(ABC):
    tool_name: str

    def __init__(self, result_root: Path):
        self.result_root = result_root

    @abstractmethod
    def pages(self) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def regions(self, page_index: int | None = None) -> Iterable[AnalysisRegion]:
        raise NotImplementedError

    def text(self, region_id: str) -> str | None:
        for region in self.regions():
            if region.id == region_id:
                return region.text
        raise KeyError(region_id)

    def provenance(self, region_id: str) -> dict[str, Any]:
        for region in self.regions():
            if region.id == region_id:
                return region.provenance
        raise KeyError(region_id)

    def crop(self, region_id: str, assets_index: dict[str, Any]) -> str | None:
        return assets_index.get("regions", {}).get(region_id, {}).get("exact_crop")
