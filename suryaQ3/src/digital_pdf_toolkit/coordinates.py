from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

BBox = tuple[float, float, float, float]


def normalize_bbox(values: Iterable[float]) -> BBox:
    x0, y0, x1, y1 = (float(value) for value in values)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def rotate_bbox(bbox: Iterable[float], width: float, height: float, rotation: int) -> BBox:
    """Rotate a top-left-origin bbox clockwise into displayed-page coordinates."""
    x0, y0, x1, y1 = normalize_bbox(bbox)
    rotation = rotation % 360
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))

    def rotate(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        if rotation == 0:
            return x, y
        if rotation == 90:
            return height - y, x
        if rotation == 180:
            return width - x, height - y
        if rotation == 270:
            return y, width - x
        raise ValueError(f"Unsupported page rotation: {rotation}")

    rotated = [rotate(point) for point in corners]
    return normalize_bbox((
        min(point[0] for point in rotated),
        min(point[1] for point in rotated),
        max(point[0] for point in rotated),
        max(point[1] for point in rotated),
    ))


def displayed_size(width: float, height: float, rotation: int) -> tuple[float, float]:
    return (height, width) if rotation % 180 else (width, height)


def bbox_contains(outer: Iterable[float], inner: Iterable[float], tolerance: float = 0.0) -> bool:
    ox0, oy0, ox1, oy1 = normalize_bbox(outer)
    ix0, iy0, ix1, iy1 = normalize_bbox(inner)
    return (
        ix0 >= ox0 - tolerance
        and iy0 >= oy0 - tolerance
        and ix1 <= ox1 + tolerance
        and iy1 <= oy1 + tolerance
    )


@dataclass(frozen=True)
class RegionCoordinates:
    bbox: BBox
    space: str
    source_width: float | None = None
    source_height: float | None = None

    def to_pixels(
        self,
        render_width: int,
        render_height: int,
        page_width_pt: float,
        page_height_pt: float,
    ) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.bbox
        if self.space == "canonical_pt":
            sx, sy = render_width / page_width_pt, render_height / page_height_pt
        elif self.space == "normalized_1":
            sx, sy = render_width, render_height
        elif self.space == "normalized_1000":
            sx, sy = render_width / 1000.0, render_height / 1000.0
        elif self.space == "image_px":
            if not self.source_width or not self.source_height:
                raise ValueError("image_px coordinates require source dimensions")
            sx, sy = render_width / self.source_width, render_height / self.source_height
        else:
            raise ValueError(f"Unknown coordinate space: {self.space}")
        result = (
            max(0, round(x0 * sx)),
            max(0, round(y0 * sy)),
            min(render_width, round(x1 * sx)),
            min(render_height, round(y1 * sy)),
        )
        return result

