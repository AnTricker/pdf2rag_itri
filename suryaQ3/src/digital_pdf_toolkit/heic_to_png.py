from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

from .sorting import natural_key


def convert_folder(input_dir: Path, output_dir: Path | None = None) -> list[Path]:
    """Convert top-level HEIC/HEIF files into naturally ordered RGB PNG pages."""
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input is not a folder: {input_dir}")

    sources = sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in {".heic", ".heif"}),
        key=natural_key,
    )
    if not sources:
        raise ValueError(f"No top-level .heic or .heif files found in: {input_dir}")

    destination = (output_dir or input_dir / "scan_source_pages").resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Output folder must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    try:
        from pillow_heif import register_heif_opener
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("HEIC conversion requires pillow-heif") from exc
    register_heif_opener()

    outputs: list[Path] = []
    for index, source in enumerate(sources, start=1):
        target = destination / f"p{index:04d}.png"
        with Image.open(source) as opened:
            normalized = ImageOps.exif_transpose(opened).convert("RGB")
            normalized.save(target, format="PNG")
        outputs.append(target)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="heic-to-png",
        description="Convert one folder of HEIC/HEIF images into ordered PNG pages.",
    )
    parser.add_argument("input", type=Path, help="Folder containing top-level HEIC/HEIF files")
    parser.add_argument("--output", type=Path, help="Output folder (default: <input>/scan_source_pages)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = convert_folder(args.input, args.output)
        print(f"Output folder: {outputs[0].parent}")
        print(f"Converted pages: {len(outputs)}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
