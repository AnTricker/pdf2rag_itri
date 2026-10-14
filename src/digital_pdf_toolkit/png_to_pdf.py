from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .sorting import natural_key


def convert_folder(input_dir: Path, output_pdf: Path | None = None) -> Path:
    """Combine naturally ordered top-level PNG files into one PDF."""
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input is not a folder: {input_dir}")

    sources = sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"),
        key=natural_key,
    )
    if not sources:
        raise ValueError(f"No top-level .png files found in: {input_dir}")

    target = (output_pdf or input_dir / f"{input_dir.name}.pdf").expanduser().resolve()
    if target.exists():
        raise ValueError(f"Output PDF already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PNG to PDF conversion requires PyMuPDF") from exc

    output = fitz.open()
    try:
        for source in sources:
            image_document = fitz.open(source)
            try:
                pdf_bytes = image_document.convert_to_pdf()
            finally:
                image_document.close()
            image_pdf = fitz.open("pdf", pdf_bytes)
            try:
                output.insert_pdf(image_pdf)
            finally:
                image_pdf.close()
        output.save(target)
    finally:
        output.close()
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="png-to-pdf",
        description="Combine one folder of naturally ordered PNG files into one PDF.",
    )
    parser.add_argument("input", type=Path, help="Folder containing top-level PNG files")
    parser.add_argument("--output", type=Path, help="Output PDF (default: <input>/<folder-name>.pdf)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = convert_folder(args.input, args.output)
        print(f"Output PDF: {output}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
