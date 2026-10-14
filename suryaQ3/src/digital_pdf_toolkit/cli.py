from __future__ import annotations

import argparse
import sys
from pathlib import Path


def toolkit_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="digital-pdf-toolkit")
    modes = parser.add_subparsers(dest="command", required=True)

    extract = modes.add_parser("extract", help="Extract native facts from one PDF")
    extract.add_argument("input", type=Path)
    extract.add_argument("--config", type=Path)

    surya2 = modes.add_parser("surya2", help="Run Surya against a PNG folder or one PDF")
    surya2.add_argument("input", type=Path)
    surya2.add_argument("--config", type=Path)

    marker = modes.add_parser("marker", help="Run Marker against one PDF or PNG folder")
    marker.add_argument("input", type=Path)
    marker.add_argument("--config", type=Path)
    qwen3vl = modes.add_parser("qwen3vl", help="Build an embedding knowledge base from Surya assets")
    qwen3vl.add_argument("input", type=Path)
    qwen3vl.add_argument("--config", type=Path)
    qwen3vl.add_argument("--resume-from", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .runs import run_extract, run_marker, run_qwen3vl, run_surya2

    try:
        if args.command == "extract":
            output = run_extract(toolkit_root(), args.input, args.config)
            print(f"Output: {output}")
            return 0
        if args.command == "marker":
            output = run_marker(toolkit_root(), args.input, args.config)
            print(f"Output: {output}")
            return 0
        if args.command == "qwen3vl":
            output = run_qwen3vl(
                toolkit_root(), args.input, args.config, resume_from=args.resume_from,
            )
            print(f"Output: {output}")
            return 0
        output, failed_batches = run_surya2(toolkit_root(), args.input, args.config)
        print(f"Output: {output}")
        return 1 if failed_batches else 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
if __name__ == "__main__":
    raise SystemExit(main())
