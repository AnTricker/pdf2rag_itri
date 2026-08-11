from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional, Protocol


class IngestionApplication(Protocol):
    def ingest(self, input_pdf: Path, *, mode: Optional[str] = None) -> Path: ...


def run_cli(
    argv: Optional[Sequence[str]] = None,
    *,
    application_factory: Optional[Callable[[], IngestionApplication]] = None,
) -> int:
    parser = argparse.ArgumentParser(prog="local-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--input", required=True, type=Path)
    ingest_parser.add_argument("--mode", choices=("text", "image", "multi"))
    arguments = parser.parse_args(argv)

    if application_factory is None:
        raise RuntimeError("default application wiring has not been configured")
    application = application_factory()
    if arguments.command == "ingest":
        application.ingest(arguments.input, mode=arguments.mode)
        return 0
    return 2
