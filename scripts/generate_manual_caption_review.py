from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "runtime" / "outputs"
OUTPUT_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}$")
REVIEW_PATTERN = re.compile(r"^(\d+)_records\.pretty\.json$")


def table_text(value: object) -> str:
    text = html.escape(str(value), quote=False).replace("|", "&#124;")
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def latest_review(pending_root: Path) -> tuple[int, Path]:
    candidates = []
    for path in pending_root.iterdir():
        match = REVIEW_PATTERN.fullmatch(path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError("output has no review revisions")
    return max(candidates, key=lambda item: item[0])


def caption_text(record: dict[str, object]) -> str:
    content_type = record.get("content_type")
    if content_type == "table":
        return table_text(
            json.dumps(
                {
                    "extraction": record.get("table_extraction"),
                    "summary": record.get("table_summary"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    if content_type == "figure":
        return table_text(
            json.dumps(
                record.get("figure_extraction"),
                ensure_ascii=False,
                indent=2,
            )
        )
    return ""


def build_review(review_path: Path) -> str:
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"expected review envelope records: {review_path}")

    lines = [
        "# Manual Review",
        "",
        f"Source: {review_path.name}",
        "",
        "|filename|Pic|type|caption|",
        "|---|---|---|---|",
    ]
    for record in records:
        if not isinstance(record, dict) or record.get("modality") != "image":
            continue
        source = record.get("source")
        if not isinstance(source, dict):
            continue
        artifact_path = source.get("artifact_path")
        if not isinstance(artifact_path, str):
            continue
        filename = Path(artifact_path).name
        image_path = quote(artifact_path.replace("\\", "/"))
        lines.append(
            f"|{table_text(filename)}|![{filename}]({image_path})|"
            f"{table_text(record.get('content_type') or '')}|"
            f"{caption_text(record)}|"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Markdown from a reviewable output revision."
    )
    parser.add_argument("--output", required=True, help="timestamp output identifier")
    parser.add_argument("--review", type=int, help="review version; defaults to latest")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
        help="configured outputs root",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not OUTPUT_PATTERN.fullmatch(args.output):
        parser.error("output must be a timestamp identifier")
    outputs_root = args.outputs_root.resolve()
    output_root = (outputs_root / args.output).resolve(strict=True)
    if output_root.parent != outputs_root:
        parser.error("output escapes outputs root")

    if args.review is None:
        version, review_path = latest_review(output_root / "pending")
    else:
        version = args.review
        review_path = (
            output_root / "pending" / f"{version:03d}_records.pretty.json"
        )
        if not review_path.is_file():
            parser.error(f"review not found: {review_path}")

    output_path = output_root / f"manual_caption_review_{version:03d}.md"
    if output_path.exists() and not args.force:
        parser.error(f"output already exists; use --force: {output_path}")
    output_path.write_text(build_review(review_path), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())