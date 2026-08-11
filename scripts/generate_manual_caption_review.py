from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = REPO_ROOT / "runtime" / "index" / "current"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def table_text(value: object) -> str:
    text = html.escape(str(value), quote=False).replace("|", "&#124;")
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")

def build_review(snapshot_dir: Path) -> str:
    crops_dir = snapshot_dir / "crops"
    if not crops_dir.is_dir():
        raise FileNotFoundError(f"crop directory not found: {crops_dir}")

    records_path = snapshot_dir / "records.pretty.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"expected a JSON array: {records_path}")

    captions_by_artifact: dict[str, list[dict[str, object]]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("modality") != "image":
            continue
        source = record.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("artifact_path"), str):
            continue
        captions_by_artifact.setdefault(source["artifact_path"], []).append(record)

    images = sorted(
        path
        for path in crops_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    lines = [
        "# Manual Review",
        "",
        "### 記錄每張圖片問題",
        "",
        "|filename|Pic|type|caption|",
        "|---|---|---|---|",
    ]
    for image in images:
        artifact_path = f"crops/{image.name}"
        image_path = f"crops/{quote(image.name)}"
        blocks = captions_by_artifact.get(artifact_path, [])
        type_parts = []
        caption_parts = []
        for index, block in enumerate(blocks, start=1):
            image_metadata = block.get("image")
            detector_label = (
                image_metadata.get("detector_label", "")
                if isinstance(image_metadata, dict)
                else ""
            )
            type_parts.append(f"[{index}] {table_text(detector_label)}")
            caption_parts.append(
                f"[{index}] language: {table_text(block.get('language', ''))}"
                f"<br>content: {table_text(block.get('content', ''))}"
            )
        lines.append(
            f"|{table_text(image.name)}|![{image.name}]({image_path})|"
            f"{'<br>'.join(type_parts)}|{'<br><br>'.join(caption_parts)}|"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown checklist for manually reviewing crop captions."
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="snapshot directory containing crops/ (default: runtime/index/current)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing manual_caption_review.md",
    )
    args = parser.parse_args()

    snapshot_dir = args.snapshot.resolve()
    output_path = snapshot_dir / "manual_caption_review.md"
    if output_path.exists() and not args.force:
        parser.error(f"output already exists; use --force to overwrite: {output_path}")

    content = build_review(snapshot_dir)
    output_path.write_text(content, encoding="utf-8")
    image_count = content.count("|![")
    print(f"Wrote {output_path} with {image_count} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
