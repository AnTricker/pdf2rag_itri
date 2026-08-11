from __future__ import annotations

import shutil
from pathlib import Path

import requests

from .config import AppConfig


def run_doctor(config: AppConfig) -> tuple[bool, list[dict[str, object]]]:
    checks: list[dict[str, object]] = []

    def add(name: str, ready: bool, detail: str) -> None:
        checks.append({"name": name, "ready": ready, "detail": detail})

    add("outputs_root", config.index_root.parent.exists(), str(config.index_root))
    input_ready = config.input_pdf is None or config.input_pdf.is_file()
    add("input_pdf", input_ready, str(config.input_pdf) if config.input_pdf else "not configured")
    image_enabled = config.ingest_mode in {"image", "multi"}
    if image_enabled:
        try:
            if config.poppler_path:
                poppler_ready = (config.poppler_path / "pdftoppm.exe").is_file() or (config.poppler_path / "pdftoppm").is_file()
            else:
                poppler_ready = shutil.which("pdftoppm") is not None
        except OSError:
            poppler_ready = False
        add("poppler", poppler_ready, str(config.poppler_path or "PATH"))
        detector_ready = config.detector_model is not None and config.detector_model.is_file()
        add("detector_model", detector_ready, str(config.detector_model or "not configured"))
    else:
        add("poppler", True, "not required in text mode")
        add("detector_model", True, "not required in text mode")
    embedding_ready = bool(config.embedding_model) and Path(config.embedding_model).expanduser().exists()
    add("embedding_model", embedding_ready, config.embedding_model or "not configured")

    for name, url, model in (
        ("llm", config.llm_url, config.llm_model),
        ("vlm", config.vlm_url, config.vlm_model),
    ):
        if name == "vlm" and not image_enabled:
            add(name, True, "not required in text mode")
            continue
        if not model:
            add(name, False, "not configured")
            continue
        try:
            response = requests.get(f"{url}/api/tags", timeout=5)
            response.raise_for_status()
            names = [item.get("name", "") for item in response.json().get("models", [])]
            ready = any(item == model or item.startswith(f"{model}:") for item in names)
            add(name, ready, f"model={model}")
        except requests.RequestException:
            add(name, False, "endpoint unavailable")
    return all(bool(item["ready"]) for item in checks), checks
