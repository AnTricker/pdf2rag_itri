from __future__ import annotations

import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .assets import build_surya_assets
from .config import load_config
from .events import EventLogger
from .extract import extract_document
from .embedding import discover_surya_indexes
from .sorting import natural_key
from .io import read_json, write_json
from .metadata import utc_now, write_metadata
from .png_to_pdf import convert_folder as convert_png_folder
from .process import ProcessResult, run_process


BATCH_SIZE = 10


def _output_root(toolkit_root: Path, config: dict[str, Any]) -> Path:
    value = Path(config["project"]["output_root"]).expanduser()
    return value if value.is_absolute() else toolkit_root / value


def allocate_run_root(toolkit_root: Path, config: dict[str, Any], mode: str) -> Path:
    output_root = _output_root(toolkit_root, config)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%m%d%H%M")
    base = f"{stamp}_{mode}"
    candidate = output_root / base
    suffix = 2
    while candidate.exists():
        candidate = output_root / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def _environment_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "core_environment": config["conda"]["core_env"],
    }


def run_extract(toolkit_root: Path, input_pdf: Path, custom_config: Path | None = None) -> Path:
    input_pdf = input_pdf.expanduser().resolve()
    if not input_pdf.is_file() or input_pdf.suffix.lower() != ".pdf":
        raise ValueError(f"extract requires one PDF file: {input_pdf}")
    config = load_config(toolkit_root, custom_config)
    run_root = allocate_run_root(toolkit_root, config, "extract")
    logger = EventLogger(run_root, config["logging"].get("redact_keys"))
    command = ["extract", str(input_pdf)] + (["--config", str(custom_config)] if custom_config else [])
    write_json(run_root / "command.json", {"command": command, "resolved_config": config})
    write_json(run_root / "environment.json", _environment_payload(config))
    status: dict[str, Any] = {
        "mode": "extract", "state": "running", "input": str(input_pdf),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_root / "status.json", status)
    try:
        extract_document(input_pdf, run_root, int(config["project"]["render_dpi"]), logger)
        status.update({"state": "completed", "finished_at": datetime.now(timezone.utc).isoformat()})
        write_json(run_root / "status.json", status)
        return run_root
    except Exception as exc:
        status.update({"state": "failed", "error": str(exc), "finished_at": datetime.now(timezone.utc).isoformat()})
        write_json(run_root / "status.json", status)
        logger.emit("extract", "FAILED", str(exc), level="ERROR")
        raise


def _pngs(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".png"),
        key=natural_key,
    )


def _ensure_unique_stems(paths: list[Path]) -> None:
    seen: dict[str, Path] = {}
    for path in paths:
        stem = path.stem.casefold()
        if stem in seen:
            raise ValueError(f"Duplicate PNG stems are not supported: {seen[stem].name}, {path.name}")
        seen[stem] = path


def _records(batch_paths: list[tuple[str | None, list[Path]]]) -> list[tuple[str | None, list[dict[str, Any]]]]:
    flat = [path for _, paths in batch_paths for path in paths]
    _ensure_unique_stems(flat)
    page_index = 0
    output: list[tuple[str | None, list[dict[str, Any]]]] = []
    for batch, paths in batch_paths:
        values = []
        for position, path in enumerate(paths, start=1):
            values.append({
                "path": str(path.resolve()), "filename": path.name, "page_index": page_index,
                "batch": batch, "batch_position": position,
            })
            page_index += 1
        output.append((batch, values))
    return output


def prepare_png_batches(input_dir: Path) -> list[tuple[str | None, list[dict[str, Any]]]]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"PNG input is not a folder: {input_dir}")
    top_level = _pngs(input_dir)
    numeric = sorted(
        (path for path in input_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if top_level and numeric:
        raise ValueError("PNG folder cannot mix top-level PNG files with numeric batch folders")
    if numeric:
        expected_names = [str(index) for index in range(1, len(numeric) + 1)]
        if [path.name for path in numeric] != expected_names:
            raise ValueError(f"Batch folders must be consecutively named: {expected_names}")
        values: list[tuple[str | None, list[Path]]] = []
        for directory in numeric:
            paths = _pngs(directory)
            if not 1 <= len(paths) <= BATCH_SIZE:
                raise ValueError(f"Batch {directory.name} must contain 1-10 PNG files")
            values.append((directory.name, paths))
        return _records(values)
    if not top_level:
        raise ValueError(f"No top-level PNG files or numeric batches found in: {input_dir}")
    _ensure_unique_stems(top_level)
    if len(top_level) <= BATCH_SIZE:
        return _records([(None, top_level)])
    batches: list[tuple[str | None, list[Path]]] = []
    for offset in range(0, len(top_level), BATCH_SIZE):
        name = str(offset // BATCH_SIZE + 1)
        destination = input_dir / name
        destination.mkdir(exist_ok=False)
        moved: list[Path] = []
        for source in top_level[offset:offset + BATCH_SIZE]:
            target = destination / source.name
            source.replace(target)
            moved.append(target)
        batches.append((name, moved))
    return _records(batches)


def render_pdf(pdf_path: Path, rendered_root: Path, dpi: int) -> list[Path]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - real core environment
        raise RuntimeError("PDF rendering requires PyMuPDF") from exc
    rendered_root.mkdir(parents=True, exist_ok=False)
    document = fitz.open(pdf_path)
    try:
        if document.needs_pass:
            raise PermissionError("PDF requires a password")
        outputs = []
        for index, page in enumerate(document, start=1):
            target = rendered_root / f"p{index:04d}.png"
            page.get_pixmap(dpi=dpi, alpha=False).save(target)
            outputs.append(target)
        if not outputs:
            raise ValueError("PDF contains no pages")
        return outputs
    finally:
        document.close()


def _normalize_results(result_root: Path) -> Path:
    matches = list(result_root.rglob("results.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Surya results.json, found {len(matches)}")
    source = matches[0]
    target = result_root / "results.json"
    if source != target:
        if target.exists():
            raise ValueError(f"Cannot normalize duplicate Surya result: {target}")
        shutil.move(str(source), str(target))
        parent = source.parent
        while parent != result_root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    return target


def _native_command(config: dict[str, Any], input_dir: Path, result_root: Path) -> tuple[list[str], list[str]]:
    replacements = {"input_dir": str(input_dir.resolve()), "output_dir": str(result_root.resolve())}
    native = [str(part).format(**replacements) for part in config["surya2"]["command"]]
    command = [
        "conda", "run", "--no-capture-output", "-n",
        config["surya2"]["environment"], *native,
    ]
    return command, native


def _validate_config(config: dict[str, Any]) -> None:
    if float(config["metadata"]["sampling_interval_seconds"]) <= 0:
        raise ValueError("metadata.sampling_interval_seconds must be greater than zero")
    if int(config["project"]["heartbeat_seconds"]) <= 0:
        raise ValueError("project.heartbeat_seconds must be greater than zero")
    command = config["surya2"].get("command")
    if not isinstance(command, list) or not command:
        raise ValueError("surya2.command must be a non-empty list")


def _marker_command(config: dict[str, Any], input_pdf: Path, result_root: Path) -> tuple[list[str], list[str]]:
    replacements = {
        "worker_script": str((Path(__file__).parent / "marker_worker.py").resolve()),
        "input_pdf": str(input_pdf.resolve()),
        "output_dir": str(result_root.resolve()),
        "mode": str(config["marker"]["mode"]),
        "inference_backend": str(config["marker"]["inference_backend"]),
    }
    native = [str(part).format(**replacements) for part in config["marker"]["command"]]
    command = [
        "conda", "run", "--no-capture-output", "-n",
        config["marker"]["environment"], *native,
    ]
    return command, native


def _embedding_command(
    config: dict[str, Any], input_path: Path, run_root: Path, config_json: Path,
    *, resume: bool = False,
) -> tuple[list[str], list[str]]:
    replacements = {
        "input": str(input_path.resolve()),
        "output_dir": str(run_root.resolve()),
        "config_json": str(config_json.resolve()),
    }
    native = [str(part).format(**replacements) for part in config["embedding"]["command"]]
    if resume:
        native.append("--resume")
    command = [
        "conda", "run", "--no-capture-output", "-n",
        config["embedding"]["environment"], *native,
    ]
    return command, native


def _legacy_attempt(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt": 0,
        "kind": "initial",
        "state": status.get("state", "unknown"),
        "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
        "duration_seconds": status.get("duration_seconds"),
        "exit_code": status.get("exit_code"),
        "error": status.get("error"),
    }


def _next_resume_attempt(run_root: Path, status: dict[str, Any]) -> int:
    recorded = [
        int(item["attempt"])
        for item in status.get("attempts", [])
        if isinstance(item, dict) and isinstance(item.get("attempt"), int)
    ]
    logged = []
    for path in run_root.glob("stdout.resume-*.log"):
        try:
            logged.append(int(path.stem.rsplit("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max([0, *recorded, *logged]) + 1


def _validated_embedding(config: dict[str, Any]) -> dict[str, Any]:
    embedding = config.get("embedding", {})
    if embedding.get("mode") != "qwen3vl":
        raise ValueError("embedding.mode must be qwen3vl")
    command_template = embedding.get("command")
    if not isinstance(command_template, list) or not command_template:
        raise ValueError("embedding.command must be a non-empty list")
    settings = embedding.get("modes", {}).get("qwen3vl", {})
    if int(settings.get("dimension", 0)) <= 0:
        raise ValueError("qwen3vl dimension must be greater than zero")
    return embedding


def run_qwen3vl(
    toolkit_root: Path,
    input_path: Path,
    custom_config: Path | None = None,
    resume_from: Path | None = None,
) -> Path:
    input_path = input_path.expanduser().resolve()
    indexes = discover_surya_indexes(input_path)
    if resume_from is not None and custom_config is not None:
        raise ValueError("--config cannot be used with --resume-from")
    if resume_from is None:
        config = load_config(toolkit_root, custom_config)
        embedding = _validated_embedding(config)
        run_root = allocate_run_root(toolkit_root, config, "qwen3vl")
        config_json = run_root / "resolved_config.json"
        write_json(config_json, config)
        status: dict[str, Any] = {
            "mode": "qwen3vl", "state": "running", "input": str(input_path),
            "source_index_count": len(indexes),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        attempt_number = 0
        attempt = {
            "attempt": 0, "kind": "initial", "state": "running",
            "started_at": status["started_at"],
        }
        status["attempts"] = [attempt]
        stdout_path = run_root / "stdout.log"
        stderr_path = run_root / "stderr.log"
    else:
        run_root = resume_from.expanduser().resolve()
        if not run_root.is_dir():
            raise ValueError(f"Resume run folder does not exist: {run_root}")
        if (run_root / "knowledge_base").exists():
            raise ValueError(f"Completed knowledge base cannot be resumed: {run_root}")
        resumable = [
            path for path in (run_root / "knowledge_base.failed", run_root / ".knowledge_base.tmp")
            if path.exists()
        ]
        if len(resumable) != 1:
            raise ValueError(
                "Resume requires exactly one knowledge_base.failed or .knowledge_base.tmp folder"
            )
        config_json = run_root / "resolved_config.json"
        command_path = run_root / "command.json"
        status_path = run_root / "status.json"
        if not config_json.is_file() or not command_path.is_file() or not status_path.is_file():
            raise ValueError("Resume run is missing resolved_config.json, command.json, or status.json")
        config = read_json(config_json)
        embedding = _validated_embedding(config)
        prior_command = read_json(command_path)
        expected_indexes = [str(path) for path in indexes]
        if prior_command.get("input") != str(input_path):
            raise ValueError("Resume Surya input does not match the original run")
        if prior_command.get("source_indexes") != expected_indexes:
            raise ValueError("Resume Surya index paths do not match the original run")
        if prior_command.get("embedding") != config.get("embedding"):
            raise ValueError("Resume embedding config does not match resolved_config.json")
        if prior_command.get("embedding_preprocess") != config.get("embedding_preprocess", {}):
            raise ValueError("Resume preprocessing config does not match resolved_config.json")
        status = read_json(status_path)
        if not isinstance(status.get("attempts"), list):
            status["attempts"] = [_legacy_attempt(status)]
        attempt_number = _next_resume_attempt(run_root, status)
        attempt = {
            "attempt": attempt_number, "kind": "resume", "state": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        status["attempts"].append(attempt)
        status.update({
            "state": "running", "error": None, "exit_code": None,
            "finished_at": None,
        })
        stdout_path = run_root / f"stdout.resume-{attempt_number:02d}.log"
        stderr_path = run_root / f"stderr.resume-{attempt_number:02d}.log"

    command, native = _embedding_command(
        config, input_path, run_root, config_json, resume=resume_from is not None,
    )
    logger = EventLogger(run_root, config["logging"].get("redact_keys"))
    if resume_from is None:
        write_json(run_root / "command.json", {
            "command": command,
            "native_command": native,
            "input": str(input_path),
            "source_indexes": [str(path) for path in indexes],
            "embedding": embedding,
            "embedding_preprocess": config.get("embedding_preprocess", {}),
        })
        environment = _environment_payload(config)
        environment["embedding_environment"] = embedding["environment"]
        write_json(run_root / "environment.json", environment)
    else:
        write_json(run_root / f"command.resume-{attempt_number:02d}.json", {
            "command": command, "native_command": native, "input": str(input_path),
            "source_indexes": [str(path) for path in indexes],
        })
    write_json(run_root / "status.json", status)
    interval = float(config["metadata"]["sampling_interval_seconds"])
    started_at = utc_now()
    started = time.monotonic()
    process_result: ProcessResult | None = None
    warnings: list[str] = []
    try:
        process_result = run_process(
            command, toolkit_root, stdout_path, stderr_path,
            logger, int(config["project"]["heartbeat_seconds"]), interval, mode="qwen3vl",
        )
        if process_result.exit_code != 0:
            raise RuntimeError(f"qwen3vl embedding exited with code {process_result.exit_code}")
        required = (
            Path("knowledge_base/manifest.json"),
            Path("knowledge_base/records.jsonl"),
            Path("knowledge_base/embedding_inputs/metadata.json"),
            Path("knowledge_base/vectors/text.npy"),
            Path("knowledge_base/vectors/image.npy"),
        )
        missing = [str(path) for path in required if not (run_root / path).is_file()]
        if missing:
            raise RuntimeError(f"qwen3vl did not produce required output: {', '.join(missing)}")
        status.update({
            "state": "completed", "exit_code": process_result.exit_code,
            "duration_seconds": round(process_result.duration_seconds, 3),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        attempt.update({
            "state": "completed", "exit_code": process_result.exit_code,
            "duration_seconds": round(process_result.duration_seconds, 3),
            "finished_at": status["finished_at"],
        })
        logger.emit("qwen3vl", "DONE", f"indexes={len(indexes)}")
        return run_root
    except Exception as exc:
        status.update({
            "state": "failed", "exit_code": process_result.exit_code if process_result else None,
            "error": str(exc), "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        attempt.update({
            "state": "failed",
            "exit_code": process_result.exit_code if process_result else None,
            "duration_seconds": round(
                process_result.duration_seconds if process_result else time.monotonic() - started, 3,
            ),
            "error": str(exc), "finished_at": status["finished_at"],
        })
        logger.emit("qwen3vl", "FAILED", str(exc), level="ERROR")
        raise
    finally:
        duration = process_result.duration_seconds if process_result else time.monotonic() - started
        samples = process_result.samples if process_result else []
        warnings.extend(process_result.warnings if process_result else ["Embedding process did not start; hardware samples are unavailable"])
        try:
            metadata_root = run_root if resume_from is None else run_root / f"resume-{attempt_number:02d}"
            write_metadata(
                metadata_root, command=command, image_count=len(indexes),
                started_at=process_result.started_at if process_result else started_at,
                finished_at=process_result.finished_at if process_result else utc_now(), duration=duration,
                exit_code=process_result.exit_code if process_result else None, samples=samples,
                warnings=warnings, interval=interval, mode="qwen3vl", workload_unit="source_index",
            )
        except Exception as exc:
            logger.emit("metadata", "WARNING", str(exc))
        write_json(run_root / "status.json", status)


def _pdf_page_count(input_pdf: Path) -> int:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - real core environment
        raise RuntimeError("PDF inspection requires PyMuPDF") from exc
    document = fitz.open(input_pdf)
    try:
        if document.needs_pass:
            raise PermissionError("PDF requires a password")
        if document.page_count == 0:
            raise ValueError("PDF contains no pages")
        return document.page_count
    finally:
        document.close()


def run_marker(toolkit_root: Path, input_path: Path, custom_config: Path | None = None) -> Path:
    input_path = input_path.expanduser().resolve()
    is_pdf = input_path.is_file() and input_path.suffix.lower() == ".pdf"
    source_images = _pngs(input_path) if input_path.is_dir() else []
    if not is_pdf and not source_images:
        raise ValueError(f"marker requires one PDF file or PNG folder: {input_path}")
    config = load_config(toolkit_root, custom_config)
    marker_command = config["marker"].get("command")
    if not isinstance(marker_command, list) or not marker_command:
        raise ValueError("marker.command must be a non-empty list")
    if float(config["metadata"]["sampling_interval_seconds"]) <= 0:
        raise ValueError("metadata.sampling_interval_seconds must be greater than zero")
    if int(config["project"]["heartbeat_seconds"]) <= 0:
        raise ValueError("project.heartbeat_seconds must be greater than zero")

    run_root = allocate_run_root(toolkit_root, config, "marker")
    if is_pdf:
        effective_input = input_path
        page_count = _pdf_page_count(effective_input)
    else:
        effective_input = convert_png_folder(input_path, run_root / "input.image-only.pdf")
        page_count = len(source_images)
        write_json(
            run_root / "input_manifest.json",
            {
                "input": str(input_path),
                "effective_input": str(effective_input),
                "source_type": "png_folder",
                "pages": [
                    {"page_index": index, "source_image": str(path.resolve())}
                    for index, path in enumerate(source_images)
                ],
            },
        )
    logger = EventLogger(run_root, config["logging"].get("redact_keys"))
    command, native = _marker_command(config, effective_input, run_root)
    write_json(
        run_root / "command.json",
        {
            "command": command,
            "native_command": native,
            "input": str(input_path),
            "effective_input": str(effective_input),
            "marker": config["marker"],
        },
    )
    environment = _environment_payload(config)
    environment.update({
        "marker_environment": config["marker"]["environment"],
        "declared_marker_version": config["marker"].get("version"),
    })
    write_json(run_root / "environment.json", environment)
    status: dict[str, Any] = {
        "mode": "marker", "state": "running", "input": str(input_path),
        "effective_input": str(effective_input),
        "page_count": page_count, "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_root / "status.json", status)
    interval = float(config["metadata"]["sampling_interval_seconds"])
    started_at = utc_now()
    started = time.monotonic()
    process_result: ProcessResult | None = None
    warnings: list[str] = []
    try:
        process_result = run_process(
            command, toolkit_root, run_root / "stdout.log", run_root / "stderr.log",
            logger, int(config["project"]["heartbeat_seconds"]), interval, mode="marker",
        )
        if process_result.exit_code != 0:
            raise RuntimeError(f"Marker exited with code {process_result.exit_code}")
        required = (
            Path("result.json"),
            Path("result.md"),
            Path("result_meta.json"),
            Path("block_provenance.json"),
            Path("result/DocumentBuilder.json"),
            Path("result/PdfConverter.build_document.json"),
        )
        missing = [str(path) for path in required if not (run_root / path).is_file()]
        if missing:
            raise RuntimeError(f"Marker did not produce required output: {', '.join(missing)}")
        status.update({
            "state": "completed", "exit_code": process_result.exit_code,
            "duration_seconds": round(process_result.duration_seconds, 3),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.emit("marker", "DONE", f"pages={page_count}")
        return run_root
    except Exception as exc:
        status.update({
            "state": "failed", "exit_code": process_result.exit_code if process_result else None,
            "error": str(exc), "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.emit("marker", "FAILED", str(exc), level="ERROR")
        raise
    finally:
        duration = process_result.duration_seconds if process_result else time.monotonic() - started
        samples = process_result.samples if process_result else []
        warnings.extend(process_result.warnings if process_result else ["Marker process did not start; hardware samples are unavailable"])
        try:
            write_metadata(
                run_root, command=command, image_count=page_count,
                started_at=process_result.started_at if process_result else started_at,
                finished_at=process_result.finished_at if process_result else utc_now(), duration=duration,
                exit_code=process_result.exit_code if process_result else None, samples=samples,
                warnings=warnings, interval=interval, mode="marker", workload_unit="page",
            )
        except Exception as exc:
            logger.emit("metadata", "WARNING", str(exc))
        write_json(run_root / "status.json", status)


def run_surya_batch(
    toolkit_root: Path,
    result_root: Path,
    images: list[dict[str, Any]],
    config: dict[str, Any],
) -> bool:
    result_root.mkdir(parents=True, exist_ok=True)
    logger = EventLogger(result_root, config["logging"].get("redact_keys"))
    input_dir = Path(images[0]["path"]).parent
    command, native = _native_command(config, input_dir, result_root)
    write_json(result_root / "command.json", {"command": command, "native_command": native, "surya2": config["surya2"]})
    environment = _environment_payload(config)
    environment.update({
        "surya_environment": config["surya2"]["environment"],
        "declared_surya_version": config["surya2"].get("version"),
    })
    write_json(result_root / "environment.json", environment)
    status: dict[str, Any] = {
        "mode": "surya2", "state": "running", "input": str(input_dir),
        "image_count": len(images), "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(result_root / "status.json", status)
    interval = float(config["metadata"]["sampling_interval_seconds"])
    started_at = utc_now()
    started = time.monotonic()
    process_result: ProcessResult | None = None
    warnings: list[str] = []
    try:
        process_result = run_process(
            command, toolkit_root, result_root / "stdout.log", result_root / "stderr.log",
            logger, int(config["project"]["heartbeat_seconds"]), interval,
        )
        if process_result.exit_code != 0:
            raise RuntimeError(f"Surya exited with code {process_result.exit_code}")
        _normalize_results(result_root)
        build_surya_assets(result_root, images, logger)
        status.update({
            "state": "completed", "exit_code": process_result.exit_code,
            "duration_seconds": round(process_result.duration_seconds, 3),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.emit("surya2", "DONE", f"images={len(images)}")
        return True
    except Exception as exc:
        status.update({
            "state": "failed", "exit_code": process_result.exit_code if process_result else None,
            "error": str(exc), "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.emit("surya2", "FAILED", str(exc), level="ERROR")
        return False
    finally:
        duration = process_result.duration_seconds if process_result else time.monotonic() - started
        samples = process_result.samples if process_result else []
        warnings.extend(process_result.warnings if process_result else ["Surya process did not start; hardware samples are unavailable"])
        try:
            write_metadata(
                result_root, command=command, image_count=len(images), started_at=process_result.started_at if process_result else started_at,
                finished_at=process_result.finished_at if process_result else utc_now(), duration=duration,
                exit_code=process_result.exit_code if process_result else None, samples=samples,
                warnings=warnings, interval=interval,
            )
        except Exception as exc:  # metadata never changes the Surya state
            logger.emit("metadata", "WARNING", str(exc))
        write_json(result_root / "status.json", status)


def run_surya2(toolkit_root: Path, input_path: Path, custom_config: Path | None = None) -> tuple[Path, list[str]]:
    input_path = input_path.expanduser().resolve()
    is_pdf = input_path.is_file() and input_path.suffix.lower() == ".pdf"
    if not is_pdf and not input_path.is_dir():
        raise ValueError(f"surya2 requires one PNG folder or PDF file: {input_path}")
    config = load_config(toolkit_root, custom_config)
    _validate_config(config)
    run_root: Path
    if is_pdf:
        run_root = allocate_run_root(toolkit_root, config, "surya2")
        rendered_root = run_root / "rendered_png"
        render_pdf(input_path, rendered_root, int(config["project"]["render_dpi"]))
        batches = prepare_png_batches(rendered_root)
        batched = batches[0][0] is not None
    else:
        batches = prepare_png_batches(input_path)
        batched = batches[0][0] is not None
        run_root = allocate_run_root(toolkit_root, config, "surya2")

    failed: list[str] = []
    completed = 0
    for batch, images in batches:
        if batched:
            result_root = run_root / str(batch)
            label = str(batch)
        elif is_pdf:
            result_root = run_root / "result"
            label = "result"
        else:
            result_root = run_root
            label = "result"
        if run_surya_batch(toolkit_root, result_root, images, config):
            completed += 1
        else:
            failed.append(label)
    if batched:
        print(f"Completed: {completed} batches")
        print(f"Failed: {len(failed)} batches" + (f" ({', '.join(failed)})" if failed else ""))
    return run_root, failed
