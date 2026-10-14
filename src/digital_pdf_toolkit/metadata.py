from __future__ import annotations

import csv
import json
import os
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAMPLE_FIELDS = (
    "elapsed_seconds", "timestamp_utc", "gpu_utilization_percent", "vram_used_gb",
    "gpu_temperature_c", "gpu_power_w", "process_cpu_percent", "process_rss_mib",
    "system_ram_used_mib", "system_ram_used_percent", "disk_read_mib", "disk_write_mib",
)
SENSITIVE_FLAGS = ("token", "secret", "password", "api-key", "api_key", "authorization")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value.replace(",", ""))
        return float(match.group(0)) if match else None
    return None


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            values.extend(_flatten(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        if value:
            values.extend(_flatten(value[0], prefix))
    else:
        values.append((prefix.lower().replace(" ", "_"), value))
    return values


def _pick(flat: list[tuple[str, Any]], groups: tuple[tuple[str, ...], ...]) -> float | None:
    for required in groups:
        for key, value in flat:
            if all(part in key for part in required):
                number = _number(value)
                if number is not None:
                    return number
    return None


class AmdSampler:
    def __init__(self) -> None:
        self.executable = shutil.which("amd-smi") if platform.system() == "Linux" else None
        self.warning = None if self.executable else "amd-smi is unavailable; GPU metrics are null"

    def sample(self) -> dict[str, float | None]:
        empty = {
            "gpu_utilization_percent": None, "vram_used_gb": None,
            "gpu_temperature_c": None, "gpu_power_w": None,
        }
        if not self.executable:
            return empty
        probes = [
            [self.executable, "monitor", "-g", "0", "--json"],
            [self.executable, "metric", "-g", "0", "-u", "-m", "-p", "-t", "--json"],
        ]
        payload: Any = None
        for command in probes:
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
                if result.returncode == 0:
                    payload = json.loads(result.stdout)
                    break
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
                self.warning = f"amd-smi sampling failed: {exc}"
        if payload is None:
            self.warning = self.warning or "amd-smi returned no usable data; GPU metrics are null"
            return empty
        flat = _flatten(payload)
        vram = _pick(flat, (("vram", "used"), ("used", "vram")))
        if vram is not None and vram > 128:
            vram /= 1024
        return {
            "gpu_utilization_percent": _pick(flat, (("gfx", "util"), ("gpu", "util"))),
            "vram_used_gb": vram,
            "gpu_temperature_c": _pick(flat, (("gpu", "temp"), ("temperature",))),
            "gpu_power_w": _pick(flat, (("power", "usage"), ("power",))),
        }


def _proc_stat(pid: int) -> tuple[int, int, int] | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = text[text.rfind(")") + 2:].split()
        return int(fields[1]), int(fields[11]) + int(fields[12]), int(fields[21])
    except (OSError, ValueError, IndexError):
        return None


def _process_tree(root_pid: int) -> dict[int, tuple[int, int]]:
    if platform.system() != "Linux":
        return {}
    records: dict[int, tuple[int, int, int]] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return {}
    for entry in entries:
        if entry.name.isdigit() and (value := _proc_stat(int(entry.name))) is not None:
            records[int(entry.name)] = value
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _, _) in records.items():
            if ppid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return {pid: (records[pid][1], records[pid][2]) for pid in selected if pid in records}


def _process_io(pid: int) -> tuple[int, int]:
    values = {"read_bytes": 0, "write_bytes": 0}
    try:
        for line in Path(f"/proc/{pid}/io").read_text(encoding="utf-8").splitlines():
            key, _, raw = line.partition(":")
            if key in values:
                values[key] = int(raw.strip())
    except (OSError, ValueError):
        pass
    return values["read_bytes"], values["write_bytes"]


def _system_memory() -> tuple[float | None, float | None]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, raw = line.partition(":")
            if match := re.search(r"\d+", raw):
                values[key] = int(match.group(0)) * 1024
    except OSError:
        return None, None
    total, available = values.get("MemTotal"), values.get("MemAvailable")
    if not total or available is None:
        return None, None
    used = total - available
    return used / 1048576, used / total * 100


class ProcessSampler:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.supported = platform.system() == "Linux"
        self.clock_ticks = os.sysconf("SC_CLK_TCK") if self.supported else 1
        self.page_size = os.sysconf("SC_PAGE_SIZE") if self.supported else 1
        self.last_at = time.monotonic()
        self.last_cpu: dict[int, int] = {}
        self.last_io: dict[int, tuple[int, int]] = {}
        self.disk_read = 0
        self.disk_write = 0

    def sample(self) -> dict[str, float | None]:
        empty = {
            "process_cpu_percent": None, "process_rss_mib": None,
            "system_ram_used_mib": None, "system_ram_used_percent": None,
            "disk_read_mib": None, "disk_write_mib": None,
        }
        if not self.supported:
            return empty
        now = time.monotonic()
        elapsed = max(now - self.last_at, 1e-9)
        cpu_delta = 0
        rss_pages = 0
        tree = _process_tree(self.pid)
        for pid, (cpu_ticks, pages) in tree.items():
            cpu_delta += max(0, cpu_ticks - self.last_cpu.get(pid, cpu_ticks))
            self.last_cpu[pid] = cpu_ticks
            rss_pages += pages
            current_io = _process_io(pid)
            previous_io = self.last_io.get(pid, current_io)
            self.disk_read += max(0, current_io[0] - previous_io[0])
            self.disk_write += max(0, current_io[1] - previous_io[1])
            self.last_io[pid] = current_io
        self.last_at = now
        memory, memory_percent = _system_memory()
        return {
            "process_cpu_percent": cpu_delta / self.clock_ticks / elapsed * 100,
            "process_rss_mib": rss_pages * self.page_size / 1048576,
            "system_ram_used_mib": memory,
            "system_ram_used_percent": memory_percent,
            "disk_read_mib": self.disk_read / 1048576,
            "disk_write_mib": self.disk_write / 1048576,
        }


class MetadataSampler:
    def __init__(self, pid: int) -> None:
        self.started = time.monotonic()
        self.process = ProcessSampler(pid)
        self.amd = AmdSampler()

    def sample(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": time.monotonic() - self.started,
            "timestamp_utc": utc_now(),
            **self.amd.sample(),
            **self.process.sample(),
        }

    @property
    def warnings(self) -> list[str]:
        values = []
        if platform.system() != "Linux":
            values.append("hardware sampling is Linux-only; hardware metrics are null")
        if self.amd.warning:
            values.append(self.amd.warning)
        return values


def _redact(command: list[str]) -> list[str]:
    output: list[str] = []
    hide_next = False
    for value in command:
        if hide_next:
            output.append("***")
            hide_next = False
        elif any(term in value.lower() for term in SENSITIVE_FLAGS):
            output.append(value.split("=", 1)[0] + "=***" if "=" in value else value)
            hide_next = "=" not in value
        else:
            output.append(value)
    return output


def _metric(samples: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    values = [float(sample[field]) for sample in samples if sample.get(field) is not None]
    return {"average": sum(values) / len(values), "peak": max(values)} if values else {"average": None, "peak": None}


def write_metadata(
    result_root: Path,
    *,
    command: list[str],
    image_count: int,
    started_at: str,
    finished_at: str,
    duration: float,
    exit_code: int | None,
    samples: list[dict[str, Any]],
    warnings: list[str],
    interval: float,
    mode: str = "surya2",
    workload_unit: str = "png",
) -> None:
    metadata_root = result_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    count_key = "png_count" if workload_unit == "png" else f"{workload_unit}_count"
    singular = "image" if workload_unit == "png" else workload_unit
    plural = "images" if workload_unit == "png" else f"{workload_unit}s"
    summary = {
        "schema_version": 1,
        "run": {
            "mode": mode, "command": _redact(command), "hostname": platform.node(),
            "platform": platform.platform(), "python": platform.python_version(),
            "started_at_utc": started_at, "finished_at_utc": finished_at, "exit_code": exit_code,
        },
        "sampling": {"interval_seconds": interval, "sample_count": len(samples), "gpu_source": "amd-smi (GPU 0)"},
        "workload": {
            count_key: image_count, "total_seconds": duration,
            f"seconds_per_{singular}": duration / image_count if image_count else None,
            f"{plural}_per_second": image_count / duration if image_count and duration else None,
        },
        "metrics": {field: _metric(samples, field) for field in SAMPLE_FIELDS[2:]},
        "warnings": list(dict.fromkeys(warnings)),
    }
    (metadata_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (metadata_root / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows({field: sample.get(field) for field in SAMPLE_FIELDS} for sample in samples)
