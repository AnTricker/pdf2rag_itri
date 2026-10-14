from __future__ import annotations

import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from .events import EventLogger
from .metadata import MetadataSampler


@dataclass
class ProcessResult:
    exit_code: int
    duration_seconds: float
    started_at: str
    finished_at: str
    samples: list[dict]
    warnings: list[str]


def _pump(stream: TextIO, destination: TextIO, label: str, messages: queue.Queue[tuple[str, str | None]]) -> None:
    try:
        for line in iter(stream.readline, ""):
            destination.write(line)
            destination.flush()
            messages.put((label, line.rstrip()))
    finally:
        messages.put((label, None))


def run_process(
    command: list[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    logger: EventLogger,
    heartbeat_seconds: int,
    sampling_interval: float,
    mode: str = "surya2",
) -> ProcessResult:
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    logger.emit(mode, "START", "process started")
    messages: queue.Queue[tuple[str, str | None]] = queue.Queue()
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        sampler = MetadataSampler(process.pid)
        samples: list[dict] = []
        next_sample = started
        assert process.stdout is not None and process.stderr is not None
        threads = [
            threading.Thread(target=_pump, args=(process.stdout, stdout_file, "stdout", messages), daemon=True),
            threading.Thread(target=_pump, args=(process.stderr, stderr_file, "stderr", messages), daemon=True),
        ]
        for thread in threads:
            thread.start()
        completed_streams = 0
        last_heartbeat = time.monotonic()
        while completed_streams < 2 or process.poll() is None:
            try:
                label, line = messages.get(timeout=0.25)
                if line is None:
                    completed_streams += 1
                elif line and ("page" in line.lower() or "progress" in line.lower()):
                    logger.emit(mode, "RUN", line)
            except queue.Empty:
                pass
            now = time.monotonic()
            if process.poll() is None and now >= next_sample:
                try:
                    samples.append(sampler.sample())
                except Exception as exc:  # metadata is best-effort
                    sampler.amd.warning = f"metadata sampling failed: {exc}"
                next_sample = now + sampling_interval
            if time.monotonic() - last_heartbeat >= heartbeat_seconds and process.poll() is None:
                elapsed = round(time.monotonic() - started)
                logger.emit(mode, "RUN", f"elapsed={elapsed}s, process active")
                last_heartbeat = time.monotonic()
        for thread in threads:
            thread.join(timeout=1)
        exit_code = process.wait()
    return ProcessResult(
        exit_code=exit_code,
        duration_seconds=time.monotonic() - started,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        samples=samples,
        warnings=sampler.warnings,
    )
