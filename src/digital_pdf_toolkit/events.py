from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLogger:
    def __init__(self, run_root: Path, redact_keys: list[str] | None = None):
        self.run_root = run_root
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.run_log = run_root / "run.log"
        self.events_log = run_root / "events.jsonl"
        keys = redact_keys or ["token", "secret", "password", "api_key", "authorization"]
        self.redact_pattern = re.compile(
            rf"(?i)({'|'.join(re.escape(key) for key in keys)})\s*[:=]\s*([^\s,;]+)"
        )

    def sanitize(self, value: str) -> str:
        return self.redact_pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)

    def emit(
        self,
        stage: str,
        state: str,
        message: str,
        *,
        level: str = "INFO",
        data: dict[str, Any] | None = None,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        clean = self.sanitize(message)
        event = {
            "timestamp": timestamp,
            "level": level,
            "stage": stage,
            "state": state,
            "message": clean,
            "data": data or {},
        }
        terminal = f"[{timestamp[11:19]}] [{stage}] {state:<7} {clean}"
        print(terminal, file=sys.stderr if level == "ERROR" else sys.stdout, flush=True)
        with self.run_log.open("a", encoding="utf-8") as handle:
            handle.write(terminal + "\n")
        with self.events_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
