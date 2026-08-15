from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class SessionJsonlLogger:
    machine_only_keys = {
        "record_id",
        "used_record_ids",
        "document_id",
        "snapshot_id",
        "content_sha256",
        "document_sha256",
        "crop_url",
    }

    def __init__(self, root: Optional[Path], *, enabled: bool) -> None:
        self.root = root
        self.enabled = enabled and root is not None
        self._lock = threading.RLock()
        self._session_stems: dict[str, str] = {}

    def append(self, session_id: str, event: str, **data: Any) -> None:
        if not self.enabled or self.root is None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        }
        with self._lock:
            stem = self._session_stems.setdefault(
                session_id,
                datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S-%f"),
            )
            path = self.root / f"{stem}.jsonl"
            with path.open("a", encoding="utf-8", newline="\n") as output:
                output.write(json.dumps(payload, ensure_ascii=False, default=str))
                output.write("\n")

        self._write_pretty(session_id)

    def end(self, session_id: str, reason: str) -> None:
        self.append(session_id, "session_end", reason=reason)
        self._write_pretty(session_id)

    def _write_pretty(self, session_id: str) -> None:
        if not self.enabled or self.root is None:
            return
        with self._lock:
            stem = self._session_stems.get(session_id)
            if stem is None:
                return
            source_path = self.root / f"{stem}.jsonl"
            if not source_path.is_file():
                return
            events = [
                self._human_fields(json.loads(line))
                for line in source_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            destination = self.root / f"{stem}.pretty.json"
            temporary = self.root / f".{stem}.pretty.tmp"
            temporary.write_text(
                json.dumps(events, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(destination)

    @classmethod
    def _human_fields(cls, value: Any) -> Any:
        if isinstance(value, dict):
            output = {}
            for key, item in value.items():
                if key in {'raw_output', 'prompt', 'thinking'}:
                    continue
                if (
                    key in cls.machine_only_keys
                    or key.endswith("_sha256")
                    or key.endswith("_id")
                    or key.endswith("_ids")
                ):
                    continue
                if key == "raw_output" and isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except json.JSONDecodeError:
                        pass
                if key == "context" and isinstance(item, str):
                    item = re.sub(r"Record ID: [^;\]]+", "Record", item)
                output[key] = cls._human_fields(item)
            return output
        if isinstance(value, list):
            return [cls._human_fields(item) for item in value]
        return value
