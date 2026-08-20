from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MonitoringStore:
    def __init__(self, database_path: Path, access_log_root: Path) -> None:
        self.database_path = database_path.resolve()
        self.access_log_root = access_log_root.resolve()
        self._lock = threading.RLock()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.access_log_root.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS session_map (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    access_session_id TEXT NOT NULL,
                    chat_session_id TEXT,
                    ip_addr TEXT NOT NULL,
                    first_seen_utc TEXT NOT NULL,
                    last_seen_utc TEXT NOT NULL,
                    ended_at_utc TEXT,
                    end_reason TEXT,
                    access_log_path TEXT NOT NULL,
                    chat_log_path TEXT,
                    has_chat INTEGER NOT NULL DEFAULT 0
                );
                CREATE UNIQUE INDEX IF NOT EXISTS ux_session_map_access
                    ON session_map(access_session_id) WHERE chat_session_id IS NULL;
                CREATE UNIQUE INDEX IF NOT EXISTS ux_session_map_chat
                    ON session_map(chat_session_id) WHERE chat_session_id IS NOT NULL;
                CREATE INDEX IF NOT EXISTS ix_session_map_ip ON session_map(ip_addr);
                CREATE INDEX IF NOT EXISTS ix_session_map_last_seen ON session_map(last_seen_utc);
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(session_map)")
            }
            if "has_chat" not in columns:
                connection.execute(
                    "ALTER TABLE session_map ADD COLUMN has_chat INTEGER NOT NULL DEFAULT 0"
                )
                self._backfill_has_chat(connection)

    @staticmethod
    def _backfill_has_chat(connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT id, chat_log_path FROM session_map WHERE chat_session_id IS NOT NULL"
        ).fetchall()
        for row in rows:
            path_value = row["chat_log_path"]
            has_chat = False
            if path_value:
                pretty_path = Path(path_value)
                suffix = ".pretty.json"
                raw_name = (
                    pretty_path.name[:-len(suffix)] + ".jsonl"
                    if pretty_path.name.endswith(suffix)
                    else pretty_path.name
                )
                raw_path = pretty_path.with_name(raw_name)
                if raw_path.is_file():
                    for line in raw_path.read_text(encoding="utf-8").splitlines():
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if event.get("event") in {"job_queued", "request_received"}:
                            has_chat = True
                            break
            connection.execute(
                "UPDATE session_map SET has_chat = ? WHERE id = ?",
                (int(has_chat), row["id"]),
            )

    def access_row(self, access_session_id: str) -> Optional[dict[str, Any]]:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM session_map WHERE access_session_id = ? "
                "AND chat_session_id IS NULL",
                (access_session_id,),
            ).fetchone()
        return dict(row) if row else None

    def create_access(self, access_session_id: str, ip_addr: str) -> dict[str, Any]:
        now = _utc_now()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = (self.access_log_root / f"{timestamp}_{access_session_id}.jsonl").resolve()
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO session_map "
                "(access_session_id, chat_session_id, ip_addr, first_seen_utc, "
                "last_seen_utc, access_log_path) VALUES (?, NULL, ?, ?, ?, ?)",
                (access_session_id, ip_addr, now, now, str(path)),
            )
        return self.access_row(access_session_id) or {
            "access_session_id": access_session_id,
            "ip_addr": ip_addr,
            "access_log_path": str(path),
        }

    def add_session(self, access_session_id: str, chat_session_id: str,
                    ip_addr: str) -> None:
        access = self.access_row(access_session_id)
        if access is None:
            access = self.create_access(access_session_id, ip_addr)
        now = _utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO session_map "
                "(access_session_id, chat_session_id, ip_addr, first_seen_utc, "
                "last_seen_utc, access_log_path, chat_log_path, has_chat) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, 0)",
                (access_session_id, chat_session_id, ip_addr, now, now,
                 access["access_log_path"]),
            )

    def materialize_chat(self, chat_session_id: str,
                         chat_log_path: Optional[Path]) -> None:
        now = _utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE session_map SET has_chat = 1, chat_log_path = ?, "
                "last_seen_utc = ? WHERE chat_session_id = ?",
                (str(chat_log_path.resolve()) if chat_log_path else None,
                 now, chat_session_id),
            )

    def touch(self, access_session_id: str, chat_session_id: Optional[str] = None) -> None:
        now = _utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE session_map SET last_seen_utc = ? WHERE access_session_id = ? "
                "AND chat_session_id IS NULL",
                (now, access_session_id),
            )
            if chat_session_id:
                connection.execute(
                    "UPDATE session_map SET last_seen_utc = ? WHERE chat_session_id = ?",
                    (now, chat_session_id),
                )

    def end_chat(self, chat_session_id: str, reason: str) -> None:
        now = _utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE session_map SET last_seen_utc = ?, ended_at_utc = ?, "
                "end_reason = ? WHERE chat_session_id = ? AND ended_at_utc IS NULL",
                (now, now, reason, chat_session_id),
            )

    def end_access(self, access_session_id: str, reason: str) -> None:
        now = _utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE session_map SET last_seen_utc = ?, ended_at_utc = ?, "
                "end_reason = ? WHERE access_session_id = ? AND ended_at_utc IS NULL",
                (now, now, reason, access_session_id),
            )

    def append_access(self, access_session_id: str, event: dict[str, Any]) -> None:
        row = self.access_row(access_session_id)
        if row is None:
            return
        path = Path(row["access_log_path"])
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as output:
                output.write(json.dumps(event, ensure_ascii=False, default=str))
                output.write("\n")
        self.touch(access_session_id, event.get("chat_session_id"))

    def query(self, *, ip_addr: str = "", access_session_id: str = "",
              chat_session_id: str = "", from_utc: str = "", to_utc: str = "",
              limit: int = 100) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        for column, value in (
            ("ip_addr", ip_addr),
            ("access_session_id", access_session_id),
            ("chat_session_id", chat_session_id),
        ):
            if value:
                clauses.append(f"{column} = ?")
                values.append(value)
        if from_utc:
            clauses.append("last_seen_utc >= ?")
            values.append(from_utc)
        if to_utc:
            clauses.append("last_seen_utc <= ?")
            values.append(to_utc)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(max(1, min(limit, 100)))
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM session_map {where} "
                "ORDER BY last_seen_utc DESC LIMIT ?",
                values,
            ).fetchall()
        return [dict(row) for row in rows]

    def chat_row(self, chat_session_id: str) -> Optional[dict[str, Any]]:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM session_map WHERE chat_session_id = ?",
                (chat_session_id,),
            ).fetchone()
        return dict(row) if row else None

    def access_events(self, access_session_id: str, *, page: int = 1,
                      page_size: int = 200) -> list[dict[str, Any]]:
        row = self.access_row(access_session_id)
        if row is None:
            return []
        path = Path(row["access_log_path"])
        if not path.is_file():
            return []
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        start = (max(page, 1) - 1) * page_size
        return events[start:start + page_size]

    def chat_history(self, chat_session_id: str) -> Optional[dict[str, Any]]:
        row = self.chat_row(chat_session_id)
        if row is None or not row.get("chat_log_path"):
            return None
        path = Path(row["chat_log_path"])
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
