"""Usage recorder storing token statistics."""
from __future__ import annotations

import atexit
import logging
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


class UsageRecorder:
    """Batching SQLite usage recorder."""

    def __init__(
        self,
        db_path: str | os.PathLike[str] | None = None,
        *,
        batch_size: int = 10,
        auto_flush: bool = True,
        env_var: Optional[str] = "LLM_USAGE_DB",
        default_filename: str = "usage_log.db",
        supports_thoughts: bool = False,
    ) -> None:
        resolved_path: str | os.PathLike[str]
        if db_path is not None:
            resolved_path = db_path
        else:
            env_value = os.environ.get(env_var, "").strip() if env_var else ""
            resolved_path = env_value or default_filename

        self._db_path = Path(resolved_path)
        self._batch_size = batch_size
        self._buffer: List[Tuple[Any, ...]] = []
        self._lock = threading.Lock()

        self._columns: List[Tuple[str, str]] = [
            ("timestamp", "TEXT"),
            ("model", "TEXT"),
            ("request_id", "TEXT"),
            ("trace_id", "TEXT"),
            ("prompt_tokens", "INTEGER"),
            ("completion_tokens", "INTEGER"),
            ("total_tokens", "INTEGER"),
        ]
        self._has_thoughts_column = supports_thoughts
        if self._has_thoughts_column:
            self._columns.append(("thoughts_token_count", "INTEGER"))

        self._ensure_table()
        if auto_flush:
            atexit.register(self.flush)

    def _ensure_table(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            column_defs = ", ".join(f"{name} {sql_type}" for name, sql_type in self._columns)
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    {column_defs}
                )
                """.strip()
            )
            existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(usage_log)")}
            for name, sql_type in self._columns:
                if name not in existing_columns:
                    conn.execute(f"ALTER TABLE usage_log ADD COLUMN {name} {sql_type}")
            conn.commit()

    def record(
        self,
        *,
        model: Optional[str],
        request_id: Optional[str],
        trace_id: Optional[str],
        usage: Optional[Dict[str, Any]],
    ) -> None:
        if not usage:
            return
        values: List[Any] = [
            datetime.utcnow().isoformat(),
            model,
            request_id,
            trace_id,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        ]
        if self._has_thoughts_column:
            values.append(usage.get("thoughts_token_count"))

        with self._lock:
            self._buffer.append(tuple(values))
            if len(self._buffer) >= self._batch_size:
                self._flush()

    def flush(self) -> None:
        with self._lock:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        try:
            with sqlite3.connect(self._db_path) as conn:
                column_names = ", ".join(name for name, _ in self._columns)
                placeholders = ", ".join("?" for _ in self._columns)
                conn.executemany(
                    f"INSERT INTO usage_log ({column_names}) VALUES ({placeholders})",
                    self._buffer,
                )
                conn.commit()
            self._buffer.clear()
        except Exception as exc:  # noqa: BLE001
            logger.error("写入 usage 失败: %s", exc)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.flush()


__all__ = ["UsageRecorder"]
