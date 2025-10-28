"""使用记录器。"""
from __future__ import annotations
import atexit
import logging
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class UsageRecorder:
    """批量 SQLite usage 记录器。"""
    def __init__(self, db_path: str | os.PathLike[str] | None = None, batch_size: int = 10, auto_flush: bool = True):
        self._db_path = Path(db_path or os.environ.get("GEMINI_USAGE_DB", "gemini_usage_log.db"))
        self._batch_size = batch_size
        self._buffer = []
        self._lock = threading.Lock()
        self._ensure_table()
        if auto_flush:
            atexit.register(self.flush)

    def _ensure_table(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            # 创建表（如果不存在）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, model TEXT, request_id TEXT, trace_id TEXT,
                    prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
                    thoughts_token_count INTEGER
                )
            """)

            # 迁移：检查并添加 thoughts_token_count 字段（如果表已存在但缺少该字段）
            cursor = conn.execute("PRAGMA table_info(usage_log)")
            columns = [row[1] for row in cursor.fetchall()]
            if "thoughts_token_count" not in columns:
                conn.execute("ALTER TABLE usage_log ADD COLUMN thoughts_token_count INTEGER")
                logger.info("数据库迁移：添加 thoughts_token_count 字段")

            conn.commit()

    def record(self, *, model: Optional[str], request_id: Optional[str], trace_id: Optional[str], usage: Optional[Dict[str, Any]]):
        if not usage:
            return
        data = (datetime.utcnow().isoformat(), model, request_id, trace_id,
                usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens"),
                usage.get("thoughts_token_count"))  # 添加 thoughts_token_count
        with self._lock:
            self._buffer.append(data)
            if len(self._buffer) >= self._batch_size:
                self._flush()

    def flush(self):
        with self._lock:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.executemany(
                    "INSERT INTO usage_log (timestamp, model, request_id, trace_id, prompt_tokens, completion_tokens, total_tokens, thoughts_token_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    self._buffer
                )
                conn.commit()
            self._buffer.clear()
        except Exception as exc:
            logger.error("写入 usage 失败: %s", exc)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.flush()
