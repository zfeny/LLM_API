"""Gemini Files API utilities."""
from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)


class GeminiFileUploader:
    """Gemini Files API uploader with simple local cache."""

    def __init__(self, genai_client: Any):
        self._client = genai_client
        self._uploaded_files: dict[str, Any] = {}

    def upload_file(self, file_path: str) -> Any:
        """Upload a file to Gemini Files API and cache the handle."""
        if not self.is_local_file(file_path):
            cached = self._uploaded_files.get(file_path)
            if cached is not None:
                logger.debug("使用缓存的远程文件对象: %s", file_path)
                return cached

            mime_type, _ = mimetypes.guess_type(file_path)
            handle = SimpleNamespace(uri=file_path, mime_type=mime_type or "application/octet-stream")
            self._uploaded_files[file_path] = handle
            logger.debug("使用远程文件URI: %s (%s)", handle.uri, handle.mime_type)
            return handle

        path = Path(file_path)
        normalized_path = str(path.resolve())

        cached = self._uploaded_files.get(normalized_path)
        if cached is not None:
            logger.debug("使用缓存的文件对象: %s", file_path)
            return cached

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        if not path.is_file():
            raise ValueError(f"路径不是文件: {file_path}")

        logger.debug("上传文件到 Files API: %s", file_path)
        try:
            uploaded_file = self._client.files.upload(file=str(path))
        except Exception as exc:  # noqa: BLE001
            logger.error("文件上传失败 %s: %s", file_path, exc)
            raise

        logger.debug("文件上传成功: %s -> %s (URI: %s)", path.name, uploaded_file.name, uploaded_file.uri)
        self._uploaded_files[normalized_path] = uploaded_file
        return uploaded_file

    def upload_files(self, file_paths: list[str]) -> list[Any]:
        """Upload multiple files and return the results."""
        return [self.upload_file(file_path) for file_path in file_paths]

    @staticmethod
    def is_local_file(path: str) -> bool:
        """Return True when the path points to an existing local file."""
        if path.startswith(("http://", "https://")):
            return False
        return Path(path).exists()


__all__ = ["GeminiFileUploader"]
