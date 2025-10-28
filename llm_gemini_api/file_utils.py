"""Gemini Files API 工具函数。"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class GeminiFileUploader:
    """Gemini Files API 上传器。"""

    def __init__(self, genai_client: Any):
        """
        初始化文件上传器。

        Args:
            genai_client: google.genai.Client 实例
        """
        self._client = genai_client
        # 缓存已上传的文件对象（path -> File）
        self._uploaded_files = {}

    def upload_file(self, file_path: str) -> Any:
        """
        上传文件到 Gemini Files API。

        Args:
            file_path: 本地文件路径

        Returns:
            File 对象，包含 uri 等信息

        Raises:
            FileNotFoundError: 文件不存在
            Exception: 上传失败
        """
        # 规范化路径（用于缓存key）
        path = Path(file_path)
        normalized_path = str(path.resolve())

        # 检查缓存
        if normalized_path in self._uploaded_files:
            logger.debug(f"使用缓存的文件对象: {file_path}")
            return self._uploaded_files[normalized_path]

        # 检查文件是否存在
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 检查是否是文件
        if not path.is_file():
            raise ValueError(f"路径不是文件: {file_path}")

        logger.debug("上传文件到 Files API: %s", file_path)

        try:
            # 使用 Files API 上传文件（keyword-only argument: file）
            uploaded_file = self._client.files.upload(file=str(path))

            logger.debug(
                "文件上传成功: %s -> %s (URI: %s)",
                path.name,
                uploaded_file.name,
                uploaded_file.uri,
            )

            # 缓存文件对象
            self._uploaded_files[normalized_path] = uploaded_file

            return uploaded_file

        except Exception as e:
            logger.error(f"文件上传失败 {file_path}: {e}")
            raise

    def upload_files(self, file_paths: list[str]) -> list[Any]:
        """
        批量上传文件。

        Args:
            file_paths: 文件路径列表

        Returns:
            File 对象列表
        """
        uploaded_files = []

        for file_path in file_paths:
            uploaded_file = self.upload_file(file_path)
            uploaded_files.append(uploaded_file)

        return uploaded_files

    @staticmethod
    def is_local_file(path: str) -> bool:
        """
        判断路径是否为本地文件。

        Args:
            path: 文件路径

        Returns:
            如果是本地文件返回 True
        """
        # 检查是否是 URL
        if path.startswith(("http://", "https://")):
            return False

        # 检查文件是否存在
        return Path(path).exists()
