"""Gemini configuration objects."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from llm.exceptions import LLMConfigError


@dataclass(slots=True)
class GeminiAPIConfig:
    """Runtime configuration for Gemini API."""

    api_key: str
    default_model: Optional[str]
    image_upload_enabled: bool = True  # 是否上传图片到OpenList

    @classmethod
    def from_env(cls) -> "GeminiAPIConfig":
        def require(key: str) -> str:
            value = os.environ.get(key)
            if not value or not value.strip():
                raise LLMConfigError(f"缺少环境变量: {key}")
            return value.strip()

        # 读取上传开关配置（默认为true）
        upload_enabled_str = os.environ.get("GEMINI_IMAGE_UPLOAD_ENABLED", "true").lower()
        upload_enabled = upload_enabled_str in ("true", "1", "yes", "on")

        return cls(
            api_key=require("GEMINI_API_KEY"),
            default_model=os.environ.get("GEMINI_MODEL"),
            image_upload_enabled=upload_enabled,
        )


__all__ = ["GeminiAPIConfig"]
