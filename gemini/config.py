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

    @classmethod
    def from_env(cls) -> "GeminiAPIConfig":
        def require(key: str) -> str:
            value = os.environ.get(key)
            if not value or not value.strip():
                raise LLMConfigError(f"缺少环境变量: {key}")
            return value.strip()

        return cls(
            api_key=require("GEMINI_API_KEY"),
            default_model=os.environ.get("GEMINI_MODEL"),
        )


__all__ = ["GeminiAPIConfig"]
