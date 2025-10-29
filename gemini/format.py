"""Gemini specific format handler."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, Optional

from llm.exceptions import LLMValidationError

try:
    from google.genai import types as genai_types
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("需要 google-genai SDK: pip install google-genai") from exc


class GeminiFormatHandler:
    """Handle Gemini response formatting and schema validation."""

    @staticmethod
    def get_generation_config(cfg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not cfg:
            return None

        fmt_type = cfg.get("type")
        if fmt_type == "json":
            return {"response_mime_type": "application/json"}
        if fmt_type == "json_schema":
            schema = cfg.get("schema")
            if not schema:
                raise LLMValidationError("json_schema 格式需要提供 schema")
            try:
                schema_obj = GeminiFormatHandler._normalize_schema(schema)
            except Exception as exc:  # noqa: BLE001
                raise LLMValidationError(f"json_schema 转换失败: {exc}") from exc
            return {
                "response_mime_type": "application/json",
                "response_schema": schema_obj,
            }
        return None

    @staticmethod
    def get_prompt_suffix(cfg: Optional[Dict[str, Any]]) -> Optional[str]:
        if not cfg:
            return None
        fmt_type = cfg.get("type")
        if fmt_type == "markdown":
            return "\n\n请确保回应使用 Markdown 的标题或列表组织内容，避免纯文本。"
        if fmt_type == "text":
            return None
        return None

    @staticmethod
    def merge_prompt_to_message(message: str, cfg: Optional[Dict[str, Any]]) -> str:
        suffix = GeminiFormatHandler.get_prompt_suffix(cfg)
        return f"{message}{suffix}" if suffix else message

    @staticmethod
    def process_response(value: Any, cfg: Optional[Dict[str, Any]]) -> Any:
        if not cfg:
            return value
        fmt_type = cfg.get("type")
        if fmt_type in ("json", "json_schema"):
            return GeminiFormatHandler._process_json(value, cfg)
        if fmt_type == "markdown":
            return GeminiFormatHandler._process_markdown(value)
        return value

    @staticmethod
    def _process_json(value: Any, cfg: Dict[str, Any]) -> Any:
        if isinstance(value, (dict, list)):
            data = value
        elif isinstance(value, str):
            try:
                data = json.loads(value)
            except json.JSONDecodeError as exc:
                raise LLMValidationError(f"返回内容不是合法 JSON: {exc}") from exc
        else:
            raise LLMValidationError(f"返回内容不是 JSON 格式: {type(value)}")

        if cfg.get("type") == "json_schema":
            schema = cfg.get("schema", {})
            required = schema.get("required", [])
            if required and isinstance(data, dict):
                missing = [field for field in required if field not in data]
                if missing:
                    raise LLMValidationError(f"缺少必需字段: {', '.join(missing)}")
        return data

    @staticmethod
    def _process_markdown(value: Any) -> str:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise LLMValidationError("返回内容为空")
            return text
        if value is None:
            raise LLMValidationError("无返回内容")
        return str(value)

    @staticmethod
    def _normalize_schema(schema: Any) -> genai_types.Schema:
        if isinstance(schema, genai_types.Schema):
            return schema
        if isinstance(schema, genai_types.JSONSchema):
            payload = schema.to_json_dict()
        elif isinstance(schema, dict):
            payload = schema
        else:
            raise TypeError(f"不支持的 schema 类型: {type(schema)}")
        schema_key = json.dumps(payload, sort_keys=True)
        return GeminiFormatHandler._schema_from_json_text(schema_key)

    @staticmethod
    @lru_cache(maxsize=32)
    def _schema_from_json_text(schema_text: str) -> genai_types.Schema:
        json_schema_obj = genai_types.JSONSchema(**json.loads(schema_text))
        return genai_types.Schema.from_json_schema(
            json_schema=json_schema_obj,
            api_option="GEMINI_API",
        )


__all__ = ["GeminiFormatHandler"]
