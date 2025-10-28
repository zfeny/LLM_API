"""格式处理器（指令构建 + 响应校验）。"""
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Tuple

from .exceptions import LLMValidationError
from .models import ICSMessage


class FormatHandler:
    """格式处理器（指令构建 + 响应校验）。"""
    _FORMAT_MESSAGE_CACHE: Dict[str, Tuple[ICSMessage, ...]] = {}

    @staticmethod
    def build_messages(cfg: Optional[Dict[str, Any]]) -> List[ICSMessage]:
        if not cfg:
            return []
        t = cfg.get("type")
        cache_key = FormatHandler._cache_key(cfg)
        if cache_key:
            cached = FormatHandler._FORMAT_MESSAGE_CACHE.get(cache_key)
            if cached is not None:
                return list(cached)
        system_content = None
        if t == "markdown":
            system_content = None
        elif t == "json":
            system_content = None
        elif t == "json_schema":
            schema_text = json.dumps(cfg.get("schema"), ensure_ascii=False) if cfg.get("schema") else ""
            if schema_text:
                system_content = None
        user_content = None
        if t == "markdown":
            user_content = "请确保回应使用 Markdown 的标题或列表组织内容，避免纯文本。"
        elif t == "json":
            user_content = "请仅输出合法 JSON，不要附加任何说明或代码块。"
        elif t == "json_schema":
            schema_text = json.dumps(cfg.get("schema"), ensure_ascii=False) if cfg.get("schema") else ""
            if schema_text:
                user_content = f"请严格按照以下 JSON Schema 返回完整字段: {schema_text}"
        messages: List[ICSMessage] = []
        if system_content:
            messages.append(ICSMessage(role="system", content=system_content))
        if user_content:
            messages.append(ICSMessage(role="user", content=user_content))
        if cache_key:
            FormatHandler._FORMAT_MESSAGE_CACHE[cache_key] = tuple(messages)
        return messages

    @staticmethod
    def response_format(cfg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not cfg:
            return None
        t = cfg.get("type")
        if t == "json":
            return {"type": "json_object"}
        if t == "json_schema":
            return {"type": "json_schema", "json_schema": {"name": cfg.get("name"), "schema": cfg.get("schema")}}
        return None

    @staticmethod
    def process(value: Any, cfg: Optional[Dict[str, Any]]) -> Any:
        if not cfg:
            return value
        t = cfg.get("type")
        if t in ("json", "json_schema"):
            return FormatHandler._to_json(value, cfg.get("schema") if t == "json_schema" else None)
        if t == "markdown":
            return FormatHandler._to_md(value)
        return value

    @staticmethod
    def _to_json(val, schema=None):
        v = val.model_dump() if hasattr(val, "model_dump") else val
        if isinstance(v, (dict, list)):
            data = v
        elif isinstance(v, str):
            try:
                data = json.loads(v)
            except json.JSONDecodeError as e:
                raise LLMValidationError("返回内容不是合法 JSON") from e
        else:
            raise LLMValidationError("返回内容不是 JSON 格式")
        if schema and isinstance(data, dict):
            required = schema.get("required", [])
            missing = [f for f in required if f not in data]
            if missing:
                raise LLMValidationError(f"缺少必需字段: {', '.join(missing)}")
        return data

    @staticmethod
    def _to_md(val):
        v = val.model_dump() if hasattr(val, "model_dump") else val
        if isinstance(v, str):
            s = v.strip()
            if not s:
                raise LLMValidationError("返回内容为空")
            return s
        if v is None:
            raise LLMValidationError("无返回内容")
        return str(v)

    @staticmethod
    def _cache_key(cfg: Dict[str, Any]) -> Optional[str]:
        try:
            fmt_type = cfg.get("type")
        except AttributeError:
            return None
        if not fmt_type:
            return None
        if fmt_type == "json_schema":
            schema = cfg.get("schema")
            try:
                schema_repr = json.dumps(schema, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                schema_repr = repr(schema)
            name = cfg.get("name") or ""
            return f"json_schema::{name}::{schema_repr}"
        return str(fmt_type)
