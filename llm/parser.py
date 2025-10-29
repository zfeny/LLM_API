"""YAML request parser."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .exceptions import LLMValidationError
from .models import MessageEntry

try:
    import yaml
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("需要 PyYAML: pip install pyyaml") from exc

PresetLoader = Callable[[str], List[MessageEntry]]
_preset_loader: Optional[PresetLoader] = None


def register_preset_loader(loader: PresetLoader) -> None:
    """Register a preset loader so YAML parser can expand preset references."""
    global _preset_loader
    _preset_loader = loader


class YAMLRequestParser:
    """Parser converting YAML prompts into internal message entries."""

    REQUIRED = ("user",)
    OPTIONAL = ("routing", "meta")
    FORMATS = {"text", "markdown", "json", "json_schema"}
    MESSAGE_ROLES = {"system", "user", "assistant"}

    @staticmethod
    def parse(raw: str) -> Dict[str, Any]:
        try:
            data = yaml.safe_load(raw) or {}
        except yaml.YAMLError as exc:
            raise LLMValidationError(f"YAML 解析失败: {exc}") from exc

        if not isinstance(data, dict) or "messages" not in data:
            raise LLMValidationError("YAML 顶层必须包含 'messages'")

        message_entries = YAMLRequestParser._normalize_messages(data["messages"])
        result: Dict[str, Any] = {"messages": message_entries}

        generation = data.get("generation")
        fmt = None
        if generation and isinstance(generation, dict):
            generation_copy = dict(generation)
            fmt_raw = generation_copy.pop("format", None)
            if fmt_raw:
                fmt = YAMLRequestParser._parse_fmt(fmt_raw)
            if generation_copy:
                result["generation"] = generation_copy

        for section in YAMLRequestParser.OPTIONAL:
            val = data.get(section)
            if val and isinstance(val, dict):
                result[section] = val

        if fmt:
            result["format"] = fmt
        return result

    @staticmethod
    def _parse_fmt(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, str):
            fmt_type = raw.strip().lower()
            if not fmt_type:
                raise LLMValidationError("format 字符串不能为空")
            if fmt_type not in YAMLRequestParser.FORMATS:
                raise LLMValidationError(f"format 仅支持: {', '.join(sorted(YAMLRequestParser.FORMATS))}")
            return {"type": fmt_type}

        if not isinstance(raw, dict):
            raise LLMValidationError("format 必须为字符串或字典")

        fmt_type_raw = raw.get("type")
        if fmt_type_raw is None:
            fmt_type = "text"
        elif isinstance(fmt_type_raw, str):
            fmt_type = fmt_type_raw.strip().lower() or "text"
        else:
            raise LLMValidationError("format.type 必须为字符串")

        if fmt_type not in YAMLRequestParser.FORMATS:
            raise LLMValidationError(f"format.type 仅支持: {', '.join(sorted(YAMLRequestParser.FORMATS))}")

        parsed = {"type": fmt_type}
        if fmt_type == "json_schema":
            name = raw.get("name") or raw.get("schema_name")
            schema = raw.get("schema")
            if not name or not isinstance(name, str):
                raise LLMValidationError("json_schema 需要 name")
            if not isinstance(schema, dict):
                raise LLMValidationError("json_schema 需要 schema 字典")
            parsed["name"] = name
            parsed["schema"] = schema
        return parsed

    @staticmethod
    def _normalize_messages(raw_msgs: Any) -> List[MessageEntry]:
        if isinstance(raw_msgs, list):
            entries = YAMLRequestParser._normalize_messages_from_list(raw_msgs)
        elif isinstance(raw_msgs, dict):
            entries = YAMLRequestParser._normalize_messages_from_dict(raw_msgs)
        else:
            raise LLMValidationError("messages 必须是列表或字典")

        if not any(entry.role == "user" for entry in entries):
            raise LLMValidationError(f"缺少必填字段: {', '.join(YAMLRequestParser.REQUIRED)}")
        return entries

    @staticmethod
    def _normalize_messages_from_list(raw_list: List[Any]) -> List[MessageEntry]:
        entries: List[MessageEntry] = []
        for item in raw_list:
            if isinstance(item, dict) and "preset" in item:
                preset_name = item.get("preset")
                if not isinstance(preset_name, str):
                    raise LLMValidationError("preset 值必须为字符串")
                if _preset_loader is None:
                    raise LLMValidationError("未注册预设加载器，无法解析 preset 引用")
                preset_entries = _preset_loader(preset_name.strip())
                entries.extend(preset_entries)
            else:
                entry = YAMLRequestParser._extract_role_content(item)
                entries.append(entry)
        return entries

    @staticmethod
    def _normalize_messages_from_dict(raw_dict: Dict[str, Any]) -> List[MessageEntry]:
        entries: List[MessageEntry] = []
        for raw_key, raw_val in raw_dict.items():
            role = YAMLRequestParser._normalize_role(raw_key)
            values = raw_val if isinstance(raw_val, list) else [raw_val]
            for val in values:
                if not isinstance(val, str):
                    raise LLMValidationError("messages.* 的值必须为字符串或字符串列表")
                content = val.strip()
                if role in ("user", "assistant"):
                    if not content:
                        raise LLMValidationError(f"{role} 必须为非空字符串")
                elif not content:
                    continue
                source = "custom" if role == "system" else None
                entries.append(MessageEntry(role=role, content=content, source=source))
        return entries

    @staticmethod
    def _extract_role_content(item: Any) -> MessageEntry:
        if isinstance(item, MessageEntry):
            return item

        if not isinstance(item, dict):
            raise LLMValidationError("messages 列表项必须为对象")

        if "images" in item:
            role = None
            content = ""
            for possible_role in YAMLRequestParser.MESSAGE_ROLES:
                if possible_role in item:
                    role = possible_role
                    role_content = item[possible_role]
                    if not isinstance(role_content, str):
                        raise LLMValidationError(f"{possible_role} 内容必须为字符串")
                    content = role_content.strip()
                    break
            if role is None:
                role = "user"

            images_value = item["images"]
            if isinstance(images_value, list):
                if not all(isinstance(x, str) for x in images_value):
                    raise LLMValidationError("images 列表必须全是字符串路径")
                urls = images_value
                contents: List[Any] = []
            elif isinstance(images_value, dict):
                urls = images_value.get("urls", [])
                contents = images_value.get("contents", [])
            elif images_value is None and ("urls" in item or "contents" in item):
                urls = item.get("urls", [])
                contents = item.get("contents", [])
            else:
                raise LLMValidationError("images 格式错误，需要列表、字典或 urls/contents 字段")

            if not isinstance(urls, list):
                raise LLMValidationError("urls 必须为列表")
            if contents and not isinstance(contents, list):
                raise LLMValidationError("contents 必须为列表")

            return MessageEntry(role=role, content=content, images={"urls": urls, "contents": contents})

        if "role" in item and "content" in item:
            role = YAMLRequestParser._normalize_role(item["role"])
            content = item["content"]
        elif len(item) == 1:
            raw_role, content = next(iter(item.items()))
            role = YAMLRequestParser._normalize_role(raw_role)
        else:
            raise LLMValidationError("messages 列表项需包含 role/content、单键角色或 images")

        if not isinstance(content, str):
            raise LLMValidationError("消息内容必须为字符串")
        stripped = content.strip()
        if role in ("user", "assistant"):
            if not stripped:
                raise LLMValidationError(f"{role} 必须为非空字符串")
        elif not stripped:
            stripped = ""

        source = "custom" if role == "system" else None
        return MessageEntry(role=role, content=stripped, source=source)

    @staticmethod
    def _normalize_role(raw_role: Any) -> str:
        token = str(raw_role).strip()
        if "." in token:
            prefix, suffix = token.split(".", 1)
            if prefix.strip().isdigit():
                token = suffix.strip()
        role = token.lower()
        if role not in YAMLRequestParser.MESSAGE_ROLES:
            raise LLMValidationError("messages 仅支持 system/user/assistant 角色")
        return role


__all__ = ["YAMLRequestParser", "register_preset_loader"]
