"""YAML 请求解析器。"""
from __future__ import annotations
from typing import Any, Dict, List

from .exceptions import LLMValidationError
from .models import MessageEntry

try:
    import yaml
except ImportError as e:
    raise ImportError("需要 PyYAML: pip install pyyaml") from e


class YAMLRequestParser:
    """YAML 解析器。"""
    REQUIRED = ("user",)  # 只有 user 必填
    OPTIONAL = ("routing", "meta")
    FORMATS = {"text", "markdown", "json", "json_schema"}
    MESSAGE_ROLES = {"system", "user", "assistant"}

    @staticmethod
    def parse(raw: str):
        try:
            data = yaml.safe_load(raw) or {}
        except yaml.YAMLError as e:
            raise LLMValidationError(f"YAML 解析失败: {e}") from e
        if not isinstance(data, dict) or "messages" not in data:
            raise LLMValidationError("YAML 顶层必须包含 'messages'")
        message_entries = YAMLRequestParser._normalize_messages(data["messages"])
        result = {"messages": message_entries}

        gen = data.get("generation")
        fmt = None
        if gen and isinstance(gen, dict):
            gen_copy = dict(gen)
            fmt_raw = gen_copy.pop("format", None)
            if fmt_raw:
                fmt = YAMLRequestParser._parse_fmt(fmt_raw)
            if gen_copy:
                result["generation"] = gen_copy

        for sec in YAMLRequestParser.OPTIONAL:
            val = data.get(sec)
            if val and isinstance(val, dict):
                result[sec] = val

        if fmt:
            result["format"] = fmt
        return result

    @staticmethod
    def _parse_fmt(raw):
        # 支持简化语法：format: json 等价于 format: {type: json}
        if isinstance(raw, str):
            fmt_type = raw.strip().lower()
            if not fmt_type:
                raise LLMValidationError("format 字符串不能为空")
            if fmt_type not in YAMLRequestParser.FORMATS:
                raise LLMValidationError(f"format 仅支持: {', '.join(sorted(YAMLRequestParser.FORMATS))}")
            return {"type": fmt_type}

        # 原有字典格式
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
        user_present = any(entry.role == "user" for entry in entries)
        if not user_present:
            raise LLMValidationError(f"缺少必填字段: {', '.join(YAMLRequestParser.REQUIRED)}")
        return entries

    @staticmethod
    def _normalize_messages_from_list(raw_list: List[Any]) -> List[MessageEntry]:
        entries: List[MessageEntry] = []
        for idx, item in enumerate(raw_list):
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
                    # system 允许为空，跳过
                    continue
                entries.append(MessageEntry(role=role, content=content))
        return entries

    @staticmethod
    def _extract_role_content(item: Any) -> MessageEntry:
        if isinstance(item, MessageEntry):
            return item

        if not isinstance(item, dict):
            raise LLMValidationError("messages 列表项必须为对象")

        # 检查是否包含 images 字段（多模态消息）
        # 支持多种格式：
        # 1. 简化格式（推荐）: - user: 文本\n  images: [path1, path2]
        # 2. 完整格式: - images: {urls: [...], contents: [...]}
        # 3. 平级格式: - images:\n  urls: [...]\n  contents: [...]
        if "images" in item:
            # 提取角色（如果有明确指定的角色，否则默认为 user）
            role = None
            content = ""

            # 检查是否有明确的角色字段
            for possible_role in YAMLRequestParser.MESSAGE_ROLES:
                if possible_role in item:
                    role = possible_role
                    content = item[possible_role]
                    if not isinstance(content, str):
                        raise LLMValidationError(f"{role} 内容必须为字符串")
                    content = content.strip()
                    break

            # 如果没有明确角色，默认为 user
            if role is None:
                role = "user"

            # 解析 images 字段
            images_value = item["images"]

            # 格式 1: images 是字符串列表（简化格式）
            if isinstance(images_value, list):
                # 检查是否全是字符串（文件路径）
                if all(isinstance(x, str) for x in images_value):
                    urls = images_value
                    contents = []
                else:
                    raise LLMValidationError("images 列表必须全是字符串路径")

            # 格式 2: images 是字典
            elif isinstance(images_value, dict):
                urls = images_value.get("urls", [])
                contents = images_value.get("contents", [])

            # 格式 3: images 为 null，urls 和 contents 在同级
            elif images_value is None and ("urls" in item or "contents" in item):
                urls = item.get("urls", [])
                contents = item.get("contents", [])

            else:
                raise LLMValidationError("images 格式错误，需要列表、字典或 urls/contents 字段")

            if not isinstance(urls, list):
                raise LLMValidationError("urls 必须为列表")
            if contents and not isinstance(contents, list):
                raise LLMValidationError("contents 必须为列表")

            return MessageEntry(
                role=role,
                content=content,
                images={"urls": urls, "contents": contents}
            )

        # 原有的文本消息解析逻辑
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
        content = content.strip()
        if role in ("user", "assistant"):
            if not content:
                raise LLMValidationError(f"{role} 必须为非空字符串")
        elif not content:
            # system 允许为空，保持空字符串
            content = ""

        return MessageEntry(role=role, content=content)

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
