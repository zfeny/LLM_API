"""ICS 构建器。"""
from __future__ import annotations
import uuid
from typing import Any, Dict, List

from .config import LLMAPIConfig
from .exceptions import LLMConfigError, LLMValidationError
from .format import FormatHandler
from .models import ICSMessage, ICSRequest, MessageEntry
from .parser import YAMLRequestParser


class ICSBuilder:
    """ICS 构建器。"""
    def __init__(self, config: LLMAPIConfig):
        self._config = config

    def build(self, parsed: Dict[str, Any]) -> ICSRequest:
        msgs = parsed["messages"]
        base_msgs = self._build_message_chain(msgs)

        gen = dict(parsed.get("generation", {}))
        model = gen.get("model") or self._config.default_model
        if not model:
            raise LLMConfigError("未提供模型")
        gen["model"] = model

        routing = dict(parsed.get("routing", {}))
        meta = dict(parsed.get("meta", {}))
        meta.setdefault("trace_id", str(uuid.uuid4()))

        fmt = parsed.get("format")
        extra = FormatHandler.build_messages(fmt)
        system_extra = [m for m in extra if m.role == "system"]
        other_extra = [m for m in extra if m.role != "system"]

        messages = list(base_msgs)
        if system_extra:
            insert_at = next((idx for idx, msg in enumerate(messages) if msg.role != "system"), len(messages))
            messages[insert_at:insert_at] = system_extra
        if other_extra:
            messages.extend(other_extra)

        return ICSRequest(messages=messages, generation=gen, routing=routing, meta=meta, format_config=fmt)

    def _build_message_chain(self, data: Any) -> List[ICSMessage]:
        if isinstance(data, list):
            return self._build_from_entries(data)
        if isinstance(data, dict):
            return self._build_from_legacy_dict(data)
        raise LLMValidationError("messages 结构无效")

    def _build_from_entries(self, entries: List[Dict[str, Any]]) -> List[ICSMessage]:
        messages: List[ICSMessage] = []
        for entry in entries:
            if isinstance(entry, MessageEntry):
                messages.append(ICSMessage(role=entry.role, content=entry.content))
                continue
            role = entry.get("role")
            content = entry.get("content")
            if role not in YAMLRequestParser.MESSAGE_ROLES:
                raise LLMValidationError(f"不支持的消息角色: {role}")
            if not isinstance(content, str):
                raise LLMValidationError("消息内容必须为字符串")
            messages.append(ICSMessage(role=role, content=content))
        return messages

    def _build_from_legacy_dict(self, msgs: Dict[str, Any]) -> List[ICSMessage]:
        base_msgs: List[ICSMessage] = []
        system_msg = msgs.get("system")
        if isinstance(system_msg, str) and system_msg.strip():
            base_msgs.append(ICSMessage(role="system", content=system_msg.strip()))
        user_msg = msgs.get("user")
        if not isinstance(user_msg, str) or not user_msg.strip():
            raise LLMValidationError("user 必须为非空字符串")
        base_msgs.append(ICSMessage(role="user", content=user_msg.strip()))
        return base_msgs
