"""ICS 构建器（简化版，仅支持文本消息）。"""
from __future__ import annotations
import logging
import uuid
from typing import Any, Dict, List

from .config import GeminiAPIConfig
from .exceptions import LLMConfigError, LLMValidationError
from .models import ICSMessage, ICSRequest, MessageEntry
from .parser import YAMLRequestParser

logger = logging.getLogger(__name__)


class ICSBuilder:
    """ICS 构建器。"""
    def __init__(self, config: GeminiAPIConfig):
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

        messages = list(base_msgs)

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
                # 如果有图片，构建多模态内容
                if entry.images and entry.images.get("urls"):
                    # 构建多模态内容列表
                    content_parts = []

                    # 添加文本内容（如果有）
                    if entry.content:
                        content_parts.append({
                            "type": "text",
                            "text": entry.content
                        })

                    # 添加图片引用（路径，稍后在 adapter 中处理）
                    for image_path in entry.images["urls"]:
                        content_parts.append({
                            "type": "image",
                            "path": image_path  # 存储路径，adapter 中上传
                        })

                    messages.append(ICSMessage(role=entry.role, content=content_parts))
                else:
                    # 纯文本消息
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
