"""ICS 构建器（简化版，仅支持文本消息）。"""
from __future__ import annotations
import json
import logging
import uuid
from typing import Any, Dict, List

from llm.exceptions import LLMConfigError, LLMValidationError
from llm.macros import render_macros
from llm.models import ICSMessage, ICSRequest, MessageEntry
from llm.parser import YAMLRequestParser

from .config import GeminiAPIConfig
from .preset_loader import get_preset_system_content

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
        # 第一步：按source分组system消息和其他消息
        preset_sources = {}  # {preset_name: 原始YAML内容}
        custom_systems = []  # 自定义system消息内容列表
        other_entries: List[MessageEntry] = []

        for entry in entries:
            if isinstance(entry, MessageEntry):
                if entry.role == "system":
                    # 收集system消息（按来源分组）
                    if entry.content:  # 只收集非空的system消息
                        source = entry.source or "custom"

                        if source.startswith("preset:"):
                            # 来自preset的system消息
                            preset_name = source[7:]  # 去掉"preset:"前缀
                            if preset_name not in preset_sources:
                                # 读取preset的system内容（只提取system角色的内容，不包含前缀）
                                try:
                                    raw_content = get_preset_system_content(preset_name)
                                    preset_sources[preset_name] = render_macros(raw_content) if raw_content else raw_content
                                except Exception as exc:  # noqa: BLE001
                                    logger.warning("无法读取preset '%s' 的system内容: %s", preset_name, exc)
                                    # 如果读取失败，将内容加入custom
                                    custom_systems.append(render_macros(entry.content))
                        else:
                            # 自定义system消息
                            custom_systems.append(render_macros(entry.content))
                else:
                    # 其他消息保留
                    processed_content = render_macros(entry.content) if entry.content else entry.content
                    other_entries.append(
                        MessageEntry(
                            role=entry.role,
                            content=processed_content,
                            images=entry.images,
                            source=entry.source,
                        )
                    )
            else:
                # 兼容旧格式（字典）
                role = entry.get("role")
                content = entry.get("content")
                if role not in YAMLRequestParser.MESSAGE_ROLES:
                    raise LLMValidationError(f"不支持的消息角色: {role}")
                if not isinstance(content, str):
                    raise LLMValidationError("消息内容必须为字符串")
                if role == "system" and content.strip():
                    # 旧格式的system消息视为custom
                    custom_systems.append(render_macros(content))
                elif role != "system":
                    other_entries.append(
                        MessageEntry(role=role, content=render_macros(content), images=None, source=None)
                    )

        # 第二步：构建最终消息列表
        messages: List[ICSMessage] = []

        # 组装system消息为dict对象，然后转换为JSON字符串（如果有）
        if preset_sources or custom_systems:
            system_dict = {}

            # 添加preset来源的内容
            for preset_name, rendered_content in preset_sources.items():
                system_dict[preset_name] = rendered_content

            # 添加custom来源的内容
            if custom_systems:
                system_dict["custom"] = custom_systems

            # 转换为JSON字符串（不使用indent以保持紧凑）
            merged_system = json.dumps(system_dict, ensure_ascii=False)
            messages.append(ICSMessage(role="system", content=merged_system))

        # 添加其他消息（保持顺序）
        for entry in other_entries:
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

        return messages

    def _build_from_legacy_dict(self, msgs: Dict[str, Any]) -> List[ICSMessage]:
        base_msgs: List[ICSMessage] = []
        system_msg = msgs.get("system")
        if isinstance(system_msg, str) and system_msg.strip():
            base_msgs.append(ICSMessage(role="system", content=render_macros(system_msg.strip())))
        user_msg = msgs.get("user")
        if not isinstance(user_msg, str) or not user_msg.strip():
            raise LLMValidationError("user 必须为非空字符串")
        base_msgs.append(ICSMessage(role="user", content=render_macros(user_msg.strip())))
        return base_msgs
