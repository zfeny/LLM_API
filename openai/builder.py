"""ICS 构建器。"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

from llm.exceptions import LLMConfigError, LLMValidationError
from llm.models import ICSMessage, ICSRequest, MessageEntry
from llm.parser import YAMLRequestParser

from .config import LLMAPIConfig
from .format import FormatHandler

logger = logging.getLogger(__name__)


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
                # 检查是否包含图片（多模态消息）
                if entry.images:
                    content = self._build_multimodal_content(entry)
                    messages.append(ICSMessage(role=entry.role, content=content))
                else:
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

    def _build_multimodal_content(self, entry: MessageEntry) -> List[Dict[str, Any]]:
        """构建多模态内容列表。"""
        content_parts = []

        # 添加文本内容（如果有）
        if entry.content:
            content_parts.append({
                "type": "text",
                "text": entry.content
            })

        # 处理图片
        if entry.images:
            urls = entry.images.get("urls", [])
            for url in urls:
                # 处理 URL（本地路径转在线链接）
                processed_url = self._process_image_url(url)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": processed_url}
                })

        return content_parts

    def _process_image_url(self, url: str) -> str:
        """处理图片 URL，统一转换为 base64 data URI。

        注意：Gemini 的 OpenAI 兼容接口对直接 URL 支持有限，
        使用 base64 编码是最可靠的方案。
        """
        import base64
        import mimetypes

        # 检查是否为本地文件路径
        path = Path(url)
        if path.exists():
            # 本地文件：读取并转换为 base64
            try:
                with open(path, "rb") as image_file:
                    image_data = image_file.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')

                # 检测 MIME 类型
                mime_type, _ = mimetypes.guess_type(str(path))
                if not mime_type or not mime_type.startswith('image/'):
                    mime_type = 'image/jpeg'  # 默认类型

                data_uri = f"data:{mime_type};base64,{base64_image}"
                logger.info("本地图片已编码为 base64: %s (%d 字符)", url, len(base64_image))
                return data_uri
            except Exception as exc:  # noqa: BLE001
                logger.error("读取本地图片失败 %s: %s", url, exc)
                raise LLMValidationError(f"读取本地图片失败: {url}, 错误: {exc}")

        # 检查是否为 HTTP(S) URL
        if url.startswith(("http://", "https://")):
            # 在线 URL：下载后转换为 base64
            try:
                import requests
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                image_data = response.content
                base64_image = base64.b64encode(image_data).decode('utf-8')

                # 从响应头获取 MIME 类型
                mime_type = response.headers.get('Content-Type', 'image/jpeg')
                if not mime_type.startswith('image/'):
                    mime_type = 'image/jpeg'

                data_uri = f"data:{mime_type};base64,{base64_image}"
                logger.info("在线图片已下载并编码为 base64: %s (%d 字符)", url, len(base64_image))
                return data_uri
            except ImportError:
                logger.error("requests 库未安装，无法下载在线图片")
                raise LLMValidationError("下载在线图片需要 requests 库: pip install requests")
            except Exception as exc:  # noqa: BLE001
                logger.error("下载在线图片失败 %s: %s", url, exc)
                raise LLMValidationError(f"下载在线图片失败: {url}, 错误: {exc}")

        # 既不是本地路径也不是 HTTP URL
        logger.warning("无法识别的图片路径: %s", url)
        raise LLMValidationError(f"无法识别的图片路径: {url}")

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
