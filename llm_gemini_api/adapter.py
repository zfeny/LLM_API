"""Gemini 适配器。"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from .exceptions import LLMConfigError, LLMValidationError
from .format import GeminiFormatHandler
from .models import ICSRequest

logger = logging.getLogger(__name__)


class GeminiAdapter:
    """Gemini 适配器：将 ICS 消息转换为 Gemini SDK 格式。"""

    @staticmethod
    def to_chat(ics: ICSRequest, file_uploader: Optional[Any] = None) -> Dict[str, Any]:
        """
        将 ICS 请求转换为 Gemini SDK 格式。

        返回格式：
        {
            "model": "gemini-2.5-flash",
            "system_instruction": "...",  # 可选
            "history": [...],              # 历史消息（除最后一条）
            "current_message": "...",      # 当前消息（最后一条，必须是 user）
            "generation_config": {...}     # 生成配置
        }
        """
        gen = ics.generation
        model = gen.get("model")
        if not model:
            raise LLMConfigError("缺少模型参数")

        messages = ics.messages
        if not messages:
            raise LLMValidationError("消息列表不能为空")

        # 提取 system_instruction（如果第一条是 system）
        system_instruction = None
        start_idx = 0
        if messages[0].role == "system":
            system_instruction = messages[0].content
            start_idx = 1

        # 分离历史消息和当前消息
        remaining_messages = messages[start_idx:]
        if not remaining_messages:
            raise LLMValidationError("除 system 外必须至少有一条消息")

        # 最后一条消息必须是 user
        current_message_obj = remaining_messages[-1]
        if current_message_obj.role != "user":
            raise LLMValidationError("最后一条消息必须是 user 角色")

        # 历史消息（除最后一条）
        history_messages = remaining_messages[:-1]

        # 转换历史消息为 Gemini 格式
        gemini_history = GeminiAdapter._convert_history(history_messages, file_uploader)

        # 提取当前消息内容
        current_message_content = current_message_obj.content

        # 处理 format 配置
        format_config = ics.format_config

        # 检查当前消息是否包含多模态内容
        if isinstance(current_message_content, list):
            # 多模态消息，处理图片上传
            current_message = GeminiAdapter._process_multimodal_content(
                current_message_content, file_uploader, format_config
            )
        elif isinstance(current_message_content, str):
            # 纯文本消息
            # 1. 合并格式提示到当前消息（用于提示词工程模式）
            current_message = GeminiFormatHandler.merge_prompt_to_message(current_message_content, format_config)
        else:
            raise LLMValidationError("当前消息内容必须为字符串或列表")

        # 2. 构建 generation_config
        generation_config = {
            key: gen[key]
            for key in ("max_output_tokens", "temperature", "top_p", "top_k")
            if key in gen
        }

        think_value = gen.get("think")
        if think_value is not None:
            thinking_config = {"thinking_budget": think_value}
            # 当 think=-1 时，启用思考总结
            if think_value == -1:
                thinking_config["include_thoughts"] = True
            generation_config["thinking_config"] = thinking_config

        # 2.2 合并 format 相关的 generation_config（用于平台原生模式）
        format_gen_config = GeminiFormatHandler.get_generation_config(format_config)
        if format_gen_config:
            generation_config.update(format_gen_config)

        payload = {
            "model": model,
            "history": gemini_history,
            "current_message": current_message,
        }

        if system_instruction:
            payload["system_instruction"] = system_instruction

        if generation_config:
            payload["generation_config"] = generation_config

        return payload

    @staticmethod
    def _process_multimodal_content(content_parts: List[Dict[str, Any]], file_uploader: Optional[Any], format_config: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理多模态内容，上传图片并返回 Gemini parts 格式。

        返回格式: [{"text": "..."}, {"file_data": {...}}, ...]
        """
        parts = GeminiAdapter._build_parts_from_list(content_parts, file_uploader, log_upload=False)
        GeminiAdapter._apply_format_suffix(parts, format_config)
        return parts

    @staticmethod
    def _convert_history(messages: List[Any], file_uploader: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        将历史消息转换为 Gemini 格式。

        Gemini 格式：
        [
            {"role": "user", "parts": [{"text": "..."}, {"file_data": {...}}]},
            {"role": "model", "parts": [{"text": "..."}]},
            ...
        ]
        """
        gemini_history = []
        for msg in messages:
            # 转换角色：assistant → model
            role = "model" if msg.role == "assistant" else msg.role

            # Gemini 不允许 system 在 history 中（应该已被提取）
            if role == "system":
                raise LLMValidationError("system 消息只能出现在第一条，且已被提取为 system_instruction")

            # 转换内容为 parts 格式
            parts = []

            if isinstance(msg.content, str):
                # 纯文本消息
                parts = [{"text": msg.content}]
            elif isinstance(msg.content, list):
                parts = GeminiAdapter._build_parts_from_list(msg.content, file_uploader, log_upload=True)
            else:
                raise LLMValidationError("消息内容必须为字符串或列表")

            gemini_history.append({
                "role": role,
                "parts": parts
            })

        return gemini_history

    @staticmethod
    def _build_parts_from_list(content_parts: List[Dict[str, Any]], file_uploader: Optional[Any], *, log_upload: bool) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []
        for part in content_parts:
            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise LLMValidationError("text part 需要 text 字段")
                parts.append({"text": text})
            elif part_type == "image":
                image_path = part.get("path")
                if not image_path:
                    raise LLMValidationError("图片部分缺少 path 字段")
                if file_uploader is None:
                    raise LLMValidationError("需要 file_uploader 来处理图片消息")
                if log_upload:
                    logger.debug("上传图片: %s", image_path)
                uploaded_file = file_uploader.upload_file(image_path)
                parts.append({
                    "_file_object": uploaded_file,
                    "file_data": {
                        "file_uri": uploaded_file.uri,
                        "mime_type": uploaded_file.mime_type
                    }
                })
            else:
                raise LLMValidationError(f"未知的多模态类型: {part_type}")
        return parts

    @staticmethod
    def _apply_format_suffix(parts: List[Dict[str, Any]], format_config: Optional[Dict[str, Any]]) -> None:
        format_suffix = GeminiFormatHandler.get_prompt_suffix(format_config)
        if format_suffix and parts:
            for part in reversed(parts):
                if "text" in part:
                    part["text"] += format_suffix
                    break
