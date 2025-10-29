"""Gemini 适配器。"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from llm.exceptions import LLMConfigError, LLMValidationError
from llm.models import ICSRequest

from .format import GeminiFormatHandler

try:
    from google.genai import types as genai_types
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("需要 google-genai SDK: pip install google-genai") from exc

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
            "history": [Content, ...],     # 历史消息（除最后一条）
            "current_message": Content,    # 当前消息（最后一条，必须是 user）
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

        current_message = GeminiAdapter._build_content(
            role="user",
            content=current_message_content,
            file_uploader=file_uploader,
            log_upload=False,
            format_config=format_config,
            apply_format_suffix=True,
        )

        # 2. 构建 generation_config
        generation_config = {
            key: gen[key]
            for key in ("max_output_tokens", "temperature", "top_p", "top_k")
            if key in gen
        }

        # 2.2 处理图片生成配置
        response_mode = gen.get("response", "text")  # text/image/both，默认text
        image_cfg = gen.get("image", {})

        # 处理 thinking 配置（图片生成模式不支持thinking）
        think_value = gen.get("think")
        if think_value is not None and response_mode == "text":
            thinking_config = {"thinking_budget": think_value}
            # 当 think=-1 时，启用思考总结
            if think_value == -1:
                thinking_config["include_thoughts"] = True
            generation_config["thinking_config"] = thinking_config

        # 如果需要生成图片，配置 response_modalities
        if response_mode == "image":
            generation_config["response_modalities"] = ["Image"]
        elif response_mode == "both":
            generation_config["response_modalities"] = ["Text", "Image"]
        # text模式不需要设置（默认为Text）

        # 如果需要生成图片，配置 image_config
        if response_mode in ("image", "both"):
            aspect_ratio = image_cfg.get("ratio", "1:1")
            # 确保aspect_ratio是字符串类型（YAML可能将16:9解析为数字）
            if not isinstance(aspect_ratio, str):
                aspect_ratio = str(aspect_ratio)
            generation_config["image_config"] = {
                "aspect_ratio": aspect_ratio
            }

        # 2.3 合并 format 相关的 generation_config（用于平台原生模式）
        format_gen_config = GeminiFormatHandler.get_generation_config(format_config)
        if format_gen_config:
            generation_config.update(format_gen_config)

        payload = {"model": model, "history": gemini_history, "current_message": current_message}

        if system_instruction:
            payload["system_instruction"] = system_instruction

        if generation_config:
            payload["generation_config"] = generation_config

        return payload

    @staticmethod
    def _build_content(
        role: str,
        content: Any,
        *,
        file_uploader: Optional[Any],
        log_upload: bool,
        format_config: Optional[Dict[str, Any]],
        apply_format_suffix: bool = False,
    ) -> genai_types.Content:
        if role == "system":
            raise LLMValidationError("system 消息只能作为 system_instruction 提供")

        descriptors = GeminiAdapter._build_descriptors(content, file_uploader, log_upload=log_upload)
        if apply_format_suffix:
            GeminiAdapter._apply_format_suffix(descriptors, format_config)
        parts = [GeminiAdapter._descriptor_to_part(descriptor) for descriptor in descriptors]
        return genai_types.Content(role=role, parts=parts)

    @staticmethod
    def _convert_history(messages: List[Any], file_uploader: Optional[Any] = None) -> List[genai_types.Content]:
        history: List[genai_types.Content] = []
        for msg in messages:
            role = "model" if msg.role == "assistant" else msg.role
            if role == "system":
                raise LLMValidationError("system 消息只能出现在第一条，且已被提取为 system_instruction")
            history.append(
                GeminiAdapter._build_content(
                    role=role,
                    content=msg.content,
                    file_uploader=file_uploader,
                    log_upload=True,
                    format_config=None,
                )
            )
        return history

    @staticmethod
    def _build_descriptors(content: Any, file_uploader: Optional[Any], *, log_upload: bool) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"kind": "text", "text": content}]
        if isinstance(content, list):
            return GeminiAdapter._build_descriptors_from_list(content, file_uploader, log_upload=log_upload)
        raise LLMValidationError("消息内容必须为字符串或列表")

    @staticmethod
    def _build_descriptors_from_list(
        content_parts: List[Dict[str, Any]],
        file_uploader: Optional[Any],
        *,
        log_upload: bool,
    ) -> List[Dict[str, Any]]:
        descriptors: List[Dict[str, Any]] = []
        for part in content_parts:
            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise LLMValidationError("text part 需要 text 字段")
                descriptors.append({"kind": "text", "text": text})
            elif part_type == "image":
                image_path = part.get("path")
                if not image_path:
                    raise LLMValidationError("图片部分缺少 path 字段")
                if file_uploader is None:
                    raise LLMValidationError("需要 file_uploader 来处理图片消息")
                if log_upload:
                    logger.debug("上传图片: %s", image_path)
                uploaded_file = file_uploader.upload_file(image_path)
                descriptors.append({"kind": "file", "file": uploaded_file})
            else:
                raise LLMValidationError(f"未知的多模态类型: {part_type}")
        return descriptors

    @staticmethod
    def _apply_format_suffix(parts: List[Dict[str, Any]], format_config: Optional[Dict[str, Any]]) -> None:
        if not parts:
            return
        suffix = GeminiFormatHandler.get_prompt_suffix(format_config)
        if not suffix:
            return
        for descriptor in reversed(parts):
            if descriptor.get("kind") == "text":
                descriptor["text"] = f"{descriptor['text']}{suffix}"
                break

    @staticmethod
    def _descriptor_to_part(descriptor: Dict[str, Any]) -> genai_types.Part:
        kind = descriptor.get("kind")
        if kind == "text":
            return genai_types.Part(text=descriptor["text"])
        if kind == "file":
            uploaded_file = descriptor["file"]
            return genai_types.Part.from_uri(
                file_uri=uploaded_file.uri,
                mime_type=uploaded_file.mime_type,
            )
        raise LLMValidationError(f"未知的多模态类型: {kind}")
