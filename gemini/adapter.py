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

        tools_config = gen.get("tools")
        if tools_config:
            payload_tools, inline_citations = GeminiAdapter._build_tools(tools_config)
            if payload_tools:
                payload["tools"] = payload_tools
            if inline_citations:
                payload["inline_citations"] = True

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

    @staticmethod
    def _build_tools(raw_tools: Any) -> tuple[List[genai_types.Tool], bool]:
        if not isinstance(raw_tools, list):
            raise LLMValidationError("generation.tools 必须为列表")

        tools: List[genai_types.Tool] = []
        inline_citations = False

        for tool_cfg in raw_tools:
            normalized = GeminiAdapter._normalize_tool_config(tool_cfg)
            if normalized.pop("_inline_citations", False):
                inline_citations = True
            tool_type_raw = normalized["type"]

            tool_type = tool_type_raw.strip().lower()

            if tool_type == "google_search":
                tools.append(GeminiAdapter._build_google_search(normalized))
            elif tool_type == "google_search_retrieval":
                tools.append(GeminiAdapter._build_google_search_retrieval(normalized))
            elif tool_type == "url_context":
                tools.append(GeminiAdapter._build_url_context(normalized))
            else:
                raise LLMValidationError(f"不支持的工具类型: {tool_type_raw}")

        return tools, inline_citations

    @staticmethod
    def _build_google_search(tool_cfg: Dict[str, Any]) -> genai_types.Tool:
        options = tool_cfg.get("options", {})
        if options and not isinstance(options, dict):
            raise LLMValidationError("google_search 工具 options 必须为对象")

        def pick(key: str) -> Any:
            if key in tool_cfg and tool_cfg[key] is not None:
                return tool_cfg[key]
            if isinstance(options, dict):
                return options.get(key)
            return None

        kwargs: Dict[str, Any] = {}

        exclude_domains = pick("exclude_domains")
        if exclude_domains is not None:
            if not isinstance(exclude_domains, list) or not all(isinstance(item, str) for item in exclude_domains):
                raise LLMValidationError("exclude_domains 必须为字符串列表")
            kwargs["exclude_domains"] = exclude_domains

        time_filter = pick("time_range_filter")
        if time_filter is not None:
            kwargs["timeRangeFilter"] = GeminiAdapter._parse_time_range_filter(time_filter)

        google_tool = genai_types.GoogleSearch(**kwargs)
        return genai_types.Tool(googleSearch=google_tool)

    @staticmethod
    def _build_google_search_retrieval(tool_cfg: Dict[str, Any]) -> genai_types.Tool:
        options = tool_cfg.get("options", {})
        if options and not isinstance(options, dict):
            raise LLMValidationError("google_search 工具 options 必须为对象")

        mode_value = tool_cfg.get("mode") or (options.get("mode") if isinstance(options, dict) else None)
        threshold_value = (
            tool_cfg.get("dynamic_threshold")
            if tool_cfg.get("dynamic_threshold") is not None
            else (options.get("dynamic_threshold") if isinstance(options, dict) else None)
        )

        dynamic_cfg = None
        if mode_value or threshold_value is not None:
            mode_enum = None
            if mode_value:
                mode_token = str(mode_value).strip().upper()
                if not mode_token.startswith("MODE_"):
                    mode_token = f"MODE_{mode_token}"
                mode_enum = getattr(genai_types.DynamicRetrievalConfigMode, mode_token, None)
                if mode_enum is None:
                    raise LLMValidationError(f"不支持的 google_search mode: {mode_value}")

            dynamic_kwargs: Dict[str, Any] = {}
            if mode_enum is not None:
                dynamic_kwargs["mode"] = mode_enum
            if threshold_value is not None:
                try:
                    dynamic_kwargs["dynamicThreshold"] = float(threshold_value)
                except (TypeError, ValueError) as exc:  # noqa: BLE001
                    raise LLMValidationError("dynamic_threshold 必须为数字") from exc

            dynamic_cfg = genai_types.DynamicRetrievalConfig(**dynamic_kwargs)

        search_kwargs: Dict[str, Any] = {}
        if dynamic_cfg is not None:
            search_kwargs["dynamicRetrievalConfig"] = dynamic_cfg

        google_tool = genai_types.GoogleSearchRetrieval(**search_kwargs)
        return genai_types.Tool(googleSearchRetrieval=google_tool)

    @staticmethod
    def _build_url_context(tool_cfg: Dict[str, Any]) -> genai_types.Tool:
        options = tool_cfg.get("options", {})
        if options and not isinstance(options, dict):
            raise LLMValidationError("url_context 工具 options 必须为对象")

        urls = tool_cfg.get("urls")
        if urls is None and isinstance(options, dict):
            urls = options.get("urls")

        if urls is not None:
            if not isinstance(urls, list) or not all(isinstance(item, str) for item in urls):
                raise LLMValidationError("url_context.urls 必须为字符串列表")
            if urls:
                logger.warning(
                    "url_context 工具无需显式 urls 参数。请确保在提示或附加内容中包含链接。将忽略: %s",
                    urls,
                )

        return genai_types.Tool(url_context=genai_types.UrlContext())

    @staticmethod
    def _normalize_tool_config(tool_cfg: Any) -> Dict[str, Any]:
        alias_map = {
            "search": "google_search",
            "google_search": "google_search",
            "google_search_retrieval": "google_search_retrieval",
            "url_context": "url_context",
        }

        if isinstance(tool_cfg, str):
            alias = tool_cfg.strip().lower()
            mapped = alias_map.get(alias)
            if not mapped:
                raise LLMValidationError(f"不支持的工具类型: {tool_cfg}")
            return {"type": mapped}

        if isinstance(tool_cfg, dict):
            cfg = dict(tool_cfg)
            inline_flag = False

            if "type" in cfg and isinstance(cfg["type"], str):
                mapped = alias_map.get(cfg["type"].strip().lower())
                if not mapped:
                    raise LLMValidationError(f"不支持的工具类型: {cfg['type']}")
                cfg["type"] = mapped
            elif len(cfg) == 1:
                key, value = next(iter(cfg.items()))
                alias = alias_map.get(str(key).strip().lower())
                if not alias:
                    raise LLMValidationError(f"不支持的工具类型: {key}")
                cfg = {"type": alias}
                inline_flag = GeminiAdapter._parse_inline_flag(value)
                if isinstance(value, dict):
                    cfg.update(value)
            else:
                raise LLMValidationError("工具配置对象缺少 type 字段")

            if "inline_citations" in cfg:
                inline_flag = bool(cfg.pop("inline_citations"))
            if inline_flag:
                cfg["_inline_citations"] = True
            return cfg

        raise LLMValidationError("tools 列表项必须是字符串或对象")

    @staticmethod
    def _parse_inline_flag(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            token = value.strip().lower()
            return token in {"inline", "in_line", "true", "yes", "on", "1"}
        return False

    @staticmethod
    def _parse_time_range_filter(raw: Any) -> genai_types.Interval:
        if isinstance(raw, genai_types.Interval):
            return raw
        if isinstance(raw, dict):
            start_raw = raw.get("start") or raw.get("start_time") or raw.get("startTime")
            end_raw = raw.get("end") or raw.get("end_time") or raw.get("endTime")

            from datetime import datetime

            def convert(value: Any) -> Optional[datetime]:
                if value is None:
                    return None
                if isinstance(value, datetime):
                    return value
                if isinstance(value, str):
                    try:
                        return datetime.fromisoformat(value)
                    except ValueError as exc:  # noqa: BLE001
                        raise LLMValidationError("time_range_filter 时间需要 ISO 格式字符串") from exc
                raise LLMValidationError("time_range_filter 时间必须为 datetime 或 ISO 字符串")

            start_dt = convert(start_raw)
            end_dt = convert(end_raw)
            return genai_types.Interval(start_time=start_dt, end_time=end_dt)

        raise LLMValidationError("time_range_filter 必须为 Interval 或字典")
