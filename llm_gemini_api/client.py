"""LLM 客户端。"""
from __future__ import annotations
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from ._utils import _retry
from .adapter import GeminiAdapter
from .builder import ICSBuilder
from .config import GeminiAPIConfig, RetryConfig
from .exceptions import LLMConfigError, LLMTransportError, LLMValidationError
from .file_utils import GeminiFileUploader
from .format import GeminiFormatHandler
from .parser import YAMLRequestParser
from .recorder import UsageRecorder

try:
    import google.genai as genai
    from google.genai import types as genai_types
except ImportError as e:
    raise ImportError("需要 google-genai SDK: pip install google-genai") from e

logger = logging.getLogger(__name__)


class _BaseLLMClient:
    """LLM 客户端基类（提取公共方法）。"""
    # 异常类作为类属性，方便外部通过 LLMClient.ConfigError 访问
    ConfigError = LLMConfigError
    ValidationError = LLMValidationError
    TransportError = LLMTransportError

    def __init__(self, config: GeminiAPIConfig, recorder: Optional[UsageRecorder], retry_config: Optional[RetryConfig]):
        self._config = config
        self._builder = ICSBuilder(config)
        self._recorder = recorder or UsageRecorder()
        self._retry_config = retry_config or RetryConfig()
        # 配置新版 SDK 客户端
        self._genai_client = genai.Client(api_key=config.api_key)
        # 初始化文件上传器（用于多模态功能）
        self._file_uploader = GeminiFileUploader(self._genai_client)

    def _extract_result(self, resp: Any, format_config: Optional[Dict[str, Any]] = None) -> Any:
        """从 Gemini 响应中提取并处理结果。"""
        try:
            # 检查是否有思考内容（Thinking mode）
            thoughts = []
            answer_parts = []

            # 尝试从 response.candidates[0].content.parts 提取
            try:
                if hasattr(resp, "candidates") and resp.candidates:
                    candidate = resp.candidates[0]
                    if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                        for part in candidate.content.parts:
                            # 检查是否是思考内容
                            if hasattr(part, "thought") and part.thought:
                                # 这是思考总结
                                if hasattr(part, "text") and part.text:
                                    thoughts.append(part.text)
                            else:
                                # 这是答案内容
                                if hasattr(part, "text") and part.text:
                                    answer_parts.append(part.text)
            except Exception as e:
                logger.debug("无法从 parts 提取思考内容，回退到简单文本提取: %s", e)

            # 如果有思考内容，格式化输出
            if thoughts:
                thought_text = "\n\n".join(thoughts)
                answer_text = "".join(answer_parts)

                # 根据 format 配置处理答案部分
                processed_answer = GeminiFormatHandler.process_response(answer_text, format_config)

                # 如果答案是字符串类型，添加思考块
                if isinstance(processed_answer, str):
                    return f"<GEMINI_THINKING>\n{thought_text}\n</GEMINI_THINKING>\n\n{processed_answer}"
                else:
                    # 如果是 JSON 等非字符串类型，只返回答案（思考内容记录在日志中）
                    logger.info("检测到思考内容（%d 字符），但答案为非文本格式，仅返回答案", len(thought_text))
                    return processed_answer

            # 没有思考内容，使用标准提取
            raw_text = resp.text
            processed = GeminiFormatHandler.process_response(raw_text, format_config)

            return processed
        except Exception as e:
            logger.error("提取响应结果失败: %s", e)
            raise LLMTransportError(f"提取响应结果失败: {e}") from e

    def _record_usage(self, resp: Any, trace_id: Optional[str], model: str):
        """记录使用量。"""
        try:
            # 尝试从不同位置获取 usage_metadata
            usage_metadata = None

            # 方式1: 直接从 resp 获取
            if hasattr(resp, "usage_metadata"):
                usage_metadata = resp.usage_metadata
            # 方式2: 从 resp.result 获取
            elif hasattr(resp, "result") and hasattr(resp.result, "usage_metadata"):
                usage_metadata = resp.result.usage_metadata

            if usage_metadata:
                # 获取 thoughts_token_count（如果有，Thinking mode）
                thoughts_tokens = getattr(usage_metadata, "thoughts_token_count", None)

                usage_dict = {
                    "prompt_tokens": getattr(usage_metadata, "prompt_token_count", None),
                    "completion_tokens": getattr(usage_metadata, "candidates_token_count", None),
                    "total_tokens": getattr(usage_metadata, "total_token_count", None),
                    "thoughts_token_count": thoughts_tokens,  # 添加到 usage_dict
                }

                # 获取 model_version（如果有）
                model_version = None
                if hasattr(resp, "result") and hasattr(resp.result, "model_version"):
                    model_version = resp.result.model_version

                # 使用 model_version 或传入的 model
                actual_model = model_version or model

                # 获取 request_id（如果有）
                request_id = None

                self._recorder.record(
                    model=actual_model,
                    request_id=request_id,
                    trace_id=trace_id,
                    usage=usage_dict
                )

                # 日志输出（包含 thoughts_tokens 如果有）
                if thoughts_tokens:
                    logger.info(
                        "记录使用量 trace_id=%s model=%s prompt=%s completion=%s total=%s thoughts=%s",
                        trace_id,
                        actual_model,
                        usage_dict.get("prompt_tokens"),
                        usage_dict.get("completion_tokens"),
                        usage_dict.get("total_tokens"),
                        thoughts_tokens
                    )
                else:
                    logger.info(
                        "记录使用量 trace_id=%s model=%s prompt=%s completion=%s total=%s",
                        trace_id,
                        actual_model,
                        usage_dict.get("prompt_tokens"),
                        usage_dict.get("completion_tokens"),
                        usage_dict.get("total_tokens")
                    )
        except Exception as e:
            logger.warning("记录使用量失败: %s", e)


class LLMClient(_BaseLLMClient):
    """同步 LLM 客户端。"""
    def __init__(self, config: GeminiAPIConfig, recorder: Optional[UsageRecorder] = None, retry_config: Optional[RetryConfig] = None):
        super().__init__(config, recorder, retry_config)

    @classmethod
    def from_env(cls, recorder: Optional[UsageRecorder] = None, retry_config: Optional[RetryConfig] = None):
        return cls(GeminiAPIConfig.from_env(), recorder, retry_config)

    def invoke_from_yaml(self, yaml_prompt: str, *, dry_run: bool = False, include_debug: bool = False, raw_response: bool = False) -> Union[str, Dict[str, Any], Any]:
        start = time.time()
        trace_id = str(uuid.uuid4())
        logger.info("处理请求 trace_id=%s dry_run=%s", trace_id, dry_run)

        # 1. 解析 YAML
        parsed = YAMLRequestParser.parse(yaml_prompt)
        ics = self._builder.build(parsed)
        if "trace_id" not in ics.meta:
            ics.meta["trace_id"] = trace_id

        # 2. 转换为 Gemini 格式（传入 file_uploader 以支持多模态）
        gemini_payload = GeminiAdapter.to_chat(ics, self._file_uploader)

        if dry_run:
            return {"ics_request": ics.to_payload(), "gemini_payload": gemini_payload}

        # 3. 调用 Gemini SDK
        resp = self._send(gemini_payload, ics.meta.get("trace_id"))
        logger.info("完成 trace_id=%s 耗时=%.2fs", trace_id, time.time() - start)

        # 4. 返回原始响应（如果请求）
        if raw_response:
            return resp

        # 5. 提取并处理结果
        result = self._extract_result(resp, ics.format_config)
        if not include_debug:
            return result
        return {
            "result": result,
            "ics_request": ics.to_payload(),
            "gemini_payload": gemini_payload,
        }

    def _convert_parts_for_new_sdk(self, parts: List[Dict[str, Any]]) -> List[Any]:
        """
        转换 parts 格式以供新 SDK 使用。

        提取 File 对象（如果有），并将文本包装为 Part 对象。
        """
        converted = []
        for part in parts:
            if "_file_object" in part:
                # 使用 file_uri 和 mime_type 创建 Part
                file_obj = part["_file_object"]
                converted.append(genai_types.Part.from_uri(
                    file_uri=file_obj.uri,
                    mime_type=file_obj.mime_type
                ))
            elif "text" in part:
                # 文本部分：需要包装为 Part 对象
                converted.append(genai_types.Part(text=part["text"]))
            else:
                # 其他格式保持不变
                converted.append(part)
        return converted

    def _send(self, payload: Dict[str, Any], trace_id: Optional[str]):
        """发送请求到 Gemini API。"""
        return self._send_with_new_sdk(payload, trace_id)

    def _send_with_new_sdk(self, payload: Dict[str, Any], trace_id: Optional[str]):
        """使用新 SDK 发送请求（支持 thinking）。"""
        @_retry(self._retry_config, is_async=False)
        def _call():
            try:
                model_name = payload["model"]
                system_instruction = payload.get("system_instruction")
                generation_config = payload.get("generation_config", {})
                history = payload.get("history", [])
                current_message = payload["current_message"]

                # 构建 GenerateContentConfig
                config_kwargs = {}

                # 处理 thinking_config
                if "thinking_config" in generation_config:
                    thinking_cfg = generation_config["thinking_config"]
                    thinking_budget = thinking_cfg.get("thinking_budget")
                    include_thoughts = thinking_cfg.get("include_thoughts", False)

                    config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                        thinking_budget=thinking_budget,
                        include_thoughts=include_thoughts
                    )

                # 处理其他 generation_config 参数
                for key in ["temperature", "top_p", "top_k", "max_output_tokens"]:
                    if key in generation_config:
                        config_kwargs[key] = generation_config[key]

                # 处理 response_mime_type 和 response_schema (format 相关)
                if "response_mime_type" in generation_config:
                    config_kwargs["response_mime_type"] = generation_config["response_mime_type"]
                if "response_schema" in generation_config:
                    config_kwargs["response_schema"] = generation_config["response_schema"]

                # 构建完整的消息列表（system_instruction + history + current）
                contents = []

                # 如果有 system_instruction，添加到配置中
                if system_instruction:
                    config_kwargs["system_instruction"] = system_instruction

                # 添加历史消息
                for msg in history:
                    # 转换 parts：提取 File 对象（如果有）
                    converted_parts = self._convert_parts_for_new_sdk(msg["parts"])
                    # 使用 genai_types.Content 构建消息
                    contents.append(genai_types.Content(
                        role=msg["role"],
                        parts=converted_parts
                    ))

                # 添加当前消息（支持文本或多模态）
                if isinstance(current_message, str):
                    # 纯文本消息：包装为 Part 对象
                    parts = [genai_types.Part(text=current_message)]
                else:
                    # 多模态消息：转换 parts，提取 File 对象
                    parts = self._convert_parts_for_new_sdk(current_message)

                # 使用 genai_types.Content 构建当前消息
                contents.append(genai_types.Content(
                    role="user",
                    parts=parts
                ))

                # 调用新 SDK
                gen_config = genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

                response = self._genai_client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=gen_config
                )

                return response

            except Exception as e:
                error_msg = f"{e.__class__.__name__}: {str(e)}"
                logger.error("API 错误 trace_id=%s: %s", trace_id, error_msg)
                raise LLMTransportError(error_msg) from e

        resp = _call()
        self._record_usage(resp, trace_id, payload["model"])
        return resp


class AsyncLLMClient(_BaseLLMClient):
    """异步 LLM 客户端（暂不实现）。"""
    def __init__(self, config: GeminiAPIConfig, recorder: Optional[UsageRecorder] = None, retry_config: Optional[RetryConfig] = None):
        super().__init__(config, recorder, retry_config)

    @classmethod
    def from_env(cls, recorder: Optional[UsageRecorder] = None, retry_config: Optional[RetryConfig] = None):
        return cls(GeminiAPIConfig.from_env(), recorder, retry_config)

    async def invoke_from_yaml(self, yaml_prompt: str, *, dry_run: bool = False, include_debug: bool = False) -> Union[str, Dict[str, Any]]:
        raise NotImplementedError("异步客户端暂不支持，请使用同步客户端 LLMClient")
