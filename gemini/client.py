"""LLM 客户端。"""
from __future__ import annotations
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from llm.config import RetryConfig
from llm.exceptions import LLMConfigError, LLMTransportError, LLMValidationError
from llm.parser import YAMLRequestParser
from llm.recorder import UsageRecorder
from llm.utils import _retry

from .adapter import GeminiAdapter
from .builder import ICSBuilder
from .config import GeminiAPIConfig
from .file_utils import GeminiFileUploader
from .format import GeminiFormatHandler

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
        self._recorder = recorder or UsageRecorder(
            env_var="GEMINI_USAGE_DB",
            default_filename="gemini_usage_log.db",
            supports_thoughts=True,
        )
        self._retry_config = retry_config or RetryConfig()
        # 配置新版 SDK 客户端
        self._genai_client = genai.Client(api_key=config.api_key)
        # 初始化文件上传器（用于多模态功能）
        self._file_uploader = GeminiFileUploader(self._genai_client)

    def _extract_result(
        self,
        resp: Any,
        format_config: Optional[Dict[str, Any]] = None,
        *,
        inline_citations: bool = False,
    ) -> Any:
        """从 Gemini 响应中提取并处理结果（支持文本和图片）。"""
        try:
            if not hasattr(resp, "candidates") or not resp.candidates:
                raise LLMTransportError("响应中没有候选结果")

            candidate = resp.candidates[0]
            if not hasattr(candidate, "content") or not hasattr(candidate.content, "parts"):
                raise LLMTransportError("响应结构异常")

            parts = candidate.content.parts

            thoughts: List[str] = []
            answer_parts_info: List[Dict[str, Any]] = []
            image_parts: List[Any] = []

            for idx, part in enumerate(parts):
                inline_data = getattr(part, "inline_data", None)
                if inline_data is not None:
                    image_parts.append(part)
                    continue

                text_value = getattr(part, "text", None)
                if getattr(part, "thought", False):
                    if text_value:
                        thoughts.append(text_value)
                    continue

                if text_value:
                    answer_parts_info.append({"text": text_value, "part_index": idx})

            fmt_type = None
            if format_config and isinstance(format_config, dict):
                fmt_type = format_config.get("type")
            if inline_citations and fmt_type not in ("json", "json_schema"):
                self._apply_grounding_citations_to_parts(answer_parts_info, candidate)

            answer_parts = [item["text"] for item in answer_parts_info]

            if image_parts:
                result = self._extract_image_result(image_parts, answer_parts, thoughts, format_config)
                return self._append_url_metadata(result, candidate) if inline_citations else result

            if thoughts:
                thought_text = "\n\n".join(thoughts)
                answer_text = "".join(answer_parts)
                processed_answer = GeminiFormatHandler.process_response(answer_text, format_config)
                if isinstance(processed_answer, str):
                    processed_answer = f"<GEMINI_THINKING>\n{thought_text}\n</GEMINI_THINKING>\n\n{processed_answer}"
                    if inline_citations:
                        return self._append_url_metadata(processed_answer, candidate)
                    return processed_answer
                logger.info("检测到思考内容（%d 字符），但答案为非文本格式，仅返回答案", len(thought_text))
                return self._append_url_metadata(processed_answer, candidate) if inline_citations else processed_answer

            raw_text = "".join(answer_parts) if answer_parts else resp.text
            processed = GeminiFormatHandler.process_response(raw_text, format_config)
            return self._append_url_metadata(processed, candidate) if inline_citations else processed
        except Exception as e:
            logger.error("提取响应结果失败: %s", e)
            raise LLMTransportError(f"提取响应结果失败: {e}") from e

    def _extract_image_result(
        self,
        image_parts: list,
        texts: list[str],
        thoughts: list[str],
        format_config: Optional[Dict[str, Any]],
    ) -> Any:
        """提取图片响应，保存并上传到OpenList。"""
        from datetime import datetime

        images = []

        # 分离文本、思考和图片
        for part in image_parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is None:
                continue
            images.append({
                "mime_type": getattr(inline_data, "mime_type", None),
                "data": getattr(inline_data, "data", None)
            })

        # 处理图片：保存并上传
        image_results = []
        for img in images:
            try:
                # 生成时间戳文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                # 保存到本地
                local_path = self._save_generated_image(img["data"], timestamp)
                logger.info("图片已保存到本地: %s", local_path)

                # 根据配置决定是否上传到OpenList
                online_url = None
                if self._config.image_upload_enabled:
                    online_url = self._upload_to_openlist(local_path)
                    logger.info("图片已上传到OpenList: %s", online_url)
                else:
                    logger.info("图片上传已禁用，仅保存本地文件")

                image_results.append({
                    "local_path": local_path,
                    "online_url": online_url,
                    "mime_type": img["mime_type"]
                })
            except Exception as e:
                logger.error("图片处理失败: %s", e)
                # 即使失败也继续处理其他图片
                image_results.append({
                    "local_path": None,
                    "online_url": None,
                    "error": str(e)
                })

        # 根据响应模式返回
        if texts and images:
            # both模式：同时返回文本和图片
            text_content = "\n".join(texts)

            # 如果有思考内容，添加到文本中
            if thoughts:
                thought_text = "\n\n".join(thoughts)
                text_content = f"<GEMINI_THINKING>\n{thought_text}\n</GEMINI_THINKING>\n\n{text_content}"

            # 处理文本格式
            processed_text = GeminiFormatHandler.process_response(text_content, format_config)

            return {
                "text": processed_text,
                "images": image_results
            }
        elif images:
            # image模式：仅返回图片
            return image_results[0] if len(image_results) == 1 else image_results
        else:
            # 回退到文本（不应该发生，但为了安全）
            return "\n".join(texts)

    def _append_url_metadata(self, result: Any, candidate: Any) -> Any:
        """将 url_metadata 附加到输出文本后。"""
        urls = []
        metadata_list = getattr(candidate, "url_context_metadata", None)
        if metadata_list:
            for meta in metadata_list:
                url_meta_list = getattr(meta, "url_metadata", None)
                if not url_meta_list:
                    continue
                for url_meta in url_meta_list:
                    retrieved = getattr(url_meta, "retrieved_url", None)
                    if retrieved:
                        urls.append(str(retrieved))

        if not urls:
            return result

        if isinstance(result, dict):
            result = dict(result)
            result["url_metadata"] = urls
            return result

        # 对于字符串或其他简单类型，保持原值，引用信息通过 grounding 插入呈现
        return result

    def _apply_grounding_citations_to_parts(self, parts_info: List[Dict[str, Any]], candidate: Any) -> None:
        """在文本片段中插入内嵌引用。"""
        if not parts_info:
            return

        metadata = getattr(candidate, "grounding_metadata", None)
        if metadata is None:
            return

        supports = getattr(metadata, "grounding_supports", None)
        chunks = getattr(metadata, "grounding_chunks", None)
        if not supports or not chunks:
            return

        supports_by_part: Dict[int, List[Any]] = {}
        for support in supports:
            segment = getattr(support, "segment", None)
            if segment is None:
                continue
            part_index = getattr(segment, "part_index", None)
            if part_index is None:
                part_index = 0
            end_index = getattr(segment, "end_index", None)
            if end_index is None:
                continue
            supports_by_part.setdefault(part_index, []).append(support)

        if not supports_by_part:
            return

        parts_map = {info["part_index"]: info for info in parts_info if isinstance(info.get("text"), str)}
        if not parts_map:
            return

        for part_index, part_supports in supports_by_part.items():
            part_entry = parts_map.get(part_index)
            if not part_entry:
                continue

            original_text = part_entry["text"]
            if not isinstance(original_text, str) or not original_text:
                continue

            sorted_supports = sorted(
                part_supports,
                key=lambda s: getattr(getattr(s, "segment", None), "end_index", 0) or 0,
                reverse=True,
            )

            text = original_text
            for support in sorted_supports:
                segment = getattr(support, "segment", None)
                if segment is None:
                    continue
                end_index = getattr(segment, "end_index", None)
                if end_index is None:
                    continue
                if end_index > len(text):
                    end_index = len(text)
                chunk_indices = getattr(support, "grounding_chunk_indices", None) or []
                citation_links = []
                for idx in chunk_indices:
                    if 0 <= idx < len(chunks):
                        uri = self._extract_chunk_uri(chunks[idx])
                        if uri:
                            citation_links.append(f"[{idx + 1}]({uri})")
                if not citation_links:
                    continue
                citation_string = " ".join(citation_links)
                if not citation_string:
                    continue

                insertion = citation_string
                if end_index > 0 and not text[end_index - 1].isspace():
                    insertion = " " + insertion
                text = text[:end_index] + insertion + text[end_index:]

            part_entry["text"] = text

    @staticmethod
    def _extract_chunk_uri(chunk: Any) -> Optional[str]:
        """从 grounding chunk 中提取引用链接。"""
        if chunk is None:
            return None
        web = getattr(chunk, "web", None)
        if web and getattr(web, "uri", None):
            return web.uri
        retrieved = getattr(chunk, "retrieved_context", None)
        if retrieved and getattr(retrieved, "uri", None):
            return retrieved.uri
        return None

    def _save_generated_image(self, image_data: bytes, timestamp: str) -> str:
        """保存生成的图片到本地目录。

        目录结构: {GEMINI_IMAGE_OUTPUT|temp/output/image}/{年份}/{年月日}/<timestamp>.png

        Args:
            image_data: 图片的字节数据（bytes），直接来自 part.inline_data.data
            timestamp: 时间戳字符串，用于文件命名

        Returns:
            本地文件路径
        """
        import os

        # 创建输出目录
        base_dir = os.environ.get("GEMINI_IMAGE_OUTPUT") or "temp/output/image"
        year = timestamp[:4]
        date = timestamp[:8]
        output_dir = os.path.join(base_dir, year, date)
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        filename = f"{timestamp}.png"
        local_path = os.path.join(output_dir, filename)

        # 直接保存字节数据（不需要base64解码）
        with open(local_path, "wb") as f:
            f.write(image_data)

        return local_path

    def _upload_to_openlist(self, local_path: str) -> str:
        """上传图片到OpenList，自动按年/月组织目录。"""
        try:
            from openlist_api import OpenListClient

            # 使用环境变量中的配置创建客户端
            openlist_client = OpenListClient.from_env()

            # 上传并获取分享链接（自动按年/月组织目录）
            online_url = openlist_client.upload_image(local_path)

            return online_url
        except ImportError:
            logger.warning("openlist_api 模块未安装，跳过上传")
            return f"(未上传: {local_path})"
        except Exception as e:
            # 如果上传失败，返回本地路径
            logger.warning("OpenList上传失败: %s", e)
            return f"(上传失败: {local_path})"

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
        inline_citations = bool(gemini_payload.pop("inline_citations", False))

        if dry_run:
            payload_preview = dict(gemini_payload)
            if inline_citations:
                payload_preview["inline_citations"] = True
            return {"ics_request": ics.to_payload(), "gemini_payload": payload_preview}

        # 3. 调用 Gemini SDK
        resp = self._send(gemini_payload, ics.meta.get("trace_id"))
        logger.info("完成 trace_id=%s 耗时=%.2fs", trace_id, time.time() - start)

        # 4. 返回原始响应（如果请求）
        if raw_response:
            return resp

        # 5. 提取并处理结果
        result = self._extract_result(resp, ics.format_config, inline_citations=inline_citations)
        if not include_debug:
            return result
        return {
            "result": result,
            "ics_request": ics.to_payload(),
            "gemini_payload": {**gemini_payload, **({"inline_citations": True} if inline_citations else {})},
        }

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
                tools = payload.get("tools")

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

                # 处理图片生成配置
                if "response_modalities" in generation_config:
                    config_kwargs["response_modalities"] = generation_config["response_modalities"]

                if "image_config" in generation_config:
                    img_cfg = generation_config["image_config"]
                    config_kwargs["image_config"] = genai_types.ImageConfig(
                        aspect_ratio=img_cfg["aspect_ratio"]
                    )

                if tools:
                    config_kwargs["tools"] = tools

                # 构建完整的消息列表（system_instruction + history + current）
                contents = list(history)

                # 如果有 system_instruction，添加到配置中
                if system_instruction:
                    config_kwargs["system_instruction"] = system_instruction

                # 使用 Content 对象列表（history + 当前消息）
                contents.append(current_message)

                # 调用新 SDK
                gen_config = genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

                response = self._genai_client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=gen_config,
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
