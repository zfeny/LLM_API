"""LLM 客户端。"""
from __future__ import annotations
import logging
import time
import uuid
from typing import Any, Dict, Optional, Union

from llm.config import RetryConfig
from llm.exceptions import LLMConfigError, LLMTransportError, LLMValidationError
from llm.parser import YAMLRequestParser
from llm.recorder import UsageRecorder
from llm.utils import _get, _retry

from .adapter import OpenAIAdapter
from .builder import ICSBuilder
from .config import LLMAPIConfig
from .format import FormatHandler

try:
    from openai import AsyncOpenAI, OpenAI
    import openai
except ImportError as e:
    raise ImportError("需要 openai SDK: pip install openai") from e

logger = logging.getLogger(__name__)


class _BaseLLMClient:
    """LLM 客户端基类（提取公共方法）。"""
    # 异常类作为类属性，方便外部通过 LLMClient.ConfigError 访问
    ConfigError = LLMConfigError
    ValidationError = LLMValidationError
    TransportError = LLMTransportError

    def __init__(self, config: LLMAPIConfig, recorder: Optional[UsageRecorder], retry_config: Optional[RetryConfig]):
        self._config = config
        self._builder = ICSBuilder(config)
        self._recorder = recorder or UsageRecorder(env_var="LLM_USAGE_DB", default_filename="usage_log.db")
        self._retry_config = retry_config or RetryConfig()

    def _extract_result(self, resp: Any, fmt_cfg: Optional[Dict[str, Any]]) -> Any:
        choice = None
        try:
            choice = resp.choices[0]
        except:
            choices = _get(resp, "choices")
            if isinstance(choices, list) and choices:
                choice = choices[0]
        if not choice:
            return FormatHandler.process(resp.model_dump() if hasattr(resp, "model_dump") else resp, fmt_cfg)

        msg = _get(choice, "message")
        if not msg:
            return FormatHandler.process(choice, fmt_cfg)

        parsed = _get(msg, "parsed")
        if parsed is not None:
            return FormatHandler.process(parsed, fmt_cfg)

        content = _get(msg, "content")

        # 检查是否有思考总结（Gemini thinking mode）
        # 情况 1: content 是列表（parts）
        if isinstance(content, list):
            # 提取 thought 和 text parts
            thoughts = []
            texts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "thought":
                        thought_text = part.get("thought", "")
                        if thought_text:
                            thoughts.append(thought_text)
                    elif part.get("type") == "text":
                        text_content = part.get("text", "")
                        if text_content:
                            texts.append(text_content)

            # 如果有思考总结，格式化为易读文本
            if thoughts:
                thought_text = "\n\n".join(thoughts)
                answer = "".join(texts)
                formatted_output = f"<LLM_THINKING>\n{thought_text}\n</LLM_THINKING>\n\n{answer}"
                return formatted_output

            # 没有思考总结，拼接所有文本
            content = "".join(texts)

        # 情况 2: content 是字符串，可能包含 <thought> 标签
        elif isinstance(content, str) and "<thought>" in content:
            import re
            # 提取 <thought>...</thought> 内容
            thought_pattern = r'<thought>(.*?)</thought>'
            thought_matches = re.findall(thought_pattern, content, re.DOTALL)

            if thought_matches:
                # 移除所有 <thought> 标签，剩余内容为答案
                answer = re.sub(thought_pattern, '', content, flags=re.DOTALL).strip()

                # 格式化为易读的文本格式
                thought_text = "\n\n".join(m.strip() for m in thought_matches)
                formatted_output = f"<LLM_THINKING>\n{thought_text}\n</LLM_THINKING>\n\n{answer}"
                return formatted_output

        return FormatHandler.process(content, fmt_cfg)

    def _record_usage(self, resp: Any, trace_id: Optional[str]):
        usage_obj = _get(resp, "usage")
        usage_dict = None
        if usage_obj:
            if hasattr(usage_obj, "model_dump"):
                usage_dict = usage_obj.model_dump()
            elif isinstance(usage_obj, dict):
                usage_dict = usage_obj
        self._recorder.record(model=_get(resp, "model"), request_id=_get(resp, "id"), trace_id=trace_id, usage=usage_dict)


class LLMClient(_BaseLLMClient):
    """同步 LLM 客户端。"""
    def __init__(self, config: LLMAPIConfig, recorder: Optional[UsageRecorder] = None, retry_config: Optional[RetryConfig] = None):
        super().__init__(config, recorder, retry_config)
        self._openai_client = self._create_openai_client()

    @classmethod
    def from_env(cls, recorder: Optional[UsageRecorder] = None, retry_config: Optional[RetryConfig] = None):
        return cls(LLMAPIConfig.from_env(), recorder, retry_config)

    def invoke_from_yaml(self, yaml_prompt: str, *, dry_run: bool = False, include_debug: bool = False) -> Union[Any, Dict[str, Any]]:
        start = time.time()
        trace_id = str(uuid.uuid4())
        logger.info("处理请求 trace_id=%s dry_run=%s", trace_id, dry_run)

        parsed = YAMLRequestParser.parse(yaml_prompt)
        ics = self._builder.build(parsed)
        if "trace_id" not in ics.meta:
            ics.meta["trace_id"] = trace_id
        openai_payload = OpenAIAdapter.to_chat(ics)

        if dry_run:
            return {"ics_request": ics.to_payload(), "openai_request": openai_payload}

        resp = self._send(openai_payload, ics.meta.get("trace_id"))
        logger.info("完成 trace_id=%s 耗时=%.2fs", trace_id, time.time() - start)

        result = self._extract_result(resp, ics.format_config)
        if not include_debug:
            return result
        return {"result": result, "ics_request": ics.to_payload(), "openai_request": openai_payload, "response": resp.model_dump() if hasattr(resp, "model_dump") else resp}

    def _create_openai_client(self):
        kwargs = {"api_key": self._config.api_key, "base_url": self._config.base_url}
        if self._config.organization:
            kwargs["organization"] = self._config.organization
        return OpenAI(**kwargs)

    def _send(self, payload: Dict[str, Any], trace_id: Optional[str]):
        @_retry(self._retry_config, is_async=False)
        def _call():
            try:
                return self._openai_client.chat.completions.create(**payload)
            except openai.APIError as e:
                # 捕获所有 OpenAI API 错误并转换为 LLMTransportError
                error_msg = f"{e.__class__.__name__}: {str(e)}"
                logger.error("API 错误 trace_id=%s: %s", trace_id, error_msg)
                raise LLMTransportError(error_msg) from e
            except Exception as e:
                # 捕获其他未预期的错误
                error_msg = f"未预期的错误 {e.__class__.__name__}: {str(e)}"
                logger.error("传输错误 trace_id=%s: %s", trace_id, error_msg)
                raise LLMTransportError(error_msg) from e

        resp = _call()
        self._record_usage(resp, trace_id)
        return resp


class AsyncLLMClient(_BaseLLMClient):
    """异步 LLM 客户端。"""
    def __init__(self, config: LLMAPIConfig, recorder: Optional[UsageRecorder] = None, retry_config: Optional[RetryConfig] = None):
        super().__init__(config, recorder, retry_config)
        self._openai_client = self._create_openai_client()

    @classmethod
    def from_env(cls, recorder: Optional[UsageRecorder] = None, retry_config: Optional[RetryConfig] = None):
        return cls(LLMAPIConfig.from_env(), recorder, retry_config)

    async def invoke_from_yaml(self, yaml_prompt: str, *, dry_run: bool = False, include_debug: bool = False) -> Union[Any, Dict[str, Any]]:
        start = time.time()
        trace_id = str(uuid.uuid4())
        logger.info("异步处理 trace_id=%s dry_run=%s", trace_id, dry_run)

        parsed = YAMLRequestParser.parse(yaml_prompt)
        ics = self._builder.build(parsed)
        if "trace_id" not in ics.meta:
            ics.meta["trace_id"] = trace_id
        openai_payload = OpenAIAdapter.to_chat(ics)

        if dry_run:
            return {"ics_request": ics.to_payload(), "openai_request": openai_payload}

        resp = await self._send(openai_payload, ics.meta.get("trace_id"))
        logger.info("异步完成 trace_id=%s 耗时=%.2fs", trace_id, time.time() - start)

        result = self._extract_result(resp, ics.format_config)
        if not include_debug:
            return result
        return {"result": result, "ics_request": ics.to_payload(), "openai_request": openai_payload, "response": resp.model_dump() if hasattr(resp, "model_dump") else resp}

    def _create_openai_client(self):
        kwargs = {"api_key": self._config.api_key, "base_url": self._config.base_url}
        if self._config.organization:
            kwargs["organization"] = self._config.organization
        return AsyncOpenAI(**kwargs)

    async def _send(self, payload: Dict[str, Any], trace_id: Optional[str]):
        @_retry(self._retry_config, is_async=True)
        async def _call():
            try:
                return await self._openai_client.chat.completions.create(**payload)
            except openai.APIError as e:
                # 捕获所有 OpenAI API 错误并转换为 LLMTransportError
                error_msg = f"{e.__class__.__name__}: {str(e)}"
                logger.error("API 错误 trace_id=%s: %s", trace_id, error_msg)
                raise LLMTransportError(error_msg) from e
            except Exception as e:
                # 捕获其他未预期的错误
                error_msg = f"未预期的错误 {e.__class__.__name__}: {str(e)}"
                logger.error("传输错误 trace_id=%s: %s", trace_id, error_msg)
                raise LLMTransportError(error_msg) from e

        resp = await _call()
        self._record_usage(resp, trace_id)
        return resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self._openai_client.close()
