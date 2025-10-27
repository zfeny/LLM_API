"""LLM 客户端。"""
from __future__ import annotations
import logging
import time
import uuid
from typing import Any, Dict, Optional, Union

from ._utils import _get, _retry
from .adapter import OpenAIAdapter
from .builder import ICSBuilder
from .config import LLMAPIConfig, RetryConfig
from .exceptions import LLMConfigError, LLMTransportError, LLMValidationError
from .format import FormatHandler
from .parser import YAMLRequestParser
from .recorder import UsageRecorder

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
        self._recorder = recorder or UsageRecorder()
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
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
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
        if self._config.request_timeout is not None:
            kwargs["timeout"] = self._config.request_timeout
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
        if self._config.request_timeout is not None:
            kwargs["timeout"] = self._config.request_timeout
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
