"""OpenAI 适配器。"""
from __future__ import annotations
from typing import Any, Dict

from .exceptions import LLMConfigError
from .format import FormatHandler
from .models import ICSRequest


class OpenAIAdapter:
    """OpenAI 适配器。"""
    @staticmethod
    def to_chat(ics: ICSRequest) -> Dict[str, Any]:
        gen = ics.generation
        payload = {"model": gen.get("model"), "messages": [m.to_payload() for m in ics.messages]}
        if not payload["model"]:
            raise LLMConfigError("缺少模型参数")
        for k, v in gen.items():
            if k == "model":
                continue
            if k == "max_output_tokens":
                payload["max_tokens"] = v
            else:
                payload[k] = v
        fmt = FormatHandler.response_format(ics.format_config)
        if fmt:
            payload["response_format"] = fmt
        return payload
