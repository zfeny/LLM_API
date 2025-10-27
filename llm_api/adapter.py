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

        # 处理 think 参数（Gemini thinking_config）
        think_value = None

        for k, v in gen.items():
            if k == "model":
                continue
            if k == "think":
                # 记录 think 值，稍后处理
                think_value = v
                continue
            if k == "max_output_tokens":
                payload["max_tokens"] = v
            else:
                payload[k] = v

        # 如果有 think 参数，添加到 extra_body
        if think_value is not None:
            extra_body = payload.get("extra_body", {})
            # 确保 extra_body 内部结构存在（双层嵌套）
            if "extra_body" not in extra_body:
                extra_body["extra_body"] = {}
            if "google" not in extra_body["extra_body"]:
                extra_body["extra_body"]["google"] = {}

            # 添加 thinking_config
            thinking_config = {"thinkingBudget": think_value}
            # 当 think=-1 时，启用思考总结
            if think_value == -1:
                thinking_config["includeThoughts"] = True

            extra_body["extra_body"]["google"]["thinking_config"] = thinking_config
            payload["extra_body"] = extra_body

        fmt = FormatHandler.response_format(ics.format_config)
        if fmt:
            payload["response_format"] = fmt
        return payload
