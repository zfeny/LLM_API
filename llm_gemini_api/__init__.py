"""极简 YAML→ICS→Gemini 封装（Gemini 原生 SDK）。"""
import logging

# 默认不污染全局 logging 配置，将日志交给调用方处理
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def configure_logging(
    *,
    level: int | str = "INFO",
    httpx_level: int | str = "WARNING",
    google_level: int | str = "WARNING",
) -> None:
    """
    统一调整 llm_gemini_api 及相关依赖的日志级别。

    Args:
        level: llm_gemini_api.* 默认日志级别。
        httpx_level: httpx 请求日志级别。
        google_level: google-genai 相关日志级别。
    """
    logging.getLogger("llm_gemini_api").setLevel(level)
    logging.getLogger("httpx").setLevel(httpx_level)
    logging.getLogger("google_genai").setLevel(google_level)
    logging.getLogger("google.generativeai").setLevel(google_level)

# 主要公共 API
from .client import AsyncLLMClient, LLMClient
from .config import GeminiAPIConfig, RetryConfig, load_env_file
from .exceptions import LLMConfigError, LLMTransportError, LLMValidationError
from .recorder import UsageRecorder

# 次要导出（可选）
from .models import ICSMessage, ICSRequest
from .parser import YAMLRequestParser
from .preset_loader import load_preset
from .tavern_converter import convert_tavern_to_preset, batch_convert

__all__ = [
    # 客户端
    "LLMClient",
    "AsyncLLMClient",
    # 配置
    "GeminiAPIConfig",
    "RetryConfig",
    "load_env_file",
    # 异常
    "LLMConfigError",
    "LLMValidationError",
    "LLMTransportError",
    # 记录器
    "UsageRecorder",
    # 工具
    "configure_logging",
    # 可选导出
    "ICSMessage",
    "ICSRequest",
    "YAMLRequestParser",
    "load_preset",
    "convert_tavern_to_preset",
    "batch_convert",
]
