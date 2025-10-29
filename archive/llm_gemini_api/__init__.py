"""极简 YAML→ICS→Gemini 封装（Gemini 原生 SDK）。"""
import logging

from gemini import (  # type: ignore[F401]
    AsyncLLMClient,
    GeminiAPIConfig,
    ICSMessage,
    ICSRequest,
    LLMClient,
    LLMConfigError,
    LLMTransportError,
    LLMValidationError,
    RetryConfig,
    UsageRecorder,
    YAMLRequestParser,
    batch_convert,
    load_env_file,
    load_preset,
    convert_tavern_to_preset,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def configure_logging(
    *,
    level: int | str = "INFO",
    httpx_level: int | str = "WARNING",
    google_level: int | str = "WARNING",
) -> None:
    """Backward-compatible logging configuration helper."""
    logging.getLogger("gemini").setLevel(level)
    logging.getLogger("llm_gemini_api").setLevel(level)
    logging.getLogger("httpx").setLevel(httpx_level)
    logging.getLogger("google_genai").setLevel(google_level)


__all__ = [
    "LLMClient",
    "AsyncLLMClient",
    "GeminiAPIConfig",
    "RetryConfig",
    "load_env_file",
    "LLMConfigError",
    "LLMValidationError",
    "LLMTransportError",
    "UsageRecorder",
    "configure_logging",
    "ICSMessage",
    "ICSRequest",
    "YAMLRequestParser",
    "load_preset",
    "convert_tavern_to_preset",
    "batch_convert",
]
