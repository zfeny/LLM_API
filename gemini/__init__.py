"""Gemini SDK integration built on shared llm toolkit."""
import logging

from llm.config import RetryConfig, load_env_file
from llm.exceptions import LLMConfigError, LLMTransportError, LLMValidationError
from llm.models import ICSMessage, ICSRequest
from llm.parser import YAMLRequestParser, register_preset_loader
from llm.recorder import UsageRecorder as _BaseUsageRecorder

from .client import AsyncLLMClient, LLMClient
from .config import GeminiAPIConfig
from .preset_loader import load_preset
from .tavern_converter import batch_convert, convert_tavern_to_preset

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

register_preset_loader(load_preset)


class UsageRecorder(_BaseUsageRecorder):
    """Gemini-specific UsageRecorder with thoughts token support enabled by default."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("env_var", "GEMINI_USAGE_DB")
        kwargs.setdefault("default_filename", "gemini_usage_log.db")
        kwargs.setdefault("supports_thoughts", True)
        super().__init__(*args, **kwargs)

__all__ = [
    "LLMClient",
    "AsyncLLMClient",
    "GeminiAPIConfig",
    "RetryConfig",
    "load_env_file",
    "UsageRecorder",
    "LLMConfigError",
    "LLMValidationError",
    "LLMTransportError",
    "ICSMessage",
    "ICSRequest",
    "YAMLRequestParser",
    "load_preset",
    "convert_tavern_to_preset",
    "batch_convert",
]
