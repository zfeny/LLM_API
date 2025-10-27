"""极简 YAML→ICS→OpenAI 封装。"""
import logging

# 配置日志
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

# 主要公共 API
from .client import AsyncLLMClient, LLMClient
from .config import LLMAPIConfig, RetryConfig, load_env_file
from .exceptions import LLMConfigError, LLMTransportError, LLMValidationError
from .recorder import UsageRecorder

# 次要导出（可选）
from .models import ICSMessage, ICSRequest
from .parser import YAMLRequestParser

__all__ = [
    # 客户端
    "LLMClient",
    "AsyncLLMClient",
    # 配置
    "LLMAPIConfig",
    "RetryConfig",
    "load_env_file",
    # 异常
    "LLMConfigError",
    "LLMValidationError",
    "LLMTransportError",
    # 记录器
    "UsageRecorder",
    # 可选导出
    "ICSMessage",
    "ICSRequest",
    "YAMLRequestParser",
]
