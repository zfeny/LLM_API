"""配置类定义。"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .exceptions import LLMConfigError


@dataclass
class RetryConfig:
    """重试配置。"""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass(slots=True)
class LLMAPIConfig:
    """运行配置。"""
    api_key: str
    base_url: str
    default_model: Optional[str]
    request_timeout: Optional[float]
    organization: Optional[str]

    @classmethod
    def from_env(cls):
        def req(k):
            v = os.environ.get(k)
            if not v or not v.strip():
                raise LLMConfigError(f"缺少环境变量: {k}")
            return v.strip()
        timeout_raw = os.environ.get("LLM_TIMEOUT")
        timeout = float(timeout_raw.strip()) if timeout_raw and timeout_raw.strip() else None
        return cls(api_key=req("LLM_API_KEY"), base_url=req("LLM_API_BASE"),
                   default_model=os.environ.get("LLM_MODEL"), request_timeout=timeout,
                   organization=os.environ.get("LLM_ORG"))


def load_env_file(path: str | os.PathLike[str] = ".env") -> None:
    """
    加载 .env 文件到环境变量。

    使用 python-dotenv 库来解析 .env 文件。
    如果文件不存在，则静默返回。

    Args:
        path: .env 文件路径，默认为当前目录下的 .env
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    load_dotenv(dotenv_path=env_path, override=False)
