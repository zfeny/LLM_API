"""Shared configuration helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


def load_env_file(path: str | os.PathLike[str] = ".env") -> None:
    """Load a .env file into the environment if it exists."""
    env_path = Path(path)
    if not env_path.exists():
        return
    load_dotenv(dotenv_path=env_path, override=False)


__all__ = ["RetryConfig", "load_env_file"]
