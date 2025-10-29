"""Internal utility helpers."""
from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
_MISSING = object()


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute or dict value while preserving falsy values."""
    if obj is None:
        return default
    attr_val = getattr(obj, key, _MISSING)
    if attr_val is not _MISSING:
        return attr_val
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _retry(config, is_async: bool = False):
    """Retry decorator supporting sync and async callables."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                import asyncio

                last_exc = None
                for attempt in range(config.max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        last_exc = exc
                        if attempt >= config.max_retries:
                            break
                        delay = min(config.initial_delay * (config.exponential_base ** attempt), config.max_delay)
                        if config.jitter:
                            delay *= 0.5 + random.random()
                        logger.warning(
                            "请求失败 (%d/%d)，%.1fs 后重试: %s",
                            attempt + 1,
                            config.max_retries + 1,
                            delay,
                            exc,
                        )
                        await asyncio.sleep(delay)
                raise last_exc

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exc = None
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt >= config.max_retries:
                        break
                    delay = min(config.initial_delay * (config.exponential_base ** attempt), config.max_delay)
                    if config.jitter:
                        delay *= 0.5 + random.random()
                    logger.warning(
                        "请求失败 (%d/%d)，%.1fs 后重试: %s",
                        attempt + 1,
                        config.max_retries + 1,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
            raise last_exc

        return sync_wrapper

    return decorator


__all__ = ["_get", "_retry"]
