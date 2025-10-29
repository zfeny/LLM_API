"""Shared LLM toolkit."""
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = [
    "logger",
]
