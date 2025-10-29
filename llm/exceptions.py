"""Common exception definitions."""


class LLMConfigError(RuntimeError):
    """Configuration error."""


class LLMValidationError(ValueError):
    """Input validation error."""


class LLMTransportError(RuntimeError):
    """Transport layer error."""


__all__ = [
    "LLMConfigError",
    "LLMValidationError",
    "LLMTransportError",
]
