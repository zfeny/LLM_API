"""异常类定义。"""


class LLMConfigError(RuntimeError):
    """配置错误。"""


class LLMValidationError(ValueError):
    """输入验证错误。"""


class LLMTransportError(RuntimeError):
    """传输错误。"""
