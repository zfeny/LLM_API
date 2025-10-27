"""自定义异常类定义。"""


class OpenListError(Exception):
    """OpenList API 基础异常类。"""
    pass


class OpenListConfigError(OpenListError):
    """配置错误。"""
    pass


class OpenListAuthError(OpenListError):
    """认证失败。"""
    pass


class OpenListUploadError(OpenListError):
    """文件上传失败。"""
    pass


class OpenListAPIError(OpenListError):
    """API 调用失败。"""
    def __init__(self, message: str, status_code: int = None, response_data: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data
