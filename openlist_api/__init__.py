"""OpenList API 封装库。

用于上传图片到 OpenList 并获取分享直链。

使用示例:
    from openlist_api import OpenListClient
    from llm_api import load_env_file

    # 加载环境变量
    load_env_file()

    # 创建客户端
    client = OpenListClient.from_env()

    # 上传图片并获取直链
    image_url = client.upload_image("/path/to/image.jpg")
    print(image_url)
"""

import logging

# 配置日志
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

# 导出主要 API
from .client import OpenListClient
from .config import OpenListConfig
from .exceptions import (
    OpenListAPIError,
    OpenListAuthError,
    OpenListConfigError,
    OpenListError,
    OpenListUploadError,
)

__all__ = [
    # 客户端
    "OpenListClient",
    # 配置
    "OpenListConfig",
    # 异常
    "OpenListError",
    "OpenListConfigError",
    "OpenListAuthError",
    "OpenListUploadError",
    "OpenListAPIError",
]

__version__ = "0.1.0"
