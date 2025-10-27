"""配置类定义。"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

from .exceptions import OpenListConfigError


@dataclass(slots=True)
class OpenListConfig:
    """OpenList 运行配置。"""
    url: str
    account: str
    password: str
    temp_upload_path: str
    otp_code: Optional[str] = None

    @classmethod
    def from_env(cls) -> OpenListConfig:
        """从环境变量加载配置。"""
        def req(key: str) -> str:
            """获取必需的环境变量。"""
            value = os.environ.get(key)
            if not value or not value.strip():
                raise OpenListConfigError(f"缺少环境变量: {key}")
            return value.strip()

        # 获取必需配置
        url = req("OPENLIST_URL")
        account = req("OPENLIST_ACCOUNT")
        password = req("OPENLIST_PASSWORD")
        temp_upload_path = req("OPENLIST_TEMP_UPLOAD_PATH")

        # 获取可选配置
        otp_code = os.environ.get("OPENLIST_OTP_CODE")
        if otp_code:
            otp_code = otp_code.strip()

        # 确保 URL 不以 / 结尾
        url = url.rstrip("/")

        return cls(
            url=url,
            account=account,
            password=password,
            temp_upload_path=temp_upload_path,
            otp_code=otp_code
        )
