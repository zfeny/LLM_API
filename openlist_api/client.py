"""OpenList 客户端实现。"""
from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

from .config import OpenListConfig
from .exceptions import (
    OpenListAPIError,
    OpenListAuthError,
    OpenListConfigError,
    OpenListUploadError,
)
from .utils import (
    generate_date_path,
    get_all_parent_dirs,
    validate_file_exists,
)

logger = logging.getLogger(__name__)


class OpenListClient:
    """OpenList API 客户端。"""

    def __init__(self, config: OpenListConfig):
        """
        初始化客户端。

        Args:
            config: OpenList 配置
        """
        self.config = config
        self._token: Optional[str] = None
        self._session = requests.Session()

    @classmethod
    def from_env(cls) -> OpenListClient:
        """从环境变量创建客户端。"""
        config = OpenListConfig.from_env()
        return cls(config)

    def _get_headers(self, with_auth: bool = False) -> dict:
        """
        获取请求头。

        Args:
            with_auth: 是否包含认证信息

        Returns:
            请求头字典
        """
        headers = {
            "Content-Type": "application/json",
        }

        if with_auth:
            if not self._token:
                raise OpenListAuthError("未登录，请先调用 login() 方法")
            headers["Authorization"] = self._token

        return headers

    def login(self) -> str:
        """
        登录获取 JWT token。

        Returns:
            JWT token

        Raises:
            OpenListAuthError: 登录失败
        """
        url = f"{self.config.url}/api/auth/login"

        payload = {
            "username": self.config.account,
            "password": self.config.password,
        }

        # 如果提供了 OTP 代码，添加到请求中
        if self.config.otp_code:
            payload["otp_code"] = self.config.otp_code

        try:
            response = self._session.post(url, json=payload)
            response.raise_for_status()

            data = response.json()

            # 检查响应状态
            if data.get("code") != 200:
                raise OpenListAuthError(
                    f"登录失败: {data.get('message', '未知错误')}"
                )

            # 提取 token
            token = data.get("data", {}).get("token")
            if not token:
                raise OpenListAuthError("响应中未找到 token")

            self._token = token
            logger.info("登录成功")

            return token

        except requests.RequestException as e:
            raise OpenListAuthError(f"登录请求失败: {str(e)}") from e

    def create_directory(self, path: str) -> None:
        """
        创建目录。

        Args:
            path: 目录路径

        Raises:
            OpenListAPIError: 创建目录失败
        """
        # 确保已登录
        if not self._token:
            self.login()

        url = f"{self.config.url}/api/fs/mkdir"

        payload = {"path": path}

        headers = self._get_headers(with_auth=True)

        try:
            response = self._session.post(url, json=payload, headers=headers)
            response.raise_for_status()

            data = response.json()

            # 检查响应状态
            if data.get("code") == 200:
                logger.debug(f"目录创建成功: {path}")
            elif data.get("code") == 400 and "请求有误" in data.get("message", ""):
                # 目录可能已存在，不视为错误
                logger.debug(f"目录可能已存在: {path}")
            else:
                raise OpenListAPIError(
                    f"创建目录失败: {data.get('message', '未知错误')}",
                    status_code=data.get("code"),
                    response_data=data,
                )

        except requests.RequestException as e:
            raise OpenListAPIError(f"创建目录请求失败: {str(e)}") from e

    def _ensure_directory_exists(self, file_path: str) -> None:
        """
        确保文件路径的所有父目录都存在。

        Args:
            file_path: 完整文件路径
        """
        parent_dirs = get_all_parent_dirs(file_path)

        for dir_path in parent_dirs:
            try:
                self.create_directory(dir_path)
            except OpenListAPIError as e:
                # 如果目录已存在，忽略错误
                if "已存在" in str(e) or "请求有误" in str(e):
                    continue
                raise

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """
        上传文件到 OpenList。

        Args:
            local_path: 本地文件路径
            remote_path: 远程文件路径

        Raises:
            OpenListUploadError: 上传失败
        """
        # 验证本地文件存在
        validate_file_exists(local_path)

        # 确保已登录
        if not self._token:
            self.login()

        # 确保远程目录存在
        self._ensure_directory_exists(remote_path)

        url = f"{self.config.url}/api/fs/put"

        # URL 编码远程路径
        encoded_path = quote(remote_path, safe='')

        # 设置请求头
        headers = {
            "Authorization": self._token,
            "Content-Type": "application/octet-stream",
            "File-Path": encoded_path,
            "As-Task": "false",
        }

        logger.debug(f"上传请求 URL: {url}")
        logger.debug(f"原始路径: {remote_path}")
        logger.debug(f"编码后路径: {encoded_path}")
        logger.debug(f"上传请求头: {headers}")

        try:
            # 以二进制流模式上传文件
            with open(local_path, "rb") as f:
                response = self._session.put(
                    url,
                    data=f,
                    headers=headers,
                )

                logger.debug(f"响应状态码: {response.status_code}")
                logger.debug(f"响应内容: {response.text}")

                response.raise_for_status()

                data = response.json()

                # 检查响应状态
                if data.get("code") != 200:
                    raise OpenListUploadError(
                        f"上传文件失败: {data.get('message', '未知错误')}",
                    )

                logger.info(f"文件上传成功: {remote_path}")

        except requests.RequestException as e:
            raise OpenListUploadError(f"上传文件请求失败: {str(e)}") from e
        except IOError as e:
            raise OpenListUploadError(f"读取本地文件失败: {str(e)}") from e

    def create_share(self, paths: list[str], password: Optional[str] = None,
                     expiration: Optional[str] = None) -> tuple[str, str]:
        """
        创建分享链接。

        Args:
            paths: 文件/文件夹路径列表
            password: 可选的分享密码
            expiration: 可选的过期时间

        Returns:
            (share_id, share_url) 元组

        Raises:
            OpenListAPIError: 创建分享失败
        """
        # 确保已登录
        if not self._token:
            self.login()

        url = f"{self.config.url}/api/share/create"

        # 路径需要双斜杠开头
        formatted_paths = [f"/{path}" if not path.startswith("//") else path for path in paths]

        # 按照 Alist 的要求构建请求体
        payload = {
            "files": formatted_paths,
            "expires": expiration,
            "pwd": password or "",
            "extract_folder": "",
            "header": "",
            "max_accessed": 0,
            "order_by": "",
            "order_direction": "",
            "readme": "",
            "remark": ""
        }

        headers = self._get_headers(with_auth=True)

        logger.debug(f"创建分享请求 URL: {url}")
        logger.debug(f"创建分享请求体: {payload}")
        logger.debug(f"创建分享请求头: {headers}")

        try:
            response = self._session.post(url, json=payload, headers=headers)

            logger.debug(f"创建分享响应状态码: {response.status_code}")
            logger.debug(f"创建分享响应内容: {response.text}")

            response.raise_for_status()

            data = response.json()

            # 检查响应状态
            if data.get("code") != 200:
                raise OpenListAPIError(
                    f"创建分享失败: {data.get('message', '未知错误')}",
                    status_code=data.get("code"),
                    response_data=data,
                )

            # 提取分享信息
            share_data = data.get("data", {})
            share_id = share_data.get("id")

            if not share_id:
                raise OpenListAPIError("响应中未找到分享 ID")

            # 手动构建分享链接
            share_url = f"{self.config.url}/sd/{share_id}"

            logger.info(f"分享创建成功: {share_url}")

            return share_id, share_url

        except requests.RequestException as e:
            raise OpenListAPIError(f"创建分享请求失败: {str(e)}") from e

    def upload_image(self, local_path: str) -> str:
        """
        上传图片并返回分享直链。

        这是主要的公共方法，封装了完整的上传流程：
        1. 验证本地文件
        2. 生成远程路径（按年/月组织）
        3. 登录
        4. 创建必要的目录
        5. 上传文件
        6. 创建分享链接
        7. 返回直链

        Args:
            local_path: 图片本地路径

        Returns:
            图片分享直链

        Raises:
            FileNotFoundError: 文件不存在
            OpenListError: 上传或分享失败
        """
        # 验证文件存在
        validate_file_exists(local_path)

        # 获取文件名
        filename = os.path.basename(local_path)

        # 生成远程路径（按年月组织）
        remote_path = generate_date_path(
            self.config.temp_upload_path,
            filename
        )

        logger.info(f"开始上传: {local_path} -> {remote_path}")

        # 上传文件
        self.upload_file(local_path, remote_path)

        # 等待一下让 Alist 索引文件
        time.sleep(0.5)

        # 创建分享链接
        _, share_url = self.create_share([remote_path])

        logger.info(f"图片上传完成，分享链接: {share_url}")

        return share_url

    def remove_files(self, dir_path: str, file_names: list[str]) -> None:
        """
        删除指定目录中的文件或文件夹。

        Args:
            dir_path: 目录路径
            file_names: 要删除的文件/文件夹名称列表

        Raises:
            OpenListAPIError: 删除失败
        """
        # 确保已登录
        if not self._token:
            self.login()

        url = f"{self.config.url}/api/fs/remove"

        payload = {
            "dir": dir_path,
            "names": file_names
        }

        headers = self._get_headers(with_auth=True)

        try:
            response = self._session.post(url, json=payload, headers=headers)
            response.raise_for_status()

            data = response.json()

            # 检查响应状态
            if data.get("code") != 200:
                raise OpenListAPIError(
                    f"删除文件失败: {data.get('message', '未知错误')}",
                    status_code=data.get("code"),
                    response_data=data,
                )

            logger.info(f"文件删除成功: {dir_path} - {file_names}")

        except requests.RequestException as e:
            raise OpenListAPIError(f"删除文件请求失败: {str(e)}") from e

    def remove_file(self, file_path: str) -> None:
        """
        删除单个文件（便捷方法）。

        Args:
            file_path: 完整的文件路径，如 /temp_images/2025/10/image.jpg

        Raises:
            OpenListAPIError: 删除失败
        """
        # 从完整路径中提取目录和文件名
        path = Path(file_path)
        dir_path = str(path.parent)
        file_name = path.name

        logger.info(f"删除文件: {file_path}")

        # 调用基础删除方法
        self.remove_files(dir_path, [file_name])
