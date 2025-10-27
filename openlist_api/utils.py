"""工具函数。"""
from datetime import datetime
from pathlib import Path
import os


def generate_date_path(base_path: str, filename: str) -> str:
    """
    根据当前日期生成文件路径。

    格式: {base_path}/{年}/{月}/{filename}
    例如: /temp_images/2025/10/image.jpg

    Args:
        base_path: 基础路径
        filename: 文件名

    Returns:
        完整的文件路径
    """
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")

    # 确保路径以 / 开头
    if not base_path.startswith("/"):
        base_path = "/" + base_path

    # 移除路径末尾的 /
    base_path = base_path.rstrip("/")

    return f"{base_path}/{year}/{month}/{filename}"


def get_directory_from_path(file_path: str) -> str:
    """
    从完整文件路径中提取目录部分。

    Args:
        file_path: 完整文件路径，如 /temp_images/2025/10/image.jpg

    Returns:
        目录路径，如 /temp_images/2025/10
    """
    return str(Path(file_path).parent)


def get_all_parent_dirs(file_path: str) -> list[str]:
    """
    获取文件路径的所有父目录（从根到叶）。

    Args:
        file_path: 完整文件路径，如 /temp_images/2025/10/image.jpg

    Returns:
        按层级排序的目录列表，如 ['/temp_images', '/temp_images/2025', '/temp_images/2025/10']
    """
    path = Path(file_path)
    parents = []

    # 获取文件所在目录
    current = path.parent

    # 收集所有父目录
    while str(current) != '/' and str(current) != '.':
        parents.append(str(current))
        current = current.parent

    # 反转列表，使其从根到叶排序
    parents.reverse()

    return parents


def validate_file_exists(file_path: str) -> None:
    """
    验证文件是否存在。

    Args:
        file_path: 文件路径

    Raises:
        FileNotFoundError: 如果文件不存在
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"路径不是文件: {file_path}")
