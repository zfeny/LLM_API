"""预设加载器。"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List

from .exceptions import LLMValidationError
from .models import MessageEntry

try:
    import yaml
except ImportError as e:
    raise ImportError("需要 PyYAML: pip install pyyaml") from e

logger = logging.getLogger(__name__)


def load_preset(preset_name: str) -> List[MessageEntry]:
    """
    从preset文件夹加载指定预设。

    Args:
        preset_name: 预设名称（不含.yaml后缀）

    Returns:
        预设消息列表（每个MessageEntry的source字段标记为preset名称）

    Raises:
        LLMValidationError: 预设文件不存在或格式错误
    """
    # 构建预设文件路径
    preset_dir = Path(__file__).parent / "preset"
    preset_file = preset_dir / f"{preset_name}.yaml"

    # 检查文件是否存在
    if not preset_file.exists():
        available_presets = [f.stem for f in preset_dir.glob("*.yaml")] if preset_dir.exists() else []
        if available_presets:
            hint = f"可用预设: {', '.join(sorted(available_presets))}"
        else:
            hint = "preset文件夹中没有可用预设"
        raise LLMValidationError(f"预设 '{preset_name}' 不存在。{hint}")

    # 读取并解析YAML文件
    try:
        with preset_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise LLMValidationError(f"预设 '{preset_name}' YAML解析失败: {e}") from e
    except Exception as e:
        raise LLMValidationError(f"预设 '{preset_name}' 读取失败: {e}") from e

    # 验证格式：必须是列表
    if not isinstance(data, list):
        raise LLMValidationError(f"预设 '{preset_name}' 格式错误：必须是消息列表")

    # 解析消息列表
    entries: List[MessageEntry] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise LLMValidationError(f"预设 '{preset_name}' 第{idx+1}项必须是字典")

        # 提取角色和内容
        if len(item) != 1:
            raise LLMValidationError(f"预设 '{preset_name}' 第{idx+1}项必须只包含一个角色键")

        role, content = next(iter(item.items()))

        # 验证角色
        if role not in ("system", "user", "assistant"):
            raise LLMValidationError(f"预设 '{preset_name}' 第{idx+1}项角色必须是 system/user/assistant")

        # 验证内容
        if not isinstance(content, str):
            raise LLMValidationError(f"预设 '{preset_name}' 第{idx+1}项内容必须是字符串")

        content = content.strip()

        # system消息允许为空，user/assistant不允许
        if role in ("user", "assistant") and not content:
            raise LLMValidationError(f"预设 '{preset_name}' 第{idx+1}项 {role} 消息不能为空")

        # 标记来源为preset名称
        entries.append(MessageEntry(role=role, content=content, source=f"preset:{preset_name}"))

    logger.debug(f"成功加载预设 '{preset_name}'，包含 {len(entries)} 条消息")
    return entries


def get_preset_raw_content(preset_name: str) -> str:
    """
    获取preset文件的原始YAML内容。

    Args:
        preset_name: 预设名称（不含.yaml后缀）

    Returns:
        原始YAML文件内容字符串

    Raises:
        LLMValidationError: 预设文件不存在
    """
    preset_dir = Path(__file__).parent / "preset"
    preset_file = preset_dir / f"{preset_name}.yaml"

    if not preset_file.exists():
        raise LLMValidationError(f"预设 '{preset_name}' 不存在")

    try:
        with preset_file.open("r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise LLMValidationError(f"预设 '{preset_name}' 读取失败: {e}") from e


def get_preset_system_content(preset_name: str) -> str:
    """
    获取preset中所有system消息的合并内容（只提取内容，不包含"- system:"前缀）。

    Args:
        preset_name: 预设名称（不含.yaml后缀）

    Returns:
        合并后的system消息内容（用\\n\\n连接多条system消息）

    Raises:
        LLMValidationError: 预设文件不存在或读取失败
    """
    entries = load_preset(preset_name)

    # 提取所有system角色的内容
    system_contents = [entry.content for entry in entries if entry.role == "system" and entry.content]

    if not system_contents:
        return ""

    # 用双换行符连接多条system消息
    return "\n\n".join(system_contents)
