"""Preset loader with cycle detection."""
from __future__ import annotations

import contextlib
import logging
import threading
from pathlib import Path
from typing import List

from llm.exceptions import LLMValidationError
from llm.models import MessageEntry

try:
    import yaml
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("需要 PyYAML: pip install pyyaml") from exc

logger = logging.getLogger(__name__)

_loading_state = threading.local()

PRESET_MODULE_ROOT = Path(__file__).resolve().parents[1] / "llm" / "preset_module"
PRESET_DIR = PRESET_MODULE_ROOT / "preset"
GROUP_DIR = PRESET_MODULE_ROOT / "groups"

for _directory in (PRESET_DIR, GROUP_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def _loading_guard(name: str):
    stack = getattr(_loading_state, "stack", [])
    if name in stack:
        cycle = " -> ".join(stack + [name])
        raise LLMValidationError(f"检测到循环依赖: {cycle}")
    stack.append(name)
    _loading_state.stack = stack
    try:
        yield
    finally:
        stack.pop()
        if stack:
            _loading_state.stack = stack
        else:
            delattr(_loading_state, "stack")


def _find_preset_file(preset_name: str, preset_dir: Path) -> Path:
    if "/" in preset_name or "\\" in preset_name:
        preset_file = preset_dir / f"{preset_name}.yaml"
        if preset_file.exists():
            return preset_file
    else:
        preset_file = preset_dir / f"{preset_name}.yaml"
        if preset_file.exists():
            return preset_file
        for yaml_file in preset_dir.rglob(f"{preset_name}.yaml"):
            return yaml_file

    available_presets: List[str] = []
    if preset_dir.exists():
        for yaml_file in preset_dir.rglob("*.yaml"):
            rel_path = yaml_file.relative_to(preset_dir)
            preset_id = str(rel_path.with_suffix(""))
            available_presets.append(preset_id.replace("\\", "/"))

    if available_presets:
        hint = f"可用预设: {', '.join(sorted(available_presets)[:10])}"
        if len(available_presets) > 10:
            hint += f" (还有{len(available_presets) - 10}个...)"
    else:
        hint = "preset文件夹中没有可用预设"

    raise LLMValidationError(f"预设 '{preset_name}' 不存在。{hint}")


def load_preset(preset_name: str) -> List[MessageEntry]:
    with _loading_guard(preset_name):
        preset_file = _find_preset_file(preset_name, PRESET_DIR)

        try:
            with preset_file.open("r", encoding="utf-8") as stream:
                data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise LLMValidationError(f"预设 '{preset_name}' YAML解析失败: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMValidationError(f"预设 '{preset_name}' 读取失败: {exc}") from exc

        if not isinstance(data, list):
            raise LLMValidationError(f"预设 '{preset_name}' 格式错误：必须是消息列表")

        entries: List[MessageEntry] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise LLMValidationError(f"预设 '{preset_name}' 第{idx + 1}项必须是字典")

            if "preset" in item:
                preset_ref = item["preset"]
                if not isinstance(preset_ref, str):
                    raise LLMValidationError(f"预设 '{preset_name}' 第{idx + 1}项 preset 值必须是字符串")
                logger.debug("加载引用预设: %s", preset_ref)
                entries.extend(load_preset(preset_ref))
                continue

            if "preset-group" in item:
                group_name = item["preset-group"]
                if not isinstance(group_name, str):
                    raise LLMValidationError(f"预设 '{preset_name}' 第{idx + 1}项 preset-group 值必须是字符串")
                logger.debug("展开预设组: %s", group_name)
                entries.extend(load_preset_group(group_name))
                continue

            if len(item) != 1:
                raise LLMValidationError(f"预设 '{preset_name}' 第{idx + 1}项必须只包含一个角色键、preset键或preset-group键")

            role, content = next(iter(item.items()))
            if role not in ("system", "user", "assistant"):
                raise LLMValidationError(f"预设 '{preset_name}' 第{idx + 1}项角色必须是 system/user/assistant")
            if not isinstance(content, str):
                raise LLMValidationError(f"预设 '{preset_name}' 第{idx + 1}项内容必须是字符串")

            stripped = content.strip()
            if role in ("user", "assistant") and not stripped:
                raise LLMValidationError(f"预设 '{preset_name}' 第{idx + 1}项 {role} 消息不能为空")

            entries.append(MessageEntry(role=role, content=stripped, source=f"preset:{preset_name}"))

        logger.debug("成功加载预设 '%s'，包含 %d 条消息", preset_name, len(entries))
        return entries


def get_preset_raw_content(preset_name: str) -> str:
    preset_file = PRESET_DIR / f"{preset_name}.yaml"
    if not preset_file.exists():
        raise LLMValidationError(f"预设 '{preset_name}' 不存在")
    try:
        with preset_file.open("r", encoding="utf-8") as stream:
            return stream.read()
    except Exception as exc:  # noqa: BLE001
        raise LLMValidationError(f"预设 '{preset_name}' 读取失败: {exc}") from exc


def get_preset_system_content(preset_name: str) -> str:
    entries = load_preset(preset_name)
    system_contents = [entry.content for entry in entries if entry.role == "system" and entry.content]
    if not system_contents:
        return ""
    return "\n\n".join(system_contents)


def load_preset_group(group_name: str) -> List[MessageEntry]:
    with _loading_guard(group_name):
        group_file = _find_preset_file(group_name, GROUP_DIR)

        try:
            with group_file.open("r", encoding="utf-8") as stream:
                data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise LLMValidationError(f"预设组 '{group_name}' YAML解析失败: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMValidationError(f"预设组 '{group_name}' 读取失败: {exc}") from exc

        if not isinstance(data, list):
            raise LLMValidationError(f"预设组 '{group_name}' 格式错误：必须是列表")

        all_entries: List[MessageEntry] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise LLMValidationError(f"预设组 '{group_name}' 第{idx + 1}项必须是字典")

            if "preset" in item:
                preset_name = item["preset"]
                if not isinstance(preset_name, str):
                    raise LLMValidationError(f"预设组 '{group_name}' 第{idx + 1}项 preset 值必须是字符串")
                logger.debug("预设组 '%s' 加载预设: %s", group_name, preset_name)
                all_entries.extend(load_preset(preset_name))
            elif "preset-group" in item:
                nested_group = item["preset-group"]
                if not isinstance(nested_group, str):
                    raise LLMValidationError(f"预设组 '{group_name}' 第{idx + 1}项 preset-group 值必须是字符串")
                logger.debug("预设组 '%s' 加载嵌套预设组: %s", group_name, nested_group)
                all_entries.extend(load_preset_group(nested_group))
            else:
                raise LLMValidationError(
                    f"预设组 '{group_name}' 第{idx + 1}项必须包含 'preset' 或 'preset-group' 键"
                )

        logger.debug("预设组 '%s' 展开完成，共 %d 条消息", group_name, len(all_entries))
        return all_entries


__all__ = [
    "load_preset",
    "load_preset_group",
    "get_preset_raw_content",
    "get_preset_system_content",
]
