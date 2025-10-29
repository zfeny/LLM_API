"""SillyTavern (酒馆) 预设转换器。

将 SillyTavern JSON 格式的预设文件转换为 llm_gemini_api 的 YAML preset 格式。
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError as e:
    raise ImportError("需要 PyYAML: pip install pyyaml") from e


def convert_tavern_to_preset(json_path: str | Path, output_path: str | Path) -> bool:
    """
    将单个 SillyTavern JSON 预设文件转换为 YAML 格式。

    Args:
        json_path: 输入的 JSON 文件路径
        output_path: 输出的 YAML 文件路径

    Returns:
        转换是否成功

    Raises:
        FileNotFoundError: JSON 文件不存在
        json.JSONDecodeError: JSON 格式错误
    """
    json_path = Path(json_path)
    output_path = Path(output_path)

    logger.info(f"转换: {json_path.name} -> {output_path.name}")

    # 读取 JSON 文件
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"文件不存在: {json_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {json_path} - {e}")
        raise

    # 提取 prompts
    prompts = data.get("prompts", [])
    if not prompts:
        logger.warning(f"未找到 prompts: {json_path.name}")
        return False

    # 转换消息
    yaml_content = []

    for prompt in prompts:
        # 提取基本信息
        name = prompt.get("name", "")
        identifier = prompt.get("identifier", "")
        role = prompt.get("role", "")
        content = prompt.get("content", "")
        marker = prompt.get("marker", False)

        # 构建注释
        comment_parts = []
        if name:
            comment_parts.append(name)
        if identifier:
            comment_parts.append(f"({identifier})")
        if marker:
            comment_parts.append("- marker")

        comment = " ".join(comment_parts)

        # 如果是 marker 类型，只添加注释，不添加消息
        if marker:
            yaml_content.append({"_comment": comment})
            continue

        # 验证角色
        if role not in ("system", "user", "assistant"):
            logger.debug(f"跳过未知角色: {role} ({comment})")
            continue

        # 跳过空的 user/assistant 消息（system 消息允许为空）
        if role in ("user", "assistant") and not content.strip():
            logger.debug(f"跳过空消息: {role} ({comment})")
            continue

        # 构建消息条目
        entry = {"_comment": comment} if comment else {}
        entry[role] = content

        yaml_content.append(entry)

    # 提取生成参数
    generation_params = {}
    if "temperature" in data:
        generation_params["temperature"] = data["temperature"]
    if "top_p" in data:
        generation_params["top_p"] = data["top_p"]
    if "top_k" in data:
        generation_params["top_k"] = data["top_k"]

    # 写入 YAML 文件
    try:
        with output_path.open("w", encoding="utf-8") as f:
            # 写入消息
            for item in yaml_content:
                # 写入注释
                if "_comment" in item:
                    comment = item.pop("_comment")
                    f.write(f"# {comment}\n")

                # 如果只有注释（marker类型），跳过消息写入
                if not item:
                    f.write("\n")
                    continue

                # 获取角色和内容
                role = list(item.keys())[0]
                content = item[role]

                # 手动写入 YAML 格式
                # 判断内容是否包含特殊字符或换行
                if not content or content == "":
                    # 空内容
                    f.write(f"- {role}: ''\n")
                elif "\n" in content or content.startswith(" ") or content.endswith(" "):
                    # 多行内容，使用 literal style (|)
                    f.write(f"- {role}: |\n")
                    for line in content.split("\n"):
                        f.write(f"    {line}\n")
                else:
                    # 单行内容，使用 yaml.dump 转义
                    content_str = yaml.dump(content, allow_unicode=True, default_style="'").strip()
                    f.write(f"- {role}: {content_str}\n")

            # 注意：generation 参数不写入 preset 文件
            # preset 只包含消息列表，generation 参数应在使用 preset 时的请求 YAML 中指定
            if generation_params:
                logger.debug(f"跳过 generation 参数: {generation_params}")

        logger.info(f"✓ 转换成功: {output_path.name}")
        return True

    except Exception as e:
        logger.error(f"写入 YAML 失败: {output_path} - {e}")
        return False


def batch_convert(
    json_dir: str | Path = "llm_gemini_api/json",
    preset_dir: str | Path = "llm_gemini_api/preset",
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    批量转换 json_dir 下的所有 JSON 文件。

    Args:
        json_dir: JSON 文件目录
        preset_dir: YAML 输出目录
        overwrite: 是否覆盖已存在的 YAML 文件

    Returns:
        转换结果统计：
        {
            "total": 总文件数,
            "success": 成功数,
            "failed": 失败数,
            "skipped": 跳过数,
            "files": {
                "success": [成功文件列表],
                "failed": [失败文件列表],
                "skipped": [跳过文件列表]
            }
        }
    """
    json_dir = Path(json_dir)
    preset_dir = Path(preset_dir)

    # 确保输出目录存在
    preset_dir.mkdir(parents=True, exist_ok=True)

    # 扫描 JSON 文件
    json_files = list(json_dir.glob("*.json"))

    if not json_files:
        logger.warning(f"未找到 JSON 文件: {json_dir}")
        return {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "files": {"success": [], "failed": [], "skipped": []}
        }

    logger.info(f"找到 {len(json_files)} 个 JSON 文件")

    # 统计
    stats = {
        "total": len(json_files),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "files": {
            "success": [],
            "failed": [],
            "skipped": []
        }
    }

    # 逐个转换
    for json_file in json_files:
        # 构建输出文件路径（替换扩展名）
        output_file = preset_dir / f"{json_file.stem}.yaml"

        # 检查是否已存在
        if output_file.exists() and not overwrite:
            logger.info(f"⊙ 跳过已存在: {output_file.name}")
            stats["skipped"] += 1
            stats["files"]["skipped"].append(json_file.name)
            continue

        # 转换
        try:
            success = convert_tavern_to_preset(json_file, output_file)
            if success:
                stats["success"] += 1
                stats["files"]["success"].append(json_file.name)
            else:
                stats["failed"] += 1
                stats["files"]["failed"].append(json_file.name)
        except Exception as e:
            logger.error(f"✗ 转换失败: {json_file.name} - {e}")
            stats["failed"] += 1
            stats["files"]["failed"].append(json_file.name)

    # 输出统计
    logger.info("=" * 60)
    logger.info(f"转换完成: 成功 {stats['success']}, 失败 {stats['failed']}, 跳过 {stats['skipped']}")
    logger.info("=" * 60)

    return stats


def main():
    """命令行入口。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    # 批量转换
    stats = batch_convert()

    # 输出详细结果
    if stats["files"]["success"]:
        print("\n✓ 成功转换:")
        for filename in stats["files"]["success"]:
            print(f"  - {filename}")

    if stats["files"]["failed"]:
        print("\n✗ 转换失败:")
        for filename in stats["files"]["failed"]:
            print(f"  - {filename}")

    if stats["files"]["skipped"]:
        print("\n⊙ 已跳过:")
        for filename in stats["files"]["skipped"]:
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
