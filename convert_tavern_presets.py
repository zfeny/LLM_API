#!/usr/bin/env python3
"""SillyTavern 预设批量转换工具。

使用方法:
    python convert_tavern_presets.py

功能:
    批量转换 llm_gemini_api/json/ 下的所有 JSON 文件为 YAML 格式，
    输出到 llm_gemini_api/preset/ 目录。
"""
from llm_gemini_api.tavern_converter import batch_convert
import logging

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    print("=" * 60)
    print("SillyTavern 预设转换工具")
    print("=" * 60)

    # 批量转换
    stats = batch_convert(
        json_dir="llm_gemini_api/json",
        preset_dir="llm_gemini_api/json2yaml",
        overwrite=False  # 不覆盖已存在的文件
    )

    # 输出详细结果
    print("\n" + "=" * 60)
    print("转换结果")
    print("=" * 60)

    if stats["files"]["success"]:
        print(f"\n✓ 成功转换 ({stats['success']} 个):")
        for filename in stats["files"]["success"]:
            print(f"  - {filename}")

    if stats["files"]["failed"]:
        print(f"\n✗ 转换失败 ({stats['failed']} 个):")
        for filename in stats["files"]["failed"]:
            print(f"  - {filename}")

    if stats["files"]["skipped"]:
        print(f"\n⊙ 已跳过 ({stats['skipped']} 个):")
        for filename in stats["files"]["skipped"]:
            print(f"  - {filename}")

    print(f"\n总计: {stats['total']} 个文件")
    print("=" * 60)
