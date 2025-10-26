"""演示如何调用极简 LLM 封装。"""

from __future__ import annotations

import json
import logging

from scripts.llm_api import (
    LLMClient,
    LLMConfigError,
    LLMTransportError,
    LLMValidationError,
    load_env_file,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_env_file()

    try:
        client = LLMClient.from_env()
    except LLMConfigError as exc:
        logging.error("初始化客户端失败：%s", exc)
        return

    yaml_prompt = """
messages:
  system: |
    你是一个信息整理助手，只能返回合法 JSON 对象。
  user: |
    请总结 Python 的核心优势，并返回一个 JSON 对象，结构如下：
    {
      "language": "Python",
      "summary": "一句话概述",
      "advantages": [
        {"name": "优势标题", "detail": "具体说明"},
        ...
      ]
    }
generation:
  model: gemini-2.5-flash
  format:
    type: json

"""

    try:
        output = client.invoke_from_yaml(yaml_prompt)
        # 如需只构造请求而不调用，可显式传入 dry_run=True、include_debug=True
    except LLMValidationError as exc:
        logging.error("YAML 输入非法：%s", exc)
        return
    except LLMTransportError as exc:
        logging.error("调用下游 LLM 失败：%s", exc)
        return

    print(output)


if __name__ == "__main__":
    main()
