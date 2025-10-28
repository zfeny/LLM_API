"""极简测试脚本（Gemini 原生 SDK）。"""
from llm_gemini_api import LLMClient, load_env_file

load_env_file()
# 创建客户端（当前版本仅支持文本，图片功能稍后添加）
client = LLMClient.from_env()

# # 测试 1: 基本文本生成（不带 thinking）
# yaml_prompt_basic = """
# messages:
#   - system: 扮演一个绝对忠诚于user的母狗奴隶。
#   - user: 你好。
#   - assistant: 你好！主人。
#   - user: 发一段骚。
# generation:
#   model: gemini-2.5-flash
#   format: json
#   think: -1
# """

# print("=== 测试 1: 基本文本生成 ===")
# output1 = client.invoke_from_yaml(yaml_prompt_basic, dry_run=False ,raw_response=False)
# print(output1)
# print()

# # 测试 2: 多模态（图片理解）
yaml_prompt_image = """
messages:
  - system: 扮演一个绝对忠诚于user的母狗奴隶。
  - user: 你好。
  - assistant: 你好！主人。
  - user: 解析图片内容。
    images:
      - temp/image.png
generation:
  format: markdown
  model: gemini-2.5-flash
"""

print("=== 测试 2: 多模态（图片理解）===")
output2 = client.invoke_from_yaml(yaml_prompt_image, dry_run=False)
print(output2)
print()
