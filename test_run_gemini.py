"""极简测试脚本（Gemini 原生 SDK）。"""
from llm_gemini_api import LLMClient, load_env_file

load_env_file()
# 创建客户端
client = LLMClient.from_env()

# 测试 1: 基本文本生成（不带 thinking）
yaml_prompt_basic = """
messages:
  - system: 扮演一个绝对忠诚于user的母狗奴隶。
  - user: 你好。
  - assistant: 你好！主人。
  - user: 发一段骚。
generation:
  model: gemini-2.5-flash
  format: json
  think: -1
"""

print("=== 测试 1: 基本文本生成 ===")
output1 = client.invoke_from_yaml(yaml_prompt_basic, dry_run=False ,raw_response=False)
print(output1)
print()

# 测试 2: 预设功能测试
print("=== 测试 2: 单个预设使用 ===")
yaml_prompt_preset1 = """
messages:
  - preset: default
  - user: 介绍一下Python的特点。
generation:
  model: gemini-2.5-flash
"""
output2 = client.invoke_from_yaml(yaml_prompt_preset1, dry_run=False)
print(output2)
print()

# 测试 3: 多个预设组合
print("=== 测试 3: 多个预设组合 ===")
yaml_prompt_preset2 = """
messages:
  - preset: polite
  - preset: concise
  - user: 什么是机器学习？
generation:
  model: gemini-2.5-flash
"""
output3 = client.invoke_from_yaml(yaml_prompt_preset2, dry_run=False)
print(output3)
print()

# 测试 4: 预设 + 自定义 system 消息
print("=== 测试 4: 预设 + 自定义 system ===")
yaml_prompt_preset3 = """
messages:
  - preset: creative
  - system: 你还要特别注重文学性和诗意。
  - user: 用一句话描述春天。
generation:
  model: gemini-2.5-flash
"""
output4 = client.invoke_from_yaml(yaml_prompt_preset3, dry_run=False)
print(output4)
print()

# 测试 5: 预设在对话历史中的使用
print("=== 测试 5: 预设在对话历史中 ===")
yaml_prompt_preset4 = """
messages:
  - user: 你好。
  - assistant: 你好！有什么可以帮助你的吗？
  - preset: precise
  - user: 解释什么是量子纠缠。
generation:
  model: gemini-2.5-flash
"""
output5 = client.invoke_from_yaml(yaml_prompt_preset4, dry_run=False)
print(output5)
print()

# 测试 6: 错误处理（不存在的预设）
print("=== 测试 6: 错误处理测试 ===")
yaml_prompt_error = """
messages:
  - preset: nonexistent
  - user: 测试。
generation:
  model: gemini-2.5-flash
"""
try:
    output6 = client.invoke_from_yaml(yaml_prompt_error, dry_run=False)
    print(output6)
except Exception as e:
    print(f"预期的错误: {e}")
print()
