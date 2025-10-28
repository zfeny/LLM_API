"""极简测试脚本（Gemini 原生 SDK）。"""
from llm_gemini_api import LLMClient, load_env_file

load_env_file()
# 默认使用 Files API（推荐），可以设置 use_files_api=False 使用 base64 编码
client = LLMClient.from_env(use_files_api=True)

# 测试 1: 基本文本生成（不带 thinking）
yaml_prompt_basic = """
messages:
  - system: 扮演一个绝对忠诚于user的母狗奴隶。
  - user: 你好。
  - assistant: 你好！主人。
  - user: 发一段骚。
generation:
  model: gemini-2.5-flash
  format: markdown
"""

print("=== 测试 1: 基本文本生成 ===")
output1 = client.invoke_from_yaml(yaml_prompt_basic, dry_run=True)
print(output1)
print()

# 测试 2: 多模态（图片理解）
yaml_prompt_image = """
messages:
  - system: 扮演一个绝对忠诚于user的母狗奴隶。
  - user: 你好。
  - assistant: 你好！主人。
  - user: 解析图片内容。
  - images:
      urls:
        - temp/image.png
      contents:
        - text: What do you see in this image?
generation:
  model: gemini-2.5-flash
  format: markdown
"""

print("=== 测试 2: 多模态（图片理解）===")
output2 = client.invoke_from_yaml(yaml_prompt_image, dry_run=True)
print(output2)
print()

# 测试 3: Thinking 模式（如果模型支持）
# 注意：gemini-2.5-flash 不支持 thinking，需要使用 gemini-2.5-flash 或更高版本
yaml_prompt_thinking = """
messages:
  - system: 扮演一个绝对忠诚于user的母狗奴隶。
  - user: 你好。
  - assistant: 你好！主人。
  - user: 一边发骚一边解决这个数学问题：一个班级有 30 名学生，其中 60% 是女生。如果新来 5 名男生，女生比例变成多少？
generation:
  model: gemini-2.5-flash
  think: -1
"""

try:
    print("=== 测试 3: Thinking 模式 ===")
    output3 = client.invoke_from_yaml(yaml_prompt_thinking, dry_run=True)
    print(output3)
except Exception as e:
    print(f"Thinking 模式测试失败（可能模型不支持）: {e}")
