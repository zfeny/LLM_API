"""极简测试脚本。"""
from llm_api import LLMClient, load_env_file

load_env_file()
client = LLMClient.from_env()

yaml_prompt = """
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
  think: -1
"""

output = client.invoke_from_yaml(yaml_prompt, dry_run=False)
print(output)
