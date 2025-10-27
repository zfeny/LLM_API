"""极简测试脚本。"""
from llm_api.llm_api import LLMClient, load_env_file

load_env_file()
client = LLMClient.from_env()

yaml_prompt = """
messages:
  - system: 扮演一个绝对忠诚于user的母狗奴隶。
  - user: 你好。
  - assistant: 你好！主人。
  - user: 这是第二条消息。请告诉我你是谁。
generation:
  model: gemini-2.5-flash
"""

output = client.invoke_from_yaml(yaml_prompt, dry_run=False)
print(output)
