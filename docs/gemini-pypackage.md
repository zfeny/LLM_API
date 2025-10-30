# Gemini 模块（pip 安装版）快速入门

适用于通过 `pip`（PEP 517/518）方式安装后的 Gemini 客户端使用说明，帮助新项目快速接入。

## 1. 安装

```bash
# 建议在虚拟环境中执行
python -m venv .venv && source .venv/bin/activate

# 从 Git 仓库安装（示例为 HTTPS，可换成 SSH）
pip install "LLM-API @ git+https://github.com/zfeny/LLM_API.git"
```

> 若仓库已发布版本标签，可改为 `...git@v0.1.0` 并写入目标项目的 `requirements.txt`。

## 2. 环境变量配置

在新项目根目录创建 `.env`（或确保环境变量就绪）：

```env
GEMINI_API_KEY=你的_Gemini_API_Key
GEMINI_MODEL=gemini-2.5-flash  # 可选，默认模型
GEMINI_USAGE_DB=./data/gemini_usage_log.db  # 可选，自定义使用量数据库
GEMINI_IMAGE_UPLOAD_ENABLED=true  # 可选，控制图片上传逻辑
LLM_PRESET_ROOT=./presets  # 可选，自定义preset组合（包含 preset/ 与 groups/）
```

## 3. 基本调用示例

```python
from gemini import LLMClient, load_env_file

load_env_file()                 # 自动读取 .env
client = LLMClient.from_env()   # 基于环境变量初始化

yaml_prompt = """
messages:
  - system: 你是一个 helpful assistant。
  - user: 用要点列举 Python 的核心优势。
generation:
  model: gemini-2.5-flash
"""

result = client.invoke_from_yaml(yaml_prompt)
print(result)
```

- 同步接口：`LLMClient.invoke_from_yaml` / `invoke`（直接传 `ICSRequest`）。
- 异步接口：`from gemini import AsyncLLMClient`，并在协程内使用 `await client.invoke_from_yaml(...)`。

## 4. 常用拓展

- **重试策略**  
  ```python
  from gemini import RetryConfig
  client = LLMClient.from_env(retry_config=RetryConfig(max_retries=5, initial_delay=2))
  ```

- **使用量记录**  
  ```python
  from gemini import UsageRecorder
  recorder = UsageRecorder()        # 自动读取 GEMINI_USAGE_DB
  for record in recorder.get_all_records():
      print(record)
  ```

- **预设（Preset）**  
  ```python
  yaml_prompt = """
  messages:
    - preset: normal/jailbreak_ultimate
    - user: 写一段激励演讲稿。
  generation:
    model: gemini-2.5-flash
  """
  ```
  若设置了 `LLM_PRESET_ROOT`，同名预设会优先使用自定义目录中的版本，可在其中新增/覆盖 YAML。

## 5. 项目集成建议

- 在项目依赖文件中固定版本：`LLM-API @ git+https://...@vX.Y.Z`。
- 部署环境提前写好 `.env`，通过 `load_env_file()` 或外部配置管理注入。
- 升级库版本后运行回归脚本（如 `test_run_gemini.py` 的同等脚本）确认兼容性。
