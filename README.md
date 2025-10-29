# LLM API 统一封装框架

一个轻量、灵活、生产级的大语言模型 API 封装框架，支持多种 LLM 提供商，提供统一的 YAML 配置接口。

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 核心特性

- **🎨 统一接口**：一套 YAML 语法适配多个 LLM 提供商
- **⚡ 轻量高效**：精简代码，零冗余依赖
- **🔄 生产就绪**：自动重试、使用量追踪、批量写入
- **🛡️ 类型安全**：完整类型注解，IDE 友好
- **📊 可观测性**：自动记录 API 调用和使用量
- **🧩 模块化设计**：独立模块，按需导入

---

## 📦 支持的 LLM 提供商

| 提供商 | 模块名 | 特色功能 | 文档 |
|-------|--------|---------|------|
| **Google Gemini** | `gemini` | 思考模式、多模态图片理解 | [📖 查看文档](docs/gemini.md) |
| **OpenAI 兼容** | `openai` | 异步并发、通用兼容 | [📖 查看文档](docs/openai.md) |

> **OpenAI 兼容**模块支持所有遵循 OpenAI API 规范的服务：OpenAI 官方、Azure OpenAI、本地部署服务等

---

## 🚀 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/yourusername/LLM_API.git
cd LLM_API
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# OpenAI 兼容 API
LLM_API_KEY=your_openai_api_key_here
LLM_API_BASE=https://api.openai.com/v1
LLM_MODEL=gpt-4
```

### 3. 使用示例

#### Gemini 模块

```python
from gemini import LLMClient, load_env_file

load_env_file()
client = LLMClient.from_env()

yaml_prompt = """
messages:
  - system: 你是一个helpful assistant。
  - user: 介绍一下Python的特点。
generation:
  model: gemini-2.5-flash
"""

response = client.invoke_from_yaml(yaml_prompt)
print(response)
```

#### OpenAI 兼容模块

```python
from openai import LLMClient, load_env_file

load_env_file()
client = LLMClient.from_env()

yaml_prompt = """
messages:
  - user: 用Markdown总结Python的核心优势。
generation:
  model: gpt-4
  format: markdown
"""

response = client.invoke_from_yaml(yaml_prompt)
print(response)
```

---

## ✨ 主要功能

### 统一的 YAML 配置

```yaml
messages:
  - system: 系统提示词
  - user: 用户消息
generation:
  model: gemini-2.5-flash
  temperature: 0.7
  max_output_tokens: 2048
  format: markdown  # 可选：格式化输出
```

### 思考模式（Gemini 专属）

```yaml
generation:
  model: gemini-2.5-flash
  think: -1  # 启用深度思考
```

### 多模态图片理解（Gemini 专属）

```yaml
messages:
  - user: 描述这张图片。
    images:
      - path/to/image.jpg
generation:
  model: gemini-2.5-flash
```

### 格式化输出

支持 Markdown、JSON、JSON Schema 三种格式：

```yaml
generation:
  format:
    type: json_schema
    json_schema:
      name: UserInfo
      schema:
        type: object
        properties:
          name: {type: string}
          age: {type: integer}
```

### 自动使用量追踪

所有 API 调用自动记录到 SQLite 数据库：

```python
from gemini import UsageRecorder

recorder = UsageRecorder()
records = recorder.get_all_records()

for record in records:
    print(f"模型: {record['model']}, Token: {record['total_tokens']}")
```

---

## 📚 详细文档

- **[Gemini 模块完整指南](docs/gemini.md)** - 思考模式、多模态、Files API
- **[OpenAI 兼容模块指南](docs/openai.md)** - 异步并发、自定义配置
- **[OpenList 开发规范](docs/openlist.md)** - 项目开发流程

---

## 🏗️ 项目结构

```
LLM_API/
├── llm/                # 公共基类与共享工具（含 preset_module/ 资源）
├── gemini/             # Gemini 原生 API 封装
├── openai/             # OpenAI 兼容 API 封装
├── docs/                    # 详细文档
├── test_run_gemini.py       # Gemini 测试示例
├── test_run.py              # OpenAI 测试示例
└── README.md                # 本文件
```

### 三层架构

```
YAML 输入 → Parser → ICS 中间层 → Adapter → SDK
```

---

## 🎓 典型场景

### 代码生成

```python
yaml_prompt = """
messages:
  - system: 你是Python专家。
  - user: 编写斐波那契函数，使用动态规划。
generation:
  model: gemini-2.5-flash
  format: markdown
"""
```

### 图片分析

```python
yaml_prompt = """
messages:
  - user: 识别图片中的物体。
    images: [photo.jpg]
generation:
  model: gemini-2.5-flash
  format:
    type: json_schema
    json_schema:
      name: ImageAnalysis
      schema:
        type: object
        properties:
          objects: {type: array}
          scene: {type: string}
"""
```

### 高并发服务

```python
from openai import AsyncLLMClient

async def process_batch(prompts):
    client = AsyncLLMClient.from_env()
    tasks = [client.invoke_from_yaml(p) for p in prompts]
    return await asyncio.gather(*tasks)
```

---

## ⚙️ 配置说明

### 重试配置

```python
from gemini import RetryConfig

retry_config = RetryConfig(
    max_retries=5,
    initial_delay=1.0,
    exponential_base=2.0
)
client = LLMClient.from_env(retry_config=retry_config)
```

### 使用量记录

```python
from gemini import UsageRecorder

recorder = UsageRecorder(
    db_path="custom_usage.db",
    batch_size=20
)
client = LLMClient.from_env(recorder=recorder)
```

---

## 🐛 常见问题

### 如何选择模块？

- **Gemini 原生**：需要思考模式、多模态等 Gemini 专属功能
- **OpenAI 兼容**：使用 OpenAI API 或需要异步并发

### 两个模块能同时使用吗？

可以，完全独立：

```python
from gemini import LLMClient as GeminiClient
from openai import LLMClient as OpenAIClient
```

### 如何迁移代码？

只需更改导入语句，YAML 格式基本兼容。

详细问题请查看各模块文档。

---

## 📊 性能特性

| 特性 | 效果 |
|------|------|
| 批量写入 | 减少 90% 数据库连接 |
| 异步 I/O | 高并发 8x 性能提升 |
| 自动重试 | 成功率 +80% |
| 文件缓存 | 避免重复上传 |

---

## 📝 更新日志

### v2.0.0 (2025-10-28)

- ✅ 添加 Gemini 原生封装模块
- ✅ 思考模式、多模态图片理解
- ✅ Files API 集成
- 🔄 模块化重构

### v1.0.0 (2025-10-27)

- ✅ OpenAI 兼容封装
- ✅ 异步客户端、自动重试
- ✅ 使用量追踪

---

## 🤝 贡献指南

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/xxx`
3. 提交更改：`git commit -m 'Add xxx'`
4. 推送分支：`git push origin feature/xxx`
5. 创建 Pull Request

详细开发规范请查看 [OpenList 文档](docs/openlist.md)。

---

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Google Generative AI Python SDK](https://github.com/google/generative-ai-python)
- [PyYAML](https://github.com/yaml/pyyaml)

---

**Happy Coding! 🚀**
