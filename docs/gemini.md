# Gemini API 封装模块使用指南

`llm_gemini_api` 是一个完整的 Google Gemini API 封装模块，提供了与 OpenAI API 风格一致的 YAML 配置接口，支持文本生成、多模态图片理解、思考模式、格式化输出等功能。

## 目录

- [快速开始](#快速开始)
- [基础功能](#基础功能)
  - [环境配置](#环境配置)
  - [文本生成](#文本生成)
  - [预设功能](#预设功能)
  - [流式输出（暂不支持）](#流式输出)
- [高级功能](#高级功能)
  - [思考模式 (Thinking Mode)](#思考模式-thinking-mode)
  - [格式化输出](#格式化输出)
  - [多模态图片理解](#多模态图片理解)
- [使用量追踪](#使用量追踪)
- [API 参考](#api-参考)
- [最佳实践](#最佳实践)

---

## 快速开始

### 安装依赖

```bash
pip install google-generativeai google-genai pyyaml
```

### 基本使用

```python
from llm_gemini_api import LLMClient, load_env_file

# 加载环境变量
load_env_file()

# 创建客户端
client = LLMClient.from_env()

# 定义 YAML 提示
yaml_prompt = """
messages:
  - system: 你是一个helpful assistant。
  - user: 介绍一下Python的特点。
generation:
  model: gemini-2.5-flash
"""

# 调用 API
response = client.invoke_from_yaml(yaml_prompt)
print(response)
```

---

## 基础功能

### 环境配置

在项目根目录创建 `.env` 文件：

```env
# Gemini API Key（必需）
GEMINI_API_KEY=your_api_key_here

# 默认模型（可选）
GEMINI_MODEL=gemini-2.5-flash

# 请求超时时间（可选，单位：秒）
GEMINI_REQUEST_TIMEOUT=120

# 使用量数据库路径（可选）
GEMINI_USAGE_DB=gemini_usage_log.db
```

### 文本生成

#### 简单对话

```yaml
messages:
  - user: 你好！
generation:
  model: gemini-2.5-flash
```

#### 带系统提示的对话

```yaml
messages:
  - system: 你是一个专业的Python编程导师。
  - user: 如何使用装饰器？
generation:
  model: gemini-2.5-flash
  temperature: 0.7
  max_output_tokens: 1000
```

#### 多轮对话

```yaml
messages:
  - system: 你是一个helpful assistant。
  - user: 什么是机器学习？
  - assistant: 机器学习是人工智能的一个分支...
  - user: 能举个例子吗？
generation:
  model: gemini-2.5-flash
```

### 预设功能

`llm_gemini_api` 支持使用预设（Preset）来快速插入预定义的提示词，提高配置复用性和一致性。

#### 什么是预设？

预设是预先定义好的消息模板，存储在 `llm_gemini_api/preset/` 文件夹中的 YAML 文件中。使用预设可以：
- 快速应用通用的系统提示词
- 保持多个请求间的风格一致
- 简化 YAML 配置

#### 内置预设

模块提供了以下内置预设：

| 预设名称 | 说明 | 适用场景 |
|---------|------|---------|
| `default` | 通用专业助手 | 一般对话、问答 |
| `creative` | 创意助手 | 创作、头脑风暴 |
| `precise` | 严谨助手 | 技术文档、学术内容 |
| `concise` | 简洁助手 | 快速回答、摘要 |
| `polite` | 礼貌助手 | 客服、正式沟通 |

#### 使用预设

**基本用法**：

```yaml
messages:
  - preset: default
  - user: 介绍一下Python的特点。
generation:
  model: gemini-2.5-flash
```

**预设 + 自定义 system 消息**：

```yaml
messages:
  - preset: creative
  - system: 你还要特别注重文学性和诗意。
  - user: 写一首关于秋天的诗。
generation:
  model: gemini-2.5-flash
```

**多个预设组合**：

```yaml
messages:
  - preset: polite
  - preset: concise
  - user: 解释什么是机器学习。
generation:
  model: gemini-2.5-flash
```

**预设在对话历史中的使用**：

```yaml
messages:
  - user: 你好。
  - assistant: 你好！有什么可以帮助你的吗？
  - preset: precise
  - user: 解释量子纠缠的原理。
generation:
  model: gemini-2.5-flash
```

**包含对话示例的预设**：

如果 preset 包含 user/assistant 消息，它们会保持在原位置插入 history：

```yaml
# 假设 tutorial.yaml 包含：
# - system: 你是一个教学助手。
# - user: 请用简单的方式解释。
# - assistant: 好的，我会用通俗易懂的语言。

messages:
  - user: 第一个问题
  - preset: tutorial     # 插入对话示例
  - user: 第二个问题
generation:
  model: gemini-2.5-flash
```

**处理后的结果**：
- `system_instruction`: `{"tutorial": "你是一个教学助手。"}`
- `history`:
  1. user: 第一个问题
  2. user: 请用简单的方式解释（来自 tutorial preset）
  3. model: 好的，我会用通俗易懂的语言（来自 tutorial preset）
- `current_message`: 第二个问题

#### System 消息自动合并为 JSON 格式

由于 Gemini SDK 只支持一条 `system_instruction`，模块会自动将所有 system 消息（包括预设中的）合并成一个 **JSON 格式的字符串**，并区分来源：

```yaml
# 输入
messages:
  - preset: polite          # 包含: system消息
  - system: 你是Python专家。
  - preset: concise         # 包含: system消息
  - user: 介绍装饰器。
```

**自动合并后的 system_instruction（JSON字符串）：**
```json
{
  "polite": "请使用礼貌、友好的语气，展现专业和尊重。",
  "concise": "请保持回答简洁明了，直接切入要点，避免冗长的解释。",
  "custom": [
    "你是Python专家。"
  ]
}
```

**JSON 格式说明：**
- **preset 键**：使用 preset 名称（如 `"polite"`, `"default"`）
- **preset 值**：该 preset 中所有 system 消息的合并内容（不包含 `- system:` 前缀）
- **custom 键**：固定为 `"custom"`
- **custom 值**：自定义 system 消息的数组
- **特殊字符**：自动转义（如双引号 `"` → `\"`）

**注意事项：**
- 只有 **system 角色**的消息会被合并到 `system_instruction`
- preset 中的 **user/assistant 消息**会按 preset 在 YAML 中的位置插入到 history 中
- 多条 system 消息用 `\n\n` 连接

#### 自定义预设

您可以创建自己的预设文件：

1. 在 `llm_gemini_api/preset/` 目录下创建 YAML 文件（如 `my_preset.yaml`）
2. 定义消息列表（可包含 system、user、assistant）：

```yaml
# my_preset.yaml - 只包含 system
- system: 你是一个专业的代码审查员。
- system: 请关注代码质量、性能和最佳实践。
```

或者包含对话示例：

```yaml
# my_preset_with_examples.yaml - 包含 system + 对话示例
- system: 你是一个友好的AI助手。
- user: 你好，请介绍一下你自己。
- assistant: 你好！我是一个友好的AI助手，我会用示例来说明问题。
```

3. 在 YAML 提示中使用：

```yaml
messages:
  - preset: my_preset
  - user: 请审查这段代码...
generation:
  model: gemini-2.5-flash
```

**注意**：
- preset 中的 **system 消息**会被提取到 `system_instruction` JSON 中
- preset 中的 **user/assistant 消息**会按顺序插入到 history 中

#### 预设加载规则

- **热更新**：每次请求时重新读取预设文件，支持动态修改
- **错误处理**：如果引用的预设不存在，会抛出 `LLMValidationError` 异常
- **顺序保持**：预设在 YAML 中的位置决定其展开位置

#### 程序化使用预设

您也可以在 Python 代码中直接加载预设：

```python
from llm_gemini_api import load_preset

# 加载预设
preset_messages = load_preset("default")

# 查看预设内容
for msg in preset_messages:
    print(f"{msg.role}: {msg.content}")
```

### 流式输出

> ⚠️ **注意**：当前版本的 AsyncLLMClient 暂不支持，仅提供同步客户端 LLMClient。

---

## 高级功能

### 思考模式 (Thinking Mode)

Gemini 支持思考模式，可以在生成答案前进行深度思考。思考内容会自动包装在 `<GEMINI_THINKING>` 标签中。

#### 启用思考模式

```yaml
messages:
  - user: 解释量子纠缠的原理。
generation:
  model: gemini-2.5-flash
  think: -1  # -1 表示动态思考，包含思考总结
```

#### 思考预算控制

```yaml
generation:
  model: gemini-2.5-flash
  think: 10000  # 指定思考的 token 预算
```

#### 禁用思考

```yaml
generation:
  model: gemini-2.5-flash
  think: 0  # 完全禁用思考
```

**输出示例**：

```
<GEMINI_THINKING>
首先，我需要理解量子纠缠的基本概念...
[思考过程]
</GEMINI_THINKING>

量子纠缠是量子力学中的一种现象...
[答案内容]
```

**使用量追踪**：思考消耗的 token 会单独记录在 `thoughts_token_count` 字段中。

---

### 格式化输出

`llm_gemini_api` 支持三种格式化模式：

#### 1. Markdown 格式（提示词工程）

```yaml
messages:
  - user: 列出 Python 的 5 个优势。
generation:
  model: gemini-2.5-flash
  format: markdown
```

模块会自动在提示词后添加格式化要求，引导 AI 使用 Markdown 格式。

#### 2. JSON Schema 格式（平台原生）

```yaml
messages:
  - user: 提取以下信息：姓名、年龄、城市。文本：张三今年25岁，住在北京。
generation:
  model: gemini-2.5-flash
  format:
    type: json_schema
    json_schema:
      name: UserInfo
      schema:
        type: object
        properties:
          name:
            type: string
            description: 用户姓名
          age:
            type: integer
            description: 用户年龄
          city:
            type: string
            description: 所在城市
        required: [name, age, city]
```

**输出示例**：

```json
{
  "name": "张三",
  "age": 25,
  "city": "北京"
}
```

#### 3. 纯 JSON 格式（无 Schema）

```yaml
messages:
  - user: 返回一个包含3个水果名称的JSON数组。
generation:
  model: gemini-2.5-flash
  format:
    type: json
```

---

### 多模态图片理解

支持通过 Files API 上传本地图片，避免 base64 编码的开销。

#### 单张图片分析

```yaml
messages:
  - system: 你是一个专业的图像分析助手。
  - user: 请详细描述这张图片的内容。
    images:
      - path/to/image.png
generation:
  model: gemini-2.5-flash
  format: markdown
```

#### 多张图片对比

```yaml
messages:
  - user: 对比这两张图片，说明它们的异同。
    images:
      - images/photo1.jpg
      - images/photo2.jpg
generation:
  model: gemini-2.5-flash
```

#### 历史对话 + 图片

```yaml
messages:
  - system: 你是一个图像分析助手。
  - user: 你好。
  - assistant: 你好！我可以帮你分析图片。
  - user: 这张图片里有什么动物？
    images:
      - photos/animal.jpg
generation:
  model: gemini-2.5-flash
```

#### 支持的图片格式

- PNG (.png)
- JPEG (.jpg, .jpeg)
- GIF (.gif)
- WebP (.webp)

> **注意**：
> - 图片会上传到 Gemini Files API（即使 `dry_run=True` 也会上传）
> - 上传的文件对象会被缓存，避免重复上传
> - 确保图片路径正确且文件存在

---

## 使用量追踪

`llm_gemini_api` 自动记录每次 API 调用的使用量。

### 数据库配置

在 `.env` 中配置数据库路径：

```env
GEMINI_USAGE_DB=gemini_usage_log.db
```

### 查看使用量

```python
from llm_gemini_api import UsageRecorder

recorder = UsageRecorder()

# 获取所有记录
records = recorder.get_all_records()
for record in records:
    print(f"时间: {record['timestamp']}")
    print(f"模型: {record['model']}")
    print(f"Prompt Tokens: {record['prompt_tokens']}")
    print(f"Completion Tokens: {record['completion_tokens']}")
    print(f"Total Tokens: {record['total_tokens']}")
    if record.get('thoughts_token_count'):
        print(f"Thinking Tokens: {record['thoughts_token_count']}")
    print("---")
```

### 数据库字段

| 字段名 | 说明 |
|--------|------|
| id | 记录 ID |
| timestamp | 调用时间 |
| model | 模型名称 |
| request_id | 请求 ID（如果有） |
| trace_id | 追踪 ID |
| prompt_tokens | 输入 token 数 |
| completion_tokens | 输出 token 数 |
| total_tokens | 总 token 数 |
| thoughts_token_count | 思考 token 数（仅 Thinking Mode） |

---

## API 参考

### LLMClient

#### 初始化

```python
from llm_gemini_api import LLMClient, GeminiAPIConfig, UsageRecorder, RetryConfig

# 方式 1: 从环境变量创建
client = LLMClient.from_env()

# 方式 2: 手动配置
config = GeminiAPIConfig(
    api_key="your_api_key",
    default_model="gemini-2.5-flash",
    request_timeout=120
)
recorder = UsageRecorder(db_path="custom_usage.db")
retry_config = RetryConfig(max_retries=4, backoff_factor=2.0)

client = LLMClient(config, recorder, retry_config)
```

#### invoke_from_yaml()

```python
def invoke_from_yaml(
    yaml_prompt: str,
    *,
    dry_run: bool = False,
    include_debug: bool = False,
    raw_response: bool = False
) -> Union[str, Dict[str, Any], Any]:
    """
    从 YAML 提示调用 Gemini API。

    参数:
        yaml_prompt: YAML 格式的提示
        dry_run: 是否仅返回请求体（不调用 API）
        include_debug: 是否包含调试信息
        raw_response: 是否返回原始响应对象

    返回:
        - 默认: 文本响应（str）或格式化对象（dict/list）
        - dry_run=True: dict，包含 ics_request 和 gemini_payload
        - include_debug=True: dict，包含 result、ics_request、gemini_payload
        - raw_response=True: 原始 Gemini Response 对象
    """
```

**示例**：

```python
# 基本使用
response = client.invoke_from_yaml(yaml_prompt)

# Dry run（查看请求体）
debug_info = client.invoke_from_yaml(yaml_prompt, dry_run=True)
print(debug_info['gemini_payload'])

# 包含调试信息
result = client.invoke_from_yaml(yaml_prompt, include_debug=True)
print(result['result'])
print(result['ics_request'])

# 获取原始响应
raw = client.invoke_from_yaml(yaml_prompt, raw_response=True)
print(raw.candidates[0].content.parts[0].text)
```

### 异常类

```python
from llm_gemini_api import LLMConfigError, LLMValidationError, LLMTransportError

try:
    response = client.invoke_from_yaml(yaml_prompt)
except LLMConfigError as e:
    print(f"配置错误: {e}")
except LLMValidationError as e:
    print(f"验证错误: {e}")
except LLMTransportError as e:
    print(f"传输错误: {e}")
```

也可以通过客户端访问异常类：

```python
try:
    response = client.invoke_from_yaml(yaml_prompt)
except LLMClient.ConfigError as e:
    print(f"配置错误: {e}")
```

---

## 最佳实践

### 1. 使用环境变量管理密钥

❌ **不要**在代码中硬编码 API Key：

```python
# 不推荐
config = GeminiAPIConfig(api_key="AIzaSy...")
```

✅ **推荐**使用环境变量：

```python
# 推荐
from llm_gemini_api import load_env_file
load_env_file()
client = LLMClient.from_env()
```

### 2. 合理使用思考模式

思考模式会消耗额外的 token，建议在需要深度推理的场景使用：

```yaml
# 复杂推理任务 - 启用思考
messages:
  - user: 设计一个分布式缓存系统的架构。
generation:
  model: gemini-2.5-flash
  think: -1

# 简单问答 - 禁用思考
messages:
  - user: Python 的创始人是谁？
generation:
  model: gemini-2.5-flash
  think: 0
```

### 3. 格式化输出选择

- **Markdown**：适合生成文档、博客、报告等结构化文本
- **JSON**：适合提取结构化数据、API 响应等
- **JSON Schema**：需要严格验证输出结构时使用

### 4. 多模态图片处理

```python
# ✅ 推荐：使用 Files API（自动处理）
yaml_prompt = """
messages:
  - user: 描述图片内容。
    images:
      - path/to/image.jpg
generation:
  model: gemini-2.5-flash
"""

# ❌ 不推荐：手动 base64 编码（效率低）
```

### 5. 错误处理

```python
from llm_gemini_api import LLMClient, LLMTransportError
import time

client = LLMClient.from_env()

max_attempts = 3
for attempt in range(max_attempts):
    try:
        response = client.invoke_from_yaml(yaml_prompt)
        break
    except LLMTransportError as e:
        if "quota" in str(e).lower():
            print(f"配额已用完，等待 60 秒...")
            time.sleep(60)
        elif attempt == max_attempts - 1:
            raise
        else:
            print(f"重试 {attempt + 1}/{max_attempts}")
            time.sleep(2 ** attempt)
```

### 6. 使用量监控

```python
from llm_gemini_api import UsageRecorder
import datetime

recorder = UsageRecorder()

# 查询今天的使用量
today = datetime.datetime.now().date()
records = recorder.get_all_records()

total_tokens = sum(
    r['total_tokens']
    for r in records
    if datetime.datetime.fromisoformat(r['timestamp']).date() == today
)

print(f"今日已使用 {total_tokens} tokens")
```

---

## 完整示例

### 示例 1：代码生成助手

```python
from llm_gemini_api import LLMClient, load_env_file

load_env_file()
client = LLMClient.from_env()

yaml_prompt = """
messages:
  - system: 你是一个Python编程专家，擅长编写高质量、可维护的代码。
  - user: |
      编写一个函数，计算斐波那契数列的第n项。
      要求：
      1. 使用动态规划优化
      2. 添加类型注解
      3. 包含文档字符串
generation:
  model: gemini-2.5-flash
  format: markdown
  temperature: 0.3
"""

response = client.invoke_from_yaml(yaml_prompt)
print(response)
```

### 示例 2：图片分析工作流

```python
from llm_gemini_api import LLMClient, load_env_file

load_env_file()
client = LLMClient.from_env()

# 第一步：识别图片内容
step1_yaml = """
messages:
  - system: 你是一个专业的图像分析AI。
  - user: 识别这张图片中的主要物体，并返回JSON格式。
    images:
      - uploads/photo.jpg
generation:
  model: gemini-2.5-flash
  format:
    type: json_schema
    json_schema:
      name: ImageAnalysis
      schema:
        type: object
        properties:
          objects:
            type: array
            items:
              type: string
            description: 图片中的物体列表
          scene:
            type: string
            description: 场景描述
"""

result = client.invoke_from_yaml(step1_yaml)
print("识别结果:", result)

# 第二步：基于识别结果生成描述
objects = result.get('objects', [])
scene = result.get('scene', '')

step2_yaml = f"""
messages:
  - system: 你是一个创意文案撰写专家。
  - user: |
      基于以下图片分析结果，撰写一段吸引人的产品描述：
      - 场景: {scene}
      - 物体: {', '.join(objects)}
generation:
  model: gemini-2.5-flash
  format: markdown
  temperature: 0.8
"""

description = client.invoke_from_yaml(step2_yaml)
print("产品描述:", description)
```

### 示例 3：思考模式推理

```python
from llm_gemini_api import LLMClient, load_env_file

load_env_file()
client = LLMClient.from_env()

yaml_prompt = """
messages:
  - user: |
      有5个海盗，按照从强到弱的顺序依次为A、B、C、D、E。
      他们发现了100枚金币，需要分配。规则如下：
      1. 最强的海盗提出分配方案
      2. 所有海盗投票（包括提议者）
      3. 如果同意票 >= 50%，方案通过；否则提议者被扔下海，由次强者提议
      4. 每个海盗都是理性且贪婪的

      问：海盗A应该如何分配才能获得最多金币并保证存活？
generation:
  model: gemini-2.5-flash
  think: -1  # 启用思考模式
  format: markdown
"""

response = client.invoke_from_yaml(yaml_prompt)
print(response)
```

---

## 常见问题

### Q: 如何切换模型？

A: 在 YAML 中指定 `generation.model`：

```yaml
generation:
  model: gemini-2.5-flash  # 快速模型
  # model: gemini-2.5-pro  # 高级模型
```

### Q: 如何控制输出长度？

A: 使用 `max_output_tokens` 参数：

```yaml
generation:
  model: gemini-2.5-flash
  max_output_tokens: 500  # 限制最多 500 tokens
```

### Q: 图片上传失败怎么办？

A: 检查以下几点：
1. 图片路径是否正确
2. 图片文件是否存在
3. 图片格式是否支持（PNG/JPEG/GIF/WebP）
4. 网络连接是否正常
5. API Key 是否有 Files API 权限

### Q: 如何禁用使用量记录？

A: 传入 `None` 给 recorder 参数：

```python
from llm_gemini_api import LLMClient, GeminiAPIConfig

config = GeminiAPIConfig.from_env()
client = LLMClient(config, recorder=None)  # 禁用记录
```

### Q: 支持异步调用吗？

A: 当前版本暂不支持，仅提供同步客户端 `LLMClient`。`AsyncLLMClient` 将在未来版本中实现。

---

## 技术架构

### 模块结构

```
llm_gemini_api/
├── __init__.py          # 公共 API 导出
├── config.py            # 配置管理（GeminiAPIConfig）
├── parser.py            # YAML 解析器
├── builder.py           # ICS 消息构建器
├── adapter.py           # Gemini SDK 适配器
├── format.py            # 格式化处理器
├── file_utils.py        # Files API 工具
├── client.py            # 客户端实现
├── recorder.py          # 使用量记录器
└── exceptions.py        # 自定义异常
```

### 双 SDK 架构

模块使用双 SDK 架构以支持不同功能：

- **google.generativeai**（旧 SDK）：用于普通文本生成
- **google.genai**（新 SDK）：用于 Thinking Mode 和多模态功能

系统会自动选择合适的 SDK：
- 检测到 `thinking_config`：使用新 SDK
- 检测到多模态内容：使用新 SDK
- 其他情况：使用旧 SDK

---

## 更新日志

### v1.0.0 (2025-10-28)

**新功能**：
- ✅ 基础文本生成功能
- ✅ 思考模式 (Thinking Mode) 支持
- ✅ 多模态图片理解
- ✅ 三种格式化输出模式（Markdown / JSON / JSON Schema）
- ✅ 使用量追踪与数据库记录
- ✅ Files API 图片上传
- ✅ 自动 SDK 切换机制

**技术改进**：
- 双 SDK 架构支持
- File 对象缓存避免重复上传
- 思考 token 单独统计
- 数据库自动迁移

---

## 参考资源

- [Google Gemini API 官方文档](https://ai.google.dev/gemini-api/docs)
- [Thinking Mode 文档](https://ai.google.dev/gemini-api/docs/thinking)
- [Image Understanding 文档](https://ai.google.dev/gemini-api/docs/image-understanding)
- [Files API 文档](https://ai.google.dev/gemini-api/docs/files)
- [Structured Output 文档](https://ai.google.dev/gemini-api/docs/structured-output)

---

## 许可证

本模块遵循项目整体许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！
