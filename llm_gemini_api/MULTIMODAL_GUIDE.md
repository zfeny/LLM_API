# Gemini 多模态功能使用指南

## 概述

llm_gemini_api 现已支持多模态功能，可以在对话中发送图片并让 Gemini 理解分析图片内容。

## 功能特性

1. **简化的 YAML 语法**：直接在消息中指定本地图片路径
2. **Files API 上传**：自动上传本地图片到 Gemini Files API（避免 base64 编码）
3. **多图片支持**：单条消息可包含多张图片
4. **历史对话支持**：历史消息中的图片也会被正确处理
5. **格式化支持**：支持与 format 配置（markdown/json）结合使用

## YAML 语法

### 基本语法

```yaml
messages:
  - user: 请描述这张图片。
    images:
      - path/to/image1.png
      - path/to/image2.jpg
generation:
  model: gemini-2.5-flash
```

### 完整示例

#### 1. 单张图片理解

```yaml
messages:
  - system: 你是一个专业的图像分析助手。
  - user: 请详细描述这张图片的内容。
    images:
      - temp/screenshot.png
generation:
  model: gemini-2.5-flash
  format: markdown
```

#### 2. 多张图片对比

```yaml
messages:
  - user: 对比这两张图片，说明它们的异同。
    images:
      - images/photo1.jpg
      - images/photo2.jpg
generation:
  model: gemini-2.5-flash
```

#### 3. 历史对话 + 图片

```yaml
messages:
  - system: 你是一个图像分析助手。
  - user: 你好。
  - assistant: 你好！我可以帮你分析图片。
  - user: 这张图片里有什么？
    images:
      - temp/image.png
generation:
  model: gemini-2.5-flash
```

## Python API 使用

```python
from llm_gemini_api import LLMClient, load_env_file

load_env_file()
client = LLMClient.from_env()

yaml_prompt = """
messages:
  - user: 请描述这张图片。
    images:
      - path/to/image.png
generation:
  model: gemini-2.5-flash
"""

# 调用 API
result = client.invoke_from_yaml(yaml_prompt)
print(result)

# Dry run（查看 payload 结构，但仍会上传文件）
debug_info = client.invoke_from_yaml(yaml_prompt, dry_run=True)
print(debug_info)
```

## 技术实现

### 工作流程

1. **解析阶段**（parser.py）
   - 识别 `images` 字段
   - 解析图片路径列表
   - 构建 MessageEntry 对象

2. **构建阶段**（builder.py）
   - 将消息转换为 ICSMessage 格式
   - 标记多模态内容：`{"type": "image", "path": "..."}`

3. **适配阶段**（adapter.py）
   - 调用 GeminiFileUploader 上传图片
   - 获取 file_uri 和 mime_type
   - 转换为 Gemini parts 格式：`{"file_data": {"file_uri": "...", "mime_type": "..."}}`

4. **发送阶段**（client.py）
   - 处理文本和 file_data parts
   - 调用 Gemini SDK 生成内容

### Files API

使用 google.genai 的 Files API 上传图片：

```python
# 在 file_utils.py 中
uploaded_file = client.files.upload(file="path/to/image.png")
# 返回: File(uri="https://...", mime_type="image/png", name="files/...")
```

上传的文件会自动获得 URI，用于后续的生成请求。

## 支持的图片格式

Gemini Files API 支持常见图片格式：
- PNG (.png)
- JPEG (.jpg, .jpeg)
- GIF (.gif)
- WebP (.webp)

## 注意事项

1. **文件必须存在**：图片路径必须指向实际存在的本地文件
2. **API 配额**：每次上传会消耗 Files API 配额
3. **Dry run 限制**：即使 dry_run=True，图片仍会被上传到 Files API（但不会调用生成 API）
4. **文件生命周期**：上传的文件在 Gemini 服务器上有一定的保留期限

## 错误处理

常见错误及解决方案：

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `FileNotFoundError` | 图片文件不存在 | 检查图片路径是否正确 |
| `文件上传失败` | Files API 错误 | 检查网络连接和 API Key |
| `多模态消息需要 file_uploader` | 内部错误 | 确保使用最新版本的 llm_gemini_api |

## 测试示例

参见项目中的测试文件：
- `test_multimodal_gemini.py` - 完整的多模态测试
- `test_run_gemini.py` - 基本用法示例

## 相关文档

- [Gemini Image Understanding](https://ai.google.dev/gemini-api/docs/image-understanding)
- [Gemini Files API](https://ai.google.dev/gemini-api/docs/files)
