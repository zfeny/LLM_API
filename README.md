# LLM API 封装使用说明

## 项目简介
`scripts/llm_api.py` 提供了一个极简的 YAML → ICS → OpenAI 请求封装层：

- 上层以 YAML 描述提示词与生成配置；
- 中间层负责解析、默认值填充、可选格式控制；
- 底层通过 OpenAI SDK 调用任意兼容接口（例如 Gemini OpenAI 兼容端点），并在本地记录 usage。

## 环境准备
1. 克隆/下载项目后，安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 复制 `.env.example` 为 `.env`，并填写至少以下变量：
   ```ini
   LLM_API_KEY=你的密钥
   LLM_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai/
   # 可选：LLM_MODEL、LLM_TIMEOUT、LLM_ORG、LLM_USAGE_DB
   ```

## YAML 输入规范
最小结构如下：

```yaml
messages:
  system: 你是一个严谨的助手。
  user: 请介绍 Python 的核心优势。
generation:
  model: gemini-2.5-flash
  format:
    type: markdown  # 可选：text / markdown / json / json_schema
routing:            # 可选，保留给未来的多通道路由
  policy: default   # 当前仅作为占位，值会透传到 ICS/OpenAI 侧，暂不参与逻辑
meta:               # 可选
  trace_id: 手动指定追踪 ID
```

- `messages.system` / `messages.user` 必填且为字符串。
- `generation.model` 可在 YAML 中指定；若缺失会回退到环境变量 `LLM_MODEL`。
- `generation.format` 支持：
  - `markdown`：自动附加 system 指令并校验返回是否为非空 Markdown 文本。
  - `json`：只要返回合法 JSON 即通过。
  - `json_schema`：需提供 `name` 与 `schema`，会校验必填字段是否存在。
- `routing`：预留给未来的多模型/多通道路由策略，当前版本不会基于此决定目标，只会把内容透传至构造的 ICS/OpenAI 请求，便于后续扩展（可用 `policy`、`targets` 等字段自行记录策略意图）。

## 快速体验
项目根目录自带 `test_run.py` ：

```bash
python test_run.py
```

默认会加载 `.env`、读取示例 YAML 并发起真实请求。若只想查看构造的请求，可显式传 `dry_run=True`：

```python
client.invoke_from_yaml(yaml_prompt, dry_run=True, include_debug=True)
```

`include_debug=True` 时返回值形如：

```json
{
  "result": {...},
  "ics_request": {...},
  "openai_request": {...},
  "response": {...}
}
```

否则默认直接返回符合格式要求的最终结果（字符串或 JSON 对象）。

## Usage 记录
真实调用会自动将 `model/request_id/trace_id/token` 信息写入 SQLite（默认 `usage_log.db`），路径可通过环境变量 `LLM_USAGE_DB` 自定义，便于后续统计与观察。

## 常见扩展
- 需要多模型路由、重试或更多日志，可在 `ICSBuilder` 与 `LLMClient` 中扩展逻辑。
- 更多 YAML 字段（如 `temperature`、`max_output_tokens`）可在 `generation` 内直接声明，封装层会自动透传至 OpenAI API。
