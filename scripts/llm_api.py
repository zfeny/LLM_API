"""极简的 YAML→ICS→OpenAI 封装模块。"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - 依赖检查
    raise ImportError("需要 openai SDK，请先执行 pip install openai") from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover - 依赖检查
    raise ImportError("需要 PyYAML 依赖，请先执行 pip install pyyaml") from exc


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


class LLMConfigError(RuntimeError):
    """当环境配置缺失或不合法时抛出。"""


class LLMValidationError(ValueError):
    """当上层 YAML 输入不符合要求时抛出。"""


class LLMTransportError(RuntimeError):
    """当下游 OpenAI 兼容接口返回异常时抛出。"""


def load_env_file(path: str | os.PathLike[str] = ".env") -> None:
    """从 .env 文件中加载键值对到 ``os.environ``。"""

    env_path = Path(path)
    if not env_path.exists():
        logger.debug("未找到 .env 文件 %s，跳过加载", env_path)
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#"):
            continue
        if "=" not in line:
            logger.debug("跳过无法解析的 .env 行：%s", line)
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


class UsageRecorder:
    """将 usage 数据落盘到 SQLite。"""

    def __init__(self, db_path: str | os.PathLike[str] | None = None) -> None:
        default_path = Path(os.environ.get("LLM_USAGE_DB", "usage_log.db"))
        self._db_path = Path(db_path or default_path)
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model TEXT,
                    request_id TEXT,
                    trace_id TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER
                )
                """
            )
            conn.commit()

    def record(
        self,
        *,
        model: Optional[str],
        request_id: Optional[str],
        trace_id: Optional[str],
        usage: Optional[Dict[str, Any]],
    ) -> None:
        if not usage:
            return

        timestamp = datetime.utcnow().isoformat()
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO usage_log (
                    timestamp, model, request_id, trace_id,
                    prompt_tokens, completion_tokens, total_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    model,
                    request_id,
                    trace_id,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                ),
            )
            conn.commit()


USAGE_RECORDER = UsageRecorder()


@dataclass(slots=True)
class LLMAPIConfig:
    """封装运行期必要的最小环境变量。"""

    api_key: str
    base_url: str
    default_model: Optional[str]
    request_timeout: Optional[float]
    organization: Optional[str]

    @staticmethod
    def _require_env(key: str) -> str:
        value = os.environ.get(key)
        if value is None or not value.strip():
            raise LLMConfigError(f"缺少必要的环境变量：{key}")
        return value.strip()

    @classmethod
    def from_env(cls) -> "LLMAPIConfig":
        timeout_raw = os.environ.get("LLM_TIMEOUT")
        timeout_value: Optional[float]
        if timeout_raw and timeout_raw.strip():
            try:
                timeout_value = float(timeout_raw.strip())
            except ValueError as exc:  # pragma: no cover - 转换异常
                raise LLMConfigError("LLM_TIMEOUT 必须为浮点数") from exc
        else:
            timeout_value = None

        return cls(
            api_key=cls._require_env("LLM_API_KEY"),
            base_url=cls._require_env("LLM_API_BASE"),
            default_model=os.environ.get("LLM_MODEL"),
            request_timeout=timeout_value,
            organization=os.environ.get("LLM_ORG"),
        )


# ============================================================================
# ICS 协议数据类
# ============================================================================

@dataclass(slots=True)
class ICSMessage:
    """表示 ICS 协议中的单条消息。
    
    ICS (Internal Communication Standard) 是内部通信标准格式。
    
    属性:
        role: 消息角色("system" 或 "user")
        content: 消息内容
    """

    role: str      # 消息角色:"system"(系统提示)或"user"(用户输入)
    content: str   # 消息的具体文本内容

    def to_payload(self) -> Dict[str, str]:
        """将消息对象转换为字典格式。
        
        Returns:
            Dict[str, str]: 包含 role 和 content 的字典
        
        示例:
            message = ICSMessage(role="user", content="Hello")
            message.to_payload()  # {"role": "user", "content": "Hello"}
        """
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class ICSRequest:
    """聚合消息、生成参数、路由和元信息。
    
    这是一个完整的 ICS 请求对象,包含了所有必要的信息。
    
    属性:
        messages: 消息列表(ICSMessage 对象)
        generation: 生成参数(如 model、temperature、max_output_tokens)
        routing: 路由信息(如 policy、targets)
        meta: 元信息(如 trace_id、locale、client)
    """

    messages: List[ICSMessage]  # 消息列表(必需)
    # 以下字段使用 field(default_factory=dict) 创建默认空字典
    # 注意:不能直接用 = {},因为所有实例会共享同一个字典对象!
    generation: Dict[str, Any] = field(default_factory=dict)  # 生成参数
    routing: Dict[str, Any] = field(default_factory=dict)     # 路由信息
    meta: Dict[str, Any] = field(default_factory=dict)        # 元信息
    format_config: Optional[Dict[str, Any]] = None            # 输出格式

    def to_payload(self) -> Dict[str, Any]:
        """将请求对象转换为字典格式。
        
        Returns:
            Dict[str, Any]: 包含所有字段的字典
        
        注意:
            messages 字段会调用每个 ICSMessage 的 to_payload() 方法
        """
        return {
            # 使用列表推导式,将每个 ICSMessage 对象转换为字典
            "messages": [message.to_payload() for message in self.messages],
            "generation": self.generation,  # 生成参数字典
            "routing": self.routing,        # 路由信息字典
            "meta": self.meta,              # 元信息字典
            "format": self.format_config,
        }


# ============================================================================
# YAML 解析器类
# ============================================================================

class YAMLRequestParser:
    """解析并校验上层 YAML 输入,保留未显式提供的可选段。"""

    REQUIRED_FIELDS = ("system", "user")
    OPTIONAL_SECTIONS = ("routing", "meta")
    FORMAT_TYPES = {"text", "markdown", "json", "json_schema"}

    @staticmethod
    def parse(raw_yaml: str) -> Dict[str, Any]:
        try:
            payload = yaml.safe_load(raw_yaml) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - 库内部已测试
            raise LLMValidationError(f"YAML 解析失败：{exc}") from exc

        if not isinstance(payload, dict) or "messages" not in payload:
            raise LLMValidationError("YAML 顶层必须包含 'messages' 映射")

        messages = payload["messages"]
        if not isinstance(messages, dict):
            raise LLMValidationError("'messages' 必须是包含 system/user 的映射")

        missing = [field for field in YAMLRequestParser.REQUIRED_FIELDS if field not in messages]
        if missing:
            raise LLMValidationError(f"缺少必要字段：{', '.join(missing)}")

        normalized_messages: Dict[str, str] = {}
        for field in YAMLRequestParser.REQUIRED_FIELDS:
            value = messages[field]
            if not isinstance(value, str) or not value.strip():
                raise LLMValidationError(f"messages.{field} 必须为非空字符串")
            normalized_messages[field] = value.strip()

        result: Dict[str, Any] = {"messages": normalized_messages}

        generation_value = payload.get("generation")
        format_config = None
        if generation_value is not None:
            if not isinstance(generation_value, dict):
                raise LLMValidationError("generation 必须为字典")
            generation_copy = dict(generation_value)
            format_raw = generation_copy.pop("format", None)
            if format_raw is not None:
                format_config = YAMLRequestParser._parse_format(format_raw)
            if generation_copy:
                result["generation"] = generation_copy
        for section in YAMLRequestParser.OPTIONAL_SECTIONS:
            section_value = payload.get(section)
            if section_value is None:
                continue
            if not isinstance(section_value, dict):
                raise LLMValidationError(f"{section} 必须为字典")
            result[section] = section_value

        if format_config is not None:
            result["format"] = format_config

        return result

    @staticmethod
    def _parse_format(raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            raise LLMValidationError("format 必须为字典")

        fmt_type_raw = raw.get("type", "text")
        if not isinstance(fmt_type_raw, str):
            raise LLMValidationError("format.type 必须为字符串")
        fmt_type = fmt_type_raw.strip().lower()
        if fmt_type not in YAMLRequestParser.FORMAT_TYPES:
            raise LLMValidationError(
                f"format.type 仅支持 {', '.join(sorted(YAMLRequestParser.FORMAT_TYPES))}"
            )

        parsed: Dict[str, Any] = {"type": fmt_type}

        if fmt_type == "json_schema":
            schema = raw.get("schema")
            schema_name = raw.get("name") or raw.get("schema_name")
            if not schema_name or not isinstance(schema_name, str):
                raise LLMValidationError("json_schema 格式需要 name/schema_name 字段")
            if not isinstance(schema, dict):
                raise LLMValidationError("json_schema 格式需要 schema 字段且必须为字典")
            parsed["name"] = schema_name
            parsed["schema"] = schema

        return parsed


# ============================================================================
# ICS 构建器类
# ============================================================================

class FormatInstructionBuilder:
    """根据 format 配置生成附加指令。"""

    @staticmethod
    def build_messages(format_config: Optional[Dict[str, Any]]) -> List[ICSMessage]:
        if not format_config:
            return []

        fmt_type = format_config.get("type")

        if fmt_type == "markdown":
            content = "请使用 Markdown 格式组织答案，包含必要的标题或列表，避免输出纯文本。"
        elif fmt_type == "json":
            content = "请严格输出合法 JSON，对象中不要包含额外说明文字或代码块标记。"
        elif fmt_type == "json_schema":
            schema = format_config.get("schema")
            schema_text = json.dumps(schema, ensure_ascii=False) if schema else ""
            content = "请严格按照以下 JSON Schema 返回结构化数据，字段必须齐全："
            if schema_text:
                content += f" {schema_text}"
        else:
            return []

        return [ICSMessage(role="system", content=content)]

    @staticmethod
    def response_format_payload(format_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not format_config:
            return None

        fmt_type = format_config.get("type")
        if fmt_type == "json":
            return {"type": "json_object"}
        if fmt_type == "json_schema":
            schema = format_config.get("schema")
            name = format_config.get("name")
            if not schema or not name:
                return None
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": schema,
                },
            }
        return None


class FormatResponseHandler:
    """根据格式配置校验并规范化返回值。"""

    @staticmethod
    def process(value: Any, format_config: Optional[Dict[str, Any]]) -> Any:
        if not format_config:
            return value

        fmt_type = format_config.get("type")
        if fmt_type == "json":
            return FormatResponseHandler._ensure_json(value)
        if fmt_type == "markdown":
            return FormatResponseHandler._ensure_markdown(value)
        if fmt_type == "json_schema":
            normalized = FormatResponseHandler._ensure_json(value)
            FormatResponseHandler._validate_schema(
                normalized, format_config.get("schema")
            )
            return normalized
        return value

    @staticmethod
    def _ensure_json(value: Any) -> Any:
        candidate = FormatResponseHandler._convert_model_dump(value)
        if isinstance(candidate, (dict, list)):
            return candidate
        if isinstance(candidate, str):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise LLMValidationError("LLM 返回内容不是合法 JSON") from exc
        raise LLMValidationError("LLM 返回内容不是 JSON 格式")

    @staticmethod
    def _ensure_markdown(value: Any) -> str:
        candidate = FormatResponseHandler._convert_model_dump(value)
        if isinstance(candidate, str):
            stripped = candidate.strip()
            if not stripped:
                raise LLMValidationError("LLM 返回内容为空，无法作为 Markdown 输出")
            return stripped
        if candidate is None:
            raise LLMValidationError("LLM 没有返回任何内容")
        return str(candidate)

    @staticmethod
    def _validate_schema(data: Any, schema: Optional[Dict[str, Any]]) -> None:
        if not isinstance(data, dict):
            raise LLMValidationError("JSON Schema 校验需要对象类型返回值")
        if not schema:
            return
        required = schema.get("required", [])
        if isinstance(required, list):
            missing = [field for field in required if field not in data]
            if missing:
                raise LLMValidationError(f"返回结果缺少必需字段：{', '.join(missing)}")

    @staticmethod
    def _convert_model_dump(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return value


class ICSBuilder:
    """负责组合解析结果并生成 ICS 结构。"""

    def __init__(self, config: LLMAPIConfig):
        self._config = config

    def build(self, parsed: Dict[str, Any]) -> ICSRequest:
        base_messages: Dict[str, str] = parsed["messages"]
        system_message = ICSMessage(role="system", content=base_messages["system"])
        user_message = ICSMessage(role="user", content=base_messages["user"])

        generation = dict(parsed.get("generation", {}))
        model = generation.get("model") or self._config.default_model
        if not model:
            raise LLMConfigError("未在 YAML 或环境变量中提供模型信息")
        generation["model"] = model

        routing = dict(parsed.get("routing", {}))
        meta = dict(parsed.get("meta", {}))
        meta.setdefault("trace_id", str(uuid.uuid4()))

        format_config = parsed.get("format")
        extra_messages = FormatInstructionBuilder.build_messages(format_config)

        return ICSRequest(
            messages=[system_message, user_message, *extra_messages],
            generation=generation,
            routing=routing,
            meta=meta,
            format_config=format_config,
        )


# ============================================================================
# OpenAI 适配器类
# ============================================================================

class OpenAIAdapter:
    """将 ICS 结构转换为 OpenAI 兼容请求体。
    
    这个类负责将内部的 ICS 格式转换为 OpenAI API 所期望的格式。
    主要是字段名称的映射和格式调整。
    """

    @staticmethod
    def to_chat_completion(ics_request: ICSRequest) -> Dict[str, Any]:
        """将 ICS 请求转换为 OpenAI Chat Completion API 格式。
        
        Args:
            ics_request: ICS 请求对象
        
        Returns:
            Dict[str, Any]: OpenAI API 格式的字典
        
        字段映射:
            ICS                  → OpenAI
            max_output_tokens    → max_tokens
            (其他字段名相同)
        """
        generation = ics_request.generation

        payload: Dict[str, Any] = {
            "model": generation.get("model"),
            "messages": [message.to_payload() for message in ics_request.messages],
        }

        if not payload["model"]:
            raise LLMConfigError("缺少模型参数，无法构建请求")

        for key, value in generation.items():
            if key == "model":
                continue
            if key == "max_output_tokens":
                payload["max_tokens"] = value
                continue
            payload[key] = value

        response_format = FormatInstructionBuilder.response_format_payload(
            ics_request.format_config
        )
        if response_format:
            payload["response_format"] = response_format

        return payload


# ============================================================================
# LLM 客户端类(主入口)
# ============================================================================

class LLMClient:
    """负责串联解析、构建 ICS 与下游请求。
    
    这是整个模块的主入口类,负责协调所有组件完成完整的工作流程:
    YAML → ICS → OpenAI → 发送请求 → 返回结果
    
    属性:
        _config: LLM API 配置对象
        _builder: ICS 构建器对象
    """

    def __init__(self, config: LLMAPIConfig):
        """初始化 LLM 客户端。
        
        Args:
            config: LLM API 配置对象
        """
        self._config = config  # 保存配置
        self._builder = ICSBuilder(config)  # 创建 ICS 构建器
        self._openai_client = self._create_openai_client()

    @classmethod
    def from_env(cls) -> "LLMClient":
        """从环境变量创建客户端(工厂方法)。
        
        Returns:
            LLMClient: 客户端实例
        
        使用示例:
            client = LLMClient.from_env()
        """
        return cls(LLMAPIConfig.from_env())

    def invoke_from_yaml(
        self,
        yaml_prompt: str,
        *,
        dry_run: bool = False,
        include_debug: bool = False,
    ) -> Dict[str, Any]:
        """从 YAML 格式的提示词调用 LLM API。
        
        Args:
            yaml_prompt: YAML 格式的提示词字符串
            dry_run: 是否为模拟运行(不实际发送请求),默认为 False
        
        Returns:
            Dict[str, Any]: 包含以下字段的字典
                - ics_request: ICS 请求格式
                - openai_request: OpenAI API 请求格式
                - response: API 响应(仅在 dry_run=False 时)
        
        Raises:
            LLMValidationError: YAML 格式不正确
            LLMTransportError: 网络请求失败
        
        工作流程:
            1. 解析 YAML → 基础消息
            2. 构建 ICS 请求
            3. 转换为 OpenAI 格式
            4. 发送请求(如果 dry_run=False)
            5. 返回结果
        """
        logger.debug("开始处理 YAML 输入，dry_run=%s", dry_run)
        
        # 步骤 1: 解析 YAML
        parsed_payload = YAMLRequestParser.parse(yaml_prompt)
        
        # 步骤 2: 构建 ICS 请求
        ics_request = self._builder.build(parsed_payload)
        
        # 步骤 3: 转换为 OpenAI 格式
        openai_payload = OpenAIAdapter.to_chat_completion(ics_request)

        # 如果是模拟运行,只返回构建的请求,不发送
        if dry_run:
            return {
                "ics_request": ics_request.to_payload(),
                "openai_request": openai_payload,
            }

        # 步骤 4: 实际发送请求
        response = self._send_openai_request(
            openai_payload,
            trace_id=ics_request.meta.get("trace_id"),
        )

        result_value = self._extract_result(response, ics_request.format_config)

        if not include_debug:
            return result_value

        return {
            "result": result_value,
            "ics_request": ics_request.to_payload(),
            "openai_request": openai_payload,
            "response": response.model_dump() if hasattr(response, "model_dump") else response,
        }

    def _create_openai_client(self) -> OpenAI:
        """构建 OpenAI SDK 客户端。"""

        client_kwargs: Dict[str, Any] = {
            "api_key": self._config.api_key,
            "base_url": self._config.base_url,
        }
        if self._config.organization:
            client_kwargs["organization"] = self._config.organization
        if self._config.request_timeout is not None:
            client_kwargs["timeout"] = self._config.request_timeout

        return OpenAI(**client_kwargs)

    def _send_openai_request(
        self,
        payload: Dict[str, Any],
        *,
        trace_id: Optional[str],
    ) -> Any:
        """通过 SDK 发送请求。"""

        try:
            response = self._openai_client.chat.completions.create(**payload)
        except Exception as exc:  # pragma: no cover - SDK 内部包含多种异常
            raise LLMTransportError(f"OpenAI SDK 调用失败：{exc}") from exc

        self._record_usage(response, trace_id)
        return response

    def _extract_result(
        self,
        response: Any,
        format_config: Optional[Dict[str, Any]],
    ) -> Any:
        """根据响应和格式配置提取核心结果。"""

        choice = None
        try:
            choice = response.choices[0]
        except (AttributeError, IndexError, KeyError, TypeError):
            if hasattr(response, "choices") and isinstance(response.choices, list) and response.choices:
                choice = response.choices[0]

        if choice is None:
            raw_response = response.model_dump() if hasattr(response, "model_dump") else response
            return FormatResponseHandler.process(raw_response, format_config)

        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        if message is None:
            return FormatResponseHandler.process(choice, format_config)

        parsed = getattr(message, "parsed", None)
        if parsed is None and isinstance(message, dict):
            parsed = message.get("parsed")

        if parsed is not None:
            normalized = parsed
        else:
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")

            if isinstance(content, list):
                text_parts: List[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = "".join(text_parts)

            normalized = content

        return FormatResponseHandler.process(normalized, format_config)

    def _record_usage(self, response: Any, trace_id: Optional[str]) -> None:
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None and isinstance(response, dict):
            usage_obj = response.get("usage")

        if usage_obj is None:
            usage_dict = None
        elif hasattr(usage_obj, "model_dump"):
            usage_dict = usage_obj.model_dump()
        elif isinstance(usage_obj, dict):
            usage_dict = usage_obj
        else:
            usage_dict = None

        response_id = getattr(response, "id", None)
        if response_id is None and isinstance(response, dict):
            response_id = response.get("id")

        response_model = getattr(response, "model", None)
        if response_model is None and isinstance(response, dict):
            response_model = response.get("model")

        USAGE_RECORDER.record(
            model=response_model,
            request_id=response_id,
            trace_id=trace_id,
            usage=usage_dict,
        )


# ============================================================================
# 导出的公共接口
# ============================================================================

__all__ = [
    "LLMClient",           # 主客户端类
    "LLMAPIConfig",        # 配置类
    "LLMConfigError",      # 配置错误异常
    "LLMTransportError",   # 传输错误异常
    "LLMValidationError",  # 验证错误异常
    "load_env_file",       # 环境变量加载函数
]
