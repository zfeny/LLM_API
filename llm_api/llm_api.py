"""极简 YAML→ICS→OpenAI 封装。"""
from __future__ import annotations
import atexit, functools, json, logging, os, random, sqlite3, threading, time, uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError as e:
    raise ImportError("需要 openai SDK: pip install openai") from e
try:
    import yaml
except ImportError as e:
    raise ImportError("需要 PyYAML: pip install pyyaml") from e
try:
    from dotenv import load_dotenv as _dotenv_load
except ImportError:
    _dotenv_load = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

T = TypeVar("T")
_MISSING = object()

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """统一获取对象属性或字典值，保留合法假值。"""
    if obj is None:
        return default
    attr_val = getattr(obj, key, _MISSING)
    if attr_val is not _MISSING:
        return attr_val
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default

@dataclass
class RetryConfig:
    """重试配置。"""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

def _retry(config: RetryConfig, is_async: bool = False):
    """统一重试装饰器（支持同步和异步）。"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                import asyncio
                last_exc = None
                for attempt in range(config.max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:
                        last_exc = exc
                        if attempt >= config.max_retries:
                            break
                        delay = min(config.initial_delay * (config.exponential_base ** attempt), config.max_delay)
                        if config.jitter:
                            delay *= 0.5 + random.random()
                        logger.warning("请求失败 (%d/%d)，%.1fs 后重试: %s", attempt + 1, config.max_retries + 1, delay, exc)
                        await asyncio.sleep(delay)
                raise last_exc
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                last_exc = None
                for attempt in range(config.max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as exc:
                        last_exc = exc
                        if attempt >= config.max_retries:
                            break
                        delay = min(config.initial_delay * (config.exponential_base ** attempt), config.max_delay)
                        if config.jitter:
                            delay *= 0.5 + random.random()
                        logger.warning("请求失败 (%d/%d)，%.1fs 后重试: %s", attempt + 1, config.max_retries + 1, delay, exc)
                        time.sleep(delay)
                raise last_exc
            return sync_wrapper
    return decorator

class LLMConfigError(RuntimeError):
    """配置错误。"""

class LLMValidationError(ValueError):
    """输入验证错误。"""

class LLMTransportError(RuntimeError):
    """传输错误。"""

def load_env_file(path: str | os.PathLike[str] = ".env") -> None:
    """加载 .env 文件。"""
    env_path = Path(path)
    if not env_path.exists():
        return
    if _dotenv_load:
        _dotenv_load(dotenv_path=env_path, override=False)
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)

class UsageRecorder:
    """批量 SQLite usage 记录器。"""
    def __init__(self, db_path: str | os.PathLike[str] | None = None, batch_size: int = 10, auto_flush: bool = True):
        self._db_path = Path(db_path or os.environ.get("LLM_USAGE_DB", "usage_log.db"))
        self._batch_size = batch_size
        self._buffer = []
        self._lock = threading.Lock()
        self._ensure_table()
        if auto_flush:
            atexit.register(self.flush)

    def _ensure_table(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, model TEXT, request_id TEXT, trace_id TEXT,
                    prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER
                )
            """)
            conn.commit()

    def record(self, *, model: Optional[str], request_id: Optional[str], trace_id: Optional[str], usage: Optional[Dict[str, Any]]):
        if not usage:
            return
        data = (datetime.utcnow().isoformat(), model, request_id, trace_id,
                usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens"))
        with self._lock:
            self._buffer.append(data)
            if len(self._buffer) >= self._batch_size:
                self._flush()

    def flush(self):
        with self._lock:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.executemany("INSERT INTO usage_log (timestamp, model, request_id, trace_id, prompt_tokens, completion_tokens, total_tokens) VALUES (?, ?, ?, ?, ?, ?, ?)", self._buffer)
                conn.commit()
            self._buffer.clear()
        except Exception as exc:
            logger.error("写入 usage 失败: %s", exc)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.flush()

@dataclass(slots=True)
class LLMAPIConfig:
    """运行配置。"""
    api_key: str
    base_url: str
    default_model: Optional[str]
    request_timeout: Optional[float]
    organization: Optional[str]

    @classmethod
    def from_env(cls):
        def req(k):
            v = os.environ.get(k)
            if not v or not v.strip():
                raise LLMConfigError(f"缺少环境变量: {k}")
            return v.strip()
        timeout_raw = os.environ.get("LLM_TIMEOUT")
        timeout = float(timeout_raw.strip()) if timeout_raw and timeout_raw.strip() else None
        return cls(api_key=req("LLM_API_KEY"), base_url=req("LLM_API_BASE"),
                   default_model=os.environ.get("LLM_MODEL"), request_timeout=timeout,
                   organization=os.environ.get("LLM_ORG"))

@dataclass(slots=True)
class ICSMessage:
    """ICS 消息。"""
    role: str
    content: str
    def to_payload(self):
        return {"role": self.role, "content": self.content}

@dataclass(slots=True)
class MessageEntry:
    """解析阶段内部消息表示，减少中间字典。"""
    role: str
    content: str

@dataclass(slots=True)
class ICSRequest:
    """ICS 请求。"""
    messages: List[ICSMessage]
    generation: Dict[str, Any] = field(default_factory=dict)
    routing: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    format_config: Optional[Dict[str, Any]] = None

    def to_payload(self):
        return {
            "messages": [m.to_payload() for m in self.messages],
            "generation": self.generation,
            "routing": self.routing,
            "meta": self.meta,
            "format": self.format_config,
        }

class YAMLRequestParser:
    """YAML 解析器。"""
    REQUIRED = ("user",)  # 只有 user 必填
    OPTIONAL = ("routing", "meta")
    FORMATS = {"text", "markdown", "json", "json_schema"}
    MESSAGE_ROLES = {"system", "user", "assistant"}

    @staticmethod
    def parse(raw: str):
        try:
            data = yaml.safe_load(raw) or {}
        except yaml.YAMLError as e:
            raise LLMValidationError(f"YAML 解析失败: {e}") from e
        if not isinstance(data, dict) or "messages" not in data:
            raise LLMValidationError("YAML 顶层必须包含 'messages'")
        message_entries = YAMLRequestParser._normalize_messages(data["messages"])
        result = {"messages": message_entries}

        gen = data.get("generation")
        fmt = None
        if gen and isinstance(gen, dict):
            gen_copy = dict(gen)
            fmt_raw = gen_copy.pop("format", None)
            if fmt_raw:
                fmt = YAMLRequestParser._parse_fmt(fmt_raw)
            if gen_copy:
                result["generation"] = gen_copy

        for sec in YAMLRequestParser.OPTIONAL:
            val = data.get(sec)
            if val and isinstance(val, dict):
                result[sec] = val

        if fmt:
            result["format"] = fmt
        return result

    @staticmethod
    def _parse_fmt(raw):
        if not isinstance(raw, dict):
            raise LLMValidationError("format 必须为字典")
        fmt_type_raw = raw.get("type")
        if fmt_type_raw is None:
            fmt_type = "text"
        elif isinstance(fmt_type_raw, str):
            fmt_type = fmt_type_raw.strip().lower() or "text"
        else:
            raise LLMValidationError("format.type 必须为字符串")
        if fmt_type not in YAMLRequestParser.FORMATS:
            raise LLMValidationError(f"format.type 仅支持: {', '.join(sorted(YAMLRequestParser.FORMATS))}")
        parsed = {"type": fmt_type}
        if fmt_type == "json_schema":
            name = raw.get("name") or raw.get("schema_name")
            schema = raw.get("schema")
            if not name or not isinstance(name, str):
                raise LLMValidationError("json_schema 需要 name")
            if not isinstance(schema, dict):
                raise LLMValidationError("json_schema 需要 schema 字典")
            parsed["name"] = name
            parsed["schema"] = schema
        return parsed

    @staticmethod
    def _normalize_messages(raw_msgs: Any) -> List[MessageEntry]:
        if isinstance(raw_msgs, list):
            entries = YAMLRequestParser._normalize_messages_from_list(raw_msgs)
        elif isinstance(raw_msgs, dict):
            entries = YAMLRequestParser._normalize_messages_from_dict(raw_msgs)
        else:
            raise LLMValidationError("messages 必须是列表或字典")
        user_present = any(entry.role == "user" for entry in entries)
        if not user_present:
            raise LLMValidationError(f"缺少必填字段: {', '.join(YAMLRequestParser.REQUIRED)}")
        return entries

    @staticmethod
    def _normalize_messages_from_list(raw_list: List[Any]) -> List[MessageEntry]:
        entries: List[MessageEntry] = []
        for idx, item in enumerate(raw_list):
            role, content = YAMLRequestParser._extract_role_content(item)
            entries.append(MessageEntry(role=role, content=content))
        return entries

    @staticmethod
    def _normalize_messages_from_dict(raw_dict: Dict[str, Any]) -> List[MessageEntry]:
        entries: List[MessageEntry] = []
        for raw_key, raw_val in raw_dict.items():
            role = YAMLRequestParser._normalize_role(raw_key)
            values = raw_val if isinstance(raw_val, list) else [raw_val]
            for val in values:
                if not isinstance(val, str):
                    raise LLMValidationError("messages.* 的值必须为字符串或字符串列表")
                content = val.strip()
                if role in ("user", "assistant"):
                    if not content:
                        raise LLMValidationError(f"{role} 必须为非空字符串")
                elif not content:
                    # system 允许为空，跳过
                    continue
                entries.append(MessageEntry(role=role, content=content))
        return entries

    @staticmethod
    def _extract_role_content(item: Any) -> tuple[str, str]:
        if isinstance(item, MessageEntry):
            return item.role, item.content
        if isinstance(item, dict):
            if "role" in item and "content" in item:
                role = YAMLRequestParser._normalize_role(item["role"])
                content = item["content"]
            elif len(item) == 1:
                raw_role, content = next(iter(item.items()))
                role = YAMLRequestParser._normalize_role(raw_role)
            else:
                raise LLMValidationError("messages 列表项需包含 role/content 或单键角色")
        else:
            raise LLMValidationError("messages 列表项必须为对象")
        if not isinstance(content, str):
            raise LLMValidationError("消息内容必须为字符串")
        content = content.strip()
        if role in ("user", "assistant"):
            if not content:
                raise LLMValidationError(f"{role} 必须为非空字符串")
        elif not content:
            # system 允许为空，保持空字符串
            content = ""
        return role, content

    @staticmethod
    def _normalize_role(raw_role: Any) -> str:
        token = str(raw_role).strip()
        if "." in token:
            prefix, suffix = token.split(".", 1)
            if prefix.strip().isdigit():
                token = suffix.strip()
        role = token.lower()
        if role not in YAMLRequestParser.MESSAGE_ROLES:
            raise LLMValidationError("messages 仅支持 system/user/assistant 角色")
        return role

class FormatHandler:
    """格式处理器（指令构建 + 响应校验）。"""
    _FORMAT_MESSAGE_CACHE: Dict[str, Tuple[ICSMessage, ...]] = {}

    @staticmethod
    def build_messages(cfg: Optional[Dict[str, Any]]) -> List[ICSMessage]:
        if not cfg:
            return []
        t = cfg.get("type")
        cache_key = FormatHandler._cache_key(cfg)
        if cache_key:
            cached = FormatHandler._FORMAT_MESSAGE_CACHE.get(cache_key)
            if cached is not None:
                return list(cached)
        system_content = None
        if t == "markdown":
            system_content = None
        elif t == "json":
            system_content = None
        elif t == "json_schema":
            schema_text = json.dumps(cfg.get("schema"), ensure_ascii=False) if cfg.get("schema") else ""
            if schema_text:
                system_content = None
        user_content = None
        if t == "markdown":
            user_content = "请确保回应使用 Markdown 的标题或列表组织内容，避免纯文本。"
        elif t == "json":
            user_content = "请仅输出合法 JSON，不要附加任何说明或代码块。"
        elif t == "json_schema":
            schema_text = json.dumps(cfg.get("schema"), ensure_ascii=False) if cfg.get("schema") else ""
            if schema_text:
                user_content = f"请严格按照以下 JSON Schema 返回完整字段: {schema_text}"
        messages: List[ICSMessage] = []
        if system_content:
            messages.append(ICSMessage(role="system", content=system_content))
        if user_content:
            messages.append(ICSMessage(role="user", content=user_content))
        if cache_key:
            FormatHandler._FORMAT_MESSAGE_CACHE[cache_key] = tuple(messages)
        return messages

    @staticmethod
    def response_format(cfg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not cfg:
            return None
        t = cfg.get("type")
        if t == "json":
            return {"type": "json_object"}
        if t == "json_schema":
            return {"type": "json_schema", "json_schema": {"name": cfg.get("name"), "schema": cfg.get("schema")}}
        return None

    @staticmethod
    def process(value: Any, cfg: Optional[Dict[str, Any]]) -> Any:
        if not cfg:
            return value
        t = cfg.get("type")
        if t in ("json", "json_schema"):
            return FormatHandler._to_json(value, cfg.get("schema") if t == "json_schema" else None)
        if t == "markdown":
            return FormatHandler._to_md(value)
        return value

    @staticmethod
    def _to_json(val, schema=None):
        v = val.model_dump() if hasattr(val, "model_dump") else val
        if isinstance(v, (dict, list)):
            data = v
        elif isinstance(v, str):
            try:
                data = json.loads(v)
            except json.JSONDecodeError as e:
                raise LLMValidationError("返回内容不是合法 JSON") from e
        else:
            raise LLMValidationError("返回内容不是 JSON 格式")
        if schema and isinstance(data, dict):
            required = schema.get("required", [])
            missing = [f for f in required if f not in data]
            if missing:
                raise LLMValidationError(f"缺少必需字段: {', '.join(missing)}")
        return data

    @staticmethod
    def _to_md(val):
        v = val.model_dump() if hasattr(val, "model_dump") else val
        if isinstance(v, str):
            s = v.strip()
            if not s:
                raise LLMValidationError("返回内容为空")
            return s
        if v is None:
            raise LLMValidationError("无返回内容")
        return str(v)

    @staticmethod
    def _cache_key(cfg: Dict[str, Any]) -> Optional[str]:
        try:
            fmt_type = cfg.get("type")
        except AttributeError:
            return None
        if not fmt_type:
            return None
        if fmt_type == "json_schema":
            schema = cfg.get("schema")
            try:
                schema_repr = json.dumps(schema, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                schema_repr = repr(schema)
            name = cfg.get("name") or ""
            return f"json_schema::{name}::{schema_repr}"
        return str(fmt_type)

class ICSBuilder:
    """ICS 构建器。"""
    def __init__(self, config: LLMAPIConfig):
        self._config = config

    def build(self, parsed: Dict[str, Any]) -> ICSRequest:
        msgs = parsed["messages"]
        base_msgs = self._build_message_chain(msgs)

        gen = dict(parsed.get("generation", {}))
        model = gen.get("model") or self._config.default_model
        if not model:
            raise LLMConfigError("未提供模型")
        gen["model"] = model

        routing = dict(parsed.get("routing", {}))
        meta = dict(parsed.get("meta", {}))
        meta.setdefault("trace_id", str(uuid.uuid4()))

        fmt = parsed.get("format")
        extra = FormatHandler.build_messages(fmt)
        system_extra = [m for m in extra if m.role == "system"]
        other_extra = [m for m in extra if m.role != "system"]

        messages = list(base_msgs)
        if system_extra:
            insert_at = next((idx for idx, msg in enumerate(messages) if msg.role != "system"), len(messages))
            messages[insert_at:insert_at] = system_extra
        if other_extra:
            messages.extend(other_extra)

        return ICSRequest(messages=messages, generation=gen, routing=routing, meta=meta, format_config=fmt)

    def _build_message_chain(self, data: Any) -> List[ICSMessage]:
        if isinstance(data, list):
            return self._build_from_entries(data)
        if isinstance(data, dict):
            return self._build_from_legacy_dict(data)
        raise LLMValidationError("messages 结构无效")

    def _build_from_entries(self, entries: List[Dict[str, Any]]) -> List[ICSMessage]:
        messages: List[ICSMessage] = []
        for entry in entries:
            if isinstance(entry, MessageEntry):
                messages.append(ICSMessage(role=entry.role, content=entry.content))
                continue
            role = entry.get("role")
            content = entry.get("content")
            if role not in YAMLRequestParser.MESSAGE_ROLES:
                raise LLMValidationError(f"不支持的消息角色: {role}")
            if not isinstance(content, str):
                raise LLMValidationError("消息内容必须为字符串")
            messages.append(ICSMessage(role=role, content=content))
        return messages

    def _build_from_legacy_dict(self, msgs: Dict[str, Any]) -> List[ICSMessage]:
        base_msgs: List[ICSMessage] = []
        system_msg = msgs.get("system")
        if isinstance(system_msg, str) and system_msg.strip():
            base_msgs.append(ICSMessage(role="system", content=system_msg.strip()))
        user_msg = msgs.get("user")
        if not isinstance(user_msg, str) or not user_msg.strip():
            raise LLMValidationError("user 必须为非空字符串")
        base_msgs.append(ICSMessage(role="user", content=user_msg.strip()))
        return base_msgs

class OpenAIAdapter:
    """OpenAI 适配器。"""
    @staticmethod
    def to_chat(ics: ICSRequest) -> Dict[str, Any]:
        gen = ics.generation
        payload = {"model": gen.get("model"), "messages": [m.to_payload() for m in ics.messages]}
        if not payload["model"]:
            raise LLMConfigError("缺少模型参数")
        for k, v in gen.items():
            if k == "model":
                continue
            if k == "max_output_tokens":
                payload["max_tokens"] = v
            else:
                payload[k] = v
        fmt = FormatHandler.response_format(ics.format_config)
        if fmt:
            payload["response_format"] = fmt
        return payload

class _BaseLLMClient:
    """LLM 客户端基类（提取公共方法）。"""
    def __init__(self, config: LLMAPIConfig, recorder: Optional[UsageRecorder], retry_config: Optional[RetryConfig]):
        self._config = config
        self._builder = ICSBuilder(config)
        self._recorder = recorder or UsageRecorder()
        self._retry_config = retry_config or RetryConfig()

    def _extract_result(self, resp: Any, fmt_cfg: Optional[Dict[str, Any]]) -> Any:
        choice = None
        try:
            choice = resp.choices[0]
        except:
            choices = _get(resp, "choices")
            if isinstance(choices, list) and choices:
                choice = choices[0]
        if not choice:
            return FormatHandler.process(resp.model_dump() if hasattr(resp, "model_dump") else resp, fmt_cfg)

        msg = _get(choice, "message")
        if not msg:
            return FormatHandler.process(choice, fmt_cfg)

        parsed = _get(msg, "parsed")
        if parsed is not None:
            return FormatHandler.process(parsed, fmt_cfg)

        content = _get(msg, "content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        return FormatHandler.process(content, fmt_cfg)

    def _record_usage(self, resp: Any, trace_id: Optional[str]):
        usage_obj = _get(resp, "usage")
        usage_dict = None
        if usage_obj:
            if hasattr(usage_obj, "model_dump"):
                usage_dict = usage_obj.model_dump()
            elif isinstance(usage_obj, dict):
                usage_dict = usage_obj
        self._recorder.record(model=_get(resp, "model"), request_id=_get(resp, "id"), trace_id=trace_id, usage=usage_dict)

class LLMClient(_BaseLLMClient):
    """同步 LLM 客户端。"""
    def __init__(self, config: LLMAPIConfig, recorder: Optional[UsageRecorder] = None, retry_config: Optional[RetryConfig] = None):
        super().__init__(config, recorder, retry_config)
        self._openai_client = self._create_openai_client()

    @classmethod
    def from_env(cls, recorder: Optional[UsageRecorder] = None, retry_config: Optional[RetryConfig] = None):
        return cls(LLMAPIConfig.from_env(), recorder, retry_config)

    def invoke_from_yaml(self, yaml_prompt: str, *, dry_run: bool = False, include_debug: bool = False) -> Union[Any, Dict[str, Any]]:
        start = time.time()
        trace_id = str(uuid.uuid4())
        logger.info("处理请求 trace_id=%s dry_run=%s", trace_id, dry_run)

        parsed = YAMLRequestParser.parse(yaml_prompt)
        ics = self._builder.build(parsed)
        if "trace_id" not in ics.meta:
            ics.meta["trace_id"] = trace_id
        openai_payload = OpenAIAdapter.to_chat(ics)

        if dry_run:
            return {"ics_request": ics.to_payload(), "openai_request": openai_payload}

        resp = self._send(openai_payload, ics.meta.get("trace_id"))
        logger.info("完成 trace_id=%s 耗时=%.2fs", trace_id, time.time() - start)

        result = self._extract_result(resp, ics.format_config)
        if not include_debug:
            return result
        return {"result": result, "ics_request": ics.to_payload(), "openai_request": openai_payload, "response": resp.model_dump() if hasattr(resp, "model_dump") else resp}

    def _create_openai_client(self):
        kwargs = {"api_key": self._config.api_key, "base_url": self._config.base_url}
        if self._config.organization:
            kwargs["organization"] = self._config.organization
        if self._config.request_timeout is not None:
            kwargs["timeout"] = self._config.request_timeout
        return OpenAI(**kwargs)

    def _send(self, payload: Dict[str, Any], trace_id: Optional[str]):
        @_retry(self._retry_config, is_async=False)
        def _call():
            return self._openai_client.chat.completions.create(**payload)
        resp = _call()
        self._record_usage(resp, trace_id)
        return resp

class AsyncLLMClient(_BaseLLMClient):
    """异步 LLM 客户端。"""
    def __init__(self, config: LLMAPIConfig, recorder: Optional[UsageRecorder] = None, retry_config: Optional[RetryConfig] = None):
        super().__init__(config, recorder, retry_config)
        self._openai_client = self._create_openai_client()

    @classmethod
    def from_env(cls, recorder: Optional[UsageRecorder] = None, retry_config: Optional[RetryConfig] = None):
        return cls(LLMAPIConfig.from_env(), recorder, retry_config)

    async def invoke_from_yaml(self, yaml_prompt: str, *, dry_run: bool = False, include_debug: bool = False) -> Union[Any, Dict[str, Any]]:
        start = time.time()
        trace_id = str(uuid.uuid4())
        logger.info("异步处理 trace_id=%s dry_run=%s", trace_id, dry_run)

        parsed = YAMLRequestParser.parse(yaml_prompt)
        ics = self._builder.build(parsed)
        if "trace_id" not in ics.meta:
            ics.meta["trace_id"] = trace_id
        openai_payload = OpenAIAdapter.to_chat(ics)

        if dry_run:
            return {"ics_request": ics.to_payload(), "openai_request": openai_payload}

        resp = await self._send(openai_payload, ics.meta.get("trace_id"))
        logger.info("异步完成 trace_id=%s 耗时=%.2fs", trace_id, time.time() - start)

        result = self._extract_result(resp, ics.format_config)
        if not include_debug:
            return result
        return {"result": result, "ics_request": ics.to_payload(), "openai_request": openai_payload, "response": resp.model_dump() if hasattr(resp, "model_dump") else resp}

    def _create_openai_client(self):
        kwargs = {"api_key": self._config.api_key, "base_url": self._config.base_url}
        if self._config.organization:
            kwargs["organization"] = self._config.organization
        if self._config.request_timeout is not None:
            kwargs["timeout"] = self._config.request_timeout
        return AsyncOpenAI(**kwargs)

    async def _send(self, payload: Dict[str, Any], trace_id: Optional[str]):
        @_retry(self._retry_config, is_async=True)
        async def _call():
            return await self._openai_client.chat.completions.create(**payload)
        resp = await _call()
        self._record_usage(resp, trace_id)
        return resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self._openai_client.close()

__all__ = ["LLMClient", "AsyncLLMClient", "LLMAPIConfig", "LLMConfigError", "LLMTransportError", "LLMValidationError", "UsageRecorder", "RetryConfig", "load_env_file"]
