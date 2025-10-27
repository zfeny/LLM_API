"""数据模型定义。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
