"""数据模型定义。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass(slots=True)
class ICSMessage:
    """ICS 消息。支持文本和多模态内容。"""
    role: str
    content: Union[str, List[Dict[str, Any]]]  # 可以是字符串或多模态内容列表

    def to_payload(self):
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class MessageEntry:
    """解析阶段内部消息表示，减少中间字典。"""
    role: str
    content: str
    images: Optional[Dict[str, Any]] = None  # 图片信息：{urls: [...], contents: [...]}
    source: Optional[str] = None  # 来源信息：preset名称或"custom"


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
