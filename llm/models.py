"""Common data models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass(slots=True)
class ICSMessage:
    """ICS message supporting text and multimodal content."""

    role: str
    content: Union[str, List[Dict[str, Any]]]

    def to_payload(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class MessageEntry:
    """Intermediate message representation parsed from YAML."""

    role: str
    content: str
    images: Optional[Dict[str, Any]] = None
    source: Optional[str] = None


@dataclass(slots=True)
class ICSRequest:
    """Full ICS request."""

    messages: List[ICSMessage]
    generation: Dict[str, Any] = field(default_factory=dict)
    routing: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    format_config: Optional[Dict[str, Any]] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "messages": [m.to_payload() for m in self.messages],
            "generation": self.generation,
            "routing": self.routing,
            "meta": self.meta,
            "format": self.format_config,
        }


__all__ = ["ICSMessage", "MessageEntry", "ICSRequest"]
