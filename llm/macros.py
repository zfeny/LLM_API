"""Prompt macro renderer for preset templating."""
from __future__ import annotations

import logging
import random
import re
from typing import List, Optional


logger = logging.getLogger(__name__)

# Public API -----------------------------------------------------------------


def render_macros(text: str) -> str:
    """Expand supported macros (random/roll) inside the given text."""
    if not text or "{{" not in text:
        return text

    return _replace_macros(text)


# Internal helpers -----------------------------------------------------------

def _replace_macros(text: str) -> str:
    result: List[str] = []
    index = 0
    length = len(text)

    while index < length:
        start = text.find("{{", index)
        if start == -1:
            result.append(text[index:])
            break

        result.append(text[index:start])
        end = _find_macro_end(text, start)
        if end is None:
            # 未找到匹配的结束符，直接附加剩余文本
            result.append(text[start:])
            break

        inner = text[start + 2 : end - 2]
        inner_stripped = inner.strip()
        replacement = _evaluate_macro(inner_stripped)

        if replacement is None:
            # 未识别的宏，按原样返回
            logger.warning("未识别的宏指令: {{%s}}", inner_stripped)
            result.append(text[start:end])
        else:
            if "{{" in replacement:
                replacement = _replace_macros(replacement)
            result.append(replacement)

        index = end

    return "".join(result)


def _find_macro_end(text: str, start: int) -> Optional[int]:
    """Locate the index **after** the matching ``}}`` for a macro."""
    depth = 0
    i = start
    while i < len(text) - 1:
        token = text[i : i + 2]
        if token == "{{":
            depth += 1
            i += 2
            continue
        if token == "}}":
            depth -= 1
            i += 2
            if depth == 0:
                return i
            continue
        i += 1
    return None


def _evaluate_macro(inner: str) -> Optional[str]:
    lowered = inner.lower()
    if lowered.startswith("random"):
        return _handle_random(inner)
    if lowered.startswith("roll"):
        return _handle_roll(inner)
    return None


def _handle_random(inner: str) -> Optional[str]:
    rest = inner[6:].lstrip()  # len("random") == 6
    if rest.startswith("::"):
        body = rest[2:]
    elif rest.startswith(":"):
        body = rest[1:]
    else:
        return None

    options = _split_options(body)
    if not options:
        return ""
    return random.choice(options)


def _split_options(body: str) -> List[str]:
    options: List[str] = []
    current: List[str] = []
    i = 0
    depth = 0

    while i < len(body):
        if body.startswith("{{", i):
            depth += 1
            current.append("{{")
            i += 2
            continue
        if body.startswith("}}", i) and depth > 0:
            depth -= 1
            current.append("}}")
            i += 2
            continue

        if depth == 0:
            if body.startswith("::", i):
                options.append("".join(current).strip())
                current = []
                i += 2
                continue
            if body[i] == ",":
                options.append("".join(current).strip())
                current = []
                i += 1
                continue

        current.append(body[i])
        i += 1

    options.append("".join(current).strip())
    return [opt for opt in options if opt]


_ROLL_RE = re.compile(r"^\s*(\d*)d(\d+)\s*$", re.IGNORECASE)


def _handle_roll(inner: str) -> Optional[str]:
    rest = inner[4:].strip()  # len("roll") == 4
    match = _ROLL_RE.match(rest)
    if not match:
        return None

    count = int(match.group(1) or "1")
    sides = int(match.group(2))

    if count <= 0 or sides <= 0:
        return None

    total = sum(random.randint(1, sides) for _ in range(count))
    return str(total)


__all__ = ["render_macros"]
