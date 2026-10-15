"""Common generation result contract for local and API backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union


@dataclass
class GenerationResult:
    """Backend output split into auditable raw text and scorer-visible text."""

    final_text: str
    raw_text: str
    thinking_trace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def split_thinking_trace(raw_text: str) -> Tuple[Optional[str], str]:
    """Separate a Qwen-style ``<think>`` prefix from the final response.

    An unclosed thinking block is retained as trace and yields an empty final
    response.  It must become a schema failure rather than being scored as if
    hidden reasoning were the requested JSON.
    """
    text = str(raw_text or "")
    start = len(text) - len(text.lstrip())
    if not text.startswith("<think>", start):
        return None, text
    end = text.find("</think>", start)
    if end < 0:
        return text, ""
    end += len("</think>")
    trace = text[:end]
    final = text[end:].lstrip()
    return trace, final


def result_from_raw(raw_text: str, separate_thinking: bool = False,
                    metadata: Optional[Dict[str, Any]] = None) -> GenerationResult:
    """Build a structured result from decoded model text."""
    raw = str(raw_text or "")
    if separate_thinking:
        trace, final = split_thinking_trace(raw)
    else:
        trace, final = None, raw
    return GenerationResult(final_text=final, raw_text=raw,
                            thinking_trace=trace,
                            metadata=dict(metadata or {}))


def normalize_generation_result(
        value: Union[str, GenerationResult]) -> GenerationResult:
    """Accept a plain string from a backend and wrap it as a GenerationResult."""
    if isinstance(value, GenerationResult):
        return value
    if isinstance(value, str):
        return result_from_raw(value)
    raise TypeError(
        "backend must return str or GenerationResult, got "
        f"{type(value).__name__}")
