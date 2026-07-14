from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..tokens import TokenUsage


@dataclass
class RuntimeResult:
    output: str
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    runtime_meta: dict[str, Any] = field(default_factory=dict)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
