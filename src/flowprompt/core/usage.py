"""Token usage and cost accounting for LLM calls.

Every call made through ``Prompt.run()`` / ``Prompt.arun()`` records a
:class:`CallUsage` (tokens, cost, latency, cache hit) to any active
:func:`track_usage` collector. ``compare()`` uses this to report real cost
per variant; you can use it directly:

    >>> from flowprompt import track_usage
    >>> with track_usage() as calls:
    ...     MyPrompt(text="hi").run(model="gpt-4o-mini")
    >>> sum(c.cost_usd or 0 for c in calls)

Collectors are stored in a :class:`contextvars.ContextVar`, so they are
isolated per thread and per asyncio task and can be nested.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

__all__ = ["CallUsage", "estimate_cost", "track_usage", "usage_from_response"]


@dataclass
class CallUsage:
    """Usage of one LLM call.

    Attributes:
        model: Model identifier passed to the provider.
        prompt_tokens: Input tokens reported by the provider.
        completion_tokens: Output tokens reported by the provider.
        cost_usd: Cost in USD, or None when no price is known for the model.
            Cache hits cost 0.
        latency_ms: Wall-clock latency of the call.
        cached: True when the response came from the FlowPrompt cache.
    """

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: float = 0.0
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


_collectors: ContextVar[tuple[list[CallUsage], ...]] = ContextVar(
    "flowprompt_usage_collectors", default=()
)


@contextmanager
def track_usage() -> Iterator[list[CallUsage]]:
    """Collect the usage of every LLM call made inside the block."""
    records: list[CallUsage] = []
    token = _collectors.set((*_collectors.get(), records))
    try:
        yield records
    finally:
        _collectors.reset(token)


def record_usage(usage: CallUsage) -> None:
    """Append ``usage`` to every active collector."""
    for collector in _collectors.get():
        collector.append(usage)


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def estimate_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Cost of a call from litellm's model price map, or None if unknown."""
    try:
        import litellm

        info = litellm.model_cost.get(model)
        if info is None and "/" in model:
            info = litellm.model_cost.get(model.split("/", 1)[1])
    except Exception:
        return None
    if not info:
        return None
    in_price = info.get("input_cost_per_token")
    out_price = info.get("output_cost_per_token")
    if in_price is None and out_price is None:
        return None
    return prompt_tokens * float(in_price or 0.0) + completion_tokens * float(
        out_price or 0.0
    )


def usage_from_response(
    response: Any, model: str, latency_ms: float = 0.0
) -> CallUsage:
    """Build a CallUsage from a litellm (OpenAI-style) response object."""
    usage = getattr(response, "usage", None)
    if isinstance(usage, dict):
        prompt_tokens = _as_int(usage.get("prompt_tokens"))
        completion_tokens = _as_int(usage.get("completion_tokens"))
    else:
        prompt_tokens = _as_int(getattr(usage, "prompt_tokens", 0))
        completion_tokens = _as_int(getattr(usage, "completion_tokens", 0))

    cost: float | None = None
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        reported = hidden.get("response_cost")
        if isinstance(reported, (int, float)) and not isinstance(reported, bool):
            cost = float(reported)
    if cost is None and (prompt_tokens or completion_tokens):
        cost = estimate_cost(model, prompt_tokens, completion_tokens)

    return CallUsage(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
        latency_ms=latency_ms,
    )
