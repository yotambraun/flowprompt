"""OpenTelemetry integration for FlowPrompt observability.

Provides automatic tracing of prompt executions with:
- Span creation for each prompt run
- Token usage tracking
- Cost estimation
- Latency measurement
- Error tracking
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Fallback pricing per 1M tokens, used only when litellm.model_cost is unavailable
_FALLBACK_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
}


@dataclass
class UsageInfo:
    """Token usage and cost information.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens used.
        cost_usd: Estimated cost in USD.
        model: Model used for the request.
        cached: Whether the response was cached.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    cached: bool = False

    def calculate_cost(self, model: str) -> float:
        """Calculate cost based on model pricing.

        Tries litellm.model_cost for up-to-date per-token pricing first,
        then falls back to the built-in _FALLBACK_PRICING table.
        """
        # Try litellm's maintained pricing database first
        try:
            import litellm

            cost_info = litellm.model_cost.get(model)
            if cost_info is not None:
                input_cost = self.prompt_tokens * cost_info.get(
                    "input_cost_per_token", 0.0
                )
                output_cost = self.completion_tokens * cost_info.get(
                    "output_cost_per_token", 0.0
                )
                return input_cost + output_cost
        except Exception:
            pass

        # Fallback to built-in pricing table
        model_key = model.split("/")[-1].lower()

        pricing = None
        for key, prices in _FALLBACK_PRICING.items():
            if key in model_key or model_key in key:
                pricing = prices
                break

        if pricing is None:
            return 0.0

        input_cost = (self.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


@dataclass
class SpanContext:
    """Context for a traced prompt execution.

    Attributes:
        name: Name of the span (usually prompt class name).
        start_time: When the span started.
        end_time: When the span ended.
        duration_ms: Duration in milliseconds.
        status: Status of the execution (success/error).
        usage: Token usage information.
        attributes: Additional span attributes.
        error: Error information if failed.
    """

    name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float | None = None
    status: str = "unset"
    usage: UsageInfo = field(default_factory=UsageInfo)
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def finish(self, status: str = "ok", error: str | None = None) -> None:
        """Mark the span as finished."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        self.error = error

    def set_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        cached: bool = False,
    ) -> None:
        """Set token usage information."""
        self.usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=model,
            cached=cached,
        )
        self.usage.cost_usd = self.usage.calculate_cost(model)


class Tracer:
    """FlowPrompt tracer for observability.

    Collects spans from prompt executions and exports them
    to configured backends (OpenTelemetry, Langfuse, etc.).

    Example:
        >>> tracer = Tracer()
        >>> with tracer.span("MyPrompt") as span:
        ...     result = prompt.run(model="gpt-4o")
        ...     span.set_usage(100, 50, "gpt-4o")
        >>> print(tracer.get_summary())
    """

    def __init__(
        self,
        service_name: str = "flowprompt",
        enabled: bool = True,
    ) -> None:
        """Initialize the tracer.

        Args:
            service_name: Name of the service for tracing.
            enabled: Whether tracing is enabled.
        """
        self._service_name = service_name
        self._enabled = enabled
        self._spans: list[SpanContext] = []
        self._otel_tracer: Any = None

        # Try to initialize OpenTelemetry
        self._init_otel()

    def _init_otel(self) -> None:
        """Initialize OpenTelemetry if available."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider

            # Only set up if not already configured
            if not isinstance(trace.get_tracer_provider(), TracerProvider):
                provider = TracerProvider()
                trace.set_tracer_provider(provider)

            self._otel_tracer = trace.get_tracer(self._service_name)
        except ImportError:
            self._otel_tracer = None

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[SpanContext]:
        """Create a traced span for a prompt execution.

        Args:
            name: Name of the span.
            attributes: Additional attributes to attach.

        Yields:
            SpanContext for tracking the execution.
        """
        if not self._enabled:
            yield SpanContext(name=name)
            return

        context = SpanContext(
            name=name,
            attributes=attributes or {},
        )

        # Also create OTel span if available
        otel_span = None
        if self._otel_tracer:
            otel_span = self._otel_tracer.start_span(name)
            if attributes:
                for k, v in attributes.items():
                    otel_span.set_attribute(k, str(v))

        try:
            yield context
            context.finish(status="ok")
        except Exception as e:
            context.finish(status="error", error=str(e))
            if otel_span:
                try:
                    from opentelemetry import trace

                    otel_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                except ImportError:
                    pass
            raise
        finally:
            self._spans.append(context)
            if otel_span:
                # Add usage info to OTel span
                if context.usage:
                    otel_span.set_attribute(
                        "llm.prompt_tokens", context.usage.prompt_tokens
                    )
                    otel_span.set_attribute(
                        "llm.completion_tokens", context.usage.completion_tokens
                    )
                    otel_span.set_attribute(
                        "llm.total_tokens", context.usage.total_tokens
                    )
                    otel_span.set_attribute("llm.cost_usd", context.usage.cost_usd)
                    otel_span.set_attribute("llm.model", context.usage.model)
                    otel_span.set_attribute("llm.cached", context.usage.cached)
                otel_span.set_attribute("duration_ms", context.duration_ms or 0)
                otel_span.end()

    def get_spans(self) -> list[SpanContext]:
        """Get all recorded spans."""
        return self._spans.copy()

    def clear_spans(self) -> None:
        """Clear all recorded spans."""
        self._spans.clear()

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all traced spans.

        Returns:
            Dictionary with aggregated statistics.
        """
        if not self._spans:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "avg_latency_ms": 0.0,
                "error_rate": 0.0,
            }

        total_tokens = sum(s.usage.total_tokens for s in self._spans)
        total_cost = sum(s.usage.cost_usd for s in self._spans)
        errors = sum(1 for s in self._spans if s.status == "error")
        latencies = [s.duration_ms for s in self._spans if s.duration_ms]

        return {
            "total_requests": len(self._spans),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "error_rate": errors / len(self._spans) if self._spans else 0.0,
            "by_model": self._group_by_model(),
        }

    def _group_by_model(self) -> dict[str, dict[str, Any]]:
        """Group statistics by model."""
        by_model: dict[str, dict[str, Any]] = {}
        for span in self._spans:
            model = span.usage.model or "unknown"
            if model not in by_model:
                by_model[model] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost_usd": 0.0,
                }
            by_model[model]["requests"] += 1
            by_model[model]["tokens"] += span.usage.total_tokens
            by_model[model]["cost_usd"] += span.usage.cost_usd
        return by_model


# Global tracer instance
_global_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def configure_tracer(
    service_name: str = "flowprompt",
    enabled: bool = True,
) -> Tracer:
    """Configure the global tracer."""
    global _global_tracer
    _global_tracer = Tracer(service_name=service_name, enabled=enabled)
    return _global_tracer
