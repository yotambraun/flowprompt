"""A deterministic stand-in for LLM calls, for tests, CI and demos.

``FakeLLM`` replaces ``litellm.completion`` / ``litellm.acompletion`` while it
is active, so any code that goes through ``Prompt.run()``, ``arun()``,
``stream()`` or ``compare()`` runs offline, without API keys, and
deterministically:

    >>> from flowprompt.testing import FakeLLM
    >>> with FakeLLM(lambda messages: "positive") as fake:
    ...     Sentiment(text="I love it").run(model="gpt-4o-mini")
    'positive'
    >>> len(fake.calls)
    1

When the prompt declares an ``Output`` model and the responder returns
``None`` (the default responder does), a minimal JSON object matching the
schema is generated. Token usage is reported (about four characters per
token), so cost accounting works for models with a known price.
"""

from __future__ import annotations

import inspect
import json
import threading
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

__all__ = ["FakeLLM"]

Responder = Callable[..., Any]


def _example_from_schema(
    schema: dict[str, Any], defs: dict[str, Any] | None = None
) -> Any:
    """Build a minimal value that validates against a JSON schema."""
    defs = defs if defs is not None else schema.get("$defs", {})
    if "$ref" in schema:
        ref = schema["$ref"].split("/")[-1]
        return _example_from_schema(defs.get(ref, {}), defs)
    if "default" in schema:
        return schema["default"]
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            options = [o for o in schema[key] if o.get("type") != "null"]
            return _example_from_schema(options[0] if options else {}, defs)
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        props = schema.get("properties", {})
        return {name: _example_from_schema(sub, defs) for name, sub in props.items()}
    if kind == "array":
        return []
    if kind == "integer":
        low = schema.get("minimum", schema.get("exclusiveMinimum", 0))
        return int(low) + (1 if "exclusiveMinimum" in schema else 0)
    if kind == "number":
        low = schema.get("minimum", 0.0)
        high = schema.get("maximum", low + 1.0)
        return (float(low) + float(high)) / 2
    if kind == "boolean":
        return True
    if kind == "null":
        return None
    return "example"


def _schema_from_request(
    messages: list[dict[str, Any]], response_format: Any
) -> dict[str, Any] | None:
    if isinstance(response_format, dict):
        if response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", {}).get("schema")
            if isinstance(schema, dict):
                return schema
        if response_format.get("type") == "json_object":
            marker = "matching this schema:\n"
            for message in messages:
                content = str(message.get("content", ""))
                if marker in content:
                    try:
                        parsed = json.loads(content.split(marker, 1)[1])
                        return parsed if isinstance(parsed, dict) else None
                    except json.JSONDecodeError:
                        return None
    return None


def _accepts_two_args(fn: Callable[..., Any]) -> bool:
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return True
    positional = [
        p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


class FakeLLM:
    """Context manager that answers LLM calls locally.

    Args:
        responder: Produces the reply. Either a string (always returned), or
            a callable receiving the list of chat messages (and, if it accepts
            a second argument, the full request as a dict, e.g. to check
            ``request["model"]`` or ``request["response_format"]``) and
            returning a string, a dict / pydantic model (serialised to JSON),
            or None to use the default reply. Defaults to echoing the last
            user message, or schema-shaped JSON for structured prompts.
        latency_s: Optional artificial delay per call, in seconds.

    Attributes:
        calls: The keyword arguments of every call made while active.
    """

    def __init__(
        self,
        responder: Responder | str | None = None,
        *,
        latency_s: float = 0.0,
    ) -> None:
        self._responder = responder
        self._wants_request = callable(responder) and _accepts_two_args(responder)
        self._latency_s = latency_s
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._saved: dict[str, Any] = {}

    # -- reply construction -------------------------------------------------

    def _reply(self, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> str:
        value: Any = None
        if isinstance(self._responder, str):
            value = self._responder
        elif callable(self._responder):
            if self._wants_request:
                value = self._responder(messages, kwargs)  # type: ignore[call-arg]
            else:
                value = self._responder(messages)
        if value is None:
            schema = _schema_from_request(messages, kwargs.get("response_format"))
            if schema is not None:
                value = _example_from_schema(schema)
            else:
                last_user = next(
                    (
                        str(m.get("content", ""))
                        for m in reversed(messages)
                        if m.get("role") == "user"
                    ),
                    "",
                )
                value = f"Fake reply to: {last_user[:80]}"
        if hasattr(value, "model_dump_json"):
            return str(value.model_dump_json())
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)

    def _response(self, model: str, messages: list[dict[str, Any]], text: str) -> Any:
        prompt_tokens = _tokens(" ".join(str(m.get("content", "")) for m in messages))
        completion_tokens = _tokens(text)
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        message = SimpleNamespace(role="assistant", content=text)
        choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
        return SimpleNamespace(model=model, choices=[choice], usage=usage)

    def _stream(self, text: str) -> Iterator[Any]:
        words = text.split(" ")
        for i, word in enumerate(words):
            delta = SimpleNamespace(content=word if i == 0 else " " + word)
            last = i == len(words) - 1
            choice = SimpleNamespace(
                delta=delta, finish_reason="stop" if last else None
            )
            yield SimpleNamespace(choices=[choice], usage=None)

    def _handle(self, kwargs: dict[str, Any]) -> tuple[Any, str]:
        messages = list(kwargs.get("messages") or [])
        with self._lock:
            self.calls.append(dict(kwargs))
        text = self._reply(messages, kwargs)
        return self._response(str(kwargs.get("model", "")), messages, text), text

    def completion(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        if self._latency_s:
            import time

            time.sleep(self._latency_s)
        response, text = self._handle(kwargs)
        if kwargs.get("stream"):
            return self._stream(text)
        return response

    async def acompletion(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        if self._latency_s:
            import asyncio

            await asyncio.sleep(self._latency_s)
        response, text = self._handle(kwargs)
        if kwargs.get("stream"):

            async def agen() -> Any:
                for chunk in self._stream(text):
                    yield chunk

            return agen()
        return response

    # -- activation ---------------------------------------------------------

    def __enter__(self) -> FakeLLM:
        import litellm

        self._saved = {
            "completion": litellm.completion,
            "acompletion": litellm.acompletion,
        }
        litellm.completion = self.completion
        litellm.acompletion = self.acompletion
        return self

    def __exit__(self, *exc: Any) -> None:
        import litellm

        litellm.completion = self._saved["completion"]
        litellm.acompletion = self._saved["acompletion"]
