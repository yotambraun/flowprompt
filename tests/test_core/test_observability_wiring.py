"""Caching, tracing and cost tracking must be connected to Prompt.run()."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

import flowprompt.core.cache as cache_module
import flowprompt.tracing.otel as otel_module
from flowprompt import Prompt, configure_cache, configure_tracer


class Hello(Prompt[Any]):
    system: str = "Be brief."
    user: str = "Say hello to {name}."


class Structured(Prompt[Any]):
    system: str = "Extract."
    user: str = "From: {text}"

    class Output(BaseModel):
        name: str


def _response(content: str, prompt_tokens: int = 1000, completion_tokens: int = 500):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


@pytest.fixture(autouse=True)
def _reset_globals() -> Iterator[None]:
    saved_cache = cache_module._global_cache
    saved_tracer = otel_module._global_tracer
    saved_flag = getattr(cache_module, "_cache_configured", None)
    yield
    cache_module._global_cache = saved_cache
    otel_module._global_tracer = saved_tracer
    if saved_flag is not None:
        cache_module._cache_configured = saved_flag


def test_configured_cache_serves_identical_calls() -> None:
    cache = configure_cache(enabled=True, default_ttl=60)
    with patch("litellm.completion", return_value=_response("Hi Ada")) as llm:
        first = Hello(name="Ada").run(model="gpt-4o-mini")
        second = Hello(name="Ada").run(model="gpt-4o-mini")
    assert first == second == "Hi Ada"
    assert llm.call_count == 1
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 1


def test_cache_key_depends_on_inputs_and_model() -> None:
    configure_cache(enabled=True)
    with patch("litellm.completion", return_value=_response("x")) as llm:
        Hello(name="Ada").run(model="gpt-4o-mini")
        Hello(name="Bob").run(model="gpt-4o-mini")
        Hello(name="Ada").run(model="gpt-4o")
        Hello(name="Ada").run(model="gpt-4o-mini", temperature=0.7)
    assert llm.call_count == 4


def test_cache_returns_validated_structured_output() -> None:
    configure_cache(enabled=True)
    with patch("litellm.completion", return_value=_response('{"name": "Ada"}')) as llm:
        a = Structured(text="Ada").run(model="gpt-4o-mini")
        b = Structured(text="Ada").run(model="gpt-4o-mini")
    assert llm.call_count == 1
    assert a.name == b.name == "Ada"


def test_cache_is_off_unless_configured() -> None:
    cache_module._global_cache = None
    if hasattr(cache_module, "_cache_configured"):
        cache_module._cache_configured = False
    with patch("litellm.completion", return_value=_response("x")) as llm:
        Hello(name="Ada").run(model="gpt-4o-mini")
        Hello(name="Ada").run(model="gpt-4o-mini")
    assert llm.call_count == 2


def test_disabled_cache_does_not_serve() -> None:
    configure_cache(enabled=False)
    with patch("litellm.completion", return_value=_response("x")) as llm:
        Hello(name="Ada").run(model="gpt-4o-mini")
        Hello(name="Ada").run(model="gpt-4o-mini")
    assert llm.call_count == 2


def test_tracer_records_requests_tokens_and_cost() -> None:
    tracer = configure_tracer(service_name="test")
    with patch("litellm.completion", return_value=_response("Hi")):
        Hello(name="Ada").run(model="gpt-4o-mini")
        Hello(name="Bob").run(model="gpt-4o-mini")
    summary = tracer.get_summary()
    assert summary["total_requests"] == 2
    assert summary["total_tokens"] == 3000
    assert summary["total_cost_usd"] > 0
    span = tracer.get_spans()[0]
    assert span.name == "Hello"
    assert span.usage.model == "gpt-4o-mini"
    assert span.status == "ok"


def test_tracer_records_errors() -> None:
    tracer = configure_tracer(service_name="test")
    with (
        patch("litellm.completion", side_effect=RuntimeError("down")),
        pytest.raises(RuntimeError),
    ):
        Hello(name="Ada").run(model="gpt-4o-mini", max_retries=1)
    assert tracer.get_summary()["error_rate"] == 1.0


@pytest.mark.asyncio
async def test_async_run_uses_cache_and_tracer() -> None:
    configure_cache(enabled=True)
    tracer = configure_tracer(service_name="test")

    async def fake(**_: Any) -> Any:
        return _response("Hi")

    with patch("litellm.acompletion", side_effect=fake) as llm:
        await Hello(name="Ada").arun(model="gpt-4o-mini")
        await Hello(name="Ada").arun(model="gpt-4o-mini")
    assert llm.call_count == 1
    spans = tracer.get_spans()
    assert len(spans) == 2
    assert [s.usage.cached for s in spans] == [False, True]


def test_track_usage_records_tokens_and_cost_from_response() -> None:
    from flowprompt import track_usage
    from flowprompt.core.usage import estimate_cost

    with (
        patch("litellm.completion", return_value=_response("Hi")),
        track_usage() as calls,
    ):
        Hello(name="Ada").run(model="gpt-4o-mini")
    assert len(calls) == 1
    call = calls[0]
    assert (call.prompt_tokens, call.completion_tokens) == (1000, 500)
    expected = estimate_cost("gpt-4o-mini", 1000, 500)
    assert expected is not None and expected > 0
    assert call.cost_usd == pytest.approx(expected)
    assert call.cached is False


def test_track_usage_unknown_model_has_no_cost() -> None:
    from flowprompt import track_usage

    with (
        patch("litellm.completion", return_value=_response("Hi")),
        track_usage() as calls,
    ):
        Hello(name="Ada").run(model="my-local-model-xyz")
    assert calls[0].cost_usd is None
    assert calls[0].total_tokens == 1500


def test_track_usage_counts_retries_and_cache_hits() -> None:
    from flowprompt import track_usage

    configure_cache(enabled=True)
    responses = [_response("not json"), _response('{"name": "Ada"}')]
    with (
        patch("litellm.completion", side_effect=responses),
        track_usage() as calls,
    ):
        Structured(text="Ada").run(model="gpt-4o-mini")
        Structured(text="Ada").run(model="gpt-4o-mini")
    # Two billed attempts (first reply failed validation), then a cache hit.
    assert [c.cached for c in calls] == [False, False, True]
    assert calls[2].cost_usd == 0.0


def test_track_usage_nests_and_isolates() -> None:
    from flowprompt import track_usage

    with patch("litellm.completion", return_value=_response("Hi")):
        with track_usage() as outer:
            Hello(name="A").run(model="gpt-4o-mini")
            with track_usage() as inner:
                Hello(name="B").run(model="gpt-4o-mini")
        Hello(name="C").run(model="gpt-4o-mini")
    assert len(outer) == 2
    assert len(inner) == 1
