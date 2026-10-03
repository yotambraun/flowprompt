"""LiteLLM provider for unified multi-provider LLM access."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from flowprompt.core.usage import CallUsage, record_usage, usage_from_response
from flowprompt.providers.base import BaseProvider

if TYPE_CHECKING:
    from flowprompt.core.prompt import Prompt

OutputT = TypeVar("OutputT", bound=BaseModel)


class LiteLLMProvider(BaseProvider):
    """Provider using LiteLLM for unified access to multiple LLM providers.

    LiteLLM supports 100+ LLM providers including:
    - OpenAI (gpt-4o, gpt-4o-mini, etc.)
    - Anthropic (claude-3-5-sonnet, claude-3-opus, etc.)
    - Google (gemini-2.0-flash, gemini-pro, etc.)
    - Local models via Ollama (llama3, mistral, etc.)
    - Azure OpenAI, AWS Bedrock, and many more

    Example:
        >>> provider = LiteLLMProvider()
        >>> result = provider.complete(
        ...     prompt=my_prompt,
        ...     model="gpt-4o",
        ...     temperature=0.0
        ... )
    """

    @staticmethod
    def _build_response_format(
        model: str,
        output_model: type[BaseModel],
        messages: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        """Build the response_format parameter for structured output.

        Uses native JSON schema mode when the model supports it,
        otherwise falls back to json_object mode with schema in the
        system message.

        Args:
            model: The model identifier.
            output_model: The Pydantic model class for the expected output.
            messages: The messages list (may be mutated for fallback path).

        Returns:
            The response_format dict to pass to litellm, or None.
        """
        import litellm

        schema = output_model.model_json_schema()

        # Try native JSON schema mode
        try:
            if litellm.supports_response_schema(model=model):
                schema_name = output_model.__name__
                return {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    },
                }
        except Exception:
            pass

        # Fallback: json_object mode with schema appended to system message
        schema_str = json.dumps(schema, indent=2)
        messages[0]["content"] += (
            f"\n\nRespond with valid JSON matching this schema:\n{schema_str}"
        )
        return {"type": "json_object"}

    # ------------------------------------------------------------------
    # Shared helpers for complete() / acomplete()
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None,
        kwargs: dict[str, Any],
    ) -> str:
        payload = json.dumps(
            {"messages": messages, "response_format": response_format, "kw": kwargs},
            sort_keys=True,
            default=repr,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _parse(content: str, output_model: type[BaseModel] | None) -> Any:
        if output_model is None:
            return content
        return output_model.model_validate(json.loads(content))

    @staticmethod
    def _finish_span(
        span: Any, model: str, usages: list[CallUsage], cached: bool
    ) -> None:
        if span is None:
            return
        span.set_usage(
            sum(u.prompt_tokens for u in usages),
            sum(u.completion_tokens for u in usages),
            model,
            cached=cached,
        )
        span.usage.cost_usd = sum(u.cost_usd or 0.0 for u in usages)

    def _prepare(
        self,
        prompt: Prompt[OutputT],
        model: str,
        temperature: float,
        max_tokens: int | None,
        kwargs: dict[str, Any],
    ) -> tuple[
        list[dict[str, str]],
        type[BaseModel] | None,
        dict[str, Any] | None,
        Any,
        str | None,
    ]:
        from flowprompt.core.cache import get_active_cache

        messages = prompt.to_messages()
        output_model = prompt.output_model
        response_format = None
        if output_model is not None:
            response_format = self._build_response_format(model, output_model, messages)

        cache = get_active_cache()
        key = None
        if cache is not None:
            key = self._cache_key(
                messages,
                response_format,
                {"max_tokens": max_tokens, "temperature": temperature, **kwargs},
            )
        return messages, output_model, response_format, cache, key

    @staticmethod
    def _from_cache(
        cache: Any,
        key: str | None,
        model: str,
        temperature: float,
        output_model: type[BaseModel] | None,
    ) -> tuple[bool, Any, CallUsage | None]:
        if cache is None or key is None:
            return False, None, None
        entry = cache.get(key, model, temperature)
        if entry is None:
            return False, None, None
        try:
            value = LiteLLMProvider._parse(entry.content, output_model)
        except (json.JSONDecodeError, ValidationError):
            cache.invalidate(key, model, temperature)
            return False, None, None
        usage = CallUsage(
            model=model,
            prompt_tokens=int(entry.metadata.get("prompt_tokens", 0)),
            completion_tokens=int(entry.metadata.get("completion_tokens", 0)),
            cost_usd=0.0,
            cached=True,
        )
        record_usage(usage)
        return True, value, usage

    @staticmethod
    def _store(
        cache: Any,
        key: str | None,
        model: str,
        temperature: float,
        content: str,
        usage: CallUsage,
    ) -> None:
        if cache is None or key is None:
            return
        cache.set(
            key,
            model,
            content,
            temperature,
            metadata={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            },
        )

    def complete(
        self,
        prompt: Prompt[OutputT],
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> OutputT | str:
        """Execute a synchronous completion request via LiteLLM.

        Uses the global cache when :func:`flowprompt.configure_cache` has been
        called, reports a span to the global tracer when one exists, and
        records token usage and cost for :func:`flowprompt.track_usage`.

        Args:
            prompt: The Prompt instance to execute.
            model: The model identifier (e.g., "gpt-4o", "anthropic/claude-3-5-sonnet").
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            max_retries: Number of retries on failure.
            **kwargs: Additional LiteLLM parameters.

        Returns:
            If prompt has an Output class, returns validated instance.
            Otherwise, returns raw string response.

        Raises:
            ImportError: If litellm is not installed.
            ValidationError: If response doesn't match Output schema.
        """
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "LiteLLM is required for this provider. "
                "Install it with: pip install flowprompt[all] or pip install litellm"
            ) from e

        from flowprompt.tracing.otel import get_active_tracer

        messages, output_model, response_format, cache, key = self._prepare(
            prompt, model, temperature, max_tokens, kwargs
        )
        tracer = get_active_tracer()
        span_cm = (
            tracer.span(type(prompt).__name__, {"model": model})
            if tracer is not None
            else nullcontext(None)
        )
        usages: list[CallUsage] = []
        with span_cm as span:
            hit, value, usage = self._from_cache(
                cache, key, model, temperature, output_model
            )
            if hit:
                self._finish_span(span, model, [usage] if usage else [], cached=True)
                return cast(OutputT, value) if output_model else value

            last_error: Exception | None = None
            for attempt in range(max_retries):
                try:
                    start = time.perf_counter()
                    response = litellm.completion(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                        response_format=response_format,
                        **kwargs,
                    )
                    call = usage_from_response(
                        response, model, (time.perf_counter() - start) * 1000
                    )
                    usages.append(call)
                    record_usage(call)
                    self._finish_span(span, model, usages, cached=False)

                    content = response.choices[0].message.content or ""
                    try:
                        value = self._parse(content, output_model)
                    except (json.JSONDecodeError, ValidationError) as e:
                        if attempt < max_retries - 1:
                            last_error = e
                            continue
                        raise
                    self._store(cache, key, model, temperature, content, call)
                    return cast(OutputT, value) if output_model else value

                except Exception as e:
                    last_error = e
                    if attempt >= max_retries - 1:
                        raise

            raise last_error or RuntimeError("Unexpected error in completion")

    async def acomplete(
        self,
        prompt: Prompt[OutputT],
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> OutputT | str:
        """Execute an asynchronous completion request via LiteLLM.

        Same caching, tracing and usage accounting as :meth:`complete`.

        Args:
            prompt: The Prompt instance to execute.
            model: The model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            max_retries: Number of retries on failure.
            **kwargs: Additional LiteLLM parameters.

        Returns:
            If prompt has an Output class, returns validated instance.
            Otherwise, returns raw string response.
        """
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "LiteLLM is required for this provider. "
                "Install it with: pip install flowprompt[all] or pip install litellm"
            ) from e

        from flowprompt.tracing.otel import get_active_tracer

        messages, output_model, response_format, cache, key = self._prepare(
            prompt, model, temperature, max_tokens, kwargs
        )
        tracer = get_active_tracer()
        span_cm = (
            tracer.span(type(prompt).__name__, {"model": model})
            if tracer is not None
            else nullcontext(None)
        )
        usages: list[CallUsage] = []
        with span_cm as span:
            hit, value, usage = self._from_cache(
                cache, key, model, temperature, output_model
            )
            if hit:
                self._finish_span(span, model, [usage] if usage else [], cached=True)
                return cast(OutputT, value) if output_model else value

            last_error: Exception | None = None
            for attempt in range(max_retries):
                try:
                    start = time.perf_counter()
                    response = await litellm.acompletion(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                        response_format=response_format,
                        **kwargs,
                    )
                    call = usage_from_response(
                        response, model, (time.perf_counter() - start) * 1000
                    )
                    usages.append(call)
                    record_usage(call)
                    self._finish_span(span, model, usages, cached=False)

                    content = response.choices[0].message.content or ""
                    try:
                        value = self._parse(content, output_model)
                    except (json.JSONDecodeError, ValidationError) as e:
                        if attempt < max_retries - 1:
                            last_error = e
                            continue
                        raise
                    self._store(cache, key, model, temperature, content, call)
                    return cast(OutputT, value) if output_model else value

                except Exception as e:
                    last_error = e
                    if attempt >= max_retries - 1:
                        raise

            raise last_error or RuntimeError("Unexpected error in async completion")

    def stream(
        self,
        prompt: Prompt[OutputT],
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stream a completion request via LiteLLM.

        Args:
            prompt: The Prompt instance to execute.
            model: The model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            **kwargs: Additional LiteLLM parameters.

        Returns:
            A StreamingResponse iterator.
        """
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "LiteLLM is required for this provider. "
                "Install it with: pip install flowprompt[all] or pip install litellm"
            ) from e

        from flowprompt.core.streaming import StreamingResponse

        messages = prompt.to_messages()

        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            stream=True,
            **kwargs,
        )

        return StreamingResponse(response, prompt)

    async def astream(
        self,
        prompt: Prompt[OutputT],
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stream a completion request asynchronously via LiteLLM.

        Args:
            prompt: The Prompt instance to execute.
            model: The model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            **kwargs: Additional LiteLLM parameters.

        Returns:
            An AsyncStreamingResponse iterator.
        """
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "LiteLLM is required for this provider. "
                "Install it with: pip install flowprompt[all] or pip install litellm"
            ) from e

        from flowprompt.core.streaming import AsyncStreamingResponse

        messages = prompt.to_messages()

        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            stream=True,
            **kwargs,
        )

        return AsyncStreamingResponse(response, prompt)
