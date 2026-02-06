"""LiteLLM provider for unified multi-provider LLM access."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

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

        messages = prompt.to_messages()
        output_model = prompt.output_model

        # Add JSON response format if Output model is defined
        response_format = None
        if output_model is not None:
            response_format = self._build_response_format(
                model, output_model, messages
            )

        # Execute completion with retries
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    response_format=response_format,
                    **kwargs,
                )

                content = response.choices[0].message.content or ""

                # Parse and validate if Output model is defined
                if output_model is not None:
                    try:
                        data = json.loads(content)
                        return cast(OutputT, output_model.model_validate(data))
                    except (json.JSONDecodeError, ValidationError) as e:
                        if attempt < max_retries - 1:
                            last_error = e
                            continue
                        raise

                return content

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

        messages = prompt.to_messages()
        output_model = prompt.output_model

        # Add JSON response format if Output model is defined
        response_format = None
        if output_model is not None:
            response_format = self._build_response_format(
                model, output_model, messages
            )

        # Execute async completion with retries
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    response_format=response_format,
                    **kwargs,
                )

                content = response.choices[0].message.content or ""

                # Parse and validate if Output model is defined
                if output_model is not None:
                    try:
                        data = json.loads(content)
                        return cast(OutputT, output_model.model_validate(data))
                    except (json.JSONDecodeError, ValidationError) as e:
                        if attempt < max_retries - 1:
                            last_error = e
                            continue
                        raise

                return content

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
