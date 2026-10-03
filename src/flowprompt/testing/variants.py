"""Variants: anything that turns an input into an output.

The comparison engine only needs a callable ``variant(input: dict) -> output``
(sync or async). Prompt classes and "same prompt, different model" setups
are thin adapters onto that protocol:

    compare(
        {
            "baseline": ExtractUser,                     # Prompt class, default model
            "mini": (ExtractUser, "gpt-4o-mini"),         # Prompt class on another model
            "regex": lambda inp: my_rule_based(inp["text"]),  # any callable
            "pipeline": PromptVariant(ExtractUser, model="gpt-4o", temperature=0.2),
        },
        inputs=..., expected=...,
    )

LLM calls made through FlowPrompt inside any variant (including your own
callables) are metered automatically, so cost per variant works for custom
pipelines too.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["PromptVariant", "FunctionVariant", "as_variant", "model_variants"]


def _is_prompt_class(obj: Any) -> bool:
    from flowprompt.core.prompt import Prompt

    return isinstance(obj, type) and issubclass(obj, Prompt)


@dataclass
class PromptVariant:
    """Adapter running a Prompt class on one model.

    Args:
        prompt: The Prompt subclass. Each input dict is passed as keyword
            arguments to its constructor.
        model: Model identifier. None means "use compare()'s ``model``".
        temperature: Sampling temperature. None means "use compare()'s".
        run_kwargs: Extra keyword arguments for ``Prompt.run()``.
    """

    prompt: type
    model: str | None = None
    temperature: float | None = None
    run_kwargs: dict[str, Any] = field(default_factory=dict)

    def bind(self, model: str, temperature: float) -> PromptVariant:
        """Fill in defaults from compare()."""
        return PromptVariant(
            self.prompt,
            self.model if self.model is not None else model,
            self.temperature if self.temperature is not None else temperature,
            dict(self.run_kwargs),
        )

    @property
    def label(self) -> str:
        name = getattr(self.prompt, "__name__", str(self.prompt))
        return f"{name} @ {self.model}" if self.model else name

    def __call__(self, inputs: dict[str, Any]) -> Any:
        prompt = self.prompt(**inputs)
        return prompt.run(
            model=self.model or "gpt-4o",
            temperature=self.temperature if self.temperature is not None else 0.0,
            **self.run_kwargs,
        )

    async def acall(self, inputs: dict[str, Any]) -> Any:
        prompt = self.prompt(**inputs)
        return await prompt.arun(
            model=self.model or "gpt-4o",
            temperature=self.temperature if self.temperature is not None else 0.0,
            **self.run_kwargs,
        )


@dataclass
class FunctionVariant:
    """Adapter for a plain callable ``fn(input: dict) -> output`` (sync or async)."""

    fn: Callable[[dict[str, Any]], Any]

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.fn) or inspect.iscoroutinefunction(
            getattr(self.fn, "__call__", None)  # noqa: B004
        )

    @property
    def label(self) -> str:
        return getattr(self.fn, "__name__", type(self.fn).__name__)

    def __call__(self, inputs: dict[str, Any]) -> Any:
        if self.is_async:
            return asyncio.run(self.fn(inputs))
        result = self.fn(inputs)
        if inspect.isawaitable(result):
            return asyncio.run(_await(result))
        return result

    async def acall(self, inputs: dict[str, Any]) -> Any:
        if self.is_async:
            return await self.fn(inputs)
        result = await asyncio.to_thread(self.fn, inputs)
        if inspect.isawaitable(result):
            return await result
        return result


async def _await(awaitable: Any) -> Any:
    return await awaitable


Variant = PromptVariant | FunctionVariant


def as_variant(obj: Any, *, model: str, temperature: float) -> Variant:
    """Adapt a user-supplied variant definition to the engine protocol.

    Accepts a Prompt subclass, a ``(PromptClass, "model")`` tuple, a
    :class:`PromptVariant`, or any callable taking the input dict.
    """
    if isinstance(obj, PromptVariant):
        return obj.bind(model, temperature)
    if isinstance(obj, FunctionVariant):
        return obj
    if _is_prompt_class(obj):
        return PromptVariant(obj, model, temperature)
    if isinstance(obj, tuple) and len(obj) == 2 and _is_prompt_class(obj[0]):
        return PromptVariant(obj[0], str(obj[1]), temperature)
    if callable(obj):
        return FunctionVariant(obj)
    raise TypeError(
        f"Cannot use {obj!r} as a variant: pass a Prompt subclass, a "
        "(PromptClass, model) tuple, a PromptVariant, or a callable taking "
        "the input dict."
    )


def model_variants(
    prompt: type, models: list[str], *, temperature: float | None = None
) -> dict[str, PromptVariant]:
    """Variants that run the same Prompt class on several models.

    Example:
        >>> compare(model_variants(ExtractUser, ["gpt-4o-mini", "gpt-4o"]), inputs, ...)
    """
    return {m: PromptVariant(prompt, m, temperature) for m in models}
