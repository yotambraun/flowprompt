"""Base Prompt class for type-safe prompt management."""

from __future__ import annotations

import hashlib
import sys
from typing import Any, ClassVar, Generic, TypeVar, cast

from jinja2 import Template
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic.fields import FieldInfo

OutputT = TypeVar("OutputT", bound=BaseModel)


def _is_plain_value(value: Any) -> bool:
    """Return True for values that can only be meant as a field default."""
    if isinstance(value, FieldInfo):
        return True
    return not (callable(value) or hasattr(value, "__get__"))


def _annotate_inherited_field_overrides(
    bases: tuple[type, ...], namespace: dict[str, Any]
) -> None:
    """Add annotations for unannotated overrides of inherited fields.

    Pydantic requires every field override to carry a type annotation, so
    ``class MyPrompt(Prompt): system = "..."`` would otherwise fail with
    ``PydanticUserError``. FlowPrompt's documented style omits annotations,
    so we reuse the annotation declared on the base class (e.g. ``str`` for
    ``system`` and ``user``) before pydantic inspects the namespace.
    """
    inherited: dict[str, Any] = {}
    for base in bases:
        for klass in reversed(base.__mro__):
            fields = klass.__dict__.get("__pydantic_fields__")
            if isinstance(fields, dict):
                for field_name, info in fields.items():
                    inherited[field_name] = info.annotation
    if not inherited:
        return

    missing = {
        key: inherited[key]
        for key, value in namespace.items()
        if key in inherited and not key.startswith("_") and _is_plain_value(value)
    }
    if not missing:
        return

    if "__annotations__" in namespace:
        annotations = namespace["__annotations__"]
    elif sys.version_info >= (3, 14):  # pragma: no cover - exercised on 3.14+
        # PEP 649: annotations are produced lazily by ``__annotate__``.
        from annotationlib import (
            Format,
            call_annotate_function,
            get_annotate_from_class_namespace,
        )

        annotate = get_annotate_from_class_namespace(namespace)
        annotations = (
            dict(call_annotate_function(annotate, format=Format.FORWARDREF))
            if annotate
            else {}
        )
        namespace["__annotations__"] = annotations
    else:
        annotations = {}
        namespace["__annotations__"] = annotations

    for key, annotation in missing.items():
        if key not in annotations:
            annotations[key] = annotation


class PromptMeta(type(BaseModel)):  # type: ignore[misc]
    """Metaclass for Prompt that handles Output class registration."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> PromptMeta:
        if bases:
            _annotate_inherited_field_overrides(bases, namespace)
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Register Output class if defined
        if (
            "Output" in namespace
            and isinstance(namespace["Output"], type)
            and issubclass(namespace["Output"], BaseModel)
        ):
            cls._output_model = namespace["Output"]

        return cast(PromptMeta, cls)


class Prompt(BaseModel, Generic[OutputT], metaclass=PromptMeta):
    """Base class for type-safe prompt definitions.

    Define prompts as Python classes with type hints, templates, and
    structured output validation. Prompts are immutable and hashable.

    Attributes:
        system: The system message for the LLM.
        user: The user message template with {variable} placeholders.
        __version__: Optional version string for tracking prompt changes.

    Example:
        >>> from pydantic import BaseModel
        >>> from flowprompt import Prompt, Field
        >>>
        >>> class ExtractUser(Prompt):
        ...     system: str = "You are a precise data extractor."
        ...     user: str = Field(template="Extract from: {text}")
        ...
        ...     class Output(BaseModel):
        ...         name: str
        ...         age: int
        >>>
        >>> result = ExtractUser(text="John is 25").run(model="gpt-4o")
    """

    model_config = ConfigDict(
        frozen=True,
        extra="allow",
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    # Class-level attributes
    __version__: ClassVar[str] = "0.0.0"
    _output_model: ClassVar[type[BaseModel] | None] = None

    # Default prompt fields (can be overridden)
    system: str = "You are a helpful assistant."
    user: str = ""

    @model_validator(mode="after")
    def _interpolate_templates(self) -> Prompt[OutputT]:
        """Interpolate template strings with provided values."""
        # Get all extra fields (template variables)
        extra_data = {}
        for key in self.model_fields_set:
            if key not in ("system", "user"):
                extra_data[key] = getattr(self, key, None)

        # Also include any extra values passed during construction
        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__:
            extra_data.update(self.__pydantic_extra__)

        if not extra_data:
            return self

        # Interpolate system message
        if "{" in self.system:
            interpolated = self._interpolate_string(self.system, extra_data)
            object.__setattr__(self, "system", interpolated)

        # Interpolate user message
        if "{" in self.user:
            interpolated = self._interpolate_string(self.user, extra_data)
            object.__setattr__(self, "user", interpolated)

        return self

    @staticmethod
    def _interpolate_string(text: str, data: dict[str, Any]) -> str:
        """Interpolate a string with data, supporting both Python and Jinja2 syntax.

        Supports:
        - Python format: {name} -> value
        - Jinja2 format: {{ name }} -> value
        - Jinja2 control: {% if %} ... {% endif %}
        """
        # Check if it's using Jinja2 syntax ({{ }} or {% %})
        if "{{" in text or "{%" in text:
            template = Template(text)
            return template.render(**data)

        # Otherwise, use Python str.format() for simple {name} syntax
        try:
            return text.format(**data)
        except KeyError:
            # If format fails, try Jinja2 as fallback
            template = Template(text)
            return template.render(**data)

    @property
    def content_hash(self) -> str:
        """Generate a unique hash based on prompt content.

        Returns:
            A SHA-256 hash of the prompt's system and user messages.
        """
        content = f"{self.system}|{self.user}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def version(self) -> str:
        """Get the prompt version.

        Returns:
            The version string defined in __version__ class attribute.
        """
        return self.__class__.__version__

    @property
    def output_model(self) -> type[BaseModel] | None:
        """Get the Output model class if defined.

        Returns:
            The Output BaseModel class, or None if not defined.
        """
        return getattr(self.__class__, "_output_model", None)

    def to_messages(self) -> list[dict[str, str]]:
        """Convert prompt to a list of message dicts.

        Returns:
            A list of message dictionaries with 'role' and 'content' keys.

        Example:
            >>> prompt = MyPrompt(text="hello")
            >>> prompt.to_messages()
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Process: hello"}
            ]
        """
        messages = []

        if self.system:
            messages.append({"role": "system", "content": self.system})

        if self.user:
            messages.append({"role": "user", "content": self.user})

        return messages

    def run(
        self,
        model: str = "gpt-4o",
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> OutputT | str:
        """Execute the prompt against an LLM.

        Args:
            model: The model identifier (e.g., "gpt-4o", "anthropic/claude-3-5-sonnet").
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            max_retries: Number of retries on failure.
            **kwargs: Additional provider-specific parameters.

        Returns:
            If Output class is defined, returns a validated instance of Output.
            Otherwise, returns the raw string response.

        Example:
            >>> result = MyPrompt(text="hello").run(model="gpt-4o")
            >>> print(result.name)  # If Output has a name field
        """
        from flowprompt.providers.litellm import LiteLLMProvider

        provider = LiteLLMProvider()
        return provider.complete(
            prompt=self,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

    async def arun(
        self,
        model: str = "gpt-4o",
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> OutputT | str:
        """Execute the prompt asynchronously.

        Args:
            model: The model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            max_retries: Number of retries on failure.
            **kwargs: Additional provider-specific parameters.

        Returns:
            If Output class is defined, returns a validated instance of Output.
            Otherwise, returns the raw string response.
        """
        from flowprompt.providers.litellm import LiteLLMProvider

        provider = LiteLLMProvider()
        return await provider.acomplete(
            prompt=self,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

    def stream(
        self,
        model: str = "gpt-4o",
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stream the prompt response.

        Yields chunks of the response as they arrive from the LLM.

        Args:
            model: The model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            **kwargs: Additional provider-specific parameters.

        Returns:
            A StreamingResponse iterator.

        Example:
            >>> for chunk in prompt.stream(model="gpt-4o"):
            ...     print(chunk.delta, end="", flush=True)
        """
        from flowprompt.providers.litellm import LiteLLMProvider

        provider = LiteLLMProvider()
        return provider.stream(
            prompt=self,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )

    async def astream(
        self,
        model: str = "gpt-4o",
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stream the prompt response asynchronously.

        Args:
            model: The model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            **kwargs: Additional provider-specific parameters.

        Returns:
            An AsyncStreamingResponse iterator.

        Example:
            >>> async for chunk in prompt.astream(model="gpt-4o"):
            ...     print(chunk.delta, end="", flush=True)
        """
        from flowprompt.providers.litellm import LiteLLMProvider

        provider = LiteLLMProvider()
        return await provider.astream(
            prompt=self,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        return (
            f"{self.__class__.__name__}("
            f"version={self.version!r}, "
            f"hash={self.content_hash!r})"
        )
