"""Tests for LiteLLM provider."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from flowprompt import Prompt
from flowprompt.providers.litellm import LiteLLMProvider


class MockChoice:
    """Mock LiteLLM choice object."""

    def __init__(self, content: str) -> None:
        self.message = MagicMock(content=content)


class MockResponse:
    """Mock LiteLLM response object."""

    def __init__(self, content: str) -> None:
        self.choices = [MockChoice(content)]


class TestLiteLLMProvider:
    """Test LiteLLM provider functionality."""

    def test_complete_simple_prompt(self) -> None:
        """Test completing a simple prompt without structured output."""

        class SimplePrompt(Prompt[Any]):
            system: str = "You are helpful."
            user: str = "Say hello."

        prompt = SimplePrompt()
        provider = LiteLLMProvider()

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = MockResponse("Hello there!")

            result = provider.complete(prompt, model="gpt-4o")

            assert result == "Hello there!"
            mock_completion.assert_called_once()

    def test_complete_with_structured_output(self) -> None:
        """Test completing a prompt with structured output."""

        class UserOutput(BaseModel):
            name: str
            age: int

        class ExtractPrompt(Prompt[UserOutput]):
            system: str = "Extract user info."
            user: str = "John is 25 years old."

            class Output(BaseModel):
                name: str
                age: int

        prompt = ExtractPrompt()
        provider = LiteLLMProvider()

        json_response = json.dumps({"name": "John", "age": 25})

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = MockResponse(json_response)

            result = provider.complete(prompt, model="gpt-4o")

            assert isinstance(result, BaseModel)
            assert result.name == "John"  # type: ignore[attr-defined]
            assert result.age == 25  # type: ignore[attr-defined]

    def test_complete_with_temperature(self) -> None:
        """Test that temperature is passed correctly."""

        class SimplePrompt(Prompt[Any]):
            system: str = "You are helpful."
            user: str = "Be creative."

        prompt = SimplePrompt()
        provider = LiteLLMProvider()

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = MockResponse("Creative response!")

            provider.complete(prompt, model="gpt-4o", temperature=0.8)

            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["temperature"] == 0.8

    def test_complete_retries_on_validation_error(self) -> None:
        """Test that completion retries on validation errors."""

        class UserOutput(BaseModel):
            name: str
            age: int

        class ExtractPrompt(Prompt[UserOutput]):
            system: str = "Extract user info."
            user: str = "John is 25."

            class Output(BaseModel):
                name: str
                age: int

        prompt = ExtractPrompt()
        provider = LiteLLMProvider()

        # First call returns invalid JSON, second returns valid
        with patch("litellm.completion") as mock_completion:
            mock_completion.side_effect = [
                MockResponse("invalid json"),
                MockResponse('{"name": "John", "age": 25}'),
            ]

            result = provider.complete(prompt, model="gpt-4o", max_retries=3)

            assert result.name == "John"  # type: ignore[attr-defined]
            assert mock_completion.call_count == 2


class TestNativeJsonSchema:
    """Test native JSON schema response format."""

    def test_native_schema_when_supported(self) -> None:
        """When model supports response_schema, use json_schema format."""

        class UserOutput(BaseModel):
            name: str
            age: int

        class ExtractPrompt(Prompt[UserOutput]):
            system: str = "Extract user info."
            user: str = "John is 25 years old."

            class Output(BaseModel):
                name: str
                age: int

        prompt = ExtractPrompt()
        provider = LiteLLMProvider()
        json_response = json.dumps({"name": "John", "age": 25})

        with (
            patch("litellm.supports_response_schema", return_value=True),
            patch("litellm.completion") as mock_completion,
        ):
            mock_completion.return_value = MockResponse(json_response)
            result = provider.complete(prompt, model="gpt-4o")

            # Verify response_format uses json_schema
            call_kwargs = mock_completion.call_args[1]
            rf = call_kwargs["response_format"]
            assert rf["type"] == "json_schema"
            assert rf["json_schema"]["strict"] is True
            assert rf["json_schema"]["name"] == "Output"

            # Verify system message was NOT mutated with schema text
            messages = call_kwargs["messages"]
            assert "Respond with valid JSON" not in messages[0]["content"]

            assert result.name == "John"  # type: ignore[attr-defined]

    def test_fallback_when_not_supported(self) -> None:
        """When model does NOT support response_schema, fall back to json_object."""

        class UserOutput(BaseModel):
            name: str
            age: int

        class ExtractPrompt(Prompt[UserOutput]):
            system: str = "Extract user info."
            user: str = "John is 25 years old."

            class Output(BaseModel):
                name: str
                age: int

        prompt = ExtractPrompt()
        provider = LiteLLMProvider()
        json_response = json.dumps({"name": "John", "age": 25})

        with (
            patch("litellm.supports_response_schema", return_value=False),
            patch("litellm.completion") as mock_completion,
        ):
            mock_completion.return_value = MockResponse(json_response)
            result = provider.complete(prompt, model="gpt-3.5-turbo")

            # Verify response_format is json_object fallback
            call_kwargs = mock_completion.call_args[1]
            rf = call_kwargs["response_format"]
            assert rf["type"] == "json_object"

            # Verify system message WAS mutated with schema text
            messages = call_kwargs["messages"]
            assert "Respond with valid JSON" in messages[0]["content"]

            assert result.name == "John"  # type: ignore[attr-defined]

    def test_fallback_when_check_raises(self) -> None:
        """When supports_response_schema raises, fall back gracefully."""

        class UserOutput(BaseModel):
            name: str
            age: int

        class ExtractPrompt(Prompt[UserOutput]):
            system: str = "Extract user info."
            user: str = "John is 25 years old."

            class Output(BaseModel):
                name: str
                age: int

        prompt = ExtractPrompt()
        provider = LiteLLMProvider()
        json_response = json.dumps({"name": "John", "age": 25})

        with (
            patch(
                "litellm.supports_response_schema",
                side_effect=Exception("unknown model"),
            ),
            patch("litellm.completion") as mock_completion,
        ):
            mock_completion.return_value = MockResponse(json_response)
            result = provider.complete(prompt, model="some-unknown-model")

            call_kwargs = mock_completion.call_args[1]
            rf = call_kwargs["response_format"]
            assert rf["type"] == "json_object"
            assert result.name == "John"  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_async_native_schema(self) -> None:
        """acomplete should also use native schema when supported."""

        class UserOutput(BaseModel):
            name: str
            age: int

        class ExtractPrompt(Prompt[UserOutput]):
            system: str = "Extract user info."
            user: str = "Alice is 30."

            class Output(BaseModel):
                name: str
                age: int

        prompt = ExtractPrompt()
        provider = LiteLLMProvider()
        json_response = json.dumps({"name": "Alice", "age": 30})

        with (
            patch("litellm.supports_response_schema", return_value=True),
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
        ):
            mock_acompletion.return_value = MockResponse(json_response)
            result = await provider.acomplete(prompt, model="gpt-4o")

            call_kwargs = mock_acompletion.call_args[1]
            rf = call_kwargs["response_format"]
            assert rf["type"] == "json_schema"
            assert result.name == "Alice"  # type: ignore[attr-defined]


class TestLiteLLMProviderAsync:
    """Test async LiteLLM provider functionality."""

    @pytest.mark.asyncio
    async def test_acomplete_simple_prompt(self) -> None:
        """Test async completing a simple prompt."""

        class SimplePrompt(Prompt[Any]):
            system: str = "You are helpful."
            user: str = "Say hello."

        prompt = SimplePrompt()
        provider = LiteLLMProvider()

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = MockResponse("Hello async!")

            result = await provider.acomplete(prompt, model="gpt-4o")

            assert result == "Hello async!"

    @pytest.mark.asyncio
    async def test_acomplete_with_structured_output(self) -> None:
        """Test async completing with structured output."""

        class UserOutput(BaseModel):
            name: str
            age: int

        class ExtractPrompt(Prompt[UserOutput]):
            system: str = "Extract user info."
            user: str = "Alice is 30."

            class Output(BaseModel):
                name: str
                age: int

        prompt = ExtractPrompt()
        provider = LiteLLMProvider()

        json_response = json.dumps({"name": "Alice", "age": 30})

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = MockResponse(json_response)

            result = await provider.acomplete(prompt, model="gpt-4o")

            assert result.name == "Alice"  # type: ignore[attr-defined]
            assert result.age == 30  # type: ignore[attr-defined]
