"""Tests for the compare() convenience function."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flowprompt import Prompt
from flowprompt.testing.compare import (
    ComparisonResult,
    VariantResult,
    acompare,
    compare,
)


# ---------------------------------------------------------------------------
# Test prompt classes
# ---------------------------------------------------------------------------


class PromptA(Prompt[Any]):
    system: str = "You are helpful."
    user: str = "Process: {text}"


class PromptB(Prompt[Any]):
    system: str = "You are concise."
    user: str = "Summarize: {text}"


class PromptC(Prompt[Any]):
    system: str = "You are detailed."
    user: str = "Explain: {text}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_run(return_value: str = "output"):
    """Create a mock for Prompt.run that returns a string."""
    return MagicMock(return_value=return_value)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Test input validation for compare() and acompare()."""

    def test_fewer_than_2_prompts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            compare({"only_one": PromptA}, inputs=[{"text": "hi"}])

    def test_zero_inputs_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            compare({"a": PromptA, "b": PromptB}, inputs=[])

    @pytest.mark.asyncio
    async def test_acompare_fewer_than_2_prompts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            await acompare({"only_one": PromptA}, inputs=[{"text": "hi"}])

    @pytest.mark.asyncio
    async def test_acompare_zero_inputs_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            await acompare({"a": PromptA, "b": PromptB}, inputs=[])


# ---------------------------------------------------------------------------
# Basic comparison
# ---------------------------------------------------------------------------


class TestCompare:
    """Test the synchronous compare() function."""

    def test_basic_comparison(self) -> None:
        """Two variants, both succeed -- should return a ComparisonResult."""
        with patch.object(Prompt, "run", return_value="ok"):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hello"}],
                model="gpt-4o-mini",
            )

        assert isinstance(result, ComparisonResult)
        assert result.total_runs == 2
        assert "a" in result.variants
        assert "b" in result.variants
        assert result.variants["a"].samples == 1
        assert result.variants["b"].samples == 1

    def test_winner_determination(self) -> None:
        """When one variant has clearly better success, it should be the winner."""
        call_count = {"a": 0, "b": 0}

        def mock_run(self, model="gpt-4o", **kwargs):
            # Determine which prompt class this is by checking the user field
            name = "a" if "Process" in self.user else "b"
            call_count[name] += 1
            if name == "a":
                return "good"
            raise RuntimeError("fail")

        inputs = [{"text": f"input_{i}"} for i in range(20)]

        with patch.object(Prompt, "run", mock_run):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=inputs,
                model="gpt-4o-mini",
            )

        # a succeeds 100%, b fails 100%
        assert result.variants["a"].success_rate == 1.0
        assert result.variants["b"].success_rate == 0.0
        # With enough samples and clear difference, winner should be a
        if result.statistical_result and result.statistical_result.significant:
            assert result.winner == "a"

    def test_no_winner_when_equal(self) -> None:
        """When both variants perform equally, there should be no winner."""
        with patch.object(Prompt, "run", return_value="ok"):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hello"}],
                model="gpt-4o-mini",
            )

        # Equal performance, should not have a significant winner
        # (with 1 sample each, it's impossible to be significant)
        assert result.winner is None

    def test_custom_success_fn(self) -> None:
        """success_fn should determine what counts as success."""
        with patch.object(Prompt, "run", return_value="short"):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                model="gpt-4o-mini",
                success_fn=lambda output: len(output) > 10,
            )

        # "short" is len 5, so both should have 0% success
        assert result.variants["a"].success_rate == 0.0
        assert result.variants["b"].success_rate == 0.0

    def test_custom_metric_fn(self) -> None:
        """metric_fn should be called on each output."""
        with patch.object(Prompt, "run", return_value="hello world"):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                model="gpt-4o-mini",
                metric_fn=lambda output: len(output),
            )

        # Both variants produced "hello world" (len 11)
        assert result.variants["a"].samples == 1
        assert result.variants["b"].samples == 1

    def test_error_handling(self) -> None:
        """When a prompt raises an exception, it should be recorded as a failure."""
        with patch.object(Prompt, "run", side_effect=RuntimeError("boom")):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                model="gpt-4o-mini",
            )

        assert result.variants["a"].successes == 0
        assert result.variants["a"].errors == ["boom"]
        assert result.variants["b"].successes == 0
        assert result.variants["b"].errors == ["boom"]

    def test_runs_per_input(self) -> None:
        """runs_per_input should multiply the number of runs."""
        with patch.object(Prompt, "run", return_value="ok"):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                model="gpt-4o-mini",
                runs_per_input=3,
            )

        assert result.variants["a"].samples == 3
        assert result.variants["b"].samples == 3
        assert result.total_runs == 6

    def test_three_variants(self) -> None:
        """compare() should work with more than 2 variants."""
        with patch.object(Prompt, "run", return_value="ok"):
            result = compare(
                {"a": PromptA, "b": PromptB, "c": PromptC},
                inputs=[{"text": "hi"}],
                model="gpt-4o-mini",
            )

        assert len(result.variants) == 3
        assert result.total_runs == 3


# ---------------------------------------------------------------------------
# Output serialization
# ---------------------------------------------------------------------------


class TestComparisonResultOutput:
    """Test __str__ and to_dict of ComparisonResult."""

    def test_str_output(self) -> None:
        with patch.object(Prompt, "run", return_value="ok"):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                model="gpt-4o-mini",
            )

        text = str(result)
        assert "Comparison Results" in text
        assert "a:" in text
        assert "b:" in text

    def test_to_dict(self) -> None:
        with patch.object(Prompt, "run", return_value="ok"):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                model="gpt-4o-mini",
            )

        d = result.to_dict()
        assert "winner" in d
        assert "variants" in d
        assert "a" in d["variants"]
        assert "b" in d["variants"]
        assert d["total_runs"] == 2
        assert d["confidence_level"] == 0.95

    def test_to_dict_with_no_statistical_result(self) -> None:
        """to_dict should handle None statistical_result."""
        cr = ComparisonResult(
            winner=None,
            variants={},
            statistical_result=None,
            confidence_level=0.95,
            total_runs=0,
        )
        d = cr.to_dict()
        assert d["statistical_result"] is None


# ---------------------------------------------------------------------------
# Async variant
# ---------------------------------------------------------------------------


class TestAcompare:
    """Test the async acompare() function."""

    @pytest.mark.asyncio
    async def test_acompare_basic(self) -> None:
        """acompare should produce the same structure as compare."""
        with patch.object(Prompt, "arun", new_callable=AsyncMock, return_value="ok"):
            result = await acompare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hello"}],
                model="gpt-4o-mini",
            )

        assert isinstance(result, ComparisonResult)
        assert result.total_runs == 2
        assert result.variants["a"].samples == 1
        assert result.variants["b"].samples == 1
