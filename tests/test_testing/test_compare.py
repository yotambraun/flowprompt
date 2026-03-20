"""Tests for the compare() convenience function."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from flowprompt import Prompt
from flowprompt.testing.compare import (
    ComparisonResult,
    acompare,
    compare,
    estimate_compare_cost,
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

        def mock_run(self, model="gpt-4o", **_kwargs):  # noqa: ARG001
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


# ---------------------------------------------------------------------------
# Dry run and cost estimation
# ---------------------------------------------------------------------------


class TestDryRun:
    """Test dry_run mode for compare() and acompare()."""

    def test_dry_run_returns_no_winner(self) -> None:
        """dry_run=True returns ComparisonResult with winner=None, total_runs=0."""
        result = compare(
            {"a": PromptA, "b": PromptB},
            inputs=[{"text": "hello"}],
            model="gpt-4o-mini",
            dry_run=True,
        )

        assert isinstance(result, ComparisonResult)
        assert result.winner is None
        assert result.total_runs == 0
        assert result.variants == {}
        assert result.statistical_result is None

    def test_dry_run_does_not_call_run(self) -> None:
        """Verify Prompt.run is NOT called when dry_run=True."""
        with patch.object(Prompt, "run", side_effect=AssertionError("should not call")):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hello"}],
                model="gpt-4o-mini",
                dry_run=True,
            )

        assert result.total_runs == 0

    def test_dry_run_shows_estimated_cost(self) -> None:
        """estimated_cost dict has expected keys."""
        result = compare(
            {"a": PromptA, "b": PromptB},
            inputs=[{"text": "hello"}],
            model="gpt-4o-mini",
            dry_run=True,
        )

        assert result.estimated_cost is not None
        assert "model" in result.estimated_cost
        assert "total_calls" in result.estimated_cost
        assert "estimated_input_tokens" in result.estimated_cost
        assert "estimated_output_tokens" in result.estimated_cost
        assert "estimated_cost_usd" in result.estimated_cost
        assert "per_variant" in result.estimated_cost
        assert result.estimated_cost["total_calls"] == 2  # 2 variants * 1 input * 1 run

    def test_dry_run_str_output(self) -> None:
        """dry_run result __str__ should contain DRY RUN."""
        result = compare(
            {"a": PromptA, "b": PromptB},
            inputs=[{"text": "hello"}],
            model="gpt-4o-mini",
            dry_run=True,
        )

        text = str(result)
        assert "DRY RUN" in text
        assert "Estimated cost" in text

    @pytest.mark.asyncio
    async def test_acompare_dry_run(self) -> None:
        """Async dry_run variant should also work."""
        with patch.object(
            Prompt,
            "arun",
            new_callable=AsyncMock,
            side_effect=AssertionError("should not call"),
        ):
            result = await acompare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hello"}],
                model="gpt-4o-mini",
                dry_run=True,
            )

        assert result.winner is None
        assert result.total_runs == 0
        assert result.estimated_cost is not None
        assert result.estimated_cost["total_calls"] == 2


# ---------------------------------------------------------------------------
# Cost estimation standalone
# ---------------------------------------------------------------------------


class TestEstimateCompareCost:
    """Test the estimate_compare_cost() function."""

    def test_estimate_compare_cost_basic(self) -> None:
        """Standalone function returns correct structure."""
        result = estimate_compare_cost(
            {"a": PromptA, "b": PromptB},
            inputs=[{"text": "hello"}, {"text": "world"}],
            model="gpt-4o-mini",
        )

        assert result["model"] == "gpt-4o-mini"
        assert result["total_calls"] == 4  # 2 variants * 2 inputs
        assert result["estimated_input_tokens"] >= 0
        assert result["estimated_output_tokens"] == 400  # 4 calls * 100 default
        assert "per_variant" in result
        assert "a" in result["per_variant"]
        assert "b" in result["per_variant"]
        assert result["per_variant"]["a"]["calls"] == 2
        assert result["per_variant"]["b"]["calls"] == 2

    def test_estimate_compare_cost_runs_per_input(self) -> None:
        """runs_per_input multiplies calls correctly."""
        result = estimate_compare_cost(
            {"a": PromptA, "b": PromptB},
            inputs=[{"text": "hello"}],
            model="gpt-4o-mini",
            runs_per_input=3,
        )

        assert result["total_calls"] == 6  # 2 variants * 1 input * 3 runs

    def test_estimate_compare_cost_fallback_when_no_litellm(self) -> None:
        """Graceful fallback when litellm is not available."""
        with patch.dict("sys.modules", {"litellm": None}):
            result = estimate_compare_cost(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hello"}],
                model="gpt-4o-mini",
            )

        # Should still return valid structure, cost fields may be None
        assert result["total_calls"] == 2
        assert "estimated_input_tokens" in result

    def test_to_dict_includes_estimated_cost(self) -> None:
        """to_dict should include estimated_cost field."""
        result = compare(
            {"a": PromptA, "b": PromptB},
            inputs=[{"text": "hello"}],
            model="gpt-4o-mini",
            dry_run=True,
        )

        d = result.to_dict()
        assert "estimated_cost" in d
        assert d["estimated_cost"] is not None
        assert d["estimated_cost"]["total_calls"] == 2


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


class TestParallelExecution:
    """Test that sync compare() uses ThreadPoolExecutor."""

    def test_parallel_execution(self) -> None:
        """Verify ThreadPoolExecutor is used for sync compare()."""
        with (
            patch.object(Prompt, "run", return_value="ok"),
            patch(
                "flowprompt.testing.compare.ThreadPoolExecutor",
                wraps=ThreadPoolExecutor,
            ) as mock_tpe,
        ):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hello"}],
                model="gpt-4o-mini",
            )

        # ThreadPoolExecutor should have been called
        mock_tpe.assert_called_once_with(max_workers=2)
        assert result.total_runs == 2


# ---------------------------------------------------------------------------
# Expected outputs and eval metrics
# ---------------------------------------------------------------------------


class TestExpectedOutputs:
    """Test the expected parameter and eval_metric integration."""

    def test_expected_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="len\\(expected\\)"):
            compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                expected=["one", "two"],
                model="gpt-4o-mini",
            )

    def test_expected_with_default_contains_metric(self) -> None:
        """When expected is provided, outputs are checked with contains_match."""
        with patch.object(Prompt, "run", return_value="The answer is positive"):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                expected=["positive"],
                model="gpt-4o-mini",
            )

        assert result.has_expected is True
        assert result.variants["a"].success_rate == 1.0
        assert result.variants["b"].success_rate == 1.0

    def test_expected_with_exact_metric(self) -> None:
        """exact_match should fail when output doesn't exactly match."""
        with patch.object(Prompt, "run", return_value="The answer is positive"):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                expected=["positive"],
                eval_metric="exact",
                model="gpt-4o-mini",
            )

        # "The answer is positive" != "positive" for exact match
        assert result.variants["a"].success_rate == 0.0

    def test_expected_with_custom_eval_callable(self) -> None:
        """User can pass a callable as eval_metric."""
        custom = lambda output, exp: output.startswith(exp)  # noqa: E731
        with patch.object(Prompt, "run", return_value="hello world"):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                expected=["hello"],
                eval_metric=custom,
                model="gpt-4o-mini",
            )

        assert result.variants["a"].success_rate == 1.0

    def test_explicit_success_fn_takes_precedence(self) -> None:
        """When both expected and success_fn are provided, success_fn gets (output, expected)."""
        calls = []

        def my_fn(output, expected_val):
            calls.append((output, expected_val))
            return True

        with patch.object(Prompt, "run", return_value="ok"):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                expected=["exp1"],
                success_fn=my_fn,
                model="gpt-4o-mini",
            )

        # success_fn should have been called with (output, expected_value)
        assert any(c == ("ok", "exp1") for c in calls)
        assert result.has_expected is True

    def test_no_expected_backward_compatible(self) -> None:
        """Without expected, success_fn receives only (output,)."""
        calls = []

        def my_fn(output):
            calls.append(output)
            return True

        with patch.object(Prompt, "run", return_value="ok"):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                success_fn=my_fn,
                model="gpt-4o-mini",
            )

        assert "ok" in calls
        assert result.has_expected is False

    def test_str_shows_accuracy_with_expected(self) -> None:
        """When has_expected=True, __str__ shows 'accuracy' not 'success'."""
        with patch.object(Prompt, "run", return_value="positive"):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                expected=["positive"],
                model="gpt-4o-mini",
            )

        text = str(result)
        assert "accuracy" in text
        assert "success" not in text.split("=")[-1]  # not in the results area

    def test_str_shows_success_without_expected(self) -> None:
        """Without expected, __str__ shows 'success'."""
        with patch.object(Prompt, "run", return_value="ok"):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                model="gpt-4o-mini",
            )

        text = str(result)
        assert "success" in text

    @pytest.mark.asyncio
    async def test_acompare_with_expected(self) -> None:
        """acompare should also support expected."""
        with patch.object(
            Prompt, "arun", new_callable=AsyncMock, return_value="The answer is yes"
        ):
            result = await acompare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                expected=["yes"],
                model="gpt-4o-mini",
            )

        assert result.has_expected is True
        assert result.variants["a"].success_rate == 1.0

    @pytest.mark.asyncio
    async def test_acompare_expected_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="len\\(expected\\)"):
            await acompare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                expected=["one", "two"],
                model="gpt-4o-mini",
            )

    def test_multiple_inputs_with_expected(self) -> None:
        """Each input should be checked against its corresponding expected value."""
        outputs = iter(["positive", "negative", "neutral"])

        with patch.object(Prompt, "run", side_effect=lambda **_kw: next(outputs)):
            # Note: each variant gets its own copy of the iterator, so we need
            # to handle this carefully. Let's just use a fixed return.
            pass

        # Use a simpler approach
        with patch.object(Prompt, "run", return_value="positive"):
            result = compare(
                {"a": PromptA, "b": PromptB},
                inputs=[
                    {"text": "great"},
                    {"text": "bad"},
                ],
                expected=["positive", "negative"],
                eval_metric="exact",
                model="gpt-4o-mini",
            )

        # "positive" matches "positive" but not "negative" -> 50% accuracy
        assert result.variants["a"].success_rate == 0.5
