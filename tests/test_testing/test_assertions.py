"""Tests for the PromptTestResult assertion wrapper."""

from __future__ import annotations

import pytest

from flowprompt.testing.assertions import PromptTestResult
from flowprompt.testing.compare import ComparisonResult, VariantResult
from flowprompt.testing.statistics import StatisticalResult


def _make_result(
    *,
    winner: str | None = None,
    p_value: float = 0.01,
    significant: bool = True,
    variants: dict[str, VariantResult] | None = None,
) -> ComparisonResult:
    """Create a ComparisonResult for testing."""
    if variants is None:
        variants = {
            "a": VariantResult(name="a", samples=10, successes=9, success_rate=0.9),
            "b": VariantResult(name="b", samples=10, successes=5, success_rate=0.5),
        }
    sr = StatisticalResult(
        test_name="z_test",
        p_value=p_value,
        significant=significant,
        effect_size=0.4,
    )
    return ComparisonResult(
        winner=winner,
        variants=variants,
        statistical_result=sr,
        confidence_level=0.95,
        total_runs=20,
    )


class TestPromptTestResultProperties:
    def test_is_significant_true(self) -> None:
        cr = _make_result(significant=True)
        ptr = PromptTestResult(cr)
        assert ptr.is_significant is True

    def test_is_significant_false(self) -> None:
        cr = _make_result(significant=False, p_value=0.5)
        ptr = PromptTestResult(cr)
        assert ptr.is_significant is False

    def test_winner(self) -> None:
        cr = _make_result(winner="a")
        ptr = PromptTestResult(cr)
        assert ptr.winner == "a"

    def test_p_value(self) -> None:
        cr = _make_result(p_value=0.03)
        ptr = PromptTestResult(cr)
        assert ptr.p_value == 0.03

    def test_p_value_none_when_no_stats(self) -> None:
        cr = ComparisonResult(
            winner=None,
            variants={},
            statistical_result=None,
            confidence_level=0.95,
            total_runs=0,
        )
        ptr = PromptTestResult(cr)
        assert ptr.p_value is None


class TestAssertSignificant:
    def test_passes_when_significant(self) -> None:
        cr = _make_result(p_value=0.01, significant=True)
        ptr = PromptTestResult(cr)
        ptr.assert_significant()  # should not raise

    def test_fails_when_not_significant(self) -> None:
        cr = _make_result(p_value=0.5, significant=False)
        ptr = PromptTestResult(cr)
        with pytest.raises(pytest.fail.Exception, match="not significant"):
            ptr.assert_significant()

    def test_fails_when_no_statistical_result(self) -> None:
        cr = ComparisonResult(
            winner=None,
            variants={},
            statistical_result=None,
            confidence_level=0.95,
            total_runs=0,
        )
        ptr = PromptTestResult(cr)
        with pytest.raises(pytest.fail.Exception, match="No statistical result"):
            ptr.assert_significant()

    def test_custom_threshold(self) -> None:
        cr = _make_result(p_value=0.08, significant=False)
        ptr = PromptTestResult(cr)
        with pytest.raises(pytest.fail.Exception):
            ptr.assert_significant(threshold=0.05)
        # Should pass with a higher threshold
        ptr2 = PromptTestResult(_make_result(p_value=0.08, significant=False))
        ptr2.assert_significant(threshold=0.1)


class TestAssertWinner:
    def test_passes_when_correct(self) -> None:
        cr = _make_result(winner="a")
        ptr = PromptTestResult(cr)
        ptr.assert_winner("a")  # should not raise

    def test_fails_when_wrong(self) -> None:
        cr = _make_result(winner="b")
        ptr = PromptTestResult(cr)
        with pytest.raises(pytest.fail.Exception, match="Expected winner 'a'"):
            ptr.assert_winner("a")

    def test_fails_when_no_winner(self) -> None:
        cr = _make_result(winner=None)
        ptr = PromptTestResult(cr)
        with pytest.raises(pytest.fail.Exception, match="Expected winner 'a'"):
            ptr.assert_winner("a")


class TestAssertNoErrors:
    def test_passes_with_no_errors(self) -> None:
        cr = _make_result()
        ptr = PromptTestResult(cr)
        ptr.assert_no_errors()  # should not raise

    def test_fails_with_errors(self) -> None:
        variants = {
            "a": VariantResult(
                name="a", samples=2, successes=1, errors=["boom", "crash"]
            ),
            "b": VariantResult(name="b", samples=2, successes=2),
        }
        cr = _make_result(variants=variants)
        ptr = PromptTestResult(cr)
        with pytest.raises(pytest.fail.Exception, match="Variants had errors"):
            ptr.assert_no_errors()


class TestDelegation:
    def test_getattr_delegates(self) -> None:
        cr = _make_result()
        ptr = PromptTestResult(cr)
        assert ptr.total_runs == 20
        assert ptr.confidence_level == 0.95

    def test_str(self) -> None:
        cr = _make_result()
        ptr = PromptTestResult(cr)
        assert "Comparison Results" in str(ptr)

    def test_repr(self) -> None:
        cr = _make_result()
        ptr = PromptTestResult(cr)
        assert "PromptTestResult" in repr(ptr)
