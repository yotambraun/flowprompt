"""Known-answer tests for the statistics used by compare().

Every variant in compare() runs on the *same* inputs, so the comparison is
paired. These tests pin the behaviour with hand-computable cases.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from flowprompt import Prompt
from flowprompt.testing.compare import compare


class Control(Prompt[Any]):
    system: str = "control"
    user: str = "control {i}"


class TreatmentA(Prompt[Any]):
    system: str = "treatment_a"
    user: str = "treatment_a {i}"


class TreatmentB(Prompt[Any]):
    system: str = "treatment_b"
    user: str = "treatment_b {i}"


def _fake_run(correct: dict[str, set[int]]) -> Any:
    """Return a Prompt.run replacement: 'yes' when the variant is right on i."""

    def run(self: Prompt[Any], model: str = "x", **_: Any) -> str:  # noqa: ARG001
        variant, idx = self.user.split()
        return "yes" if int(idx) in correct[variant] else "no"

    return run


INPUTS = [{"i": i} for i in range(10)]
EXPECTED = ["yes"] * 10


def _compare(correct: dict[str, set[int]], prompts: dict[str, type], **kw: Any):
    with patch.object(Prompt, "run", _fake_run(correct)):
        return compare(
            prompts,
            inputs=INPUTS,
            expected=EXPECTED,
            eval_metric="exact",
            model="fake",
            **kw,
        )


def test_pass_fail_uses_exact_mcnemar() -> None:
    # Control right on 0-7 (8/10), treatment right on 0-3 (4/10).
    # Discordant pairs: 4 favour control, 0 favour treatment.
    # Exact McNemar: p = 2 * P(Bin(4, 0.5) <= 0) = 2/16 = 0.125.
    # (The unpaired pooled z-test gives 0.068 for the same data.)
    result = _compare(
        {"control": set(range(8)), "treatment_a": set(range(4))},
        {"control": Control, "treatment_a": TreatmentA},
    )
    sr = result.statistical_result
    assert sr is not None
    assert sr.p_value == pytest.approx(0.125)
    assert sr.significant is False
    assert sr.method == "mcnemar_exact"
    assert sr.n_inputs == 10
    assert sr.difference == pytest.approx(-0.4)
    assert result.winner is None


def test_repeated_runs_are_not_counted_as_independent_samples() -> None:
    # Deterministic outputs repeated 5 times carry no extra information:
    # the p-value must stay at the 1-run value (0.125), not shrink.
    result = _compare(
        {"control": set(range(8)), "treatment_a": set(range(4))},
        {"control": Control, "treatment_a": TreatmentA},
        runs_per_input=5,
    )
    sr = result.statistical_result
    assert sr is not None
    assert sr.p_value == pytest.approx(0.125)
    assert sr.n_inputs == 10
    assert sr.significant is False


def test_treatment_wins_when_control_scores_zero() -> None:
    # Control 0/10, treatment 6/10: exact McNemar p = 2/64 = 0.03125.
    result = _compare(
        {"control": set(), "treatment_a": set(range(6))},
        {"control": Control, "treatment_a": TreatmentA},
    )
    sr = result.statistical_result
    assert sr is not None
    assert sr.p_value == pytest.approx(0.03125)
    assert sr.significant is True
    assert result.winner == "treatment_a"
    assert sr.difference == pytest.approx(0.6)
    assert sr.relative_lift is None  # undefined when the control rate is 0


def test_control_wins_when_significantly_better() -> None:
    result = _compare(
        {"control": set(range(6)), "treatment_a": set()},
        {"control": Control, "treatment_a": TreatmentA},
    )
    assert result.winner == "control"
    assert result.statistical_result.difference == pytest.approx(-0.6)


def test_three_variants_apply_holm_correction() -> None:
    # Each treatment alone: p = 0.03125 (6 discordant, all favour it).
    # Holm over 2 comparisons: adjusted p = 0.0625 for both -> no winner.
    result = _compare(
        {"control": set(), "treatment_a": set(range(6)), "treatment_b": set(range(6))},
        {"control": Control, "treatment_a": TreatmentA, "treatment_b": TreatmentB},
    )
    assert len(result.comparisons) == 2
    for comp in result.comparisons:
        assert comp.p_value == pytest.approx(0.03125)
        assert comp.adjusted_p == pytest.approx(0.0625)
        assert comp.significant is False
    assert result.winner is None


def test_no_ground_truth_is_reported_explicitly() -> None:
    # Without expected outputs or a scorer, outputs cannot be graded. The
    # comparison falls back to "completed without error" and says so.
    with patch.object(Prompt, "run", return_value="ok"):
        result = compare(
            {"control": Control, "treatment_a": TreatmentA},
            inputs=INPUTS,
            model="fake",
        )
    assert result.has_ground_truth is False
    sr = result.statistical_result
    assert sr is not None
    assert "error" not in sr.details
    assert sr.p_value == pytest.approx(1.0)
    assert "no expected outputs" in str(result).lower()


def test_assert_significant_uses_holm_adjusted_p() -> None:
    from flowprompt.testing.assertions import PromptTestResult

    result = _compare(
        {"control": set(), "treatment_a": set(range(6)), "treatment_b": set(range(6))},
        {"control": Control, "treatment_a": TreatmentA, "treatment_b": TreatmentB},
    )
    wrapped = PromptTestResult(result)
    assert wrapped.p_value == pytest.approx(0.0625)
    with pytest.raises(pytest.fail.Exception, match="adjusted p=0.0625"):
        wrapped.assert_significant()
