"""Known-answer and calibration tests for flowprompt.testing.paired."""

from __future__ import annotations

import math
import random

import pytest

from flowprompt.testing.paired import (
    SequentialMcNemar,
    agresti_min_interval,
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap_interval,
    paired_sign_flip_test,
    paired_t_interval,
    paired_test,
    plan_sample_size,
    wilson_interval,
)

# ---------------------------------------------------------------------------
# Exact McNemar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [
        (4, 0, 2 / 16),  # 2 * P(Bin(4,.5) <= 0)
        (0, 6, 2 / 64),
        (1, 9, 2 * 11 / 1024),  # 2 * (1 + 10) / 2^10
        (5, 5, 1.0),
        (0, 0, 1.0),
        (3, 7, 2 * (1 + 10 + 45 + 120) / 1024),
    ],
)
def test_mcnemar_exact_known_answers(b: int, c: int, expected: float) -> None:
    assert mcnemar_exact(b, c) == pytest.approx(expected, abs=1e-15)
    assert mcnemar_exact(c, b) == pytest.approx(expected, abs=1e-15)


def test_mcnemar_mid_p_known_answer() -> None:
    # mid-p = 2 * (P(X < 0) + 0.5 * P(X = 0)) = 2 * 0.5 / 16
    assert mcnemar_exact(4, 0, mid_p=True) == pytest.approx(0.0625)


def test_mcnemar_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_mcnemar_type_one_error_is_at_most_alpha() -> None:
    # Exact rejection probability under H0, summing over the discordant count.
    n, psi = 40, 0.3
    size = 0.0
    for n_d in range(n + 1):
        p_nd = math.comb(n, n_d) * psi**n_d * (1 - psi) ** (n - n_d)
        reject = sum(
            math.comb(n_d, c) / 2**n_d
            for c in range(n_d + 1)
            if mcnemar_exact(n_d - c, c) < 0.05
        )
        size += p_nd * reject
    assert size <= 0.05


# ---------------------------------------------------------------------------
# Sign-flip permutation test
# ---------------------------------------------------------------------------


def test_sign_flip_equals_mcnemar_for_single_run_pass_fail() -> None:
    rng = random.Random(1)
    for _ in range(50):
        n = rng.randint(1, 40)
        diffs = [rng.choice([-1, 0, 0, 1]) for _ in range(n)]
        b = diffs.count(-1)
        c = diffs.count(1)
        p, how, err = paired_sign_flip_test(diffs)
        assert how == "exact"
        assert err == 0.0
        assert p == pytest.approx(mcnemar_exact(b, c), abs=1e-12)


def test_sign_flip_hand_computed_grid_case() -> None:
    # Sign assignments of (1.5, 2.0, 0.5): totals 4,3,0,-1,1,0,-3,-4.
    # |total| >= 3 in 4 of 8 cases.
    p, how, _ = paired_sign_flip_test([1.5, 2.0, -0.5])
    assert how == "exact"
    assert p == pytest.approx(0.5)


def test_sign_flip_full_enumeration_off_grid() -> None:
    diffs = [math.pi / 10, math.e / 10, -math.sqrt(2) / 10]
    # Totals: pi+e-sqrt2 is the observed |sum|; by brute force:
    sums = []
    for mask in range(8):
        s = sum(d if (mask >> i) & 1 else -d for i, d in enumerate(diffs))
        sums.append(abs(s))
    expected = sum(1 for s in sums if s >= abs(sum(diffs)) - 1e-12) / 8
    p, how, _ = paired_sign_flip_test(diffs)
    assert how == "exact"
    assert p == pytest.approx(expected)


def test_sign_flip_monte_carlo_is_reproducible_and_reports_error() -> None:
    rng = random.Random(7)
    diffs = [rng.gauss(0.05, 0.3) for _ in range(40)]
    p1, how, err = paired_sign_flip_test(diffs)
    p2, _, _ = paired_sign_flip_test(diffs)
    assert how == "monte_carlo"
    assert p1 == p2
    assert 0 < err < 0.01
    # Agrees with the normal approximation to the permutation distribution.
    z = abs(sum(diffs)) / math.sqrt(sum(d * d for d in diffs))
    normal_p = math.erfc(z / math.sqrt(2))
    assert p1 == pytest.approx(normal_p, abs=0.03)


def test_sign_flip_all_ties() -> None:
    assert paired_sign_flip_test([0.0, 0.0, 0.0]) == (1.0, "exact", 0.0)


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


def test_agresti_min_known_answer() -> None:
    # n'=12, b'=4.5, c'=0.5: diff=-1/3, var=(5-16/12)/144
    lo, hi = agresti_min_interval(4, 0, 10)
    half = 1.959963984540054 * math.sqrt((5 - 16 / 12) / 144)
    assert lo == pytest.approx(-1 / 3 - half, abs=1e-6)
    assert hi == pytest.approx(-1 / 3 + half, abs=1e-6)


def test_wilson_known_answer() -> None:
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4)
    assert hi == pytest.approx(0.9433, abs=1e-4)


def test_paired_t_interval_known_answer() -> None:
    diffs = [-1.0] * 4 + [0.0] * 6
    lo, hi = paired_t_interval(diffs)
    se = math.sqrt((4 * 0.36 + 6 * 0.16) / 9 / 10)
    t975_df9 = 2.2621571627409915
    assert lo == pytest.approx(-0.4 - t975_df9 * se, abs=1e-6)
    assert hi == pytest.approx(-0.4 + t975_df9 * se, abs=1e-6)


def test_agresti_min_coverage() -> None:
    # Paired pass/fail with p10=0.10, p01=0.20 -> true difference +0.10.
    rng = random.Random(11)
    n, reps, covered = 40, 2000, 0
    for _ in range(reps):
        b = c = 0
        for _ in range(n):
            u = rng.random()
            if u < 0.10:
                b += 1
            elif u < 0.30:
                c += 1
        lo, hi = agresti_min_interval(b, c, n)
        covered += lo <= 0.10 <= hi
    coverage = covered / reps
    assert 0.93 <= coverage <= 0.98, coverage


def test_bootstrap_interval_coverage() -> None:
    rng = random.Random(5)
    n, reps, covered = 30, 300, 0
    for _ in range(reps):
        diffs = [rng.gauss(0.2, 1.0) for _ in range(n)]
        lo, hi = paired_bootstrap_interval(
            diffs, resamples=1000, seed=rng.randrange(10**9)
        )
        covered += lo <= 0.2 <= hi
    coverage = covered / reps
    # 3 Monte Carlo standard errors around 0.95 (BCa is slightly liberal).
    assert 0.89 <= coverage <= 0.99, coverage


def test_bootstrap_interval_degenerate_cases() -> None:
    assert paired_bootstrap_interval([0.25, 0.25, 0.25]) == (0.25, 0.25)
    lo, hi = paired_bootstrap_interval([1.0])
    assert math.isinf(lo) and math.isinf(hi)


# ---------------------------------------------------------------------------
# Holm
# ---------------------------------------------------------------------------


def test_holm_known_answers() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.03125, 0.03125]) == pytest.approx([0.0625, 0.0625])
    assert holm_adjust([0.5]) == [0.5]
    assert holm_adjust([0.9, 0.8]) == pytest.approx([1.0, 1.0])
    assert holm_adjust([0.04, 0.01, 0.2]) == pytest.approx([0.08, 0.03, 0.2])


# ---------------------------------------------------------------------------
# paired_test
# ---------------------------------------------------------------------------


def test_paired_test_single_run_uses_mcnemar() -> None:
    r = paired_test([1] * 8 + [0] * 2, [1] * 4 + [0] * 6)
    assert r.method == "mcnemar_exact"
    assert r.p_value == pytest.approx(0.125)
    assert r.difference == pytest.approx(-0.4)
    assert r.effect_size == pytest.approx(-0.4)
    assert r.relative_lift == pytest.approx(-0.5)
    assert r.details["control_only_correct"] == 4
    assert r.details["treatment_only_correct"] == 0
    assert r.n_inputs == 10


def test_paired_test_no_disagreement() -> None:
    r = paired_test([1, 0, 1], [1, 0, 1])
    assert r.p_value == 1.0
    assert "no disagreement" in r.details["note"]


def test_paired_test_repeats_are_aggregated_per_input() -> None:
    control = [[1, 1, 1]] * 8 + [[0, 0, 0]] * 2
    treatment = [[1, 1, 1]] * 4 + [[0, 0, 0]] * 6
    r = paired_test(control, treatment)
    assert r.method == "paired_permutation"
    assert r.p_value == pytest.approx(0.125)
    assert r.n_inputs == 10


def test_paired_test_relative_lift_none_when_control_zero() -> None:
    r = paired_test([0] * 10, [1] * 6 + [0] * 4)
    assert r.relative_lift is None
    assert r.difference == pytest.approx(0.6)
    assert r.significant


def test_paired_test_length_mismatch() -> None:
    with pytest.raises(ValueError):
        paired_test([1, 0], [1])


# ---------------------------------------------------------------------------
# Sample-size planning
# ---------------------------------------------------------------------------


def test_plan_known_answer_connor() -> None:
    za, zb = 1.959963984540054, 0.8416212335729143
    expected = (za * math.sqrt(0.2) + zb * math.sqrt(0.2 - 0.01)) ** 2 / 0.01
    plan = plan_sample_size(0.1, discordance=0.2)
    assert plan.n_inputs == math.ceil(expected) == 155
    assert plan.total_calls == 310
    assert plan.estimated_cost_usd is None
    assert "155 inputs" in plan.summary()


def test_plan_power_is_achieved_by_simulation() -> None:
    # p10=0.05, p01=0.15 -> discordance 0.2, difference 0.1.
    plan = plan_sample_size(0.1, discordance=0.2)
    rng = random.Random(3)
    reps, hits = 800, 0
    for _ in range(reps):
        b = c = 0
        for _ in range(plan.n_inputs):
            u = rng.random()
            if u < 0.05:
                b += 1
            elif u < 0.20:
                c += 1
        hits += mcnemar_exact(b, c) < 0.05
    power = hits / reps
    # Exact test is slightly conservative; allow 3 MC standard errors.
    assert 0.72 <= power <= 0.88, power


def test_plan_with_cost_and_variants() -> None:
    plan = plan_sample_size(0.1, discordance=0.2, n_variants=3, cost_per_call=0.001)
    # Bonferroni alpha = 0.025 per comparison makes n larger.
    assert plan.n_inputs > 155
    assert plan.alpha == pytest.approx(0.025)
    assert plan.estimated_cost_usd == pytest.approx(plan.total_calls * 0.001)


def test_plan_numeric_scores() -> None:
    plan = plan_sample_size(0.1, sd_difference=0.5)
    expected = ((1.959963984540054 + 0.8416212335729143) * 0.5 / 0.1) ** 2
    assert plan.n_inputs == math.ceil(expected)


def test_plan_default_discordance_is_documented_and_conservative() -> None:
    plan = plan_sample_size(0.1)
    assert "independent" in plan.assumptions
    assert plan.n_inputs > plan_sample_size(0.1, discordance=0.2).n_inputs


def test_plan_validation() -> None:
    with pytest.raises(ValueError):
        plan_sample_size(0.0)
    with pytest.raises(ValueError):
        plan_sample_size(0.1, power=1.5)
    with pytest.raises(ValueError):
        plan_sample_size(0.1, n_variants=1)


# ---------------------------------------------------------------------------
# Sequential McNemar
# ---------------------------------------------------------------------------


def test_sequential_known_answer() -> None:
    test = SequentialMcNemar(alpha=0.05)
    for _ in range(5):
        test.update(False, True)
    # M = 2^5 * B(6, 1) / B(1, 1) = 32 / 6
    assert test.evidence == pytest.approx(32 / 6)
    assert test.p_value == pytest.approx(6 / 32)
    assert not test.rejected
    # Concordant pairs leave the evidence unchanged.
    test.update(True, True)
    assert test.evidence == pytest.approx(32 / 6)


def test_sequential_rejects_after_strong_evidence() -> None:
    test = SequentialMcNemar(alpha=0.05)
    for _ in range(9):
        test.update(False, True)
    # M = 2^9 / 10 = 51.2 >= 20
    assert test.rejected
    assert test.stopped_at is not None


def test_sequential_false_positive_rate_under_continuous_monitoring() -> None:
    rng = random.Random(2024)
    reps, max_n, rejections = 1000, 300, 0
    for _ in range(reps):
        test = SequentialMcNemar(alpha=0.05)
        for _ in range(max_n):
            u = rng.random()
            control_ok = u < 0.15 or u >= 0.30 and u < 0.80
            treatment_ok = (0.15 <= u < 0.30) or (0.30 <= u < 0.80)
            if test.update(control_ok, treatment_ok):
                rejections += 1
                break
    rate = rejections / reps
    # Ville's inequality: <= alpha. Allow 3 MC standard errors.
    assert rate <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / reps), rate


def test_sequential_validation() -> None:
    with pytest.raises(ValueError):
        SequentialMcNemar(alpha=0)
    with pytest.raises(ValueError):
        SequentialMcNemar(prior_strength=0)
