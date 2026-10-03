"""Paired statistics for comparing prompt variants on the same inputs.

When every variant is evaluated on the same set of inputs, the observations
are *paired*: input 7 may be hard for every prompt. Treating the variants as
independent samples (as an unpaired two-proportion z-test does) throws that
information away and miscalibrates the p-value. The functions here work at
the level of the input, which is the unit that was actually sampled.

Methods
-------
* **Exact McNemar test** (pass/fail outcomes, one run per input). Only the
  discordant inputs (one variant right, the other wrong) carry information.
  Under the null hypothesis each discordant input is equally likely to favour
  either variant, so the count favouring the treatment is Binomial(n_d, 1/2).
  McNemar (1947), Psychometrika 12:153-157.
* **Paired sign-flip permutation test** (numeric scores, or repeated runs
  averaged per input). Under the null hypothesis of no difference the sign of
  each per-input difference is exchangeable. For pass/fail data with one run
  per input this test *is* the exact McNemar test. p-values are exact when the
  differences lie on a common grid (e.g. pass/fail averaged over a fixed number
  of runs), otherwise Monte Carlo (fixed seed) or a normal approximation for
  large samples. Fisher (1935); Good, *Permutation Tests* (2005).
* **Agresti-Min interval** for a difference of paired proportions: Wald
  interval after adding 1/2 to each cell of the 2x2 table. Agresti & Min
  (2005), Statistics in Medicine 24:729-740.
* **Paired t interval** for the mean per-input difference of numeric scores.
* **Holm-Bonferroni** step-down adjustment for several comparisons against a
  control. Holm (1979), Scandinavian Journal of Statistics 6:65-70.
* **Sample-size planning** for the paired design: Connor (1987), Biometrics
  43:207-211, for pass/fail outcomes; the usual paired normal formula for
  numeric scores.
* **Sequential (always-valid) McNemar test** for early stopping, based on a
  Beta-mixture likelihood ratio martingale and Ville's inequality. Robbins
  (1970), Ann. Math. Statist. 41:1397-1409; Howard, Ramdas, McAuliffe and
  Sekhon (2021), Ann. Statist. 49:1055-1080; Johari, Koomen, Pekelis and
  Walsh (2022), Operations Research 70:1806-1821.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from flowprompt.testing.statistics import (
    StatisticalResult,
    _normal_cdf,
    _normal_ppf,
    _t_cdf,
)

__all__ = [
    "SampleSizePlan",
    "SequentialMcNemar",
    "agresti_min_interval",
    "holm_adjust",
    "mcnemar_exact",
    "paired_bootstrap_interval",
    "paired_sign_flip_test",
    "paired_t_interval",
    "paired_test",
    "plan_sample_size",
    "wilson_interval",
]

_EXACT_GRID_BUDGET = 4_000_000
_MAX_EXACT_ENUMERATION = 16
_MONTE_CARLO_DRAWS = 20_000


# ---------------------------------------------------------------------------
# Elementary pieces
# ---------------------------------------------------------------------------


def _binom_cdf_half(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 1/2), computed exactly with integers."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) for i in range(k + 1)) / (1 << n)


def mcnemar_exact(b: int, c: int, *, mid_p: bool = False) -> float:
    """Two-sided exact McNemar p-value.

    Args:
        b: Inputs where the control was right and the treatment wrong.
        c: Inputs where the control was wrong and the treatment right.
        mid_p: Return the mid-p variant (less conservative; Fagerland,
            Lydersen and Laake 2013 recommend it for small samples).

    Returns:
        The two-sided p-value ``2 * P(X <= min(b, c))`` (capped at 1) with
        ``X ~ Binomial(b + c, 1/2)``.

    Example:
        >>> round(mcnemar_exact(4, 0), 4)
        0.125
    """
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    if mid_p:
        point = _binom_cdf_half(k, n) - _binom_cdf_half(k - 1, n)
        return min(1.0, 2.0 * (_binom_cdf_half(k - 1, n) + 0.5 * point))
    return min(1.0, 2.0 * _binom_cdf_half(k, n))


def wilson_interval(
    successes: int, n: int, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a single proportion."""
    if n <= 0:
        return (0.0, 1.0)
    z = _normal_ppf(1 - (1 - confidence_level) / 2)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def agresti_min_interval(
    b: int, c: int, n: int, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Confidence interval for p_treatment - p_control with paired data.

    Adds 1/2 to each of the four cells of the paired 2x2 table, then uses
    the Wald interval (Agresti & Min 2005). Good coverage even for small n.

    Args:
        b: Control right, treatment wrong.
        c: Control wrong, treatment right.
        n: Total number of paired inputs.
        confidence_level: Confidence level of the interval.
    """
    if n <= 0:
        return (-1.0, 1.0)
    z = _normal_ppf(1 - (1 - confidence_level) / 2)
    n_adj = n + 2.0
    b_adj = b + 0.5
    c_adj = c + 0.5
    diff = (c_adj - b_adj) / n_adj
    var = ((b_adj + c_adj) - (c_adj - b_adj) ** 2 / n_adj) / n_adj**2
    half = z * math.sqrt(max(var, 0.0))
    return (max(-1.0, diff - half), min(1.0, diff + half))


def _t_ppf(q: float, df: float) -> float:
    """Quantile of Student's t distribution (bisection on the CDF)."""
    if df > 1000:
        return _normal_ppf(q)
    lo, hi = -1000.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def paired_t_interval(
    differences: Sequence[float], confidence_level: float = 0.95
) -> tuple[float, float]:
    """t-based confidence interval for the mean of paired differences."""
    n = len(differences)
    if n == 0:
        return (-math.inf, math.inf)
    mean = sum(differences) / n
    if n == 1:
        return (-math.inf, math.inf)
    var = sum((d - mean) ** 2 for d in differences) / (n - 1)
    if var == 0:
        return (mean, mean)
    t = _t_ppf(1 - (1 - confidence_level) / 2, n - 1)
    half = t * math.sqrt(var / n)
    return (mean - half, mean + half)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (same order as input).

    Controls the family-wise error rate at alpha under arbitrary dependence
    between the tests.

    Example:
        >>> holm_adjust([0.01, 0.04, 0.03])
        [0.03, 0.06, 0.06]
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)
        adjusted[idx] = running
    return [round(p, 15) for p in adjusted]


# ---------------------------------------------------------------------------
# Paired sign-flip permutation test
# ---------------------------------------------------------------------------


def _common_grid(values: Sequence[float]) -> list[int] | None:
    """Return integer multiples if all values lie on a 1/k grid (k <= 60)."""
    for k in range(1, 61):
        ints = []
        for v in values:
            scaled = v * k
            r = round(scaled)
            if abs(scaled - r) > 1e-9:
                break
            ints.append(int(r))
        else:
            return ints
    return None


def paired_sign_flip_test(
    differences: Sequence[float],
    *,
    seed: int = 0,
    draws: int = _MONTE_CARLO_DRAWS,
) -> tuple[float, str, float]:
    """Two-sided paired sign-flip permutation test for a zero mean difference.

    Under the null hypothesis that the two variants are interchangeable on
    every input, the sign of each per-input difference is a fair coin flip.
    The p-value is the probability, over all sign assignments, of a total at
    least as extreme as the observed one. Inputs where both variants tie
    (difference 0) do not change the test statistic.

    The p-value is exact when the differences lie on a common grid (pass/fail
    scores averaged over a fixed number of runs) or when at most 16 inputs
    differ; otherwise it is estimated by Monte Carlo with a fixed seed.

    Args:
        differences: Per-input differences (treatment - control).
        seed: Seed for the Monte Carlo branch (results are reproducible).
        draws: Number of Monte Carlo sign assignments.

    Returns:
        ``(p_value, how, monte_carlo_error)`` where ``how`` is ``"exact"`` or
        ``"monte_carlo"`` and ``monte_carlo_error`` is the standard error of
        the Monte Carlo estimate (0.0 when exact).
    """
    nonzero = [d for d in differences if abs(d) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0, "exact", 0.0
    observed = abs(sum(nonzero))
    tol = 1e-9 * max(1.0, observed)

    grid = _common_grid(nonzero)
    if grid is not None:
        weights = [abs(w) for w in grid]
        span = sum(weights)
        if n * (2 * span + 1) <= _EXACT_GRID_BUDGET:
            # Distribution of sum(+-w_i) with fair signs, by convolution.
            dist = [0.0] * (2 * span + 1)
            dist[span] = 1.0
            for w in weights:
                new = [0.0] * (2 * span + 1)
                for idx, prob in enumerate(dist):
                    if prob:
                        new[idx - w] += 0.5 * prob
                        new[idx + w] += 0.5 * prob
                dist = new
            obs_int = abs(sum(grid))
            p = sum(prob for idx, prob in enumerate(dist) if abs(idx - span) >= obs_int)
            return min(1.0, p), "exact", 0.0

    if n <= _MAX_EXACT_ENUMERATION:
        count = 0
        total = 1 << n
        for mask in range(total):
            s = 0.0
            for i, d in enumerate(nonzero):
                s += d if (mask >> i) & 1 else -d
            if abs(s) >= observed - tol:
                count += 1
        return count / total, "exact", 0.0

    rng = random.Random(seed)
    hits = 0
    for _ in range(draws):
        bits = rng.getrandbits(n)
        s = 0.0
        for i, d in enumerate(nonzero):
            s += d if (bits >> i) & 1 else -d
        if abs(s) >= observed - tol:
            hits += 1
    p = (hits + 1) / (draws + 1)
    return p, "monte_carlo", math.sqrt(p * (1 - p) / draws)


def paired_bootstrap_interval(
    differences: Sequence[float],
    confidence_level: float = 0.95,
    *,
    resamples: int = 4000,
    seed: int = 0,
) -> tuple[float, float]:
    """BCa bootstrap confidence interval for the mean per-input difference.

    Resamples whole inputs (a cluster bootstrap when an input has several
    runs), then applies the bias-corrected and accelerated adjustment of
    Efron (1987), JASA 82:171-185. Inputs where the variants tie contribute
    a difference of 0 and are kept. Deterministic for a given seed.
    """
    n = len(differences)
    if n == 0:
        return (-math.inf, math.inf)
    mean = sum(differences) / n
    if n == 1 or all(abs(d - mean) < 1e-15 for d in differences):
        return (mean, mean) if n > 1 else (-math.inf, math.inf)

    rng = random.Random(seed)
    boots = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += differences[int(rng.random() * n)]
        boots.append(total / n)
    boots.sort()

    below = sum(1 for b in boots if b < mean) + 0.5 * sum(1 for b in boots if b == mean)
    frac = min(max(below / resamples, 1e-6), 1 - 1e-6)
    z0 = _normal_ppf(frac)

    total_sum = sum(differences)
    jack = [(total_sum - d) / (n - 1) for d in differences]
    jack_mean = sum(jack) / n
    num = sum((jack_mean - j) ** 3 for j in jack)
    den = 6.0 * (sum((jack_mean - j) ** 2 for j in jack) ** 1.5)
    accel = num / den if den > 0 else 0.0

    alpha = 1 - confidence_level

    def adjusted(q: float) -> float:
        z = _normal_ppf(q)
        return _normal_cdf(z0 + (z0 + z) / (1 - accel * (z0 + z)))

    def quantile(q: float) -> float:
        pos = min(max(q * (resamples - 1), 0.0), resamples - 1.0)
        lo = int(math.floor(pos))
        hi = min(lo + 1, resamples - 1)
        return boots[lo] + (boots[hi] - boots[lo]) * (pos - lo)

    return (quantile(adjusted(alpha / 2)), quantile(adjusted(1 - alpha / 2)))


# ---------------------------------------------------------------------------
# The test used by compare()
# ---------------------------------------------------------------------------


def _as_runs(values: Sequence[Any]) -> list[list[float]]:
    out: list[list[float]] = []
    for v in values:
        if isinstance(v, (list, tuple)):
            out.append([float(x) for x in v])
        else:
            out.append([float(v)])
    return out


def _is_binary(runs: list[list[float]]) -> bool:
    return all(x in (0.0, 1.0) for r in runs for x in r)


def paired_test(
    control: Sequence[float | Sequence[float]],
    treatment: Sequence[float | Sequence[float]],
    *,
    confidence_level: float = 0.95,
    control_name: str = "control",
    treatment_name: str = "treatment",
) -> StatisticalResult:
    """Compare two variants evaluated on the same inputs.

    Each element of ``control`` / ``treatment`` is the score for one input,
    or a list of scores from repeated runs on that input. Repeated runs are
    averaged per input first, so the input (not the run) is the unit of
    analysis.

    * Pass/fail scores with one run per input: exact McNemar test with an
      Agresti-Min confidence interval for the difference in accuracy.
    * Anything else: paired sign-flip permutation test on the per-input mean
      differences with a BCa bootstrap confidence interval (inputs are
      resampled as whole clusters, so repeated runs are not pseudo-replicated).

    Returns:
        A StatisticalResult whose ``difference`` is treatment minus control
        (in accuracy points for pass/fail data), with ``ci_low``/``ci_high``,
        ``p_value``, ``method`` and ``n_inputs`` filled in. ``effect_size``
        equals ``difference``; ``relative_lift`` is ``difference /
        control_mean`` or None when the control mean is 0.
    """
    c_runs = _as_runs(control)
    t_runs = _as_runs(treatment)
    if len(c_runs) != len(t_runs):
        raise ValueError("control and treatment must have one entry per input")
    if any(len(r) == 0 for r in c_runs + t_runs):
        raise ValueError("every input needs at least one score per variant")
    n = len(c_runs)
    if n == 0:
        return StatisticalResult(
            significant=False,
            p_value=1.0,
            confidence_level=confidence_level,
            test_name="paired",
            method="none",
            n_inputs=0,
            details={"note": "no inputs"},
        )

    c_means = [sum(r) / len(r) for r in c_runs]
    t_means = [sum(r) / len(r) for r in t_runs]
    diffs = [t - c for c, t in zip(c_means, t_means, strict=True)]
    c_mean = sum(c_means) / n
    t_mean = sum(t_means) / n
    difference = t_mean - c_mean
    alpha = 1 - confidence_level

    one_run = all(len(r) == 1 for r in c_runs + t_runs)
    binary = _is_binary(c_runs) and _is_binary(t_runs)
    details: dict[str, Any] = {
        "control_name": control_name,
        "treatment_name": treatment_name,
        "control_mean": c_mean,
        "treatment_mean": t_mean,
    }

    if one_run and binary:
        b = sum(
            1 for c, t in zip(c_means, t_means, strict=True) if c == 1.0 and t == 0.0
        )
        cc = sum(
            1 for c, t in zip(c_means, t_means, strict=True) if c == 0.0 and t == 1.0
        )
        p_value = mcnemar_exact(b, cc)
        ci = agresti_min_interval(b, cc, n, confidence_level)
        method = "mcnemar_exact"
        details.update(
            {
                "control_only_correct": b,
                "treatment_only_correct": cc,
                "both_correct": sum(
                    1
                    for c, t in zip(c_means, t_means, strict=True)
                    if c == 1.0 and t == 1.0
                ),
                "both_wrong": sum(
                    1
                    for c, t in zip(c_means, t_means, strict=True)
                    if c == 0.0 and t == 0.0
                ),
                "discordant": b + cc,
                "discordance_rate": (b + cc) / n,
                "ci_method": "agresti_min",
            }
        )
        if b + cc == 0:
            details["note"] = "no disagreement: both variants got the same inputs right"
    else:
        p_value, how, mc_error = paired_sign_flip_test(diffs)
        ci = paired_bootstrap_interval(diffs, confidence_level)
        method = "paired_permutation"
        sd = (
            math.sqrt(sum((d - difference) ** 2 for d in diffs) / (n - 1))
            if n > 1
            else 0.0
        )
        details.update(
            {
                "p_value_computation": how,
                "monte_carlo_error": mc_error,
                "sd_difference": sd,
                "runs_per_input": max(len(r) for r in c_runs + t_runs),
                "ci_method": "bca_bootstrap",
            }
        )

    relative_lift = difference / c_mean if c_mean != 0 else None
    return StatisticalResult(
        significant=p_value < alpha,
        p_value=p_value,
        confidence_level=confidence_level,
        effect_size=difference,
        confidence_interval=ci,
        test_name=method,
        details=details,
        method=method,
        difference=difference,
        ci_low=ci[0],
        ci_high=ci[1],
        n_inputs=n,
        relative_lift=relative_lift,
    )


# ---------------------------------------------------------------------------
# Sample-size planning
# ---------------------------------------------------------------------------


@dataclass
class SampleSizePlan:
    """How many inputs a paired comparison needs.

    Attributes:
        n_inputs: Inputs needed (each evaluated by every variant).
        min_detectable_difference: Absolute difference the plan targets.
        power: Target probability of detecting that difference.
        alpha: Significance level per comparison (after Bonferroni for
            several treatments).
        n_variants: Number of variants that will run on each input.
        runs_per_input: Runs per input and variant.
        total_calls: ``n_inputs * n_variants * runs_per_input``.
        estimated_cost_usd: Estimated cost of the calls, or None if no price
            is known.
        method: Formula used.
        assumptions: Plain-language statement of the assumptions.
    """

    n_inputs: int
    min_detectable_difference: float
    power: float
    alpha: float
    n_variants: int = 2
    runs_per_input: int = 1
    total_calls: int = 0
    estimated_cost_usd: float | None = None
    method: str = ""
    assumptions: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line plain-English summary."""
        text = (
            f"To detect a {self.min_detectable_difference * 100:.0f}-point difference "
            f"at {self.power:.0%} power you need ~{self.n_inputs} inputs"
        )
        if self.estimated_cost_usd is not None:
            text += f" (~{self.total_calls} calls, ≈ ${self.estimated_cost_usd:.2f})"
        else:
            text += f" (~{self.total_calls} calls)"
        return text

    def __str__(self) -> str:
        return self.summary()


def _price_per_call(
    model: str | None, input_tokens: int | None, output_tokens: int | None
) -> float | None:
    if not model or input_tokens is None or output_tokens is None:
        return None
    from flowprompt.core.usage import estimate_cost

    return estimate_cost(model, input_tokens, output_tokens)


def plan_sample_size(
    min_detectable_difference: float = 0.1,
    *,
    discordance: float | None = None,
    baseline_accuracy: float | None = None,
    power: float = 0.8,
    alpha: float = 0.05,
    sd_difference: float | None = None,
    n_variants: int = 2,
    runs_per_input: int = 1,
    cost_per_call: float | None = None,
    model: str | None = None,
    input_tokens_per_call: int | None = None,
    output_tokens_per_call: int | None = None,
) -> SampleSizePlan:
    """Plan how many inputs a paired prompt comparison needs.

    For pass/fail outcomes the paired-proportions formula of Connor (1987)
    is used::

        n = (z_{1-a/2} * sqrt(psi) + z_{power} * sqrt(psi - d^2))^2 / d^2

    where ``d`` is the difference to detect and ``psi`` the discordance rate
    (fraction of inputs where the two variants disagree). The discordance
    matters as much as the accuracies: two prompts that fail on the same hard
    inputs disagree rarely and need far fewer inputs. Estimate it from a
    small pilot (``ComparisonResult.sample_size_plan`` uses the discordance it
    observed). If it is not given, it is computed from ``baseline_accuracy``
    (default 0.5) assuming the variants err independently, which is the
    conservative choice for prompts that are positively correlated.

    For numeric scores pass ``sd_difference`` (standard deviation of the
    per-input differences) instead; then ``n = ((z_{1-a/2} + z_{power}) *
    sd / d)^2``.

    With more than two variants, ``alpha`` is split across the
    ``n_variants - 1`` comparisons against the control (Bonferroni), a
    conservative stand-in for the Holm procedure used in the analysis.

    Args:
        min_detectable_difference: Absolute difference to detect, e.g. 0.1
            for 10 accuracy points.
        discordance: Expected fraction of inputs where the variants disagree.
        baseline_accuracy: Expected accuracy of the control, used only to
            derive a default discordance (default 0.5).
        power: Target power (default 0.8).
        alpha: Two-sided significance level (default 0.05).
        sd_difference: For numeric scores: SD of per-input differences.
        n_variants: Variants evaluated on each input (default 2).
        runs_per_input: Runs per input and variant (default 1). Repeats do
            not reduce ``n_inputs``; they only add calls.
        cost_per_call: Known average cost of one call in USD.
        model: Model name to look up a price (litellm cost map) when
            ``cost_per_call`` is not given.
        input_tokens_per_call: Average prompt tokens per call (for pricing).
        output_tokens_per_call: Average completion tokens per call.

    Returns:
        A SampleSizePlan. ``estimated_cost_usd`` is None when no price is
        known.

    Example:
        >>> plan_sample_size(0.1, discordance=0.2).n_inputs
        155
    """
    d = abs(min_detectable_difference)
    if not 0 < d <= 1 and sd_difference is None:
        raise ValueError("min_detectable_difference must be in (0, 1]")
    if d == 0:
        raise ValueError("min_detectable_difference must be non-zero")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if n_variants < 2:
        raise ValueError("n_variants must be at least 2")

    comparisons = n_variants - 1
    alpha_each = alpha / comparisons
    z_a = _normal_ppf(1 - alpha_each / 2)
    z_b = _normal_ppf(power)

    if sd_difference is not None:
        n_float = ((z_a + z_b) * sd_difference / d) ** 2
        method = "paired normal approximation"
        assumptions = (
            f"SD of per-input differences = {sd_difference:.3g}; "
            f"alpha = {alpha_each:.3g} per comparison"
        )
        psi = None
    else:
        if discordance is None:
            p1 = 0.5 if baseline_accuracy is None else baseline_accuracy
            p2 = min(1.0, p1 + d)
            if p2 - p1 < d - 1e-12:
                p2 = max(0.0, p1 - d)
            psi = p1 * (1 - p2) + p2 * (1 - p1)
            assumptions = (
                f"control accuracy {p1:.0%}; variants assumed to err "
                f"independently (discordance {psi:.0%}, conservative)"
            )
        else:
            psi = discordance
            assumptions = f"discordance rate {psi:.0%}"
        psi = max(psi, d)
        if psi > 1:
            raise ValueError("discordance must be <= 1")
        n_float = (z_a * math.sqrt(psi) + z_b * math.sqrt(max(psi - d * d, 0.0))) ** 2
        n_float /= d * d
        method = "Connor (1987) paired proportions"
        assumptions += f"; alpha = {alpha_each:.3g} per comparison"

    n_inputs = max(2, int(math.ceil(n_float - 1e-9)))
    total_calls = n_inputs * n_variants * runs_per_input

    price = cost_per_call
    if price is None:
        price = _price_per_call(model, input_tokens_per_call, output_tokens_per_call)
    cost = price * total_calls if price is not None else None

    return SampleSizePlan(
        n_inputs=n_inputs,
        min_detectable_difference=d,
        power=power,
        alpha=alpha_each,
        n_variants=n_variants,
        runs_per_input=runs_per_input,
        total_calls=total_calls,
        estimated_cost_usd=cost,
        method=method,
        assumptions=assumptions,
        details={"discordance": psi, "z_alpha": z_a, "z_power": z_b},
    )


# ---------------------------------------------------------------------------
# Sequential testing
# ---------------------------------------------------------------------------


class SequentialMcNemar:
    """Always-valid McNemar test you may check after every input.

    Classical p-values are only valid if you look once, at a sample size
    fixed in advance. "Peeking" after every batch and stopping at the first
    p < 0.05 inflates the false-positive rate far beyond 5%. This class
    tracks a test martingale instead:

        M_n = Integral prod_i [theta^x_i (1 - theta)^(1 - x_i) / (1/2)] dBeta(theta; a, a)
            = 2^n_d * B(a + c, a + b) / B(a, a)

    where ``b``/``c`` count discordant inputs favouring the control/treatment
    and ``n_d = b + c``. Under the null hypothesis (no difference between
    variants) ``M_n`` is a non-negative martingale with ``M_0 = 1``, so by
    Ville's inequality ``P(sup_n M_n >= 1/alpha) <= alpha``. Stopping the
    first time ``M_n >= 1/alpha`` therefore keeps the false-positive rate at
    most alpha no matter how often you look. The always-valid p-value is
    ``min(1, 1 / max_k M_k)``.

    The price of continuous monitoring is power: for a fixed sample size the
    sequential test needs more data than the one-look exact McNemar test.

    Args:
        alpha: Significance level (default 0.05).
        prior_strength: ``a`` in the symmetric Beta(a, a) mixing prior
            (default 1, uniform). Larger values favour detecting smaller
            effects later; smaller values favour large effects early.

    Example:
        >>> test = SequentialMcNemar(alpha=0.05)
        >>> for control_ok, treatment_ok in [(False, True)] * 9:
        ...     test.update(control_ok, treatment_ok)
        >>> test.rejected
        True
    """

    def __init__(self, alpha: float = 0.05, prior_strength: float = 1.0) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if prior_strength <= 0:
            raise ValueError("prior_strength must be positive")
        self.alpha = alpha
        self.prior_strength = prior_strength
        self.n = 0
        self.control_only = 0
        self.treatment_only = 0
        self._max_log_m = 0.0
        self.stopped_at: int | None = None

    @property
    def log_evidence(self) -> float:
        """log M_n for the current data."""
        a = self.prior_strength
        b, c = self.control_only, self.treatment_only
        n_d = b + c

        def log_beta(x: float, y: float) -> float:
            return math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)

        return n_d * math.log(2.0) + log_beta(a + c, a + b) - log_beta(a, a)

    @property
    def evidence(self) -> float:
        """The test martingale M_n (likelihood-ratio evidence against H0)."""
        return math.exp(min(self.log_evidence, 700.0))

    @property
    def p_value(self) -> float:
        """Always-valid p-value: min(1, 1 / max_k M_k)."""
        return min(1.0, math.exp(-self._max_log_m))

    @property
    def rejected(self) -> bool:
        """True once the evidence has crossed 1/alpha at any look."""
        return self._max_log_m >= math.log(1 / self.alpha)

    @property
    def difference(self) -> float:
        """Observed accuracy difference (treatment - control) so far."""
        if self.n == 0:
            return 0.0
        return (self.treatment_only - self.control_only) / self.n

    def update(self, control_correct: bool, treatment_correct: bool) -> bool:
        """Add one paired observation and return ``rejected``."""
        self.n += 1
        if control_correct and not treatment_correct:
            self.control_only += 1
        elif treatment_correct and not control_correct:
            self.treatment_only += 1
        log_m = self.log_evidence
        if log_m > self._max_log_m:
            self._max_log_m = log_m
        if self.rejected and self.stopped_at is None:
            self.stopped_at = self.n
        return self.rejected

    def __repr__(self) -> str:
        return (
            f"SequentialMcNemar(n={self.n}, b={self.control_only}, "
            f"c={self.treatment_only}, p={self.p_value:.4f}, "
            f"rejected={self.rejected})"
        )
