"""Statistical analysis for A/B testing.

Provides statistical significance testing:
- Two-proportion z-test
- Chi-squared test
- T-test for means
- Bayesian A/B analysis
- Sequential testing
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from flowprompt.testing.experiment import VariantStats

# Type alias for significance test functions
SignificanceTestFunc = Callable[
    [VariantStats, VariantStats, float], "StatisticalResult"
]


@dataclass
class StatisticalResult:
    """Result of statistical significance test.

    Attributes:
        significant: Whether the result is statistically significant (using
            ``adjusted_p`` when a multiple-comparison correction was applied).
        p_value: Raw (unadjusted) p-value of the test.
        confidence_level: Confidence level used.
        effect_size: Estimated effect size. For the paired tests used by
            ``compare()`` this is the absolute difference (treatment minus
            control, e.g. 0.12 = 12 accuracy points). The legacy unpaired
            tests report the relative improvement here.
        confidence_interval: Confidence interval for the difference.
        power: Statistical power of the test.
        sample_size_recommendation: Recommended sample size if not significant.
        test_name: Name of the statistical test used.
        details: Additional test details.
        method: Name of the method (e.g. ``"mcnemar_exact"``).
        difference: Treatment minus control (absolute).
        ci_low: Lower confidence bound for ``difference``.
        ci_high: Upper confidence bound for ``difference``.
        n_inputs: Number of paired inputs the test is based on.
        adjusted_p: p-value after the multiple-comparison correction
            (Holm), or None when there was a single comparison.
        relative_lift: ``difference / control_rate``; None when undefined.
        control: Name of the control variant.
        treatment: Name of the treatment variant.
    """

    significant: bool
    p_value: float
    confidence_level: float = 0.95
    effect_size: float = 0.0
    confidence_interval: tuple[float, float] | None = None
    power: float | None = None
    sample_size_recommendation: int | None = None
    test_name: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    method: str = ""
    difference: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_inputs: int | None = None
    adjusted_p: float | None = None
    relative_lift: float | None = None
    control: str | None = None
    treatment: str | None = None

    def summary(self) -> str:
        """Generate a human-readable summary."""
        status = "SIGNIFICANT" if self.significant else "NOT SIGNIFICANT"
        lines = [
            f"Statistical Result ({self.method or self.test_name}):",
            f"  Status: {status}",
            f"  P-value: {self.p_value:.4f}",
        ]
        if self.adjusted_p is not None:
            lines.append(f"  Adjusted p-value (Holm): {self.adjusted_p:.4f}")
        lines.append(f"  Confidence Level: {self.confidence_level:.0%}")
        if self.difference is not None:
            ci = ""
            if self.ci_low is not None and self.ci_high is not None:
                ci = f" [{self.ci_low:+.3f}, {self.ci_high:+.3f}]"
            lines.append(f"  Difference: {self.difference:+.3f}{ci}")
        else:
            lines.append(f"  Effect Size: {self.effect_size:+.2%}")
        if self.power:
            lines.append(f"  Power: {self.power:.2%}")
        return "\n".join(lines)


def _normal_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal distribution."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _normal_ppf(p: float) -> float:
    """Percent point function (inverse CDF) for standard normal distribution.

    Uses approximation for inverse error function.
    """
    if p <= 0:
        return float("-inf")
    if p >= 1:
        return float("inf")
    if p == 0.5:
        return 0.0

    # Use rational approximation
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def two_proportion_z_test(
    control_stats: VariantStats,
    treatment_stats: VariantStats,
    confidence_level: float = 0.95,
) -> StatisticalResult:
    """Two-proportion z-test for comparing conversion rates.

    Tests whether the treatment variant has a significantly different
    success rate than the control variant.

    Args:
        control_stats: Statistics for the control variant.
        treatment_stats: Statistics for the treatment variant.
        confidence_level: Confidence level (e.g., 0.95 for 95%).

    Returns:
        StatisticalResult with test results.
    """
    n1 = control_stats.samples
    n2 = treatment_stats.samples
    p1 = control_stats.success_rate
    p2 = treatment_stats.success_rate

    if n1 == 0 or n2 == 0:
        return StatisticalResult(
            significant=False,
            p_value=1.0,
            confidence_level=confidence_level,
            effect_size=0.0,
            test_name="two_proportion_z_test",
            details={"error": "Insufficient samples"},
        )

    # Pooled proportion
    p_pooled = (control_stats.successes + treatment_stats.successes) / (n1 + n2)

    # Standard error
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))

    if se == 0:
        return StatisticalResult(
            significant=False,
            p_value=1.0,
            confidence_level=confidence_level,
            effect_size=0.0,
            test_name="two_proportion_z_test",
            details={"error": "Zero standard error"},
        )

    # Z-statistic
    z = (p2 - p1) / se

    # Two-tailed p-value
    p_value = 2 * (1 - _normal_cdf(abs(z)))

    # Effect size (relative improvement)
    effect_size = (p2 - p1) / p1 if p1 > 0 else 0.0

    # Significance threshold
    alpha = 1 - confidence_level
    significant = p_value < alpha

    # Calculate power
    z_alpha = _normal_ppf(1 - alpha / 2)
    power = _calculate_power(n1, n2, p1, p2, z_alpha)

    # Sample size recommendation if not significant
    sample_recommendation = None
    if not significant:
        sample_recommendation = _recommend_sample_size(p1, p2, confidence_level, 0.8)

    return StatisticalResult(
        significant=significant,
        p_value=p_value,
        confidence_level=confidence_level,
        effect_size=effect_size,
        power=power,
        sample_size_recommendation=sample_recommendation,
        test_name="two_proportion_z_test",
        details={
            "z_statistic": z,
            "control_rate": p1,
            "treatment_rate": p2,
            "pooled_rate": p_pooled,
            "standard_error": se,
        },
    )


def chi_squared_test(
    control_stats: VariantStats,
    treatment_stats: VariantStats,
    confidence_level: float = 0.95,
) -> StatisticalResult:
    """Chi-squared test for independence.

    Tests whether success/failure is independent of variant assignment.

    Args:
        control_stats: Statistics for the control variant.
        treatment_stats: Statistics for the treatment variant.
        confidence_level: Confidence level.

    Returns:
        StatisticalResult with test results.
    """
    # Observed counts
    o11 = control_stats.successes
    o12 = control_stats.samples - control_stats.successes
    o21 = treatment_stats.successes
    o22 = treatment_stats.samples - treatment_stats.successes

    total = o11 + o12 + o21 + o22
    if total == 0:
        return StatisticalResult(
            significant=False,
            p_value=1.0,
            confidence_level=confidence_level,
            effect_size=0.0,
            test_name="chi_squared_test",
            details={"error": "No samples"},
        )

    # Row and column totals
    row1 = o11 + o12
    row2 = o21 + o22
    col1 = o11 + o21
    col2 = o12 + o22

    # Expected counts
    e11 = row1 * col1 / total
    e12 = row1 * col2 / total
    e21 = row2 * col1 / total
    e22 = row2 * col2 / total

    # Chi-squared statistic with Yates' correction
    chi2 = 0.0
    for o, e in [(o11, e11), (o12, e12), (o21, e21), (o22, e22)]:
        if e > 0:
            chi2 += (abs(o - e) - 0.5) ** 2 / e

    # P-value from chi-squared distribution (df=1)
    # Using approximation for chi-squared CDF
    p_value = 1 - _chi2_cdf(chi2, df=1)

    # Effect size (Cramer's V)
    cramer_v = math.sqrt(chi2 / total) if total > 0 else 0.0

    # Relative effect
    p1 = control_stats.success_rate
    p2 = treatment_stats.success_rate
    effect_size = (p2 - p1) / p1 if p1 > 0 else 0.0

    alpha = 1 - confidence_level
    significant = p_value < alpha

    return StatisticalResult(
        significant=significant,
        p_value=p_value,
        confidence_level=confidence_level,
        effect_size=effect_size,
        test_name="chi_squared_test",
        details={
            "chi2_statistic": chi2,
            "cramers_v": cramer_v,
            "control_rate": p1,
            "treatment_rate": p2,
        },
    )


def _chi2_cdf(x: float, df: int = 1) -> float:
    """Cumulative distribution function for chi-squared distribution.

    Uses regularized incomplete gamma function approximation.
    """
    if x <= 0:
        return 0.0

    # For df=1, use simpler formula
    if df == 1:
        return 2 * _normal_cdf(math.sqrt(x)) - 1

    # General case using series expansion
    k = df / 2
    x_half = x / 2

    # Regularized gamma function approximation
    return _regularized_gamma_p(k, x_half)


def _regularized_gamma_p(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function P(a, x)."""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0

    # Series expansion for P(a, x)
    if x < a + 1:
        # Use series expansion
        ap = a
        sum_val = 1.0 / a
        delta = sum_val
        for _ in range(100):
            ap += 1
            delta *= x / ap
            sum_val += delta
            if abs(delta) < abs(sum_val) * 1e-10:
                break
        return sum_val * math.exp(-x + a * math.log(x) - math.lgamma(a))
    else:
        # Use continued fraction
        return 1.0 - _regularized_gamma_q(a, x)


def _regularized_gamma_q(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function Q(a, x)."""
    # Continued fraction representation
    b = x + 1 - a
    c = 1e30
    d = 1 / b
    h = d

    for i in range(1, 100):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-10:
            break

    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def t_test_means(
    control_stats: VariantStats,
    treatment_stats: VariantStats,
    confidence_level: float = 0.95,
) -> StatisticalResult:
    """Welch's t-test for comparing means.

    Tests whether the treatment has a significantly different mean
    metric value than the control.

    Args:
        control_stats: Statistics for the control variant.
        treatment_stats: Statistics for the treatment variant.
        confidence_level: Confidence level.

    Returns:
        StatisticalResult with test results.
    """
    n1 = control_stats.samples
    n2 = treatment_stats.samples
    m1 = control_stats.mean_metric
    m2 = treatment_stats.mean_metric
    s1 = control_stats.std_metric
    s2 = treatment_stats.std_metric

    if n1 < 2 or n2 < 2:
        return StatisticalResult(
            significant=False,
            p_value=1.0,
            confidence_level=confidence_level,
            effect_size=0.0,
            test_name="welch_t_test",
            details={"error": "Insufficient samples (need at least 2 per group)"},
        )

    # Welch's t-statistic
    se = math.sqrt(s1**2 / n1 + s2**2 / n2)

    if se == 0:
        return StatisticalResult(
            significant=False,
            p_value=1.0,
            confidence_level=confidence_level,
            effect_size=0.0,
            test_name="welch_t_test",
            details={"error": "Zero standard error"},
        )

    t_stat = (m2 - m1) / se

    # Welch-Satterthwaite degrees of freedom
    num = (s1**2 / n1 + s2**2 / n2) ** 2
    denom = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
    df = num / denom if denom > 0 else 1

    # P-value from t-distribution (using normal approximation for large df)
    if df > 100:
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    else:
        p_value = 2 * _t_cdf(-abs(t_stat), df)

    # Effect size (Cohen's d)
    pooled_std = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohens_d = (m2 - m1) / pooled_std if pooled_std > 0 else 0.0

    # Relative effect
    effect_size = (m2 - m1) / m1 if m1 != 0 else 0.0

    alpha = 1 - confidence_level
    significant = p_value < alpha

    return StatisticalResult(
        significant=significant,
        p_value=p_value,
        confidence_level=confidence_level,
        effect_size=effect_size,
        test_name="welch_t_test",
        details={
            "t_statistic": t_stat,
            "degrees_of_freedom": df,
            "cohens_d": cohens_d,
            "control_mean": m1,
            "treatment_mean": m2,
            "control_std": s1,
            "treatment_std": s2,
        },
    )


def _t_cdf(x: float, df: float) -> float:
    """Cumulative distribution function for t-distribution."""
    # Use regularized incomplete beta function
    t2 = x * x
    p = df / (df + t2)

    if x < 0:
        return 0.5 * _regularized_beta(df / 2, 0.5, p)
    else:
        return 1 - 0.5 * _regularized_beta(df / 2, 0.5, p)


def _regularized_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use continued fraction representation
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1 - x)
    )

    if x < (a + 1) / (a + b + 2):
        return bt * _beta_cf(a, b, x) / a
    else:
        return 1 - bt * _beta_cf(b, a, 1 - x) / b


def _beta_cf(a: float, b: float, x: float) -> float:
    """Continued fraction for incomplete beta function."""
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1 / d
    h = d

    for m in range(1, 100):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        delta = d * c
        h *= delta

        if abs(delta - 1) < 1e-10:
            break

    return h


def _calculate_power(
    n1: int,
    n2: int,
    p1: float,
    p2: float,
    z_alpha: float,
) -> float:
    """Calculate statistical power for two-proportion test."""
    if n1 == 0 or n2 == 0:
        return 0.0

    p_pooled = (n1 * p1 + n2 * p2) / (n1 + n2)
    se_null = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
    se_alt = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

    if se_null == 0 or se_alt == 0:
        return 0.0

    effect = abs(p2 - p1)
    z_power = (effect - z_alpha * se_null) / se_alt

    return _normal_cdf(z_power)


def _recommend_sample_size(
    p1: float,
    p2: float,
    confidence_level: float,
    power: float,
) -> int:
    """Recommend sample size per group for desired power."""
    alpha = 1 - confidence_level
    z_alpha = _normal_ppf(1 - alpha / 2)
    z_beta = _normal_ppf(power)

    p_bar = (p1 + p2) / 2
    effect = abs(p2 - p1)

    if effect == 0:
        return 10000  # Default large value

    # Sample size formula for two-proportion test
    n = 2 * p_bar * (1 - p_bar) * (z_alpha + z_beta) ** 2 / effect**2

    return int(math.ceil(n))


def bayesian_ab_test(
    control_stats: VariantStats,
    treatment_stats: VariantStats,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    num_samples: int = 10000,
) -> StatisticalResult:
    """Bayesian A/B test using Beta-Binomial model.

    Computes the probability that treatment is better than control
    using Monte Carlo sampling.

    Args:
        control_stats: Statistics for the control variant.
        treatment_stats: Statistics for the treatment variant.
        prior_alpha: Prior alpha for Beta distribution.
        prior_beta: Prior beta for Beta distribution.
        num_samples: Number of Monte Carlo samples.

    Returns:
        StatisticalResult with Bayesian analysis results.
    """
    import random

    # Posterior parameters
    alpha_c = prior_alpha + control_stats.successes
    beta_c = prior_beta + (control_stats.samples - control_stats.successes)
    alpha_t = prior_alpha + treatment_stats.successes
    beta_t = prior_beta + (treatment_stats.samples - treatment_stats.successes)

    # Monte Carlo sampling
    treatment_wins = 0
    control_wins = 0
    relative_lifts: list[float] = []

    for _ in range(num_samples):
        # Sample from posteriors
        p_c = random.betavariate(alpha_c, beta_c)
        p_t = random.betavariate(alpha_t, beta_t)

        if p_t > p_c:
            treatment_wins += 1
        elif p_c > p_t:
            control_wins += 1

        if p_c > 0:
            relative_lifts.append((p_t - p_c) / p_c)

    # Probability treatment is better
    prob_treatment_better = treatment_wins / num_samples

    # Expected relative lift
    expected_lift = sum(relative_lifts) / len(relative_lifts) if relative_lifts else 0.0

    # 95% credible interval for lift
    relative_lifts.sort()
    ci_lower = (
        relative_lifts[int(0.025 * len(relative_lifts))] if relative_lifts else 0.0
    )
    ci_upper = (
        relative_lifts[int(0.975 * len(relative_lifts))] if relative_lifts else 0.0
    )

    return StatisticalResult(
        significant=prob_treatment_better > 0.95 or prob_treatment_better < 0.05,
        p_value=1
        - prob_treatment_better,  # Not a true p-value, but useful for comparison
        confidence_level=0.95,
        effect_size=expected_lift,
        test_name="bayesian_ab_test",
        details={
            "prob_treatment_better": prob_treatment_better,
            "expected_lift": expected_lift,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "posterior_control_alpha": alpha_c,
            "posterior_control_beta": beta_c,
            "posterior_treatment_alpha": alpha_t,
            "posterior_treatment_beta": beta_t,
        },
    )


def run_significance_test(
    control_stats: VariantStats,
    treatment_stats: VariantStats,
    test_type: str = "z_test",
    confidence_level: float = 0.95,
) -> StatisticalResult:
    """Run a significance test between control and treatment.

    Args:
        control_stats: Statistics for the control variant.
        treatment_stats: Statistics for the treatment variant.
        test_type: Type of test ("z_test", "chi_squared", "t_test", "bayesian").
        confidence_level: Confidence level.

    Returns:
        StatisticalResult with test results.
    """
    tests: dict[str, SignificanceTestFunc] = {
        "z_test": two_proportion_z_test,
        "chi_squared": chi_squared_test,
        "t_test": t_test_means,
        "bayesian": lambda c, t, conf: bayesian_ab_test(c, t),  # noqa: ARG005
    }

    if test_type not in tests:
        raise ValueError(
            f"Unknown test type: {test_type}. Available: {list(tests.keys())}"
        )

    return tests[test_type](control_stats, treatment_stats, confidence_level)
