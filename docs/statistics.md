# Statistical methods

This page documents exactly what `compare()` computes, so you can check it
and cite it. For the motivation, with simulations, read
[Your prompt A/B test is probably lying to you](false-winners.md).

## The unit of analysis is the input

`compare()` runs every variant on the same inputs, so the data are
**paired**: input 17 may be hard for every prompt. Each variant gets one
score per input. With `runs_per_input > 1` the runs are averaged per input
first, so an input run five times still counts once. Every test below works
on these per-input scores, and `n_inputs` in the result is the sample size.

Errors raised by a variant score 0 for that run and are counted in
`VariantResult.errors`.

## Which test is used

| Data | Test | Interval for the difference |
|---|---|---|
| pass/fail, one run per input | exact McNemar | Agresti-Min |
| pass/fail with repeats, or numeric scores | paired sign-flip permutation | BCa bootstrap over inputs |

The choice is made from the data (`test_type="auto"`, the default). For
pass/fail data with one run per input both tests give the same p-value: the
sign-flip test on differences in {-1, 0, +1} *is* the exact McNemar test.

### Exact McNemar test

Let `b` be the inputs where the control is right and the treatment wrong,
and `c` the reverse. Inputs where both agree carry no information about
which is better. Under the null hypothesis each of the `b + c` discordant
inputs favours either variant with probability 1/2, so

    p = min(1, 2 * P(X <= min(b, c))),   X ~ Binomial(b + c, 1/2)

computed exactly with integer arithmetic. If `b + c = 0` then `p = 1` and the
result notes that the variants never disagreed. The exact test is
conservative (its false-positive rate is at most alpha, usually below);
`mcnemar_exact(b, c, mid_p=True)` gives the mid-p version recommended by
Fagerland et al. (2013) when you prefer a rate closer to alpha.

Worked example: control right on 8/10, treatment on 4/10, with `b = 4`,
`c = 0`: `p = 2 * (1/2)^4 = 0.125`. An unpaired two-proportion z-test on the
same data reports 0.068.

### Agresti-Min interval

For the difference in accuracy `(c - b) / n`, FlowPrompt adds 1/2 to each
cell of the paired 2x2 table and uses the Wald interval on the adjusted
table (Agresti and Min, 2005). It has good coverage even for small `n`
(our test suite checks about 95% coverage by simulation).

The interval and the exact test are different procedures. In small samples
they can disagree near the boundary (an interval that just excludes 0 with
`p` slightly above 0.05). The verdict always follows the test, which never
exceeds the stated false-positive rate.

### Paired sign-flip permutation test

For per-input differences `d_i = treatment_i - control_i`, the null
hypothesis is that the two variants are interchangeable on every input, so
each `d_i` is equally likely to have either sign. The p-value is the share
of the `2^n` sign assignments whose total is at least as extreme as the
observed `|sum(d_i)|`. Inputs where the variants tie (`d_i = 0`) do not
change the statistic but are kept for the interval.

- **Exact** when the differences lie on a common grid, which they do for
  pass/fail scores averaged over a fixed number of runs (dynamic
  programming over the attainable totals), or when at most 16 inputs
  differ (full enumeration).
- **Monte Carlo** otherwise: 20,000 random sign assignments with a fixed
  seed, `p = (hits + 1) / (draws + 1)`. The Monte Carlo standard error is
  reported in `details["monte_carlo_error"]`.

### BCa bootstrap interval

Inputs are resampled with replacement (4,000 resamples, fixed seed), so
repeated runs of an input always move together: a cluster bootstrap. The
percentile interval is corrected for bias and skewness with the BCa method
(Efron, 1987). Results are deterministic.

### Per-variant intervals

Accuracy with one run per input uses the Wilson score interval. Repeated
runs or numeric scores use a t interval over the per-input means.

## Several variants: Holm-Bonferroni

With more than two variants, each treatment is compared with the control
(`comparisons="control"`, the default) or every pair is compared
(`comparisons="all"`). The raw p-values are adjusted with Holm's step-down
procedure, which controls the family-wise error rate at alpha under any
dependence between the comparisons and is never less powerful than
Bonferroni. Both `p_value` (raw) and `adjusted_p` are reported;
`significant` uses `adjusted_p`.

## How the winner is chosen

- `comparisons="control"`: among treatments that are significantly better
  than the control (after adjustment), the one with the largest
  difference. If no treatment is better but every treatment is
  significantly worse, the control wins. Otherwise there is no winner.
  Treatments are not tested against each other in this mode; the result
  says so when several beat the control.
- `comparisons="all"`: the variant that is significantly better than every
  other variant, if there is one.

The decision uses the sign of the absolute difference. `relative_lift`
(difference divided by the control's score) is reported for information
and is `None` when the control scores 0.

## When there is not enough data

With `n` inputs, the smallest p-value any paired test can produce is
`2 / 2^n` (every input favours the same variant), and Holm multiplies it by
up to the number of comparisons. If even that cannot reach alpha, the
verdict is "Not enough data" with the minimum number of inputs needed
(6 inputs for two variants at alpha = 0.05).

## Sample size

For pass/fail outcomes, `plan_sample_size` uses Connor's (1987) formula for
paired proportions:

    n = (z_{1-a/2} * sqrt(psi) + z_{power} * sqrt(psi - d^2))^2 / d^2

where `d` is the difference to detect and `psi` the discordance rate. Pass a
discordance measured in a pilot (`result.sample_size_plan()` does this
automatically). Without one, `psi` is computed from `baseline_accuracy`
(default 0.5) assuming independent errors, which overstates `n` for prompts
that fail on the same inputs. For numeric scores, pass the standard
deviation of per-input differences: `n = ((z_{1-a/2} + z_{power}) * sd / d)^2`.
With `k` variants, alpha is divided by `k - 1` (Bonferroni, a conservative
stand-in for Holm).

## Sequential testing

`SequentialMcNemar` lets you look after every input and stop as soon as the
evidence is strong, without inflating the false-positive rate. It tracks
the mixture likelihood ratio

    M_n = 2^(b + c) * B(a + c, a + b) / B(a, a)

with a symmetric Beta(a, a) prior (default `a = 1`). Under the null
hypothesis `M_n` is a non-negative martingale with `M_0 = 1`, so by Ville's
inequality `P(sup_n M_n >= 1/alpha) <= alpha` (Robbins, 1970; Howard et al.,
2021; Johari et al., 2022). The always-valid p-value is
`min(1, 1 / max_k M_k)`. The test suite checks known values (five
discordant inputs favouring the treatment give `M = 32/6`) and, by
simulation with continuous monitoring over 300 inputs, a false-positive
rate at or below alpha.

## Deprecated unpaired tests

`test_type="z_test"`, `"chi_squared"`, `"t_test"` and `"bayesian"` select the
unpaired tests used before 0.5.0. They ignore the pairing and count repeated
runs as samples, and they emit a `DeprecationWarning` when used with
`compare()`. They remain the right tools for the live-traffic
`ABTestRunner`, where variants receive different requests.

## References

- McNemar, Q. (1947). *Psychometrika*, 12(2), 153-157.
- Holm, S. (1979). *Scandinavian Journal of Statistics*, 6(2), 65-70.
- Efron, B. (1987). Better bootstrap confidence intervals. *JASA*, 82(397), 171-185.
- Connor, R. J. (1987). *Biometrics*, 43(1), 207-211.
- Agresti, A., and Min, Y. (2005). *Statistics in Medicine*, 24(5), 729-740.
- Fagerland, M. W., Lydersen, S., and Laake, P. (2013). *BMC Medical Research Methodology*, 13, 91.
- Robbins, H. (1970). Statistical methods related to the law of the iterated logarithm. *Annals of Mathematical Statistics*, 41(5), 1397-1409.
- Howard, S. R., Ramdas, A., McAuliffe, J., and Sekhon, J. (2021). Time-uniform, nonparametric, nonasymptotic confidence sequences. *Annals of Statistics*, 49(2), 1055-1080.
- Johari, R., Koomen, P., Pekelis, L., and Walsh, D. (2022). Always valid inference: continuous monitoring of A/B tests. *Operations Research*, 70(3), 1806-1821.
