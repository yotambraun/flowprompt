"""Compare prompt variants on the same inputs, with paired statistics.

    from flowprompt import compare

    result = compare(
        {"v1": PromptV1, "v2": PromptV2},
        inputs=[{"text": "hello"}, {"text": "world"}],
        expected=["greeting", "noun"],
        model="gpt-4o-mini",
    )
    print(result)                 # text report
    result.save_report("ab.html") # or .md

How it works
------------
1. **Run.** Every variant runs on every input (``runs_per_input`` times).
   A variant is any callable ``fn(input) -> output``; Prompt classes and
   ``(PromptClass, "model")`` pairs are adapters (see
   :mod:`flowprompt.testing.variants`). LLM usage and cost are metered.
2. **Score.** Each output is graded by a scorer (``expected`` +
   ``eval_metric``), a ``success_fn``, or a numeric ``metric_fn``. Errors
   count as failures (score 0) and are reported per variant.
3. **Test.** Because all variants see the same inputs, comparisons are
   *paired* and the *input* is the unit of analysis: repeated runs are
   averaged per input first. Pass/fail with one run per input uses the exact
   McNemar test; anything else uses a paired sign-flip permutation test with
   a bootstrap confidence interval. With more than two variants each
   treatment is compared with the control and p-values are Holm-adjusted.
4. **Decide.** The winner is the variant that is significantly better
   (after adjustment), by the sign of the difference.
"""

from __future__ import annotations

import asyncio
import math
import time
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flowprompt.core.usage import CallUsage, track_usage
from flowprompt.testing.experiment import VariantStats
from flowprompt.testing.paired import (
    SampleSizePlan,
    holm_adjust,
    paired_t_interval,
    paired_test,
    plan_sample_size,
    wilson_interval,
)
from flowprompt.testing.statistics import StatisticalResult, run_significance_test
from flowprompt.testing.variants import PromptVariant, Variant, as_variant

_LEGACY_TESTS = ("z_test", "chi_squared", "t_test", "bayesian")
_PAIRED_TESTS = ("auto", "paired", "mcnemar", "permutation")


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """One run of one variant on one input."""

    input_index: int
    run_index: int
    output: Any = None
    score: float = 0.0
    error: str | None = None
    latency_ms: float = 0.0
    cost_usd: float | None = None
    tokens: int = 0
    llm_calls: int = 0
    cached: bool = False


@dataclass
class VariantResult:
    """Results for a single variant.

    Attributes:
        name: Variant name.
        samples: Number of runs completed (inputs x runs_per_input).
        successes: Runs that passed (pass/fail outcomes) or completed without
            error (numeric outcomes).
        success_rate: ``successes / samples``.
        mean_latency_ms: Average latency per run in milliseconds.
        total_cost_usd: Total metered cost in USD (0.0 when unknown; see
            ``cost_known``).
        outputs: Outputs of the runs that did not raise.
        errors: Error messages of runs that raised (counted as failures).
        label: Human-readable description (prompt class, model, function).
        mean_score: Input-level mean score (accuracy for pass/fail).
        ci_low: Lower confidence bound for ``mean_score``.
        ci_high: Upper confidence bound for ``mean_score``.
        p95_latency_ms: 95th percentile latency per run.
        cost_known: True when every LLM call had a known price.
        cost_per_correct: ``total_cost_usd / number of correct answers``.
        total_tokens: Tokens used across all LLM calls.
        llm_calls: Number of LLM calls (including retries and cache hits).
        scores: Per-input lists of run scores (errors scored 0).
        records: Every individual run.
    """

    name: str
    samples: int = 0
    successes: int = 0
    success_rate: float = 0.0
    mean_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    outputs: list[Any] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    label: str = ""
    mean_score: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p95_latency_ms: float = 0.0
    cost_known: bool = False
    cost_per_correct: float | None = None
    total_tokens: int = 0
    llm_calls: int = 0
    scores: list[list[float]] = field(default_factory=list)
    records: list[RunRecord] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)


@dataclass
class ComparisonResult:
    """Result of comparing variants.

    Attributes:
        winner: Name of the variant that is significantly better than the
            others (after multiple-comparison adjustment), or None.
        variants: Per-variant results keyed by name.
        statistical_result: The comparison that decides the outcome (the
            winner's comparison, or the closest call when there is none).
        confidence_level: Confidence level used.
        total_runs: Total number of runs across all variants.
        estimated_cost: Pre-run cost estimate (also filled in dry-run mode).
        has_expected: True when ``expected`` outputs were provided.
        comparisons: All pairwise comparisons that were tested.
        control: Name of the control (baseline) variant.
        outcome: What was measured: ``"accuracy"`` (expected outputs),
            ``"success"`` (success_fn), ``"score"`` (numeric metric) or
            ``"no_error"`` (nothing to grade against).
        has_ground_truth: False when outputs could not be graded (no
            expected outputs, success_fn or metric_fn).
        n_inputs: Number of inputs.
        runs_per_input: Runs per input and variant.
        correction: Multiple-comparison correction applied (``"holm"``) or
            None for a single comparison.
        comparison_mode: ``"control"`` (each variant vs the control) or
            ``"all"`` (all pairs).
        model: Default model used for Prompt variants.
        notes: Caveats worth showing next to the result.
    """

    winner: str | None
    variants: dict[str, VariantResult]
    statistical_result: StatisticalResult | None
    confidence_level: float
    total_runs: int
    estimated_cost: dict[str, Any] | None = None
    has_expected: bool = False
    comparisons: list[StatisticalResult] = field(default_factory=list)
    control: str | None = None
    outcome: str = "success"
    has_ground_truth: bool = True
    n_inputs: int = 0
    runs_per_input: int = 1
    correction: str | None = None
    comparison_mode: str = "control"
    model: str | None = None
    notes: list[str] = field(default_factory=list)

    # -- derived values ---------------------------------------------------

    @property
    def alpha(self) -> float:
        return 1 - self.confidence_level

    @property
    def is_dry_run(self) -> bool:
        return self.total_runs == 0 and self.estimated_cost is not None

    @property
    def total_cost_usd(self) -> float:
        """Metered cost of all runs (0.0 when no price was known)."""
        return sum(v.total_cost_usd for v in self.variants.values())

    @property
    def cost_known(self) -> bool:
        calls = [v for v in self.variants.values() if v.llm_calls]
        return bool(calls) and all(v.cost_known for v in calls)

    @property
    def is_proportion(self) -> bool:
        """True when scores are pass/fail rates (shown in points)."""
        return self.outcome in ("accuracy", "success", "no_error")

    @property
    def enough_data(self) -> bool:
        """False when no difference could reach significance at this size.

        With ``n`` inputs the smallest attainable two-sided p-value of the
        paired tests is ``2 / 2**n`` (every input favours the same variant);
        with ``m`` comparisons Holm multiplies it by up to ``m``.
        """
        if not self.comparisons or self.n_inputs == 0:
            return False
        m = len(self.comparisons) if self.correction else 1
        min_p = min(1.0, m * 2.0 / 2.0**self.n_inputs)
        return min_p < self.alpha

    @property
    def min_inputs_for_significance(self) -> int:
        m = len(self.comparisons) if self.correction else 1
        n = 1
        while m * 2.0 / 2.0**n >= self.alpha:
            n += 1
        return n

    @property
    def verdict(self) -> str:
        """One plain-English sentence summarising the outcome."""
        from flowprompt.testing.report import verdict_text

        return verdict_text(self)

    def sample_size_plan(
        self,
        min_detectable_difference: float = 0.1,
        *,
        power: float = 0.8,
    ) -> SampleSizePlan | None:
        """Inputs needed to detect a given difference, using this run as a pilot.

        Uses the discordance (pass/fail) or the SD of per-input differences
        (numeric scores) observed in the deciding comparison, and the
        observed cost per run when it is known.
        """
        sr = self.statistical_result
        if sr is None or not self.variants:
            return None
        n_variants = len(self.variants)
        runs = sum(v.samples for v in self.variants.values())
        cost_per_call: float | None = None
        if self.cost_known and runs:
            cost_per_call = self.total_cost_usd / runs
        elif self.estimated_cost and self.estimated_cost.get("estimated_cost_usd"):
            total = self.estimated_cost.get("total_calls") or 0
            if total:
                cost_per_call = self.estimated_cost["estimated_cost_usd"] / total
        kwargs: dict[str, Any] = {
            "power": power,
            "alpha": self.alpha,
            "n_variants": n_variants if self.comparison_mode == "control" else 2,
            "runs_per_input": self.runs_per_input,
            "cost_per_call": cost_per_call,
        }
        if "discordance_rate" in sr.details:
            observed = float(sr.details["discordance_rate"])
            return plan_sample_size(
                min_detectable_difference,
                discordance=max(observed, min_detectable_difference),
                **kwargs,
            )
        sd = sr.details.get("sd_difference")
        if sd is not None and sd > 0:
            return plan_sample_size(
                min_detectable_difference, sd_difference=float(sd), **kwargs
            )
        control = self.variants.get(self.control or "")
        baseline = control.mean_score if control else None
        return plan_sample_size(
            min_detectable_difference, baseline_accuracy=baseline, **kwargs
        )

    # -- rendering ----------------------------------------------------------

    def __str__(self) -> str:
        """Plain-text report."""
        from flowprompt.testing.report import render_text

        return render_text(self)

    def to_markdown(self) -> str:
        """Markdown report (for PR comments, job summaries, notebooks)."""
        from flowprompt.testing.report import render_markdown

        return render_markdown(self)

    def to_html(self) -> str:
        """Self-contained HTML report (no external assets)."""
        from flowprompt.testing.report import render_html

        return render_html(self)

    def save_report(self, path: str | Path) -> Path:
        """Write the report; the format follows the extension (.html/.md/.txt/.json)."""
        import json

        target = Path(path)
        suffix = target.suffix.lower()
        if suffix in (".html", ".htm"):
            content = self.to_html()
        elif suffix in (".md", ".markdown"):
            content = self.to_markdown()
        elif suffix == ".json":
            content = json.dumps(self.to_dict(), indent=2, default=str)
        else:
            content = str(self)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""

        def stat(sr: StatisticalResult) -> dict[str, Any]:
            return {
                "significant": sr.significant,
                "p_value": sr.p_value,
                "adjusted_p": sr.adjusted_p,
                "effect_size": sr.effect_size,
                "difference": sr.difference,
                "ci_low": sr.ci_low,
                "ci_high": sr.ci_high,
                "relative_lift": sr.relative_lift,
                "n_inputs": sr.n_inputs,
                "method": sr.method,
                "test_name": sr.test_name,
                "control": sr.control,
                "treatment": sr.treatment,
            }

        return {
            "winner": self.winner,
            "verdict": self.verdict if self.variants else None,
            "confidence_level": self.confidence_level,
            "total_runs": self.total_runs,
            "n_inputs": self.n_inputs,
            "runs_per_input": self.runs_per_input,
            "outcome": self.outcome,
            "has_ground_truth": self.has_ground_truth,
            "control": self.control,
            "correction": self.correction,
            "estimated_cost": self.estimated_cost,
            "total_cost_usd": self.total_cost_usd if self.cost_known else None,
            "variants": {
                name: {
                    "label": v.label,
                    "samples": v.samples,
                    "successes": v.successes,
                    "success_rate": v.success_rate,
                    "mean_score": v.mean_score,
                    "ci_low": v.ci_low,
                    "ci_high": v.ci_high,
                    "mean_latency_ms": v.mean_latency_ms,
                    "p95_latency_ms": v.p95_latency_ms,
                    "total_cost_usd": v.total_cost_usd,
                    "cost_known": v.cost_known,
                    "cost_per_correct": v.cost_per_correct,
                    "total_tokens": v.total_tokens,
                    "llm_calls": v.llm_calls,
                    "errors": v.errors,
                }
                for name, v in self.variants.items()
            },
            "statistical_result": stat(self.statistical_result)
            if self.statistical_result
            else None,
            "comparisons": [stat(c) for c in self.comparisons],
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Cost estimate (dry run)
# ---------------------------------------------------------------------------


def estimate_compare_cost(
    prompts: dict[str, Any],
    inputs: list[dict[str, Any]],
    model: str = "gpt-4o",
    *,
    runs_per_input: int = 1,
    estimated_output_tokens: int = 100,
) -> dict[str, Any]:
    """Estimate the cost of running compare() without making API calls.

    Prompt variants are rendered for each input and their input tokens are
    counted with litellm; prices come from litellm's model cost map. Custom
    callables cannot be inspected, so they count calls but no tokens.

    Returns:
        Dict with keys: model, total_calls, estimated_input_tokens,
        estimated_output_tokens, estimated_cost_usd, per_variant. Cost fields
        are None when a price is unknown.
    """
    total_calls = len(prompts) * len(inputs) * runs_per_input
    per_variant: dict[str, dict[str, Any]] = {}
    total_input_tokens = 0
    total_cost: float | None = 0.0
    priced_any = False

    def prices(m: str) -> tuple[float | None, float | None]:
        try:
            import litellm

            info = litellm.model_cost.get(m)
            if info:
                return info.get("input_cost_per_token"), info.get(
                    "output_cost_per_token"
                )
        except Exception:
            pass
        return None, None

    for name, obj in prompts.items():
        try:
            variant: Variant | None = as_variant(obj, model=model, temperature=0.0)
        except TypeError:
            variant = None
        variant_calls = len(inputs) * runs_per_input
        variant_input_tokens = 0
        variant_cost: float | None = None
        if isinstance(variant, PromptVariant):
            vmodel = variant.model or model
            for inp in inputs:
                try:
                    messages = variant.prompt(**inp).to_messages()
                    try:
                        import litellm

                        tokens = litellm.token_counter(model=vmodel, messages=messages)
                    except Exception:
                        text = " ".join(
                            m.get("content", "")
                            for m in messages
                            if isinstance(m, dict)
                        )
                        tokens = len(text) // 4
                    variant_input_tokens += tokens * runs_per_input
                except Exception:
                    pass
            in_price, out_price = prices(vmodel)
            if in_price is not None and out_price is not None:
                variant_cost = (
                    variant_input_tokens * in_price
                    + variant_calls * estimated_output_tokens * out_price
                )
                priced_any = True
        total_input_tokens += variant_input_tokens
        if variant_cost is None:
            total_cost = None
        elif total_cost is not None:
            total_cost += variant_cost
        per_variant[name] = {
            "calls": variant_calls,
            "input_tokens": variant_input_tokens,
            "cost_usd": variant_cost,
        }

    if not priced_any:
        total_cost = None
    if total_cost is None:
        custom = [n for n, info in per_variant.items() if info["cost_usd"] is None]
        warnings.warn(
            f"Could not estimate the cost of: {', '.join(custom)} (no price for the "
            f"model, or a custom callable). Cost fields will be None.",
            stacklevel=2,
        )

    return {
        "model": model,
        "total_calls": total_calls,
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_calls * estimated_output_tokens,
        "estimated_cost_usd": total_cost,
        "per_variant": per_variant,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class _Scoring:
    outcome: str
    has_ground_truth: bool
    fn: Callable[[Any, int], float]
    metric_fn: Callable[[Any], float] | None


def _make_scoring(
    expected: list[Any] | None,
    eval_metric: str | Callable[..., Any],
    success_fn: Callable[..., Any] | None,
    metric_fn: Callable[[Any], float] | None,
) -> _Scoring:
    if success_fn is not None:
        fn_success = success_fn
        if expected is not None:
            exp = expected
            return _Scoring(
                "success",
                True,
                lambda out, i: float(fn_success(out, exp[i])),
                metric_fn,
            )
        return _Scoring(
            "success", True, lambda out, _i: float(fn_success(out)), metric_fn
        )
    if expected is not None:
        from flowprompt.testing.scorers import resolve_scorer

        scorer = resolve_scorer(eval_metric)
        exp = expected
        return _Scoring(
            "accuracy", True, lambda out, i: float(scorer(out, exp[i])), metric_fn
        )
    if metric_fn is not None:
        mfn = metric_fn
        return _Scoring("score", True, lambda out, _i: float(mfn(out)), None)
    return _Scoring("no_error", False, lambda _out, _i: 1.0, None)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _record(
    idx: int,
    run: int,
    calls: list[CallUsage],
    started: float,
    output: Any = None,
    score: float = 0.0,
    error: str | None = None,
) -> RunRecord:
    costs = [c.cost_usd for c in calls]
    return RunRecord(
        input_index=idx,
        run_index=run,
        output=output,
        score=score,
        error=error,
        latency_ms=(time.perf_counter() - started) * 1000,
        cost_usd=sum(c for c in costs if c is not None)
        if calls and all(c is not None for c in costs)
        else None,
        tokens=sum(c.total_tokens for c in calls),
        llm_calls=len(calls),
        cached=bool(calls) and all(c.cached for c in calls),
    )


def _score_output(scoring: _Scoring, output: Any, idx: int) -> tuple[float, str | None]:
    try:
        return scoring.fn(output, idx), None
    except Exception as exc:  # a broken scorer must not crash the experiment
        return 0.0, f"scorer error: {exc}"


def _run_variant_sync(
    variant: Variant,
    inputs: list[dict[str, Any]],
    runs_per_input: int,
    scoring: _Scoring,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    for idx, inp in enumerate(inputs):
        for run in range(runs_per_input):
            started = time.perf_counter()
            with track_usage() as calls:
                try:
                    output = variant(dict(inp))
                except Exception as exc:
                    records.append(_record(idx, run, calls, started, error=str(exc)))
                    continue
            score, err = _score_output(scoring, output, idx)
            records.append(_record(idx, run, calls, started, output, score, err))
    return records


async def _run_variant_async(
    variant: Variant,
    inputs: list[dict[str, Any]],
    runs_per_input: int,
    scoring: _Scoring,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    for idx, inp in enumerate(inputs):
        for run in range(runs_per_input):
            started = time.perf_counter()
            with track_usage() as calls:
                try:
                    output = await variant.acall(dict(inp))
                except Exception as exc:
                    records.append(_record(idx, run, calls, started, error=str(exc)))
                    continue
            score, err = _score_output(scoring, output, idx)
            records.append(_record(idx, run, calls, started, output, score, err))
    return records


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _summarise_variant(
    name: str,
    label: str,
    records: list[RunRecord],
    n_inputs: int,
    scoring: _Scoring,
    confidence_level: float,
) -> VariantResult:
    vr = VariantResult(name=name, label=label, records=records)
    per_input: list[list[float]] = [[] for _ in range(n_inputs)]
    binary = True
    for rec in records:
        vr.samples += 1
        per_input[rec.input_index].append(rec.score)
        if rec.error is not None:
            vr.errors.append(rec.error)
        else:
            vr.outputs.append(rec.output)
        if rec.score not in (0.0, 1.0):
            binary = False
        passed = rec.error is None and (
            rec.score >= 1.0 if scoring.outcome != "score" else True
        )
        if passed:
            vr.successes += 1
        vr.total_tokens += rec.tokens
        vr.llm_calls += rec.llm_calls
    vr.scores = per_input

    if vr.samples:
        vr.success_rate = vr.successes / vr.samples
        latencies = [r.latency_ms for r in records]
        vr.mean_latency_ms = sum(latencies) / len(latencies)
        vr.p95_latency_ms = _percentile(latencies, 0.95)

    means = [sum(s) / len(s) for s in per_input if s]
    if means:
        vr.mean_score = sum(means) / len(means)
        one_run = all(len(s) == 1 for s in per_input if s)
        if binary and one_run:
            k = int(round(sum(means)))
            vr.ci_low, vr.ci_high = wilson_interval(k, len(means), confidence_level)
        else:
            lo, hi = paired_t_interval(means, confidence_level)
            if binary:
                lo, hi = max(0.0, lo), min(1.0, hi)
            vr.ci_low, vr.ci_high = lo, hi

    metered = [r for r in records if r.llm_calls]
    vr.cost_known = bool(metered) and all(r.cost_usd is not None for r in metered)
    vr.total_cost_usd = sum(r.cost_usd or 0.0 for r in records)
    if vr.cost_known and scoring.outcome in ("accuracy", "success"):
        correct = sum(r.score for r in records if r.error is None)
        vr.cost_per_correct = vr.total_cost_usd / correct if correct > 0 else None
    return vr


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _legacy_test(
    control: VariantResult,
    treatment: VariantResult,
    test_type: str,
    confidence_level: float,
) -> StatisticalResult:
    def stats(v: VariantResult) -> VariantStats:
        from flowprompt.testing.experiment import ExperimentResult

        vs = VariantStats(name=v.name)
        for rec in v.records:
            vs.update(
                ExperimentResult(
                    experiment_id="_compare",
                    variant_name=v.name,
                    success=rec.error is None and rec.score >= 1.0,
                    metric_value=rec.score,
                    latency_ms=rec.latency_ms,
                )
            )
        return vs

    cs, ts = stats(control), stats(treatment)
    sr = run_significance_test(
        cs, ts, test_type=test_type, confidence_level=confidence_level
    )
    if test_type == "t_test":
        sr.difference = ts.mean_metric - cs.mean_metric
    else:
        sr.difference = ts.success_rate - cs.success_rate
    sr.method = sr.test_name
    sr.n_inputs = len(control.scores)
    c_rate = cs.mean_metric if test_type == "t_test" else cs.success_rate
    sr.relative_lift = sr.difference / c_rate if c_rate else None
    return sr


def _analyse(
    variant_results: dict[str, VariantResult],
    *,
    control: str,
    comparison_mode: str,
    test_type: str,
    confidence_level: float,
) -> tuple[list[StatisticalResult], str | None, StatisticalResult | None, str | None]:
    names = list(variant_results)
    if comparison_mode == "all":
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    else:
        pairs = [(control, n) for n in names if n != control]

    comparisons: list[StatisticalResult] = []
    for c_name, t_name in pairs:
        c_res, t_res = variant_results[c_name], variant_results[t_name]
        if test_type in _LEGACY_TESTS:
            sr = _legacy_test(c_res, t_res, test_type, confidence_level)
        else:
            sr = paired_test(
                c_res.scores,
                t_res.scores,
                confidence_level=confidence_level,
                control_name=c_name,
                treatment_name=t_name,
            )
        sr.control, sr.treatment = c_name, t_name
        comparisons.append(sr)

    alpha = 1 - confidence_level
    correction: str | None = None
    if len(comparisons) > 1:
        correction = "holm"
        adjusted = holm_adjust([c.p_value for c in comparisons])
        for comp, adj in zip(comparisons, adjusted, strict=True):
            comp.adjusted_p = adj
            comp.significant = adj < alpha
    else:
        for comp in comparisons:
            comp.significant = comp.p_value < alpha

    winner: str | None = None
    deciding: StatisticalResult | None = None

    def beats(a: str, b: str) -> StatisticalResult | None:
        for comp in comparisons:
            diff = comp.difference or 0.0
            if (
                comp.treatment == a
                and comp.control == b
                and comp.significant
                and diff > 0
            ):
                return comp
            if (
                comp.control == a
                and comp.treatment == b
                and comp.significant
                and diff < 0
            ):
                return comp
        return None

    if comparison_mode == "all":
        for cand in names:
            wins = [beats(cand, other) for other in names if other != cand]
            if wins and all(w is not None for w in wins):
                winner = cand
                deciding = min(wins, key=lambda c: abs(c.difference or 0.0))  # type: ignore[arg-type,union-attr]
                break
    else:
        better = [c for c in comparisons if c.significant and (c.difference or 0.0) > 0]
        worse = [c for c in comparisons if c.significant and (c.difference or 0.0) < 0]
        if better:
            deciding = max(better, key=lambda c: c.difference or 0.0)
            winner = deciding.treatment
        elif comparisons and len(worse) == len(comparisons):
            deciding = min(worse, key=lambda c: abs(c.difference or 0.0))
            winner = control

    if deciding is None and comparisons:
        deciding = min(
            comparisons,
            key=lambda c: c.adjusted_p if c.adjusted_p is not None else c.p_value,
        )
    return comparisons, winner, deciding, correction


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _validate(
    prompts: dict[str, Any],
    inputs: list[dict[str, Any]],
    expected: list[Any] | None,
    runs_per_input: int,
    test_type: str,
    control: str | None,
    comparisons: str,
    fn_name: str,
) -> None:
    if len(prompts) < 2:
        raise ValueError(f"{fn_name}() requires at least 2 prompt variants")
    if len(inputs) < 1:
        raise ValueError(f"{fn_name}() requires at least 1 input")
    if expected is not None and len(expected) != len(inputs):
        raise ValueError(
            f"len(expected)={len(expected)} must equal len(inputs)={len(inputs)}"
        )
    if runs_per_input < 1:
        raise ValueError("runs_per_input must be at least 1")
    if test_type not in _PAIRED_TESTS + _LEGACY_TESTS:
        raise ValueError(
            f"Unknown test type: {test_type}. "
            f"Available: {list(_PAIRED_TESTS + _LEGACY_TESTS)}"
        )
    if control is not None and control not in prompts:
        raise ValueError(f"control {control!r} is not one of the variants")
    if comparisons not in ("control", "all"):
        raise ValueError("comparisons must be 'control' or 'all'")
    if test_type in _LEGACY_TESTS:
        warnings.warn(
            f"test_type={test_type!r} uses an unpaired test that ignores that all "
            "variants ran on the same inputs and counts repeated runs as "
            "independent samples. Use the default test_type='auto' (paired). "
            "Legacy test types will be removed in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )


def _dry_run_result(
    cost_estimate: dict[str, Any], confidence_level: float, has_expected: bool
) -> ComparisonResult:
    cost_str = (
        f"${cost_estimate['estimated_cost_usd']:.2f}"
        if cost_estimate.get("estimated_cost_usd") is not None
        else "unknown"
    )
    print(
        f"Estimated cost: {cost_str} for {cost_estimate['total_calls']} API calls "
        f"({cost_estimate['estimated_input_tokens']} input tokens)"
    )
    return ComparisonResult(
        winner=None,
        variants={},
        statistical_result=None,
        confidence_level=confidence_level,
        total_runs=0,
        estimated_cost=cost_estimate,
        has_expected=has_expected,
    )


def _finish(
    raw: dict[str, list[RunRecord]],
    adapted: dict[str, Variant],
    inputs: list[dict[str, Any]],
    scoring: _Scoring,
    *,
    confidence_level: float,
    control: str,
    comparison_mode: str,
    test_type: str,
    cost_estimate: dict[str, Any] | None,
    has_expected: bool,
    runs_per_input: int,
    model: str,
) -> ComparisonResult:
    variant_results = {
        name: _summarise_variant(
            name,
            getattr(adapted[name], "label", name),
            raw[name],
            len(inputs),
            scoring,
            confidence_level,
        )
        for name in adapted
    }
    comparisons, winner, deciding, correction = _analyse(
        variant_results,
        control=control,
        comparison_mode=comparison_mode,
        test_type=test_type,
        confidence_level=confidence_level,
    )
    notes: list[str] = []
    if not scoring.has_ground_truth:
        notes.append(
            "No expected outputs or scorer were given, so outputs were not graded: "
            "the comparison only measures how often each variant completed "
            "without an error. Pass expected=... (or success_fn/metric_fn) to "
            "measure quality."
        )
    for name, vr in variant_results.items():
        if vr.errors:
            notes.append(
                f"{name}: {len(vr.errors)} of {vr.samples} runs raised an error and "
                "were scored as failures."
            )
    if runs_per_input > 1:
        notes.append(
            f"Each input ran {runs_per_input} times per variant; runs were averaged "
            "per input before testing, so the test is based on "
            f"{len(inputs)} inputs, not {len(inputs) * runs_per_input} runs."
        )
    result = ComparisonResult(
        winner=winner,
        variants=variant_results,
        statistical_result=deciding,
        confidence_level=confidence_level,
        total_runs=sum(v.samples for v in variant_results.values()),
        estimated_cost=cost_estimate,
        has_expected=has_expected,
        comparisons=comparisons,
        control=control,
        outcome=scoring.outcome,
        has_ground_truth=scoring.has_ground_truth,
        n_inputs=len(inputs),
        runs_per_input=runs_per_input,
        correction=correction,
        comparison_mode=comparison_mode,
        model=model,
        notes=notes,
    )
    better = [
        c.treatment
        for c in comparisons
        if c.significant and (c.difference or 0.0) > 0 and c.treatment != winner
    ]
    if comparison_mode == "control" and winner and better:
        result.notes.append(
            f"{', '.join(str(b) for b in better)} also beat {control} significantly; "
            f"{winner} has the largest difference, but it was not tested against "
            f"{'it' if len(better) == 1 else 'them'} directly. Use "
            "comparisons='all' to test every pair."
        )
    if not result.enough_data:
        result.notes.append(
            f"Not enough data: with {len(inputs)} inputs no difference can reach "
            f"significance at alpha={result.alpha:g}; at least "
            f"{result.min_inputs_for_significance} inputs are needed."
        )
    return result


def compare(
    prompts: dict[str, Any],
    inputs: list[dict[str, Any]],
    model: str = "gpt-4o",
    *,
    expected: list[Any] | None = None,
    eval_metric: str | Callable[..., Any] = "contains",
    success_fn: Callable[..., Any] | None = None,
    metric_fn: Callable[[Any], float] | None = None,
    confidence_level: float = 0.95,
    runs_per_input: int = 1,
    temperature: float = 0.0,
    test_type: str = "auto",
    dry_run: bool = False,
    control: str | None = None,
    comparisons: str = "control",
) -> ComparisonResult:
    """Compare variants on the same inputs with paired significance tests.

    Args:
        prompts: Dict mapping variant names to variants: Prompt subclasses,
            ``(PromptClass, "model")`` tuples, :class:`PromptVariant` objects
            or any callable ``fn(input: dict) -> output`` (sync or async).
        inputs: Input dicts; every variant runs on every input.
        model: Default model for Prompt variants.
        expected: Expected outputs (same length as ``inputs``). Outputs are
            graded with ``eval_metric``.
        eval_metric: Scorer for ``(output, expected)``: ``"exact"``,
            ``"contains"`` (default), ``"regex"``, ``"numeric"``,
            ``"similarity"``, or a callable returning bool (pass/fail) or a
            float score. See :mod:`flowprompt.testing.scorers`.
        success_fn: Custom pass/fail function ``(output)`` or, with
            ``expected``, ``(output, expected)``. Takes precedence.
        metric_fn: Numeric score ``(output) -> float``. Used as the outcome
            when neither ``expected`` nor ``success_fn`` is given.
        confidence_level: Confidence level (default 0.95, i.e. alpha 0.05).
        runs_per_input: Runs per input and variant. Repeats are averaged per
            input; they measure run-to-run variation but do not increase the
            sample size, which is the number of inputs.
        temperature: Default temperature for Prompt variants.
        test_type: ``"auto"`` (default): exact McNemar for pass/fail with one
            run per input, otherwise a paired sign-flip permutation test.
            ``"z_test"``, ``"chi_squared"``, ``"t_test"`` and ``"bayesian"``
            select the deprecated unpaired tests.
        dry_run: Only estimate the cost; make no calls.
        control: Name of the baseline variant (default: the first one).
        comparisons: ``"control"`` (default) compares each variant with the
            control; ``"all"`` compares every pair. Holm's correction is
            applied whenever there is more than one comparison.

    Returns:
        A :class:`ComparisonResult`. ``print(result)`` shows a report;
        ``result.save_report("report.html")`` writes Markdown or HTML.

    Raises:
        ValueError: On fewer than 2 variants, no inputs, mismatched
            ``expected`` or an unknown option.

    Example:
        >>> result = compare(
        ...     {"short": Short, "detailed": Detailed},
        ...     inputs=[{"text": t} for t in texts],
        ...     expected=labels,
        ...     eval_metric="exact",
        ...     model="gpt-4o-mini",
        ... )
        >>> print(result.verdict)
    """
    _validate(
        prompts,
        inputs,
        expected,
        runs_per_input,
        test_type,
        control,
        comparisons,
        "compare",
    )
    has_expected = expected is not None
    adapted = {
        n: as_variant(v, model=model, temperature=temperature)
        for n, v in prompts.items()
    }
    with warnings.catch_warnings():
        if not dry_run:  # the estimate is only shown in dry-run mode
            warnings.simplefilter("ignore")
        cost_estimate = estimate_compare_cost(
            prompts, inputs, model, runs_per_input=runs_per_input
        )
    if dry_run:
        return _dry_run_result(cost_estimate, confidence_level, has_expected)

    scoring = _make_scoring(expected, eval_metric, success_fn, metric_fn)
    raw: dict[str, list[RunRecord]] = {}
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = {
            name: executor.submit(
                _run_variant_sync, variant, inputs, runs_per_input, scoring
            )
            for name, variant in adapted.items()
        }
        for name, future in futures.items():
            raw[name] = future.result()

    return _finish(
        raw,
        adapted,
        inputs,
        scoring,
        confidence_level=confidence_level,
        control=control or next(iter(prompts)),
        comparison_mode=comparisons,
        test_type=test_type,
        cost_estimate=cost_estimate,
        has_expected=has_expected,
        runs_per_input=runs_per_input,
        model=model,
    )


async def acompare(
    prompts: dict[str, Any],
    inputs: list[dict[str, Any]],
    model: str = "gpt-4o",
    *,
    expected: list[Any] | None = None,
    eval_metric: str | Callable[..., Any] = "contains",
    success_fn: Callable[..., Any] | None = None,
    metric_fn: Callable[[Any], float] | None = None,
    confidence_level: float = 0.95,
    runs_per_input: int = 1,
    temperature: float = 0.0,
    test_type: str = "auto",
    dry_run: bool = False,
    control: str | None = None,
    comparisons: str = "control",
) -> ComparisonResult:
    """Async version of :func:`compare`; variants run concurrently.

    Same arguments and result. Prompt variants use ``Prompt.arun()``;
    async callables are awaited; sync callables run in a worker thread.
    """
    _validate(
        prompts,
        inputs,
        expected,
        runs_per_input,
        test_type,
        control,
        comparisons,
        "acompare",
    )
    has_expected = expected is not None
    adapted = {
        n: as_variant(v, model=model, temperature=temperature)
        for n, v in prompts.items()
    }
    with warnings.catch_warnings():
        if not dry_run:  # the estimate is only shown in dry-run mode
            warnings.simplefilter("ignore")
        cost_estimate = estimate_compare_cost(
            prompts, inputs, model, runs_per_input=runs_per_input
        )
    if dry_run:
        return _dry_run_result(cost_estimate, confidence_level, has_expected)

    scoring = _make_scoring(expected, eval_metric, success_fn, metric_fn)
    results = await asyncio.gather(
        *(
            _run_variant_async(variant, inputs, runs_per_input, scoring)
            for name, variant in adapted.items()
        )
    )
    raw = dict(zip(adapted, results, strict=True))
    return _finish(
        raw,
        adapted,
        inputs,
        scoring,
        confidence_level=confidence_level,
        control=control or next(iter(prompts)),
        comparison_mode=comparisons,
        test_type=test_type,
        cost_estimate=cost_estimate,
        has_expected=has_expected,
        runs_per_input=runs_per_input,
        model=model,
    )
