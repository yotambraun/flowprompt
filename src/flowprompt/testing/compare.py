"""One-call prompt comparison for A/B testing.

Provides a simple interface for comparing prompt variants:

    from flowprompt import compare

    result = compare(
        {"v1": PromptV1, "v2": PromptV2},
        inputs=[{"text": "hello"}, {"text": "world"}],
        model="gpt-4o-mini",
    )
    print(result)
"""

from __future__ import annotations

import asyncio
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from flowprompt.testing.experiment import VariantStats
from flowprompt.testing.statistics import StatisticalResult, run_significance_test


@dataclass
class VariantResult:
    """Results for a single prompt variant.

    Attributes:
        name: Variant name.
        samples: Number of runs completed.
        successes: Number of successful runs.
        success_rate: Fraction of successful runs (0.0-1.0).
        mean_latency_ms: Average response latency in milliseconds.
        total_cost_usd: Total estimated cost in USD.
        outputs: List of outputs from each run.
        errors: List of errors encountered.
    """

    name: str
    samples: int = 0
    successes: int = 0
    success_rate: float = 0.0
    mean_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    outputs: list[Any] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Result of comparing prompt variants.

    Attributes:
        winner: Name of the winning variant, or None if no significant winner.
        variants: Per-variant results keyed by name.
        statistical_result: Result of the significance test.
        confidence_level: Confidence level used for the test.
        total_runs: Total number of runs across all variants.
        estimated_cost: Cost estimation dict, populated in dry_run or normal mode.
        has_expected: True when ``expected`` outputs were provided to compare().
    """

    winner: str | None
    variants: dict[str, VariantResult]
    statistical_result: StatisticalResult | None
    confidence_level: float
    total_runs: int
    estimated_cost: dict[str, Any] | None = None
    has_expected: bool = False

    def __str__(self) -> str:
        """Pretty-print the comparison result."""
        # Dry run mode
        if self.total_runs == 0 and self.estimated_cost is not None:
            lines = ["Comparison Results (DRY RUN)", "=" * 40]
            est = self.estimated_cost
            cost_str = (
                f"${est['estimated_cost_usd']:.2f}"
                if est.get("estimated_cost_usd") is not None
                else "unknown"
            )
            lines.append(f"  Estimated cost: {cost_str} for {est['total_calls']} calls")
            per_variant = est.get("per_variant", {})
            if per_variant:
                lines.append("  Per variant:")
                for vname, vinfo in per_variant.items():
                    vcost = (
                        f"~${vinfo['cost_usd']:.2f}"
                        if vinfo.get("cost_usd") is not None
                        else "unknown"
                    )
                    lines.append(
                        f"    {vname}: {vinfo['calls']} calls, "
                        f"~{vinfo['input_tokens']} tokens, {vcost}"
                    )
            return "\n".join(lines)

        # Normal mode
        lines = ["Comparison Results", "=" * 40]
        rate_label = "accuracy" if self.has_expected else "success"

        for name, v in self.variants.items():
            marker = " << WINNER" if name == self.winner else ""
            lines.append(
                f"  {name}: {v.success_rate:.0%} {rate_label}, "
                f"{v.mean_latency_ms:.0f}ms avg, "
                f"{v.samples} runs{marker}"
            )
            if v.errors:
                lines.append(f"    errors: {len(v.errors)}")

        if self.statistical_result:
            sr = self.statistical_result
            lines.append("")
            sig = "SIGNIFICANT" if sr.significant else "not significant"
            lines.append(f"  p={sr.p_value:.4f} ({sig})")
            lines.append(f"  effect size: {sr.effect_size:+.2%}")

        if not self.winner:
            lines.append("")
            lines.append("  No clear winner (results not statistically significant)")

        if (
            self.estimated_cost
            and self.estimated_cost.get("estimated_cost_usd") is not None
        ):
            cost = self.estimated_cost["estimated_cost_usd"]
            num_variants = len(self.variants)
            avg = cost / num_variants if num_variants > 0 else 0
            lines.append("")
            lines.append(f"  Cost: ${cost:.2f} total (${avg:.2f}/variant avg)")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "winner": self.winner,
            "confidence_level": self.confidence_level,
            "total_runs": self.total_runs,
            "estimated_cost": self.estimated_cost,
            "variants": {
                name: {
                    "samples": v.samples,
                    "successes": v.successes,
                    "success_rate": v.success_rate,
                    "mean_latency_ms": v.mean_latency_ms,
                    "total_cost_usd": v.total_cost_usd,
                    "errors": v.errors,
                }
                for name, v in self.variants.items()
            },
            "statistical_result": {
                "significant": self.statistical_result.significant,
                "p_value": self.statistical_result.p_value,
                "effect_size": self.statistical_result.effect_size,
                "test_name": self.statistical_result.test_name,
            }
            if self.statistical_result
            else None,
        }


def estimate_compare_cost(
    prompts: dict[str, type],
    inputs: list[dict[str, Any]],
    model: str = "gpt-4o",
    *,
    runs_per_input: int = 1,
    estimated_output_tokens: int = 100,
) -> dict[str, Any]:
    """Estimate the cost of running compare() without making API calls.

    Instantiates each prompt variant with each input, counts input tokens
    using litellm, and looks up pricing to produce a cost estimate.

    Args:
        prompts: Dict mapping variant names to Prompt subclasses.
        inputs: List of input dicts to test each variant against.
        model: Model to use for token counting and pricing lookup.
        runs_per_input: Number of times to run each input per variant.
        estimated_output_tokens: Estimated output tokens per call.

    Returns:
        Dict with keys: model, total_calls, estimated_input_tokens,
        estimated_output_tokens, estimated_cost_usd, per_variant.
        Cost fields may be None if litellm pricing is unavailable.
    """
    total_calls = len(prompts) * len(inputs) * runs_per_input
    per_variant: dict[str, dict[str, Any]] = {}
    total_input_tokens = 0

    # Try to get pricing info
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    try:
        import litellm

        cost_info = litellm.model_cost.get(model)
        if cost_info:
            input_cost_per_token = cost_info.get("input_cost_per_token")
            output_cost_per_token = cost_info.get("output_cost_per_token")
    except Exception:
        pass

    for name, prompt_class in prompts.items():
        variant_input_tokens = 0
        variant_calls = len(inputs) * runs_per_input

        for inp in inputs:
            try:
                prompt = prompt_class(**inp)
                messages = prompt.to_messages()
                try:
                    import litellm

                    tokens = litellm.token_counter(model=model, messages=messages)
                except Exception:
                    # Rough fallback: ~4 chars per token
                    text = " ".join(
                        m.get("content", "") for m in messages if isinstance(m, dict)
                    )
                    tokens = len(text) // 4
                variant_input_tokens += tokens * runs_per_input
            except Exception:
                pass

        total_input_tokens += variant_input_tokens

        variant_cost: float | None = None
        if input_cost_per_token is not None and output_cost_per_token is not None:
            variant_cost = (
                variant_input_tokens * input_cost_per_token
                + variant_calls * estimated_output_tokens * output_cost_per_token
            )

        per_variant[name] = {
            "calls": variant_calls,
            "input_tokens": variant_input_tokens,
            "cost_usd": variant_cost,
        }

    total_estimated_output_tokens = total_calls * estimated_output_tokens
    total_cost: float | None = None
    if input_cost_per_token is not None and output_cost_per_token is not None:
        total_cost = (
            total_input_tokens * input_cost_per_token
            + total_estimated_output_tokens * output_cost_per_token
        )
    else:
        warnings.warn(
            f"Could not look up pricing for model '{model}'. Cost fields will be None.",
            stacklevel=2,
        )

    return {
        "model": model,
        "total_calls": total_calls,
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_estimated_output_tokens,
        "estimated_cost_usd": total_cost,
        "per_variant": per_variant,
    }


def _run_variant(
    name: str,
    prompt_class: type,
    inputs: list[dict[str, Any]],
    model: str,
    temperature: float,
    runs_per_input: int,
    success_fn: Callable[..., bool] | None,
    metric_fn: Callable[[Any], float] | None,
    expected_outputs: list[Any] | None = None,
) -> tuple[VariantResult, VariantStats]:
    """Run a single variant against all inputs and collect stats."""
    from flowprompt.testing.experiment import ExperimentResult

    vr = VariantResult(name=name)
    vs = VariantStats(name=name)

    for idx, inp in enumerate(inputs):
        for _ in range(runs_per_input):
            start = time.time()
            try:
                prompt = prompt_class(**inp)
                output = prompt.run(model=model, temperature=temperature)
                latency_ms = (time.time() - start) * 1000

                if expected_outputs is not None and success_fn is not None:
                    success = success_fn(output, expected_outputs[idx])
                elif success_fn is not None:
                    success = success_fn(output)
                else:
                    success = True
                metric_value = metric_fn(output) if metric_fn else float(success)

                vr.outputs.append(output)
                vr.samples += 1
                if success:
                    vr.successes += 1

                er = ExperimentResult(
                    experiment_id="_compare",
                    variant_name=name,
                    input_data=inp,
                    output=output,
                    success=success,
                    metric_value=metric_value,
                    latency_ms=latency_ms,
                )
                vs.update(er)

            except Exception as exc:
                latency_ms = (time.time() - start) * 1000
                vr.errors.append(str(exc))
                vr.samples += 1

                er = ExperimentResult(
                    experiment_id="_compare",
                    variant_name=name,
                    input_data=inp,
                    output=None,
                    success=False,
                    metric_value=0.0,
                    latency_ms=latency_ms,
                )
                vs.update(er)

    if vr.samples > 0:
        vr.success_rate = vr.successes / vr.samples
        vr.mean_latency_ms = vs.mean_latency_ms
        vr.total_cost_usd = vs.total_cost_usd

    return vr, vs


async def _arun_variant(
    name: str,
    prompt_class: type,
    inputs: list[dict[str, Any]],
    model: str,
    temperature: float,
    runs_per_input: int,
    success_fn: Callable[..., bool] | None,
    metric_fn: Callable[[Any], float] | None,
    expected_outputs: list[Any] | None = None,
) -> tuple[VariantResult, VariantStats]:
    """Run a single variant asynchronously against all inputs."""
    from flowprompt.testing.experiment import ExperimentResult

    vr = VariantResult(name=name)
    vs = VariantStats(name=name)

    for idx, inp in enumerate(inputs):
        for _ in range(runs_per_input):
            start = time.time()
            try:
                prompt = prompt_class(**inp)
                output = await prompt.arun(model=model, temperature=temperature)
                latency_ms = (time.time() - start) * 1000

                if expected_outputs is not None and success_fn is not None:
                    success = success_fn(output, expected_outputs[idx])
                elif success_fn is not None:
                    success = success_fn(output)
                else:
                    success = True
                metric_value = metric_fn(output) if metric_fn else float(success)

                vr.outputs.append(output)
                vr.samples += 1
                if success:
                    vr.successes += 1

                er = ExperimentResult(
                    experiment_id="_compare",
                    variant_name=name,
                    input_data=inp,
                    output=output,
                    success=success,
                    metric_value=metric_value,
                    latency_ms=latency_ms,
                )
                vs.update(er)

            except Exception as exc:
                latency_ms = (time.time() - start) * 1000
                vr.errors.append(str(exc))
                vr.samples += 1

                er = ExperimentResult(
                    experiment_id="_compare",
                    variant_name=name,
                    input_data=inp,
                    output=None,
                    success=False,
                    metric_value=0.0,
                    latency_ms=latency_ms,
                )
                vs.update(er)

    if vr.samples > 0:
        vr.success_rate = vr.successes / vr.samples
        vr.mean_latency_ms = vs.mean_latency_ms
        vr.total_cost_usd = vs.total_cost_usd

    return vr, vs


def _build_result(
    variant_results: dict[str, VariantResult],
    variant_stats: dict[str, VariantStats],
    confidence_level: float,
    test_type: str,
    estimated_cost: dict[str, Any] | None = None,
    *,
    has_expected: bool = False,
) -> ComparisonResult:
    """Build a ComparisonResult from collected variant data."""
    total_runs = sum(vr.samples for vr in variant_results.values())

    # Run significance test between the first two variants
    names = list(variant_stats.keys())
    statistical_result = None
    winner = None

    if len(names) >= 2:
        # Use first variant as control, find best treatment
        control_name = names[0]
        control_stats = variant_stats[control_name]
        best_name = None
        best_effect = 0.0
        best_result = None

        for name in names[1:]:
            treatment_stats = variant_stats[name]
            if control_stats.samples > 0 and treatment_stats.samples > 0:
                result = run_significance_test(
                    control_stats,
                    treatment_stats,
                    test_type=test_type,
                    confidence_level=confidence_level,
                )
                if best_result is None or (
                    result.significant and result.effect_size > best_effect
                ):
                    best_result = result
                    best_effect = result.effect_size
                    best_name = name

        statistical_result = best_result

        if statistical_result and statistical_result.significant:
            winner = best_name if statistical_result.effect_size > 0 else control_name

    return ComparisonResult(
        winner=winner,
        variants=variant_results,
        statistical_result=statistical_result,
        confidence_level=confidence_level,
        total_runs=total_runs,
        estimated_cost=estimated_cost,
        has_expected=has_expected,
    )


def compare(
    prompts: dict[str, type],
    inputs: list[dict[str, Any]],
    model: str = "gpt-4o",
    *,
    expected: list[Any] | None = None,
    eval_metric: str | Callable[..., bool] = "contains",
    success_fn: Callable[..., bool] | None = None,
    metric_fn: Callable[[Any], float] | None = None,
    confidence_level: float = 0.95,
    runs_per_input: int = 1,
    temperature: float = 0.0,
    test_type: str = "z_test",
    dry_run: bool = False,
) -> ComparisonResult:
    """Compare prompt variants with statistical significance testing.

    This is the simplest way to A/B test prompts. Pass a dict of named
    prompt classes and a list of test inputs, and get back a result
    telling you which prompt performs better.

    Args:
        prompts: Dict mapping variant names to Prompt subclasses.
        inputs: List of input dicts to test each variant against.
        model: Model to use for all variants.
        expected: Optional list of expected outputs (same length as inputs).
            When provided, auto-generates a success_fn from eval_metric.
        eval_metric: Metric for evaluating outputs against expected values.
            Either a string name ("exact", "contains", "similarity") or
            a callable ``(output, expected) -> bool``. Default "contains".
            Only used when ``expected`` is provided and ``success_fn`` is None.
        success_fn: Optional function to determine if an output is successful.
            Defaults to treating all non-error outputs as successful.
            When both ``expected`` and ``success_fn`` are provided, the
            success_fn receives ``(output, expected_value)`` as arguments.
        metric_fn: Optional function to compute a numeric metric from output.
        confidence_level: Confidence level for significance testing (default 0.95).
        runs_per_input: Number of times to run each input per variant (default 1).
        temperature: Temperature for LLM calls (default 0.0).
        test_type: Statistical test type ("z_test", "chi_squared", "t_test", "bayesian").
        dry_run: If True, estimate cost without making API calls.

    Returns:
        ComparisonResult with winner, per-variant stats, and significance test.
        In dry_run mode, returns result with winner=None and cost estimate.

    Raises:
        ValueError: If fewer than 2 prompts or 0 inputs are provided,
            or if ``expected`` length doesn't match ``inputs``.

    Example:
        >>> from flowprompt import Prompt, compare
        >>>
        >>> class Short(Prompt):
        ...     system = "Be brief."
        ...     user = "Summarize: {text}"
        >>>
        >>> class Detailed(Prompt):
        ...     system = "Be thorough."
        ...     user = "Summarize in detail: {text}"
        >>>
        >>> result = compare(
        ...     {"short": Short, "detailed": Detailed},
        ...     inputs=[{"text": "Python is great"}],
        ...     expected=["Python is a programming language"],
        ...     model="gpt-4o-mini",
        ... )
        >>> print(result)
    """
    if len(prompts) < 2:
        raise ValueError("compare() requires at least 2 prompt variants")
    if len(inputs) < 1:
        raise ValueError("compare() requires at least 1 input")
    if expected is not None and len(expected) != len(inputs):
        raise ValueError(
            f"len(expected)={len(expected)} must equal len(inputs)={len(inputs)}"
        )

    has_expected = expected is not None

    # When expected is provided and no explicit success_fn, build one from eval_metric
    if expected is not None and success_fn is None:
        from flowprompt.testing.eval_metrics import resolve_eval_metric

        resolved = resolve_eval_metric(eval_metric)

        def _eval_success(output: Any, exp: Any) -> bool:
            return resolved(str(output), str(exp))

        success_fn = _eval_success

    cost_estimate = estimate_compare_cost(
        prompts, inputs, model, runs_per_input=runs_per_input
    )

    if dry_run:
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

    variant_results: dict[str, VariantResult] = {}
    variant_stats: dict[str, VariantStats] = {}

    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = {
            name: executor.submit(
                _run_variant,
                name,
                prompt_class,
                inputs,
                model,
                temperature,
                runs_per_input,
                success_fn,
                metric_fn,
                expected if has_expected else None,
            )
            for name, prompt_class in prompts.items()
        }
        for name, future in futures.items():
            vr, vs = future.result()
            variant_results[name] = vr
            variant_stats[name] = vs

    return _build_result(
        variant_results,
        variant_stats,
        confidence_level,
        test_type,
        cost_estimate,
        has_expected=has_expected,
    )


async def acompare(
    prompts: dict[str, type],
    inputs: list[dict[str, Any]],
    model: str = "gpt-4o",
    *,
    expected: list[Any] | None = None,
    eval_metric: str | Callable[..., bool] = "contains",
    success_fn: Callable[..., bool] | None = None,
    metric_fn: Callable[[Any], float] | None = None,
    confidence_level: float = 0.95,
    runs_per_input: int = 1,
    temperature: float = 0.0,
    test_type: str = "z_test",
    dry_run: bool = False,
) -> ComparisonResult:
    """Async version of compare() that runs variants in parallel.

    Same arguments and return type as compare(). Uses asyncio.gather
    to run all variants concurrently.

    Args:
        expected: Optional list of expected outputs (same length as inputs).
        eval_metric: Metric for evaluating outputs against expected values.
        dry_run: If True, estimate cost without making API calls.

    Example:
        >>> result = await acompare(
        ...     {"v1": PromptV1, "v2": PromptV2},
        ...     inputs=[{"text": "hello"}],
        ...     expected=["greeting"],
        ...     model="gpt-4o-mini",
        ... )
    """
    if len(prompts) < 2:
        raise ValueError("acompare() requires at least 2 prompt variants")
    if len(inputs) < 1:
        raise ValueError("acompare() requires at least 1 input")
    if expected is not None and len(expected) != len(inputs):
        raise ValueError(
            f"len(expected)={len(expected)} must equal len(inputs)={len(inputs)}"
        )

    has_expected = expected is not None

    if expected is not None and success_fn is None:
        from flowprompt.testing.eval_metrics import resolve_eval_metric

        resolved = resolve_eval_metric(eval_metric)

        def _eval_success(output: Any, exp: Any) -> bool:
            return resolved(str(output), str(exp))

        success_fn = _eval_success

    cost_estimate = estimate_compare_cost(
        prompts, inputs, model, runs_per_input=runs_per_input
    )

    if dry_run:
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

    tasks = [
        _arun_variant(
            name,
            prompt_class,
            inputs,
            model,
            temperature,
            runs_per_input,
            success_fn,
            metric_fn,
            expected if has_expected else None,
        )
        for name, prompt_class in prompts.items()
    ]

    results = await asyncio.gather(*tasks)

    variant_results: dict[str, VariantResult] = {}
    variant_stats: dict[str, VariantStats] = {}
    for vr, vs in results:
        variant_results[vr.name] = vr
        variant_stats[vs.name] = vs

    return _build_result(
        variant_results,
        variant_stats,
        confidence_level,
        test_type,
        cost_estimate,
        has_expected=has_expected,
    )
