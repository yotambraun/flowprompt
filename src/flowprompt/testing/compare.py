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
from dataclasses import dataclass, field
from typing import Any, Callable

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
    """

    winner: str | None
    variants: dict[str, VariantResult]
    statistical_result: StatisticalResult | None
    confidence_level: float
    total_runs: int

    def __str__(self) -> str:
        """Pretty-print the comparison result."""
        lines = ["Comparison Results", "=" * 40]

        for name, v in self.variants.items():
            marker = " << WINNER" if name == self.winner else ""
            lines.append(
                f"  {name}: {v.success_rate:.0%} success, "
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

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "winner": self.winner,
            "confidence_level": self.confidence_level,
            "total_runs": self.total_runs,
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


def _run_variant(
    name: str,
    prompt_class: type,
    inputs: list[dict[str, Any]],
    model: str,
    temperature: float,
    runs_per_input: int,
    success_fn: Callable[[Any], bool] | None,
    metric_fn: Callable[[Any], float] | None,
) -> tuple[VariantResult, VariantStats]:
    """Run a single variant against all inputs and collect stats."""
    from flowprompt.testing.experiment import ExperimentResult

    vr = VariantResult(name=name)
    vs = VariantStats(name=name)

    for inp in inputs:
        for _ in range(runs_per_input):
            start = time.time()
            try:
                prompt = prompt_class(**inp)
                output = prompt.run(model=model, temperature=temperature)
                latency_ms = (time.time() - start) * 1000

                success = success_fn(output) if success_fn else True
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
    success_fn: Callable[[Any], bool] | None,
    metric_fn: Callable[[Any], float] | None,
) -> tuple[VariantResult, VariantStats]:
    """Run a single variant asynchronously against all inputs."""
    from flowprompt.testing.experiment import ExperimentResult

    vr = VariantResult(name=name)
    vs = VariantStats(name=name)

    for inp in inputs:
        for _ in range(runs_per_input):
            start = time.time()
            try:
                prompt = prompt_class(**inp)
                output = await prompt.arun(model=model, temperature=temperature)
                latency_ms = (time.time() - start) * 1000

                success = success_fn(output) if success_fn else True
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
            if statistical_result.effect_size > 0:
                winner = best_name
            else:
                winner = control_name

    return ComparisonResult(
        winner=winner,
        variants=variant_results,
        statistical_result=statistical_result,
        confidence_level=confidence_level,
        total_runs=total_runs,
    )


def compare(
    prompts: dict[str, type],
    inputs: list[dict[str, Any]],
    model: str = "gpt-4o",
    *,
    success_fn: Callable[[Any], bool] | None = None,
    metric_fn: Callable[[Any], float] | None = None,
    confidence_level: float = 0.95,
    runs_per_input: int = 1,
    temperature: float = 0.0,
    test_type: str = "z_test",
) -> ComparisonResult:
    """Compare prompt variants with statistical significance testing.

    This is the simplest way to A/B test prompts. Pass a dict of named
    prompt classes and a list of test inputs, and get back a result
    telling you which prompt performs better.

    Args:
        prompts: Dict mapping variant names to Prompt subclasses.
        inputs: List of input dicts to test each variant against.
        model: Model to use for all variants.
        success_fn: Optional function to determine if an output is successful.
            Defaults to treating all non-error outputs as successful.
        metric_fn: Optional function to compute a numeric metric from output.
        confidence_level: Confidence level for significance testing (default 0.95).
        runs_per_input: Number of times to run each input per variant (default 1).
        temperature: Temperature for LLM calls (default 0.0).
        test_type: Statistical test type ("z_test", "chi_squared", "t_test", "bayesian").

    Returns:
        ComparisonResult with winner, per-variant stats, and significance test.

    Raises:
        ValueError: If fewer than 2 prompts or 0 inputs are provided.

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
        ...     model="gpt-4o-mini",
        ... )
        >>> print(result)
    """
    if len(prompts) < 2:
        raise ValueError("compare() requires at least 2 prompt variants")
    if len(inputs) < 1:
        raise ValueError("compare() requires at least 1 input")

    variant_results: dict[str, VariantResult] = {}
    variant_stats: dict[str, VariantStats] = {}

    for name, prompt_class in prompts.items():
        vr, vs = _run_variant(
            name, prompt_class, inputs, model, temperature,
            runs_per_input, success_fn, metric_fn,
        )
        variant_results[name] = vr
        variant_stats[name] = vs

    return _build_result(variant_results, variant_stats, confidence_level, test_type)


async def acompare(
    prompts: dict[str, type],
    inputs: list[dict[str, Any]],
    model: str = "gpt-4o",
    *,
    success_fn: Callable[[Any], bool] | None = None,
    metric_fn: Callable[[Any], float] | None = None,
    confidence_level: float = 0.95,
    runs_per_input: int = 1,
    temperature: float = 0.0,
    test_type: str = "z_test",
) -> ComparisonResult:
    """Async version of compare() that runs variants in parallel.

    Same arguments and return type as compare(). Uses asyncio.gather
    to run all variants concurrently.

    Example:
        >>> result = await acompare(
        ...     {"v1": PromptV1, "v2": PromptV2},
        ...     inputs=[{"text": "hello"}],
        ...     model="gpt-4o-mini",
        ... )
    """
    if len(prompts) < 2:
        raise ValueError("acompare() requires at least 2 prompt variants")
    if len(inputs) < 1:
        raise ValueError("acompare() requires at least 1 input")

    tasks = [
        _arun_variant(
            name, prompt_class, inputs, model, temperature,
            runs_per_input, success_fn, metric_fn,
        )
        for name, prompt_class in prompts.items()
    ]

    results = await asyncio.gather(*tasks)

    variant_results: dict[str, VariantResult] = {}
    variant_stats: dict[str, VariantStats] = {}
    for vr, vs in results:
        variant_results[vr.name] = vr
        variant_stats[vs.name] = vs

    return _build_result(variant_results, variant_stats, confidence_level, test_type)
