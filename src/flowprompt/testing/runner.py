"""A/B test runner for executing experiments.

Provides the main interface for running A/B tests:
- Experiment lifecycle management
- Automatic variant allocation
- Result collection and analysis
- Early stopping
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from flowprompt.testing.allocation import TrafficAllocator, get_allocator
from flowprompt.testing.experiment import (
    AllocationStrategy,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    ExperimentStore,
    VariantConfig,
    VariantStats,
)
from flowprompt.testing.statistics import StatisticalResult, run_significance_test

if TYPE_CHECKING:
    from flowprompt.core.prompt import Prompt

OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass
class ExperimentSummary:
    """Summary of an experiment's current state.

    Attributes:
        experiment: The experiment configuration.
        status: Current status.
        total_samples: Total samples across all variants.
        variant_stats: Statistics per variant.
        winner: Winning variant if determined.
        statistical_result: Result of significance test.
        recommendations: Actionable recommendations.
    """

    experiment: ExperimentConfig
    status: ExperimentStatus
    total_samples: int
    variant_stats: dict[str, VariantStats]
    winner: VariantConfig | None = None
    statistical_result: StatisticalResult | None = None
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment.id,
            "experiment_name": self.experiment.name,
            "status": self.status.value,
            "total_samples": self.total_samples,
            "variant_stats": {
                name: {
                    "samples": stats.samples,
                    "success_rate": stats.success_rate,
                    "mean_metric": stats.mean_metric,
                    "mean_latency_ms": stats.mean_latency_ms,
                    "total_cost_usd": stats.total_cost_usd,
                    "confidence_interval": stats.confidence_interval,
                }
                for name, stats in self.variant_stats.items()
            },
            "winner": self.winner.name if self.winner else None,
            "statistical_result": {
                "significant": self.statistical_result.significant,
                "p_value": self.statistical_result.p_value,
                "effect_size": self.statistical_result.effect_size,
            }
            if self.statistical_result
            else None,
            "recommendations": self.recommendations,
        }

    def summary_text(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Experiment: {self.experiment.name} ({self.experiment.id})",
            f"Status: {self.status.value}",
            f"Total Samples: {self.total_samples}",
            "",
            "Variant Performance:",
        ]

        for name, stats in self.variant_stats.items():
            variant = self.experiment.get_variant(name)
            control_marker = " (control)" if variant and variant.is_control else ""
            lines.append(
                f"  {name}{control_marker}: "
                f"n={stats.samples}, "
                f"success={stats.success_rate:.2%}, "
                f"CI=[{stats.confidence_interval[0]:.2%}, {stats.confidence_interval[1]:.2%}]"
            )

        if self.statistical_result:
            lines.extend(
                [
                    "",
                    "Statistical Analysis:",
                    f"  P-value: {self.statistical_result.p_value:.4f}",
                    f"  Effect Size: {self.statistical_result.effect_size:+.2%}",
                    f"  Significant: {'Yes' if self.statistical_result.significant else 'No'}",
                ]
            )

        if self.winner:
            lines.extend(["", f"Winner: {self.winner.name}"])

        if self.recommendations:
            lines.extend(["", "Recommendations:"])
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


class ABTestRunner:
    """Main class for running A/B test experiments.

    Manages experiment lifecycle, allocates traffic, collects results,
    and performs statistical analysis.

    Example:
        >>> from flowprompt.testing import ABTestRunner, ExperimentConfig, VariantConfig
        >>>
        >>> config = ExperimentConfig(
        ...     name="prompt_comparison",
        ...     variants=[
        ...         VariantConfig(name="control", prompt_class="PromptV1", is_control=True),
        ...         VariantConfig(name="treatment", prompt_class="PromptV2"),
        ...     ]
        ... )
        >>>
        >>> runner = ABTestRunner()
        >>> runner.create_experiment(config)
        >>> runner.start_experiment(config.id)
        >>>
        >>> # Run prompts
        >>> variant = runner.get_variant(config.id, user_id="user123")
        >>> result = runner.run_prompt(config.id, variant.name, input_data={"text": "hello"})
        >>>
        >>> # Get summary
        >>> summary = runner.get_summary(config.id)
        >>> print(summary.summary_text())
    """

    def __init__(
        self,
        store: ExperimentStore | None = None,
        prompt_registry: dict[str, type[Prompt[Any]]] | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            store: Store for experiments and results. Creates in-memory store if None.
            prompt_registry: Registry mapping prompt class names to classes.
        """
        self._store = store or ExperimentStore()
        self._prompt_registry = prompt_registry or {}
        self._allocators: dict[str, TrafficAllocator] = {}

    def register_prompt(
        self,
        name: str,
        prompt_class: type[Prompt[Any]],
    ) -> None:
        """Register a prompt class for use in experiments.

        Args:
            name: Name to register the prompt under.
            prompt_class: The prompt class.
        """
        self._prompt_registry[name] = prompt_class

    def create_experiment(self, config: ExperimentConfig) -> ExperimentConfig:
        """Create a new experiment.

        Args:
            config: Experiment configuration.

        Returns:
            The created experiment configuration.
        """
        config.status = ExperimentStatus.DRAFT
        self._store.save_experiment(config)

        # Create allocator
        self._allocators[config.id] = get_allocator(config.allocation_strategy)

        return config

    def start_experiment(self, experiment_id: str) -> ExperimentConfig:
        """Start an experiment.

        Args:
            experiment_id: ID of the experiment to start.

        Returns:
            The updated experiment configuration.
        """
        config = self._store.get_experiment(experiment_id)
        if config is None:
            raise ValueError(f"Experiment not found: {experiment_id}")

        config.status = ExperimentStatus.RUNNING
        config.start_time = datetime.utcnow()
        self._store.save_experiment(config)

        return config

    def pause_experiment(self, experiment_id: str) -> ExperimentConfig:
        """Pause an experiment.

        Args:
            experiment_id: ID of the experiment to pause.

        Returns:
            The updated experiment configuration.
        """
        config = self._store.get_experiment(experiment_id)
        if config is None:
            raise ValueError(f"Experiment not found: {experiment_id}")

        config.status = ExperimentStatus.PAUSED
        self._store.save_experiment(config)

        return config

    def complete_experiment(
        self,
        experiment_id: str,
        winner: str | None = None,
    ) -> ExperimentConfig:
        """Complete an experiment.

        Args:
            experiment_id: ID of the experiment to complete.
            winner: Name of the winning variant (optional).

        Returns:
            The updated experiment configuration.
        """
        config = self._store.get_experiment(experiment_id)
        if config is None:
            raise ValueError(f"Experiment not found: {experiment_id}")

        config.status = ExperimentStatus.COMPLETED
        config.end_time = datetime.utcnow()
        if winner:
            config.metadata["winner"] = winner
        self._store.save_experiment(config)

        return config

    def get_variant(
        self,
        experiment_id: str,
        user_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> VariantConfig:
        """Get the variant to use for a request.

        Args:
            experiment_id: ID of the experiment.
            user_id: Optional user ID for sticky assignment.
            context: Optional context for allocation.

        Returns:
            The allocated variant configuration.
        """
        config = self._store.get_experiment(experiment_id)
        if config is None:
            raise ValueError(f"Experiment not found: {experiment_id}")

        if config.status != ExperimentStatus.RUNNING:
            # Return control if not running
            control = config.get_control()
            if control is None:
                raise ValueError("No variants configured")
            return control

        allocator = self._allocators.get(experiment_id)
        if allocator is None:
            allocator = get_allocator(config.allocation_strategy)
            self._allocators[experiment_id] = allocator

        return allocator.allocate(config, user_id, context)

    def run_prompt(
        self,
        experiment_id: str,
        variant_name: str,
        input_data: dict[str, Any],
        model: str | None = None,
        user_id: str | None = None,
        success_fn: Callable[[Any], bool] | None = None,
        metric_fn: Callable[[Any], float] | None = None,
    ) -> ExperimentResult:
        """Run a prompt for an experiment and record the result.

        Args:
            experiment_id: ID of the experiment.
            variant_name: Name of the variant to run.
            input_data: Input data for the prompt.
            model: Model override (uses variant's model if None).
            user_id: Optional user ID.
            success_fn: Function to determine if result is successful.
            metric_fn: Function to compute metric value from result.

        Returns:
            The recorded experiment result.
        """
        config = self._store.get_experiment(experiment_id)
        if config is None:
            raise ValueError(f"Experiment not found: {experiment_id}")

        variant = config.get_variant(variant_name)
        if variant is None:
            raise ValueError(f"Variant not found: {variant_name}")

        # Get prompt class
        prompt_class = self._prompt_registry.get(variant.prompt_class)
        if prompt_class is None:
            raise ValueError(f"Prompt class not registered: {variant.prompt_class}")

        # Run prompt
        start_time = time.time()
        try:
            prompt = prompt_class(**input_data)
            output = prompt.run(
                model=model or variant.model,
                temperature=variant.temperature,
            )
            success = success_fn(output) if success_fn else True
            metric_value = metric_fn(output) if metric_fn else float(success)
            error = None
        except Exception as e:
            output = None
            success = False
            metric_value = 0.0
            error = str(e)

        latency_ms = (time.time() - start_time) * 1000

        # Create result
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant_name=variant_name,
            user_id=user_id,
            input_data=input_data,
            output=output,
            success=success,
            metric_value=metric_value,
            latency_ms=latency_ms,
            metadata={"error": error} if error else {},
        )

        # Record result
        self._store.record_result(result)

        # Update allocator
        stats = self._store.get_stats(experiment_id).get(variant_name)
        if stats:
            allocator = self._allocators.get(experiment_id)
            if allocator:
                allocator.update(experiment_id, variant_name, stats)

        # Check for auto-completion
        self._check_completion(experiment_id)

        return result

    def record_result(
        self,
        experiment_id: str,
        variant_name: str,
        output: Any,
        input_data: dict[str, Any] | None = None,
        success: bool = True,
        metric_value: float = 1.0,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentResult:
        """Manually record an experiment result.

        Use this when running prompts outside the runner.

        Args:
            experiment_id: ID of the experiment.
            variant_name: Name of the variant.
            output: The prompt output.
            input_data: Input data used.
            success: Whether the result was successful.
            metric_value: Metric value for the result.
            latency_ms: Response latency.
            cost_usd: Cost of the request.
            user_id: Optional user ID.
            metadata: Additional metadata.

        Returns:
            The recorded experiment result.
        """
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant_name=variant_name,
            user_id=user_id,
            input_data=input_data or {},
            output=output,
            success=success,
            metric_value=metric_value,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            metadata=metadata or {},
        )

        self._store.record_result(result)

        # Update allocator
        stats = self._store.get_stats(experiment_id).get(variant_name)
        if stats:
            allocator = self._allocators.get(experiment_id)
            if allocator:
                allocator.update(experiment_id, variant_name, stats)

        # Check for auto-completion
        self._check_completion(experiment_id)

        return result

    def get_summary(
        self,
        experiment_id: str,
        test_type: str = "z_test",
    ) -> ExperimentSummary:
        """Get a summary of experiment results.

        Args:
            experiment_id: ID of the experiment.
            test_type: Type of statistical test to use.

        Returns:
            ExperimentSummary with current state and analysis.
        """
        config = self._store.get_experiment(experiment_id)
        if config is None:
            raise ValueError(f"Experiment not found: {experiment_id}")

        stats = self._store.get_stats(experiment_id)
        total_samples = sum(s.samples for s in stats.values())

        # Perform statistical analysis
        control = config.get_control()
        statistical_result = None
        winner = None
        recommendations = []

        if control and len(stats) >= 2:
            control_stats = stats.get(control.name)

            # Compare each treatment to control. Live traffic assigns
            # different requests to different variants, so the samples are
            # independent and the unpaired tests apply. With several
            # treatments the p-values are Holm-adjusted.
            results: list[tuple[str, StatisticalResult]] = []
            for name, treatment_stats in stats.items():
                if name == control.name:
                    continue
                if control_stats and treatment_stats:
                    result = run_significance_test(
                        control_stats,
                        treatment_stats,
                        test_type=test_type,
                        confidence_level=config.confidence_level,
                    )
                    if test_type == "t_test":
                        diff = treatment_stats.mean_metric - control_stats.mean_metric
                    else:
                        diff = treatment_stats.success_rate - control_stats.success_rate
                    result.difference = diff
                    result.control, result.treatment = control.name, name
                    results.append((name, result))

            if len(results) > 1:
                from flowprompt.testing.paired import holm_adjust

                adjusted = holm_adjust([r.p_value for _, r in results])
                alpha = 1 - config.confidence_level
                for (_, r), adj in zip(results, adjusted, strict=True):
                    r.adjusted_p = adj
                    r.significant = adj < alpha

            better = [
                (n, r) for n, r in results if r.significant and (r.difference or 0) > 0
            ]
            worse = [
                (n, r) for n, r in results if r.significant and (r.difference or 0) < 0
            ]
            if better:
                best_name, statistical_result = max(
                    better, key=lambda item: item[1].difference or 0.0
                )
                winner = config.get_variant(best_name)
            elif results and len(worse) == len(results):
                _, statistical_result = min(
                    worse, key=lambda item: abs(item[1].difference or 0.0)
                )
                winner = control
            elif results:
                _, statistical_result = min(
                    results,
                    key=lambda item: (
                        item[1].adjusted_p
                        if item[1].adjusted_p is not None
                        else item[1].p_value
                    ),
                )

            # Generate recommendations
            if total_samples < config.min_samples:
                recommendations.append(
                    f"Continue collecting data (need {config.min_samples - total_samples} more samples)"
                )
            elif not statistical_result or not statistical_result.significant:
                if statistical_result and statistical_result.sample_size_recommendation:
                    recommendations.append(
                        f"Consider increasing sample size to {statistical_result.sample_size_recommendation} per variant"
                    )
                else:
                    recommendations.append("No significant difference detected yet")
            else:
                recommendations.append(
                    f"Consider deploying {winner.name if winner else 'the best variant'} as the new default"
                )

        return ExperimentSummary(
            experiment=config,
            status=config.status,
            total_samples=total_samples,
            variant_stats=stats,
            winner=winner,
            statistical_result=statistical_result,
            recommendations=recommendations,
        )

    def _check_completion(self, experiment_id: str) -> None:
        """Check if experiment should auto-complete."""
        config = self._store.get_experiment(experiment_id)
        if config is None or config.status != ExperimentStatus.RUNNING:
            return

        stats = self._store.get_stats(experiment_id)
        total_samples = sum(s.samples for s in stats.values())

        # Check max samples
        if config.max_samples and total_samples >= config.max_samples:
            self.complete_experiment(experiment_id)
            return

        # Check for early stopping (significant result with enough samples).
        # Note: testing after every result once min_samples is reached is
        # "peeking" and raises the false-positive rate above alpha. Set
        # min_samples to the planned sample size (see plan_sample_size) or
        # use max_samples for a fixed-horizon test.
        if total_samples >= config.min_samples:
            summary = self.get_summary(experiment_id)
            if summary.winner is not None:
                self.complete_experiment(experiment_id, winner=summary.winner.name)


def create_simple_experiment(
    name: str,
    control_prompt: type[Prompt[Any]],
    treatment_prompts: list[tuple[str, type[Prompt[Any]]]],
    model: str = "gpt-4o",
    allocation_strategy: AllocationStrategy = AllocationStrategy.RANDOM,
    min_samples: int = 100,
    confidence_level: float = 0.95,
) -> tuple[ExperimentConfig, ABTestRunner]:
    """Convenience function to create a simple A/B test.

    Args:
        name: Experiment name.
        control_prompt: The control prompt class.
        treatment_prompts: List of (name, prompt_class) tuples for treatments.
        model: Model to use.
        allocation_strategy: Traffic allocation strategy.
        min_samples: Minimum samples before analysis.
        confidence_level: Confidence level for significance.

    Returns:
        Tuple of (experiment_config, runner).

    Example:
        >>> config, runner = create_simple_experiment(
        ...     name="prompt_improvement",
        ...     control_prompt=PromptV1,
        ...     treatment_prompts=[("v2", PromptV2), ("v3", PromptV3)],
        ... )
        >>> runner.start_experiment(config.id)
    """
    # Create variants
    variants = [
        VariantConfig(
            name="control",
            prompt_class="control",
            model=model,
            is_control=True,
        )
    ]

    for treatment_name, _ in treatment_prompts:
        variants.append(
            VariantConfig(
                name=treatment_name,
                prompt_class=treatment_name,
                model=model,
            )
        )

    # Create config
    config = ExperimentConfig(
        name=name,
        variants=variants,
        allocation_strategy=allocation_strategy,
        min_samples=min_samples,
        confidence_level=confidence_level,
    )

    # Create runner and register prompts
    runner = ABTestRunner()
    runner.register_prompt("control", control_prompt)
    for treatment_name, prompt_class in treatment_prompts:
        runner.register_prompt(treatment_name, prompt_class)

    runner.create_experiment(config)

    return config, runner
