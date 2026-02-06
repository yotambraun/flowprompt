"""A/B Testing Framework for FlowPrompt.

This module provides comprehensive A/B testing capabilities:
- Experiment configuration and management
- Traffic allocation strategies (random, weighted, multi-armed bandits)
- Statistical significance testing
- Results analysis and reporting

Example:
    >>> from flowprompt import Prompt
    >>> from flowprompt.testing import (
    ...     ABTestRunner, ExperimentConfig, VariantConfig, create_simple_experiment
    ... )
    >>>
    >>> class PromptV1(Prompt):
    ...     system = "You are helpful."
    ...     user = "Process: {text}"
    >>>
    >>> class PromptV2(Prompt):
    ...     system = "You are a helpful assistant. Be concise."
    ...     user = "Process the following: {text}"
    >>>
    >>> # Create experiment
    >>> config, runner = create_simple_experiment(
    ...     name="prompt_comparison",
    ...     control_prompt=PromptV1,
    ...     treatment_prompts=[("v2", PromptV2)],
    ... )
    >>>
    >>> # Run experiment
    >>> runner.start_experiment(config.id)
    >>> variant = runner.get_variant(config.id, user_id="user123")
    >>> result = runner.run_prompt(config.id, variant.name, input_data={"text": "hello"})
    >>>
    >>> # Analyze results
    >>> summary = runner.get_summary(config.id)
    >>> print(summary.summary_text())
"""

from flowprompt.testing.allocation import (
    EpsilonGreedyAllocator,
    RandomAllocator,
    RoundRobinAllocator,
    ThompsonSamplingAllocator,
    TrafficAllocator,
    UCBAllocator,
    WeightedAllocator,
    get_allocator,
)
from flowprompt.testing.compare import (
    ComparisonResult,
    VariantResult,
    acompare,
    compare,
    estimate_compare_cost,
)
from flowprompt.testing.experiment import (
    AllocationStrategy,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    ExperimentStore,
    VariantConfig,
    VariantStats,
)
from flowprompt.testing.runner import (
    ABTestRunner,
    ExperimentSummary,
    create_simple_experiment,
)
from flowprompt.testing.statistics import (
    StatisticalResult,
    bayesian_ab_test,
    chi_squared_test,
    run_significance_test,
    t_test_means,
    two_proportion_z_test,
)

__all__ = [
    # Experiment
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentStatus",
    "ExperimentStore",
    "VariantConfig",
    "VariantStats",
    "AllocationStrategy",
    # Allocation
    "TrafficAllocator",
    "RandomAllocator",
    "RoundRobinAllocator",
    "WeightedAllocator",
    "EpsilonGreedyAllocator",
    "UCBAllocator",
    "ThompsonSamplingAllocator",
    "get_allocator",
    # Statistics
    "StatisticalResult",
    "two_proportion_z_test",
    "chi_squared_test",
    "t_test_means",
    "bayesian_ab_test",
    "run_significance_test",
    # Comparison
    "compare",
    "acompare",
    "estimate_compare_cost",
    "ComparisonResult",
    "VariantResult",
    # Runner
    "ABTestRunner",
    "ExperimentSummary",
    "create_simple_experiment",
]
