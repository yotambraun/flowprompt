# A/B Testing Guide

**Production experimentation framework for prompt comparison**

FlowPrompt's A/B testing module provides a complete framework for running controlled experiments in production, comparing different prompts or configurations, and making data-driven decisions.

## Table of Contents

- [Quick Compare](#quick-compare)
- [Full Experiment Control](#full-experiment-control)
- [Experiment Configuration](#experiment-configuration)
- [Traffic Allocation](#traffic-allocation)
- [Running Experiments](#running-experiments)
- [Statistical Analysis](#statistical-analysis)
- [Best Practices](#best-practices)

## Quick Compare

The fastest way to compare prompts -- no setup needed:

```python
from flowprompt import Prompt, compare

class PromptV1(Prompt):
    system = "You are a helpful assistant."
    user = "Process: {text}"

class PromptV2(Prompt):
    system = "You are a helpful assistant. Be concise and clear."
    user = "Please process the following text: {text}"

result = compare(
    {"v1": PromptV1, "v2": PromptV2},
    inputs=[{"text": "Hello world"}, {"text": "Test input"}],
    model="gpt-4o-mini",
    success_fn=lambda out: len(out) > 10,
)
print(result)  # Shows winner, success rates, p-value
```

For async execution with parallel variant runs:

```python
from flowprompt import acompare

result = await acompare(
    {"v1": PromptV1, "v2": PromptV2},
    inputs=test_data,
    model="gpt-4o-mini",
)
```

`compare()` returns a `ComparisonResult` with:
- `winner` -- name of the winning variant (or `None` if not significant)
- `variants` -- per-variant `VariantResult` with success rate, latency, errors
- `statistical_result` -- full `StatisticalResult` with p-value and effect size
- `to_dict()` -- serialize to dict for logging/storage

## Full Experiment Control

For production traffic splitting, sticky user assignment, or multi-armed bandits, use the full experiment API:

```python
from flowprompt import Prompt
from flowprompt.testing import create_simple_experiment
from pydantic import BaseModel

# Define your prompt variants
class PromptV1(Prompt):
    system = "You are a helpful assistant."
    user = "Process: {text}"

    class Output(BaseModel):
        result: str

class PromptV2(Prompt):
    system = "You are a helpful assistant. Be concise and clear."
    user = "Please process the following text: {text}"

    class Output(BaseModel):
        result: str

# Create experiment (automatically creates runner and registers prompts)
config, runner = create_simple_experiment(
    name="prompt_comparison",
    control_prompt=PromptV1,
    treatment_prompts=[("v2", PromptV2)],
    model="gpt-4o",
    min_samples=100
)

# Start the experiment
runner.start_experiment(config.id)

# Run prompts for users
for user_id in range(100):
    # Get variant for this user
    variant = runner.get_variant(config.id, user_id=f"user{user_id}")

    # Run the prompt and record result
    result = runner.run_prompt(
        config.id,
        variant.name,
        input_data={"text": f"Sample text {user_id}"}
    )

# Get statistical summary
summary = runner.get_summary(config.id)
print(summary.summary_text())

# Check if there's a winner
if summary.winner:
    print(f"Winner: {summary.winner.name}")
    print(f"Effect size: {summary.statistical_result.effect_size:+.2%}")
```

## Experiment Configuration

### Creating Experiments

Define experiments with full control over variants and settings:

```python
from flowprompt.testing import (
    ABTestRunner,
    ExperimentConfig,
    VariantConfig,
    AllocationStrategy
)

# Create detailed experiment configuration
config = ExperimentConfig(
    name="prompt_optimization_test",
    description="Testing improved instruction clarity",
    variants=[
        VariantConfig(
            name="control",
            prompt_class="PromptV1",
            model="gpt-4o",
            temperature=0.0,
            is_control=True,
            weight=1.0
        ),
        VariantConfig(
            name="treatment_a",
            prompt_class="PromptV2",
            model="gpt-4o",
            temperature=0.0,
            weight=1.0
        ),
        VariantConfig(
            name="treatment_b",
            prompt_class="PromptV3",
            model="gpt-4o",
            temperature=0.3,
            weight=0.5  # Less traffic
        ),
    ],
    allocation_strategy=AllocationStrategy.RANDOM,
    min_samples=100,
    max_samples=1000,
    confidence_level=0.95,
    metric="success_rate"
)

# Create runner and register prompts
runner = ABTestRunner()
runner.register_prompt("PromptV1", PromptV1)
runner.register_prompt("PromptV2", PromptV2)
runner.register_prompt("PromptV3", PromptV3)

# Create and start experiment
runner.create_experiment(config)
runner.start_experiment(config.id)
```

### Configuration Options

**ExperimentConfig:**
- `name`: Human-readable experiment name
- `description`: What you're testing
- `variants`: List of variant configurations
- `allocation_strategy`: How to distribute traffic
- `min_samples`: Minimum samples before statistical analysis
- `max_samples`: Auto-complete experiment after this many samples
- `confidence_level`: Required confidence level (default 0.95)
- `metric`: Primary metric to optimize ("success_rate", "mean_metric")

**VariantConfig:**
- `name`: Variant identifier
- `prompt_class`: Name of registered prompt class
- `model`: Model to use for this variant
- `temperature`: Temperature setting
- `weight`: Traffic weight (for weighted allocation)
- `is_control`: Mark as control/baseline variant
- `metadata`: Additional configuration

### Loading from YAML

Store experiment configurations in YAML files:

```yaml
# experiment.yaml
name: prompt_comparison
description: Testing instruction improvements
variants:
  - name: control
    prompt_class: PromptV1
    model: gpt-4o
    temperature: 0.0
    is_control: true
    weight: 1.0
  - name: treatment
    prompt_class: PromptV2
    model: gpt-4o
    temperature: 0.0
    weight: 1.0
allocation_strategy: random
min_samples: 100
confidence_level: 0.95
```

Load and use:

```python
config = ExperimentConfig.from_file("experiment.yaml")
runner.create_experiment(config)
```

## Traffic Allocation

FlowPrompt supports multiple traffic allocation strategies:

### Random Allocation

Random assignment with optional user stickiness (same user always gets same variant).

```python
config = ExperimentConfig(
    name="random_test",
    variants=[...],
    allocation_strategy=AllocationStrategy.RANDOM
)

# Sticky by default - same user_id always gets same variant
variant = runner.get_variant(config.id, user_id="user123")
```

### Round Robin

Equal distribution by cycling through variants.

```python
config = ExperimentConfig(
    name="roundrobin_test",
    variants=[...],
    allocation_strategy=AllocationStrategy.ROUND_ROBIN
)
```

### Weighted Allocation

Distribute traffic according to variant weights.

```python
config = ExperimentConfig(
    name="weighted_test",
    variants=[
        VariantConfig(name="control", ..., weight=2.0),    # 50% traffic
        VariantConfig(name="treatment_a", ..., weight=1.0), # 25% traffic
        VariantConfig(name="treatment_b", ..., weight=1.0), # 25% traffic
    ],
    allocation_strategy=AllocationStrategy.WEIGHTED
)
```

### Multi-Armed Bandits

Adaptive allocation strategies that learn which variants perform better:

#### Epsilon-Greedy

Explores with probability epsilon, exploits (best variant) otherwise.

```python
config = ExperimentConfig(
    name="epsilon_greedy_test",
    variants=[...],
    allocation_strategy=AllocationStrategy.EPSILON_GREEDY
)

# Epsilon decays over time, balancing exploration and exploitation
```

#### UCB (Upper Confidence Bound)

Balances exploration and exploitation using confidence bounds.

```python
config = ExperimentConfig(
    name="ucb_test",
    variants=[...],
    allocation_strategy=AllocationStrategy.UCB
)

# Allocates more traffic to promising variants while maintaining exploration
```

#### Thompson Sampling

Bayesian approach using Beta distributions.

```python
config = ExperimentConfig(
    name="thompson_test",
    variants=[...],
    allocation_strategy=AllocationStrategy.THOMPSON_SAMPLING
)

# Samples from posterior distributions to balance exploration/exploitation
```

## Running Experiments

### Basic Usage

```python
# Get variant for a request
variant = runner.get_variant(
    experiment_id=config.id,
    user_id="user123",  # Optional, for sticky assignment
    context={"location": "US"}  # Optional context
)

# Run the prompt
result = runner.run_prompt(
    experiment_id=config.id,
    variant_name=variant.name,
    input_data={"text": "Sample input"},
    model="gpt-4o",  # Optional override
    success_fn=lambda output: len(output.result) > 0,  # Custom success check
    metric_fn=lambda output: len(output.result) / 100  # Custom metric
)
```

### Custom Success and Metric Functions

Define what "success" means for your use case:

```python
def is_successful(output):
    """Custom success criteria."""
    return (
        output.result is not None and
        len(output.result) > 10 and
        "error" not in output.result.lower()
    )

def compute_metric(output):
    """Custom metric computation."""
    if output.result is None:
        return 0.0
    # Score based on length and quality
    length_score = min(len(output.result) / 200, 1.0)
    quality_score = 1.0 if "excellent" in output.result else 0.5
    return (length_score + quality_score) / 2

result = runner.run_prompt(
    config.id,
    variant.name,
    input_data={"text": "input"},
    success_fn=is_successful,
    metric_fn=compute_metric
)
```

### Recording External Results

If you run prompts outside the runner, record results manually:

```python
# Run prompt yourself
prompt = PromptV1(text="sample")
output = prompt.run(model="gpt-4o")

# Record the result
runner.record_result(
    experiment_id=config.id,
    variant_name="control",
    output=output,
    input_data={"text": "sample"},
    success=True,
    metric_value=0.85,
    latency_ms=250.0,
    cost_usd=0.0015,
    user_id="user123"
)
```

### Experiment Lifecycle

```python
# Start experiment
runner.start_experiment(config.id)

# Pause if needed
runner.pause_experiment(config.id)

# Resume (start again)
runner.start_experiment(config.id)

# Complete manually
runner.complete_experiment(config.id, winner="treatment")

# Auto-completion happens when:
# - max_samples is reached
# - Significant result detected with min_samples met
```

## Statistical Analysis

### Getting Results

```python
# Get comprehensive summary
summary = runner.get_summary(
    experiment_id=config.id,
    test_type="z_test"  # or "chi_squared", "t_test", "bayesian"
)

# Access results
print(summary.to_dict())
print(summary.summary_text())
```

### Statistical Tests

FlowPrompt provides multiple statistical tests:

#### Two-Proportion Z-Test (Default)

Best for comparing conversion/success rates:

```python
summary = runner.get_summary(config.id, test_type="z_test")

if summary.statistical_result:
    print(f"P-value: {summary.statistical_result.p_value:.4f}")
    print(f"Significant: {summary.statistical_result.significant}")
    print(f"Effect size: {summary.statistical_result.effect_size:+.2%}")
```

#### Chi-Squared Test

Tests independence of success/failure and variant:

```python
summary = runner.get_summary(config.id, test_type="chi_squared")
```

#### T-Test for Means

Compares mean metric values:

```python
summary = runner.get_summary(config.id, test_type="t_test")
```

#### Bayesian A/B Test

Provides probability that treatment is better:

```python
summary = runner.get_summary(config.id, test_type="bayesian")

if summary.statistical_result:
    details = summary.statistical_result.details
    print(f"P(treatment better): {details['prob_treatment_better']:.2%}")
    print(f"Expected lift: {details['expected_lift']:+.2%}")
    print(f"95% CI: [{details['ci_lower']:+.2%}, {details['ci_upper']:+.2%}]")
```

### Interpreting Results

```python
summary = runner.get_summary(config.id)

# Check status
print(f"Status: {summary.status.value}")
print(f"Total samples: {summary.total_samples}")

# Variant performance
for name, stats in summary.variant_stats.items():
    print(f"\n{name}:")
    print(f"  Samples: {stats.samples}")
    print(f"  Success rate: {stats.success_rate:.2%}")
    print(f"  Mean metric: {stats.mean_metric:.4f}")
    print(f"  Latency: {stats.mean_latency_ms:.1f}ms")
    print(f"  Cost: ${stats.total_cost_usd:.4f}")
    print(f"  95% CI: [{stats.confidence_interval[0]:.2%}, "
          f"{stats.confidence_interval[1]:.2%}]")

# Statistical significance
if summary.statistical_result:
    result = summary.statistical_result
    print(f"\nStatistical Analysis:")
    print(f"  Test: {result.test_name}")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  Significant: {'Yes' if result.significant else 'No'}")
    print(f"  Effect size: {result.effect_size:+.2%}")

    if result.power:
        print(f"  Statistical power: {result.power:.2%}")

    if result.sample_size_recommendation:
        print(f"  Recommended sample size: {result.sample_size_recommendation}")

# Winner determination
if summary.winner:
    print(f"\nWinner: {summary.winner.name}")

# Recommendations
if summary.recommendations:
    print("\nRecommendations:")
    for rec in summary.recommendations:
        print(f"  - {rec}")
```

### Viewing Raw Results

```python
# Get all results for an experiment
results = runner._store.get_results(config.id)

# Get results for specific variant
control_results = runner._store.get_results(config.id, variant_name="control")

# Access individual result
for result in results[:5]:
    print(f"Variant: {result.variant_name}")
    print(f"Input: {result.input_data}")
    print(f"Output: {result.output}")
    print(f"Success: {result.success}")
    print(f"Metric: {result.metric_value}")
    print()
```

## Best Practices

### 1. Define Clear Success Criteria

Before starting an experiment, define what success means:

```python
def is_successful(output):
    """
    Success criteria:
    1. Output is not empty
    2. Contains required fields
    3. Passes validation
    """
    if not output.result:
        return False
    if len(output.result) < 10:
        return False
    if not validate_format(output.result):
        return False
    return True
```

### 2. Calculate Required Sample Size

Determine sample size before starting:

```python
# For detecting a 10% improvement with 80% power
# Control success rate: 50%
# Treatment success rate: 55% (10% relative improvement)
# Typical requirement: 1,500-3,000 samples per variant

config = ExperimentConfig(
    name="my_test",
    variants=[...],
    min_samples=3000,  # Total across variants
    confidence_level=0.95
)
```

### 3. Use Sticky Assignment

Ensure consistent user experience:

```python
# Good: Same user always gets same variant
variant = runner.get_variant(config.id, user_id=user.id)

# Bad: User might see different variants
variant = runner.get_variant(config.id)  # No user_id
```

### 4. Monitor Early and Often

Check experiment health regularly:

```python
# Check every 100 samples
if summary.total_samples % 100 == 0:
    print(summary.summary_text())

    # Check for issues
    for name, stats in summary.variant_stats.items():
        if stats.samples < summary.total_samples * 0.2:
            print(f"Warning: {name} has low sample count")

        if stats.success_rate < 0.5:
            print(f"Warning: {name} has low success rate")
```

### 5. Consider Multiple Metrics

Don't optimize for success rate alone:

```python
# Track multiple aspects
summary = runner.get_summary(config.id)

for name, stats in summary.variant_stats.items():
    # Success rate
    print(f"{name} success: {stats.success_rate:.2%}")

    # Quality (mean metric)
    print(f"{name} quality: {stats.mean_metric:.4f}")

    # Performance
    print(f"{name} latency: {stats.mean_latency_ms:.1f}ms")

    # Cost efficiency
    cost_per_success = stats.total_cost_usd / stats.successes if stats.successes > 0 else 0
    print(f"{name} cost/success: ${cost_per_success:.4f}")
```

### 6. Avoid Peeking Too Early

Wait for minimum samples before making decisions:

```python
summary = runner.get_summary(config.id)

if summary.total_samples < config.min_samples:
    print("Warning: Not enough samples for reliable results")
    print(f"Current: {summary.total_samples}, Need: {config.min_samples}")
else:
    if summary.statistical_result.significant:
        print("Significant result detected!")
```

### 7. Use Persistence

Store experiment data for later analysis:

```python
from flowprompt.testing import ExperimentStore

# Create persistent store
store = ExperimentStore(storage_path=".experiments")

# Create runner with store
runner = ABTestRunner(store=store)

# Data is automatically saved to disk
# Survives restarts and can be analyzed offline
```

### 8. Test One Thing at a Time

Isolate variables for clear conclusions:

```python
# Good: Test instruction changes only
class Control(Prompt):
    system = "You are helpful."
    user = "Process: {text}"

class Treatment(Prompt):
    system = "You are helpful. Be concise."  # Only change
    user = "Process: {text}"

# Bad: Multiple changes
class Treatment(Prompt):
    system = "You are helpful. Be concise."  # Changed
    user = "Please process: {text}"           # Also changed
    # Can't tell which change caused the difference!
```

## Advanced Usage

### Multi-Variant Tests

Compare more than two variants:

```python
config = ExperimentConfig(
    name="multi_variant_test",
    variants=[
        VariantConfig(name="control", ..., is_control=True),
        VariantConfig(name="treatment_a", ...),
        VariantConfig(name="treatment_b", ...),
        VariantConfig(name="treatment_c", ...),
    ],
    min_samples=200  # 50 per variant minimum
)

# Each treatment is compared to control
summary = runner.get_summary(config.id)
# Winner is best performing treatment that beats control
```

### Sequential Testing

Monitor experiments in real-time with early stopping:

```python
# Run experiment with continuous monitoring
for i in range(1000):
    variant = runner.get_variant(config.id, user_id=f"user{i}")
    runner.run_prompt(config.id, variant.name, input_data={...})

    # Check every 50 samples
    if i % 50 == 0 and i >= config.min_samples:
        summary = runner.get_summary(config.id)
        if summary.statistical_result and summary.statistical_result.significant:
            print(f"Early stopping at {i} samples")
            runner.complete_experiment(config.id, winner=summary.winner.name)
            break
```

### Custom Allocators

Implement custom allocation logic:

```python
from flowprompt.testing.allocation import TrafficAllocator

class CustomAllocator(TrafficAllocator):
    def allocate(self, experiment, user_id=None, context=None):
        # Your custom logic
        if context and context.get("premium_user"):
            return experiment.variants[0]  # Premium users get best variant
        else:
            return random.choice(experiment.variants)

    def update(self, experiment_id, variant_name, stats):
        # Update allocator state based on results
        pass

# Use custom allocator
runner._allocators[config.id] = CustomAllocator()
```

### Integration with Monitoring

Export metrics to your monitoring system:

```python
import time

while experiment_running:
    summary = runner.get_summary(config.id)

    # Export to monitoring (e.g., Prometheus, DataDog)
    for name, stats in summary.variant_stats.items():
        metrics.gauge(f"experiment.{config.id}.{name}.success_rate", stats.success_rate)
        metrics.gauge(f"experiment.{config.id}.{name}.latency", stats.mean_latency_ms)
        metrics.gauge(f"experiment.{config.id}.{name}.samples", stats.samples)

    time.sleep(60)  # Update every minute
```

## Next Steps

- Learn about [Optimization](optimization.md) to improve prompts before A/B testing
- Check the [API Reference](api.md) for detailed documentation
- See [Examples](../examples/) for more A/B testing patterns
