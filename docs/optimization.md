# Optimization Guide

**Automatic prompt optimization for better performance**

FlowPrompt provides DSPy-style optimization to automatically improve your prompts using example data and metrics. The optimization module supports multiple strategies to find the best prompt configuration for your use case.

## Table of Contents

- [Quick Start](#quick-start)
- [Metrics](#metrics)
- [Example Management](#example-management)
- [Optimization Strategies](#optimization-strategies)
- [Configuration](#configuration)
- [Best Practices](#best-practices)

## Quick Start

Here's a simple example of optimizing a prompt:

```python
from flowprompt import Prompt
from flowprompt.optimize import optimize, ExampleDataset, Example, ExactMatch
from pydantic import BaseModel


# Define your prompt
class ExtractUser(Prompt):
    system = "Extract user information from text."
    user = "Text: {text}"

    class Output(BaseModel):
        name: str
        age: int


# Create training examples
dataset = ExampleDataset(
    [
        Example(
            input={"text": "John is 25 years old"}, output={"name": "John", "age": 25}
        ),
        Example(input={"text": "Alice, age 30"}, output={"name": "Alice", "age": 30}),
        Example(
            input={"text": "Bob Smith is 35"}, output={"name": "Bob Smith", "age": 35}
        ),
        # Add more examples...
    ]
)

# Optimize the prompt
result = optimize(
    ExtractUser,
    dataset=dataset,
    metric=ExactMatch(),
    strategy="fewshot",
    model="gpt-4o",
)

# Use the optimized prompt
OptimizedPrompt = result.best_prompt_class
print(f"Improvement: {result.best_score:.2%}")
print(result.summary())

# Use it in production
optimized_result = OptimizedPrompt(text="Jane is 28").run(model="gpt-4o")
```

## Metrics

Metrics evaluate how well your prompt performs. FlowPrompt provides several built-in metrics:

### ExactMatch

Perfect for classification or exact extraction tasks.

```python
from flowprompt.optimize import ExactMatch

metric = ExactMatch(
    case_sensitive=True,  # Case-sensitive comparison
    strip=True,  # Strip whitespace before comparing
)

# Returns 1.0 if prediction exactly matches ground truth, 0.0 otherwise
# Final score is the average across all examples
```

### F1Score

Token-level F1 score for extractive tasks where partial matches matter.

```python
from flowprompt.optimize import F1Score

# Default tokenizer splits on whitespace
metric = F1Score()


# Custom tokenizer
def custom_tokenizer(text: str) -> list[str]:
    return text.lower().split()


metric = F1Score(tokenizer=custom_tokenizer)
```

### StructuredAccuracy

For structured (Pydantic) outputs, compares individual fields.

```python
from flowprompt.optimize import StructuredAccuracy

# Compare all fields
metric = StructuredAccuracy()

# Compare specific fields only
metric = StructuredAccuracy(fields=["name", "age"])

# Returns field-level accuracy in metadata
result = metric.evaluate(predictions, ground_truth)
print(result.metadata["field_accuracy"])  # {'name': 0.95, 'age': 0.90}
```

### ContainsMatch

Checks if the output contains the expected substring.

```python
from flowprompt.optimize import ContainsMatch

metric = ContainsMatch(case_sensitive=False)
```

### RegexMatch

Validates output format using regular expressions.

```python
from flowprompt.optimize import RegexMatch
import re

# Check for phone number format
metric = RegexMatch(r"\d{3}-\d{4}", flags=re.IGNORECASE)
```

### CustomMetric

Create your own metrics.

```python
from flowprompt.optimize import CustomMetric


def my_metric(predictions, ground_truth):
    correct = sum(p == gt for p, gt in zip(predictions, ground_truth))
    return correct / len(predictions)


metric = CustomMetric("my_metric", my_metric)
```

### CompositeMetric

Combine multiple metrics with weights.

```python
from flowprompt.optimize import CompositeMetric, ExactMatch, F1Score

metric = CompositeMetric(
    [
        (ExactMatch(), 0.6),  # 60% weight on exact matches
        (F1Score(), 0.4),  # 40% weight on F1 score
    ]
)
```

## Example Management

### Creating Datasets

```python
from flowprompt.optimize import ExampleDataset, Example

# Create individual examples
example = Example(
    input={"text": "John is 25"},
    output={"name": "John", "age": 25},
    metadata={"source": "training", "difficulty": "easy"},
)

# Build a dataset
dataset = ExampleDataset([example1, example2, example3])

# Add examples dynamically
dataset.add(new_example)
dataset.add_many([ex1, ex2, ex3])

# Check size
print(len(dataset))  # Number of examples
```

### Splitting Datasets

```python
# Split into train and test
train_data, test_data = dataset.split(
    train_ratio=0.7,
    seed=42,  # For reproducibility
)

# Sample random examples
sample = dataset.sample(n=10, seed=42)

# Filter examples
filtered = dataset.filter(lambda ex: ex.metadata.get("difficulty") == "hard")
```

### Example Selection

The `ExampleSelector` chooses optimal few-shot examples:

```python
from flowprompt.optimize import ExampleSelector

selector = ExampleSelector(
    strategy="random",  # or "diverse", "similar", "bootstrap"
    k=3,  # Number of examples to select
    seed=42,
)

# Select examples
selected = selector.select(dataset)

# For similarity-based selection, provide input
selected = selector.select(dataset, input_data={"text": "Alice is 30"})

# Update performance scores (for bootstrap strategy)
for example in selected:
    selector.update_performance(example, score=0.9)
```

**Selection Strategies:**

- **random**: Random selection (baseline)
- **diverse**: Maximizes diversity across examples
- **similar**: Selects examples similar to the current input
- **bootstrap**: Selects based on past performance

### Example Bootstrapping

Automatically collect high-quality examples from production:

```python
from flowprompt.optimize import ExampleBootstrapper

bootstrapper = ExampleBootstrapper(
    min_confidence=0.8,  # Minimum confidence to accept
    max_examples=100,  # Maximum examples to collect
    validate_fn=lambda input, output: output.age > 0,  # Optional validator
)

# Record successful executions
success = bootstrapper.record(
    input_data={"text": "John is 25"},
    output={"name": "John", "age": 25},
    confidence=0.95,
    metadata={"timestamp": "2024-01-01"},
)

# Get collected examples as dataset
collected = bootstrapper.get_dataset()

# Clear when needed
count = bootstrapper.clear()
```

## Optimization Strategies

### Few-Shot Optimization

Finds the best few-shot examples to include in your prompt.

```python
from flowprompt.optimize import FewShotOptimizer

optimizer = FewShotOptimizer(
    num_examples=3,
    selection_strategy="bootstrap",  # Best performing examples
)

result = optimizer.optimize(
    prompt_class=ExtractUser, dataset=dataset, metric=ExactMatch(), model="gpt-4o"
)
```

The optimized prompt will include the selected examples in the system message.

### Instruction Optimization

Uses an LLM to generate and test improved prompt instructions.

```python
from flowprompt.optimize import InstructionOptimizer

optimizer = InstructionOptimizer(
    optimizer_model="gpt-4o",  # Model for generating improvements
    num_candidates=5,  # Candidates per iteration
)

result = optimizer.optimize(
    prompt_class=ExtractUser,
    dataset=dataset,
    metric=ExactMatch(),
    model="gpt-4o-mini",  # Model for evaluation
)
```

The optimizer iteratively generates better instructions based on performance feedback.

### Optuna Optimization

Hyperparameter search using Optuna (requires `optuna` package).

```python
from flowprompt.optimize import OptunaOptimizer

optimizer = OptunaOptimizer(
    n_trials=50,  # Number of trials
    timeout=600,  # Timeout in seconds (optional)
)

result = optimizer.optimize(
    prompt_class=ExtractUser, dataset=dataset, metric=ExactMatch(), model="gpt-4o"
)

# View optimization history
for trial in result.history:
    print(
        f"Trial {trial['trial']}: score={trial['score']:.4f}, "
        f"temp={trial['temperature']:.2f}, examples={trial['num_examples']}"
    )
```

Optimizes temperature, few-shot example count, and other hyperparameters.

### Bootstrap Optimization

Self-improvement through bootstrapping unlabeled data.

```python
from flowprompt.optimize import BootstrapOptimizer

optimizer = BootstrapOptimizer(
    bootstrap_rounds=3,
    confidence_threshold=0.8,
    validator_fn=lambda input, output: output["age"] > 0,
)

result = optimizer.optimize(
    prompt_class=ExtractUser,
    dataset=dataset,  # Can include unlabeled data
    metric=ExactMatch(),
    model="gpt-4o",
)
```

Runs the prompt on unlabeled data, collects high-confidence outputs, and uses them as training examples.

## Configuration

### OptimizationConfig

Fine-tune the optimization process:

```python
from flowprompt.optimize import optimize, OptimizationConfig

config = OptimizationConfig(
    max_iterations=10,  # Maximum optimization iterations
    num_candidates=5,  # Candidates per iteration
    train_size=20,  # Training examples per iteration
    eval_size=50,  # Evaluation examples
    temperature_range=(0.0, 0.7),  # Temperature range to explore
    seed=42,  # Random seed
    early_stopping_patience=3,  # Stop after N iterations without improvement
    early_stopping_threshold=0.01,  # Minimum improvement threshold
)

result = optimize(
    ExtractUser, dataset=dataset, metric=ExactMatch(), strategy="fewshot", config=config
)
```

### Accessing Results

```python
# Best performing prompt class
OptimizedPrompt = result.best_prompt_class

# Best score achieved
print(f"Score: {result.best_score:.2%}")

# Configuration that achieved best score
print(result.best_config)

# Optimization history
for i, entry in enumerate(result.history):
    print(f"Iteration {entry['iteration']}: {entry['score']:.4f}")

# Summary statistics
print(f"Total iterations: {result.iterations}")
print(f"Improvements made: {result.improvements}")

# Human-readable summary
print(result.summary())
```

## Best Practices

### 1. Create Diverse Training Data

Include a variety of examples covering different patterns and edge cases:

```python
dataset = ExampleDataset(
    [
        # Different formats
        Example(input={"text": "John is 25"}, output={"name": "John", "age": 25}),
        Example(
            input={"text": "Alice, 30 years old"}, output={"name": "Alice", "age": 30}
        ),
        Example(
            input={"text": "Bob Smith (age: 35)"},
            output={"name": "Bob Smith", "age": 35},
        ),
        # Edge cases
        Example(input={"text": "Jane"}, output={"name": "Jane", "age": None}),
        Example(input={"text": "No data here"}, output={"name": "", "age": None}),
    ]
)
```

### 2. Choose the Right Metric

- Use **ExactMatch** for classification tasks
- Use **F1Score** for extractive tasks where partial credit matters
- Use **StructuredAccuracy** for complex structured outputs
- Combine metrics with **CompositeMetric** when you care about multiple aspects

### 3. Start Simple

Begin with few-shot optimization before trying more complex strategies:

```python
# Start here
result = optimize(ExtractUser, dataset, ExactMatch(), strategy="fewshot")

# If few-shot doesn't improve enough, try instruction optimization
result = optimize(ExtractUser, dataset, ExactMatch(), strategy="instruction")

# For final tuning, use Optuna
result = optimize(ExtractUser, dataset, ExactMatch(), strategy="optuna")
```

### 4. Use Train/Test Splits

Always evaluate on held-out data:

```python
train_data, test_data = dataset.split(train_ratio=0.7, seed=42)

# Optimize on training data
result = optimize(
    ExtractUser,
    dataset=train_data,  # Only training data
    metric=ExactMatch(),
    strategy="fewshot",
)

# Evaluate on test data
OptimizedPrompt = result.best_prompt_class
test_predictions = []
test_ground_truth = []

for example in test_data:
    pred = OptimizedPrompt(**example.input).run(model="gpt-4o")
    test_predictions.append(pred)
    test_ground_truth.append(example.output)

test_score = ExactMatch().evaluate(test_predictions, test_ground_truth)
print(f"Test score: {test_score.value:.2%}")
```

### 5. Monitor Costs

Optimization requires multiple LLM calls. Monitor and control costs:

```python
from flowprompt import get_tracer

# Enable tracing
result = optimize(ExtractUser, dataset, ExactMatch(), strategy="fewshot")

# Check costs
summary = get_tracer().get_summary()
print(f"Optimization cost: ${summary['total_cost_usd']:.2f}")
print(f"Total calls: {summary['total_requests']}")
```

### 6. Save Optimized Prompts

Export optimized prompts for reuse:

```python
# Get the optimized prompt
OptimizedPrompt = result.best_prompt_class

# Access the improved instructions
print(OptimizedPrompt.system)
print(OptimizedPrompt.user)

# Save to a YAML file for version control
# (Manual export - use the prompt values)
```

### 7. Iterate and Refine

Optimization is iterative. Use insights from one run to improve the next:

```python
# Round 1: Baseline few-shot
result1 = optimize(ExtractUser, dataset, ExactMatch(), strategy="fewshot")

# Round 2: Refine instructions based on errors
result2 = optimize(
    result1.best_prompt_class, dataset, ExactMatch(), strategy="instruction"
)

# Round 3: Final hyperparameter tuning
result3 = optimize(result2.best_prompt_class, dataset, ExactMatch(), strategy="optuna")
```

## Advanced Usage

### Custom Optimization Loop

For maximum control, implement your own optimization loop:

```python
from flowprompt.optimize import ExampleSelector, format_examples_for_prompt

selector = ExampleSelector(strategy="diverse", k=3)
best_score = 0.0
best_examples = []

for iteration in range(10):
    # Select candidate examples
    candidates = selector.select(dataset)

    # Create prompt with examples
    examples_str = format_examples_for_prompt(candidates)

    class TestPrompt(ExtractUser):
        system = f"{ExtractUser.system}\n\nExamples:\n{examples_str}"

    # Evaluate
    predictions = []
    ground_truth = []
    for ex in test_data:
        pred = TestPrompt(**ex.input).run(model="gpt-4o")
        predictions.append(pred)
        ground_truth.append(ex.output)

    score = ExactMatch().evaluate(predictions, ground_truth).value

    if score > best_score:
        best_score = score
        best_examples = candidates
        print(f"Iteration {iteration}: New best score {score:.2%}")

    # Update selector
    for ex in candidates:
        selector.update_performance(ex, score)
```

### Parallel Optimization

Optimize multiple prompts in parallel:

```python
from concurrent.futures import ProcessPoolExecutor

prompts = [PromptV1, PromptV2, PromptV3]

with ProcessPoolExecutor() as executor:
    futures = [
        executor.submit(optimize, prompt, dataset, ExactMatch(), "fewshot")
        for prompt in prompts
    ]

    results = [f.result() for f in futures]

# Compare results
for i, result in enumerate(results):
    print(f"Prompt {i + 1}: {result.best_score:.2%}")
```

## Next Steps

- Learn about [A/B Testing](ab-testing.md) to compare optimized prompts in production
- Check the [API Reference](api.md) for detailed documentation
- See [Examples](https://github.com/yotambraun/flowprompt/tree/main/examples) for more optimization patterns
