# API Reference

## Core Classes

### Prompt

The base class for all prompts.

```python
from flowprompt import Prompt
from pydantic import BaseModel

class MyPrompt(Prompt):
    system = "System message"
    user = "User message with {variable}"

    class Output(BaseModel):
        field: str
```

**Class Attributes:**
- `system` (str): The system message template
- `user` (str): The user message template
- `__version__` (str, optional): Version string for the prompt
- `Output` (BaseModel, optional): Pydantic model for structured output

**Instance Properties:**
- `version`: Returns the prompt version
- `content_hash`: Returns a hash of the prompt content

**Methods:**

#### `run(model: str, **kwargs) -> Output`
Execute the prompt synchronously.

```python
result = MyPrompt(variable="value").run(model="gpt-4o")
```

#### `arun(model: str, **kwargs) -> Output`
Execute the prompt asynchronously.

```python
result = await MyPrompt(variable="value").arun(model="gpt-4o")
```

#### `stream(model: str, **kwargs) -> StreamingResponse`
Stream the response synchronously.

```python
for chunk in MyPrompt(variable="value").stream(model="gpt-4o"):
    print(chunk.delta)
```

#### `astream(model: str, **kwargs) -> AsyncStreamingResponse`
Stream the response asynchronously.

```python
async for chunk in MyPrompt(variable="value").astream(model="gpt-4o"):
    print(chunk.delta)
```

---

## Streaming

### StreamChunk

Represents a single chunk in a streaming response.

**Attributes:**
- `delta` (str): The text content of this chunk
- `finish_reason` (str | None): Why the stream ended (if last chunk)

### StreamingResponse

Synchronous streaming iterator.

### AsyncStreamingResponse

Asynchronous streaming iterator.

---

## Caching

### configure_cache()

Configure the global cache settings.

```python
from flowprompt import configure_cache, FileCache

# Memory cache (default)
configure_cache(enabled=True, default_ttl=3600)

# File cache
configure_cache(
    backend=FileCache(".flowprompt_cache"),
    default_ttl=86400
)
```

**Parameters:**
- `enabled` (bool): Enable or disable caching
- `default_ttl` (int): Default time-to-live in seconds
- `backend` (CacheBackend): Cache backend to use

### get_cache()

Get the global cache instance.

```python
from flowprompt import get_cache

cache = get_cache()
print(cache.stats)  # {'hits': 10, 'misses': 5, 'hit_rate': 0.67}
```

### MemoryCache

In-memory cache backend.

### FileCache

File-based persistent cache backend.

```python
from flowprompt import FileCache

cache = FileCache("/path/to/cache/dir")
```

---

## Tracing

### get_tracer()

Get the global tracer instance.

```python
from flowprompt import get_tracer

tracer = get_tracer()
summary = tracer.get_summary()
```

**Summary Fields:**
- `total_requests`: Number of requests made
- `total_tokens`: Total tokens used
- `total_input_tokens`: Input tokens used
- `total_output_tokens`: Output tokens used
- `total_cost_usd`: Total cost in USD
- `avg_latency_ms`: Average latency in milliseconds

### configure_tracer()

Configure the global tracer.

```python
from flowprompt import configure_tracer

configure_tracer(enabled=True)
```

---

## Prompt Loading

### load_prompt()

Load a single prompt from a YAML or JSON file.

```python
from flowprompt import load_prompt

MyPrompt = load_prompt("prompts/my_prompt.yaml")
result = MyPrompt(variable="value").run(model="gpt-4o")
```

### load_prompts()

Load all prompts from a directory.

```python
from flowprompt import load_prompts

prompts = load_prompts("prompts/")
result = prompts["MyPrompt"](variable="value").run(model="gpt-4o")
```

### PromptConfig

Configuration model for YAML/JSON prompt files.

**Fields:**
- `name` (str): Prompt name
- `version` (str, optional): Version string
- `description` (str, optional): Description
- `system` (str): System message
- `user` (str): User message template
- `output_schema` (dict, optional): JSON Schema for output

---

## Field

Extended field configuration for prompt variables.

```python
from flowprompt import Prompt, Field

class MyPrompt(Prompt):
    text: str = Field(description="Input text to process")
    max_length: int = Field(default=100, ge=1, le=1000)
```

---

## Optimization

### optimize()

Optimize a prompt using a dataset and metric.

```python
from flowprompt.optimize import optimize, ExampleDataset, Example, ExactMatch

dataset = ExampleDataset([
    Example(input={"text": "John is 25"}, output={"name": "John", "age": 25}),
])

result = optimize(
    MyPrompt,
    dataset=dataset,
    metric=ExactMatch(),
    strategy="fewshot",  # or "instruction", "optuna", "bootstrap"
    model="gpt-4o"
)

OptimizedPrompt = result.best_prompt_class
```

**Parameters:**
- `prompt_class` (type[Prompt]): Prompt class to optimize
- `dataset` (ExampleDataset): Training/evaluation examples
- `metric` (Metric): Metric to optimize for
- `model` (str): Model to use for evaluation
- `strategy` (str): Optimization strategy
- `config` (OptimizationConfig, optional): Configuration options

**Returns:** `OptimizationResult` with the optimized prompt.

### Metrics

#### ExactMatch

Exact string match metric.

```python
from flowprompt.optimize import ExactMatch

metric = ExactMatch(case_sensitive=True, strip=True)
```

#### F1Score

Token-level F1 score.

```python
from flowprompt.optimize import F1Score

metric = F1Score()
```

#### StructuredAccuracy

Field-level accuracy for structured outputs.

```python
from flowprompt.optimize import StructuredAccuracy

metric = StructuredAccuracy(fields=["name", "age"])
```

#### ContainsMatch

Check if output contains expected substring.

```python
from flowprompt.optimize import ContainsMatch

metric = ContainsMatch(case_sensitive=False)
```

#### RegexMatch

Check if output matches regex pattern.

```python
from flowprompt.optimize import RegexMatch
import re

metric = RegexMatch(r"\d{3}-\d{4}", flags=re.IGNORECASE)
```

#### CustomMetric

Create custom metrics.

```python
from flowprompt.optimize import CustomMetric

def my_metric(predictions, ground_truth):
    return sum(p == gt for p, gt in zip(predictions, ground_truth)) / len(predictions)

metric = CustomMetric("my_metric", my_metric)
```

#### CompositeMetric

Combine multiple metrics with weights.

```python
from flowprompt.optimize import CompositeMetric, ExactMatch, F1Score

metric = CompositeMetric([
    (ExactMatch(), 0.6),
    (F1Score(), 0.4),
])
```

### ExampleDataset

Dataset container for optimization.

```python
from flowprompt.optimize import ExampleDataset, Example

dataset = ExampleDataset([
    Example(
        input={"text": "John is 25"},
        output={"name": "John", "age": 25}
    ),
])

# Split into train/test
train, test = dataset.split(train_ratio=0.7, seed=42)
```

### Optimizers

#### FewShotOptimizer

Optimizes few-shot example selection.

```python
from flowprompt.optimize import FewShotOptimizer

optimizer = FewShotOptimizer(
    num_examples=3,
    selection_strategy="bootstrap"  # or "random", "diverse", "similar"
)
result = optimizer.optimize(MyPrompt, dataset, metric, model="gpt-4o")
```

#### InstructionOptimizer

Optimizes prompt instructions using LLM feedback.

```python
from flowprompt.optimize import InstructionOptimizer

optimizer = InstructionOptimizer(
    optimizer_model="gpt-4o",
    num_candidates=5
)
result = optimizer.optimize(MyPrompt, dataset, metric)
```

#### OptunaOptimizer

Hyperparameter optimization using Optuna.

```python
from flowprompt.optimize import OptunaOptimizer

optimizer = OptunaOptimizer(n_trials=50, timeout=600)
result = optimizer.optimize(MyPrompt, dataset, metric)
```

#### BootstrapOptimizer

Self-improvement through bootstrapping.

```python
from flowprompt.optimize import BootstrapOptimizer

optimizer = BootstrapOptimizer(
    bootstrap_rounds=3,
    confidence_threshold=0.8
)
result = optimizer.optimize(MyPrompt, dataset, metric)
```

---

## A/B Testing

### compare()

One-call prompt comparison with statistical significance testing.

```python
from flowprompt import compare

result = compare(
    {"v1": PromptV1, "v2": PromptV2},
    inputs=[{"text": "sample"}],
    model="gpt-4o-mini",
    success_fn=lambda out: len(out) > 10,
    confidence_level=0.95,
    runs_per_input=3,
    temperature=0.0,
    test_type="z_test",
)
```

**Parameters:**
- `prompts` (dict[str, type]): Dict mapping variant names to Prompt subclasses (min 2)
- `inputs` (list[dict]): Test inputs to run each variant against (min 1)
- `model` (str): Model to use
- `expected` (list, optional): Expected outputs, same length as inputs. When provided, auto-generates success_fn from eval_metric.
- `eval_metric` (str | callable): Metric for expected output evaluation. Built-in: `"exact"`, `"contains"` (default), `"similarity"`. Or a `(output, expected) -> bool` callable.
- `success_fn` (callable, optional): Function to determine success. When both `expected` and `success_fn` are provided, receives `(output, expected_value)`.
- `metric_fn` (callable, optional): Function to compute numeric metric
- `confidence_level` (float): Confidence level for significance (default 0.95)
- `runs_per_input` (int): Runs per input per variant (default 1)
- `temperature` (float): LLM temperature (default 0.0)
- `test_type` (str): Statistical test ("z_test", "chi_squared", "t_test", "bayesian")

**Returns:** `ComparisonResult`

### acompare()

Async variant of `compare()` that runs variants in parallel via `asyncio.gather`.

```python
from flowprompt import acompare

result = await acompare(
    {"v1": PromptV1, "v2": PromptV2},
    inputs=[{"text": "sample"}],
    model="gpt-4o-mini",
)
```

### ComparisonResult

Result of comparing prompt variants.

**Attributes:**
- `winner` (str | None): Name of the winning variant, or None
- `variants` (dict[str, VariantResult]): Per-variant results
- `statistical_result` (StatisticalResult | None): Significance test result
- `confidence_level` (float): Confidence level used
- `total_runs` (int): Total runs across all variants

**Methods:**
- `__str__()`: Pretty-printed summary
- `to_dict()`: Serialize to dictionary

### VariantResult

Results for a single prompt variant.

**Attributes:**
- `name` (str): Variant name
- `samples` (int): Number of runs
- `successes` (int): Number of successful runs
- `success_rate` (float): Success fraction (0.0-1.0)
- `mean_latency_ms` (float): Average latency
- `total_cost_usd` (float): Total cost
- `outputs` (list): Outputs from each run
- `errors` (list[str]): Errors encountered

### Eval Metrics

Built-in functions for comparing outputs against expected values.

```python
from flowprompt.testing import exact_match, contains_match, similarity_match

exact_match("Hello World", "hello world")       # True (case-insensitive)
contains_match("The answer is 42.", "42")        # True
similarity_match("hello world", "hello worlds")  # True (threshold=0.7)
```

#### exact_match(output, expected) -> bool
Case-insensitive exact match after stripping whitespace.

#### contains_match(output, expected) -> bool
Check if expected value appears in the output (case-insensitive).

#### similarity_match(output, expected, threshold=0.7) -> bool
Compare using `difflib.SequenceMatcher`. Returns True if ratio >= threshold.

#### resolve_eval_metric(metric) -> Callable
Resolve a string name (`"exact"`, `"contains"`, `"similarity"`) or callable to a metric function.

### Pytest Fixtures

Auto-discovered via `pytest11` entry point. Install with `pip install flowprompt-ai[pytest]`.

#### fp (session-scoped)
`FlowPromptHelper` with `.compare()`, `.acompare()`, `.estimate_cost()`, `.Prompt()`.

#### fp_compare (function-scoped)
Returns a wrapper that calls `compare()` and returns a `PromptTestResult`.

### PromptTestResult

Wraps `ComparisonResult` with pytest-friendly assertions.

**Properties:**
- `is_significant` (bool): Whether the test found significance
- `winner` (str | None): Winning variant name
- `p_value` (float | None): P-value from significance test

**Methods:**
- `assert_significant(threshold=0.05)`: Fail unless p-value <= threshold
- `assert_winner(expected)`: Fail unless the named variant won
- `assert_no_errors()`: Fail if any variant had errors

All other attributes delegate to the underlying `ComparisonResult`.

### ABTestRunner

Main class for running A/B test experiments.

```python
from flowprompt.testing import ABTestRunner, ExperimentConfig, VariantConfig

runner = ABTestRunner()
runner.register_prompt("control", ControlPrompt)
runner.register_prompt("treatment", TreatmentPrompt)

config = ExperimentConfig(
    name="prompt_test",
    variants=[
        VariantConfig(name="control", prompt_class="control", is_control=True),
        VariantConfig(name="treatment", prompt_class="treatment"),
    ],
    min_samples=100,
    confidence_level=0.95,
)

runner.create_experiment(config)
runner.start_experiment(config.id)
```

**Methods:**

#### create_experiment(config)

Create a new experiment.

#### start_experiment(experiment_id)

Start running an experiment.

#### get_variant(experiment_id, user_id, context)

Get the allocated variant for a request.

#### run_prompt(experiment_id, variant_name, input_data, ...)

Run a prompt and record the result.

#### get_summary(experiment_id, test_type)

Get statistical summary of results.

```python
summary = runner.get_summary(config.id)
print(summary.summary_text())
```

### ExperimentConfig

Configuration for an A/B test experiment.

**Fields:**
- `name` (str): Experiment name
- `variants` (list[VariantConfig]): Variant configurations
- `allocation_strategy` (AllocationStrategy): Traffic allocation strategy
- `min_samples` (int): Minimum samples before analysis
- `max_samples` (int, optional): Maximum samples before auto-completion
- `confidence_level` (float): Confidence level for significance (default 0.95)

### VariantConfig

Configuration for a single variant.

**Fields:**
- `name` (str): Variant name
- `prompt_class` (str): Prompt class name
- `model` (str, optional): Model to use
- `temperature` (float, optional): Temperature setting
- `is_control` (bool): Whether this is the control variant

### AllocationStrategy

Traffic allocation strategies:

- `RANDOM`: Random allocation
- `ROUND_ROBIN`: Round-robin allocation
- `WEIGHTED`: Weighted allocation
- `EPSILON_GREEDY`: Epsilon-greedy (exploration/exploitation)
- `UCB`: Upper Confidence Bound
- `THOMPSON_SAMPLING`: Bayesian Thompson sampling

---

## Multimodal

### MultimodalPrompt

Base class for prompts with multimodal content.

```python
from flowprompt.multimodal import MultimodalPrompt, ImageContent
from pydantic import BaseModel

class ImageAnalyzer(MultimodalPrompt):
    system = "Analyze images in detail."
    user = "What do you see?"

    class Output(BaseModel):
        description: str
        objects: list[str]

result = ImageAnalyzer(
    images=[ImageContent.from_file("photo.jpg")]
).run(model="gpt-4o")
```

**Methods:**

#### add_image(image, detail)

Add an image to the prompt.

```python
prompt = prompt.add_image("photo.jpg", detail="high")
```

#### add_document(document, extract_images)

Add a document to the prompt.

```python
prompt = prompt.add_document("report.pdf", extract_images=True)
```

#### with_audio(audio)

Set audio content.

```python
prompt = prompt.with_audio("recording.mp3")
```

#### with_video(video, frame_interval, max_frames)

Set video content.

```python
prompt = prompt.with_video("video.mp4", frame_interval=1.0, max_frames=10)
```

### VisionPrompt

Specialized prompt for vision/image tasks.

```python
from flowprompt.multimodal import VisionPrompt

class Analyzer(VisionPrompt):
    system = "You are an image expert."
    user = "Describe this image."

result = Analyzer().with_image("photo.jpg").run(model="gpt-4o")
```

**Class Methods:**

#### VisionPrompt.describe(image, detail_level)

Create a prompt to describe an image.

```python
result = VisionPrompt.describe("photo.jpg", detail_level="comprehensive").run(model="gpt-4o")
```

#### VisionPrompt.compare(images, comparison_type)

Create a prompt to compare images.

```python
result = VisionPrompt.compare(
    ["before.jpg", "after.jpg"],
    comparison_type="differences"
).run(model="gpt-4o")
```

### DocumentPrompt

Specialized prompt for document analysis.

```python
from flowprompt.multimodal import DocumentPrompt

class Summarizer(DocumentPrompt):
    system = "Summarize documents."
    user = "What are the key points?"

result = Summarizer().with_document("report.pdf").run(model="gpt-4o")
```

**Class Methods:**

#### DocumentPrompt.summarize(document, length)

Create a prompt to summarize a document.

```python
result = DocumentPrompt.summarize("report.pdf", length="brief").run(model="gpt-4o")
```

#### DocumentPrompt.extract_info(document, info_type)

Extract specific information from a document.

```python
result = DocumentPrompt.extract_info(
    "contract.pdf",
    info_type="dates"
).run(model="gpt-4o")
```

### ImageContent

Image content handler.

```python
from flowprompt.multimodal import ImageContent

# From file
image = ImageContent.from_file("photo.jpg", detail="high")

# From URL
image = ImageContent.from_url("https://example.com/image.jpg")

# From base64
image = ImageContent.from_base64(b64_string, format="png")
```

### DocumentContent

Document content handler.

```python
from flowprompt.multimodal import DocumentContent

# From file (PDF, DOCX, TXT)
doc = DocumentContent.from_file("report.pdf", extract_images=True, max_pages=20)

# Access extracted text
print(doc.text)

# Access page images
for page_img in doc.pages:
    print(page_img.to_message_content())
```

### AudioContent

Audio content handler.

```python
from flowprompt.multimodal import AudioContent

# From file
audio = AudioContent.from_file("recording.mp3")

# From URL
audio = AudioContent.from_url("https://example.com/audio.mp3")
```

### VideoContent

Video content handler.

```python
from flowprompt.multimodal import VideoContent

# From file with frame extraction
video = VideoContent.from_file(
    "video.mp4",
    frame_interval=1.0,  # Extract frame every 1 second
    max_frames=10
)

# Frames are converted to ImageContent
for frame in video.frames:
    print(frame)
```

---

## CLI Commands

### flowprompt init

Initialize a new FlowPrompt project.

```bash
flowprompt init my-project
```

### flowprompt run

Run a prompt from a YAML/JSON file.

```bash
flowprompt run prompts/my_prompt.yaml --var text="Hello" --model gpt-4o
```

### flowprompt test

Test all prompts in a directory.

```bash
flowprompt test --prompts-dir prompts/
```

### flowprompt stats

View usage statistics.

```bash
flowprompt stats
```

### flowprompt cache-stats

View cache statistics.

```bash
flowprompt cache-stats
```

### flowprompt cache-clear

Clear the cache.

```bash
flowprompt cache-clear
```

### flowprompt list-prompts

List all available prompts.

```bash
flowprompt list-prompts --prompts-dir prompts/
```
