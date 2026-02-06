# FlowPrompt

**Stop guessing which prompt works. Measure it.**

[![PyPI](https://img.shields.io/pypi/v/flowprompt-ai.svg)](https://pypi.org/project/flowprompt-ai/)
[![Downloads](https://static.pepy.tech/badge/flowprompt-ai)](https://pepy.tech/project/flowprompt-ai)
[![Downloads/Month](https://static.pepy.tech/badge/flowprompt-ai/month)](https://pepy.tech/project/flowprompt-ai)
[![Python](https://img.shields.io/pypi/pyversions/flowprompt-ai.svg)](https://pypi.org/project/flowprompt-ai/)
[![License](https://img.shields.io/pypi/l/flowprompt-ai.svg)](https://github.com/yotambraun/flowprompt/blob/main/LICENSE)
[![Tests](https://github.com/yotambraun/flowprompt/workflows/CI/badge.svg)](https://github.com/yotambraun/flowprompt/actions)
[![codecov](https://codecov.io/gh/yotambraun/flowprompt/graph/badge.svg?token=3IDNOYK3D3)](https://codecov.io/gh/yotambraun/flowprompt)

---

## 30-Second Quickstart

Define prompts as Python classes. No API key needed to preview messages:

```python
from flowprompt import Prompt
from pydantic import BaseModel

class ExtractUser(Prompt):
    system = "Extract user info from text."
    user = "Text: {text}"

    class Output(BaseModel):
        name: str
        age: int

# Preview messages -- works without an API key
print(ExtractUser(text="John is 25").to_messages())
# [{'role': 'system', 'content': 'Extract user info from text.'},
#  {'role': 'user', 'content': 'Text: John is 25'}]

# Run against any LLM
result = ExtractUser(text="John is 25").run(model="gpt-4o")
print(result.name)  # "John"
print(result.age)   # 25
```

---

## Compare Prompts in 5 Lines

The killer feature: find which prompt actually works better, with statistical significance.

```python
from flowprompt import Prompt, compare

class Concise(Prompt):
    system = "Be concise."
    user = "Summarize: {text}"

class Detailed(Prompt):
    system = "Be thorough and detailed."
    user = "Provide a comprehensive summary of: {text}"

result = compare(
    {"concise": Concise, "detailed": Detailed},
    inputs=[{"text": "Python is a programming language..."}, ...],
    model="gpt-4o-mini",
    success_fn=lambda out: len(out) > 20,
)
print(result)
# Comparison Results
# ========================================
#   concise: 90% success, 245ms avg, 50 runs << WINNER
#   detailed: 72% success, 410ms avg, 50 runs
#
#   p=0.0231 (SIGNIFICANT)
#   effect size: -20.00%
```

---

## Installation

```bash
pip install flowprompt-ai
```

> **Note:** The package is installed as `flowprompt-ai` but imported as `flowprompt`

**Optional extras:**

```bash
pip install flowprompt-ai[all]        # Everything
pip install flowprompt-ai[cli]        # CLI tools
pip install flowprompt-ai[tracing]    # OpenTelemetry support
pip install flowprompt-ai[multimodal] # Images, PDFs, audio, video
```

---

## A/B Testing

FlowPrompt is the only Python LLM framework with built-in A/B testing.

**Quick comparison** with `compare()`:

```python
from flowprompt import compare

result = compare(
    {"v1": PromptV1, "v2": PromptV2, "v3": PromptV3},
    inputs=test_data,
    model="gpt-4o-mini",
    confidence_level=0.95,
)

if result.winner:
    print(f"Winner: {result.winner} (p={result.statistical_result.p_value:.4f})")
```

**Full experiment control** when you need production traffic splitting, sticky user assignment, or multi-armed bandits:

```python
from flowprompt.testing import create_simple_experiment

config, runner = create_simple_experiment(
    name="prompt_comparison",
    control_prompt=PromptV1,
    treatment_prompts=[("v2", PromptV2)],
    min_samples=100,
)

runner.start_experiment(config.id)
variant = runner.get_variant(config.id, user_id="user123")
result = runner.run_prompt(config.id, variant.name, input_data={"text": "..."})

summary = runner.get_summary(config.id)
if summary.winner:
    print(f"Winner: {summary.winner.name}")
```

**Six allocation strategies:** Random, Round-Robin, Weighted, Epsilon-Greedy, UCB, Thompson Sampling.

**Four statistical tests:** Z-test, Chi-squared, Welch's t-test, Bayesian.

---

## Structured Outputs

Define expected output as a Pydantic model. Parsing and validation are automatic.

```python
from pydantic import BaseModel, Field

class Sentiment(Prompt):
    system = "Analyze the sentiment of the given text."
    user = "Text: {text}"

    class Output(BaseModel):
        sentiment: str = Field(description="positive, negative, or neutral")
        confidence: float = Field(ge=0.0, le=1.0)

result = Sentiment(text="I love this!").run(model="gpt-4o")
print(result.sentiment)   # "positive"
print(result.confidence)  # 0.95
```

Models that support native JSON schema get guaranteed valid output. Others fall back to JSON mode with schema hints.

---

## Multi-Provider Support

Switch between 100+ providers with a single parameter.

```python
result = prompt.run(model="gpt-4o")                              # OpenAI
result = prompt.run(model="anthropic/claude-3-5-sonnet-20241022") # Anthropic
result = prompt.run(model="gemini/gemini-2.0-flash-exp")          # Google
result = prompt.run(model="ollama/llama3")                        # Local
```

---

## More Features

| Feature | Example |
|---------|---------|
| **Caching** | `configure_cache(enabled=True, default_ttl=3600)` -- cut costs 50-90% |
| **Optimization** | DSPy-style auto-improvement with `flowprompt.optimize` |
| **Streaming** | `for chunk in prompt.stream(model="gpt-4o"): ...` |
| **Observability** | `get_tracer().get_summary()` -- costs, tokens, latency |
| **YAML prompts** | `load_prompt("prompts/my_prompt.yaml")` |
| **Multimodal** | Images, PDFs, audio via `flowprompt.multimodal` |
| **CLI** | `flowprompt optimize prompt.py examples.json` |

---

## Comparison

| Feature | FlowPrompt | LangChain | Instructor | DSPy |
|---------|:----------:|:---------:|:----------:|:----:|
| **A/B testing** | **Built-in** | No | No | No |
| Structured outputs | Yes | Partial | **Best-in-class** | Yes |
| Auto-optimization | Yes | No | No | **Best-in-class** |
| Multi-provider | Yes | Yes | Yes | Partial |
| Caching | Yes | Yes | Yes | Yes |
| Cost tracking | Yes | Partial | No | No |
| Streaming | Yes | Yes | Yes | Yes |
| Import time | <100ms | ~2s | <100ms | ~6s |

---

## Documentation

- **[Quick Start Guide](docs/quickstart.md)** -- Get started in 5 minutes
- **[A/B Testing Guide](docs/ab-testing.md)** -- Run experiments
- **[API Reference](docs/api.md)** -- Complete API documentation
- **[Optimization Guide](docs/optimization.md)** -- Improve prompts automatically
- **[Examples](examples/)** -- Runnable example scripts

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/yotambraun/flowprompt.git
cd flowprompt
uv venv && uv sync --all-extras
uv run pytest
```

---

## License

MIT License -- see [LICENSE](LICENSE) for details.

---

**Made with care by [Yotam Braun](https://github.com/yotambraun)**

[GitHub](https://github.com/yotambraun/flowprompt) | [PyPI](https://pypi.org/project/flowprompt-ai/) | [Issues](https://github.com/yotambraun/flowprompt/issues)
