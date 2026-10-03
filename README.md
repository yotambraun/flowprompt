# FlowPrompt

**Stop guessing which prompt works. Measure it.**

[![PyPI](https://img.shields.io/pypi/v/flowprompt-ai.svg)](https://pypi.org/project/flowprompt-ai/)
[![Downloads](https://static.pepy.tech/badge/flowprompt-ai)](https://pepy.tech/project/flowprompt-ai)
[![Downloads/Month](https://static.pepy.tech/badge/flowprompt-ai/month)](https://pepy.tech/project/flowprompt-ai)
[![Python](https://img.shields.io/pypi/pyversions/flowprompt-ai.svg)](https://pypi.org/project/flowprompt-ai/)
[![License](https://img.shields.io/pypi/l/flowprompt-ai.svg)](https://github.com/yotambraun/flowprompt/blob/main/LICENSE)
[![Tests](https://github.com/yotambraun/flowprompt/workflows/CI/badge.svg)](https://github.com/yotambraun/flowprompt/actions)
[![codecov](https://codecov.io/gh/yotambraun/flowprompt/graph/badge.svg?token=3IDNOYK3D3)](https://codecov.io/gh/yotambraun/flowprompt)

FlowPrompt defines prompts as typed Python classes and compares prompts,
models and pipelines with paired significance tests, so "B is better than A"
means something.

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
print(result.age)  # 25
```

---

## Compare Prompts

Run both prompts on the same labelled examples and let FlowPrompt test the
difference:

```python
from flowprompt import Prompt, compare


class Concise(Prompt):
    system = (
        "Classify the sentiment. Reply with one word: positive, negative or neutral."
    )
    user = "Review: {text}"


class Chatty(Prompt):
    system = "You are a helpful assistant. Analyze the sentiment of the review."
    user = "Review: {text}"


reviews = [
    ("Absolutely love it, works perfectly.", "positive"),
    ("Broke after two days. Waste of money.", "negative"),
    ("It arrived on Tuesday.", "neutral"),
    ("Best purchase I made this year!", "positive"),
    ("The battery life is awful.", "negative"),
    ("Comes in a blue box.", "neutral"),
    ("Great value and fast shipping.", "positive"),
    ("Customer support never answered.", "negative"),
    ("The manual is twelve pages long.", "neutral"),
    ("Oh great, another charger that melts.", "negative"),
    ("Exceeded my expectations.", "positive"),
    ("Stopped working, returning it.", "negative"),
]

result = compare(
    {"concise": Concise, "chatty": Chatty},
    inputs=[{"text": text} for text, _ in reviews],
    expected=[label for _, label in reviews],
    eval_metric="exact",
    model="gpt-4o-mini",
)
print(result)
```

Output (answered offline by a simulated model in which the chatty prompt
often replies in a full sentence, which the exact-match grader rejects; with
a real model your numbers will differ):

```text
Comparison Results
========================================================================
2 variants | 12 inputs | accuracy vs expected outputs | model gpt-4o-mini

  Variant           Accuracy      95% CI  Latency mean/p95       Cost  Cost/correct  Errors
  ----------------  --------  ----------  ----------------  ---------  ------------  ------
  concise * winner     91.7%  64.6-98.5%        12 / 12 ms  $0.000063    $0.0000058       0
  chatty               33.3%  13.8-60.9%        21 / 26 ms  $0.000099     $0.000025       0

Paired comparisons vs concise
  Comparison         Difference          95% CI       p  Result
  -----------------  ----------  --------------  ------  -----------
  chatty vs concise   -58.3 pts  [-85.7, -14.3]  0.0391  significant

Verdict: concise beats chatty by 58.3 points (95% CI 14.3 to 85.7), p=0.0391: significant.
Method: Paired design: every variant ran on the same 12 inputs. Test: exact McNemar test. Interval: Agresti-Min interval. Significance level: 0.05.
Cost: $0.00016 total for 24 LLM calls.
```

With only 12 reviews the interval is wide (14 to 86 points): the direction
is clear, the size is not. `result.save_report("report.html")` writes the
same result as a shareable page, and `result.to_markdown()` as a pull
request comment.

---

## A/B Testing You Can Trust

Most prompt comparisons put two accuracy numbers side by side and run a
test that assumes the two prompts saw unrelated data. They did not: both
answered the same inputs. FlowPrompt's `compare()` is built around that:

- **Paired tests.** Only inputs where the prompts disagree carry evidence.
  Pass/fail results use the exact McNemar test; scores use a paired
  permutation test.
- **The input is the unit of analysis.** With `runs_per_input=5`, the five
  runs are averaged per input. Repeats measure run-to-run variation; they
  are not extra data.
- **Multiple-comparison correction.** With three or more variants, p-values
  are Holm-adjusted before a winner is named.
- **Effect sizes with confidence intervals**, a plain-English verdict, and an
  honest "not enough data" when no difference could be significant.
- **Planning.** `plan_sample_size(0.10, discordance=0.2)` tells you how many
  inputs you need to detect a 10-point difference.

Why it matters: in 240,000 simulated comparisons of *equally good* prompts,
the analysis FlowPrompt used up to version 0.3.0 declared a false winner
**52.5% of the time** (five prompts, five runs per input, temperature 0).
The current analysis did so 1.2% of the time, within the 5% it promises.
Read [Your prompt A/B test is probably lying to you](docs/false-winners.md)
for the details and how to reproduce them.

Variants are not limited to Prompt classes:

<!-- not-run -->
```python
from flowprompt.testing import model_variants

compare(
    {
        "baseline": ExtractUser,  # Prompt class
        "mini": (ExtractUser, "gpt-4o-mini"),  # same prompt, another model
        "rules": lambda inp: my_regex_extractor(inp["text"]),  # any callable
    },
    inputs=inputs,
    expected=expected,
    eval_metric="exact",  # or "contains", "regex", "numeric", "similarity", a function
)
compare(
    model_variants(ExtractUser, ["gpt-4o-mini", "gpt-4o"]),
    inputs=inputs,
    expected=expected,
)
```

For live traffic (sticky assignment, weighted splits, epsilon-greedy, UCB
and Thompson sampling bandits) use `ABTestRunner`; see the
[A/B testing guide](docs/ab-testing.md#live-traffic-experiments).

---

## Test Prompts in CI

Fail the build when a prompt change makes things significantly worse:

```bash
pip install "flowprompt-ai[cli]"
flowprompt compare variants.py dataset.jsonl --model gpt-4o-mini --metric exact \
    --control current --markdown "$GITHUB_STEP_SUMMARY" --fail-on-regression
```

Or from pytest, with the auto-discovered plugin:

<!-- not-run -->
```python
import pytest


@pytest.mark.prompt_test
def test_candidate_is_not_worse(fp_compare):
    result = fp_compare(
        {"current": CurrentPrompt, "candidate": CandidatePrompt},
        inputs=inputs,
        expected=expected,
        eval_metric="exact",
        model="gpt-4o-mini",
    )
    result.assert_no_errors()
    assert result.winner != "current", result.verdict
```

`FakeLLM` answers LLM calls offline for fast, deterministic unit tests. See
[Prompt tests in CI](docs/ci.md) for a complete GitHub Actions workflow.

---

## Installation

```bash
pip install flowprompt-ai
```

> **Note:** The package is installed as `flowprompt-ai` but imported as `flowprompt`

**Optional extras:**

```bash
pip install flowprompt-ai[all]        # Everything
pip install flowprompt-ai[pytest]     # Pytest fixtures & markers
pip install flowprompt-ai[cli]        # CLI tools (flowprompt compare, ...)
pip install flowprompt-ai[tracing]    # OpenTelemetry support
pip install flowprompt-ai[multimodal] # Images, PDFs, audio, video
```

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
print(result.sentiment)  # "positive"
print(result.confidence)  # 0.95
```

Models that support native JSON schema get schema-constrained output. Others
fall back to JSON mode with the schema in the system message.

---

## Multi-Provider Support

Switch between 100+ providers (through LiteLLM) with a single parameter.

<!-- not-run -->
```python
result = prompt.run(model="gpt-4o")  # OpenAI
result = prompt.run(model="anthropic/claude-3-5-sonnet-20241022")  # Anthropic
result = prompt.run(model="gemini/gemini-2.0-flash-exp")  # Google
result = prompt.run(model="ollama/llama3")  # Local
```

---

## More Features

| Feature | Example |
|---------|---------|
| **Cost tracking** | `with track_usage() as calls: ...` -- tokens and cost per call, from the provider's usage |
| **Caching** | `configure_cache(enabled=True, default_ttl=3600)` -- identical requests are served from the cache |
| **Observability** | `configure_tracer()` then `tracer.get_summary()` -- requests, tokens, cost, latency, errors |
| **Reports** | `result.save_report("ab.html")` -- self-contained HTML, Markdown or JSON |
| **Optimization** | DSPy-style few-shot and instruction optimization with `flowprompt.optimize` |
| **Streaming** | `for chunk in prompt.stream(model="gpt-4o"): ...` |
| **YAML prompts** | `load_prompt("prompts/my_prompt.yaml")` |
| **Multimodal** | Images, PDFs, audio via `flowprompt.multimodal` |
| **CLI** | `flowprompt compare`, `flowprompt optimize`, `flowprompt run` |

---

## How It Compares

Statistical features for comparing prompt variants, checked against each
project's source code and documentation in October 2026:

| | FlowPrompt | promptfoo | LangSmith | Langfuse | inspect-ai |
|---|:---:|:---:|:---:|:---:|:---:|
| Side-by-side scores per variant | Yes | Yes | Yes | Yes | Yes |
| Confidence intervals per variant | Yes | No | No | No | Yes |
| Paired significance test between variants | Yes | No | No | No | No |
| Multiple-comparison correction | Yes | No | No | No | No |
| Sample-size planning | Yes | No | No | No | No |

We also checked DSPy, DeepEval, pydantic-evals, Mirascope and Opik; none of
them runs a significance test between variants either. These tools do many
things FlowPrompt does not (hosted dashboards, tracing UIs, large assertion
libraries); this table only covers the statistics. Corrections are welcome.

---

## Documentation

The documentation lives in [`docs/`](docs/) and is published with MkDocs.

- **[Quick Start Guide](docs/quickstart.md)** -- Get started in 5 minutes
- **[A/B Testing Guide](docs/ab-testing.md)** -- Variants, scorers, reports, sample size, live experiments
- **[Prompt tests in CI](docs/ci.md)** -- FakeLLM, `flowprompt compare`, GitHub Actions
- **[Statistical methods](docs/statistics.md)** -- Exactly what is computed, with references
- **[Caching, tracing and cost](docs/observability.md)**
- **[API Reference](docs/api.md)** -- Complete API documentation
- **[Optimization Guide](docs/optimization.md)** -- Improve prompts automatically
- **[Examples](examples/)** -- Runnable example scripts (all run offline in CI)

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/yotambraun/flowprompt.git
cd flowprompt
uv venv && uv sync --all-extras
uv run pytest
uv run python scripts/run_examples.py
```

---

## License

MIT License -- see [LICENSE](LICENSE) for details.

---

**Made with care by [Yotam Braun](https://github.com/yotambraun)**

[GitHub](https://github.com/yotambraun/flowprompt) | [PyPI](https://pypi.org/project/flowprompt-ai/) | [Issues](https://github.com/yotambraun/flowprompt/issues)
