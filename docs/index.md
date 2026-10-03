# FlowPrompt

**Stop guessing which prompt works. Measure it.**

FlowPrompt is a Python library for prompts as typed classes and for
comparing prompts, models and pipelines with statistics you can trust.

- **Typed prompts.** Prompts are Pydantic classes with templates and a
  structured `Output` model, run against 100+ providers through LiteLLM.
- **Paired A/B tests.** `compare()` runs every variant on the same inputs and
  tests the difference with the exact McNemar test (pass/fail) or a paired
  permutation test (scores), corrected for multiple comparisons.
- **Answers, not just p-values.** Each result reports accuracy with
  confidence intervals, latency, cost and cost per correct answer, a
  plain-English verdict, and how many inputs you would need to detect a
  smaller difference.
- **Built for CI.** Markdown and HTML reports, a `flowprompt compare`
  command that fails the build on a significant regression, a pytest plugin,
  and `FakeLLM` for offline tests.

## A first comparison

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


result = compare(
    {"concise": Concise, "chatty": Chatty},
    inputs=[{"text": text} for text, _ in reviews],
    expected=[label for _, label in reviews],
    eval_metric="exact",
    model="gpt-4o-mini",
)
print(result.verdict)
result.save_report("report.html")
```

![Comparison report](assets/report-example.png)

The complete script, which runs offline with a simulated model, is
[`examples/12_ab_test_report.py`](https://github.com/yotambraun/flowprompt/blob/main/examples/12_ab_test_report.py).

## Where to go next

- [Installation](installation.md) and the [Quick start](quickstart.md)
- [A/B testing guide](ab-testing.md): variants, scorers, reports, sample size
- [Prompt tests in CI](ci.md): `FakeLLM`, `flowprompt compare`, GitHub Actions
- [Statistical methods](statistics.md): exactly what is computed, with references
- [Your prompt A/B test is probably lying to you](false-winners.md): a
  simulation study of how often common analyses crown a false winner
- [Caching, tracing and cost](observability.md)
- [Optimization](optimization.md) and [Multimodal prompts](multimodal.md)
- [API reference](api.md)

## Getting help

- [GitHub Issues](https://github.com/yotambraun/flowprompt/issues) to report bugs or request features
- [Discussions](https://github.com/yotambraun/flowprompt/discussions) for questions

FlowPrompt is released under the MIT License.
