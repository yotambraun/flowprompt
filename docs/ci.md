# Prompt tests in CI

Prompt changes deserve the same review as code changes. FlowPrompt gives you
three ways to put them under test, from cheapest to most thorough.

## 1. Offline tests with `FakeLLM`

`FakeLLM` answers every LLM call made through FlowPrompt locally and
deterministically, so you can test prompt rendering, parsing and your own
pipeline logic without an API key:

```python
from flowprompt import Prompt
from flowprompt.testing import FakeLLM
from pydantic import BaseModel


class ExtractUser(Prompt):
    system = "Extract the user."
    user = "Text: {text}"

    class Output(BaseModel):
        name: str
        age: int


def test_extract_user_parses_output():
    with FakeLLM(lambda messages: {"name": "Ada", "age": 36}) as fake:
        user = ExtractUser(text="Ada is 36").run(model="gpt-4o-mini")
    assert (user.name, user.age) == ("Ada", 36)
    assert fake.calls[0]["model"] == "gpt-4o-mini"
```

The responder receives the chat messages and returns a string, a dict or a
pydantic model. Without a responder, `FakeLLM` returns schema-shaped JSON for
prompts with an `Output` model and an echo otherwise. It reports token usage,
so cost accounting can be tested too.

## 2. `flowprompt compare` as a regression gate

`flowprompt compare` runs a paired comparison over a JSONL dataset and can
fail the build when a variant is significantly worse than the control:

```bash
pip install "flowprompt-ai[cli]"
flowprompt compare examples/ci/variants.py examples/ci/sentiment.jsonl \
    --model gpt-4o-mini --metric exact --control current \
    --markdown summary.md --html prompt-report.html --fail-on-regression
```

- **Variants file**: a Python file defining `VARIANTS = {"name": ...}` (Prompt
  classes, `(PromptClass, "model")` tuples or callables), or simply two or
  more Prompt classes.
- **Dataset**: one JSON object per line, either
  `{"input": {"text": "..."}, "expected": "positive"}` or flat template
  variables plus `expected`:

  ```json
  {"text": "Great value (#0)", "expected": "positive"}
  {"text": "Stopped working (#1)", "expected": "negative"}
  ```

- **Exit codes**: `0` finished (no regression), `1` a variant is
  significantly worse than the control (with `--fail-on-regression`),
  `2` invalid input.
- `--markdown` *appends* the report, so it can point at
  `$GITHUB_STEP_SUMMARY`. `--html` and `--json` write standalone reports.

Output of the command above, with the model answered offline by a simulated
model (the candidate prompt often answers in a full sentence, which the
exact-match grader rejects):

```text
Comparison Results
========================================================================
2 variants | 40 inputs | accuracy vs expected outputs | model gpt-4o-mini

  Variant           Accuracy      95% CI  Latency mean/p95      Cost  Cost/correct  Errors
  ----------------  --------  ----------  ----------------  --------  ------------  ------
  current * winner     92.5%  80.1-97.4%          0 / 0 ms  $0.00019    $0.0000052       0
  candidate            25.0%  14.2-40.2%          0 / 0 ms  $0.00027     $0.000027       0

Paired comparisons vs current
  Comparison            Difference          95% CI        p  Result
  --------------------  ----------  --------------  -------  -----------
  candidate vs current   -67.5 pts  [-80.9, -47.7]  <0.0001  significant

Verdict: current beats candidate by 67.5 points (95% CI 47.7 to 80.9), p<0.0001: significant.
Method: Paired design: every variant ran on the same 40 inputs. Test: exact McNemar test. Interval: Agresti-Min interval. Significance level: 0.05.
Cost: $0.00046 total for 80 LLM calls.
Wrote summary.md
Wrote prompt-report.html
Regression: candidate significantly worse than current. Failing (exit code 1).
```

## 3. A GitHub Actions workflow

Add `.github/workflows/prompts.yml` to compare the prompts on every pull
request that touches them. The Markdown report appears in the job summary
and the HTML report is attached as an artifact.

```yaml
name: Prompt regression check

on:
  pull_request:
    paths: ["prompts/**", "evals/**"]

permissions:
  contents: read

jobs:
  compare:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install "flowprompt-ai[cli]"
      - name: Compare prompts on the eval set
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: >
          flowprompt compare evals/variants.py evals/dataset.jsonl
          --model gpt-4o-mini --metric exact --control current
          --markdown "$GITHUB_STEP_SUMMARY" --html prompt-report.html
          --fail-on-regression
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: prompt-report
          path: prompt-report.html
```

To also post the report as a pull request comment, write it to a file and
use the GitHub CLI (this needs `pull-requests: write` permission):

```yaml
      - name: Comment on the pull request
        if: always()
        env:
          GH_TOKEN: ${{ github.token }}
        run: gh pr comment ${{ github.event.pull_request.number }} --body-file report.md
```

with `--markdown report.md` in the compare step.

### Choosing the dataset size

A regression gate is only as good as its power. Before wiring it in, check
which differences your dataset can detect:

```python
from flowprompt import plan_sample_size

print(plan_sample_size(0.10, discordance=0.2))
# To detect a 10-point difference at 80% power you need ~155 inputs (~310 calls)
```

A non-significant result on a small dataset means "no evidence of a
regression", not "no regression". The report prints a planning line in that
case.

## 4. pytest

The pytest plugin (`fp_compare`, `fp`, `@pytest.mark.prompt_test`) is
described in the [A/B testing guide](ab-testing.md#pytest-integration).
