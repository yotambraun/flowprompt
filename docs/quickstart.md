# Quick Start

This guide will get you up and running with FlowPrompt in under 5 minutes.

## Your First Prompt

Create a simple prompt that extracts information from text:

```python
from flowprompt import Prompt
from pydantic import BaseModel

class ExtractUser(Prompt):
    """Extract user information from text."""

    system = "You are a precise data extractor."
    user = "Extract the name and age from: {text}"

    class Output(BaseModel):
        name: str
        age: int

# Run the prompt
result = ExtractUser(text="John Smith is 25 years old").run(model="gpt-4o")
print(f"Name: {result.name}")  # Name: John Smith
print(f"Age: {result.age}")    # Age: 25
```

## Compare Prompts

The fastest way to find which prompt works better -- with statistical significance:

```python
from flowprompt import Prompt, compare

class PromptV1(Prompt):
    system = "You are a precise data extractor."
    user = "Extract the name and age from: {text}"

class PromptV2(Prompt):
    system = "Extract structured data. Be accurate and concise."
    user = "From the following text, extract name and age: {text}"

result = compare(
    {"v1": PromptV1, "v2": PromptV2},
    inputs=[
        {"text": "John Smith is 25 years old"},
        {"text": "Alice (age 30) joined today"},
        {"text": "Bob, 42, from NYC"},
    ],
    expected=["John Smith, 25", "Alice, 30", "Bob, 42"],  # ground-truth answers
    model="gpt-4o-mini",
)
print(result)
# Shows winner, accuracy rates, latency, and statistical significance
```

Use `dry_run=True` to preview estimated cost before spending API credits:

```python
result = compare(
    {"v1": PromptV1, "v2": PromptV2},
    inputs=test_data,
    model="gpt-4o-mini",
    dry_run=True,  # No API calls -- just estimates cost
)
print(result)  # Shows estimated cost and number of API calls
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

> **Tip:** FlowPrompt includes a pytest plugin for running prompt tests in CI.
> Install `flowprompt-ai[pytest]` and use the `fp_compare` fixture. See the
> [A/B Testing Guide](ab-testing.md#pytest-integration) for details.

## Streaming Responses

For real-time output, use streaming:

```python
for chunk in ExtractUser(text="John is 25").stream(model="gpt-4o"):
    print(chunk.delta, end="", flush=True)
```

Async streaming:

```python
async for chunk in ExtractUser(text="John is 25").astream(model="gpt-4o"):
    print(chunk.delta, end="", flush=True)
```

## Using Different Providers

Switch providers by changing the model string:

```python
# OpenAI
result = prompt.run(model="gpt-4o")

# Anthropic Claude
result = prompt.run(model="anthropic/claude-3-5-sonnet-20241022")

# Google Gemini
result = prompt.run(model="gemini/gemini-2.0-flash-exp")

# Local Ollama
result = prompt.run(model="ollama/llama3")
```

## Enable Caching

Reduce costs by caching identical requests:

```python
from flowprompt import configure_cache

# Enable caching with 1-hour TTL
configure_cache(enabled=True, default_ttl=3600)

# First call hits the API
result1 = ExtractUser(text="John is 25").run(model="gpt-4o")

# Second call returns cached result
result2 = ExtractUser(text="John is 25").run(model="gpt-4o")  # Instant!
```

## Track Usage and Costs

Monitor your API usage:

```python
from flowprompt import get_tracer

# Run some prompts
result = ExtractUser(text="John is 25").run(model="gpt-4o")

# View statistics
summary = get_tracer().get_summary()
print(f"Total cost: ${summary['total_cost_usd']:.4f}")
print(f"Total tokens: {summary['total_tokens']}")
```

## Load Prompts from Files

Create `prompts/extract_user.yaml`:

```yaml
name: ExtractUser
version: "1.0.0"
system: You are a precise data extractor.
user: "Extract from: {{ text }}"
output_schema:
  type: object
  properties:
    name:
      type: string
    age:
      type: integer
  required: [name, age]
```

Load and use it:

```python
from flowprompt import load_prompt

ExtractUser = load_prompt("prompts/extract_user.yaml")
result = ExtractUser(text="John is 25").run(model="gpt-4o")
```

## Using the CLI

Initialize a new project:

```bash
flowprompt init my-project
cd my-project
```

Run a prompt:

```bash
flowprompt run prompts/extract_user.yaml --var text="John is 25"
```

View statistics:

```bash
flowprompt stats
```

## Next Steps

- Read the [API Reference](api.md) for detailed documentation
- Check out the [examples](https://github.com/yotambraun/flowprompt/tree/main/examples) directory
- Join our [GitHub Discussions](https://github.com/yotambraun/flowprompt/discussions)
