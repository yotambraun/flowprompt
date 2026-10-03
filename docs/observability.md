# Caching, tracing and cost

Every call made through `Prompt.run()` and `Prompt.arun()` can be cached,
traced and metered. All three are opt-in and cost nothing when unused.

## Cost and token usage

`track_usage()` collects one record per LLM call made inside the block:

```python
from flowprompt import Prompt, track_usage


class Summarize(Prompt):
    system = "Summarize in one sentence."
    user = "{text}"


with track_usage() as calls:
    Summarize(text="FlowPrompt measures prompts.").run(model="gpt-4o-mini")

for call in calls:
    print(
        call.model,
        call.prompt_tokens,
        call.completion_tokens,
        call.cost_usd,
        call.cached,
    )
```

- Token counts come from the provider's response (`usage`).
- Cost uses the provider-reported cost when litellm returns one, otherwise
  litellm's model price map. It is `None` when the model has no known price
  (for example a local model), never a made-up number.
- Retries are recorded as the separate, billed calls they are; cache hits
  are recorded with `cached=True` and cost 0.
- Collectors nest and are isolated per thread and per asyncio task.

`compare()` uses the same mechanism, which is how it reports cost and cost
per correct answer per variant, including for custom callables that call
FlowPrompt prompts internally.

## Caching

```python
from flowprompt import configure_cache

cache = configure_cache(enabled=True, default_ttl=3600)  # in-memory, 1 hour
```

After `configure_cache()`, identical requests are answered from the cache.
"Identical" means the same rendered messages, model, temperature, output
schema and generation parameters. Structured outputs are re-validated on a
cache hit. Streaming calls are not cached. Use
`configure_cache(backend=FileCache(".flowprompt_cache"))` for a cache that
survives restarts.

The cache is off until `configure_cache()` is called; calling `get_cache()`
only to inspect statistics does not turn it on.

> **Caching and repeated runs.** A cache returns the same answer every time.
> If you use `compare(..., runs_per_input=5)` at `temperature > 0` to measure
> run-to-run variation, disable the cache for that comparison.

## Tracing

```python
from flowprompt import configure_tracer

tracer = configure_tracer(service_name="my-app")
# ... run prompts ...
print(tracer.get_summary())
```

Once a tracer exists (`configure_tracer()` or `get_tracer()`), each call
records a span named after the prompt class, with model, tokens, cost,
latency, cache status and errors. If the `opentelemetry` packages are
installed (`pip install "flowprompt-ai[tracing]"`), spans are also exported
through the OpenTelemetry API.

## Example

Run offline with `FakeLLM` (the second call is a cache hit):

```text
gpt-4o-mini 13 4 4.35e-06 False
gpt-4o-mini 13 4 0.0 True
{'hits': 1, 'misses': 1, 'hit_rate': 0.5, 'size': 1, 'enabled': True}
{'total_requests': 2, 'total_tokens': 34, 'total_cost_usd': 4e-06, ...}
```
