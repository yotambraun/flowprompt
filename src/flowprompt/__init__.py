"""FlowPrompt: Type-safe prompt management with automatic optimization for LLMs.

FlowPrompt combines Pydantic's type safety with powerful features:
- Streaming responses for real-time output
- Prompt caching for cost reduction
- OpenTelemetry tracing for observability
- YAML/JSON prompt loading for collaboration
- Multi-provider support via LiteLLM
- **NEW in v0.2.0**: Automatic optimization (DSPy-style)
- **NEW in v0.2.0**: A/B testing framework
- **NEW in v0.2.0**: Multimodal support (images, audio, documents)

Example:
    >>> from flowprompt import Prompt, Field
    >>> from pydantic import BaseModel
    >>>
    >>> class ExtractUser(Prompt):
    ...     system = "You are a precise data extractor."
    ...     user = "Extract from: {text}"
    ...
    ...     class Output(BaseModel):
    ...         name: str
    ...         age: int
    >>>
    >>> result = ExtractUser(text="John is 25").run(model="gpt-4o")
    >>> print(result.name)  # "John"

Streaming:
    >>> for chunk in ExtractUser(text="John is 25").stream(model="gpt-4o"):
    ...     print(chunk.delta, end="", flush=True)

Caching:
    >>> from flowprompt import configure_cache
    >>> configure_cache(enabled=True, default_ttl=3600)

Tracing:
    >>> from flowprompt import get_tracer
    >>> print(get_tracer().get_summary())

YAML Loading:
    >>> from flowprompt import load_prompt, load_prompts
    >>> MyPrompt = load_prompt("prompts/my_prompt.yaml")

Optimization (NEW in v0.2.0):
    >>> from flowprompt.optimize import optimize, ExampleDataset, Example, ExactMatch
    >>> dataset = ExampleDataset([Example(input={"text": "John is 25"}, output="John, 25")])
    >>> result = optimize(MyPrompt, dataset=dataset, metric=ExactMatch())

Prompt Comparison (NEW in v0.3.0):
    >>> from flowprompt import compare
    >>> result = compare({"v1": PromptV1, "v2": PromptV2}, inputs=[{"text": "hi"}])
    >>> print(result)

A/B Testing:
    >>> from flowprompt.testing import ABTestRunner, create_simple_experiment
    >>> config, runner = create_simple_experiment("test", PromptV1, [("v2", PromptV2)])
    >>> runner.start_experiment(config.id)

Multimodal (NEW in v0.2.0):
    >>> from flowprompt.multimodal import VisionPrompt
    >>> result = VisionPrompt.describe("image.jpg").run(model="gpt-4o")
"""

__version__ = "0.3.0"

# Core
# Caching
from flowprompt.core.cache import (
    CacheBackend,
    CacheEntry,
    FileCache,
    MemoryCache,
    PromptCache,
    configure_cache,
    get_cache,
)
from flowprompt.core.field import Field
from flowprompt.core.prompt import Prompt

# Streaming
from flowprompt.core.streaming import (
    AsyncStreamingResponse,
    StreamChunk,
    StreamingResponse,
)

# Storage / YAML Loading
from flowprompt.storage.yaml_loader import (
    PromptConfig,
    PromptRegistry,
    load_prompt,
    load_prompts,
)

# Testing (convenience API)
from flowprompt.testing.assertions import PromptTestResult
from flowprompt.testing.compare import (
    ComparisonResult,
    acompare,
    compare,
    estimate_compare_cost,
)
from flowprompt.testing.eval_metrics import (
    contains_match,
    exact_match,
    similarity_match,
)

# Tracing
from flowprompt.tracing.otel import (
    SpanContext,
    Tracer,
    UsageInfo,
    configure_tracer,
    get_tracer,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "Prompt",
    "Field",
    # Streaming
    "StreamChunk",
    "StreamingResponse",
    "AsyncStreamingResponse",
    # Caching
    "PromptCache",
    "CacheEntry",
    "CacheBackend",
    "MemoryCache",
    "FileCache",
    "get_cache",
    "configure_cache",
    # Tracing
    "Tracer",
    "SpanContext",
    "UsageInfo",
    "get_tracer",
    "configure_tracer",
    # Testing (convenience API)
    "compare",
    "acompare",
    "estimate_compare_cost",
    "ComparisonResult",
    "PromptTestResult",
    "exact_match",
    "contains_match",
    "similarity_match",
    # Storage
    "PromptConfig",
    "PromptRegistry",
    "load_prompt",
    "load_prompts",
]

# Note: The following modules are available but not imported at the top level
# to keep imports fast. Import them explicitly when needed:
#
# - flowprompt.optimize: Automatic prompt optimization
# - flowprompt.testing: A/B testing framework
# - flowprompt.multimodal: Multimodal support (images, audio, documents)
