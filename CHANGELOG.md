# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Redis cache backend
- Langfuse integration

## [0.3.0] - 2026-02-06

### Added
- **`compare()` convenience function**: One-call prompt A/B testing with statistical significance
  - `compare(prompts, inputs, model)` -- compare prompt variants in a single call
  - `acompare()` async variant with parallel execution via `asyncio.gather`
  - `ComparisonResult` and `VariantResult` dataclasses with `__str__()` and `to_dict()`
  - Top-level exports: `from flowprompt import compare, acompare, ComparisonResult`
- **Prompt Lab example** (`examples/10_prompt_lab.py`): End-to-end sentiment analysis comparison demo

### Fixed
- **Native JSON schema for structured outputs**: Use `litellm.supports_response_schema()` to detect models with native `json_schema` response format instead of always appending schema text to the system message. Falls back gracefully for older models.
- **Dynamic model pricing**: `UsageInfo.calculate_cost()` now looks up `litellm.model_cost` for current per-token pricing instead of using a hardcoded 9-model table. Built-in fallback prices remain for when litellm is unavailable.

### Changed
- README rewritten to lead with `compare()` as the hero feature, honest comparison table
- Quickstart docs updated with "Compare Prompts" section as second thing users learn
- `MODEL_PRICING` renamed to `_FALLBACK_PRICING` (private, backward-compatible)

## [0.2.1] - 2026-01-16

### Added
- **CLI `optimize` command**: One-command prompt optimization from terminal
  - `flowprompt optimize my_prompt.py examples.json --strategy fewshot`
  - Shows before/after accuracy comparison
  - Supports fewshot, instruction, and bootstrap strategies
- Added `a-b-testing`, `prompt-testing`, `langchain-alternative` keywords for discoverability

### Fixed
- Fixed 404 documentation URL (now points to GitHub docs)
- Fixed package name in installation docs (`flowprompt` → `flowprompt-ai`)
- Fixed silent exception swallowing in optimizer (now logs warnings)
- Added upper bounds to dependencies for security (litellm, jinja2, pyyaml)
- Removed `dev` from `[all]` extra (don't ship linters to users)
- Made basic example actually runnable with API key detection

### Changed
- README repositioned to lead with unique value proposition: "Stop guessing which prompt works. Measure it."
- A/B Testing moved to top of feature comparison tables

## [0.2.0] - 2026-01-10

### Added

#### Prompt Optimization (DSPy-style)
- **FewShotOptimizer**: Automatic few-shot example selection and optimization
- **InstructionOptimizer**: LLM-powered instruction refinement
- **OptunaOptimizer**: Hyperparameter search with Optuna integration
- **BootstrapOptimizer**: Self-improving bootstrapping optimization
- **ExampleDataset**: Structured dataset management for optimization
- **Metrics**: ExactMatch, ContainsMatch, F1Score, StructuredAccuracy, RegexMatch, CustomMetric
- **Convenience function**: Simple `optimize()` function for quick optimization

#### A/B Testing Framework
- **ExperimentConfig**: Define A/B experiments with multiple variants
- **Allocation strategies**: Random, RoundRobin, Weighted, EpsilonGreedy, UCB1, ThompsonSampling
- **Statistical tests**: Z-test, Chi-squared, T-test, Bayesian A/B testing
- **ExperimentStore**: Persist and analyze experiment results
- **Multi-armed bandit support**: Adaptive allocation based on performance

#### Multimodal Support
- **ImageContent**: Support for base64, URL, and file-based images
- **VideoContent**: Video frame extraction and processing
- **DocumentContent**: PDF, DOCX, HTML, and plain text processing
- **MultimodalPrompt**: Unified prompt class for multimodal inputs

### Fixed
- Resolved ruff linting errors across the codebase
- Added proper exception chaining with `from err`
- Fixed unused variable warnings
- Added `strict=True` to zip() calls for safety
- Improved code quality with set comprehensions

## [0.1.0] - 2026-01-10

### Added

#### Core Features
- **Prompt class with Pydantic v2 integration**: Type-safe prompt definitions with full IDE autocomplete and validation
- **Structured output validation**: Automatic parsing and validation of LLM responses via Pydantic models
- **Multi-provider support via LiteLLM**: Works with OpenAI, Anthropic, Google, Ollama, and 100+ providers
- **Async support**: Full async/await support with `arun()` method

#### Streaming
- **Sync streaming**: Real-time streaming responses with `stream()` method
- **Async streaming**: Async streaming support with `astream()` method
- **StreamChunk model**: Structured streaming chunks with delta content and metadata

#### Caching
- **Prompt caching system**: Built-in caching for 50-90% cost reduction
- **Memory cache backend**: Fast in-memory caching with TTL support
- **File cache backend**: Persistent file-based caching for cross-session reuse
- **Cache statistics**: Track hits, misses, and hit rates
- **Configurable TTL**: Time-to-live settings for cache entries

#### Observability
- **OpenTelemetry tracing**: Full distributed tracing support
- **Cost tracking**: Automatic cost calculation per request
- **Token tracking**: Input/output token counting
- **Latency metrics**: Request timing and performance monitoring
- **Usage summaries**: Aggregate statistics across all requests

#### Prompt Loading
- **YAML prompt files**: Load prompts from YAML files for team collaboration
- **JSON prompt files**: Alternative JSON format support
- **Prompt registry**: Load and manage multiple prompts from directories
- **Schema validation**: JSON Schema support for output validation in files

#### CLI Tools
- `flowprompt init`: Initialize new FlowPrompt projects with best-practice structure
- `flowprompt test`: Test all prompts in a directory
- `flowprompt run`: Run individual prompts from command line
- `flowprompt stats`: View usage and cost statistics
- `flowprompt cache-stats`: View cache performance metrics
- `flowprompt cache-clear`: Clear the prompt cache
- `flowprompt list-prompts`: List all available prompts

#### Template System
- **Python format strings**: Simple `{variable}` interpolation
- **Jinja2 templates**: Full Jinja2 support for complex logic (conditionals, loops)
- **Mixed template support**: Use both formats as needed

#### Versioning
- **Prompt versioning**: Explicit version tracking with `__version__` attribute
- **Content hashing**: Automatic content hash generation for change detection
- **Version comparison**: Track prompt evolution over time

### Technical
- Python 3.10+ support
- Pydantic v2 integration
- MIT License
- Modern tooling: ruff, mypy, pytest, pre-commit
- GitHub Actions CI/CD ready
- Comprehensive test suite
- Full type annotations

[unreleased]: https://github.com/yotambraun/flowprompt/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/yotambraun/flowprompt/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/yotambraun/flowprompt/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/yotambraun/flowprompt/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yotambraun/flowprompt/releases/tag/v0.1.0
