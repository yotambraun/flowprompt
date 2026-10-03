# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Always-valid (sequential) tests for live-traffic experiments
- Redis cache backend
- Langfuse integration

## [0.5.0] - Unreleased

This release makes `compare()` statistically sound and fixes the crash that
broke the README quickstart and every example. **If you
used `compare()` before, re-check conclusions drawn from it**: the old
analysis could report significant differences that were not there (see
Fixed), and results now change accordingly.

### Fixed
- **Prompts defined as in the README crashed.** `class MyPrompt(Prompt):
  system = "..."` (no type annotation) raised `PydanticUserError: Field
  'system' defined on a base class was overridden by a non-annotated
  attribute` on every Pydantic 2 release we tested (2.0.3 through 2.13.5),
  so the quickstart and all examples failed; the test suite only used
  annotated fields and did not catch it. Unannotated overrides of prompt fields now work on Pydantic 2.0
  through 2.13 and Python 3.10 through 3.14; annotated definitions are
  unchanged.
- **`compare()` p-values ignored that variants share inputs.** Every variant
  runs on the same inputs, but results were tested with an unpaired pooled
  z-test. Pass/fail results now use the exact McNemar test. Example: 8/10
  versus 4/10 correct, with the four disagreements all favouring the first
  prompt, gives p = 0.125 (previously reported as 0.068).
- **Repeated runs were counted as extra samples.** With `runs_per_input=5`
  and deterministic outputs, the p-value for the example above dropped from
  0.068 to below 0.0001 without any new information. Runs are now averaged
  per input; the number of inputs is the sample size.
- **Wrong winner when the control scored 0.** The winner was chosen by the
  relative lift, which is undefined (and was set to 0) when the control
  scores 0, so a significantly better treatment lost. Winners now follow
  the sign of the absolute difference. The same bug in `ABTestRunner`
  (summary and auto-completion) is fixed too.
- **No correction for multiple comparisons.** With three or more variants,
  p-values are now Holm-adjusted before a winner is named, in `compare()`
  and in `ABTestRunner`.
- **Comparisons without expected outputs** reported a failed test ("Zero
  standard error"). They now say plainly that outputs were not graded and
  compare error rates only (`has_ground_truth=False`).
- **Caching, tracing and cost tracking were not connected to `run()`.** Two
  identical calls made two LLM calls, the tracer recorded nothing and
  `compare()` reported $0.00. `configure_cache()` now serves identical
  requests from the cache, a configured tracer records every call, and
  token usage and cost are taken from the provider response.
- **YAML prompts with an `output_schema`** failed with
  `PydanticSchemaGenerationError`. Output models are now built with
  `pydantic.create_model` (enums, nested objects, typed arrays, required
  fields).

### Changed
- **`compare()` uses paired tests by default** (`test_type="auto"`): the
  exact McNemar test for pass/fail with one run per input, otherwise a
  paired sign-flip permutation test with a bootstrap confidence interval.
  Expect larger, honest p-values on small evaluation sets; with fewer than
  six inputs no difference can be significant, and the result says so.
- `StatisticalResult.effect_size` from `compare()` is the absolute
  difference (0.12 = 12 accuracy points); the relative lift moved to
  `relative_lift` (None when undefined). The unpaired tests keep their old
  meaning.
- `test_type="z_test"`, `"chi_squared"`, `"t_test"` and `"bayesian"` still
  work with `compare()` but emit a `DeprecationWarning`.
- `print(result)` shows a report: accuracy with confidence intervals,
  latency (mean and p95), cost and cost per correct answer, each paired
  comparison with its interval and adjusted p-value, and a one-line verdict.
- `PromptTestResult.assert_significant()` and `.p_value` use the
  Holm-adjusted p-value when several variants were compared.
- The cache is only used after `configure_cache()`; `get_cache()` alone
  does not turn it on.

### Added
- Variants can be any callable `fn(input) -> output` (sync or async),
  `(PromptClass, "model")` tuples, `PromptVariant` or
  `model_variants(Prompt, [...])`, so models and pipelines can be compared,
  not only prompts.
- Scorers: `exact`, `contains`, `regex`, `numeric` (with tolerances),
  `similarity`, or any function returning a bool or a score
  (`flowprompt.testing.scorers`).
- `ComparisonResult`: `comparisons`, `verdict`, `enough_data`, `notes`,
  `to_markdown()`, `to_html()` (self-contained, light and dark),
  `save_report()`, `sample_size_plan()`; `StatisticalResult` gains
  `method`, `difference`, `ci_low`, `ci_high`, `n_inputs`, `adjusted_p`,
  `relative_lift`, `control`, `treatment`.
- `plan_sample_size()`: inputs needed for a given difference and power
  (Connor 1987 for paired pass/fail outcomes), with a cost estimate when a
  price is known.
- `SequentialMcNemar`: an always-valid paired test for stopping early
  without inflating false positives.
- Statistics helpers: `paired_test`, `mcnemar_exact` (with mid-p),
  `paired_sign_flip_test`, `paired_bootstrap_interval`,
  `agresti_min_interval`, `wilson_interval`, `holm_adjust`.
- `track_usage()` and `CallUsage` for per-call tokens, cost, latency and
  cache hits.
- `FakeLLM`: answers LLM calls offline and deterministically for tests,
  CI and demos.
- `flowprompt compare VARIANTS.py DATASET.jsonl` with Markdown, HTML and
  JSON reports and `--fail-on-regression` for CI.
- `compare(control=..., comparisons="all")` to choose the baseline or test
  every pair.
- Documentation site (MkDocs Material), pages on CI, statistical methods
  and observability, and a simulation study:
  [Your prompt A/B test is probably lying to you](docs/false-winners.md)
  (`benchmarks/false_winners.py`).
- CI runs every example and the README offline, and tests prompt
  definitions on Pydantic 2.0, 2.9 and the latest 2.x.

### Removed
- README claims that could not be verified: "<100ms import" (importing
  `flowprompt` takes about 0.15 s; litellm, about 2 s, loads on the first
  call), "the only Python LLM framework with built-in A/B testing", and the
  unverified comparison table.

## [0.4.0] - 2026-03-20

> **Not published to PyPI.** This version was tagged on GitHub, but the
> package version in the source was still 0.3.0, so the PyPI upload could
> not succeed. Everything listed here ships in 0.5.0.

### Added
- **`expected` parameter for `compare()` and `acompare()`**: Ground-truth evaluation for prompt outputs
  - Pass `expected=[...]` alongside `inputs` to measure accuracy instead of just success
  - Output displays "accuracy" instead of "success" when expected values are provided
- **Built-in eval metrics**: `exact_match`, `contains_match`, `similarity_match` (stdlib only, no new deps)
  - `eval_metric` parameter accepts string names or custom `(output, expected) -> bool` callables
  - `resolve_eval_metric()` for programmatic metric resolution
- **Pytest integration** (auto-discovered via `pytest11` entry point, zero config):
  - `fp` fixture (session-scoped): `FlowPromptHelper` with `.compare`, `.acompare`, `.estimate_cost`, `.Prompt`
  - `fp_compare` fixture (function-scoped): wraps `compare()`, returns `PromptTestResult`
  - `@pytest.mark.prompt_test` and `@pytest.mark.slow_prompt` markers
  - `--no-slow-prompts` CLI option to skip expensive tests
- **`PromptTestResult`** assertion wrapper:
  - `.assert_significant(threshold=0.05)` -- fail unless statistically significant
  - `.assert_winner(expected)` -- fail unless the named variant won
  - `.assert_no_errors()` -- fail if any variant recorded errors
  - `.is_significant`, `.winner`, `.p_value` convenience properties
- New `pytest` optional dependency: `pip install flowprompt-ai[pytest]`
- New example: `examples/11_pytest_testing.py`

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

[unreleased]: https://github.com/yotambraun/flowprompt/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/yotambraun/flowprompt/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/yotambraun/flowprompt/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/yotambraun/flowprompt/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/yotambraun/flowprompt/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yotambraun/flowprompt/releases/tag/v0.1.0
