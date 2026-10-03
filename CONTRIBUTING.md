# Contributing to FlowPrompt

Thank you for your interest in contributing to FlowPrompt! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yotambraun/flowprompt.git
   cd flowprompt
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   uv venv
   uv sync --all-extras
   ```

3. **Activate the virtual environment**
   ```bash
   # Linux/macOS
   source .venv/bin/activate

   # Windows
   .venv\Scripts\activate
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/test_core/test_prompt.py

# Run with verbose output
uv run pytest -v

# Run every example and the README code offline (no API key needed)
uv run python scripts/run_examples.py
uv run pytest tests/test_docs_examples.py --no-cov
```

Tests never call a real LLM. Use `flowprompt.testing.FakeLLM` (or mock
`litellm.completion`) for anything that would. Python blocks in `README.md`
are executed by the test suite; mark an intentional fragment with
`<!-- not-run -->` on the line before the code fence.

### Statistics changes

Changes to `flowprompt/testing/paired.py`, `compare.py` or `statistics.py`
need known-answer tests (hand-computed p-values or intervals) and, where it
applies, a calibration check by simulation (see `tests/test_testing/test_paired.py`).
If behaviour changes, rerun the false-winner study and update
`docs/false-winners.md`:

```bash
uv run python benchmarks/false_winners.py --quick   # smoke run, ~10 s
uv run python benchmarks/false_winners.py           # full study, ~6 min
uv run python benchmarks/plot_false_winners.py      # redraw docs/assets charts
```

### Documentation

```bash
uv sync --group docs
uv run mkdocs serve             # live preview at http://127.0.0.1:8000
uv run mkdocs build --strict    # what CI runs
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Linting with ruff
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Format code (also formats Python blocks in Markdown files)
uv run ruff format .

# Type checking with mypy
uv run mypy src/flowprompt
```

### Pre-commit Hooks

We recommend using pre-commit hooks to ensure code quality before commits:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run manually on all files
uv run pre-commit run --all-files
```

## Code Style Guidelines

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Use type hints for all function signatures
- Write docstrings for public APIs (Google style)
- Keep functions focused and small
- Prefer composition over inheritance

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add streaming response support
fix: handle empty response from LLM
docs: update README with caching examples
test: add tests for YAML prompt loading
refactor: simplify template interpolation logic
```

Prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `test:` - Adding/updating tests
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write tests for new functionality
   - Update documentation if needed
   - Ensure all tests pass
   - Run linting and type checking

3. **Submit a pull request**
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure CI checks pass

4. **Code Review**
   - Address reviewer feedback
   - Keep discussions constructive

## Project Structure

```
flowprompt/
├── src/flowprompt/       # Main package source
│   ├── core/             # Core Prompt, Field, Cache, Streaming
│   ├── providers/        # LLM provider implementations
│   ├── testing/          # A/B testing, compare(), experiments
│   ├── optimize/         # DSPy-style prompt optimization
│   ├── multimodal/       # Image, audio, document support
│   ├── tracing/          # OpenTelemetry integration
│   ├── storage/          # YAML/JSON prompt loading
│   └── cli/              # Command-line interface
├── tests/                # Test suite
├── examples/             # Usage examples (run offline in CI)
├── scripts/              # Developer scripts (run_examples.py)
├── benchmarks/           # Reproducible studies (false_winners.py)
└── docs/                 # Documentation (MkDocs site, mkdocs.yml)
```

## Adding New Features

### Adding a New Provider

1. Create a new file in `src/flowprompt/providers/`
2. Inherit from `BaseProvider`
3. Implement `complete()`, `acomplete()`, `stream()`, `astream()` methods
4. Add tests in `tests/test_providers/`
5. Update documentation

### Adding a New Cache Backend

1. Create a class inheriting from `CacheBackend` in `src/flowprompt/core/cache.py`
2. Implement `get()`, `set()`, `delete()`, `clear()` methods
3. Add tests
4. Update documentation and examples

## Reporting Issues

When reporting issues, please include:

- FlowPrompt version (`flowprompt --version`)
- Python version
- Operating system
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback (if applicable)

## Feature Requests

We welcome feature requests! Please:

- Check existing issues to avoid duplicates
- Describe the use case clearly
- Explain why this would benefit other users

## Questions?

- Open a GitHub issue for bugs or features
- Start a GitHub Discussion for questions
- Check existing documentation first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to FlowPrompt!
