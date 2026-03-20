"""Pytest plugin for FlowPrompt prompt testing.

Auto-discovered via the ``pytest11`` entry point. Provides:

- **Markers**: ``@pytest.mark.prompt_test``, ``@pytest.mark.slow_prompt``
- **Fixtures**: ``fp`` (session-scoped helper), ``fp_compare`` (function-scoped)
- **CLI option**: ``--no-slow-prompts`` to skip expensive tests
"""

from __future__ import annotations

from typing import Any

import pytest

# ------------------------------------------------------------------
# Markers & CLI option
# ------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "prompt_test: marks a prompt A/B test")
    config.addinivalue_line(
        "markers",
        "slow_prompt: marks an expensive prompt test (skip with --no-slow-prompts)",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --no-slow-prompts CLI flag."""
    parser.addoption(
        "--no-slow-prompts",
        action="store_true",
        default=False,
        help="Skip tests marked with @pytest.mark.slow_prompt",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Deselect slow_prompt tests when --no-slow-prompts is used."""
    if not config.getoption("--no-slow-prompts"):
        return

    skip_slow = pytest.mark.skip(reason="skipped via --no-slow-prompts")
    for item in items:
        if "slow_prompt" in item.keywords:
            item.add_marker(skip_slow)


# ------------------------------------------------------------------
# Helper class
# ------------------------------------------------------------------


class FlowPromptHelper:
    """Session-scoped helper providing convenient access to FlowPrompt APIs."""

    @staticmethod
    def compare(
        prompts: dict[str, type],
        inputs: list[dict[str, Any]],
        model: str = "gpt-4o",
        **kwargs: Any,
    ) -> Any:
        from flowprompt.testing.compare import compare

        return compare(prompts, inputs, model, **kwargs)

    @staticmethod
    async def acompare(
        prompts: dict[str, type],
        inputs: list[dict[str, Any]],
        model: str = "gpt-4o",
        **kwargs: Any,
    ) -> Any:
        from flowprompt.testing.compare import acompare

        return await acompare(prompts, inputs, model, **kwargs)

    @staticmethod
    def estimate_cost(
        prompts: dict[str, type],
        inputs: list[dict[str, Any]],
        model: str = "gpt-4o",
        **kwargs: Any,
    ) -> dict[str, Any]:
        from flowprompt.testing.compare import estimate_compare_cost

        return estimate_compare_cost(prompts, inputs, model, **kwargs)

    @staticmethod
    def Prompt() -> type:  # noqa: N802
        from flowprompt.core.prompt import Prompt

        return Prompt


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="session")
def fp() -> FlowPromptHelper:
    """Session-scoped FlowPrompt helper."""
    return FlowPromptHelper()


@pytest.fixture()
def fp_compare():
    """Function-scoped fixture returning a compare() wrapper.

    Returns a ``PromptTestResult`` wrapping the ``ComparisonResult``.
    """
    from flowprompt.testing.assertions import PromptTestResult
    from flowprompt.testing.compare import compare

    def _compare(
        prompts: dict[str, type],
        inputs: list[dict[str, Any]],
        model: str = "gpt-4o",
        **kwargs: Any,
    ) -> PromptTestResult:
        result = compare(prompts, inputs, model, **kwargs)
        return PromptTestResult(result)

    return _compare
