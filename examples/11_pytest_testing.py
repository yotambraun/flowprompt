"""Pytest Testing: Write prompt tests that run in CI.

This example shows how to use FlowPrompt's pytest fixtures to
write repeatable prompt evaluation tests.

Requirements:
    pip install flowprompt-ai[pytest]
    pip install pytest

Usage:
    pytest examples/11_pytest_testing.py -v

    # Skip slow/expensive tests:
    pytest examples/11_pytest_testing.py -v --no-slow-prompts

Note:
    These tests mock LLM calls so they run without an API key.
    In real usage, you'd remove the mocks and run against a real model.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from flowprompt import Prompt

# =============================================================================
# Step 1: Define prompt variants
# =============================================================================


class SentimentBasic(Prompt):
    """Minimal sentiment classifier."""

    system = "Classify sentiment as positive, negative, or neutral."
    user = "{text}"


class SentimentDetailed(Prompt):
    """Detailed sentiment classifier with instructions."""

    system = (
        "You are a sentiment analysis expert. "
        "Classify the sentiment as exactly one of: positive, negative, neutral. "
        "Reply with a single word."
    )
    user = "Text: {text}"


# =============================================================================
# Step 2: Write tests using fp_compare fixture
# =============================================================================


@pytest.mark.prompt_test
class TestSentimentPrompts:
    """Example prompt tests using the fp_compare fixture."""

    def test_compare_with_expected_outputs(self, fp_compare) -> None:
        """Compare prompts against known correct answers."""
        # Mock LLM to return predictable outputs for demonstration
        with patch.object(Prompt, "run", return_value="positive"):
            result = fp_compare(
                {"basic": SentimentBasic, "detailed": SentimentDetailed},
                inputs=[
                    {"text": "I love this product!"},
                    {"text": "Best purchase ever!"},
                ],
                expected=["positive", "positive"],
                eval_metric="exact",
                model="gpt-4o-mini",
            )

        # Both variants should get 100% accuracy on these inputs
        result.assert_no_errors()
        assert result.variants["basic"].success_rate == 1.0

    def test_fp_helper_estimate_cost(self, fp) -> None:
        """Use the fp fixture to estimate cost before running."""
        cost = fp.estimate_cost(
            {"basic": SentimentBasic, "detailed": SentimentDetailed},
            inputs=[{"text": "test"}],
            model="gpt-4o-mini",
        )
        assert cost["total_calls"] == 2


@pytest.mark.slow_prompt
class TestExpensivePrompts:
    """These tests are skipped with --no-slow-prompts."""

    def test_large_comparison(self, fp_compare) -> None:
        """Would normally run many inputs -- mocked here for demo."""
        inputs = [{"text": f"Input {i}"} for i in range(20)]
        expected = ["positive"] * 20

        with patch.object(Prompt, "run", return_value="positive"):
            result = fp_compare(
                {"basic": SentimentBasic, "detailed": SentimentDetailed},
                inputs=inputs,
                expected=expected,
                eval_metric="exact",
                model="gpt-4o-mini",
            )

        result.assert_no_errors()
