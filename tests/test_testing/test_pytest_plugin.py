"""Tests for the pytest plugin."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from flowprompt import Prompt
from flowprompt.testing.assertions import PromptTestResult
from flowprompt.testing.pytest_plugin import FlowPromptHelper

# ---------------------------------------------------------------------------
# Test prompt classes
# ---------------------------------------------------------------------------


class PromptA(Prompt[Any]):
    system: str = "You are helpful."
    user: str = "Process: {text}"


class PromptB(Prompt[Any]):
    system: str = "You are concise."
    user: str = "Summarize: {text}"


# ---------------------------------------------------------------------------
# FlowPromptHelper
# ---------------------------------------------------------------------------


class TestFlowPromptHelper:
    def test_compare(self) -> None:
        helper = FlowPromptHelper()
        with patch.object(Prompt, "run", return_value="ok"):
            result = helper.compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                model="gpt-4o-mini",
            )
        assert result.total_runs == 2

    def test_estimate_cost(self) -> None:
        helper = FlowPromptHelper()
        cost = helper.estimate_cost(
            {"a": PromptA, "b": PromptB},
            inputs=[{"text": "hi"}],
            model="gpt-4o-mini",
        )
        assert cost["total_calls"] == 2

    def test_prompt_returns_class(self) -> None:
        helper = FlowPromptHelper()
        assert helper.Prompt() is Prompt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TestFixtures:
    def test_fp_fixture(self, fp: FlowPromptHelper) -> None:
        assert isinstance(fp, FlowPromptHelper)

    def test_fp_compare_fixture(self, fp_compare) -> None:
        with patch.object(Prompt, "run", return_value="ok"):
            result = fp_compare(
                {"a": PromptA, "b": PromptB},
                inputs=[{"text": "hi"}],
                model="gpt-4o-mini",
            )
        assert isinstance(result, PromptTestResult)
        assert result.total_runs == 2


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


@pytest.mark.prompt_test
class TestPromptTestMarker:
    def test_marker_exists(self) -> None:
        """This test simply verifies the marker is registered."""
        pass


@pytest.mark.slow_prompt
class TestSlowPromptMarker:
    def test_marker_exists(self) -> None:
        """This test verifies slow_prompt marker is registered."""
        pass
