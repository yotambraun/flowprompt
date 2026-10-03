"""The live-traffic runner must pick the winner by the sign of the difference."""

from __future__ import annotations

from flowprompt.core.prompt import Prompt
from flowprompt.testing.experiment import ExperimentStatus
from flowprompt.testing.runner import create_simple_experiment


class Control(Prompt):
    system = "control"
    user = "{text}"


class Treatment(Prompt):
    system = "treatment"
    user = "{text}"


def _record(runner, exp_id: str, name: str, successes: int, failures: int) -> None:
    for i in range(successes + failures):
        runner.record_result(exp_id, name, output="x", success=i < successes)


def test_summary_winner_when_control_rate_is_zero() -> None:
    config, runner = create_simple_experiment(
        "zero-control", Control, [("treatment", Treatment)], min_samples=10_000
    )
    runner.start_experiment(config.id)
    _record(runner, config.id, "control", successes=0, failures=30)
    _record(runner, config.id, "treatment", successes=18, failures=12)
    summary = runner.get_summary(config.id)
    assert summary.statistical_result is not None
    assert summary.statistical_result.significant
    assert summary.winner is not None
    assert summary.winner.name == "treatment"


def test_auto_completion_names_the_better_variant() -> None:
    config, runner = create_simple_experiment(
        "zero-control-auto", Control, [("treatment", Treatment)], min_samples=60
    )
    runner.start_experiment(config.id)
    _record(runner, config.id, "control", successes=0, failures=30)
    _record(runner, config.id, "treatment", successes=18, failures=12)
    exp = runner._store.get_experiment(config.id)
    assert exp.status == ExperimentStatus.COMPLETED
    assert exp.metadata.get("winner") == "treatment"
