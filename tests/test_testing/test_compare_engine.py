"""compare() as a general experiment engine: variants, scorers, edge cases."""

from __future__ import annotations

import asyncio
import json
import warnings
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from flowprompt import Prompt, compare
from flowprompt.testing import FakeLLM, PromptVariant, acompare, model_variants, scorers

INPUTS = [{"q": i} for i in range(12)]
ANSWERS = [str(i * 2) for i in range(12)]


def doubler(inp: dict[str, Any]) -> str:
    return str(inp["q"] * 2)


def off_by_one_on_odd(inp: dict[str, Any]) -> str:
    q = inp["q"]
    return str(q * 2 + (q % 2))


class Echo(Prompt[Any]):
    system: str = "Answer with the number only."
    user: str = "What is 2 * {q}?"


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


def test_plain_callables_are_variants() -> None:
    result = compare(
        {"good": doubler, "sloppy": off_by_one_on_odd},
        inputs=INPUTS,
        expected=ANSWERS,
        eval_metric="exact",
    )
    assert result.variants["good"].mean_score == pytest.approx(1.0)
    assert result.variants["sloppy"].mean_score == pytest.approx(0.5)
    # 6 discordant inputs, all favour the control: p = 2/64.
    assert result.statistical_result.p_value == pytest.approx(0.03125)
    assert result.winner == "good"
    assert result.variants["good"].label == "doubler"


def test_async_callables_in_compare_and_acompare() -> None:
    async def async_doubler(inp: dict[str, Any]) -> str:
        await asyncio.sleep(0)
        return str(inp["q"] * 2)

    sync_result = compare(
        {"async": async_doubler, "sloppy": off_by_one_on_odd},
        inputs=INPUTS,
        expected=ANSWERS,
        eval_metric="exact",
    )
    async_result = asyncio.run(
        acompare(
            {"async": async_doubler, "sloppy": off_by_one_on_odd},
            inputs=INPUTS,
            expected=ANSWERS,
            eval_metric="exact",
        )
    )
    for r in (sync_result, async_result):
        assert r.variants["async"].mean_score == pytest.approx(1.0)
        assert r.winner == "async"


def test_same_prompt_on_different_models() -> None:
    def responder(_messages: list[dict[str, Any]]) -> str:
        return "42"

    with FakeLLM(responder) as fake:
        result = compare(
            model_variants(Echo, ["gpt-4o-mini", "gpt-4o"]),
            inputs=INPUTS[:3],
            expected=["42"] * 3,
        )
    models = sorted({c["model"] for c in fake.calls})
    assert models == ["gpt-4o", "gpt-4o-mini"]
    assert set(result.variants) == {"gpt-4o-mini", "gpt-4o"}
    assert result.variants["gpt-4o"].label == "Echo @ gpt-4o"


def test_tuple_and_prompt_variant_shorthands() -> None:
    with FakeLLM("ok") as fake:
        compare(
            {
                "default": Echo,
                "tuple": (Echo, "gpt-4o-mini"),
                "explicit": PromptVariant(
                    Echo, model="claude-3-5-haiku", temperature=0.3
                ),
            },
            inputs=INPUTS[:2],
            model="gpt-4o",
        )
    seen = {(c["model"], c["temperature"]) for c in fake.calls}
    assert seen == {("gpt-4o", 0.0), ("gpt-4o-mini", 0.0), ("claude-3-5-haiku", 0.3)}


def test_invalid_variant_raises() -> None:
    with pytest.raises(TypeError, match="Cannot use"):
        compare({"a": doubler, "b": 42}, inputs=INPUTS)


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scorer", "output", "expected", "ok"),
    [
        (scorers.exact(), " Positive ", "positive", True),
        (scorers.exact(case_sensitive=True), "Positive", "positive", False),
        (scorers.contains(), "The answer is 42.", "42", True),
        (scorers.regex(), "Order #A-1234 shipped", r"#A-\d{4}", True),
        (scorers.regex(full_match=True), "A-1234 shipped", r"A-\d{4}", False),
        (scorers.numeric(abs_tol=0.01), "Total: 3.141", "3.14", True),
        (scorers.numeric(abs_tol=0.001), "Total: 3.2", 3.14, False),
        (scorers.numeric(), "no number here", 1, False),
        (scorers.numeric(rel_tol=0.05), "1,000", 1040, True),
        (scorers.similarity(0.8), "hello world", "hello wrld", True),
    ],
)
def test_builtin_scorers(scorer: Any, output: Any, expected: Any, ok: bool) -> None:
    assert scorer(output, expected) is ok


def test_exact_scorer_with_structured_output() -> None:
    class Out(BaseModel):
        name: str

    assert scorers.exact()(Out(name="Ada"), {"name": "Ada"}) is True
    assert scorers.exact()(Out(name="Ada"), {"name": "Bob"}) is False


def test_scorer_names_resolve() -> None:
    for name in ("exact", "contains", "regex", "numeric", "similarity"):
        assert callable(scorers.resolve_scorer(name))
    with pytest.raises(ValueError, match="Unknown eval metric"):
        scorers.resolve_scorer("nope")


def test_numeric_scores_use_paired_permutation() -> None:
    def graded(output: Any, expected: Any) -> float:
        return 1.0 - min(1.0, abs(int(output) - int(expected)) / 10)

    result = compare(
        {"good": doubler, "sloppy": off_by_one_on_odd},
        inputs=INPUTS,
        expected=ANSWERS,
        eval_metric=graded,
    )
    sr = result.statistical_result
    assert sr.method == "paired_permutation"
    assert sr.difference == pytest.approx(-0.05)
    assert result.outcome == "accuracy"


def test_metric_fn_without_expected() -> None:
    result = compare(
        {"long": lambda inp: "x" * (inp["q"] + 5), "short": lambda inp: "x" * inp["q"]},
        inputs=INPUTS,
        metric_fn=len,
    )
    assert result.outcome == "score"
    sr = result.statistical_result
    assert sr.method == "paired_permutation"
    assert sr.difference == pytest.approx(-5.0)
    assert sr.significant
    assert result.winner == "long"
    assert "pts" not in str(result)  # numeric scores are not percentages


def test_broken_scorer_counts_as_failure_not_crash() -> None:
    def explode(_output: Any, _expected: Any) -> bool:
        raise RuntimeError("bad scorer")

    result = compare(
        {"a": doubler, "b": doubler},
        inputs=INPUTS[:3],
        expected=ANSWERS[:3],
        eval_metric=explode,
    )
    assert result.variants["a"].mean_score == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_errors_are_counted_and_scored_as_failures() -> None:
    def flaky(inp: dict[str, Any]) -> str:
        if inp["q"] % 3 == 0:
            raise TimeoutError("timeout")
        return doubler(inp)

    result = compare(
        {"stable": doubler, "flaky": flaky},
        inputs=INPUTS,
        expected=ANSWERS,
        eval_metric="exact",
    )
    flaky_res = result.variants["flaky"]
    assert flaky_res.error_count == 4
    assert flaky_res.errors == ["timeout"] * 4
    assert flaky_res.mean_score == pytest.approx(8 / 12)
    assert any("4 of 12 runs raised an error" in n for n in result.notes)


def test_all_outputs_identical_gives_no_difference() -> None:
    result = compare(
        {"a": doubler, "b": doubler},
        inputs=INPUTS,
        expected=ANSWERS,
        eval_metric="exact",
    )
    sr = result.statistical_result
    assert sr.p_value == 1.0
    assert sr.difference == 0.0
    assert result.winner is None
    assert "No difference" in result.verdict


def test_tiny_sample_reports_not_enough_data() -> None:
    result = compare(
        {"good": doubler, "sloppy": lambda _inp: "wrong"},
        inputs=INPUTS[:4],
        expected=ANSWERS[:4],
        eval_metric="exact",
    )
    # Even 4/4 disagreements give p = 2/16 = 0.125.
    assert not result.enough_data
    assert result.winner is None
    assert result.verdict.startswith("Not enough data")
    assert result.min_inputs_for_significance == 6


def test_all_pairs_mode_with_holm() -> None:
    def always_wrong(_inp: dict[str, Any]) -> str:
        return "?"

    inputs = [{"q": i} for i in range(20)]
    answers = [str(i * 2) for i in range(20)]
    result = compare(
        {"good": doubler, "sloppy": off_by_one_on_odd, "bad": always_wrong},
        inputs=inputs,
        expected=answers,
        eval_metric="exact",
        comparisons="all",
    )
    assert len(result.comparisons) == 3
    assert result.correction == "holm"
    assert all(c.adjusted_p is not None for c in result.comparisons)
    assert result.winner == "good"


def test_explicit_control() -> None:
    result = compare(
        {"sloppy": off_by_one_on_odd, "good": doubler},
        inputs=INPUTS,
        expected=ANSWERS,
        eval_metric="exact",
        control="good",
    )
    assert result.control == "good"
    assert result.comparisons[0].control == "good"
    assert result.winner == "good"


def test_bad_options_raise() -> None:
    with pytest.raises(ValueError, match="control"):
        compare({"a": doubler, "b": doubler}, inputs=INPUTS, control="zzz")
    with pytest.raises(ValueError, match="comparisons"):
        compare({"a": doubler, "b": doubler}, inputs=INPUTS, comparisons="some")
    with pytest.raises(ValueError, match="runs_per_input"):
        compare({"a": doubler, "b": doubler}, inputs=INPUTS, runs_per_input=0)
    with pytest.raises(ValueError, match="Unknown test type"):
        compare({"a": doubler, "b": doubler}, inputs=INPUTS, test_type="magic")


def test_legacy_test_type_still_works_with_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compare(
            {"good": doubler, "sloppy": off_by_one_on_odd},
            inputs=INPUTS,
            expected=ANSWERS,
            eval_metric="exact",
            test_type="z_test",
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert result.statistical_result.test_name == "two_proportion_z_test"
    assert result.statistical_result.difference == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# Cost, latency, planning and reports
# ---------------------------------------------------------------------------


def _fake_answer(messages: list[dict[str, Any]]) -> str:
    q = int(messages[-1]["content"].split()[-1].rstrip("?"))
    return str(q * 2)


def test_cost_latency_and_cost_per_correct_with_fake_llm() -> None:
    with FakeLLM(_fake_answer):
        result = compare(
            {"a": Echo, "b": (Echo, "gpt-4o")},
            inputs=INPUTS,
            expected=ANSWERS,
            eval_metric="exact",
            model="gpt-4o-mini",
        )
    a, b = result.variants["a"], result.variants["b"]
    assert a.cost_known and b.cost_known
    assert 0 < a.total_cost_usd < b.total_cost_usd  # gpt-4o costs more
    assert a.cost_per_correct == pytest.approx(a.total_cost_usd / 12)
    assert a.llm_calls == 12
    assert a.total_tokens > 0
    assert a.p95_latency_ms >= 0
    assert result.cost_known
    assert result.total_cost_usd == pytest.approx(a.total_cost_usd + b.total_cost_usd)


def test_unknown_price_is_reported_as_unknown() -> None:
    with FakeLLM("0"):
        result = compare(
            {"a": Echo, "b": Echo}, inputs=INPUTS[:2], model="my-local-model-xyz"
        )
    assert not result.cost_known
    assert "Cost: unknown" in str(result)


def test_sample_size_plan_from_result() -> None:
    result = compare(
        {"good": doubler, "sloppy": off_by_one_on_odd},
        inputs=INPUTS,
        expected=ANSWERS,
        eval_metric="exact",
    )
    plan = result.sample_size_plan(0.1)
    assert plan is not None
    # observed discordance 0.5 -> Connor formula
    assert plan.details["discordance"] == pytest.approx(0.5)
    assert plan.n_inputs > 0


def test_planning_line_shown_when_not_significant() -> None:
    def sometimes(inp: dict[str, Any]) -> str:
        return doubler(inp) if inp["q"] != 3 else "?"

    result = compare(
        {"a": doubler, "b": sometimes},
        inputs=INPUTS,
        expected=ANSWERS,
        eval_metric="exact",
    )
    assert result.winner is None
    assert "To detect a 10-point difference at 80% power" in str(result)


def test_reports_render_from_the_same_result(tmp_path: Path) -> None:
    with FakeLLM(_fake_answer):
        result = compare(
            {"a": Echo, "b": off_by_one_on_odd},
            inputs=INPUTS,
            expected=ANSWERS,
            eval_metric="exact",
            model="gpt-4o-mini",
        )
    text = str(result)
    md = result.to_markdown()
    page = result.to_html()
    for rendered in (text, md, page):
        assert "exact McNemar" in rendered
    assert text.isascii()
    assert md.startswith("### Prompt comparison:")
    assert "| Variant |" in md
    assert page.startswith("<!doctype html>")
    assert "<script" not in page and "http" not in page.split("<body>")[1]
    for suffix in (".html", ".md", ".txt", ".json"):
        path = result.save_report(tmp_path / f"report{suffix}")
        assert path.read_text(encoding="utf-8")
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["winner"] == result.winner
    assert data["comparisons"][0]["method"] == "mcnemar_exact"


def test_repeated_runs_note_and_unit_of_analysis() -> None:
    result = compare(
        {"good": doubler, "sloppy": off_by_one_on_odd},
        inputs=INPUTS,
        expected=ANSWERS,
        eval_metric="exact",
        runs_per_input=3,
    )
    assert result.total_runs == 72
    assert result.statistical_result.n_inputs == 12
    assert any("averaged per input" in n for n in result.notes)
