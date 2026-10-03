"""Tests for `flowprompt compare`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flowprompt.cli.compare_cmd import (
    EXIT_OK,
    EXIT_REGRESSION,
    EXIT_USAGE,
    load_dataset,
    load_variants,
    run_compare_command,
)

VARIANTS = """
from flowprompt import Prompt

def baseline(inp):
    return str(inp["q"] * 2)

def candidate(inp):
    q = inp["q"]
    return str(q * 2 + (q % 2))   # wrong on odd inputs

VARIANTS = {"baseline": baseline, "candidate": candidate}
"""

PROMPTS = """
from flowprompt import Prompt

class Short(Prompt):
    system = "Be brief."
    user = "Q: {q}"

class Long(Prompt):
    system = "Be thorough."
    user = "Question: {q}"
"""


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    variants = tmp_path / "variants.py"
    variants.write_text(VARIANTS)
    data = tmp_path / "data.jsonl"
    data.write_text(
        "\n".join(json.dumps({"q": i, "expected": str(i * 2)}) for i in range(20))
    )
    return variants, data


def test_regression_gate_fails_ci(files: tuple[Path, Path], tmp_path: Path) -> None:
    variants, data = files
    out: list[str] = []
    md = tmp_path / "summary.md"
    code = run_compare_command(
        str(variants),
        str(data),
        metric="exact",
        control="baseline",
        markdown=str(md),
        html=str(tmp_path / "r.html"),
        json_path=str(tmp_path / "r.json"),
        fail_on_regression=True,
        echo=out.append,
    )
    assert code == EXIT_REGRESSION
    text = "\n".join(out)
    assert "baseline beats candidate" in text
    assert "Regression: candidate significantly worse than baseline" in text
    assert md.read_text().startswith("### Prompt comparison")
    assert (tmp_path / "r.html").read_text().startswith("<!doctype html>")
    assert json.loads((tmp_path / "r.json").read_text())["winner"] == "baseline"


def test_no_regression_exits_zero(files: tuple[Path, Path]) -> None:
    variants, data = files
    code = run_compare_command(
        str(variants),
        str(data),
        metric="exact",
        control="candidate",  # baseline is better -> improvement, not regression
        fail_on_regression=True,
        echo=lambda _msg: None,
    )
    assert code == EXIT_OK


def test_markdown_appends_for_job_summaries(
    files: tuple[Path, Path], tmp_path: Path
) -> None:
    variants, data = files
    md = tmp_path / "summary.md"
    md.write_text("# Existing summary\n")
    run_compare_command(
        str(variants), str(data), metric="exact", markdown=str(md), echo=lambda _m: None
    )
    content = md.read_text()
    assert content.startswith("# Existing summary")
    assert "### Prompt comparison" in content


def test_prompt_classes_discovered_in_definition_order(tmp_path: Path) -> None:
    path = tmp_path / "prompts.py"
    path.write_text(PROMPTS)
    variants = load_variants(path)
    assert list(variants) == ["Short", "Long"]


def test_dataset_formats(tmp_path: Path) -> None:
    nested = tmp_path / "nested.jsonl"
    nested.write_text('{"input": {"text": "hi"}, "expected": "x"}\n\n')
    inputs, expected = load_dataset(nested)
    assert inputs == [{"text": "hi"}] and expected == ["x"]

    flat = tmp_path / "flat.jsonl"
    flat.write_text('{"text": "hi"}\n{"text": "yo"}\n')
    inputs, expected = load_dataset(flat)
    assert inputs == [{"text": "hi"}, {"text": "yo"}] and expected is None


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json\n", "invalid JSON"),
        ("[1, 2]\n", "must be a JSON object"),
        ('{"a": 1, "expected": 1}\n{"a": 2}\n', "for all rows or none"),
        ("", "has no rows"),
    ],
)
def test_bad_datasets_are_usage_errors(
    files: tuple[Path, Path], tmp_path: Path, content: str, message: str
) -> None:
    variants, _ = files
    data = tmp_path / "bad.jsonl"
    data.write_text(content)
    out: list[str] = []
    code = run_compare_command(str(variants), str(data), echo=out.append)
    assert code == EXIT_USAGE
    assert message in out[0]


def test_missing_files_and_bad_variants(tmp_path: Path) -> None:
    out: list[str] = []
    assert run_compare_command("nope.py", "nope.jsonl", echo=out.append) == EXIT_USAGE
    one = tmp_path / "one.py"
    one.write_text("VARIANTS = {'a': print}\n")
    data = tmp_path / "d.jsonl"
    data.write_text('{"q": 1}\n')
    assert run_compare_command(str(one), str(data), echo=out.append) == EXIT_USAGE


def test_typer_command_wiring(files: tuple[Path, Path]) -> None:
    typer_testing = pytest.importorskip("typer.testing")
    from flowprompt.cli.main import create_app

    variants, data = files
    runner = typer_testing.CliRunner()
    result = runner.invoke(
        create_app(),
        [
            "compare",
            str(variants),
            str(data),
            "--metric",
            "exact",
            "--control",
            "baseline",
            "--fail-on-regression",
        ],
    )
    assert result.exit_code == EXIT_REGRESSION, result.output
    assert "Paired comparisons vs baseline" in result.output
