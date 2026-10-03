"""``flowprompt compare``: run a paired comparison from the command line.

    flowprompt compare variants.py dataset.jsonl --model gpt-4o-mini \\
        --metric exact --control baseline --fail-on-regression \\
        --markdown report.md --html report.html

* ``variants.py`` defines the variants: a ``VARIANTS`` dict (name -> Prompt
  class, ``(PromptClass, "model")`` tuple or callable), or else every Prompt
  subclass defined in the file, in definition order.
* ``dataset.jsonl`` has one JSON object per line. Either
  ``{"input": {...}, "expected": ...}`` or a flat object whose keys (except
  the expected key) are the template variables.

Exit codes: 0 = finished (and no regression when ``--fail-on-regression``),
1 = a variant is significantly worse than the control, 2 = bad input.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_USAGE = 2


class UsageError(Exception):
    """Invalid command-line input."""


def load_variants(path: str | Path, attribute: str = "VARIANTS") -> dict[str, Any]:
    """Load variants from a Python file."""
    from flowprompt.core.prompt import Prompt

    file = Path(path)
    if not file.exists():
        raise UsageError(f"variants file not found: {file}")
    sys.path.insert(0, str(file.parent.resolve()))
    try:
        namespace = runpy.run_path(str(file), run_name="flowprompt_variants")
    finally:
        sys.path.pop(0)
    variants = namespace.get(attribute)
    if variants is not None:
        if not isinstance(variants, dict) or len(variants) < 2:
            raise UsageError(f"{attribute} in {file} must be a dict with 2+ variants")
        return dict(variants)
    found = {
        name: obj
        for name, obj in namespace.items()
        if isinstance(obj, type)
        and issubclass(obj, Prompt)
        and obj is not Prompt
        and getattr(obj, "__module__", None) == "flowprompt_variants"
    }
    if len(found) < 2:
        raise UsageError(
            f"{file} must define {attribute} = {{...}} or at least two Prompt classes"
        )
    return found


def load_dataset(
    path: str | Path, expected_key: str = "expected", input_key: str = "input"
) -> tuple[list[dict[str, Any]], list[Any] | None]:
    """Read a JSONL dataset into (inputs, expected)."""
    file = Path(path)
    if not file.exists():
        raise UsageError(f"dataset not found: {file}")
    inputs: list[dict[str, Any]] = []
    expected: list[Any] = []
    with_expected = 0
    for lineno, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UsageError(f"{file}:{lineno}: invalid JSON ({exc.msg})") from None
        if not isinstance(row, dict):
            raise UsageError(f"{file}:{lineno}: each line must be a JSON object")
        if expected_key in row:
            with_expected += 1
        exp = row.get(expected_key)
        if isinstance(row.get(input_key), dict):
            inp = row[input_key]
        else:
            inp = {k: v for k, v in row.items() if k != expected_key}
        inputs.append(inp)
        expected.append(exp)
    if not inputs:
        raise UsageError(f"{file} has no rows")
    if with_expected not in (0, len(inputs)):
        raise UsageError(
            f"{file}: {with_expected} of {len(inputs)} rows have '{expected_key}'; "
            "provide it for all rows or none"
        )
    return inputs, (expected if with_expected else None)


def regressions(result: Any) -> list[Any]:
    """Comparisons where a treatment is significantly worse than the control."""
    if result.comparison_mode != "control":
        return []
    return [
        c
        for c in result.comparisons
        if c.significant and (c.difference or 0.0) < 0 and c.control == result.control
    ]


def run_compare_command(
    variants_file: str,
    dataset: str,
    *,
    model: str = "gpt-4o",
    metric: str = "contains",
    control: str | None = None,
    runs: int = 1,
    confidence: float = 0.95,
    temperature: float = 0.0,
    comparisons: str = "control",
    expected_key: str = "expected",
    input_key: str = "input",
    variants_attr: str = "VARIANTS",
    markdown: str | None = None,
    html: str | None = None,
    json_path: str | None = None,
    fail_on_regression: bool = False,
    echo: Any = print,
) -> int:
    """Run the comparison and return a process exit code."""
    from flowprompt.testing.compare import compare

    try:
        variants = load_variants(variants_file, variants_attr)
        inputs, expected = load_dataset(dataset, expected_key, input_key)
        result = compare(
            variants,
            inputs,
            model,
            expected=expected,
            eval_metric=metric,
            confidence_level=confidence,
            runs_per_input=runs,
            temperature=temperature,
            control=control,
            comparisons=comparisons,
        )
    except UsageError as exc:
        echo(f"Error: {exc}")
        return EXIT_USAGE
    except ValueError as exc:
        echo(f"Error: {exc}")
        return EXIT_USAGE

    echo(str(result))
    for target, writer in ((markdown, "md"), (html, "html"), (json_path, "json")):
        if target:
            path = Path(target)
            if writer == "md":
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:  # job summaries append
                    fh.write(result.to_markdown() + "\n")
            elif writer == "html":
                result.save_report(path.with_suffix(path.suffix or ".html"))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(result.to_dict(), indent=2, default=str),
                    encoding="utf-8",
                )
            echo(f"Wrote {path}")

    if fail_on_regression:
        bad = regressions(result)
        if bad:
            names = ", ".join(str(c.treatment) for c in bad)
            echo(
                f"Regression: {names} significantly worse than {result.control}. "
                "Failing (exit code 1)."
            )
            return EXIT_REGRESSION
    return EXIT_OK
