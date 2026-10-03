"""Scorers: how an output is graded against the expected answer.

A scorer is any callable ``(output, expected) -> bool | float``. ``True`` /
``False`` (or 1.0 / 0.0) give pass/fail outcomes, which ``compare()`` tests
with the exact McNemar test; other floats are treated as numeric scores and
tested with a paired permutation test.

    >>> from flowprompt.testing import scorers
    >>> compare(variants, inputs, expected=answers, eval_metric=scorers.numeric(abs_tol=0.01))

String shortcuts accepted by ``compare(eval_metric=...)``: ``"exact"``,
``"contains"``, ``"regex"``, ``"numeric"``, ``"similarity"``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

from flowprompt.testing.eval_metrics import similarity_match

__all__ = [
    "Scorer",
    "contains",
    "exact",
    "numeric",
    "regex",
    "resolve_scorer",
    "similarity",
]

Scorer = Callable[[Any, Any], "bool | float"]

_NUMBER = re.compile(r"[-+]?(?:\d+(?:[.,]\d+)*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


def _text(value: Any) -> str:
    if hasattr(value, "model_dump_json"):
        return str(value.model_dump_json())
    return str(value)


def exact(*, case_sensitive: bool = False, strip: bool = True) -> Scorer:
    """Output equals the expected value.

    Strings are compared after stripping whitespace and, by default,
    case-insensitively. A structured (pydantic) output matches a dict
    expectation when ``output.model_dump() == expected``.
    """

    def score(output: Any, expected: Any) -> bool:
        if isinstance(expected, dict) and hasattr(output, "model_dump"):
            return bool(output.model_dump() == expected)
        a, b = _text(output), _text(expected)
        if strip:
            a, b = a.strip(), b.strip()
        if not case_sensitive:
            a, b = a.lower(), b.lower()
        return a == b

    score.__name__ = "exact"
    return score


def contains(*, case_sensitive: bool = False) -> Scorer:
    """The expected value appears somewhere in the output."""

    def score(output: Any, expected: Any) -> bool:
        a, b = _text(output).strip(), _text(expected).strip()
        if not case_sensitive:
            a, b = a.lower(), b.lower()
        return b in a

    score.__name__ = "contains"
    return score


def regex(*, flags: int = re.IGNORECASE, full_match: bool = False) -> Scorer:
    """The expected value is a regular expression the output must match.

    Args:
        flags: ``re`` flags (default case-insensitive).
        full_match: Require the whole (stripped) output to match instead of
            searching for the pattern anywhere.
    """

    def score(output: Any, expected: Any) -> bool:
        pattern = re.compile(str(expected), flags)
        text = _text(output).strip()
        match = pattern.fullmatch(text) if full_match else pattern.search(text)
        return match is not None

    score.__name__ = "regex"
    return score


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    found = _NUMBER.findall(_text(value))
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def numeric(*, abs_tol: float = 1e-6, rel_tol: float = 0.0) -> Scorer:
    """Output is numerically close to the expected number.

    Numbers are extracted from text outputs (the last number in the text is
    used, so "The answer is 42." scores against 42).
    """

    def score(output: Any, expected: Any) -> bool:
        got, want = _to_number(output), _to_number(expected)
        if got is None or want is None:
            return False
        return math.isclose(got, want, abs_tol=abs_tol, rel_tol=rel_tol)

    score.__name__ = "numeric"
    return score


def similarity(threshold: float = 0.7) -> Scorer:
    """Character-level similarity ratio (difflib) of at least ``threshold``."""

    def score(output: Any, expected: Any) -> bool:
        return similarity_match(_text(output), _text(expected), threshold)

    score.__name__ = "similarity"
    return score


_NAMED: dict[str, Callable[[], Scorer]] = {
    "exact": exact,
    "exact_match": exact,
    "contains": contains,
    "contains_match": contains,
    "regex": regex,
    "numeric": numeric,
    "similarity": similarity,
    "similarity_match": similarity,
}


def resolve_scorer(metric: str | Callable[..., Any]) -> Callable[..., Any]:
    """Turn a scorer name or callable into a scorer callable."""
    if callable(metric):
        return metric
    name = str(metric).strip().lower()
    if name not in _NAMED:
        raise ValueError(
            f"Unknown eval metric '{metric}'. Valid names: {', '.join(sorted(_NAMED))}"
        )
    return _NAMED[name]()
