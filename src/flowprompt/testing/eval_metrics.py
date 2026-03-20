"""Built-in evaluation metrics for prompt comparison.

Provides functions to compare LLM outputs against expected ground-truth values:

    from flowprompt.testing.eval_metrics import exact_match, contains_match

    exact_match("Hello World", "hello world")     # True (case-insensitive)
    contains_match("The answer is 42.", "42")      # True
"""

from __future__ import annotations

import difflib
from collections.abc import Callable


def exact_match(output: str, expected: str) -> bool:
    """Case-insensitive exact match after stripping whitespace."""
    return str(output).strip().lower() == str(expected).strip().lower()


def contains_match(output: str, expected: str) -> bool:
    """Check if expected value appears in the output (case-insensitive)."""
    return str(expected).strip().lower() in str(output).strip().lower()


def similarity_match(output: str, expected: str, threshold: float = 0.7) -> bool:
    """Check if output is similar enough to expected using SequenceMatcher.

    Args:
        output: The LLM output string.
        expected: The expected ground-truth string.
        threshold: Minimum similarity ratio (0.0-1.0). Default 0.7.

    Returns:
        True if similarity ratio >= threshold.
    """
    ratio = difflib.SequenceMatcher(
        None, str(output).strip().lower(), str(expected).strip().lower()
    ).ratio()
    return ratio >= threshold


_BUILTIN_METRICS: dict[str, Callable[..., bool]] = {
    "exact": exact_match,
    "exact_match": exact_match,
    "contains": contains_match,
    "contains_match": contains_match,
    "similarity": similarity_match,
    "similarity_match": similarity_match,
}


def resolve_eval_metric(metric: str | Callable[..., bool]) -> Callable[..., bool]:
    """Resolve a metric name or callable to a metric function.

    Args:
        metric: Either a string name ("exact", "contains", "similarity")
            or a callable ``(output, expected) -> bool``.

    Returns:
        The resolved callable.

    Raises:
        ValueError: If the string name is not recognised.
    """
    if callable(metric):
        return metric
    name = str(metric).strip().lower()
    if name not in _BUILTIN_METRICS:
        raise ValueError(
            f"Unknown eval metric '{metric}'. "
            f"Valid names: {', '.join(sorted(set(_BUILTIN_METRICS.keys())))}"
        )
    return _BUILTIN_METRICS[name]
