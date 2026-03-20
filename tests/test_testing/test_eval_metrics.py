"""Tests for eval_metrics module."""

from __future__ import annotations

import pytest

from flowprompt.testing.eval_metrics import (
    contains_match,
    exact_match,
    resolve_eval_metric,
    similarity_match,
)


class TestExactMatch:
    def test_identical(self) -> None:
        assert exact_match("hello", "hello")

    def test_case_insensitive(self) -> None:
        assert exact_match("Hello World", "hello world")

    def test_strips_whitespace(self) -> None:
        assert exact_match("  hello  ", "hello")

    def test_mismatch(self) -> None:
        assert not exact_match("hello", "world")

    def test_non_string_coercion(self) -> None:
        assert exact_match(42, "42")


class TestContainsMatch:
    def test_substring_present(self) -> None:
        assert contains_match("The answer is 42.", "42")

    def test_case_insensitive(self) -> None:
        assert contains_match("Hello World", "hello")

    def test_exact_is_also_contains(self) -> None:
        assert contains_match("hello", "hello")

    def test_not_present(self) -> None:
        assert not contains_match("hello", "world")

    def test_empty_expected(self) -> None:
        assert contains_match("anything", "")


class TestSimilarityMatch:
    def test_identical_strings(self) -> None:
        assert similarity_match("hello world", "hello world")

    def test_similar_strings(self) -> None:
        assert similarity_match("hello world", "hello worlds", threshold=0.8)

    def test_below_threshold(self) -> None:
        assert not similarity_match("hello", "completely different", threshold=0.8)

    def test_custom_threshold(self) -> None:
        assert similarity_match("abc", "abd", threshold=0.5)
        assert not similarity_match("abc", "xyz", threshold=0.5)


class TestResolveEvalMetric:
    def test_resolve_exact(self) -> None:
        fn = resolve_eval_metric("exact")
        assert fn is exact_match

    def test_resolve_exact_match(self) -> None:
        fn = resolve_eval_metric("exact_match")
        assert fn is exact_match

    def test_resolve_contains(self) -> None:
        fn = resolve_eval_metric("contains")
        assert fn is contains_match

    def test_resolve_similarity(self) -> None:
        fn = resolve_eval_metric("similarity")
        assert fn is similarity_match

    def test_resolve_callable(self) -> None:
        custom = lambda o, e: o == e  # noqa: E731
        assert resolve_eval_metric(custom) is custom

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown eval metric"):
            resolve_eval_metric("nonexistent")

    def test_case_insensitive_name(self) -> None:
        fn = resolve_eval_metric("EXACT")
        assert fn is exact_match
