"""Pytest-friendly assertion wrappers for ComparisonResult.

Provides ``PromptTestResult`` which wraps a ``ComparisonResult`` with
convenient assertion helpers designed for use in pytest test suites:

    def test_prompt(fp_compare):
        result = fp_compare(...)
        result.assert_significant()
        result.assert_winner("v1")
"""

from __future__ import annotations

from typing import Any


class PromptTestResult:
    """Wraps a ``ComparisonResult`` with pytest-friendly assertions.

    Attributes:
        result: The underlying ``ComparisonResult``.
    """

    def __init__(self, result: Any) -> None:
        self.result = result

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_significant(self) -> bool:
        """Whether the statistical test found a significant difference."""
        sr = self.result.statistical_result
        return sr is not None and sr.significant

    @property
    def winner(self) -> str | None:
        """Name of the winning variant, or ``None``."""
        return self.result.winner

    @property
    def p_value(self) -> float | None:
        """P-value from the significance test, or ``None``."""
        sr = self.result.statistical_result
        return sr.p_value if sr is not None else None

    # ------------------------------------------------------------------
    # Assertion helpers
    # ------------------------------------------------------------------

    def assert_significant(self, threshold: float = 0.05) -> None:
        """Fail unless the comparison reached statistical significance.

        Args:
            threshold: Maximum p-value to consider significant (default 0.05).
        """
        try:
            import pytest
        except ImportError:  # pragma: no cover
            raise  # let the caller see the real ImportError

        sr = self.result.statistical_result
        if sr is None:
            pytest.fail("No statistical result available (too few samples?)")

        if sr.p_value > threshold:
            lines = [
                f"Result not significant: p={sr.p_value:.4f} > {threshold}",
                "",
                "Variant breakdown:",
            ]
            for name, v in self.result.variants.items():
                marker = " << WINNER" if name == self.result.winner else ""
                lines.append(
                    f"  {name}: {v.success_rate:.0%} success, {v.samples} runs{marker}"
                )
            pytest.fail("\n".join(lines))

    def assert_winner(self, expected: str) -> None:
        """Fail unless the given variant is the winner.

        Args:
            expected: The variant name that should have won.
        """
        try:
            import pytest
        except ImportError:  # pragma: no cover
            raise

        if self.result.winner != expected:
            lines = [
                f"Expected winner '{expected}', got '{self.result.winner}'",
                "",
                "Variant breakdown:",
            ]
            for name, v in self.result.variants.items():
                marker = " << WINNER" if name == self.result.winner else ""
                lines.append(
                    f"  {name}: {v.success_rate:.0%} success, {v.samples} runs{marker}"
                )
            pytest.fail("\n".join(lines))

    def assert_no_errors(self) -> None:
        """Fail if any variant recorded errors."""
        try:
            import pytest
        except ImportError:  # pragma: no cover
            raise

        errors: dict[str, list[str]] = {}
        for name, v in self.result.variants.items():
            if v.errors:
                errors[name] = v.errors

        if errors:
            lines = ["Variants had errors:"]
            for name, errs in errors.items():
                lines.append(f"  {name}: {len(errs)} error(s)")
                for e in errs[:3]:
                    lines.append(f"    - {e}")
                if len(errs) > 3:
                    lines.append(f"    ... and {len(errs) - 3} more")
            pytest.fail("\n".join(lines))

    # ------------------------------------------------------------------
    # Delegate everything else to the underlying ComparisonResult
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self.result, name)

    def __str__(self) -> str:
        return str(self.result)

    def __repr__(self) -> str:
        return f"PromptTestResult({self.result!r})"
