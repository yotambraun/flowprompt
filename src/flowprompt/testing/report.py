"""Text, Markdown and HTML reports for a ComparisonResult.

All three renderers read the same :class:`ComparisonResult`, so the console
output, a CI job summary and a shared HTML file always agree.
"""

from __future__ import annotations

import html
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowprompt.testing.compare import ComparisonResult, VariantResult
    from flowprompt.testing.statistics import StatisticalResult

__all__ = ["render_html", "render_markdown", "render_text", "verdict_text"]

_METHOD_NAMES = {
    "mcnemar_exact": "exact McNemar test",
    "paired_permutation": "paired sign-flip permutation test",
    "two_proportion_z_test": "two-proportion z-test (unpaired, deprecated)",
    "chi_squared_test": "chi-squared test (unpaired, deprecated)",
    "welch_t_test": "Welch t-test (unpaired, deprecated)",
    "bayesian_ab_test": "Bayesian beta-binomial (unpaired, deprecated)",
}

_CI_NAMES = {
    "agresti_min": "Agresti-Min interval",
    "bca_bootstrap": "BCa bootstrap interval over inputs",
}

_OUTCOME_DESCRIPTIONS = {
    "accuracy": "accuracy vs expected outputs",
    "success": "pass rate (success_fn)",
    "score": "mean score (metric_fn)",
    "no_error": "success = completed without error (outputs not graded)",
}

_OUTCOME_LABELS = {
    "accuracy": "Accuracy",
    "success": "Pass rate",
    "score": "Mean score",
    "no_error": "Success",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _rate(result: ComparisonResult, value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if result.is_proportion:
        return f"{value * 100:.1f}%"
    return f"{value:.3g}"


def _range(
    result: ComparisonResult, lo: float | None, hi: float | None, sep: str
) -> str:
    if lo is None or hi is None or math.isinf(lo) or math.isinf(hi):
        return "-"
    if result.is_proportion:
        return f"{lo * 100:.1f}{sep}{hi * 100:.1f}%"
    return f"{lo:.3g}{sep}{hi:.3g}"


def _diff(result: ComparisonResult, value: float | None, minus: str = "-") -> str:
    if value is None:
        return "-"
    sign = "+" if value >= 0 else minus
    if result.is_proportion:
        return f"{sign}{abs(value) * 100:.1f} pts"
    return f"{sign}{abs(value):.3g}"


def _ci(result: ComparisonResult, sr: StatisticalResult, minus: str = "-") -> str:
    lo, hi = sr.ci_low, sr.ci_high
    if lo is None or hi is None or math.isinf(lo) or math.isinf(hi):
        return "-"

    def one(v: float) -> str:
        sign = "+" if v >= 0 else minus
        if result.is_proportion:
            return f"{sign}{abs(v) * 100:.1f}"
        return f"{sign}{abs(v):.3g}"

    return f"[{one(lo)}, {one(hi)}]"


def _p(p: float | None) -> str:
    if p is None:
        return "-"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def _money(value: float | None, known: bool = True) -> str:
    if value is None or not known:
        return "-"
    if value == 0:
        return "$0"
    if value < 0.01:
        decimals = min(10, -math.floor(math.log10(value)) + 1)
        return f"${value:.{decimals}f}"
    return f"${value:.4f}" if value < 1 else f"${value:,.2f}"


def _latency(v: VariantResult) -> str:
    if not v.samples:
        return "-"
    return f"{v.mean_latency_ms:.0f} / {v.p95_latency_ms:.0f} ms"


def _points(result: ComparisonResult, value: float) -> str:
    if result.is_proportion:
        return f"{abs(value) * 100:.1f} points"
    return f"{abs(value):.3g}"


def _ci_plain(result: ComparisonResult, sr: StatisticalResult, sign: float) -> str:
    lo, hi = sr.ci_low, sr.ci_high
    if lo is None or hi is None or math.isinf(lo) or math.isinf(hi):
        return ""
    lo, hi = sorted((lo * sign, hi * sign))
    scale = 100 if result.is_proportion else 1
    fmt = "{:.1f}" if result.is_proportion else "{:.3g}"
    level = f"{result.confidence_level:.0%}"
    return f" ({level} CI {fmt.format(lo * scale)} to {fmt.format(hi * scale)})"


def _p_phrase(sr: StatisticalResult) -> str:
    if sr.adjusted_p is not None:
        return f"adjusted p={_p(sr.adjusted_p)}"
    return f"p={_p(sr.p_value)}"


def method_text(result: ComparisonResult) -> str:
    """Describe how the comparison was tested."""
    if not result.comparisons:
        return ""
    methods = sorted({c.method or c.test_name for c in result.comparisons})
    names = ", ".join(_METHOD_NAMES.get(m, m) for m in methods)
    ci = sorted(
        {
            str(c.details.get("ci_method"))
            for c in result.comparisons
            if c.details.get("ci_method")
        }
    )
    parts = [
        f"Paired design: every variant ran on the same {result.n_inputs} inputs"
        + (
            f" ({result.runs_per_input} runs each, averaged per input)"
            if result.runs_per_input > 1
            else ""
        )
        + ".",
        f"Test: {names}.",
    ]
    if ci:
        parts.append("Interval: " + ", ".join(_CI_NAMES.get(c, c) for c in ci) + ".")
    if result.correction:
        parts.append(
            f"Multiple comparisons: Holm-adjusted over {len(result.comparisons)} "
            "comparisons."
        )
    parts.append(f"Significance level: {result.alpha:g}.")
    return " ".join(parts)


def verdict_text(result: ComparisonResult) -> str:
    """One sentence: who won, by how much, how sure."""
    if result.is_dry_run:
        return "Dry run: no calls were made."
    if not result.comparisons:
        return "Nothing to compare."
    if not result.enough_data:
        return (
            f"Not enough data: with {result.n_inputs} inputs no difference can be "
            f"significant; use at least {result.min_inputs_for_significance} inputs."
        )
    sr = result.statistical_result
    if sr is None:
        return "Nothing to compare."
    diff = sr.difference or 0.0
    if result.winner is not None:
        if result.winner == sr.treatment:
            loser, sign = sr.control, 1.0
        else:
            loser, sign = sr.treatment, -1.0
        others = len(result.variants) - 1
        target = str(loser)
        if others > 1 and (
            result.comparison_mode == "all" or result.winner == result.control
        ):
            target = f"every other variant (closest: {loser})"
        return (
            f"{result.winner} beats {target} by {_points(result, diff)}"
            f"{_ci_plain(result, sr, sign)}, {_p_phrase(sr)}: significant."
        )
    leader = sr.treatment if diff >= 0 else sr.control
    other = sr.control if diff >= 0 else sr.treatment
    sign = 1.0 if diff >= 0 else -1.0
    if abs(diff) < 1e-12:
        return (
            f"No difference between {sr.treatment} and {sr.control}"
            f"{_ci_plain(result, sr, 1.0)}, {_p_phrase(sr)}."
        )
    return (
        f"No significant difference: {leader} leads {other} by "
        f"{_points(result, diff)}{_ci_plain(result, sr, sign)}, "
        f"{_p_phrase(sr)}."
    )


def _plan_line(result: ComparisonResult) -> str | None:
    if result.winner is not None or not result.comparisons:
        return None
    mde = 0.1 if result.is_proportion else None
    sr = result.statistical_result
    if mde is None:
        sd = sr.details.get("sd_difference") if sr else None
        if not sd:
            return None
        mde = float(sd) / 2
    try:
        plan = result.sample_size_plan(mde)
    except ValueError:
        return None
    if plan is None:
        return None
    if result.is_proportion:
        return plan.summary() + "."
    text = (
        f"To detect a difference of {mde:.3g} at {plan.power:.0%} power you need "
        f"~{plan.n_inputs} inputs"
    )
    if plan.estimated_cost_usd is not None:
        text += f" (≈ ${plan.estimated_cost_usd:.2f})"
    return text + "."


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _meta(result: ComparisonResult, code: bool = False) -> list[str]:
    meta = [
        _plural(len(result.variants), "variant"),
        _plural(result.n_inputs, "input")
        + (
            f" x {_plural(result.runs_per_input, 'run')}"
            if result.runs_per_input > 1
            else ""
        ),
        _OUTCOME_DESCRIPTIONS.get(result.outcome, result.outcome),
    ]
    if result.model:
        meta.append(f"model `{result.model}`" if code else f"model {result.model}")
    return meta


def _rows(result: ComparisonResult) -> list[tuple[str, VariantResult]]:
    return list(result.variants.items())


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


def _dry_run_text(result: ComparisonResult) -> str:
    lines = ["Comparison Results (DRY RUN)", "=" * 40]
    est = result.estimated_cost or {}
    cost_str = (
        f"${est['estimated_cost_usd']:.2f}"
        if est.get("estimated_cost_usd") is not None
        else "unknown"
    )
    lines.append(f"  Estimated cost: {cost_str} for {est.get('total_calls', 0)} calls")
    per_variant = est.get("per_variant", {})
    if per_variant:
        lines.append("  Per variant:")
        for vname, vinfo in per_variant.items():
            vcost = (
                f"~${vinfo['cost_usd']:.2f}"
                if vinfo.get("cost_usd") is not None
                else "unknown"
            )
            lines.append(
                f"    {vname}: {vinfo['calls']} calls, "
                f"~{vinfo['input_tokens']} tokens, {vcost}"
            )
    return "\n".join(lines)


def _table(headers: list[str], rows: list[list[str]], align: str) -> list[str]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            out.append(
                cell.rjust(widths[i]) if align[i] == "r" else cell.ljust(widths[i])
            )
        return "  " + "  ".join(out).rstrip()

    lines = [fmt(headers), "  " + "  ".join("-" * w for w in widths)]
    lines.extend(fmt(r) for r in rows)
    return lines


def render_text(result: ComparisonResult) -> str:
    """ASCII-only text report (safe on any console encoding)."""
    if result.is_dry_run:
        return _dry_run_text(result)

    label = _OUTCOME_LABELS.get(result.outcome, "Score")
    header = "Comparison Results"
    lines = [header, "=" * 72]
    if result.n_inputs:
        lines.append(" | ".join(_meta(result)))
    lines.append("")

    rows: list[list[str]] = []
    for name, v in _rows(result):
        tag = ""
        if name == result.winner:
            tag = " * winner"
        elif name == result.control and result.comparison_mode == "control":
            tag = " (control)"
        rows.append(
            [
                f"{name}{tag}",
                _rate(
                    result, v.mean_score if v.mean_score is not None else v.success_rate
                ),
                _range(result, v.ci_low, v.ci_high, "-"),
                _latency(v),
                _money(v.total_cost_usd, v.cost_known),
                _money(v.cost_per_correct),
                str(v.error_count),
            ]
        )
    ci_head = f"{result.confidence_level:.0%} CI"
    lines.extend(
        _table(
            [
                "Variant",
                label,
                ci_head,
                "Latency mean/p95",
                "Cost",
                "Cost/correct",
                "Errors",
            ],
            rows,
            "lrrrrrr",
        )
    )

    if result.comparisons:
        lines.append("")
        base = (
            f"Paired comparisons vs {result.control}"
            if result.comparison_mode == "control"
            else "Paired comparisons (all pairs)"
        )
        lines.append(base)
        crow: list[list[str]] = []
        for c in result.comparisons:
            row = [
                f"{c.treatment} vs {c.control}",
                _diff(result, c.difference),
                _ci(result, c),
                _p(c.p_value),
            ]
            if result.correction:
                row.append(_p(c.adjusted_p))
            row.append("significant" if c.significant else "not significant")
            crow.append(row)
        heads = ["Comparison", "Difference", ci_head, "p"]
        if result.correction:
            heads.append("Holm p")
        heads.append("Result")
        lines.extend(
            _table(heads, crow, "lrrr" + ("r" if result.correction else "") + "l")
        )

    lines.append("")
    lines.append(f"Verdict: {verdict_text(result)}")
    if result.comparisons:
        lines.append(f"Method: {method_text(result)}")
    plan = _plan_line(result)
    if plan:
        lines.append(f"Planning: {plan}")
    if result.variants and any(v.llm_calls for v in result.variants.values()):
        total_calls = sum(v.llm_calls for v in result.variants.values())
        if result.cost_known:
            lines.append(
                f"Cost: {_money(result.total_cost_usd)} total for {total_calls} LLM calls."
            )
        else:
            lines.append(
                f"Cost: unknown for {total_calls} LLM calls (no price for this model)."
            )
    if result.notes:
        lines.append("")
        lines.append("Notes:")
        for note in result.notes:
            lines.append(f"  - {note}")
    if not result.winner and result.comparisons and result.enough_data:
        lines.append("")
        lines.append("  No clear winner (results not statistically significant)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def render_markdown(result: ComparisonResult) -> str:
    """GitHub-flavoured Markdown report."""
    if result.is_dry_run:
        return "```\n" + _dry_run_text(result) + "\n```\n"
    label = _OUTCOME_LABELS.get(result.outcome, "Score")
    title = f"{result.winner} wins" if result.winner else "No significant winner"
    if result.comparisons and not result.enough_data:
        title = "Not enough data"
    out = [f"### Prompt comparison: {_md_escape(title)}", ""]
    out.append(f"> **{_md_escape(verdict_text(result))}**")
    out.append("")
    out.append(" · ".join(_meta(result, code=True)))
    out.append("")
    ci_head = f"{result.confidence_level:.0%} CI"
    out.append(
        f"| Variant | {label} | {ci_head} | Latency (mean / p95) | Cost | Cost per correct | Errors |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, v in _rows(result):
        cell = f"`{_md_escape(name)}`"
        if name == result.winner:
            cell = f"**`{_md_escape(name)}`** (winner)"
        elif name == result.control and result.comparison_mode == "control":
            cell += " (control)"
        out.append(
            "| "
            + " | ".join(
                [
                    cell,
                    _rate(
                        result,
                        v.mean_score if v.mean_score is not None else v.success_rate,
                    ),
                    _range(result, v.ci_low, v.ci_high, "–"),
                    _latency(v),
                    _money(v.total_cost_usd, v.cost_known),
                    _money(v.cost_per_correct),
                    str(v.error_count),
                ]
            )
            + " |"
        )
    if result.comparisons:
        out.append("")
        holm = " | Holm-adjusted p" if result.correction else ""
        out.append(f"| Comparison | Difference | {ci_head} | p{holm} | Result |")
        out.append(
            "|---|---:|---:|---:" + ("|---:" if result.correction else "") + "|---|"
        )
        for c in result.comparisons:
            row = [
                f"`{_md_escape(str(c.treatment))}` vs `{_md_escape(str(c.control))}`",
                _diff(result, c.difference, "−"),
                _ci(result, c, "−"),
                _p(c.p_value),
            ]
            if result.correction:
                row.append(_p(c.adjusted_p))
            row.append("**significant**" if c.significant else "not significant")
            out.append("| " + " | ".join(row) + " |")
    out.append("")
    method = method_text(result)
    if method:
        out.append(f"<sub>{method}</sub>")
        out.append("")
    plan = _plan_line(result)
    if plan:
        out.append(f"**Planning:** {plan}")
        out.append("")
    if result.notes:
        for note in result.notes:
            out.append(f"- {note}")
        out.append("")
    if result.cost_known:
        total_calls = sum(v.llm_calls for v in result.variants.values())
        out.append(
            f"Total cost: {_money(result.total_cost_usd)} for {total_calls} LLM calls."
        )
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
:root{--bg:#f7f7f5;--card:#ffffff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e4e4e0;
--accent:#2f6fde;--good:#16794c;--good-bg:#e8f5ee;--warn:#9a6700;--warn-bg:#fff6dd;
--neutral-bg:#eef1f6;--bar:#c9d6f2;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#121214;--card:#1b1b1f;--ink:#ececf0;
--muted:#a0a0aa;--line:#2c2c33;--accent:#7aa2ff;--good:#5fd39a;--good-bg:#173226;
--warn:#f2c35b;--warn-bg:#33290f;--neutral-bg:#20242d;--bar:#2f3d5c}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1060px;margin:0 auto;padding:40px 20px 56px}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 6px}
h1{font-size:26px;line-height:1.25;margin:0 0 6px;font-weight:650}
.meta{color:var(--muted);margin:0 0 24px}
.verdict{border-radius:12px;padding:16px 18px;margin:0 0 28px;border:1px solid var(--line);
background:var(--neutral-bg);font-size:16px}
.verdict.win{background:var(--good-bg);border-color:transparent}
.verdict.win strong{color:var(--good)}
.verdict.warn{background:var(--warn-bg);border-color:transparent}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:4px 0;margin:0 0 22px;overflow-x:auto}
h2{font-size:14px;font-weight:600;margin:28px 0 10px;color:var(--muted);letter-spacing:.02em}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:10px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th{white-space:normal;vertical-align:bottom}
tr:last-child td{border-bottom:none}
th{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
th:first-child,td:first-child{text-align:left}
td.name{font-family:var(--mono);font-size:14px}
.tag{display:inline-block;font:600 11px/1 Inter,sans-serif;padding:4px 7px;border-radius:999px;margin-left:8px;
vertical-align:1px;background:var(--neutral-bg);color:var(--muted)}
.tag.win{background:var(--good-bg);color:var(--good)}
.sig{color:var(--good);font-weight:600}
.ns{color:var(--muted)}
svg{display:block}
.viz{width:130px}
footer{color:var(--muted);font-size:13px;margin-top:26px}
footer ul{padding-left:18px;margin:8px 0}
code{font-family:var(--mono);font-size:13px}
"""


def _bar_svg(
    value: float | None, lo: float | None, hi: float | None, vmin: float, vmax: float
) -> str:
    if value is None:
        return ""
    width, height = 130, 18

    def x(v: float) -> float:
        if vmax == vmin:
            return width / 2
        return 4 + (width - 8) * (min(max(v, vmin), vmax) - vmin) / (vmax - vmin)

    parts = [
        f'<svg class="viz" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-hidden="true">',
        f'<rect x="4" y="6" width="{width - 8}" height="6" rx="3" fill="var(--line)"/>',
        f'<rect x="4" y="6" width="{max(x(value) - 4, 0):.1f}" height="6" rx="3" fill="var(--bar)"/>',
    ]
    if lo is not None and hi is not None and not (math.isinf(lo) or math.isinf(hi)):
        parts.append(
            f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="9" y2="9" stroke="var(--ink)" stroke-width="1.5"/>'
        )
        for edge in (lo, hi):
            parts.append(
                f'<line x1="{x(edge):.1f}" x2="{x(edge):.1f}" y1="5" y2="13" stroke="var(--ink)" stroke-width="1.5"/>'
            )
    parts.append(f'<circle cx="{x(value):.1f}" cy="9" r="3.5" fill="var(--accent)"/>')
    parts.append("</svg>")
    return "".join(parts)


def _forest_svg(sr: StatisticalResult, vmin: float, vmax: float) -> str:
    width, height = 130, 18
    diff = sr.difference or 0.0

    def x(v: float) -> float:
        if vmax == vmin:
            return width / 2
        return 4 + (width - 8) * (min(max(v, vmin), vmax) - vmin) / (vmax - vmin)

    color = "var(--good)" if sr.significant else "var(--muted)"
    parts = [
        f'<svg class="viz" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-hidden="true">',
        f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="1" y2="17" stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 2"/>',
    ]
    if (
        sr.ci_low is not None
        and sr.ci_high is not None
        and not (math.isinf(sr.ci_low) or math.isinf(sr.ci_high))
    ):
        parts.append(
            f'<line x1="{x(sr.ci_low):.1f}" x2="{x(sr.ci_high):.1f}" y1="9" y2="9" stroke="{color}" stroke-width="2"/>'
        )
    parts.append(f'<circle cx="{x(diff):.1f}" cy="9" r="4" fill="{color}"/>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(result: ComparisonResult) -> str:
    """Self-contained HTML report (inline CSS and SVG, light and dark)."""
    esc = html.escape
    label = _OUTCOME_LABELS.get(result.outcome, "Score")
    verdict = verdict_text(result)
    if result.is_dry_run:
        body = f"<pre>{esc(_dry_run_text(result))}</pre>"
        return _page("Prompt comparison (dry run)", body)

    if result.winner:
        title = f"{result.winner} wins"
        vclass = "win"
    elif result.comparisons and not result.enough_data:
        title = "Not enough data"
        vclass = "warn"
    else:
        title = "No significant winner"
        vclass = ""
    meta = [m.replace(" x ", " × ") for m in _meta(result)]

    # Variant table
    values: list[float] = []
    for v in result.variants.values():
        for val in (v.mean_score, v.ci_low, v.ci_high):
            if val is not None and not math.isinf(val):
                values.append(val)
    if result.is_proportion:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = (min(values), max(values)) if values else (0.0, 1.0)
    ci_head = f"{result.confidence_level:.0%} CI"
    rows = []
    for name, v in _rows(result):
        tag = ""
        if name == result.winner:
            tag = '<span class="tag win">winner</span>'
        elif name == result.control and result.comparison_mode == "control":
            tag = '<span class="tag">control</span>'
        score = v.mean_score if v.mean_score is not None else v.success_rate
        rows.append(
            "<tr>"
            f'<td class="name">{esc(name)}{tag}</td>'
            f"<td>{esc(_rate(result, score))}</td>"
            f"<td>{_bar_svg(score, v.ci_low, v.ci_high, vmin, vmax)}</td>"
            f"<td>{esc(_range(result, v.ci_low, v.ci_high, '–'))}</td>"
            f"<td>{esc(_latency(v))}</td>"
            f"<td>{esc(_money(v.total_cost_usd, v.cost_known))}</td>"
            f"<td>{esc(_money(v.cost_per_correct))}</td>"
            f"<td>{v.error_count}</td>"
            "</tr>"
        )
    variant_table = (
        '<div class="card"><table><thead><tr>'
        f"<th>Variant</th><th>{esc(label)}</th><th></th><th>{esc(ci_head)}</th>"
        "<th>Latency mean / p95</th><th>Cost</th><th>Cost per correct</th><th>Errors</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )

    comp_html = ""
    if result.comparisons:
        bounds = [0.0]
        for c in result.comparisons:
            for val in (c.difference, c.ci_low, c.ci_high):
                if val is not None and not math.isinf(val):
                    bounds.append(val)
        span = max(abs(min(bounds)), abs(max(bounds))) or 1.0
        span *= 1.1
        crow = []
        for c in result.comparisons:
            res = (
                '<span class="sig">significant</span>'
                if c.significant
                else '<span class="ns">not significant</span>'
            )
            holm = f"<td>{esc(_p(c.adjusted_p))}</td>" if result.correction else ""
            crow.append(
                "<tr>"
                f'<td class="name">{esc(str(c.treatment))} vs {esc(str(c.control))}</td>'
                f"<td>{esc(_diff(result, c.difference, '−'))}</td>"
                f"<td>{_forest_svg(c, -span, span)}</td>"
                f"<td>{esc(_ci(result, c, '−'))}</td>"
                f"<td>{esc(_p(c.p_value))}</td>{holm}<td>{res}</td>"
                "</tr>"
            )
        holm_head = "<th>Holm p</th>" if result.correction else ""
        comp_html = (
            "<h2>Paired comparisons</h2>"
            '<div class="card"><table><thead><tr>'
            f"<th>Comparison</th><th>Difference</th><th></th><th>{esc(ci_head)}</th>"
            f"<th>p</th>{holm_head}<th>Result</th></tr></thead><tbody>"
            + "".join(crow)
            + "</tbody></table></div>"
        )

    foot: list[str] = []
    method = method_text(result)
    if method:
        foot.append(f"<p>{esc(method)}</p>")
    plan = _plan_line(result)
    if plan:
        foot.append(f"<p><strong>Planning.</strong> {esc(plan)}</p>")
    if result.cost_known:
        total_calls = sum(v.llm_calls for v in result.variants.values())
        foot.append(
            f"<p>Total cost {esc(_money(result.total_cost_usd))} for {total_calls} LLM calls.</p>"
        )
    if result.notes:
        foot.append(
            "<ul>" + "".join(f"<li>{esc(n)}</li>" for n in result.notes) + "</ul>"
        )

    body = (
        '<p class="eyebrow">Prompt comparison</p>'
        f"<h1>{esc(title)}</h1>"
        f'<p class="meta">{esc(" · ".join(meta))}</p>'
        f'<div class="verdict {vclass}"><strong>{esc(verdict)}</strong></div>'
        f"<h2>Variants</h2>{variant_table}{comp_html}"
        f"<footer>{''.join(foot)}<p>Generated by FlowPrompt.</p></footer>"
    )
    return _page(f"Prompt comparison: {title}", body)


def _page(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
        f"<body><main>{body}</main></body></html>\n"
    )
