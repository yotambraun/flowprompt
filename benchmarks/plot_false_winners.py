"""Render the false-winner study charts as standalone SVG (stdlib only).

    python benchmarks/plot_false_winners.py  # reads benchmarks/results/false_winners.json

Writes docs/assets/false-winners.svg and docs/assets/power.svg.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

INK = "#1d1d1f"
MUTED = "#6b6b70"
GRID = "#e7e7e4"
OLD = "#d0803a"  # warm, muted: FlowPrompt <= 0.3.0
NEW = "#2f6fde"  # accent: current FlowPrompt
BG = "#ffffff"
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def _text(
    x: float,
    y: float,
    s: str,
    *,
    size: int = 13,
    fill: str = INK,
    anchor: str = "start",
    weight: int = 400,
    halo: bool = False,
) -> str:
    extra = (
        f' stroke="{BG}" stroke-width="5" stroke-linejoin="round" paint-order="stroke"'
        if halo
        else ""
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{extra}>'
        f"{escape(s)}</text>"
    )


def _find(cells: list[dict[str, Any]], **want: Any) -> dict[str, Any]:
    for c in cells:
        if all(c[k] == v for k, v in want.items()):
            return c
    raise KeyError(want)


def false_winner_chart(data: dict[str, Any], n_inputs: int) -> str:
    cells = data["false_winner"]
    scenarios = [
        (
            "2 prompts, 1 run each",
            {"k": 2, "runs": 1, "mode": "stochastic"},
            "old_single",
        ),
        (
            "2 prompts, 5 runs (temperature > 0)",
            {"k": 2, "runs": 5, "mode": "stochastic"},
            "old_repeats",
        ),
        (
            "2 prompts, 5 runs (temperature 0)",
            {"k": 2, "runs": 5, "mode": "deterministic"},
            "old_repeats",
        ),
        (
            "5 prompts, 1 run each",
            {"k": 5, "runs": 1, "mode": "stochastic"},
            "old_single",
        ),
        (
            "5 prompts, 5 runs (temperature 0)",
            {"k": 5, "runs": 5, "mode": "deterministic"},
            "old_repeats",
        ),
    ]
    width, left, right = 900, 300, 70
    top, row_h, bar_h, gap = 134, 58, 16, 4
    height = top + row_h * len(scenarios) + 70
    plot_w = width - left - right
    max_rate = 0.0
    for _, sel, old_key in scenarios:
        c = _find(cells, n_inputs=n_inputs, **sel)
        max_rate = max(max_rate, c[old_key]["ci_high"], c["new"]["ci_high"])
    x_max = max(0.1, min(1.0, (int(max_rate * 10) + 1) / 10))

    def x(rate: float) -> float:
        return left + plot_w * rate / x_max

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{BG}"/>',
        _text(
            32,
            40,
            "How often an A/B test crowns a winner between equally good prompts",
            size=19,
            weight=650,
        ),
        _text(
            32,
            64,
            f"Simulated evaluations, {n_inputs} inputs, true accuracy identical for every "
            f"prompt. {data['meta']['reps_per_cell']:,} experiments per bar; "
            "whiskers are 95% intervals.",
            size=13,
            fill=MUTED,
        ),
    ]
    # Legend
    lx = 32
    for color, label in (
        (OLD, "FlowPrompt 0.3.0 (unpaired z-test)"),
        (NEW, "FlowPrompt now (paired, per input, Holm)"),
    ):
        out.append(
            f'<rect x="{lx}" y="80" width="12" height="12" rx="2" fill="{color}"/>'
        )
        out.append(_text(lx + 18, 90.5, label, size=12.5, fill=INK))
        lx += 18 + len(label) * 7.0 + 26
    # Grid
    ticks = [i / 10 for i in range(0, int(round(x_max * 10)) + 1)]
    plot_bottom = top + row_h * len(scenarios) - 6
    for t in ticks:
        out.append(
            f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 10}" y2="{plot_bottom}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(
            _text(
                x(t), plot_bottom + 18, f"{t:.0%}", size=12, fill=MUTED, anchor="middle"
            )
        )
    out.append(
        _text(
            left + plot_w / 2,
            plot_bottom + 40,
            "false-winner rate (target: at most 5%)",
            size=12.5,
            fill=MUTED,
            anchor="middle",
        )
    )
    # 5% line
    out.append(
        f'<line x1="{x(0.05):.1f}" x2="{x(0.05):.1f}" y1="{top - 10}" y2="{plot_bottom}" '
        f'stroke="{INK}" stroke-width="1.2" stroke-dasharray="4 3"/>'
    )
    out.append(_text(x(0.05), top - 16, "5%", size=11.5, fill=INK, anchor="middle"))

    for row, (label, sel, old_key) in enumerate(scenarios):
        c = _find(cells, n_inputs=n_inputs, **sel)
        y0 = top + row * row_h
        out.append(
            _text(left - 14, y0 + bar_h + gap / 2 + 4, label, size=13.5, anchor="end")
        )
        for j, (key, color) in enumerate(((old_key, OLD), ("new", NEW))):
            r = c[key]
            y = y0 + j * (bar_h + gap)
            out.append(
                f'<rect x="{left}" y="{y}" width="{max(x(r["rate"]) - left, 1.5):.1f}" '
                f'height="{bar_h}" rx="3" fill="{color}"/>'
            )
            ym = y + bar_h / 2
            out.append(
                f'<line x1="{x(r["ci_low"]):.1f}" x2="{x(r["ci_high"]):.1f}" y1="{ym}" y2="{ym}" '
                f'stroke="{INK}" stroke-width="1.2" stroke-opacity="0.65"/>'
            )
            for edge in (r["ci_low"], r["ci_high"]):
                out.append(
                    f'<line x1="{x(edge):.1f}" x2="{x(edge):.1f}" y1="{ym - 4}" y2="{ym + 4}" '
                    f'stroke="{INK}" stroke-width="1.2" stroke-opacity="0.65"/>'
                )
            out.append(
                _text(
                    x(r["ci_high"]) + 8,
                    ym + 4.5,
                    f"{r['rate'] * 100:.1f}%",
                    size=12.5,
                    fill=INK,
                    weight=600 if key == "new" else 500,
                    halo=True,
                )
            )
    out.append("</svg>")
    return "\n".join(out)


def power_chart(data: dict[str, Any]) -> str:
    cells = data["power"]
    n_inputs = cells[0]["n_inputs"]
    width, height = 900, 470
    left, right, top, bottom = 80, 70, 112, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    x_max = max(c["true_difference_points"] for c in cells)

    def x(d: float) -> float:
        return left + plot_w * d / x_max

    def y(p: float) -> float:
        return top + plot_h * (1 - p)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{BG}"/>',
        _text(
            32,
            40,
            "Power: how often the truly better prompt is declared the winner",
            size=19,
            weight=650,
        ),
        _text(
            32,
            64,
            f"Two prompts, {n_inputs} inputs, one run each; inputs share difficulty, as in "
            f"real evals. {data['meta']['power_reps']:,} experiments per point; bands are 95% intervals.",
            size=13,
            fill=MUTED,
        ),
    ]
    lx = 32
    for color, label in (
        (OLD, "FlowPrompt 0.3.0 (unpaired z-test)"),
        (NEW, "FlowPrompt now (exact McNemar)"),
    ):
        out.append(
            f'<rect x="{lx}" y="80" width="12" height="12" rx="2" fill="{color}"/>'
        )
        out.append(_text(lx + 18, 90.5, label, size=12.5))
        lx += 18 + len(label) * 7.0 + 26
    for p in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        out.append(
            f'<line x1="{left}" x2="{left + plot_w}" y1="{y(p):.1f}" y2="{y(p):.1f}" stroke="{GRID}"/>'
        )
        out.append(
            _text(left - 10, y(p) + 4, f"{p:.0%}", size=12, fill=MUTED, anchor="end")
        )
    for c in cells:
        d = c["true_difference_points"]
        out.append(
            _text(
                x(d), top + plot_h + 20, f"{d:g}", size=12, fill=MUTED, anchor="middle"
            )
        )
    out.append(
        _text(
            left + plot_w / 2,
            top + plot_h + 44,
            "true accuracy difference between the prompts (percentage points)",
            size=12.5,
            fill=MUTED,
            anchor="middle",
        )
    )
    for key, color in (("old_single", OLD), ("new", NEW)):
        upper = " ".join(
            f"{x(c['true_difference_points']):.1f},{y(c[key]['ci_high']):.1f}"
            for c in cells
        )
        lower = " ".join(
            f"{x(c['true_difference_points']):.1f},{y(c[key]['ci_low']):.1f}"
            for c in reversed(cells)
        )
        out.append(
            f'<polygon points="{upper} {lower}" fill="{color}" fill-opacity="0.16"/>'
        )
        line = " ".join(
            f"{x(c['true_difference_points']):.1f},{y(c[key]['rate']):.1f}"
            for c in cells
        )
        out.append(
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.4"/>'
        )
        for c in cells:
            out.append(
                f'<circle cx="{x(c["true_difference_points"]):.1f}" cy="{y(c[key]["rate"]):.1f}" '
                f'r="3.6" fill="{BG}" stroke="{color}" stroke-width="2"/>'
            )
    last = cells[-1]
    ys = {key: y(last[key]["rate"]) for key in ("new", "old_single")}
    if abs(ys["new"] - ys["old_single"]) < 16:
        mid = (ys["new"] + ys["old_single"]) / 2
        ys = {"new": mid - 8, "old_single": mid + 8}
    for key, color in (("new", NEW), ("old_single", OLD)):
        out.append(
            _text(
                x(last["true_difference_points"]) + 10,
                ys[key] + 4,
                f"{last[key]['rate'] * 100:.0f}%",
                size=12.5,
                fill=color,
                weight=600,
                halo=True,
            )
        )
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data", default=str(ROOT / "benchmarks/results/false_winners.json")
    )
    parser.add_argument("--out", default=str(ROOT / "docs/assets"))
    parser.add_argument("--n-inputs", type=int, default=100)
    args = parser.parse_args()
    data = json.loads(Path(args.data).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "false-winners.svg").write_text(false_winner_chart(data, args.n_inputs))
    (out / "power.svg").write_text(power_chart(data))
    print(f"Wrote {out / 'false-winners.svg'} and {out / 'power.svg'}")


if __name__ == "__main__":
    main()
