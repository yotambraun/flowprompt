"""How often does a prompt A/B test declare a false winner?

Offline Monte Carlo study (no API calls). We simulate evaluation sets where
every prompt variant has the SAME true accuracy and count how often each
analysis method still declares a winner. A well-calibrated method should do
so at most 5% of the time (alpha = 0.05). We also measure power when one
variant really is better.

Methods compared (all call FlowPrompt's shipped inference functions):

  old_single    FlowPrompt <= 0.3.0 compare(): pooled two-proportion z-test
                (flowprompt.testing.run_significance_test(test_type="z_test")),
                first variant as control, old winner rule, ONE run per input.
  old_repeats   The same, with every repeated run counted as a sample.
  paired_raw    Paired, input-level test (exact McNemar / sign-flip), but no
                multiple-comparison correction (any raw p < 0.05 wins).
  new           FlowPrompt's compare(): paired, input-level, Holm-adjusted
                (flowprompt.testing.paired_test / paired_sign_flip_test +
                flowprompt.testing.holm_adjust, winner rule of compare()).

Data-generating model (realistic for LLM evals): input i has a latent
difficulty d_i ~ N(0, 1.5) on the logit scale, shared by all variants (hard
inputs are hard for every prompt). Variant v adds its own item-level noise
e_vi ~ N(0, 0.5) and a shift delta_v (0 under the null). The probability of
a correct answer is p_vi = sigmoid(mu + d_i + e_vi + delta_v), mu = 1.0
(about 70% accuracy).

  stochastic     temperature > 0: every run is a fresh Bernoulli(p_vi) draw.
  deterministic  temperature = 0: the answer for (v, i) is drawn once and
                 every repeat returns the same answer.

Usage:
    python benchmarks/false_winners.py            # full study (several minutes)
    python benchmarks/false_winners.py --quick    # smoke run (~1 min)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import flowprompt
from flowprompt import compare
from flowprompt.testing import (
    VariantStats,
    holm_adjust,
    paired_sign_flip_test,
    paired_test,
    run_significance_test,
    wilson_interval,
)

ALPHA = 0.05
MU = 1.0
SD_ITEM = 1.5
SD_NOISE = 0.5

METHODS = ("old_single", "old_repeats", "paired_raw", "new")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def simulate(
    rng: random.Random,
    n_inputs: int,
    k: int,
    runs: int,
    mode: str,
    shifts: list[float],
) -> list[list[list[int]]]:
    """Return scores[v][i] = list of 0/1 outcomes over runs."""
    difficulty = [rng.gauss(0.0, SD_ITEM) for _ in range(n_inputs)]
    scores: list[list[list[int]]] = []
    for v in range(k):
        per_input = []
        for i in range(n_inputs):
            p = sigmoid(MU + difficulty[i] + rng.gauss(0.0, SD_NOISE) + shifts[v])
            if mode == "deterministic":
                outcome = 1 if rng.random() < p else 0
                per_input.append([outcome] * runs)
            else:
                per_input.append([1 if rng.random() < p else 0 for _ in range(runs)])
        scores.append(per_input)
    return scores


# ---------------------------------------------------------------------------
# Old method: FlowPrompt <= 0.3.0 compare() decision rule
# ---------------------------------------------------------------------------


def _stats(name: str, per_input: list[list[int]], use_runs: int | None) -> VariantStats:
    outcomes = [
        x for runs in per_input for x in (runs if use_runs is None else runs[:use_runs])
    ]
    successes = sum(outcomes)
    return VariantStats(
        name=name,
        samples=len(outcomes),
        successes=successes,
        success_rate=successes / len(outcomes),
    )


def old_winner(scores: list[list[list[int]]], use_runs: int | None) -> str | None:
    """Replicates compare._build_result from FlowPrompt 0.3.0 (git c9e3cc7)."""
    names = [f"v{v}" for v in range(len(scores))]
    stats = [_stats(n, s, use_runs) for n, s in zip(names, scores, strict=True)]
    control = stats[0]
    best_name = None
    best_effect = 0.0
    best_result = None
    for name, treatment in zip(names[1:], stats[1:], strict=True):
        result = run_significance_test(
            control, treatment, test_type="z_test", confidence_level=0.95
        )
        if best_result is None or (
            result.significant and result.effect_size > best_effect
        ):
            best_result = result
            best_effect = result.effect_size
            best_name = name
    if best_result is not None and best_result.significant:
        return best_name if best_result.effect_size > 0 else names[0]
    return None


# ---------------------------------------------------------------------------
# New method: FlowPrompt compare() (paired, input-level, Holm)
# ---------------------------------------------------------------------------


def paired_p_and_diff(
    control: list[list[int]], treatment: list[list[int]]
) -> tuple[float, float]:
    """p-value and difference exactly as compare() computes them.

    One run per input: paired_test (exact McNemar). Repeated runs: the p-value
    paired_test uses, i.e. paired_sign_flip_test on per-input mean
    differences (skipping paired_test's bootstrap interval, which does not
    affect the decision and dominates the runtime).
    """
    if len(control[0]) == 1:
        sr = paired_test(control, treatment)
        return sr.p_value, float(sr.difference or 0.0)
    diffs = [
        sum(t) / len(t) - sum(c) / len(c)
        for c, t in zip(control, treatment, strict=True)
    ]
    p, _how, _err = paired_sign_flip_test(diffs)
    return p, sum(diffs) / len(diffs)


def new_winner(scores: list[list[list[int]]], correct: bool = True) -> str | None:
    """Winner rule of compare() in 'control' mode (see compare._analyse)."""
    names = [f"v{v}" for v in range(len(scores))]
    results = [paired_p_and_diff(scores[0], scores[v]) for v in range(1, len(scores))]
    p_values = [p for p, _ in results]
    decision_p = holm_adjust(p_values) if correct and len(p_values) > 1 else p_values
    better = [
        (diff, names[j + 1])
        for j, ((_, diff), p) in enumerate(zip(results, decision_p, strict=True))
        if p < ALPHA and diff > 0
    ]
    worse = [
        diff
        for (_, diff), p in zip(results, decision_p, strict=True)
        if p < ALPHA and diff < 0
    ]
    if better:
        return max(better)[1]
    if worse and len(worse) == len(results):
        return names[0]
    return None


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


def rate_summary(hits: int, reps: int) -> dict[str, float]:
    rate = hits / reps
    lo, hi = wilson_interval(hits, reps, 0.95)
    return {
        "hits": hits,
        "reps": reps,
        "rate": rate,
        "mc_se": math.sqrt(rate * (1 - rate) / reps),
        "ci_low": lo,
        "ci_high": hi,
    }


def seed_for(*parts: Any) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def null_cell(n_inputs: int, k: int, runs: int, mode: str, reps: int) -> dict[str, Any]:
    rng = random.Random(seed_for("null", n_inputs, k, runs, mode))
    hits = dict.fromkeys(METHODS, 0)
    discordance = 0.0
    for _ in range(reps):
        scores = simulate(rng, n_inputs, k, runs, mode, [0.0] * k)
        if old_winner(scores, use_runs=1) is not None:
            hits["old_single"] += 1
        if old_winner(scores, use_runs=None) is not None:
            hits["old_repeats"] += 1
        if new_winner(scores, correct=False) is not None:
            hits["paired_raw"] += 1
        if new_winner(scores, correct=True) is not None:
            hits["new"] += 1
        discordance += (
            sum(1 for c, t in zip(scores[0], scores[1], strict=True) if c[0] != t[0])
            / n_inputs
        )
    return {
        "n_inputs": n_inputs,
        "k": k,
        "runs": runs,
        "mode": mode,
        "mean_discordance": discordance / reps,
        **{m: rate_summary(hits[m], reps) for m in METHODS},
    }


def calibrate_shift(target_diff: float) -> float:
    """Logit shift giving the target population accuracy difference."""
    rng = random.Random(12345)
    draws = [
        MU + rng.gauss(0.0, SD_ITEM) + rng.gauss(0.0, SD_NOISE) for _ in range(200_000)
    ]

    def accuracy(shift: float) -> float:
        return sum(sigmoid(x + shift) for x in draws) / len(draws)

    base = accuracy(0.0)
    if target_diff == 0:
        return 0.0
    lo, hi = 0.0, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if accuracy(mid) - base < target_diff:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def base_accuracy() -> float:
    rng = random.Random(12345)
    draws = [
        MU + rng.gauss(0.0, SD_ITEM) + rng.gauss(0.0, SD_NOISE) for _ in range(200_000)
    ]
    return sum(sigmoid(x) for x in draws) / len(draws)


def power_cell(diff_points: float, n_inputs: int, reps: int) -> dict[str, Any]:
    shift = calibrate_shift(diff_points / 100)
    rng = random.Random(seed_for("power", diff_points, n_inputs))
    hits = {"old_single": 0, "new": 0}
    for _ in range(reps):
        scores = simulate(rng, n_inputs, 2, 1, "stochastic", [0.0, shift])
        # Power = correctly naming the better variant (v1).
        if old_winner(scores, use_runs=1) == "v1":
            hits["old_single"] += 1
        if new_winner(scores) == "v1":
            hits["new"] += 1
    return {
        "true_difference_points": diff_points,
        "logit_shift": shift,
        "n_inputs": n_inputs,
        **{m: rate_summary(hits[m], reps) for m in hits},
    }


def verify_against_compare(reps: int) -> dict[str, Any]:
    """Check that the direct path matches flowprompt.compare() end to end."""
    checked = 0
    agree = 0
    for n_inputs, k, runs, mode in [
        (20, 2, 1, "stochastic"),
        (20, 3, 3, "stochastic"),
        (50, 5, 1, "deterministic"),
    ]:
        rng = random.Random(seed_for("verify", n_inputs, k, runs, mode))
        for _ in range(reps):
            scores = simulate(rng, n_inputs, k, runs, mode, [0.0] * k)
            direct = new_winner(scores)
            answers = {}
            for v in range(k):
                # Each variant replays its simulated outcomes, one per call.
                queue = {i: list(scores[v][i]) for i in range(n_inputs)}

                def variant(
                    inp: dict[str, Any], q: dict[int, list[int]] = queue
                ) -> str:
                    return "right" if q[inp["i"]].pop(0) else "wrong"

                answers[f"v{v}"] = variant
            result = compare(
                answers,
                inputs=[{"i": i} for i in range(n_inputs)],
                expected=["right"] * n_inputs,
                eval_metric="exact",
                runs_per_input=runs,
            )
            checked += 1
            agree += result.winner == direct
    return {"checked": checked, "agree": agree}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="small smoke run")
    parser.add_argument("--reps", type=int, default=None, help="replications per cell")
    parser.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = parser.parse_args(argv)

    reps = args.reps or (200 if args.quick else 4000)
    power_reps = args.reps or (200 if args.quick else 4000)
    verify_reps = 10 if args.quick else 50
    sizes = [20, 100] if args.quick else [20, 50, 100, 200]

    started = time.time()
    cells = []
    for n_inputs in sizes:
        for k in (2, 3, 5):
            for runs, mode in [
                (1, "stochastic"),
                (3, "stochastic"),
                (5, "stochastic"),
                (3, "deterministic"),
                (5, "deterministic"),
            ]:
                cell = null_cell(n_inputs, k, runs, mode, reps)
                cells.append(cell)
                print(
                    f"N={n_inputs:>3} k={k} runs={runs} {mode:<13} "
                    + "  ".join(f"{m}={cell[m]['rate']:.3f}" for m in METHODS),
                    flush=True,
                )

    power = []
    for diff in (0, 2.5, 5, 7.5, 10, 12.5, 15):
        cell = power_cell(diff, 100, power_reps)
        power.append(cell)
        print(
            f"power diff={diff:>4} old={cell['old_single']['rate']:.3f} new={cell['new']['rate']:.3f}",
            flush=True,
        )

    verification = verify_against_compare(verify_reps)
    print(
        f"compare() agreement: {verification['agree']}/{verification['checked']}",
        flush=True,
    )
    runtime = time.time() - started

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "description": "False-winner rate of prompt A/B analyses when all variants are equally good.",
            "alpha": ALPHA,
            "model": {
                "mu_logit": MU,
                "sd_item_logit": SD_ITEM,
                "sd_variant_noise_logit": SD_NOISE,
                "base_accuracy": base_accuracy(),
            },
            "reps_per_cell": reps,
            "power_reps": power_reps,
            "quick": args.quick,
            "runtime_seconds": round(runtime, 1),
            "flowprompt_version": flowprompt.__version__,
            "python": platform.python_version(),
            "command": "python benchmarks/false_winners.py"
            + (" --quick" if args.quick else ""),
        },
        "false_winner": cells,
        "power": power,
        "compare_verification": verification,
    }
    (out / "false_winners.json").write_text(json.dumps(payload, indent=2) + "\n")

    with (out / "false_winners.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "table",
                "n_inputs",
                "k",
                "runs",
                "mode",
                "true_difference_points",
                "method",
                "rate",
                "mc_se",
                "ci_low",
                "ci_high",
                "reps",
            ]
        )
        for c in cells:
            for m in METHODS:
                r = c[m]
                writer.writerow(
                    [
                        "false_winner",
                        c["n_inputs"],
                        c["k"],
                        c["runs"],
                        c["mode"],
                        0,
                        m,
                        f"{r['rate']:.4f}",
                        f"{r['mc_se']:.4f}",
                        f"{r['ci_low']:.4f}",
                        f"{r['ci_high']:.4f}",
                        r["reps"],
                    ]
                )
        for c in power:
            for m in ("old_single", "new"):
                r = c[m]
                writer.writerow(
                    [
                        "power",
                        c["n_inputs"],
                        2,
                        1,
                        "stochastic",
                        c["true_difference_points"],
                        m,
                        f"{r['rate']:.4f}",
                        f"{r['mc_se']:.4f}",
                        f"{r['ci_low']:.4f}",
                        f"{r['ci_high']:.4f}",
                        r["reps"],
                    ]
                )

    print(f"Wrote {out / 'false_winners.json'} in {runtime:.0f}s")
    return 0 if verification["agree"] == verification["checked"] else 1


if __name__ == "__main__":
    sys.exit(main())
