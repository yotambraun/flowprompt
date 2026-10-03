# Roadmap: statistically valid experiments for prompts and models

Internal design note. Not linked from the README and excluded from the docs
site (`exclude_docs` in `mkdocs.yml`).

## Positioning

FlowPrompt's distinctive job is to answer "is B really better than A, by how
much, and what does it cost?" for prompts, models and LLM pipelines, with
statistics a reviewer can check. Typed prompts, caching and tracing are the
supporting cast.

## Status after 0.5.0

| Item | Status | Where |
|---|---|---|
| Compare any callables / models, not only Prompt subclasses | Done | `testing/variants.py`; `compare()` accepts callables, `(Prompt, "model")`, `PromptVariant`, `model_variants()` |
| Paired tests, input as unit of analysis, Holm | Done | `testing/paired.py`, `testing/compare.py` |
| Sample-size / power planner ("you need ~N inputs, about $Y") | Done | `plan_sample_size()`, `ComparisonResult.sample_size_plan()`, planning line in reports |
| Cost per correct answer, latency per variant | Done | `VariantResult.cost_per_correct`, `p95_latency_ms`; metered through `track_usage()` |
| Markdown / HTML reports | Done | `testing/report.py`, `save_report()` |
| CI gate | Done | `flowprompt compare --fail-on-regression`, documented Actions workflow (`docs/ci.md`) |
| Sequential testing for paired offline evals | Done (standalone) | `SequentialMcNemar`; not yet wired into `compare()` |
| Sequential testing for live traffic | Next | see 1 |
| GitHub Action that comments on PRs | Next | see 4 |
| Lighter core without litellm | Next | see 5 |

## Next

### 1. Always-valid tests for live traffic (`ABTestRunner`)

Problem: the runner re-tests after every result once `min_samples` is
reached, which is optional stopping and inflates the false-positive rate
(documented as a caveat in 0.5.0).

Plan: replace the auto-completion rule with a mixture sequential
probability ratio test for two independent proportions (normal-mixture
mSPRT, Johari, Koomen, Pekelis and Walsh, *Operations Research* 2022) or
an asymptotic confidence sequence for the difference in means (Waudby-Smith,
Arnold, Wu, Wu and Ramdas, "Time-uniform central limit theory and
asymptotic confidence sequences", *Annals of Statistics* 2024). Both have
closed forms, so no new dependencies. For several treatments, apply
Bonferroni to the e-values or use e-BH (Wang and Ramdas, *JRSS-B* 2022).

Feasibility: high. Tests: known values of the mixture statistic, and a
simulation showing a false-positive rate at or below alpha under continuous
monitoring (the same pattern as `test_sequential_false_positive_rate_*`).

### 2. Early stopping inside `compare()`

`compare(..., stop_early=True)` would evaluate inputs in interleaved batches
across variants and stop when `SequentialMcNemar` (pass/fail) or a
betting-based confidence sequence for bounded scores (Waudby-Smith and
Ramdas, *JRSS-B* 2024) rejects. The report must state that a sequential
test was used and show the always-valid p-value. Cost saving is the selling
point: with a clear winner, a 400-input run can stop after a few dozen.

### 3. Non-inferiority gate for CI

`--fail-on-regression` fails only when the candidate is *significantly*
worse. On a small dataset that can pass a real regression ("no evidence"
is not "evidence of none"). Add `--non-inferiority-margin 0.02`: pass only
if the lower confidence bound of (candidate - current) is above -2 points.
For paired pass/fail data use the score-based non-inferiority test for
matched pairs (Nam, *Biometrics* 1997; Tango, *Statistics in Medicine*
1998). Pair it with the planner so teams pick a dataset size that can
actually demonstrate non-inferiority.

### 4. A GitHub Action

A small composite action (`yotambraun/flowprompt-action`) wrapping
`flowprompt compare`: installs the package, runs the comparison, writes the
job summary, and creates or updates a single sticky PR comment (needs
`pull-requests: write`). Inputs: variants file, dataset, model, metric,
control, margin, fail mode. Today the same result takes about 15 lines of
YAML (`docs/ci.md`).

### 5. A lighter core without litellm

litellm takes about 2 s to import and pulls a large dependency tree;
importing `flowprompt` itself takes about 0.15 s because litellm is loaded
lazily. Moving litellm to an extra would break `pip install flowprompt-ai`
for existing users, so it was not done in 0.5.0.

Plan: (a) 0.6: define a small `Provider` protocol (already implied by
`BaseProvider`, `FakeLLM` and callable variants) and accept
`provider=` on `run()` / `compare()`; add an `openai`-SDK-only provider for
users who want fewer dependencies. (b) 1.0: move litellm to an extra
(`flowprompt-ai[litellm]`, included in `[all]`) with a clear error message
pointing to it, announced one minor release ahead in the changelog.

### 6. Per-input diff view and stored runs

The most useful thing after "which prompt wins" is "on which inputs did it
flip". Add a per-input table (inputs where the variants disagree, with
outputs) to the HTML report, and `ComparisonResult.save_runs("runs.jsonl")`
plus `analyze(runs)` to re-analyse (different scorer, different control)
without paying for the calls again.

### 7. Clustered and stratified evaluation sets

Real eval sets have structure (several questions per document, several
turns per conversation). Accept `group=` per input and resample groups in
the bootstrap and sign-flip test (cluster-robust paired analysis, as
recommended for LLM evals by Miller, "Adding Error Bars to Evals", 2024).
Report per-slice results with Holm across slices.

### 8. Cost-aware decisions

Report a confidence interval for cost per correct answer (ratio estimator
with a paired bootstrap) and optional decision rules such as "prefer the
cheaper variant unless the other is better by at least X points".

## Competitive check (statistics for comparing prompt variants)

Checked 2026-10-03 by reading each project's source (shallow clones of the
default branch, HEADs dated 2026-10-01 to 2026-10-03) and official docs.
Legend: (a) side-by-side aggregates, (b) interval or standard error for a
single variant, (c) hypothesis test between variants, (d) paired test.

| Project (version read) | What it offers | Paired significance test? | Evidence |
|---|---|---|---|
| promptfoo 0.123.1 | (a) results matrix, compare view, head-to-head plot, `--repeat` | No | `site/docs/usage/web-ui.md`; https://promptfoo.dev/docs/usage/web-ui |
| DSPy (ba3f919) | `Evaluate` mean score; MIPROv2 minibatch + periodic full evals, picks the max | No | `dspy/teleprompt/mipro_optimizer_v2.py`, `dspy/evaluate/` |
| LangSmith | (a) comparison view with improved/regressed counts; pairwise experiments | No | https://docs.langchain.com/langsmith/compare-experiment-results, https://docs.langchain.com/langsmith/evaluate-pairwise |
| Langfuse 4.50.0 | (a) baseline vs comparison deltas, per-item improved/regressed counts; score analytics for judge agreement only | No | `web/src/features/experiments/fns/summariseScoreColumn.ts`, `web/src/features/score-analytics/lib/statistics-utils.ts` |
| DeepEval 4.2.8 | `compare()` with `ArenaGEval` returns win counts | No | `deepeval/evaluate/compare.py` |
| pydantic-evals (pydantic-ai f06bba5) | (a) `report.print(baseline=...)` diff table, `repeat=` | No | `pydantic_evals/pydantic_evals/reporting/__init__.py` |
| inspect-ai (93f7182) | (b) `stderr(cluster=)`, `bootstrap_stderr()`, `ci()`, `ci_wilson()`; docs suggest comparing interval overlap | No (per-variant intervals only) | `src/inspect_ai/scorer/_metrics/std.py`, `docs/metrics.qmd` |
| Mirascope 2.5.0 | `llm` and `ops` modules; no eval module in v2 | No | `python/mirascope/` |
| Opik 2.2.89 | (a) experiment comparison page; Spearman / JS-divergence only as metrics | No | `apps/opik-frontend/src/v2/pages/CompareExperimentsPage/`, `sdks/python/src/opik/evaluation/metrics/heuristics/spearman.py` |

None corrects for multiple comparisons, plans sample sizes or offers
sequential tests. Small standalone PyPI packages (abeval, EVALSIG,
evalstats) reportedly offer paired tests; not verified. Caveat: LangSmith,
Confident AI and Opik Cloud backends are closed source; only their docs and
open-source frontends were checked.

## Literature relied on

McNemar (1947); Holm (1979); Efron (1987); Connor (1987); Agresti and Min
(2005); Fagerland, Lydersen and Laake (2013); Robbins (1970); Howard,
Ramdas, McAuliffe and Sekhon (2021); Johari, Koomen, Pekelis and Walsh
(2022); Wang and Ramdas (2022); Waudby-Smith and Ramdas (2024);
Waudby-Smith, Arnold, Wu, Wu and Ramdas (2024); Nam (1997); Tango (1998);
Dror, Baumer, Shlomov and Reichart (2018); Miller (2024).
