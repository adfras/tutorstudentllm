# Operations — Runs, Aggregation, and Insights

This guide explains how run logs are organized, how we aggregate them into a single database, and what we’re trying to achieve with the experiments.

## Goals

- Improve credited performance (correct + evidence) by aligning USE citations to the chosen option’s PRO card.
- Keep a single, reproducible database of all historical runs to compare dials and models over time.
- Derive model‑specific defaults that lift credited with reasonable cost/latency.

Key concept: credited closely tracks witness (the option most supported by citations). Fix witness → lift credited.

## Where Runs Live

- Raw logs: `runs/` (JSONL); subfolders like `runs/holdout`, `runs/smoke`, `runs/compare`, `runs/window`, `runs/exp`.
- Aggregates: `runs/_aggregated/` (CSV/Parquet + insights).
- Archive (safe deletion): `runs/_archive/` (tar.gz + manifest of removed raw logs).
- Catalog: `runs/_catalog/catalog.csv` (one row per run; newest first).

## Make a Single Database (All Runs)

```bash
.venv/bin/python -m scripts.aggregate_runs --runs-dir runs --out-dir runs/_aggregated
```

Outputs under `runs/_aggregated/`:

- `steps.csv.gz` / `steps.parquet` — master step‑level table (with provider/model/dials/evidence/timing/tokens/provenance).
- `runs_summary.csv.gz` / `runs_summary.parquet` — one row per run.
- `model_ranking.csv.gz` — per‑model means.
- `by_model_steps.csv.gz`, `by_model_runs.csv.gz` — per‑model rollups.
- `plots/INSIGHTS.md` — quick correlations + dial tables.

Optional helpers:

```bash
# Parquet exports
.venv/bin/python -m scripts.export_parquet --agg-dir runs/_aggregated

# One‑row‑per‑run catalog
.venv/bin/python -m scripts.catalog --agg-dir runs/_aggregated --out runs/_catalog/catalog.csv

# Component analysis (per‑model dial effects)
.venv/bin/python -m scripts.component_analysis --agg-dir runs/_aggregated --out-dir runs/_aggregated/analysis --filter-apples

# Best per‑model row by credited_final
.venv/bin/python -m scripts.best_settings --agg-dir runs/_aggregated --out runs/_aggregated/analysis/best_settings.csv --apples
```

## What We Measure (MCQ)

Each MCQ step logs:

- `evaluation.correct` — boolean.
- `evaluation.citations_evidence` → `{coverage, witness_pass, credited, reasons[]}`.
  - coverage — token overlap between cited quotes and the gold option;
  - witness_pass — cited text supports the correct option more than others;
  - credited — `correct && citations && coverage ≥ τ && witness_pass`.

Pipeline guarantees:

- LEARN: ensures a ≤15‑token PRO card exists for each option (sliced from the option text if missing) and reserves these first in the budget.
- USE: JSON spec requires the CHOSEN option’s FIRST citation to be its PRO card; abstain (IDK) on hard‑evidence failure.

## Current Findings (from your data)

- Witness → Credited is the lever (improving witness increases credited across runs).
- Budget ≈ 6 is a safe default; bigger budgets only help sometimes.
- Self‑consistency is model‑specific:
  - DeepSeek‑R1: SC=3 benefits from “deliberate” runs (more time/tokens helps).
  - Mixtral‑8x7B: SC=3 stable; SC=1 can spike but is noisier; b=6 default.
  - GPT‑4.1: In recent smokes, SC=1 with b=6 did well.

## Small, Fast Sweeps

```bash
# FAST=1 → steps=10, Mixtral only; FAST=0 → steps=30, adds GPT‑4.1
FAST=1 STEPS=10 BUDGETS="6 8" SC_LIST="1 3" bash scripts/sweep_minisuite.sh
```

## Phase 2 "Scattergun" Matrix

Use `scripts/run_scattergun_phase2.sh` to launch breadth (cheap, short) and depth (longer) sessions across the Phase 2 model roster (JSON-stable, long-context, sensible cost/token).

```bash
chmod +x scripts/run_scattergun_phase2.sh
DRY_RUN=1 PARALLEL_JOBS=4 SEED_BASE=424242 bash scripts/run_scattergun_phase2.sh
PARALLEL_JOBS=4 SEED_BASE=424242 bash scripts/run_scattergun_phase2.sh

> Already-captured logs are skipped automatically. To restart later, stop the batch (`pkill -f run_scattergun_phase2.sh`) and rerun the command above; only missing logs will resume. Delete a partial JSONL if you want that run redone from step 0.
```

The script auto-detects the latest guardrail and talk-slope tables under `runs/_aggregated/bayes/tables/`, logs to `runs/scattergun_phase2/<stage>/`, and uses three stages:

- `breadth` — 12 steps, 3 seeds, `--best-of 4`, `--sc-extract 2`, `--cards-budget 8`, lean cost for quick model scoring.
- `depth` — 40 steps, 2 seeds, `--best-of 8`, `--sc-extract 3`, `--cards-budget 12`, `--max-learn-boosts 1` to retain retrieved snippets.
- `long` — 80 steps, 1 seed, `--best-of 10`, `--sc-extract 4`, `--cards-budget 14`, captures trough behavior over long contexts.

Every run pins `ANON_SEED`, enforces strict evidence, and leaves `--idk` enabled so models abstain instead of guessing when evidence is missing. Toggle stages/models by editing the arrays near the top of the script.

Automation:

```bash
# Run jobs and wait for them to finish, then aggregate & summarize
.venv/bin/python -m scripts.wait_for_runs --pids <PID...> --logs <run.jsonl...>
```

## Cleanup (Safe Archival)

```bash
# DRY RUN — see which files would be archived/removed
.venv/bin/python -m scripts.cleanup_runs --min-age-minutes 10 --out-dir runs/_archive --dry-run
# Archive + remove old raw logs (aggregates remain)
.venv/bin/python -m scripts.cleanup_runs --min-age-minutes 10 --out-dir runs/_archive
```

The manifest lists exactly which files were archived. You can extract any file from the tarball later.
