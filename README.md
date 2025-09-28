# ICL Simulator — Closed‑Book Tutor→Student Runs

A minimal, fast, general‑purpose In‑Context Learning (ICL) simulator. It runs anonymized, closed‑book tutor→student simulations using a fixed OpenAI tutor model and pluggable students (LLM via OpenAI/DeepInfra or an algorithmic baseline). Results stream as strict JSON and can be aggregated for learning‑curve analysis.

Start here: see the full handbook at `docs/ICL_Simulator_Handbook.md`.

Latest live evaluation and analysis (Sep 11, 2025): `docs/reports/2025-09-11_live_eval.md`.
Strict-credit quick guide (Sep 12, 2025): `docs/reports/2025-09-12_strict_credit_guide.md`.
Snippet coverage snapshot (Sep 17, 2025): `docs/reports/2025-09-17_kimi_meta_snapshot.md`.
Evidence credit pipeline (Sep 18, 2025): `docs/pipeline.md` (refresh models, score runs, surface dial recommendations offline).
Pinned seed guidance: see "Run Matrix" below for why we default to `ANON_SEED` values in the 424242 block.

Quick cost/latency presets are in the Handbook’s “Cost & Latency Playbook” (fast & cheap runs, zero‑cost offline loop, and a quick model shootout with Pareto ranking).

## What It Does

- Tutor generates short, abstract MCQs (extendable to SAQ/code/proof/table QA).
- Student answers using only provided context (closed-book); anonymization scrambles entities.
- Fixed tutor model: `gpt-5-nano-2025-08-07` with JSON responses.
- Swappable student: OpenAI, DeepInfra/DeepSeek, or an algorithmic baseline.
- Per-step JSONL logging; optional progress bar; optional self-consistency voting for MCQ.

## Architecture Overview

- **Orchestrator (`sim/orchestrator.py`)** – runs LEARN → USE loops, enforces evidence gates, applies guardrail nudges, and records every step as strict JSON.
- **Learners (`sim/learner.py`)** – adapters for stateful LLM students (OpenAI, DeepInfra, DeepSeek) and the deterministic algo/oracle baselines.
- **Evidence Pipeline (`sim/factcards.py`, `sim/factcards_manager.py`, `sim/evidence.py`, `sim/validation.py`)** – orchestrator now delegates normalization/budgeting to the Fact-Card manager before running coverage/witness checks and strict credit.
- **Prompt Builders (`sim/prompts_mcq.py`)** – reusable JSON scaffolds for MCQ prompts (fact-card grounding, schema enforcement) shared across learners.
- **Retrieval & Tools (`sim/retrieval.py`, `sim/tools.py`)** – TF-IDF and option-conditioned retrievers plus Program-of-Thought/pyexec hooks for tool-augmented answers.
- **Guardrails & Profiling (`scripts/bayes/…`, `scripts/model_profiles.py`)** – fit Bayesian token/talk bands, auto-generate per-model dial overrides, and keep the sweeps reproducible.
- **Aggregation (`scripts/aggregate_runs.py`, `scripts/session_view.py`, `scripts.analyze.py`)** – merge logs, compute learning curves (LAUC/STAB/WIT), and emit summaries for dashboards or CI comparisons.

## Install

```
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` with your keys as needed:

```
OPENAI_API_KEY=sk-...
DEEPINFRA_API_KEY=di-...
# Optional default OSS model
DEEPINFRA_MODEL=deepseek-ai/DeepSeek-R1
```

## Strict Credit Quick Start

- One‑liners (presets):
  - 4.1 strict: `scripts/presets/run_strict_41.sh`
  - Mixtral strict: `scripts/presets/run_strict_mixtral.sh`

Both write logs under `runs/strict/`. See the 2025‑09‑12 guide for details and expected metrics.

## Run Examples

- Offline mock mode has been removed. Run with real providers (OpenAI/DeepInfra) or use the built‑in `oracle` or `algo` students for deterministic offline checks (no external calls).

- Algorithmic baseline (closed‑book with notes):
  `python -m sim.cli --student algo --steps 5 --closed-book --notes-file data/notes_algo.txt`

- Live LLM student (DeepSeek via DeepInfra) with progress and logging:
  `unset TUTOR_MOCK_LLM && mkdir -p runs && .venv/bin/python -m sim.cli --student llm --provider deepinfra --model deepseek-ai/DeepSeek-R1 --steps 10 --closed-book --rich --progress --log runs/full_live.jsonl`

Output prints a compact JSON summary to stdout; detailed per‑step JSONL goes to `--log` if provided.

### Recommended Recipe (High‑Credit "state002")

This recipe reproduces a strong 20‑step session (observed 75% accuracy, 60% strict credit) by pinning the anonymization seed while keeping all flags fixed.

Run (DeepInfra Mixtral; identical flags, pinned seed):

```
ANON_SEED=424242 .venv/bin/python -m sim.cli \
  --student stateful-llm --provider deepinfra \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --steps 20 --closed-book --fact-cards --require-citations \
  --self-consistency 3 --best-of 8 --rerank evidence \
  --sc-extract 3 --q-min 2 \
  --use-tools --tools tfidf_retriever,option_retriever \
  --cards-budget 12 --coverage-tau 0.35 \
  --guardrails runs/_aggregated/bayes/tables/guardrails.json \
  --talk-slopes runs/_aggregated/bayes/tables/talk_slopes_by_domain_20250913_104225.csv \
  --tokens-autonudge --trough-margin 0.10 \
  --progress --log runs/guardrailed/state002_repro.jsonl
```

Why this works more often:
- Witness JSON is enforced/synthesized and citations are trimmed to the chosen option (PRO‑first), improving witness alignment.
- Retrieved snippets are anonymized and retained (up to 3 TF‑IDF + per‑option), raising provenance pass‑rate.
- Evidence scoring prioritizes option‑linked quotes for witness and guards against stale cards.

Optional (quality filter): add `--min-cqs 0.55` to keep only higher‑quality option cards (raises citation precision without loosening gates).

### Scattergun Matrix (Multi-Model Phase 2)

- Script: `scripts/run_scattergun_phase2.sh`
- Default seed block: `SEED_BASE=424242`. We keep all Phase 2 runs in the 424242 range so anonymized entities match historical guardrailed data (state002 recipes, meta-llama repros, etc.). Using a new base would reshuffle entity names and make apples-to-apples comparisons harder unless you explicitly want fresh mappings.
- Parallelism: set `PARALLEL_JOBS` (e.g., `PARALLEL_JOBS=4`) to fan out runs while monitoring DeepInfra rate limits. The script queues commands and executes them via `xargs -P` when parallelism > 1.
- DRY run preview: `DRY_RUN=1 PARALLEL_JOBS=4 SEED_BASE=424242 bash scripts/run_scattergun_phase2.sh`.
- Launch: `PARALLEL_JOBS=4 SEED_BASE=424242 bash scripts/run_scattergun_phase2.sh`.
- Pausing/resuming: stop the batch with `pkill -f run_scattergun_phase2.sh` (also kill any `sim.cli` children). When you relaunch with the same command, the script skips logs that already exist; delete a partial JSONL if you want that model/stage rerun from scratch.
- Logs: `runs/scattergun_phase2/<stage>/<model>_seedXXXXXX_NYY.jsonl` (runs skip automatically if the log exists).
- After completion: rerun aggregation (`scripts.aggregate_runs`, `scripts.session_view`) and refit Bayes guardrails with the latest `--target-accept` you trust (0.995 recommended).

### Batch Runs With Fixed Flags (80 runs, 20 steps each)

Run 80 sessions with identical flags and only the anonymization seed changing. This loop is resume-safe: it skips logs that already contain a complete 20-step block.

```
set -a; [ -f .env ] && . .env; set +a; mkdir -p runs/guardrailed
base=424240
for i in $(seq 1 80); do \
  seed=$((base+i)); \
  log=$(printf "runs/guardrailed/witness_json_guarded_state%03d.jsonl" "$i"); \
  # Skip if a 20-step run already exists in this file
  has_complete=$(python3 - <<'PY'
import json,sys,os
p=sys.argv[1]
if not os.path.exists(p): print("0"); sys.exit(0)
cur=None
with open(p,'r',encoding='utf-8') as f:
  for ln in f:
    try: rec=json.loads(ln)
    except: continue
    if rec.get('run_header'):
      if cur is not None and cur>=20: print("1"); sys.exit(0)
      cur=0
    else:
      if cur is not None: cur+=1
print("1" if (cur is not None and cur>=20) else "0")
PY
  "$log"); \
  [ "$has_complete" = "1" ] && continue; \
  ANON_SEED=$seed .venv/bin/python -m sim.cli \
    --student stateful-llm --provider deepinfra \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --steps 20 --closed-book --fact-cards --require-citations \
    --self-consistency 3 --best-of 8 --rerank evidence \
    --sc-extract 3 --q-min 2 \
    --use-tools --tools tfidf_retriever,option_retriever \
    --cards-budget 12 --coverage-tau 0.35 \
    --guardrails runs/_aggregated/bayes/tables/guardrails.json \
    --talk-slopes runs/_aggregated/bayes/tables/talk_slopes_by_domain_20250913_104225.csv \
    --tokens-autonudge --trough-margin 0.10 \
    --progress --log "$log" \
    || echo "$i seed=$seed FAILED" >> runs/guardrailed/failures.txt; \
done
```

Aggregate when done:

```
.venv/bin/python -m scripts.analyze --log runs/guardrailed/witness_json_guarded_state*.jsonl > runs/guardrailed/aggregate.json
```

Multi-model variant (same dials, different DeepInfra backends) is scripted at `scripts/run_guardrailed_multimodel.sh`. It sweeps the pinned recipe across Mixtral-8x7B, Mixtral-8x22B, DeepSeek V3.1, and Kimi-K2 by default; override `MODELS`, `SEED_BASE`, `SEED_COUNT`, or `STEPS` to adjust coverage. The script auto-picks the newest guardrail/talk-slope tables from `runs/_aggregated/bayes/tables/` and, when available, merges in per-model overrides from `runs/_aggregated/model_profiles.yaml` (generated by `scripts.model_profiles.py`). Regenerate the aggregates (`scripts.aggregate_runs` → `scripts.session_view` → `scripts.bayes.session_bayes_report`) before large sweeps so the bands and overrides stay current.

### Optional Message Logging

You can log per‑API‑call message metadata for tutor and student to study talk ratio and cadence:

- `LOG_MESSAGES=counts` — record events without text: `{ts, role_group∈{tutor,student}, api, model, prompt_tokens, completion_tokens, total_tokens, request_ms, system_len, user_len}`.
- `LOG_MESSAGES=text` — also include truncated `system`/`user` (≤ 4k chars each). Use only if you’re comfortable storing prompts.
- `TUTOR_LOG_RATIONALES=1` — when combined with `--rich`, include `task.rationales` in MCQ step logs (hint proxy).

Events are attached to each step under `messages[]` in the JSONL and appear in the flattened CSV as a JSON string column.

Advanced (opt‑in refactor flags)
- `SIM_USE_EVIDENCE_PIPELINE=1` — uses the new unified evidence pipeline and attaches `evaluation.evidence_report` with coverage/witness/abstention details.
- `SIM_USE_FACTCARD_MANAGER=1` — routes LEARN/USE card normalization and budgeting through the new Fact‑Card Manager.
Both flags are off by default; enable them to validate the refactor without changing current behavior.

### Bayesian Guardrails (optional)

- `--guardrails PATH` — load `guardrails.json` from `scripts/bayes/session_bayes_report.py` to enforce per‑step/run token bands. Each step gets `guardrail_alerts` with status and suggested action. Add `--tokens-autonudge` to let the orchestrator gently adjust dials when outside the band.
- `--talk-slopes PATH` — load `talk_slopes_by_domain_*.csv` to pick domain‑specific tutor talk mode (lean/neutral/rich) using the posterior `prob_positive` (threshold via `--talk-ppos-threshold`, default 0.70).
- `--turns-limit N` — cap session turns; once exceeded, the simulator triggers a concise resolve‑or‑escalate prompt for the rest of the run. If omitted, it defaults to `mean_steps + ≈1SD` inferred from guardrails.

Example:
```
.venv/bin/python -m sim.cli \
  --student llm --provider deepinfra --model deepseek-ai/DeepSeek-R1 \
  --steps 60 --closed-book --fact-cards \
  --guardrails runs/2025-09-13/tables/guardrails.json \
  --talk-slopes runs/2025-09-13/tables/talk_slopes_by_domain_20250913_0756.csv \
  --tokens-autonudge --progress \
  --log runs/with_guardrails.jsonl
```

### Bayes + Evidence‑Aware Selection (recommended)

1) Train Bayes on your aggregated logs (no LLM calls):
```
.venv/bin/python -m scripts.aggregate_runs --runs-dir runs --out-jsonl runs/_aggregated/all_steps.jsonl.gz --out-csv runs/_aggregated/all_steps_flat.csv.gz
.venv/bin/python -m scripts.session_view --steps-csv runs/_aggregated/all_steps_flat.csv.gz --out-csv runs/_aggregated/session_view.csv.gz
.venv/bin/python -m scripts.bayes.session_bayes_report --csv runs/_aggregated/session_view.csv.gz --out-root runs/_aggregated/bayes --target-accept 0.98 --tune 2000
```

2) Run with guardrails + evidence‑aware rerank (Mixtral example):
```
.venv/bin/python -m sim.cli \
  --student stateful-llm --provider deepinfra --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --steps 60 --closed-book --fact-cards --require-citations \
  --self-consistency 3 --sc-extract 2 --best-of 6 --rerank evidence --q-min 2 \
  --use-tools --tools tfidf_retriever \
  --guardrails runs/_aggregated/bayes/tables/guardrails.json \
  --talk-slopes $(ls -t runs/_aggregated/bayes/tables/talk_slopes_by_domain_*.csv | head -1) \
  --tokens-autonudge --progress \
  --log runs/guardrailed/mixtral_live_grail_bo6_ev.jsonl
```

What this does
- Bayes: enforces token bands (alerts + optional nudges), sets tutor talk by domain, and derives a turns limit.
- Evidence rerank: collects a small candidate pool and selects the answer whose citations best match the gold option (witness/coverage), improving credited without leaving the safe token band.

### Advanced ICL Examples

- Controllers (ToT) with early‑stop SC and gating
  `.venv/bin/python -m sim.cli --steps 10 --closed-book --controller tot --tot-width 2 --tot-depth 2 --self-consistency 3 --adaptive-sc --uncertainty-gate --escalate-reasoning`

- Program‑of‑Thought (safe Python) for arithmetic/table calc
  `.venv/bin/python -m sim.cli --steps 10 --closed-book --use-tools --tools pyexec --reasoning pot`

- Few‑shot with embedding KNN + MMR diversification
  `.venv/bin/python -m sim.cli --steps 10 --closed-book --shots data/examples.jsonl --shots-k 6 --shots-selector knn --shots-embed-backend st --shots-diverse --shots-mmr 0.5`

- APE header injection (inline or file)
  `.venv/bin/python -m sim.cli --steps 10 --closed-book --ape-header "You are a careful reasoner."`
  or
  `.venv/bin/python -m sim.cli --steps 10 --closed-book --ape-header docs/ape/header.json`

## Witness & Credit (What Matters)

- Enable Fact‑Cards two‑pass and require citations: add `--fact-cards --require-citations`.
- Keep self‑consistency at 3 votes: add `--self-consistency 3`.
- If extraction is noisy, try SC=5 for LEARN only (use `--sc-extract 5` while keeping `--self-consistency` lower).
- Larger card budget improves stability and reduces eviction of required option‑quotes. Default is now 10; you can set `--cards-budget 9..12` depending on snippets/tools.

Notes: At least one LEARN card must quote the chosen option via `where.scope="option"` and `where.option_index=<chosen>`, and the USE pass should include ≥1 citation pointing to that option. Quotes are clamped centrally to ≤15 tokens.

Evidence engine (unified): `sim/credit.py` computes coverage, witness, snippet provenance (when tools are on), and final `credited` + reasons. Controllers and the main path delegate to it for consistent results.

Grammar: for MCQ + Fact‑Cards the CLI forces `--grammar json` (schema cannot carry citations).

Snippet rule: when tools are off, no snippet provenance is required. When tools are on, any retrieved snippet can satisfy the provenance check (previously it required snippet[0]).

## Evidence & Abstention (IDK)

- Enable calibrated abstention: add `--idk --target-confidence 0.60` (t only affects the score; abstention is hard‑evidence only).
- Hard‑evidence gate (built‑in): IDK triggers only when the chosen option lacks an option‑linked quote, coverage < τ, or the chosen option didn’t cite its required PRO card id. Confidence (t) does not cause abstention.
- Witness‑first tie‑break: on near ties, re‑pick the option with the strongest overlap to the cited quotes.
- RAG‑ON recommended: `--use-tools --tools tfidf_retriever`. Reserve room for evidence with `--cards-budget 8` (5 options + 2 pinned snippets + 1 spare).
- New log fields: `evaluation.calibrated_score`, `evaluation.abstain_reason`, and per‑step `card_validation` with any card issues.
- Validator update: option‑quote validation is now offset‑first and canonicalized‑substring second (NFKC + casefold + dash/space unification). The rule remains strict (quotes must be substrings), but minor Unicode/typographic drift no longer causes false negatives. A new counter `card_validation.validator_modes` summarizes how many option quotes validated via `offset` vs `canon-substring`.

Quality and robustness changes (Sep 2025)
- Witness JSON: Student prompts enforce a structured `witness` block; if missing, the orchestrator synthesizes a minimal witness from chosen citations.
- Citation trimming: PRO id for the chosen option is cited first, then other chosen‑option citations, then at most one context citation.
- Provenance fix: Retrieved snippets in CONTEXT are anonymized when anonymization is on; up to 3 TF‑IDF snippets plus per‑option snippets are retained for better substring matches.
- Option‑weighted witness: Evidence scoring prioritizes tokens derived from option‑linked citations for witness alignment and ignores out‑of‑range option indices in stale cards.
- Card quality (optional): `--min-cqs` filters non‑required option cards by relevance/specificity/length; `--per-option-top-k` caps extras per option; PRO stubs are always kept.

## Evidence Gating & Retrieval (LEARN / Fact‑Cards)

- Flags:
  - `--q-min` (default 1): min option‑linked quotes per option before selection.
  - `--coverage-tau`: coverage threshold τ (overrides `TUTOR_COVERAGE_TAU`).
  - `--max-learn-boosts` (default 0): extra extract rounds when gates fail.
  - `--mmr` (default 0.4): MMR lambda for `option_retriever` diversity.
  - `--span-window` (default 240): token span for retrieval context.
  - `--citations off|lenient|strict`: gating mode (strict enforces coverage/witness gates).
  - Tools: `--use-tools --tools option_retriever` enables option‑conditioned retrieval (per‑option queries).

- Selection uses evidence‑weighted scoring (when `--fact-cards` and `--citations lenient|strict`):
  `score(o) = 1.0*coverage(o) − 0.7*overlap(o,¬o) + 0.1*num_independent_quotes(o)` with a hard `q_min` constraint.

- Telemetry (per step): `evidence_telemetry` includes `min_quotes_per_option`, `coverage_by_option`, `coverage_chosen`, `witness_overlap_ratio`, `num_distinct_sources`, plus LEARN cost (`learn_tokens`, `learn_time_s`).

Changelog (Sep 2025)
- Retrieval context fix: tools now read the raw retrieval corpus while the learner sees Fact-Cards JSON (prevents empty snippets when cards are stored in `notes_text`).
- Freeze‑cards refinement: when `--cards-freeze` is used, the simulator injects ephemeral per‑option PRO cards derived from the current options (≤15‑token quotes) so the answering phase can be evaluated in isolation from extraction without failing evidence gates. The provided cards themselves remain unchanged across steps.
- Quote slicing: per‑option PRO stubs in FREEZE are now built by slicing directly from the presented option text and setting `where.start/end`. In LEARN, valid offsets on incoming cards are treated as authoritative and the quote is sliced from the option.
- Defaults: `--cards-budget` increased to 10.
 - Maintenance: removed an unused helper and pruned unused imports/locals; no user-facing behavior changes.

Example (Mixtral, 120‑step RAG‑ON)
```
ANON_SEED=424242 TUTOR_COVERAGE_TAU=0.40 \
.venv/bin/python -m sim.cli \
  --student stateful-llm --provider deepinfra \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --steps 120 --closed-book \
  --fact-cards --require-citations \
  --self-consistency 3 --sc-extract 1 \
  --use-tools --tools tfidf_retriever \
  --cards data/holdout/postcutoff_text.jsonl \
  --cards-budget 8 \
  --idk --target-confidence 0.60 \
  --progress \
  --log runs/holdout/mixtral_postcutoff_t060_rag_on.jsonl
```

## Quote Validation Details

- Offsets > strings: if a card provides `where.start/end` for `where.scope="option"`, the validator derives the quote from the exact slice of the presented option. This removes hyphen/space drift entirely.
- Canonicalized match: if offsets aren’t present, the validator compares `canon(quote) ∈ canon(option)`, where canon = NFKC + casefold + dash/space unification. The gate remains strict; this only avoids false negatives on formatting variants.
- Logs: `card_validation.validator_modes = {offset, canon-substring, fail}` helps audit what path validated each option quote.

## Multi‑Model Smoke Runs (3 terminals)

Run nine 30‑step smokes across common models by splitting them into three terminals. Adjust model slugs as needed for your DeepInfra account.

Shared prep (run in each terminal):
```
set -a; [ -f .env ] && . .env; set +a; unset TUTOR_MOCK_LLM; mkdir -p runs/smoke
```

Terminal A — DeepSeek trio
```
for MODEL in \
  "deepseek-ai/DeepSeek-V3.1" \
  "deepseek-ai/DeepSeek-R1-0528" \
  "deepseek-ai/DeepSeek-V3-0324" \
; do \
  SLUG=$(echo "$MODEL" | tr '/:' '__'); \
  ANON_SEED=424242 .venv/bin/python -m sim.cli \
    --student stateful-llm --provider deepinfra --model "$MODEL" \
    --steps 30 --closed-book --fact-cards --require-citations \
    --self-consistency 3 --sc-extract 1 \
    --use-tools --tools tfidf_retriever \
    --cards data/holdout/postcutoff_text.jsonl \
    --cards-budget 8 --idk --target-confidence 0.60 \
    --progress --log "runs/smoke/${SLUG}_N30.jsonl"; \
done
```

## Aggregation & Analysis (All Runs → One Database)

After you accumulate runs, convert everything in `runs/` (recursively) into a single database under `runs/_aggregated/`:

```bash
# 1) Aggregate all JSONL logs
.venv/bin/python -m scripts.aggregate_runs --runs-dir runs --out-dir runs/_aggregated

# 2) Generate quick insights
.venv/bin/python -m scripts.insights --agg-dir runs/_aggregated --out-dir runs/_aggregated/plots

# 3) Optional: export Parquet
.venv/bin/python -m scripts.export_parquet --agg-dir runs/_aggregated

# 4) Optional: one‑row‑per‑run catalog
.venv/bin/python -m scripts.catalog --agg-dir runs/_aggregated --out runs/_catalog/catalog.csv

# 5) Optional: component analysis (dial effects per model)
.venv/bin/python -m scripts.component_analysis --agg-dir runs/_aggregated --out-dir runs/_aggregated/analysis --filter-apples

# 6) Optional: best settings by model (max credited)
.venv/bin/python -m scripts.best_settings --agg-dir runs/_aggregated --out runs/_aggregated/analysis/best_settings.csv --apples
```

Key artifacts (autogenerated):

- `runs/_aggregated/steps.csv.gz` (and `steps.parquet`) — master step‑level table with provenance, dials, evaluation, timing, tokens.
- `runs/_aggregated/runs_summary.csv.gz` (and `runs_summary.parquet`) — one row per run.
- `runs/_aggregated/model_ranking.csv.gz` — per‑model means (credited/witness/acc/timing/tokens).
- `runs/_aggregated/by_model_steps.csv.gz`, `by_model_runs.csv.gz` — rollups by model.
- `runs/_aggregated/plots/INSIGHTS.md` — human‑readable overview.
- `runs/_catalog/catalog.csv` — “table of contents” for runs, newest first.

Housekeeping:

```bash
# Archive old JSON/JSONL from runs/ (safe: aggregates stay)
.venv/bin/python -m scripts.cleanup_runs --min-age-minutes 10 --out-dir runs/_archive --dry-run
.venv/bin/python -m scripts.cleanup_runs --min-age-minutes 10 --out-dir runs/_archive
```

Small, fast sweep (configurable):

```bash
# FAST=1 → steps=10, Mixtral only; FAST=0 → steps=30, adds GPT‑4.1
FAST=1 STEPS=10 BUDGETS="6 8" SC_LIST="1 3" bash scripts/sweep_minisuite.sh
```

See `docs/iclsim/OPERATIONS.md` for a deeper operational guide and current insights.

## Most Important Files

Core simulation:
- `sim/orchestrator.py` — run loop, LEARN/USE Fact‑Cards, witness/coverage, evaluation, logging.
- `sim/learner.py` — LLM students; USE JSON spec enforces option‑first citation and abstention on evidence failure.
- `sim/controllers.py` — LtM/ToT controllers (budget‑capped) that seed votes before SC.
- `sim/retrieval.py` — embedding KNN (lexical/sentence‑transformers/OpenAI) + MMR diversification with disk cache.
- `sim/tasks.py` — MCQ/SAQ task types and evaluation helpers.

Aggregation & analysis:
- `scripts/aggregate_runs.py` — scan `runs/` recursively (and archives); build the master DB and summaries.
- `scripts/insights.py` — quick correlations + dial tables; writes `INSIGHTS.md`.
- `scripts/export_parquet.py` — Parquet exports of aggregates (steps, runs, by‑model, ranking).
- `scripts/catalog.py` — one‑row‑per‑run catalog for easy browsing.
- `scripts/component_analysis.py` — per‑model dial effects (budget, SC, IDK; correlations).
- `scripts/best_settings.py` — best per‑model row by credited_final.

Experiment helpers:
- `scripts/sweep_minisuite.sh` — small, configurable grid sweep (providers × {budget} × {SC}).
- `scripts/wait_for_runs.py` — wait on PIDs, then aggregate + summarize.
- `scripts/cleanup_runs.py` — archive and remove old JSON/JSONL from `runs/`.

Generated data:
- `runs/_aggregated/` — master DBs (CSV/Parquet), per‑model rollups, rankings, `plots/INSIGHTS.md`.
- `runs/_archive/` — tarballs of archived raw logs + manifests.
- `runs/_catalog/` — CSV catalog of runs for quick scanning.

## Tools & Scripts Added (Phase 2–3)

- `scripts/active_prompt.py` — selects top‑K uncertain items (vote entropy/confidence) to label for Active Prompt.
- `scripts/ape_optimize.py` — small dev‑loop to choose an instruction header; writes `docs/ape/header.json`.

## Detailed ICL Routing & Dials

See `docs/icl_routing.md` for controllers (LtM/ToT), PoT, uncertainty gating, reranking, grammar constraints, embedding KNN/MMR, APE header, and compression.

## About The Runs (What We’re Achieving)

Every MCQ run writes a header (config + anonymization seed) followed by per‑step records. For MCQ steps, the evaluation includes a citations evidence block:

- `coverage` — token overlap between cited quotes and the gold option.
- `witness_pass` — whether cited text supports the correct option more than others.
- `credited` — correct AND has citations AND meets coverage τ AND passes witness.

We aim to lift credited by:

- Aligning USE citations to the chosen option’s PRO card first (prompt enforces this),
- Guaranteeing each option has a ≤15‑token option‑quote PRO card (LEARN synthesizes if missing), and
- Choosing model‑specific dials (e.g., `self_consistency`, `cards_budget`) that trade off cost/latency versus witness/credited improvements.


Terminal B — OpenAI + Qwen
```
for MODEL in \
  "openai/gpt-oss-120b" \
  "Qwen/Qwen3-235B-A22B-Thinking-2507" \
  "Qwen/QwQ-32B" \
; do \
  SLUG=$(echo "$MODEL" | tr '/:' '__'); \
  ANON_SEED=424242 .venv/bin/python -m sim.cli \
    --student stateful-llm --provider deepinfra --model "$MODEL" \
    --steps 30 --closed-book --fact-cards --require-citations \
    --self-consistency 3 --sc-extract 1 \
    --use-tools --tools tfidf_retriever \
    --cards data/holdout/postcutoff_text.jsonl \
    --cards-budget 8 --idk --target-confidence 0.60 \
    --progress --log "runs/smoke/${SLUG}_N30.jsonl"; \
done
```

Terminal C — Agentic/coding + Moonshot + Llama
```
for MODEL in \
  "Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo" \
  "moonshotai/Kimi-K2-Instruct" \
  "meta-llama/Llama-4-Scout-17B-16E-Instruct" \
; do \
  SLUG=$(echo "$MODEL" | tr '/:' '__'); \
  ANON_SEED=424242 .venv/bin/python -m sim.cli \
    --student stateful-llm --provider deepinfra --model "$MODEL" \
    --steps 30 --closed-book --fact-cards --require-citations \
    --self-consistency 3 --sc-extract 1 \
    --use-tools --tools tfidf_retriever \
    --cards data/holdout/postcutoff_text.jsonl \
    --cards-budget 8 --idk --target-confidence 0.60 \
    --progress --log "runs/smoke/${SLUG}_N30.jsonl"; \
done
```

Tip: Set `TUTOR_COVERAGE_TAU=0.25` for stricter evidence gating consistent with the Mixtral holdout profile.

## CLI Flags (selected)

- `--steps N`: number of items (default 5)
- `--task {mcq,saq,code,proof,table_qa}` (default mcq)
- `--student {llm,algo,stateful-llm}` and `--provider {openai,deepinfra,deepseek}`
- `--model NAME`: student model (e.g., `deepseek-ai/DeepSeek-R1`)
- `--closed-book`: include CONTEXT in `presented_stem` only; student must use it
- `--rich`: ask tutor for rationales/metadata when available
- `--self-consistency N`: majority vote over N MCQ answers
- `--adaptive-sc` and `--sc-quorum K`: early‑stopping SC; stop when quorum reached
- `--sc-policy {fixed,adaptive}` and per‑difficulty k (`--sc-k-easy/medium/hard`)
- `--sc-extract N`: self-consistency for Fact‑Cards extraction (LEARN); 0 = use `--self-consistency`
- `--uncertainty-gate` + `--conf-threshold` + `--entropy-threshold` + `--max-k-escalated` + `--escalate-reasoning`: escalate compute when uncertain; can switch to ToT
- `--best-of N` + `--rerank {confidence,evidence,judge}`: Best‑of‑N candidate sampling with reranking
- `--reasoning {none,cot,ltm,tot,sot,selfdisco,got,pot}`: internal scaffolds (JSON‑only output)
- `--controller {basic,ltm,tot}` + `--controller-budget` + `--tot-width/depth`: multi‑step controllers
- `--grammar {none,json,schema}`: constrained decoding mode (JSON schema when supported)
- `--ape-header PATH|TEXT`: prepend instruction header to student prompts
- Few‑shot: `--shots PATH --shots-k K --shots-selector {knn,random,as-is}`
  - Embeddings: `--shots-embed-backend {lexical,st,openai}`; diversity: `--shots-diverse --shots-mmr 0.5`
  - Reranker: `--shots-reranker ce --shots-reranker-model BAAI/bge-reranker-base` (optional)
- Prompt compression: `--compress-examples --compress-ratio R`
- `--reflection-every K`: every K steps, condense cards/notes into a brief study note (improves late‑run stability)
- `--notes-file PATH`: simulate closed‑book notes
- `--progress`: show a lightweight stderr progress bar
- `--log PATH`: append JSONL records per step
Notes: anonymization is on by default. Use `--no-anon` to disable (the `--anonymize` flag is accepted but redundant). Env overrides: `TUTOR_ANONYMIZE=1`, `TUTOR_REQUIRE_CITATIONS=1`, `TUTOR_COVERAGE_TAU=0.35`.

## Health Checks

- Keys present: ensure `.env` has `OPENAI_API_KEY` (tutor) and optionally `DEEPINFRA_API_KEY` + `DEEPINFRA_MODEL` (student).
- Offline mode: `TUTOR_MOCK_LLM=1` to bypass network.
- JSON mode: prompts include “JSON” and use `response_format=json_object` for compliant models.

## Analyze Logs

Aggregate one or more runs:

```
.venv/bin/python -m scripts.analyze --log runs/full_live.jsonl
```

The analyzer reports accuracy curves, citation credit (if enabled), and alias‑swap metrics when applicable.

It also reports: early/late credited with two‑proportion p‑value, Wilson 95% CI; witness curves; retention deltas (lag 3–5); timing and token usage.

## Context Learning & Stability (CL&S)

Measure within‑session learning speed and stability/forgetting using long, stateful runs.

- Run long stateful sessions (per model):
  - Example (DeepSeek‑R1, 60 steps):
    `.venv/bin/python -m sim.cli --student stateful-llm --provider deepinfra --model deepseek-ai/DeepSeek-R1 --steps 60 --closed-book --fact-cards --require-citations --self-consistency 2 --use-tools --tools tfidf_retriever --progress --log runs/window/deepinfra_deepseek-ai-DeepSeek-R1_N60.jsonl`
  - Mixtral‑8x7B (DeepInfra):
    `.venv/bin/python -m sim.cli --student stateful-llm --provider deepinfra --model mistralai/Mixtral-8x7B-Instruct-v0.1 --steps 60 --closed-book --fact-cards --require-citations --self-consistency 2 --use-tools --tools tfidf_retriever --progress --log runs/window/deepinfra_mistralai-Mixtral-8x7B-Instruct-v0.1_N60.jsonl`
  - OpenAI baseline:
    `.venv/bin/python -m sim.cli --student stateful-llm --provider openai --steps 60 --closed-book --fact-cards --require-citations --self-consistency 2 --use-tools --tools tfidf_retriever --progress --log runs/window/openai_gpt-4.1_N60.jsonl`

- Aggregate and score CL&S:
  - Per‑run aggregate: `.venv/bin/python -m scripts.analyze --log runs/window/<slug>_N60.jsonl > runs/window/<slug>_N60.aggregate.json`
  - CL&S CSV: `.venv/bin/python -m scripts.cls_score --glob "runs/window/*_N60.jsonl" > runs/window/CLS_N60.csv`

- Score definition (in `scripts/cls_score.py`):
  - LAUC: credited AUC over steps (0..1)
  - STAB: mean(credited[last 10%]) − mean(credited[20%..40%])
  - WIT: witness_pass_mean from analyzer
  - CL&S = 0.6·LAUC + 0.4·clip(WIT + STAB, 0, 1)

Tips
- Use `--steps 120` for stronger signals; `--self-consistency 2` balances speed/stability.
- Keep `--fact-cards --require-citations --use-tools tfidf_retriever` so “credited” reflects evidence, not guesses.
- The stateful student preserves memory and now cites Fact‑Cards when answering MCQ.

## Quick Checks: Is the model using context?

Run these two first for the strongest, quickest signal.

1) RAG delta (retrieval ON vs OFF)

Baseline (retrieval ON):

```
.venv/bin/python -m sim.cli --student stateful-llm --provider deepinfra \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 --steps 60 --closed-book \
  --fact-cards --require-citations --self-consistency 2 \
  --use-tools --tools tfidf_retriever \
  --log runs/window/mixtral_rag_on.jsonl
```

Ablation (retrieval OFF):

```
.venv/bin/python -m sim.cli --student stateful-llm --provider deepinfra \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 --steps 60 --closed-book \
  --fact-cards --require-citations --self-consistency 2 \
  --log runs/window/mixtral_rag_off.jsonl
```

Interpretation: compute RAG delta and attribution gaps

```
.venv/bin/python -m scripts.rag_delta --on runs/window/mixtral_rag_on.jsonl --off runs/window/mixtral_rag_off.jsonl
```

Healthy signs: large RAG delta in credited accuracy (≥ 0.15), small attribution gap (raw − credited ≤ 0.05).

2) Temporal holdout (post‑cutoff cards)

Run on a dated, post‑cutoff set. Provide Fact‑Cards via `--cards` (JSON or JSONL with `{cards:[...]}` objects). Optionally freeze them with `--cards-freeze` to disable LEARN updates. In freeze mode the simulator still injects per‑option PRO stubs (short quotes from the choice texts) at answer time so evidence gating remains meaningful while isolating answering behavior.

```
.venv/bin/python -m sim.cli --student stateful-llm --provider deepinfra \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 --steps 60 --closed-book \
  --fact-cards --require-citations --self-consistency 2 \
  --use-tools --tools tfidf_retriever \
  --cards PATH_TO_POSTCUTOFF_CARDS.jsonl --cards-freeze \
  --log runs/holdout/mixtral_postcutoff.jsonl
```

Notes
- The CLI accepts `--cards` as JSON or JSONL. JSONL can contain per‑line `{cards:[...]}` or `{text:"..."}` objects; cards are concatenated, text is joined.
- `--cards-freeze` preserves provided cards as the answering context (no LEARN step).

Sample holdout file
- A small synthetic, dated holdout set is provided at `data/holdout/postcutoff_text.jsonl` (uses post‑2025 dates to avoid pretraining leakage).
- Recommended run (citations ON, LEARN from text):

```
ANON_SEED=424242 TUTOR_COVERAGE_TAU=0.25 \
.venv/bin/python -m sim.cli \
  --student stateful-llm --provider deepinfra \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --steps 120 --closed-book \
  --fact-cards --require-citations \
  --self-consistency 3 --sc-extract 1 \
  --use-tools --tools tfidf_retriever \
  --reflection-every 10 \
  --cards data/holdout/postcutoff_text.jsonl \
  --progress \
  --log runs/holdout/mixtral_postcutoff_text_tau025_sc3_ref10.jsonl
```

If you already have pre‑built cards with option‑linked quotes for the exact items, you can use `--cards-freeze --cards-budget N` with a JSON/JSONL file containing `{cards:[...]}`. Otherwise, prefer the text JSONL path so LEARN can extract option‑quoted cards.

3) Nonce entity substitution (anonymization seed)

```
ANON_SEED=$RANDOM \
.venv/bin/python -m sim.cli --student stateful-llm --provider deepinfra \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --steps 60 --closed-book --fact-cards --require-citations \
  --self-consistency 2 --use-tools --tools tfidf_retriever \
  --anonymize --log runs/window/mixtral_nonce.jsonl
```

Credited accuracy should be close to baseline with retrieval ON. A large drop suggests reliance on pretraining associations.

4) Counterfactual flips (numbers/dates)

```
.venv/bin/python -m scripts.perturb_cards --in PATH_TO_CARDS.jsonl --out runs/cards_cf.jsonl --percent 7
.venv/bin/python -m sim.cli --student stateful-llm --provider deepinfra \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 --steps 60 --closed-book \
  --fact-cards --require-citations --self-consistency 2 \
  --use-tools --tools tfidf_retriever \
  --cards runs/cards_cf.jsonl --cards-freeze \
  --log runs/window/mixtral_counterfactual.jsonl
```

Readout: the model should follow edited values and cite the edited context; reversions to pretraining values indicate leakage.

Example findings (recent runs)
- Mixtral‑8x7B (120 steps): LAUC≈0.055, STAB≈0.083, WIT≈0.283 → CL&S≈0.180 (shows learning + stability).
- DeepSeek‑R1 (60 steps): WIT≈0.333, LAUC≈0, STAB≈0 → strong evidence alignment but flat learning signal at N=60.
- GPT‑4.1 (60 steps): WIT≈0.250, LAUC≈0, STAB≈0 → flat at N=60; consider N=120 to probe stability.

## Project Structure

- sim/: orchestrator, learners, tasks, evaluators, tools, anonymization
  - `sim/orchestrator.py`: core runner + dials (closed‑book, anonymize, self‑consistency, tools, fact‑cards)
    - Normalizes cards (≤15‑token verbatim quotes), guarantees per‑option PRO cards, reserves up to 2 snippet cards, computes evidence signals, runs witness tie‑break and hard‑evidence abstain gate, logs `card_validation` and `abstain_reason`.
  - `sim/learner.py`: `LLMStudent`, `AlgoStudent`, `StatefulLLMStudent`
    - USE prompts require the chosen option to cite its PRO card id first; accepts explicit `IDK`.
  - `sim/tasks.py`: task schemas and evaluators (MCQ; SAQ/code/proof/table supported)
  - `sim/evaluators.py`, `sim/tools.py`, `sim/anonymize.py`
- tutor/: wrappers for tutor/student backends
  - `tutor/llm_openai.py`: fixed tutor model wrapper (`gpt-5-nano-2025-08-07`)
  - `tutor/llm_deepinfra.py`: OpenAI‑compatible client for DeepInfra models
- docs/: handbooks, specs, and configs (no live reports)
  - `docs/ICL_Simulator_Handbook.md`: detailed guide
  - `docs/iclsim/`: skill map, templates, alias families, experiments
- runs/: live run artifacts (JSONL logs, aggregates, reports). Git‑ignored by default
- scripts/: analyzers and utilities (`analyze.py`, `rag_delta.py`, `report_runs.py`, `validate_cards.py`, etc.)
- data/: small sample inputs (e.g., `data/holdout/postcutoff_text.jsonl`)

## Policies & Options

- Anonymization on by default; disable with `--no-anon`.
- Evidence gating (optional): set `--require-citations` and enable fact‑cards to credit only supported answers.
- Memory gating (baseline): remediation when wrong; light decay scheduling planned.

## Troubleshooting

- Use `python3` if `python` is not found.
- If `ModuleNotFoundError: openai`, run `pip install -r requirements.txt` inside your venv.
- If DeepInfra rejects JSON mode, the student wrapper retries without `response_format` and falls back to letter parsing.
- For DNS/HTTP issues, confirm keys and outbound HTTPS.

## Tests

```
.venv/bin/python -m pytest -q
```

All tests run offline using `TUTOR_MOCK_LLM=1`.

Note: `pytest` is not included in `requirements.txt`. Install it in your venv if needed:

```
.venv/bin/python -m pip install pytest
```

## Multi‑Model Benchmark

- Script: `scripts/run_models.sh` runs multiple student models, aggregates results, executes an alias‑swap suite, and writes a CSV summary.
- Live progress: per‑batch progress bars (stderr) with `QUIET=0`; suite‑level ETA is always shown on stderr.
- Output: per‑model logs under `runs/compare/<provider>_<model>/`, and a summary CSV at `runs/compare/summary.csv` (or `SUMMARY_OUT=...`).

Smoke (sequential; clean bars):

```
chmod +x scripts/run_models.sh
mkdir -p runs/compare && \
SUMMARY_OUT=runs/compare/summary_smoke.csv QUIET=0 BATCH_PAR=1 SC_ANSWER=1 STEPS=4 BATCHES=2 TAU=0.40 USE_RETRIEVER=0 \
./scripts/run_models.sh
```

Smoke (parallel; bars interleave):

```
mkdir -p runs/compare && \
SUMMARY_OUT=runs/compare/summary_smoke.csv QUIET=0 BATCH_PAR=2 SC_ANSWER=1 STEPS=4 BATCHES=2 TAU=0.40 USE_RETRIEVER=0 \
./scripts/run_models.sh
```

Semifinal (top models only; edit `MODELS` in the script):

```
SUMMARY_OUT=runs/compare/summary_semi.csv QUIET=0 BATCH_PAR=2 SC_ANSWER=2 STEPS=8 BATCHES=3 TAU=0.40 USE_RETRIEVER=1 \
./scripts/run_models.sh
```

Final:

```
SUMMARY_OUT=runs/compare/summary_final.csv QUIET=0 BATCH_PAR=2 SC_ANSWER=3 STEPS=12 BATCHES=5 TAU=0.40 USE_RETRIEVER=1 \
./scripts/run_models.sh
```

CSV columns include: `credited_final_mean`, `credited_auc_mean`, `alias_B_credited_mean`, `witness_pass_mean`, `acc_*`, `median_step_seconds`, `tokens_per_step_mean`.

## Efficiency Metrics

- Per‑step logs include `duration_ms` and `student_usage.{prompt_tokens,completion_tokens,total_tokens}`.
- Aggregated timing in analyzer: `overall.timing.{median,mean,total}_seconds`.
- Aggregated usage in analyzer: `overall.usage.tokens_per_step_mean`, `tokens_total`.

## Progress & ETA

- Suite ETA prints as: `Batches: d/T | Elapsed H:MM:SS | ETA H:MM:SS` on stderr.
- Per‑batch bars (when `QUIET=0`) may interleave with `BATCH_PAR>1`.

## Troubleshooting (Benchmark)

- Tee path: keep `tee runs/compare/summary.csv` on the same line; if it wraps, tee writes to the directory and fails.
- BrokenPipe noise: batch jobs don’t write to stdout; `sim.cli` suppresses stdout when `--log` is used, and final prints are guarded. If you still see noise, set `QUIET=1`.
- Slow/stuck: start with `BATCH_PAR=2`, set `SC_ANSWER=1`, trim heavy models in `MODELS`, and favor faster OSS models first.
- Rate limits (429): reduce `BATCH_PAR`, disable `--use-tools`, or lower `STEPS` per batch.
