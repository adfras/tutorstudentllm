# ICL Simulator Handbook

Note (Sep 2025): Advanced ICL controls are now available. See `docs/icl_routing.md` for:
- Controllers: LtM (PLAN→SOLVE) and ToT (width/depth, budget‑capped)
- Program‑of‑Thought with safe Python execution (`--use-tools --tools pyexec --reasoning pot`)
- Uncertainty/entropy gating with escalation to ToT
- Best‑of‑N with reranking (confidence/evidence/tutor judge)
- Constrained decoding (`--grammar schema`), embedding KNN + MMR for few‑shot, APE header injection, and prompt compression knobs


This handbook explains the simulator end‑to‑end so a new reader can install, run, measure, and extend it with confidence.

## 1) Overview

- Purpose: Closed‑book, anonymized tutor→student simulations with strict JSON I/O, pluggable students (LLM or algorithmic), and evidence‑gated credit.
- Tutor: fixed OpenAI model `gpt-5-nano-2025-08-07` (see `tutor/llm_openai.py`).
- Tasks: MCQ primary (SAQ/code/proof/table_qa supported). Fact‑Cards v2 provides a two‑pass LEARN/USE scheme with citations.
- Logs: JSONL per step (append‑only) + optional aggregate summaries via `scripts/analyze`.
- Benchmark: `scripts/run_models.sh` runs multiple models, executes an alias‑swap suite, and writes a summary CSV.

## 2) Install & Environment

Prereqs
- Python 3.11+ recommended.
- Create a venv and install deps:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Testing dependencies
- The test suite uses `pytest` which is not included in `requirements.txt`. To run tests locally:

```
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest -q
```

Environment keys
- `.env` should contain:

```
OPENAI_API_KEY=sk-...
DEEPINFRA_API_KEY=di-...    # for DeepInfra student
DEEPINFRA_MODEL=deepseek-ai/DeepSeek-R1  # optional default
```

Useful env toggles
- `TUTOR_ANONYMIZE=1` → anonymization (default is on unless `--no-anon`).
- `TUTOR_REQUIRE_CITATIONS=1` → evidence‑gated credit on MCQ.
- `TUTOR_COVERAGE_TAU=0.40` → coverage threshold (default 0.40).
- `TUTOR_WITNESS_TIE_PASS=1` → accept witness ties when the gold option is among the top‑tied options.
- `ANON_SEED=<int>` → fix the anonymization mapping (nonce entities) for reproducible runs.

## 3) Run the Simulator

Presets (strict credit):
```
scripts/presets/run_strict_41.sh         # OpenAI gpt‑4.1 (τ=0.25, with header)
scripts/presets/run_strict_mixtral.sh    # DeepInfra Mixtral (τ=0.30)
```

Algorithmic baseline:
```
python -m sim.cli --student algo --steps 5 --closed-book --notes-file data/notes_algo.txt
```

LLM student (DeepInfra):
```
unset TUTOR_MOCK_LLM
python -m sim.cli \
  --student llm --provider deepinfra --model deepseek-ai/DeepSeek-R1 \
  --steps 10 --closed-book --progress --log runs/live.jsonl
```

Important CLI flags
- `--closed-book` places notes/tools output into CONTEXT and requires the student to use it.
- `--self-consistency N` runs N votes/drafts; MCQ uses majority vote.
- `--sc-extract N` sets self-consistency specifically for Fact‑Cards extraction (LEARN); 0 = use `--self-consistency`.
- `--fact-cards` enables LEARN/USE Fact‑Cards v2.
- `--require-citations` enables evidence‑gated credit on MCQ.
 - `--use-tools --tools tfidf_retriever` enables retrieval into CONTEXT.
 - `--idk --target-confidence t` enables calibrated abstention; t affects scoring only. Abstain triggers on hard evidence failures (no option‑linked quote, coverage < τ, or missing required PRO cite).
- `--reflection-every K` every K steps, condense cards/notes into a brief study note that persists in memory.
- Grammar is forced to JSON for MCQ + Fact‑Cards (schema cannot carry citations).
- `--log PATH` appends per‑step JSONL records to PATH.
 - `--cards PATH` provides a JSON/JSONL dataset used as session context:
   - JSONL lines can be `{"text":"..."}` (best: LEARN extracts option‑quoted cards on the fly) or `{ "cards": [...] }` to pre‑supply cards.
 - `--cards-freeze` uses the provided `cards` as‑is (skips LEARN). When freezing without pre‑built option cards, the simulator injects per‑option PRO stubs (≤15‑token quotes) at answer time so evidence gating still applies.
 - `--cards-budget N` caps the number of cards kept in memory (default 10). Recommended: 9–10 (5 option PRO cards + up to 2 pinned snippets + 2 spare) when retrieval is ON.

## 4) Fact‑Cards v2 (LEARN/USE)

Goal: extract discriminative, option‑linked “cards” with short verbatim quotes and then cite them when answering.

LEARN (extraction) JSON schema
```
{"cards": [
  {"id": "f1",
   "claim": "...",
   "quote": "...",                   # verbatim ≤ 15 tokens
   "where": {"scope": "option"|"context", "option_index?": 0, "start": 0, "end": 12},
   "tags": ["<skill_id>", ...],      # must include current skill_id
   "hypothesis": "...",              # why/what this supports
   "polarity": "pro"|"con"}
]}
```

LEARN rules (enforced centrally)
- Quotes are clamped centrally to ≤ 15 tokens after LEARN (single pass); eligibility for PRO ids also requires ≤ 15 tokens.
- At least one card per option must cite a verbatim substring via `where.scope="option"` and `where.option_index=i`.
- If tools provide `retrieved_snippets`, include at least one card quoting from any retrieved snippet (not only the first).
- Orchestrator now routes normalization/budgeting through `sim.factcards_manager.FactCardManager`, so the LEARN helper enforces tags, length, and per-option PRO stubs before scoring.

USE (answer) JSON schema
```
{"options": [
  {"id": "A|B|C|D|E", "hypothesis": "...", "score": 0.0..1.0, "citations": ["f1","f3"]},
  ...
],
 "choice": "A|B|C|..."}
```

USE rules
- For the selected choice, include ≥1 citation that points to a card whose `where.option_index` equals that option (option‑linked witness).
- All cited cards must have the current `skill_id` tag and `quote` length ≤ 15 tokens.
- Scores should roughly sum to 1.
- The learner’s fact-card prompts are constructed via `sim.prompts_mcq`, which pins the JSON schema (options + witness) shared across providers.

Why it helps: explicit option‑linked quotes remove ambiguity and increase `witness_pass` by 10–20 points in practice.

## 5) Evidence‑Gated Credit (MCQ)

Definitions (see `sim/credit.py`)
- Coverage: token overlap between the gold option text and the concatenation of cited quotes.
- Witness: pick the option whose tokens overlap most with the cited quotes; pass if it equals the correct option.
- Credited: `correct && coverage ≥ τ && witness_pass` (and, when tools are ON, at least one cited quote must be a substring of any retrieved snippet).
- τ is `coverage_tau` (default 0.40; adjust via `TUTOR_COVERAGE_TAU`).

Abstention policy (IDK)
- Hard‑evidence only: the runner abstains (IDK) only if the chosen option is missing an option‑linked quote, coverage < τ, or the chosen option did not cite its required PRO card id.
- Confidence target (`--target-confidence t`) does not cause abstention; it is used only to compute `evaluation.calibrated_score`.

Tie‑break (witness‑first)
- After self‑consistency voting, if cited evidence strongly favors a different option, the orchestrator re‑picks that option before gating.

## 6) Analyzer Metrics

Run:
```
.venv/bin/python -m scripts.analyze --log runs/live.jsonl
```

What it reports
- Accuracy: `acc_final`, `acc_auc`, per‑step cumulative curves.
- Credited: `credited_final`, `credited_auc`, and curves.
- Witness curves: `witness_final`, `witness_auc`.
- Early/Late: first vs last K steps, delta and two‑proportion p‑value; Wilson 95% CI for credited.
- Retention: lag‑based re‑tests (lag 3–5), deltas and p‑values.
- Timing: `overall.timing.{median,mean,total}_seconds`.
- Usage: `overall.usage.tokens_per_step_mean`, `tokens_total`.
 - New fields in per‑step logs: `evaluation.calibrated_score`, `evaluation.abstain_reason` (when IDK), and `card_validation` with per‑step card issues.
- Alias: per‑family B‑credited with counts and Wilson lower/upper bounds (and pooled across runs).

## 7) Context Learning & Stability (CL&S)

Goal: quantify in‑context learning speed and stability/forgetting across a long session as context grows.

How to run (per model)
- Use the stateful student with Fact‑Cards, citations, and retrieval so “credited” reflects grounded answers.

Examples (60‑step runs)
```
.venv/bin/python -m sim.cli \
  --student stateful-llm --provider deepinfra --model deepseek-ai/DeepSeek-R1 \
  --steps 60 --closed-book \
  --fact-cards --require-citations \
  --self-consistency 2 \
  --use-tools --tools tfidf_retriever \
  --progress \
  --log runs/window/deepinfra_deepseek-ai-DeepSeek-R1_N60.jsonl

.venv/bin/python -m sim.cli \
  --student stateful-llm --provider deepinfra --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --steps 60 --closed-book \
  --fact-cards --require-citations \
  --self-consistency 2 \
  --use-tools --tools tfidf_retriever \
  --progress \
  --log runs/window/deepinfra_mistralai-Mixtral-8x7B-Instruct-v0.1_N60.jsonl

.venv/bin/python -m sim.cli \
  --student stateful-llm --provider openai \
  --steps 60 --closed-book \
  --fact-cards --require-citations \
  --self-consistency 2 \
  --use-tools --tools tfidf_retriever \
  --progress \
  --log runs/window/openai_gpt-4.1_N60.jsonl
```

Aggregate and score
```
.venv/bin/python -m scripts.analyze --log runs/window/<slug>_N60.jsonl > runs/window/<slug>_N60.aggregate.json
.venv/bin/python -m scripts.cls_score --glob "runs/window/*_N60.jsonl" > runs/window/CLS_N60.csv
```

CL&S components
- LAUC: credited area under curve over steps (0..1).
- STAB: late‑minus‑plateau = mean(credited[last 10%]) − mean(credited[20%..40%]); negative implies forgetting.
- WIT: witness_pass_mean from analyzer (evidence alignment).
- Composite: `CL&S = 0.6·LAUC + 0.4·clip(WIT + STAB, 0, 1)`.

Forget onset (optional)
- `scripts.cls_score` reports `forget_onset_step` using a 5‑step moving average dropping ≥Δ (default 0.10) below the plateau and persisting for 5 steps. Tweak with `--delta`.

Tips
- Use `--steps 120` for stronger signals; `--self-consistency 2` is a good default.
- Keep `--fact-cards --require-citations --use-tools tfidf_retriever` to ensure credited reflects evidence, not guesses.
- The stateful student carries its memory and, when Fact‑Cards are present, returns citations to support credit.

## 8) Bayes Guardrails + Evidence‑Aware Selection

Purpose
- Learn policy from data (Bayes) → enforce at runtime:
  - Token band (safe range with a U‑shaped effect; avoid the mid‑range trough).
  - Tutor talk by domain (lean/neutral/rich from posterior slopes).
  - Optional turns trigger (resolve‑or‑escalate beyond mean_steps + ~1 SD).
- Restore credited while staying in‑band by choosing the candidate with the strongest evidence (witness/coverage), not just majority vote.

Train Bayes (offline; no LLM calls)
```
.venv/bin/python -m scripts.aggregate_runs --runs-dir runs --out-jsonl runs/_aggregated/all_steps.jsonl.gz --out-csv runs/_aggregated/all_steps_flat.csv.gz
.venv/bin/python -m scripts.session_view --steps-csv runs/_aggregated/all_steps_flat.csv.gz --out-csv runs/_aggregated/session_view.csv.gz
.venv/bin/python -m scripts.bayes.session_bayes_report --csv runs/_aggregated/session_view.csv.gz --out-root runs/_aggregated/bayes --target-accept 0.98 --tune 2000
```

Artifacts
- `runs/_aggregated/bayes/tables/guardrails.json` — z* and token bands (run + per‑step) and mean_steps.
- `runs/_aggregated/bayes/tables/talk_slopes_by_domain_*.csv` — posterior talk slopes and `prob_positive` per domain.

Run with guardrails + evidence rerank (Mixtral example)
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

Operational notes
- Guardrail alerts appear as `guardrail_alerts.tokens_step.state ∈ {below_band,in_band,near_trough,above_band}` with suggested actions.
- `--tokens-autonudge` toggles gentle dial shifts when outside the band (leaner/richer). Without it, alerts log only.
- Adjust trough sensitivity with `--trough-margin 0.10..0.30`.
- For model‑specific bands, re‑fit Bayes on filtered logs (e.g., Mixtral‑only) and point `--guardrails` to the new JSON.
 - To stabilize extraction vs answering costs, use `--sc-extract 5 --self-consistency 3`.

## 10) Temporal Holdout & RAG Delta

Goal: verify context grounding with post‑cutoff sources and measure the impact of retrieval (RAG) over that holdout.

Prepare a holdout file
- A small synthetic, dated set is provided: `data/holdout/postcutoff_text.jsonl`.
- Recommended format is text JSONL: one `{ "text": "..." }` per line. This lets LEARN extract option‑linked quotes on the fly.

Run (citations ON, LEARN from text, RAG ON; budget 8; IDK on)
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
  --log runs/holdout/postcutoff_rag_on.jsonl
```

RAG delta (compare with retrieval OFF)
```
ANON_SEED=424242 TUTOR_COVERAGE_TAU=0.25 \
.venv/bin/python -m sim.cli \
  --student stateful-llm --provider deepinfra \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --steps 120 --closed-book \
  --fact-cards --require-citations \
  --self-consistency 3 --sc-extract 1 \
  --reflection-every 10 \
  --cards data/holdout/postcutoff_text.jsonl \
  --log runs/holdout/postcutoff_rag_off.jsonl

.venv/bin/python -m scripts.rag_delta --on runs/holdout/postcutoff_rag_on.jsonl --off runs/holdout/postcutoff_rag_off.jsonl
```

Counterfactuals (optional)
```
.venv/bin/python -m scripts.perturb_cards --in data/holdout/postcutoff_text.jsonl --out runs/holdout/postcutoff_cf.jsonl --percent 7
.venv/bin/python -m sim.cli \
  --student stateful-llm --provider deepinfra \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --steps 60 --closed-book \
  --fact-cards --require-citations \
  --self-consistency 3 --sc-extract 1 \
  --use-tools --tools tfidf_retriever \
  --cards runs/holdout/postcutoff_cf.jsonl \
  --progress \
  --log runs/holdout/postcutoff_cf_run.jsonl
```

Analysis
```
.venv/bin/python -m scripts.analyze --log runs/holdout/postcutoff_rag_on.jsonl > runs/holdout/postcutoff_rag_on.aggregate.json
```

Example findings (recent)
- Mixtral‑8x7B (120 steps): LAUC≈0.055, STAB≈0.083, WIT≈0.283 → CL&S≈0.180 (most learning + stability).
- DeepSeek‑R1 (60 steps): WIT≈0.333, LAUC≈0, STAB≈0 → strong evidence alignment but flat learning signal at N=60.
- GPT‑4.1 (60 steps): WIT≈0.250, LAUC≈0, STAB≈0 → flat at N=60; try N=120.

## 10b) Updated Recommendations (Sep 2025)

These dials consistently improved Mixtral on the post‑cutoff holdout and reduced card issues in recent runs.

- Student/provider/model: `--student stateful-llm --provider deepinfra --model mistralai/Mixtral-8x7B-Instruct-v0.1`
- Evidence: `--fact-cards --require-citations --use-tools --tools tfidf_retriever`
- Voting: `--self-consistency 3` (answering) and `--sc-extract 5` (LEARN)
- Budget: `--cards-budget 10` (prevents eviction of per‑option PRO cards)
- Abstention: `--idk --target-confidence 0.60`
- Coverage threshold: `TUTOR_COVERAGE_TAU=0.40` (only consider 0.35 if coverage narrowly misses and the attribution gap remains ≤ 0.05)

Recommended command (RAG‑ON, stronger LEARN, budget 10)

```
ANON_SEED=424242 TUTOR_COVERAGE_TAU=0.40 \
.venv/bin/python -m sim.cli \
  --student stateful-llm --provider deepinfra \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --steps 120 --closed-book \
  --fact-cards --require-citations \
  --self-consistency 3 --sc-extract 5 \
  --use-tools --tools tfidf_retriever \
  --cards data/holdout/postcutoff_text.jsonl \
  --cards-budget 10 \
  --idk --target-confidence 0.60 \
  --progress \
  --log runs/holdout/postcutoff_rag_on_scx5_cb10.jsonl
```

Isolate answering vs extraction (freeze)

Use the same dials with `--cards-freeze` to skip LEARN and evaluate the USE phase only. The simulator injects ephemeral per‑option PRO stubs (≤ 15‑token quotes) at answer time so evidence gating remains meaningful while extraction is held fixed.

```
ANON_SEED=424242 TUTOR_COVERAGE_TAU=0.40 \
.venv/bin/python -m sim.cli \
  --student stateful-llm --provider deepinfra \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --steps 120 --closed-book \
  --fact-cards --require-citations \
  --self-consistency 3 --sc-extract 5 \
  --use-tools --tools tfidf_retriever \
  --cards data/holdout/postcutoff_text.jsonl \
  --cards-freeze \
  --cards-budget 10 \
  --idk --target-confidence 0.60 \
  --progress \
  --log runs/holdout/postcutoff_rag_on_scx5_cb10_freeze.jsonl
```

Readouts to watch
- Credited_final and credited_auc (effectiveness), witness_final (evidence alignment), attribution gap (raw − credited, target ≤ 0.05), card validation (`quote_not_in_option`, `missing_option_card:i`).

## 10c) Best Runs Summary (Sep 2025)

Holdout — Mixtral (DeepInfra)
- Stronger LEARN (SCx=5, budget=10, RAG‑ON):
  - credited_final ≈ 0.315; raw_final ≈ 0.389; gap ≈ 0.074
  - credited_auc ≈ 0.337; witness_final ≈ 0.362
  - median_step_seconds ≈ 78.1; tokens_per_step ≈ 11.38k
  - RAG delta (vs OFF) ≈ +0.117 (credited)
- Fixed retrieval separation (SCx=1, budget=8, RAG‑ON):
  - credited_final ≈ 0.300; raw_final ≈ 0.350; gap ≈ 0.050
  - witness_final ≈ 0.317; tokens_per_step ≈ 6.81k; median_step_seconds ≈ 31.6
  - RAG delta (vs OFF) ≈ +0.102 (credited)
- RAG‑OFF baseline:
  - credited_final ≈ 0.198; witness_final ≈ 0.364; gap ≈ 0.000
  - median_step_seconds ≈ 35.4; tokens_per_step ≈ 5.42k
- Freeze (USE‑only, LEARN disabled):
  - credited_final ≈ 0.000; witness_final ≈ 0.192
  - median_step_seconds ≈ 25.5; tokens_per_step ≈ 3.30k
  - Interpretation: uplift in credited comes from LEARN (extraction), not answering alone.

Notes
- SC‑extract=5 and a larger cards budget (10) reduce card issues (fewer `quote_not_in_option`, fewer `missing_option_card:i`).
- Keep attribution gap ≤ 0.05 by tuning τ and ensuring chosen options cite their PRO card IDs first in USE.


## 8) Alias‑Swap Experiments

Purpose: prove robustness against priors by scrambling aliases between A/B variants.

Run an alias suite:
```
.venv/bin/python -m scripts.experiment \
  --config docs/iclsim/experiments/icl_proof_live_classical_on.yaml \
  --out runs/alias_on_oss_v2

.venv/bin/python -m scripts.analyze --dir runs/alias_on_oss_v2 > runs/alias_on_oss_v2_summary.json
```

Success bar: B‑credited > 0.0 indicates the model is using session context.

## 9) Multi‑Model Benchmark

Runner: `scripts/run_models.sh` (writes summary CSV to `SUMMARY_OUT`, defaults to `runs/compare/summary.csv`).

Examples
```
chmod +x scripts/run_models.sh
mkdir -p runs/compare && \
SUMMARY_OUT=runs/compare/summary_smoke.csv QUIET=0 BATCH_PAR=1 SC_ANSWER=1 STEPS=4 BATCHES=2 TAU=0.40 USE_RETRIEVER=0 \
./scripts/run_models.sh
```

Parallel (bars interleave):
```
SUMMARY_OUT=runs/compare/summary_smoke.csv QUIET=0 BATCH_PAR=2 SC_ANSWER=1 STEPS=4 BATCHES=2 TAU=0.40 USE_RETRIEVER=0 \
./scripts/run_models.sh
```

What it saves (per model)
- `runs/compare/<provider>_<model>/batch_*.jsonl`
- `runs/compare/<provider>_<model>/aggregate.json`
- `runs/compare/<provider>_<model>/alias_classical_on/alias_summary.json`
- CSV row in `runs/compare/summary_*.csv`.

Tuning knobs
- `BATCH_PAR`: concurrent batches per model (typical 2–4; lower if throttled).
- `SC_ANSWER`: 1–3 (1 is faster; 3 improves stability).
- `QUIET`: 0 shows per‑batch bars; 1 silences them.

Interpreting the CSV
- Effectiveness: `credited_final_mean`, `credited_auc_mean`, alias `credited_B_mean`.
- Efficiency: `median_step_seconds`, `tokens_per_step_mean` (if available).
- Balance: compute “credited per minute” or “seconds per credited.” Plot Pareto (effectiveness vs efficiency).

## 10) Architecture & Data Flow

Key modules
- `sim/orchestrator.py`: Builds tasks, injects CONTEXT/tools, runs learner with self‑consistency, computes evaluation/crediting, logs JSONL.
- `sim/learner.py`: Students:
  - `LLMStudent` (OpenAI or DeepInfra) implements MCQ/SAQ/etc. and Fact‑Cards LEARN/USE prompts.
  - `AlgoStudent` (closed‑book overlap baseline).
- `tutor/llm_openai.py`, `tutor/llm_deepinfra.py`: Tutor/student wrappers with JSON mode; attach usage/timing.
- `sim/tools.py`: Retrieval tools (`retriever`, `tfidf_retriever`).
- `scripts/analyze.py`: Aggregates metrics across logs.
- `scripts/experiment.py`: Runs YAML‑defined suites (e.g., alias_swap).
- `scripts/run_models.sh`: Multi‑model benchmark runner with ETA.

Record shape (MCQ)
```
{
  "task": {"type": "mcq", "stem": "...", "options": [..], "correct_index": 0},
  "presented_stem": "CONTEXT...\nQUESTION: ...",
  "answer": {"chosen_index": 2, "votes": [2,2,1], "citations": ["f1","f3"]},
  "evaluation": {"correct": true, "citations_evidence": {"coverage": 0.6, "witness_pass": true, "credited": true}},
  "duration_ms": 5230,
  "student_usage": {"total_tokens": 432, ...}
}
```

## 11) Troubleshooting

BrokenPipeError noise
- Cause: a child process prints to stdout after the main pipe (tee/terminal) closes.
- Mitigation: `sim.cli` suppresses stdout when `--log` is used; final prints are guarded. The benchmark runner redirects batch stdout and prints ETA on stderr.

Tee path wraps to next line
- Keep `tee runs/compare/summary.csv` on one line; a wrapped path points tee at the directory and fails.

429 / rate limits
- Reduce `BATCH_PAR`; disable retrieval; lower `STEPS`; try a faster model.

Slow batches
- Set `SC_ANSWER=1`; reduce `STEPS`; choose faster open‑weights; keep `BATCH_PAR` at 2–3.

JSON formatting issues
- Prompts explicitly say “JSON”; wrappers request `response_format=json_object` and fall back if needed.

## 11) Extending Safely

- Add new tools via `sim/tools.py` and list them in `REGISTRY`.
- Add new tasks by extending `sim/tasks.py` and learner methods.
- Modify prompts in `sim/learner.py` (LEARN/USE) and keep schemas stable.
- Analyzer: add new metrics in `scripts/analyze.py` (ensure backwards compatibility).

## 12) Cost & Latency Playbook

Use these presets to cut cost/latency without refactors.

1) Fast & cheap (live LLM)

```
unset TUTOR_MOCK_LLM
.venv/bin/python -m sim.cli \
  --student llm --provider deepinfra --model deepseek-ai/DeepSeek-R1 \
  --steps 4 --closed-book \
  --self-consistency 1 \
  --log runs/cheap_fast.jsonl
```

Why: SC=1 reduces calls ~linearly; skipping retrieval/tools and rich tutor output keeps prompts short.

2) Zero‑cost debug loop (offline)

```
export TUTOR_MOCK_LLM=1
.venv/bin/python -m sim.cli --student algo --steps 5 --closed-book
```

Why: mock tutor + algorithmic student → $0 iteration while tuning prompts/flags/logging.

3) Quick model shootout (Pareto pick)

```
chmod +x scripts/run_models.sh
mkdir -p runs/compare && \
SUMMARY_OUT=runs/compare/summary_smoke.csv QUIET=0 BATCH_PAR=2 SC_ANSWER=1 STEPS=4 BATCHES=2 USE_RETRIEVER=0 \
./scripts/run_models.sh
```

Pick winners by effectiveness (credited_*), efficiency (median_step_seconds, tokens_per_step_mean), and balance (seconds per credited). Then trim heavy models from `MODELS`.

Biggest levers (impact → effort)
- Drop self‑consistency to 1: `--self-consistency 1` or `SC_ANSWER=1` — cuts tokens/latency roughly linearly; slight stability loss.
- Avoid retrieval/tools: omit `--use-tools` — prevents context bloat; loses some RAG gains.
- Skip rich tutor output: omit `--rich` — shorter prompts; less metadata.
- Short batches: `--steps 3..5` — faster wall‑clock; more runs to aggregate.
- Moderate parallelism: `BATCH_PAR=2..3` — concurrency without 429s; too high can throttle.
- Pick faster models: use the shootout to optimize seconds per credited.
- Offline for dev: `TUTOR_MOCK_LLM=1`, `--student algo` — free iteration; not representative of final quality.
- JSON reliability: keep the word “JSON” in prompts; optional one‑shot example to reduce retries.

See where time/tokens went

```
.venv/bin/python -m scripts.analyze --log runs/cheap_fast.jsonl
```

Reports `overall.timing.{median,mean,total}_seconds`, `overall.usage.tokens_per_step_mean`, plus per‑step `duration_ms` and `student_usage.total_tokens`.

Guardrails
- 429 / throttling: lower `BATCH_PAR`, disable `--use-tools`, reduce `--steps`.
- Bars noisy/stalls: keep `BATCH_PAR≈2`, shorter batches; suite ETA prints on stderr.

## 13) Quote Validation (Offsets‑First + Canonicalized)

- Offsets are authoritative for option‑scope cards: when `where.scope="option"` and `where.start/end` exist and are valid, the validator derives the quote by slicing the presented option text (clamped to ≤15 tokens if necessary). This guarantees quotes are true substrings and preserves punctuation/hyphens.
- Canonicalized substring check: if offsets are missing, the validator compares canonical forms: NFKC + casefold + dash/space unification. The rule remains strict (quotes must be substrings); this only avoids false negatives from Unicode/typographic drift.
- Logs: `card_validation.validator_modes = {offset, canon-substring, fail}` helps audit behavior. Healthy runs should show `fail: 0`.
- FREEZE: per‑option PRO stubs are sliced directly from the displayed options and include `where.start/end`, so drift cannot occur.

## 14) Multi‑Model Smoke Runs (3 terminals)

Launch nine 30‑step smokes across common DeepInfra models by splitting them into three terminals.

Shared prep (in each terminal):
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

Tip: set `TUTOR_COVERAGE_TAU=0.25` to match the stronger Mixtral profile used in holdout scoring.
