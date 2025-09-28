# Agent Runbook — ICL Simulator

This document guides an automation agent or contributor on how to operate and extend the project.

## Purpose

Deliver a minimal, fast, general‑purpose In‑Context Learning (ICL) simulator that:
- Runs closed‑book, anonymized tutor→student simulations.
- Uses a fixed OpenAI model (`gpt-5-nano-2025-08-07`) with strict JSON outputs.
- Supports pluggable learners (LLM student, algorithmic baseline) and task types (starting with MCQ).

## Operating Instructions

1) Environment
- Ensure `.env` contains `OPENAI_API_KEY`.
- Python venv recommended: `python3 -m venv .venv && source .venv/bin/activate`.

2) Install & Run
- `pip install -r requirements.txt`
- Run simulator (mock): `export TUTOR_MOCK_LLM=1; python -m sim.cli --steps 3 --closed-book --rich --progress`
- Algorithmic baseline: `python -m sim.cli --student algo --steps 5 --closed-book --notes-file data/notes_algo.txt`
- LLM student (OpenAI tutor + DeepSeek student via DeepInfra):
  - Ensure `.env` has `DEEPINFRA_API_KEY` and set a model in `DEEPINFRA_MODEL` or pass `--model`.
 - Example: `python -m sim.cli --student llm --provider deepinfra --model deepseek-ai/DeepSeek-R1 --steps 5 --closed-book --progress`
  - With calibrated abstention (hard‑evidence only): add `--idk --target-confidence 0.60` (t affects scoring, not gating).
 - Live run with open-source student (3-step batch):
 - `.venv/bin/python -m sim.cli --student llm --provider deepinfra --model openai/gpt-oss-20b --steps 3 --closed-book --rich --self-consistency 3 --progress --log runs/deepseek_live.jsonl`
 - Post-cutoff holdout with fixed Fact-Cards (freeze LEARN):
   - `.venv/bin/python -m sim.cli --student stateful-llm --provider deepinfra --model mistralai/Mixtral-8x7B-Instruct-v0.1 --steps 60 --closed-book --fact-cards --require-citations --self-consistency 2 --use-tools --tools tfidf_retriever --cards PATH_TO_POSTCUTOFF_CARDS.jsonl --cards-freeze --progress --log runs/holdout/mixtral_postcutoff.jsonl`
 - Post-cutoff holdout with on-the-fly cards (recommended):
   - Use `data/holdout/postcutoff_text.jsonl` and omit `--cards-freeze` so LEARN extracts option-quoted cards.
   - `ANON_SEED=424242 TUTOR_COVERAGE_TAU=0.40 .venv/bin/python -m sim.cli --student stateful-llm --provider deepinfra --model mistralai/Mixtral-8x7B-Instruct-v0.1 --steps 120 --closed-book --fact-cards --require-citations --self-consistency 3 --sc-extract 1 --use-tools --tools tfidf_retriever --cards data/holdout/postcutoff_text.jsonl --cards-budget 8 --idk --target-confidence 0.60 --progress --log runs/holdout/postcutoff_rag_on.jsonl`

Full one‑liner (fresh venv, deps, run with progress):
`set -a; [ -f .env ] && . .env; set +a; unset TUTOR_MOCK_LLM; python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt && mkdir -p runs && .venv/bin/python -m sim.cli --student llm --provider deepinfra --model deepseek-ai/DeepSeek-R1 --steps 10 --closed-book --rich --progress --log runs/full_live.jsonl`

3) Health Checks
- Simulator runs and prints JSON with `results[]` records.
- Each record contains: `task`, `answer`, `evaluation`, and `presented_stem` (with CONTEXT when closed‑book).

4) Key Verification
- Offline mock mode has been removed. Ensure `.env` has `OPENAI_API_KEY` (tutor) and optionally `DEEPINFRA_API_KEY` for live runs.
- Internal checks: the tutor and DeepInfra wrappers perform a minimal chat call to verify reachability; see `verify_key_and_model()` in `tutor/llm_openai.py` and `tutor/llm_deepinfra.py`.

Synthetic Student
- Use `--student llm` (default) or `--student algo` with optional `--notes-file`.
- For DeepSeek via DeepInfra, use `--provider deepinfra` (or `--provider deepseek`) and set `--model` or `DEEPINFRA_MODEL`.

## Constraints & Policies

- Tutor model is hard‑locked to `gpt-5-nano-2025-08-07`. Student provider/model is selectable; defaults remain OpenAI.
- Use JSON response format and prompts that explicitly include the word “JSON” to comply with model requirements.
- UI should remain minimal and responsive; prefer short stems/options and minimal payloads.
- Do not log or persist PII besides username; keep stats aggregate only.
 - Memory gating: repeats only when not mastered and due by interval; allow a single immediate remediation after a wrong answer.

Optional hardening (Provable Novice Mode):
- `TUTOR_ANONYMIZE=1` → apply per‑user codebook + numeric scrambling to stems/options/rationales.
- `TUTOR_REQUIRE_CITATIONS=1` → credit only if correct AND coverage ≥ τ AND witness re‑pick agrees. Optional `TUTOR_COVERAGE_TAU` (default 0.4).
 - `--idk --target-confidence t` → abstain only on hard evidence failures (no option quote / low coverage / missing required PRO cite). `t` is used for calibrated_score, not gating.

Adaptivity is lightweight: per‑skill `mastery` (0..1) with 7‑day half‑life decay, misconception counters, and a simple policy: remediation → continue current until mastered → review due → advance when prereqs satisfied.

## Logs & Analysis
- Pass `--notes-file` to simulate closed‑book students.
- Use `sim.orchestrator.Orchestrator.run(..., log_path=...)` to write JSONL per step. Pass `--progress` in CLI to show a live progress bar (stderr).
 - Live outputs go in `runs/` (git‑ignored). Do not check reports into `docs/`.
 - Dials via env (optional): `TUTOR_ANONYMIZE=1`, `TUTOR_REQUIRE_CITATIONS=1`, `TUTOR_COVERAGE_TAU=0.35`. Anonymization is on by default; use `--no-anon` to disable.
- Anonymization seed: set `ANON_SEED` to force nonce mapping (e.g., `ANON_SEED=$RANDOM`).
 - Logging: the simulator auto-creates parent directories for `--log` paths.
- New fields: `evaluation.calibrated_score`, `evaluation.abstain_reason` (when IDK), and `card_validation` per step summarizing any card issues (e.g., missing option card, too-long quote).

### Model Dial Profiling

Use per-run aggregates to match student models with dial overrides:
- After a sweep, regenerate aggregates and guardrails (`scripts.aggregate_runs` → `scripts.session_view` → `scripts.bayes.session_bayes_report`). The newest tables live under `runs/_aggregated/bayes/tables/`.
- Build/update model capability summaries with `scripts.model_profiles.py`. This inspects `session_view.csv.gz`, buckets models by coverage/witness/credit gaps, and writes recommended overrides to `runs/_aggregated/model_profiles.yaml`.
- When launching guardrailed runs, `scripts/run_guardrailed_multimodel.sh` pulls the overrides automatically (e.g., evidence-limited models get higher `--sc-extract`, wider card budgets, and lighter `coverage_tau`). Override detection happens per DeepInfra model id; set `MODELS` to limit scope.
- To inspect the planned overrides without running anything: `.venv/bin/python -m scripts.model_profiles --model mistralai/Mixtral-8x7B-Instruct-v0.1 --show`.
- Keep the YAML under version control if you want deterministic dial selection between CI runs; otherwise treat it as a generated artifact alongside guardrails.

## Skill Map & Categories

- Skills are defined in `docs/iclsim/skill_map.general.yaml`.
- Category is computed as the top‑level parent of a skill; used to roll up stats.

## Code Pointers (central files)
- `sim/orchestrator.py` — core simulator & dials
- `sim/learner.py` — learners (LLM, algo)
- `sim/tasks.py` — task/evaluator (MCQ)
- `sim/anonymize.py` — anonymization utilities
- `tutor/llm_openai.py` — fixed OpenAI model wrapper
 - `scripts/analyze.py`, `scripts/rag_delta.py`, `scripts/report_runs.py`, `scripts/validate_cards.py`

## Extending Safely

- To add richer feedback: add `rationales` back to MCQ prompts and render them in the UI.
- To implement adaptivity: store mastery per skill and schedule review per `docs/iclsim/adaptive_algorithms.md`.
- To switch persistence: migrate to Postgres using `docs/iclsim/db_schema.sql` and replace `server/storage.py` with DB queries.

## Validation Checklist (Simulator)
- [ ] Mock LLM runs (TUTOR_MOCK_LLM=1)
- [ ] Algo student runs with notes
- [ ] CONTEXT included in `presented_stem` when closed‑book
- [ ] JSONL per‑step logging works when `log_path` set
 - [ ] DeepInfra student emits valid choices (fallback handles letter outputs)

## Longer Live Sessions

- Use multiple 3–5 step batches to fit latency constraints and then aggregate with:
  - `.venv/bin/python -m scripts.analyze --log runs/batch1.jsonl --log runs/batch2.jsonl`
- For alias‑swap evidence gating (baseline):
  - `.venv/bin/python -m scripts.experiment --config docs/iclsim/experiments/alias_live_algo.yaml --out runs/alias_live_algo`

## Context Learning & Stability (CL&S)

- Purpose: measure within‑session learning and stability/forgetting as context grows.
- Run long, stateful sessions per model (recommend 60–120 steps, `--self-consistency 2`):
  - Example (DeepInfra Mixtral):
    `.venv/bin/python -m sim.cli --student stateful-llm --provider deepinfra --model mistralai/Mixtral-8x7B-Instruct-v0.1 --steps 60 --closed-book --fact-cards --require-citations --self-consistency 2 --use-tools --tools tfidf_retriever --progress --log runs/window/deepinfra_mistralai-Mixtral-8x7B-Instruct-v0.1_N60.jsonl`
  - OpenAI baseline:
    `.venv/bin/python -m sim.cli --student stateful-llm --provider openai --steps 60 --closed-book --fact-cards --require-citations --self-consistency 2 --use-tools --tools tfidf_retriever --progress --log runs/window/openai_gpt-4.1_N60.jsonl`
- Analyze and score:
  - `.venv/bin/python -m scripts.analyze --log runs/window/<slug>_N60.jsonl > runs/window/<slug>_N60.aggregate.json`
  - `.venv/bin/python -m scripts.cls_score --glob "runs/window/*_N60.jsonl" > runs/window/CLS_N60.csv`
- CL&S = 0.6·LAUC + 0.4·clip(WIT + STAB, 0, 1); see `scripts/cls_score.py`.

## Troubleshooting

- 400: temperature/JSON format — ensure prompts do not set `temperature` and include “JSON” when using `response_format=json_object`.
- DNS/connectivity — verify outbound HTTPS and that `OPENAI_API_KEY` is present.
- Missing skills — check `docs/iclsim/skill_map.general.yaml` and file path.
 - If credited=0 while correct>0 — ensure `TUTOR_REQUIRE_CITATIONS` off, or send `citations_text`/`citations[]` with answers and lower `TUTOR_COVERAGE_TAU` for testing.

## Dev Notes

- Tests: the repo ships tests that run fully offline with `TUTOR_MOCK_LLM=1`. `pytest` is not listed in `requirements.txt`; install it in your venv if needed: `.venv/bin/pip install pytest` then run `.venv/bin/python -m pytest -q`.
- Code hygiene: minor dead-code cleanup removed an unused helper in `tutor/utils.py` and pruned unused imports/locals. No functional changes.
