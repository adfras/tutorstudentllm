# Agent Next Steps — Longer Live Sessions

## Goals
- Increase step count for the DeepSeek (DeepInfra) student under closed‑book, anonymized conditions.
- Improve JSON reliability and measure self‑consistency gains.
- Evaluate alias‑swap evidence crediting with the DeepSeek student.

## Recommended Batching Plan
- Run 5–10 batches of 3–5 steps each to avoid environment timeouts.
- Use `--self-consistency 3` for MCQ; consider 5 if latency allows. Enable `--adaptive-sc --sc-quorum 2` to early‑stop on easy items.
- Commands (example with `openai/gpt-oss-20b`):
  - `.venv/bin/python -m sim.cli --student llm --provider deepinfra --model openai/gpt-oss-20b --steps 3 --closed-book --rich --self-consistency 3 --adaptive-sc --log runs/oss_batch1.jsonl`
  - Repeat with `oss_batch2.jsonl`, ..., `oss_batchN.jsonl`.
- Aggregate:
  - `.venv/bin/python -m scripts.analyze --log runs/oss_batch1.jsonl --log runs/oss_batch2.jsonl --log runs/oss_batch3.jsonl`

## Multi‑Model Benchmark (New)

Use the suite runner to compare models with Fact‑Cards v2 + citations and SC; consider adding ToT on hard items via uncertainty gating:

```
chmod +x scripts/run_models.sh
mkdir -p runs/compare && \
SUMMARY_OUT=runs/compare/summary_smoke.csv QUIET=0 BATCH_PAR=2 SC_ANSWER=1 STEPS=4 BATCHES=2 TAU=0.40 USE_RETRIEVER=0 \
./scripts/run_models.sh
```

Then shortlist winners (edit `MODELS` in the script) and run (optionally add `--uncertainty-gate --escalate-reasoning` in CLI templates):

```
SUMMARY_OUT=runs/compare/summary_final.csv QUIET=0 BATCH_PAR=2 SC_ANSWER=3 STEPS=12 BATCHES=5 TAU=0.40 USE_RETRIEVER=1 \
./scripts/run_models.sh
```

Analyzer adds early/late p‑values, Wilson CIs, timing/tokens. Alias suite runs per model; see `alias_summary.json` in each model’s folder.

## JSON Robustness for OSS
- Keep prompts explicit: include the word "JSON" and the required keys.
- Fallback already implemented: retry without `response_format`, parse letters (A/B/C).
- Optional enhancement: one‑shot JSON example in system prompt; reprompt once on malformed output.

## Alias‑Swap with DeepSeek Student
- Extend `scripts/experiment.py` to accept `--provider`/`--model` (or a config field) to run alias family tasks with DeepSeek.
- Start with `signal-association-01` and `feedback-learning-01` for 10–20 steps; measure B‑credited.

## Retrieval‑Augmented Context & IDK
- Enable `--use-tools --tools tfidf_retriever` to append anonymized snippets to CONTEXT. For few‑shot, prefer embedding KNN: `--shots <file> --shots-k 6 --shots-selector knn --shots-embed-backend st --shots-diverse`.
- Reserve room for evidence: set `--cards-budget 8` (5 options + 2 snippets + 1 spare).
- Enable calibrated abstention: add `--idk --target-confidence 0.60`. Abstention is hard‑evidence only; t is used only for `calibrated_score`.
- Compare runs with and without tools (same seed) to estimate retrieval gain.

## Controllers & APE (optional)
- Controllers: `--controller ltm` (PLAN→SOLVE) or `--controller tot --tot-width 2 --tot-depth 2` to seed a controller vote before SC.
- APE header: pick one with `scripts/ape_optimize.py` and inject with `--ape-header docs/ape/header.json`.

## Tracking
- After each batch, append to a run ledger (CSV/JSON) with: timestamp, steps, SC, acc_final, acc_auc.
- Keep raw JSONL logs under `runs/` for reproducibility.
