# Agent Next Steps — Longer Live Sessions

## Goals
- Increase step count for the DeepSeek (DeepInfra) student under closed‑book, anonymized conditions.
- Improve JSON reliability and measure self‑consistency gains.
- Evaluate alias‑swap evidence crediting with the DeepSeek student.

## Recommended Batching Plan
- Run 5–10 batches of 3–5 steps each to avoid environment timeouts.
- Use `--self-consistency 3` for MCQ; consider 5 if latency allows.
- Commands (example with `openai/gpt-oss-20b`):
  - `.venv/bin/python -m sim.cli --student llm --provider deepinfra --model openai/gpt-oss-20b --steps 3 --closed-book --rich --self-consistency 3 --log runs/oss_batch1.jsonl`
  - Repeat with `oss_batch2.jsonl`, ..., `oss_batchN.jsonl`.
- Aggregate:
  - `.venv/bin/python -m scripts.analyze --log runs/oss_batch1.jsonl --log runs/oss_batch2.jsonl --log runs/oss_batch3.jsonl`

## JSON Robustness for OSS
- Keep prompts explicit: include the word "JSON" and the required keys.
- Fallback already implemented: retry without `response_format`, parse letters (A/B/C).
- Optional enhancement: one‑shot JSON example in system prompt; reprompt once on malformed output.

## Alias‑Swap with DeepSeek Student
- Extend `scripts/experiment.py` to accept `--provider`/`--model` (or a config field) to run alias family tasks with DeepSeek.
- Start with `classical-conditioning-01` and `operant-conditioning-01` for 10–20 steps; measure B‑credited.

## Retrieval‑Augmented Context
- Enable `--use-tools` and include `tfidf_retriever` to append anonymized snippets to CONTEXT.
- Compare runs with and without tools (same seed) to estimate retrieval gain.

## Tracking
- After each batch, append to a run ledger (CSV/JSON) with: timestamp, steps, SC, acc_final, acc_auc.
- Keep raw JSONL logs under `runs/` for reproducibility.

