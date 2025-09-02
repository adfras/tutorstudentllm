# ICL Live Run Report — 2025‑09‑02

## Overview

- Tutor LLM: `gpt-5-nano-2025-08-07` (OpenAI, JSON prompts)
- Student LLM: DeepInfra OSS model `openai/gpt-oss-20b`
- Mode: closed‑book, anonymized stems/options, rich MCQs
- JSON handling: strict JSON requested; OSS fallback parses letter choices and retries without `response_format` when necessary

## Configurations Executed

1) Live, 3‑step (initial):
   - Cmd: `.venv/bin/python -m sim.cli --student llm --provider deepinfra --model openai/gpt-oss-20b --steps 3 --closed-book --rich --log runs/deepseek_live.jsonl`
   - Result: Student returned null choices (JSON compliance issue) → overall 0/3 correct recorded.

2) Live, 3‑step with robust fallback:
   - Cmd: `.venv/bin/python -m sim.cli --student llm --provider deepinfra --model openai/gpt-oss-20b --steps 3 --closed-book --rich --log runs/deepseek_live2.jsonl`
   - Result: 2/3 correct (choices parsed and evaluated).

3) Live, 3‑step with self‑consistency (SC=3) and context notes:
   - Cmd: `.venv/bin/python -m sim.cli --student llm --provider deepinfra --model openai/gpt-oss-20b --steps 3 --closed-book --rich --self-consistency 3 --notes-file data/notes_novice_quick.txt --log runs/deepseek_live_batch1.jsonl`
   - Result: 1/3 correct.

Aggregate across live logs: 3/9 correct (≈33%).

## Alias‑Swap Evidence Gating (Baseline)

- Config: `docs/iclsim/experiments/alias_live_algo.yaml` (algorithmic student, closed‑book notes accumulation)
- Cmd: `.venv/bin/python -m scripts.experiment --config docs/iclsim/experiments/alias_live_algo.yaml --out runs/alias_live_algo`
- Summary (from `runs/alias_live_algo/summary.json`):
  - MCQ acc_final_mean: 0.80; acc_auc_mean: 0.514
  - Credited on alias‑B (coverage ≥ τ and witness pass):
    - classical‑conditioning‑01 → credited_B_mean ≈ 0.8
    - operant‑conditioning‑01 → credited_B_mean ≈ 0.8

## Interpretation

- JSON Compliance: OSS models via DeepInfra may ignore `response_format=json_object`. Adding retry without the flag and tolerant parsing of letter choices improves reliability.
- Early Accuracy: With anonymization and closed‑book constraints, initial DeepSeek performance is modest (≈33% on small samples). Self‑consistency (SC=3) helped in some cases but not uniformly on this small run.
- Evidence Gating: The alias‑swap pipeline and coverage+witness crediting logic function correctly on the algorithmic baseline; it’s suitable for benchmarking before running OSS LLMs on alias tasks.

## Recommendations

1) Longer Sessions via Batching
   - Run multiple 3–5 step batches with `--self-consistency 3` and aggregate using `scripts.analyze`.
   - Example: run 5 batches (total 15 steps), then aggregate.

2) Prompt Hardening for OSS
   - Add a one‑shot JSON example to the system message for DeepInfra provider.
   - Accept both `chosen_index` and letter keys; reject malformed outputs with a one‑time reprompt.

3) Alias‑Swap with DeepSeek
   - Extend experiment harness to accept `--provider/--model` and run alias families with DeepSeek student, measuring B‑credited.

4) Tool‑Augmented Context
   - Enable `--use-tools` with `tfidf_retriever` to append snippets to CONTEXT and test retrieval‑augmented closed‑book runs.

5) Tracking & Dashboards
   - Use `scripts.analyze` outputs to generate stepwise cumulative accuracy plots over longer sessions.

## Artifacts

- Live logs: `runs/deepseek_live.jsonl`, `runs/deepseek_live2.jsonl`, `runs/deepseek_live_batch1.jsonl`
- Alias experiment summary: `runs/alias_live_algo/summary.md`, `runs/alias_live_algo/summary.json`

