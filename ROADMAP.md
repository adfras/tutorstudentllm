# Roadmap — General-Purpose In‑Context Learning (ICL) Simulator

This roadmap organizes work into phases to reach a robust, multi‑domain, closed‑book ICL simulator.

## Phase 1 — Task Coverage & Logging (Now)
- [x] MCQ task type and evaluator
- [x] Closed‑book anonymization of stems/options/context
- [x] Self‑verify dial (two answers + agreement)
- [x] JSONL per‑step logging (presented_stem with CONTEXT)
- [x] SAQ task type and evaluator (grade)
- [x] Run IDs + session metadata in logs

## Phase 2 — Evaluation & Analytics
- [x] JSONL schema with run header and per‑step `ts`
- [x] Analysis script: learning curves (instant/cumulative), multi‑run aggregator
- [~] Ablations: grouped by dials (context position, verify, etc.)
- [ ] Error‑driven learning dials: spacing, rare‑example emphasis

## Phase 3 — Learner Extensions
- [x] Self‑consistency (N answers → majority vote) — MCQ
- [x] Self‑consistency (N answers → best by grade) — SAQ
- [x] Stateful learner (scratchpad memory) — `StatefulLLMStudent`
- [x] Notes accumulation strategies (correct/rationales)
- [x] Tool use (retriever tool, hooks & logging)

## Phase 4 — Task Expansion
- [x] Code repair tasks (unit test evaluator)
- [x] Proof steps (keyword-based proof hook)
- [x] Table QA (CSV input + exact-match evaluator)

## Phase 5 — Domain Module
- [x] Content stores per domain (glossary, examples)
- [x] Domain‑specific anonymization (glossary merged into vocab)
- [x] Per‑run anonymization seed logged for audit
- [x] Example scheduling (inject rotating EXAMPLE block for MCQ)

## Phase 6 — Experiment Protocols
- [x] Reproducible YAML configs for domain/task mixes (`scripts/experiment.py`)
- [x] Summary report generation (JSON + Markdown via analyzer)

Notes:
- Model fixed: `gpt-5-nano-2025-08-07` with JSON outputs.
- All tests should run with `TUTOR_MOCK_LLM=1`.
