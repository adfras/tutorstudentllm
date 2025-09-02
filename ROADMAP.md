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
- [ ] Tool use (extensible hooks for future tasks)

## Phase 4 — Task Expansion
- [ ] Code repair tasks (unit test evaluator)
- [ ] Proof steps (hook to proof checker)
- [ ] Table QA (table input + evaluator)

## Phase 5 — Domain Module
- [ ] Content stores per domain + de‑anonymization audit path
- [ ] Domain‑specific anonymization + example scheduling

## Phase 6 — Experiment Protocols
- [ ] Reproducible configs for domain/task mixes
- [ ] Summary report generation per run (curves + metrics)

Notes:
- Model fixed: `gpt-5-nano-2025-08-07` with JSON outputs.
- All tests should run with `TUTOR_MOCK_LLM=1`.
