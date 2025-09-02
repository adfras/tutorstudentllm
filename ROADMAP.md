# Roadmap — General-Purpose In‑Context Learning (ICL) Simulator

This roadmap organizes work into phases to reach a robust, multi‑domain, closed‑book ICL simulator.

## Phase 1 — Task Coverage & Logging (Now)
- [x] MCQ task type and evaluator
- [x] Closed‑book anonymization of stems/options/context
- [x] Self‑verify dial (two answers + agreement)
- [x] JSONL per‑step logging (presented_stem with CONTEXT)
- [ ] SAQ task type and evaluator (grade) — In progress
- [ ] Run IDs + session metadata in logs — In progress

## Phase 2 — Evaluation & Analytics
- [ ] JSONL schema finalized (run metadata, session config, per‑step)
- [ ] Analysis scripts: learning curves, ablation of context placement/self‑verify
- [ ] Error‑driven learning dials: spacing, rare‑example emphasis

## Phase 3 — Learner Extensions
- [ ] Self‑consistency (N answers → majority vote)
- [ ] Stateful/tool‑using learners (scratchpad, tools)
- [ ] Notes accumulation strategies across steps (strict closed‑book)

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
