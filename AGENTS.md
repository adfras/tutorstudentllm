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
- Run simulator (mock): `export TUTOR_MOCK_LLM=1; python -m sim.cli --steps 3 --closed-book --rich`
- Algorithmic baseline: `python -m sim.cli --student algo --steps 5 --closed-book --notes-file data/notes_algo.txt`

3) Health Checks
- Simulator runs and prints JSON with `results[]` records.
- Each record contains: `task`, `answer`, `evaluation`, and `presented_stem` (with CONTEXT when closed‑book).

4) Key Verification
- Set `TUTOR_MOCK_LLM=1` for offline tests. Without mock, ensure `.env` has `OPENAI_API_KEY`.

Synthetic Student
- Use `--student llm` (default) or `--student algo` with optional `--notes-file`.

## Constraints & Policies

- Model is hard‑locked to `gpt-5-2025-08-07`; do not introduce overrides.
- Use JSON response format and prompts that explicitly include the word “JSON” to comply with model requirements.
- UI should remain minimal and responsive; prefer short stems/options and minimal payloads.
- Do not log or persist PII besides username; keep stats aggregate only.
 - Memory gating: repeats only when not mastered and due by interval; allow a single immediate remediation after a wrong answer.

Optional hardening (Provable Novice Mode):
- `TUTOR_ANONYMIZE=1` → apply per‑user codebook + numeric scrambling to stems/options/rationales.
- `TUTOR_REQUIRE_CITATIONS=1` → credit only if correct AND coverage ≥ τ AND witness re‑pick agrees. Optional `TUTOR_COVERAGE_TAU` (default 0.4).

Adaptivity is lightweight: per‑skill `mastery` (0..1) with 7‑day half‑life decay, misconception counters, and a simple policy: remediation → continue current until mastered → review due → advance when prereqs satisfied.

## Logs & Analysis
- Pass `--notes-file` to simulate closed‑book students.
- Use `sim.orchestrator.Orchestrator.run(..., log_path=...)` to write JSONL per step.

## Skill Map & Categories

- Skills are defined in `docs/rt-psych-tutor/skill_map.psych101.yaml`.
- Category is computed as the top‑level parent of a skill; used to roll up stats.

## Code Pointers
- `sim/orchestrator.py` — core simulator & dials
- `sim/learner.py` — learners (LLM, algo)
- `sim/tasks.py` — task/evaluator (MCQ)
- `sim/anonymize.py` — anonymization utilities
- `tutor/llm_openai.py` — fixed OpenAI model wrapper

## Extending Safely

- To add richer feedback: add `rationales` back to MCQ prompts and render them in the UI.
- To implement adaptivity: store mastery per skill and schedule review per `docs/rt-psych-tutor/adaptive_algorithms.md`.
- To switch persistence: migrate to Postgres using `docs/rt-psych-tutor/db_schema.sql` and replace `server/storage.py` with DB queries.

## Validation Checklist (Simulator)
- [ ] Mock LLM runs (TUTOR_MOCK_LLM=1)
- [ ] Algo student runs with notes
- [ ] CONTEXT included in `presented_stem` when closed‑book
- [ ] JSONL per‑step logging works when `log_path` set

## Troubleshooting

- 400: temperature/JSON format — ensure prompts do not set `temperature` and include “JSON” when using `response_format=json_object`.
- DNS/connectivity — verify outbound HTTPS and that `OPENAI_API_KEY` is present.
- Missing skills — check `docs/rt-psych-tutor/skill_map.psych101.yaml` and file path.
 - If credited=0 while correct>0 — ensure `TUTOR_REQUIRE_CITATIONS` off, or send `citations_text`/`citations[]` with answers and lower `TUTOR_COVERAGE_TAU` for testing.
