# General-Purpose In‑Context Learning (ICL) Simulator

This repository now focuses on a general-purpose ICL simulator: a modular system to study how models learn in-context across domains under closed‑book, anonymized conditions. It coordinates a Tutor (task/context provider) and a Student (LLM or algorithmic baseline), supports pluggable task types, and logs outcomes for analysis of learning dynamics.

## Highlights

- Modular orchestration: Tutor ↔ Student, step-by-step simulation.
- Pluggable learners: LLM student (OpenAI) and algorithmic baseline.
- Task abstraction: MCQ now; extendable to SAQ, code, proofs, table QA.
- Closed‑book + anonymization: force reliance on provided context, not pretraining.
- Deterministic, mockable tests: no network needed when `TUTOR_MOCK_LLM=1`.

## Layout

- `sim/` – Core simulator
  - `orchestrator.py` – runs sessions with dials (closed‑book, anonymize, etc.)
  - `learner.py` – pluggable learners (`LLMStudent`, `AlgoStudent`)
  - `tasks.py` – task abstraction and evaluators (MCQ)
  - `anonymize.py` – codebook + numeric scrambler
- `tutor/` – OpenAI wrapper (`llm_openai.py`) and skill map loader
- `docs/` – Design/reference material (including the ICL roadmap)
- `requirements.txt` – Python dependencies (no web stack)

Removed: the previous human web UI and FastAPI server are no longer part of this project.

## Synthetic Student (DeepInfra / OpenAI)

Use the included harness to simulate a “student” answering items so you can test adaptivity and analytics end‑to‑end.

Prereqs:
- Server running (see Quickstart). Tutor generation model is fixed.
- `.env` contains `DEEPINFRA_API_KEY` (for open‑source models) and `OPENAI_API_KEY`.

Common runs:

```
export $(grep -v '^#' .env | xargs)
# Open‑source (DeepInfra) student
python student_bot_deepinfra.py --username os_llama --n 100 --fast --provider deepinfra --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
python student_bot_deepinfra.py --username os_qwen  --n 100 --fast --provider deepinfra --model Qwen/Qwen2.5-7B-Instruct

# Closed‑book novice (uses only accumulated NOTES; good for learning curves)
TUTOR_ANONYMIZE=1 uvicorn server.app:app --reload &  # or set in your shell
python student_bot_deepinfra.py --username novice_qwen --n 100 --fast --provider deepinfra --model Qwen/Qwen2.5-7B-Instruct --closed-book --notes-file data/notes_novice_qwen.txt

# Algorithmic student (no LLM; NOTES‑overlap baseline)
python student_bot_deepinfra.py --username algo1 --n 100 --fast --algo --closed-book --notes-file data/notes_algo1.txt
```

Notes:
- Harness supports DeepInfra (`--provider deepinfra`) and OpenAI (`--provider openai`).
- Closed‑book mode only uses the NOTES the bot accumulates from tutor rationales.
- Algorithmic mode provides a true “no prior knowledge” baseline with simple NOTES→option overlap.

## Prerequisites

- Python 3.10+
- `.env` with `OPENAI_API_KEY` if you run non‑mock LLM.

```
OPENAI_API_KEY=sk-...
```

Model: fixed to `gpt-5-nano-2025-08-07` in `tutor/llm_openai.py`.

## Quickstart (Simulator)

1) Create a venv and install deps

```
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Run a short simulation (mocked LLM)

```
export TUTOR_MOCK_LLM=1
python -m sim.cli --steps 3 --closed-book --rich
```

3) Run with algorithmic student and optional notes file

```
python -m sim.cli --student algo --steps 5 --closed-book --notes-file data/notes_algo.txt
```

Output is a JSON blob containing the config and per‑step results.

## Extending Tasks

- Add new task types in `sim/tasks.py` with a prompt schema and evaluator.
- Update `sim/orchestrator.py` to construct the new tasks (e.g., code repair tasks) and route to learner APIs.

## Tests

Run tests in mock mode (no network):

```
.venv/bin/python -m pytest -q
```

Tests cover JSON extraction utilities, skill map loading, LLM wrapper (mock), and simulator orchestration.

## Skill Map

- Source: `docs/rt-psych-tutor/skill_map.psych101.yaml`
- Each skill has `id`, `name`, optional `parent`, and a Bloom level.

## Roadmap

See “Roadmap for a General-Purpose In-Context Learning Simulator.docx” for the architecture and experimental protocols guiding this direction.
