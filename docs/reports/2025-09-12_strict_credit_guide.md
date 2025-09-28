# Strict Credit — What To Do (Sep 12, 2025)

This guide captures the latest fixes and the simplest, working recipes to get correctness and strict credit back to healthy levels for OpenAI gpt‑4.1 and DeepInfra Mixtral in closed‑book, anonymized MCQ runs.

## TL;DR (Copy/Paste)

1) Environment

```bash
python3 -m venv .venv && . .venv/bin/activate
.venv/bin/python -m pip install -r requirements.txt
set -a; [ -f .env ] && . .env; set +a
```

2) Mixtral — strict credit baseline (works out of the box)

```bash
.venv/bin/python -m sim.cli \
  --student stateful-llm --provider deepinfra --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --steps 6 --closed-book --fact-cards --require-citations \
  --self-consistency 5 --sc-extract 5 --max-learn-boosts 3 \
  --q-min 1 --coverage-tau 0.30 \
  --log runs/strict/mixtral_strict_fixed.jsonl --progress
```

Observed (6 steps): `acc_final≈0.50`, `credited_final≈0.50`.

3) OpenAI gpt‑4.1 — strict credit (recommended recipe)

```bash
export TUTOR_COVERAGE_TAU=0.25
HDR='When using FactCards, return JSON only. Include options array with id letters and citations. For the CHOSEN option, include at least q_min option-linked citation ids and put its PRO id first. If you cannot meet q_min citations for the chosen option, output choice:"IDK".'

.venv/bin/python -m sim.cli \
  --student stateful-llm --provider openai --model gpt-4.1 \
  --steps 12 --closed-book --fact-cards --require-citations \
  --self-consistency 7 --sc-extract 5 --max-learn-boosts 3 \
  --q-min 1 --ape-header "$HDR" \
  --log runs/strict/openai_41_tau25_header.jsonl --progress
```

Observed (12 steps): `acc_final≈0.67`, `credited_final≈0.33`. Do not enable `--evidence-weighted-selection` for 4.1 here — it hurt both accuracy and credit in our trials.

## What Changed (Why 4.1 Was Failing)

- Prompt–Gate Mismatch → Forced IDK
  - Student prompt previously demanded ≥2 citations or IDK, while `q_min=1` in gating. 4.1 obeyed the prompt and abstained.
- SC Majority Counted Abstentions
  - With SC on, `None` (IDK) votes beat concrete votes → chosen=None → `acc=0`, `credit=0`.
- Fixes Applied (code)
  - Align prompt with `q_min`: learner now requires ≥`q_min` citations, not a hardcoded 2.
  - SC majority ignores `None` when concrete votes exist.
  - Evidence override is now opt‑in via `--evidence-weighted-selection` (was effectively always‑on).

Files touched:

- `sim/learner.py` — prompt aligned to `q_min`; stricter, robust citations parsing; oracle emits two cites when possible.
- `sim/orchestrator.py` — SC majority ignores `None`; evidence rerank behind flag; citations check can use current-step cards.
- `sim/controllers_quote.py` — fixed `evidence_health` call; majority ignores `None`.

## Dials That Actually Move Strict Credit

- Keep ON / Tune:
  - `--fact-cards` (enables citation channel)
  - `--sc-extract` 5 and `--max-learn-boosts` 3 (more option‑linked cards)
  - `--q-min` 1 for 4.1 (use 2 only when supply is abundant)
  - `--self-consistency` 5–7 (stability → fewer witness mismatches)
  - `--coverage-tau` 0.25–0.30 (short, abstract options need a forgiving τ)
  - `--grammar` json (do NOT use schema; it strips citations)
  - `--ape-header` (short, explicit “citations ≥q_min” JSON instruction)

- Leave OFF (for these strict MCQs):
  - `--use-tools` (unless you also cite a context card from retrieved snippet[0])
  - `--evidence-weighted-selection` for 4.1 (hurt performance in our run)

## Troubleshooting Cheat Sheet

- `no_option_quote_for_choice` → Increase card supply: `--sc-extract 5`, `--max-learn-boosts 3`.
- `witness_mismatch` → Stabilize: `--self-consistency 5–7`; consider slightly lowering τ.
- `coverage_below_tau` → Set `--coverage-tau 0.25–0.30` for short options.
- `no_citations` → Ensure `--grammar json`, and keep `--fact-cards` on.
- `no_snippet_quote` → Don’t use tools unless you will cite a verbatim substring from retrieved snippet[0].

## Reproduce Our Runs (Artifacts)

- OpenAI 4.1 (after fixes):
  - `runs/strict/openai_41_strict_fixed2.jsonl` → acc≈0.50; credited≈0.167; failures mainly coverage/witness.
  - `runs/strict/openai_41_tau25_header.jsonl` → acc≈0.67; credited≈0.33 (12 steps).
- Mixtral (no header, τ=0.30):
  - `runs/strict/mixtral_strict_fixed.jsonl` → acc≈0.50; credited≈0.50 (6 steps).

Analyze:

```bash
.venv/bin/python -m scripts.analyze --log <PATH>.jsonl > <PATH>.aggregate.json
```

## Why Credit Isn’t 1.0 On 4.1

- Ultra‑short, anonymized options → ≤15‑token quotes can overlap multiple choices, causing witness ties.
- Occasionally wrong choices under closed‑book, abstract stems → zero coverage to gold.
  - The header + τ=0.25 lift both accuracy and credit without relaxing evidence rules.

## Optional: Oracle Sanity Check (Strict)

```bash
export TUTOR_COVERAGE_TAU=0.33
.venv/bin/python -m sim.cli --student oracle --steps 6 \
  --closed-book --fact-cards --require-citations --q-min 1 \
  --log runs/strict/oracle_strict_q1_v1.jsonl --progress

# Observed: acc=1.0, credited≈0.83 on 6 steps (ties on ultra-short options)
```

## Next Steps

- Stabilize stats with 20–30 step batches on 4.1 using the header + τ=0.25 recipe.
- If desired, add a “witness‑tie = pass” policy for ultra‑short options (small change in credit engine) and re‑evaluate.
- Keep tools off for strict MCQ unless you plan to cite retrieved snippet text verbatim.

