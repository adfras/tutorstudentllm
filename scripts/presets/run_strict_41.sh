#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$DIR"/..

set -a; [ -f .env ] && . .env; set +a

export TUTOR_COVERAGE_TAU=${TUTOR_COVERAGE_TAU:-0.25}
HDR='When using FactCards, return JSON only. Include options array with id letters and citations. For the CHOSEN option, include at least q_min option-linked citation ids and put its PRO id first. If you cannot meet q_min citations for the chosen option, output choice:"IDK".'

.venv/bin/python -m sim.cli \
  --student stateful-llm --provider openai --model gpt-4.1 \
  --steps 12 --closed-book --fact-cards --require-citations \
  --self-consistency 7 --sc-extract 5 --max-learn-boosts 3 \
  --q-min 1 --ape-header "$HDR" \
  --log runs/strict/openai_41_preset.jsonl --progress

echo "Wrote runs/strict/openai_41_preset.jsonl"

