#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$DIR"/..

set -a; [ -f .env ] && . .env; set +a

.venv/bin/python -m sim.cli \
  --student stateful-llm --provider deepinfra --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --steps 12 --closed-book --fact-cards --require-citations \
  --self-consistency 5 --sc-extract 5 --max-learn-boosts 3 \
  --q-min 1 --coverage-tau 0.30 \
  --log runs/strict/mixtral_preset.jsonl --progress

echo "Wrote runs/strict/mixtral_preset.jsonl"

