#!/usr/bin/env bash
set -euo pipefail

# Run Fact-Cards v2 live batches with DeepInfra and aggregate results.
# - Debug batch at τ=0.35 (SC=3)
# - Four batches at τ=0.40 (SC=3)
# Outputs JSONL logs under runs/ and an aggregate JSON at the end.

# Export .env if present (keys and model)
if [[ -f .env ]]; then
  set -a
  . ./.env
  set +a
fi

MODEL="${DEEPINFRA_MODEL:-deepseek-ai/DeepSeek-R1}"

echo "[run_batches] Using model: ${MODEL}" >&2
mkdir -p runs

# Ensure venv and deps
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip -q install -r requirements.txt

# Debug batch (lower tau 0.35)
echo "[run_batches] Debug batch (τ=0.35, SC=3, steps=12)" >&2
TUTOR_COVERAGE_TAU=0.35 TUTOR_ANONYMIZE=1 TUTOR_REQUIRE_CITATIONS=1 \
.venv/bin/python -m sim.cli \
  --student llm --provider deepinfra --model "${MODEL}" \
  --steps 12 --closed-book \
  --fact-cards --require-citations \
  --self-consistency 3 \
  --use-tools --tools tfidf_retriever \
  --progress \
  --log runs/fc_sc3_v2_batch1_tau35.jsonl

# Four more batches (restore tau 0.40)
for i in 2 3 4 5; do
  echo "[run_batches] Batch $i (τ=0.40, SC=3, steps=12)" >&2
  TUTOR_COVERAGE_TAU=0.40 TUTOR_ANONYMIZE=1 TUTOR_REQUIRE_CITATIONS=1 \
  .venv/bin/python -m sim.cli \
    --student llm --provider deepinfra --model "${MODEL}" \
    --steps 12 --closed-book \
    --fact-cards --require-citations \
    --self-consistency 3 \
    --use-tools --tools tfidf_retriever \
    --progress \
    --log "runs/fc_sc3_v2_batch${i}.jsonl"
done

# Aggregate
echo "[run_batches] Aggregating..." >&2
.venv/bin/python -m scripts.analyze --glob 'runs/fc_sc3_v2_batch*.jsonl' > runs/fc_sc3_v2_aggregate.json
echo "[run_batches] DONE: runs/fc_sc3_v2_aggregate.json" >&2

