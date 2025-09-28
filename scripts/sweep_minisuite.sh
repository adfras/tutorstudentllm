#!/usr/bin/env bash
set -euo pipefail

# Mini-sweep: Mixtral (DeepInfra) and optionally OpenAI GPT-4.1
# Varies: cards budget in {6,8} and self_consistency_n in {1,3}
# Defaults:
# - FAST=1 → steps=10 and models=[deepinfra:mistralai/Mixtral-8x7B-Instruct-v0.1]
# - FAST=0 → steps=30 and models add OpenAI GPT-4.1
# You can override with:
#   STEPS, TAU, TARGET_CONF, OUTDIR, BUDGETS (space-separated), SC_LIST (space-separated),
#   SWEEP_MODELS_CSV (comma/space separated list like "deepinfra:mistralai/Mixtral-8x7B-Instruct-v0.1, openai:")

FAST=${FAST:-1}
STEPS=${STEPS:-$([[ "${FAST}" = "1" ]] && echo 10 || echo 30)}
TAU=${TAU:-0.25}
TARGET_CONF=${TARGET_CONF:-0.60}
OUTDIR=${OUTDIR:-runs/exp}
BUDGETS=${BUDGETS:-"6 8"}
SC_LIST=${SC_LIST:-"1 3"}

# Determine models
if [[ -n "${SWEEP_MODELS_CSV:-}" ]]; then
  # Split on commas and whitespace
  read -r -a MODELS <<< "$(echo "$SWEEP_MODELS_CSV" | tr ',' ' ' | xargs)"
else
  if [[ "${FAST}" = "1" ]]; then
    MODELS=("deepinfra:mistralai/Mixtral-8x7B-Instruct-v0.1")
  else
    MODELS=(
      "deepinfra:mistralai/Mixtral-8x7B-Instruct-v0.1"
      "openai:"
    )
  fi
fi

mkdir -p "$OUTDIR"

# Export env and pick up .env
set -a; [ -f .env ] && . .env; set +a

run_one() {
  local provider=$1; shift
  local model=$1; shift
  local sc=$1; shift
  local budget=$1; shift
  local mslug
  mslug="$( [[ -n "$model" ]] && echo "$model" | tr '/' '-' || echo default )"
  local slug
  slug="${provider//\//-}_${mslug}_sc${sc}_b${budget}"
  local log="$OUTDIR/${slug}.jsonl"
  echo "--> ${provider}:${model:-<default>} sc=${sc} budget=${budget} steps=${STEPS} -> ${log}"
  ANON_SEED=424242 TUTOR_COVERAGE_TAU=${TAU} \
  .venv/bin/python -m sim.cli \
    --student stateful-llm \
    --provider "${provider}" \
    $( [[ -n "$model" ]] && echo --model "$model" ) \
    --steps ${STEPS} \
    --closed-book \
    --fact-cards \
    --require-citations \
    --idk \
    --target-confidence ${TARGET_CONF} \
    --self-consistency ${sc} \
    --use-tools --tools tfidf_retriever \
    --cards-budget ${budget} \
    --progress \
    --log "${log}"
}

# Sweep loops
for entry in "${MODELS[@]}"; do
  provider="${entry%%:*}"
  model="${entry#*:}"
  for sc in ${SC_LIST}; do
    for b in ${BUDGETS}; do
      run_one "${provider}" "${model}" "${sc}" "${b}"
    done
  done
done

# Aggregate after runs
.venv/bin/python -m scripts.aggregate_runs --runs-dir runs --out-dir runs/_aggregated
.venv/bin/python -m scripts.insights --agg-dir runs/_aggregated --out-dir runs/_aggregated/plots

echo "Sweep complete. Aggregates in runs/_aggregated and plots in runs/_aggregated/plots."
