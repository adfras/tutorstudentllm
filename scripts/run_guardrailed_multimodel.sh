#!/usr/bin/env bash
set -euo pipefail

# Models to sweep. Override by exporting MODELS (space-separated) before calling.
DEFAULT_MODELS=(
  "mistralai/Mixtral-8x7B-Instruct-v0.1"
  "mistralai/Mixtral-8x22B-Instruct-v0.1"
  "deepseek-ai/DeepSeek-V3.1"
  "moonshotai/Kimi-K2-Instruct-0905"
)

# Allow callers to override key knobs without editing the file.
MODELS=(${MODELS:-${DEFAULT_MODELS[@]}})
SEED_BASE=${SEED_BASE:-424240}
SEED_COUNT=${SEED_COUNT:-20}
STEPS=${STEPS:-20}
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
LOG_ROOT=${LOG_ROOT:-runs/guardrailed}
GUARDRAILS_PATH=${GUARDRAILS_PATH:-runs/_aggregated/bayes/tables/guardrails.json}
TALK_SLOPES_PATH=${TALK_SLOPES_PATH:-runs/_aggregated/bayes/tables/talk_slopes_by_domain_20250913_104225.csv}
MODEL_PROFILES_PATH=${MODEL_PROFILES_PATH:-runs/_aggregated/model_profiles.yaml}
PROFILE_CMD=${PROFILE_CMD:-$PYTHON_BIN -m scripts.model_profiles}

if [[ -z ${GUARDRAILS_PATH:-} || ! -f $GUARDRAILS_PATH ]]; then
  latest_guardrails=$(ls -t runs/_aggregated/bayes/tables/guardrails*.json 2>/dev/null | head -1)
  if [[ -n $latest_guardrails ]]; then
    GUARDRAILS_PATH=$latest_guardrails
  else
    echo "[error] guardrails file not found; run scripts.bayes.session_bayes_report first" >&2
    exit 1
  fi
fi

if [[ -z ${TALK_SLOPES_PATH:-} || ! -f $TALK_SLOPES_PATH ]]; then
  latest_talk=$(ls -t runs/_aggregated/bayes/tables/talk_slopes_by_domain_*.csv 2>/dev/null | head -1)
  if [[ -n $latest_talk ]]; then
    TALK_SLOPES_PATH=$latest_talk
  else
    echo "[warn] talk slopes file not found; talk-slope guardrails will be skipped" >&2
    TALK_SLOPES_PATH=""
  fi
fi

if [[ -z ${MODEL_PROFILE_DISABLE:-} ]]; then
  if ! $PROFILE_CMD --profiles-out "$MODEL_PROFILES_PATH" >/dev/null 2>&1; then
    echo "[warn] unable to generate model profiles; proceeding without overrides" >&2
    MODEL_PROFILE_DISABLE=1
  fi
fi

mkdir -p "$LOG_ROOT"

for model in "${MODELS[@]}"; do
  slug=$(echo "$model" | tr '/.' '_')
  for i in $(seq 1 "$SEED_COUNT"); do
    seed=$((SEED_BASE + i))
    log_path=$(printf "%s/%s_state%03d.jsonl" "$LOG_ROOT" "$slug" "$i")
    last_step=$((STEPS - 1))

    if [[ -s "$log_path" ]] && rg -q "\"step\": $last_step" "$log_path" 2>/dev/null; then
      echo "[skip] $log_path already has $STEPS steps"
      continue
    fi

    echo "[run] model=$model seed=$seed -> $log_path"
    talk_args=()
    [[ -n $TALK_SLOPES_PATH ]] && talk_args=(--talk-slopes "$TALK_SLOPES_PATH")
    extra_args=()
    if [[ -z ${MODEL_PROFILE_DISABLE:-} ]]; then
      while IFS= read -r line; do
        [[ -z $line ]] && continue
        read -r -a parts <<<"$line"
        extra_args+=("${parts[@]}")
      done < <($PROFILE_CMD --model "$model" --format shell 2>/dev/null || true)
    fi
    if ! ANON_SEED="$seed" "$PYTHON_BIN" -m sim.cli \
      --student stateful-llm --provider deepinfra \
      --model "$model" \
      --steps "$STEPS" --closed-book --fact-cards --require-citations \
      --self-consistency 3 --best-of 8 --rerank evidence \
      --sc-extract 3 --q-min 2 \
      --use-tools --tools tfidf_retriever,option_retriever \
      --cards-budget 12 --coverage-tau 0.35 \
      --guardrails "$GUARDRAILS_PATH" \
      "${talk_args[@]}" \
      "${extra_args[@]}" \
      --tokens-autonudge --trough-margin 0.10 \
      --progress --log "$log_path"; then
        echo "model=$model seed=$seed" >> "$LOG_ROOT/failures.txt"
    fi
  done
done
