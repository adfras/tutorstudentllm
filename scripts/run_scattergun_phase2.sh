#!/usr/bin/env bash
# Scattergun Phase 2 runner for tutor→student ICL simulations.
#
# This script launches breadth (cheap, short) and depth (longer) runs across
# the selected DeepInfra models that satisfy: long-context, JSON reliability,
# and reasonable cost per credited answer. It handles seed management, log
# paths, and guardrail discovery. Set DRY_RUN=1 to preview commands.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENVPY="${ROOT_DIR}/.venv/bin/python"
SIM="${VENVPY} -m sim.cli"

if [[ ! -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  echo "error: virtualenv not found at ${ROOT_DIR}/.venv/bin/python" >&2
  exit 1
fi

latest_file() {
  local pattern=$1
  local file
  file=$(ls -1t ${pattern} 2>/dev/null | head -n 1 || true)
  if [[ -z "${file}" ]]; then
    return 1
  fi
  printf '%s' "${file}"
}

GUARDRAILS=${GUARDRAILS_PATH:-$(latest_file "${ROOT_DIR}/runs/_aggregated/bayes/tables/guardrails*.json")}
TALK_SLOPES=${TALK_SLOPES_PATH:-$(latest_file "${ROOT_DIR}/runs/_aggregated/bayes/tables/talk_slopes_by_domain_*.csv")}

MODEL_PROFILES_PATH=${MODEL_PROFILES_PATH:-"${ROOT_DIR}/runs/_aggregated/model_profiles.yaml"}
PROFILE_BIN=${PROFILE_BIN:-"${VENVPY}"}
read -r -a PROFILE_CMD <<<"${PROFILE_CMD:-${PROFILE_BIN} -m scripts.model_profiles}"

if [[ -z "${GUARDRAILS}" || -z "${TALK_SLOPES}" ]]; then
  echo "error: guardrail or talk-slope tables not found. Set GUARDRAILS_PATH/TALK_SLOPES_PATH." >&2
  exit 1
fi

if [[ -z "${MODEL_PROFILE_DISABLE:-}" ]]; then
  if ! "${PROFILE_CMD[@]}" --profiles-out "${MODEL_PROFILES_PATH}" >/dev/null 2>&1; then
    echo "[warn] unable to generate model profiles; proceeding without overrides" >&2
    MODEL_PROFILE_DISABLE=1
  fi
fi

LOG_ROOT="${ROOT_DIR}/runs/scattergun_phase2"
mkdir -p "${LOG_ROOT}"

SEED_BASE=${SEED_BASE:-4242420}
DRY_RUN=${DRY_RUN:-0}
PROGRESS_OPT=${PROGRESS_OPT:---progress}
PARALLEL_JOBS=${PARALLEL_JOBS:-1}

COMMANDS=()

run_sim() {
  local stage=$1
  local model=$2
  local steps=$3
  local seed=$4
  local best_of=$5
  local sc_extract=$6
  local cards_budget=$7
  local learn_boosts=$8
  local q_min=$9
  local idk_flag=${10}

  local log_dir="${LOG_ROOT}/${stage}"
  mkdir -p "${log_dir}"
  local slug=${model//\//_}
  slug=${slug//./_}
  local log_path="${log_dir}/${slug}_seed${seed}_N${steps}.jsonl"

  if [[ -s "${log_path}" ]]; then
    echo "[skip] ${log_path} exists"
    return
  fi

  local extra_args=()
  if [[ -z "${MODEL_PROFILE_DISABLE:-}" ]]; then
    while IFS= read -r line; do
      [[ -z "${line}" ]] && continue
      read -r -a parts <<<"${line}"
      extra_args+=("${parts[@]}")
    done < <("${PROFILE_CMD[@]}" --model "${model}" --format shell 2>/dev/null || true)
  fi

  local cmd=(
    env ANON_SEED=${seed} \
    ${VENVPY} -m sim.cli \
      --student stateful-llm --provider deepinfra \
      --model "${model}" \
      --steps "${steps}" --closed-book --fact-cards --require-citations \
      --self-consistency 3 --best-of "${best_of}" --rerank evidence \
      --sc-extract "${sc_extract}" --q-min "${q_min}" \
      --use-tools --tools tfidf_retriever,option_retriever \
      --cards-budget "${cards_budget}" --coverage-tau 0.35 \
      --max-learn-boosts "${learn_boosts}" ${idk_flag} \
      --guardrails "${GUARDRAILS}" \
      --talk-slopes "${TALK_SLOPES}" \
      --tokens-autonudge --trough-margin 0.10
  )

  if (( ${#extra_args[@]} )); then
    cmd+=( "${extra_args[@]}" )
  fi

  cmd+=(
    ${PROGRESS_OPT} \
    --log "${log_path}"
  )

  local cmd_str
  printf -v cmd_str '%q ' "${cmd[@]}"
  cmd_str=${cmd_str% }

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY: %s\n' "${cmd_str}"
    COMMANDS+=("${cmd_str}")
    return
  fi

  if (( PARALLEL_JOBS > 1 )); then
    printf '[queue] stage=%s model=%s seed=%s steps=%s\n' "${stage}" "${model}" "${seed}" "${steps}"
    COMMANDS+=("${cmd_str}")
    return
  fi

  printf '[run] stage=%s model=%s seed=%s steps=%s\n' "${stage}" "${model}" "${seed}" "${steps}"
  "${cmd[@]}"
}

# Stage configuration: name|steps|seeds|best_of|sc_extract|cards_budget|max_learn_boosts|q_min|idk_flag
STAGE_MATRIX=$(cat <<'EOF'
breadth|10|1|4|2|8|0|2|--idk
depth|20|1|6|3|10|1|2|--idk
long|40|1|8|4|12|1|2|--idk
EOF
)

readarray -t MODELS <<'EOF'
deepseek-ai/DeepSeek-V3.1
openai/gpt-oss-120b
Qwen/Qwen3-235B-A22B-Instruct-2507
meta-llama/Llama-4-Scout-17B-16E-Instruct
Qwen/Qwen3-Next-80B-A3B-Instruct
Qwen/Qwen3-30B-A3B
openai/gpt-oss-20b
deepseek-ai/DeepSeek-V3-0324
Qwen/Qwen3-32B
Qwen/Qwen3-14B
mistralai/Mistral-Small-3.2-24B-Instruct-2506
deepseek-ai/DeepSeek-R1-0528
moonshotai/Kimi-K2-Instruct-0905
EOF

seed_offset=0

while IFS='|' read -r stage steps seeds best_of sc_extract cards_budget learn_boosts q_min idk_flag; do
  [[ -z "${stage}" ]] && continue
  for model in "${MODELS[@]}"; do
    for ((i=0;i<seeds;i++)); do
      seed=$((SEED_BASE + seed_offset + i))
      run_sim "${stage}" "${model}" "${steps}" "${seed}" "${best_of}" "${sc_extract}" "${cards_budget}" "${learn_boosts}" "${q_min}" "${idk_flag}"
    done
    seed_offset=$((seed_offset + seeds))
  done
done <<<"${STAGE_MATRIX}"

if (( PARALLEL_JOBS > 1 )) && [[ "${DRY_RUN}" != "1" ]]; then
  if (( ${#COMMANDS[@]} > 0 )); then
    printf 'Launching %d commands with %d-way parallelism\n' "${#COMMANDS[@]}" "${PARALLEL_JOBS}"
    printf '%s\0' "${COMMANDS[@]}" | xargs -0 -I{} -P "${PARALLEL_JOBS}" bash -lc "{}"
  fi
fi

echo "All requested runs launched. Logs: ${LOG_ROOT}"
