#!/usr/bin/env bash
set -euo pipefail

# -------- Settings you can tweak ----------
STEPS=${STEPS:-12}                 # per batch
BATCHES=${BATCHES:-5}              # total items = STEPS * BATCHES
TAU=${TAU:-0.40}                   # coverage threshold
SC_EXTRACT=${SC_EXTRACT:-3}        # (extract SC not separately exposed yet)
SC_ANSWER=${SC_ANSWER:-3}          # self-consistency for answers (sim cli has one SC knob)
FACT_CARDS=${FACT_CARDS:-1}        # 1 = enable Fact-Cards two-pass; 0 = off (faster)
REQUIRE_CITATIONS=${REQUIRE_CITATIONS:-1}  # 1 = enforce citations gate; 0 = off
USE_RETRIEVER=${USE_RETRIEVER:-1}  # 1 = include retriever tool
RETRIEVER_NAME=${RETRIEVER_NAME:-tfidf_retriever}
BATCH_PAR=${BATCH_PAR:-2}          # how many batches to run at once (per model)
MODEL_PAR=${MODEL_PAR:-1}          # how many models to run at once
QUIET=${QUIET:-1}                  # 1: silence batch output; 0: show per-batch progress bars
SKIP_ALIAS=${SKIP_ALIAS:-0}        # 1: skip alias-swap experiment per model (faster smoke)

# Models to test (provider:model)
# Add/remove lines as needed; ensure your keys are in .env (OPENAI_API_KEY, DEEPINFRA_API_KEY, etc.)
# Models to test may be overridden via env:
# - Set MODELS_CSV="provider:model, provider:model" OR
# - Set MODELS_FILE=path/to/list.txt (one provider:model per line)

MODELS=(
  # DeepSeek family (different sizes and styles)
  "deepinfra:deepseek-ai/DeepSeek-R1"                      # Baseline reasoning model
  #"deepinfra:deepseek-ai/DeepSeek-V3"                      # Large MoE model, high reasoning
  #"deepinfra:deepseek-ai/DeepSeek-R1-Distill-Llama-70B"    # Efficient distilled variant
  #"deepinfra:deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"     # Mid-size distilled model

  # Other strong open-weights on DeepInfra
  #"deepinfra:meta-llama/Llama-3.3-70B-Instruct-Turbo"      # Meta LLaMA instruct
  "deepinfra:mistralai/Mixtral-8x7B-Instruct-v0.1"         # Mixtral MoE, balanced

  # OpenAI benchmark
  "openai:gpt-4.1"
)

# Optional overrides: MODELS_CSV or MODELS_FILE
if [[ -n "${MODELS_CSV:-}" ]]; then
  # Split on commas and whitespace
  read -r -a MODELS <<< "$(echo "$MODELS_CSV" | tr ',' ' ' | xargs)"
fi
if [[ -n "${MODELS_FILE:-}" && -f "${MODELS_FILE}" ]]; then
  mapfile -t MODELS < <(grep -v '^[[:space:]]*#' "${MODELS_FILE}" | awk 'NF>0')
fi


# Activate venv
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

mkdir -p runs/compare

# ---------------- Progress / ETA (suite-wide) ----------------
TOTAL_BATCHES=$(( BATCHES * ${#MODELS[@]} ))
PROGRESS_FILE=""
MONITOR_PID=""

_fmt_time() {
  local T=$1
  local H=$(( T/3600 ))
  local M=$(( (T%3600)/60 ))
  local S=$(( T%60 ))
  printf "%d:%02d:%02d" "$H" "$M" "$S"
}

_start_suite_monitor() {
  PROGRESS_FILE="$(mktemp -t icl_progress.XXXXXX)"
  local refresh_secs=${PROGRESS_REFRESH_SECS:-1}
  local start_ts
  start_ts=$(date +%s)
  (
    while :; do
      local done=0
      if [[ -f "$PROGRESS_FILE" ]]; then
        done=$(wc -l < "$PROGRESS_FILE" | tr -d ' ')
      fi
      (( done > TOTAL_BATCHES )) && done=$TOTAL_BATCHES
      local now elapsed eta left
      now=$(date +%s)
      elapsed=$(( now - start_ts ))
      if (( done > 0 )); then
        left=$(( TOTAL_BATCHES - done ))
        eta=$(( (elapsed * left) / done ))
      else
        eta=0
      fi
      printf "\rBatches: %d/%d | Elapsed %s | ETA %s" \
        "$done" "$TOTAL_BATCHES" "$( _fmt_time "$elapsed" )" "$( _fmt_time "$eta" )" >&2
      if (( done >= TOTAL_BATCHES )); then
        echo >&2
        break
      fi
      sleep "$refresh_secs"
    done
  ) & MONITOR_PID=$!
}

_stop_suite_monitor() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  [[ -f "${PROGRESS_FILE}" ]] && rm -f "${PROGRESS_FILE}"
}

run_one_model () {
  local provider="$1"
  local model="$2"
  local slug="${provider//\//-}_${model//\//-}"
  local outdir="runs/compare/${slug}"
  mkdir -p "${outdir}"

  echo "=== Running model: ${provider}:${model} -> ${outdir}"

  # Helper: check if a batch log already looks complete (has last step)
  _batch_complete() {
    local path="$1"
    # Consider complete if file exists and contains a line with '"step": STEPS-1'
    [[ -s "$path" ]] && grep -q "\"step\"[[:space:]]*:[[:space:]]*$((STEPS-1))\b" "$path"
  }

  # ---------- Simple per-model parallelism (Option A) ----------
  running=0
  pids=()
  for i in $(seq 1 ${BATCHES}); do
    # Skip if already complete (resume support)
    if _batch_complete "${outdir}/batch_${i}.jsonl"; then
      [[ -n "${PROGRESS_FILE:-}" ]] && echo 1 >> "${PROGRESS_FILE}"
      continue
    fi
    if [[ "$QUIET" -eq 1 ]]; then
      {
        TUTOR_COVERAGE_TAU=${TAU} TUTOR_ANONYMIZE=1 TUTOR_REQUIRE_CITATIONS=${REQUIRE_CITATIONS} \
        .venv/bin/python -m sim.cli \
          --student llm \
          --provider "${provider}" \
          --model "${model}" \
          --steps ${STEPS} \
          --closed-book \
          --sc-extract ${SC_EXTRACT} \
          $( [[ ${FACT_CARDS} -eq 1 ]] && echo --fact-cards ) \
          $( [[ ${REQUIRE_CITATIONS} -eq 1 ]] && echo --require-citations ) \
          --self-consistency ${SC_ANSWER} \
          $( [[ ${USE_RETRIEVER} -eq 1 ]] && echo --use-tools --tools ${RETRIEVER_NAME} ) \
          --progress \
          --log "${outdir}/batch_${i}.jsonl" \
          > /dev/null 2>> "${outdir}/batch_${i}.stderr" || true
        [[ -n "${PROGRESS_FILE:-}" ]] && echo 1 >> "${PROGRESS_FILE}"
      } &
    else
      {
        TUTOR_COVERAGE_TAU=${TAU} TUTOR_ANONYMIZE=1 TUTOR_REQUIRE_CITATIONS=${REQUIRE_CITATIONS} \
        .venv/bin/python -m sim.cli \
          --student llm \
          --provider "${provider}" \
          --model "${model}" \
          --steps ${STEPS} \
          --closed-book \
          --sc-extract ${SC_EXTRACT} \
          $( [[ ${FACT_CARDS} -eq 1 ]] && echo --fact-cards ) \
          $( [[ ${REQUIRE_CITATIONS} -eq 1 ]] && echo --require-citations ) \
          --self-consistency ${SC_ANSWER} \
          $( [[ ${USE_RETRIEVER} -eq 1 ]] && echo --use-tools --tools ${RETRIEVER_NAME} ) \
          --progress \
          --log "${outdir}/batch_${i}.jsonl" || true
        [[ -n "${PROGRESS_FILE:-}" ]] && echo 1 >> "${PROGRESS_FILE}"
      } &
    fi
    pids+=($!)
    running=$((running+1))
    if (( running >= BATCH_PAR )); then
      wait -n || true
      running=$((running-1))
      sleep 0.2
    fi
  done
  # finalize remaining
  wait "${pids[@]}" || true

  .venv/bin/python -m scripts.analyze \
    --glob "${outdir}/batch_*.jsonl" \
    > "${outdir}/aggregate.json"

  # Alias-swap acid test (optional)
  if [[ "${SKIP_ALIAS}" -ne 1 ]]; then
    # Write a per-model alias config overriding provider/model and dials
    alias_cfg="${outdir}/alias_${slug}.yaml"
    PROVIDER="${provider}" MODEL="${model}" ALIAS_OUT="${alias_cfg}" TAU="${TAU}" SC_ANSWER="${SC_ANSWER}" FACT_CARDS="${FACT_CARDS}" REQUIRE_CITATIONS="${REQUIRE_CITATIONS}" USE_RETRIEVER="${USE_RETRIEVER}" RETRIEVER_NAME="${RETRIEVER_NAME}" \
    python3 - <<'PY'
import os, yaml
src = 'docs/iclsim/experiments/icl_proof_live_classical_on.yaml'
cfg = yaml.safe_load(open(src, 'r', encoding='utf-8'))
runs = cfg.get('runs') or []
provider = os.environ.get('PROVIDER')
model = os.environ.get('MODEL')
tau = float(os.environ.get('TAU','0.40'))
sc = int(os.environ.get('SC_ANSWER','3'))
use_fc = os.environ.get('FACT_CARDS','1') in ('1','true','yes','on')
req_cit = os.environ.get('REQUIRE_CITATIONS','1') in ('1','true','yes','on')
use_retr = os.environ.get('USE_RETRIEVER','1') in ('1','true','yes','on')
retr_name = os.environ.get('RETRIEVER_NAME','retriever')
for r in runs:
    r['provider'] = provider
    r['model'] = model
    r['coverage_tau'] = tau
    d = r.setdefault('dials', {})
    d['self_consistency_n'] = sc
    d['use_fact_cards'] = use_fc
    d['require_citations'] = req_cit
    d['use_tools'] = use_retr
    d['tools'] = [retr_name] if use_retr else []
with open(os.environ['ALIAS_OUT'], 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print('Wrote alias YAML to', os.environ['ALIAS_OUT'])
PY
    .venv/bin/python -m scripts.experiment \
      --config "${alias_cfg}" \
      --out "${outdir}/alias_classical_on"

    .venv/bin/python -m scripts.analyze \
      --dir "${outdir}/alias_classical_on" \
      > "${outdir}/alias_summary.json"
  fi
}

# -------- Run all models -----------
_start_suite_monitor
trap _stop_suite_monitor EXIT
running_models=0
model_pids=()
for entry in "${MODELS[@]}"; do
  provider="${entry%%:*}"
  model="${entry#*:}"
  run_one_model "${provider}" "${model}" &
  model_pids+=($!)
  running_models=$((running_models+1))
  if (( running_models >= MODEL_PAR )); then
    wait -n || true
    running_models=$((running_models-1))
  fi
done
wait "${model_pids[@]}" || true
_stop_suite_monitor

# -------- Produce a combined table ----------
SUMMARY_OUT=${SUMMARY_OUT:-runs/compare/summary.csv}
.venv/bin/python - <<'PY' > "$SUMMARY_OUT"
import json, glob, os

def get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

rows = []
for agg_path in glob.glob("runs/compare/*/aggregate.json"):
    model_dir = os.path.dirname(agg_path)
    slug = os.path.basename(model_dir)
    alias_path = os.path.join(model_dir, "alias_summary.json")

    try:
        agg = json.load(open(agg_path))
    except Exception:
        continue
    alias = {}
    if os.path.exists(alias_path):
        try:
            alias = json.load(open(alias_path))
        except Exception:
            alias = {}

    rows.append({
        "model_slug": slug,
        "credited_final_mean": get(agg, "overall", "mcq", "credited_final_mean"),
        "credited_auc_mean": get(agg, "overall", "mcq", "credited_auc_mean"),
        "witness_pass_mean": get(agg, "overall", "mcq", "witness_final_mean"),
        "acc_final_mean": get(agg, "overall", "mcq", "acc_final_mean"),
        "acc_auc_mean": get(agg, "overall", "mcq", "acc_auc_mean"),
        "median_step_seconds": get(agg, "overall", "timing", "median_step_seconds"),
        "tokens_per_step_mean": get(agg, "overall", "usage", "tokens_per_step_mean"),
        "alias_B_credited_mean": get(alias, "overall", "alias", "signal-association-01", "credited_B_mean"),
    })

cols = [
    "model_slug","credited_final_mean","credited_auc_mean",
    "alias_B_credited_mean","witness_pass_mean",
    "acc_final_mean","acc_auc_mean",
    "median_step_seconds","tokens_per_step_mean",
]
print(",".join(cols))
for r in rows:
    print(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols))
PY
echo "Wrote summary CSV -> $SUMMARY_OUT" >&2
