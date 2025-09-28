# Evidence Coverage Modeling Pipeline

## Goals
- Track how tutor→student runs perform across models and dial settings using only
  existing logs.
- Recommend dial combinations that historically yielded the strongest credited
  performance.
- When credits become available, launch a focused 20-step verification run using
  the top recommended settings, then fold the new log back into the model — all
  without rerunning bulk sweeps.

## Pipeline Overview
```
scripts/update_credit_pipeline.py
├── scripts.aggregate_runs                 # refreshes raw -> aggregated tables
├── scripts.session_view                   # per-run session stats
├── build_dataset_and_model()              # trains ridge predictor on 277 runs
│   ├─ outputs run_training_data_clean.csv
│   └─ outputs model_credit_ridge_poly.json + coeffs
├── scripts/credit_model_diagnostics.py    # per-run predictions + residuals
└── build_recommendations()                # top-3 dial combos per provider
```

Artifacts land in `runs/_aggregated/`:

| File | Purpose |
| --- | --- |
| `run_training_data_clean.csv` | Cleaned run-level features (277 runs, ≥5 steps) |
| `model_credit_ridge_poly.json` | Ridge model (features, z-scores, coefficients, metrics) |
| `model_credit_ridge_poly_coeffs.json` | Top feature weights for quick inspection |
| `model_predictions.csv` | Per-run predicted credited rates |
| `model_predictions_by_provider.csv` | Accuracy summaries per provider |
| `model_predictions_by_stage.csv` | Accuracy summaries per stage |
| `dial_recommendations.csv` | Top 3 historical dial combos per provider |

### Diagnostics-only usage

If you just want to rescore runs or inspect residuals without retraining, call

```
.venv/bin/python scripts/credit_model_diagnostics.py \
  --data runs/_aggregated/run_training_data_clean.csv \
  --model runs/_aggregated/model_credit_ridge_poly.json \
  --out runs/_aggregated/model_predictions.csv
```

This regenerates the predictions and provider/stage breakdowns while reusing the
existing model artefact.

Run the entire pipeline with:

```bash
.venv/bin/python scripts/update_credit_pipeline.py
```

Typical runtime: ~1 minute, dominated by the aggregation step.

## Interpreting the Outputs

1. **Model metrics** (printed to stdout and stored in the artifact)
   - Hold-out MAE ≈ 0.059 credited-rate points; R² ≈ 0.84.
   - 5-fold CV MAE ≈ 0.063 ± 0.014.
   - Residual quartiles: 25% of runs within 0.016; median 0.031; 75% within 0.059.

2. **Dial recommendations** (`dial_recommendations.csv`)
   - For each provider we list the top three historical dial combinations, the
     number of runs supporting them, total steps, actual avg credit, predicted
     credit, and in-sample MAE. Focus follow-up runs on combinations with both
     high predicted credit and low error.

3. **Provider/stage summaries**
   - `model_predictions_by_provider.csv` and `_by_stage.csv` highlight where the
     predictor under-performs. High MAE pockets (e.g., `stage=strict`) flag areas
     where we should capture more data or tighten dials before trusting the model.

## Launching a 20-Step Verification Run

When credits become available:

1. Re-run the pipeline to ensure recommendations reflect the latest logs.
2. Open `dial_recommendations.csv` and pick the top row for your target provider.
3. Translate the recommended settings into the CLI command. Example template:

```bash
ANON_SEED=$RANDOM \
.venv/bin/python -m sim.cli \
  --student stateful-llm --provider deepinfra \
  --model moonshotai/Kimi-K2-Instruct-0905 \
  --steps 20 --closed-book --fact-cards --require-citations \
  --self-consistency {self_consistency_n} \
  --best-of {best_of_n} --sc-extract {sc_extract_n} --q-min {q_min} \
  --cards-budget {cards_budget} --max-learn-boosts {max_learn_boosts} \
  $( [ {idk_enabled} -eq 1 ] && echo "--idk" ) \
  $( [ {evidence_weighted_selection} -eq 1 ] && echo "--evidence-weighted-selection" ) \
  --guardrails runs/_aggregated/bayes/tables/guardrails.json \
  --talk-slopes runs/_aggregated/bayes/tables/talk_slopes_by_domain_*.csv \
  --tokens-autonudge --trough-margin 0.10 \
  --progress --log runs/tuning_phase2/<slug>_N20.jsonl
```

4. After the run finishes, re-run the pipeline (`update_credit_pipeline.py`). The
   new log will be folded into the training data, the model retrained, and the
   recommendations refreshed automatically.

## Design Decisions & Rationale

- **Ridge regression + engineered features**: balances interpretability and
  accuracy; handles correlated inputs (evidence metrics, dial settings) while
  suppressing overfitting. Batch refitting is cheap (<1 s).

- **Historical combinations only**: Recommendations come from real dial combos
  we actually executed, avoiding dangerous extrapolation into settings we’ve
  never tested.

- **One command update**: `update_credit_pipeline.py` stitches the entire flow,
  so you can refresh metrics before every decision without manual bookkeeping.

- **Deferred experimentation**: The pipeline never calls external APIs. When
  credits return, you make a single targeted run (20 steps) and immediately fold
  the evidence back into the model.

## Extending the Pipeline

- **Additional features**: Parse guardrail alerts, talk-mode statistics, or
  retrieval diagnostics from existing logs and append them to
  `run_training_data_clean.csv`; the training step will pick them up
  automatically.

- **Provider-specific models**: Filter `run_training_data_clean.csv` per provider
  and train small models for each; combine their predictions in
  `dial_recommendations.csv` to get even finer recommendations.

- **Automation hooks**: Wrap the pipeline invocation in a cron job or CI
  workflow so the artifacts stay fresh whenever new logs land in `runs/`.

All pieces are in place to make data-driven dial decisions without touching the
LLM APIs until you’re ready to verify."
