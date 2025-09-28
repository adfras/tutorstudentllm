# TODO — Evidence Snippet Coverage Improvements

1. **Tune Retrieval & Extraction**
   - Enable `--max-learn-boosts 1` for Kimi-K2 and other evidence-limited models so LEARN can re-attempt option quotes when coverage < τ.
   - Raise `--cards-budget` to 14 and persist more retrieved snippets; re-run the 60-step Kimi suite and confirm `no_snippet_quote` drops below 10.
   - Add `--evidence-weighted-selection` to prioritise candidates that cite retrieved snippets.

2. **Refresh Guardrails for Lean Models**
   - After collecting the new runs, re-run `scripts.aggregate_runs`, `scripts.session_view`, and `scripts.bayes.session_bayes_report` to rebuild bands that reflect ~29k token steps.
   - Verify guardrail alerts fall to <10% “below_band” on Kimi logs.

3. **Automated Checks**
   - Extend `scripts/model_profiles.py` to flag models with `no_snippet_quote` > 20% and automatically suggest `--max-learn-boosts`/`--evidence-weighted-selection`.
   - Add a regression check in `scripts.analyze` reporting the share of steps missing snippet quotes so alerts surface during sweeps.

4. **Report Back**
   - Capture before/after credited, witness, and coverage metrics for Kimi and Meta-Llama; document improvements in `runs/_aggregated/model_profiles.yaml` notes once validated.

5. **Apply Phase 2 Overrides**
   - Merge per-model override suggestions from `runs/_aggregated/model_profiles_phase2.yaml` into the main profile file so future runs auto-pick the evidence boosts (`--sc-extract 5`, `--cards-budget 14`, `--max-learn-boosts 1`, `--evidence-weighted-selection`, etc.).
   - Update launch scripts (`scripts/run_scattergun_phase2.sh`, `scripts/run_guardrailed_multimodel.sh`) to consume these overrides without new runs.
   - Refit guardrails against `runs/scattergun_phase2` aggregates (no external calls) and stash the new tables for the calibrated models.
