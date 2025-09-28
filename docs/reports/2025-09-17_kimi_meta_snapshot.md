# 2025-09-17 — Kimi-K2 vs. Meta-Llama-3.1 (Seed 424242)

## Run Summary

All sessions used `ANON_SEED=424242`, strict evidence gating, and shared guardrail/talk tables produced on September 17, 2025. Kimi ran the 60-step baseline recipe; Meta-Llama runs reused existing 10- and 30-step guardrailed sessions.

| Model Run | Steps | Raw Acc. | Strict Credit | Witness | Mean Student Tokens | `no_snippet_quote` Rate |
|-----------|-------|----------|---------------|---------|---------------------|--------------------------|
| `moonshotai/Kimi-K2-Instruct-0905` (`runs/baseline/kimi-k2_seed424242_N60.jsonl`) | 60 | 0.80 | 0.35 | 0.80 | 29,136 | 0.50 |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` (`runs/guardrailed/meta-llama_Meta-Llama-3_1-8B-Instruct_state001.jsonl`) | 10 | 0.30 | 0.30 | 0.30 | 46,132 | 0.10 |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` (`runs/guardrailed/meta-llama_Meta-Llama-3_1-8B-Instruct_state002.jsonl`) | 30 | 0.60 | 0.40 | 0.60 | 44,060 | 0.13 |

> Metrics derived from `runs/_aggregated/all_steps_flat.csv.gz` (September 17, 2025). `no_snippet_quote` measures the share of MCQ turns where evidence failed solely because no retrieved snippet was cited.

## Observations

- **Kimi-K2** maintains strong raw accuracy (0.80) and witness alignment (0.80) but loses half of its credited answers because the student rarely keeps retrieved snippets. 30 of 60 turns flag `no_snippet_quote`, despite tools being enabled.
- **Meta-Llama-3.1-8B** sits near parity between raw and credited scores; snippet coverage rarely fails, but overall correctness is lower than Kimi. Mean token use remains substantially higher (~44–46k).
- The newly regenerated Bayesian guardrails (`session_bayes_report_20250917_003023`) produced 81 divergences at `target_accept=0.9`, yielding wide bands; we should rerun with a higher target accept (e.g., 0.98) once the snippet fixes land.

## Next Actions

1. Re-run Kimi with `--max-learn-boosts 1`, `--cards-budget 14`, and `--evidence-weighted-selection` to retain retrieved snippets while staying inside the new token bands.
2. Validate the updated `scripts.analyze` output to ensure the `no_snippet_quote_rate` propagates into session dashboards and CI alerts.
3. Once fresh runs complete, regenerate guardrails (expect fewer divergences with better evidence coverage) and update `runs/_aggregated/model_profiles.yaml`.

