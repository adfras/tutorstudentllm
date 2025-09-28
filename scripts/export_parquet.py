#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
import pandas as pd


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Export aggregated CSVs to Parquet')
    p.add_argument('--agg-dir', default='runs/_aggregated', help='directory with steps.csv.gz and runs_summary.csv.gz')
    p.add_argument('--out-steps', default='runs/_aggregated/steps.parquet', help='output Parquet path for steps')
    p.add_argument('--out-runs', default='runs/_aggregated/runs_summary.parquet', help='output Parquet path for run summaries')
    args = p.parse_args(argv)

    os.makedirs(os.path.dirname(args.out_steps) or '.', exist_ok=True)

    steps_csv = os.path.join(args.agg_dir, 'steps.csv.gz')
    runs_csv = os.path.join(args.agg_dir, 'runs_summary.csv.gz')

    # Steps
    df_steps = pd.read_csv(steps_csv)
    try:
        df_steps.to_parquet(args.out_steps, engine='pyarrow')
    except Exception:
        # fallback with no compression options
        df_steps.to_parquet(args.out_steps)

    # Runs summary (optional if present)
    if os.path.exists(runs_csv):
        df_runs = pd.read_csv(runs_csv)
        try:
            df_runs.to_parquet(args.out_runs, engine='pyarrow')
        except Exception:
            df_runs.to_parquet(args.out_runs)

    # Optional extras: by-model and model ranking
    extras = [
        ('by_model_steps.csv.gz', os.path.join(args.agg_dir, 'by_model_steps.parquet')),
        ('by_model_runs.csv.gz', os.path.join(args.agg_dir, 'by_model_runs.parquet')),
        ('model_ranking.csv.gz', os.path.join(args.agg_dir, 'model_ranking.parquet')),
    ]
    out = {'steps_parquet': args.out_steps, 'runs_parquet': args.out_runs}
    for csv_name, parquet_path in extras:
        csv_path = os.path.join(args.agg_dir, csv_name)
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            try:
                df.to_parquet(parquet_path, engine='pyarrow')
            except Exception:
                df.to_parquet(parquet_path)
            out[os.path.basename(parquet_path)] = parquet_path
        except Exception:
            continue

    print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
