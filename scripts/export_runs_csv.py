#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Scan runs/ JSONL logs and write a single CSV with run parameters and performance",
    )
    p.add_argument("--runs-dir", default="runs", help="Directory to scan recursively (default: runs)")
    p.add_argument("--out", default="runs/all_runs.csv", help="Output CSV path (default: runs/all_runs.csv)")
    args = p.parse_args(argv)

    if not os.path.isdir(args.runs_dir):
        raise SystemExit(f"runs dir not found: {args.runs_dir}")

    # Reuse the robust aggregators from scripts.aggregate_runs
    from scripts.aggregate_runs import iter_files, aggregate_steps, build_runs_summary

    paths = sorted(iter_files(args.runs_dir, exts=(".jsonl",)))
    if not paths:
        print(json.dumps({"ok": False, "reason": "no_logs_found", "runs_dir": args.runs_dir}))
        return 1

    _rows, runs_index = aggregate_steps(paths)
    summary_rows = build_runs_summary(runs_index)

    # Compute header across all keys (preserve stable, readable order where possible)
    preferred = [
        "run_id", "source_path", "run_ts",
        "provider", "model_slug", "model_family", "model_name",
        "task", "difficulty", "domain",
        "num_steps_planned", "closed_book",
        "use_fact_cards", "require_citations", "self_consistency_n", "idk_enabled",
        "fact_cards_budget",
        # Performance
        "steps_n", "acc_final", "acc_auc", "credited_final", "credited_auc", "witness_final",
        # Cost/time
        "mean_step_seconds", "tokens_total",
    ]
    all_fields: List[str] = []
    seen = set()
    for k in preferred:
        seen.add(k)
        all_fields.append(k)
    for r in summary_rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                all_fields.append(k)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(json.dumps({"ok": True, "out": os.path.relpath(args.out), "runs": len(summary_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

