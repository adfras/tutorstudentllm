#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, gzip, os, json, time
from typing import Dict, Any, List


def load_csv_gz(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as gz:
        r = csv.DictReader(gz)
        for row in r:
            rows.append(row)
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Build a simple runs catalog (one row per run) for easy browsing')
    p.add_argument('--agg-dir', default='runs/_aggregated', help='where runs_summary.csv.gz lives')
    p.add_argument('--out', default='runs/_catalog/catalog.csv', help='output CSV path')
    args = p.parse_args(argv)

    src = os.path.join(args.agg_dir, 'runs_summary.csv.gz')
    if not os.path.exists(src):
        print(f"[error] not found: {src}")
        return 2
    rows = load_csv_gz(src)
    # Select compact columns and sort by run_ts (desc) then mtime
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        path = r.get('source_path') or ''
        try:
            mtime = int(os.path.getmtime(path)) if (path and os.path.exists(path) and not path.startswith('archive://')) else None
        except Exception:
            mtime = None
        out_rows.append({
            'run_ts': r.get('run_ts'),
            'mtime': mtime,
            'provider': r.get('provider'),
            'model': r.get('model_name') or r.get('model_slug') or r.get('model_family'),
            'task': r.get('task'),
            'budget': r.get('fact_cards_budget'),
            'sc': r.get('self_consistency_n'),
            'idk': r.get('idk_enabled'),
            'closed_book': r.get('closed_book'),
            'use_fact_cards': r.get('use_fact_cards'),
            'require_citations': r.get('require_citations'),
            'steps_n': r.get('steps_n'),
            'credited_final': r.get('credited_final'),
            'witness_final': r.get('witness_final'),
            'acc_final': r.get('acc_final'),
            'mean_step_seconds': r.get('mean_step_seconds'),
            'tokens_total': r.get('tokens_total'),
            'path': path,
        })
    # Sort by ts desc, fallback mtime desc
    def key(r):
        ts = r.get('run_ts')
        try:
            ts = int(ts) if ts not in (None, '') else 0
        except Exception:
            ts = 0
        mt = r.get('mtime') or 0
        return (ts, mt)
    out_rows.sort(key=key, reverse=True)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    cols = [
        'run_ts','mtime','provider','model','task','budget','sc','idk','closed_book','use_fact_cards','require_citations',
        'steps_n','credited_final','witness_final','acc_final','mean_step_seconds','tokens_total','path'
    ]
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(out_rows)
    print(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

