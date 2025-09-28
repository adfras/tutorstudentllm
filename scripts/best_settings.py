#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, gzip, os
from typing import Dict, Any, List


def load_csv_gz(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as gz:
        r = csv.DictReader(gz)
        for row in r:
            rows.append(row)
    return rows


def fnum(x: str | None) -> float | None:
    try:
        if x in (None, ''): return None
        return float(x)
    except Exception:
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Emit best settings by model (max credited_final row per model)')
    p.add_argument('--agg-dir', default='runs/_aggregated', help='directory containing runs_summary.csv.gz')
    p.add_argument('--out', default='runs/_aggregated/analysis/best_settings.csv', help='output CSV path')
    p.add_argument('--apples', action='store_true', help='restrict to mcq + closed_book + use_fact_cards + require_citations')
    args = p.parse_args(argv)

    src = os.path.join(args.agg_dir, 'runs_summary.csv.gz')
    if not os.path.exists(src):
        print(f'[error] not found: {src}')
        return 2
    rows = load_csv_gz(src)
    if args.apples:
        rows = [r for r in rows if (r.get('task')=='mcq' and str(r.get('closed_book')).lower() in ('true','1') and str(r.get('use_fact_cards')).lower() in ('true','1') and str(r.get('require_citations')).lower() in ('true','1'))]

    # Group by model (prefer model_name)
    by: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        k = r.get('model_name') or r.get('model_slug') or r.get('model_family') or 'unknown'
        by.setdefault(k, []).append(r)

    # Pick best row (max credited_final); include summary counts
    out_rows: List[Dict[str, Any]] = []
    for model, lst in by.items():
        # Best run by credited_final
        best = None
        best_v = -1.0
        for r in lst:
            v = fnum(r.get('credited_final'))
            if v is None: continue
            if v > best_v:
                best_v = v
                best = r
        if best is None:
            continue
        out_rows.append({
            'model': model,
            'runs': len(lst),
            'best_path': best.get('source_path'),
            'best_credited_final': fnum(best.get('credited_final')),
            'best_witness_final': fnum(best.get('witness_final')),
            'best_acc_final': fnum(best.get('acc_final')),
            'budget': best.get('fact_cards_budget'),
            'sc': best.get('self_consistency_n'),
            'idk': best.get('idk_enabled'),
            'mean_step_seconds': best.get('mean_step_seconds'),
            'tokens_total': best.get('tokens_total'),
        })

    # Sort by best_credited_final desc
    out_rows.sort(key=lambda r: (r.get('best_credited_final') is not None, r.get('best_credited_final') or 0.0), reverse=True)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    cols = ['model','runs','budget','sc','idk','best_credited_final','best_witness_final','best_acc_final','mean_step_seconds','tokens_total','best_path']
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(out_rows)
    print(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

