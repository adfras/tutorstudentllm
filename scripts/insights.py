#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, gzip, json, math, os
from typing import Dict, Any, List, Tuple


def load_csv_gz(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as gz:
        r = csv.DictReader(gz)
        for row in r:
            rows.append(row)
    return rows


def to_float(x) -> float | None:
    try:
        if x is None: return None
        if x == '': return None
        return float(x)
    except Exception:
        return None


def pearson(xs: List[float], ys: List[float]) -> float | None:
    if not xs or not ys: return None
    n = min(len(xs), len(ys))
    xs = xs[:n]; ys = ys[:n]
    mx = sum(xs)/n; my = sum(ys)/n
    num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    denx = math.sqrt(sum((x-mx)**2 for x in xs)); deny = math.sqrt(sum((y-my)**2 for y in ys))
    if denx == 0 or deny == 0:
        return 0.0
    return num/(denx*deny)


def summarize(runs_summary: List[Dict[str, str]], steps: List[Dict[str, str]]) -> Dict[str, Any]:
    # Correlations (run-level)
    w = [to_float(r.get('witness_final')) for r in runs_summary]
    c = [to_float(r.get('credited_final')) for r in runs_summary]
    ttot = [to_float(r.get('tokens_total')) for r in runs_summary]
    msec = [to_float(r.get('mean_step_seconds')) for r in runs_summary]
    w2 = [x for x in w if x is not None]; c2 = [x for x in c if x is not None]
    t2 = [x for x in ttot if x is not None]; m2 = [x for x in msec if x is not None]
    # Pairwise compress to the same indices
    def zip_valid(a,b):
        out_a, out_b = [], []
        for i in range(min(len(a), len(b))):
            if a[i] is not None and b[i] is not None:
                out_a.append(float(a[i])); out_b.append(float(b[i]))
        return out_a, out_b
    w_c_x, w_c_y = zip_valid(w, c)
    t_c_x, t_c_y = zip_valid(ttot, c)
    s_c_x, s_c_y = zip_valid(msec, c)
    # Budget and scx grouping (run-level)
    by_budget: Dict[str, List[float]] = {}
    by_sc: Dict[str, List[float]] = {}
    for r in runs_summary:
        b = r.get('fact_cards_budget')
        sc = r.get('self_consistency_n')
        cr = to_float(r.get('credited_final'))
        if cr is None: continue
        if b:
            by_budget.setdefault(str(b), []).append(cr)
        if sc:
            by_sc.setdefault(str(sc), []).append(cr)
    def mean(lst):
        return (sum(lst)/len(lst)) if lst else None
    budget_tbl = sorted(((k, len(v), mean(v)) for k,v in by_budget.items()), key=lambda x: (x[2] is not None, x[2]), reverse=True)
    sc_tbl = sorted(((k, len(v), mean(v)) for k,v in by_sc.items()), key=lambda x: (x[2] is not None, x[2]), reverse=True)

    return {
        'corr': {
            'witness_final_vs_credited_final': pearson(w_c_x, w_c_y),
            'tokens_total_vs_credited_final': pearson(t_c_x, t_c_y),
            'mean_step_seconds_vs_credited_final': pearson(s_c_x, s_c_y),
        },
        'budget_table': [{'budget': k, 'n': n, 'credited_mean': m} for k,n,m in budget_tbl],
        'sc_table': [{'sc': k, 'n': n, 'credited_mean': m} for k,n,m in sc_tbl],
    }


def write_md(path: str, title: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    c = data.get('corr') or {}
    lines.append("## Correlations")
    for k, v in c.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Mean credited by budget")
    for row in (data.get('budget_table') or []):
        lines.append(f"- budget={row['budget']}: mean={row['credited_mean']} (n={row['n']})")
    lines.append("")
    lines.append("## Mean credited by self_consistency_n (scx)")
    for row in (data.get('sc_table') or []):
        lines.append(f"- sc={row['sc']}: mean={row['credited_mean']} (n={row['n']})")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def write_top_configs(path: str, runs_summary: List[Dict[str, str]], top_n: int = 15) -> None:
    # Create a compact CSV of top configs by credited_final
    rows = []
    for r in runs_summary:
        try:
            cf = float(r.get('credited_final')) if r.get('credited_final') not in ('', None) else None
        except Exception:
            continue
        if cf is None:
            continue
        rows.append({
            'source_path': r.get('source_path'),
            'model': r.get('model_name') or r.get('model_slug') or r.get('model_family'),
            'credited_final': cf,
            'witness_final': r.get('witness_final'),
            'acc_final': r.get('acc_final'),
            'budget': r.get('fact_cards_budget'),
            'sc': r.get('self_consistency_n'),
            'steps': r.get('steps_n'),
            'tokens_total': r.get('tokens_total'),
            'mean_step_seconds': r.get('mean_step_seconds'),
        })
    rows.sort(key=lambda x: x['credited_final'], reverse=True)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['source_path','model','credited_final'])
        w.writeheader(); w.writerows(rows[:top_n])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Generate quick insights, tables, and (optional) plots from aggregated runs')
    p.add_argument('--agg-dir', default='runs/_aggregated', help='directory with aggregated CSVs')
    p.add_argument('--out-dir', default='runs/_aggregated/plots', help='output directory for insights artifacts')
    args = p.parse_args(argv)

    runs_summary_path = os.path.join(args.agg_dir, 'runs_summary.csv.gz')
    steps_path = os.path.join(args.agg_dir, 'steps.csv.gz')
    runs_summary = load_csv_gz(runs_summary_path)
    steps = load_csv_gz(steps_path)
    data = summarize(runs_summary, steps)

    os.makedirs(args.out_dir, exist_ok=True)
    write_md(os.path.join(args.out_dir, 'INSIGHTS.md'), 'ICL Simulator — Insights', data)
    write_md(os.path.join(args.out_dir, 'DIAL_INSIGHTS.md'), 'Dial Insights', data)
    write_top_configs(os.path.join(args.out_dir, 'top15_configs.csv'), runs_summary, 15)

    # Optional plots if matplotlib is installed
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        # Witness vs. Credited scatter
        wc = [(to_float(r.get('witness_final')), to_float(r.get('credited_final'))) for r in runs_summary]
        wc = [(x,y) for x,y in wc if x is not None and y is not None]
        if wc:
            xs = [x for x,_ in wc]; ys = [y for _,y in wc]
            plt.figure(figsize=(5,4)); plt.scatter(xs, ys, alpha=0.5)
            plt.xlabel('witness_final'); plt.ylabel('credited_final'); plt.title('Witness vs Credited')
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, 'witness_vs_credited.png'), dpi=160)
            plt.close()
        # Mean credited by budget
        by_b = {}
        for r in runs_summary:
            b = r.get('fact_cards_budget'); cf = to_float(r.get('credited_final'))
            if b and cf is not None:
                by_b.setdefault(str(b), []).append(cf)
        if by_b:
            labs = sorted(by_b.keys(), key=lambda k: int(k))
            vals = [sum(by_b[k])/len(by_b[k]) for k in labs]
            plt.figure(figsize=(5,4)); plt.bar(labs, vals)
            plt.xlabel('cards_budget'); plt.ylabel('mean credited_final'); plt.title('Credited by Budget')
            plt.grid(axis='y', alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, 'mean_credited_by_budget.png'), dpi=160)
            plt.close()
        # Mean credited by scx
        by_sc = {}
        for r in runs_summary:
            sc = r.get('self_consistency_n'); cf = to_float(r.get('credited_final'))
            if sc and cf is not None:
                by_sc.setdefault(str(sc), []).append(cf)
        if by_sc:
            labs = sorted(by_sc.keys(), key=lambda k: int(k))
            vals = [sum(by_sc[k])/len(by_sc[k]) for k in labs]
            plt.figure(figsize=(5,4)); plt.bar(labs, vals)
            plt.xlabel('self_consistency_n'); plt.ylabel('mean credited_final'); plt.title('Credited by SCX')
            plt.grid(axis='y', alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, 'mean_credited_by_scx.png'), dpi=160)
            plt.close()
    except Exception:
        pass

    print(json.dumps({'out_dir': args.out_dir}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

