#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, gzip, json, os, statistics as st
from typing import Any, Dict, List, Tuple


def load_csv_gz(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as gz:
        r = csv.DictReader(gz)
        for row in r:
            rows.append(row)
    return rows


def fnum(x) -> float | None:
    try:
        if x in (None, ''):
            return None
        return float(x)
    except Exception:
        return None


def as_bool(x) -> bool | None:
    if x in (None, ''):
        return None
    s = str(x).strip().lower()
    if s in ('true','1','yes','y','on'):
        return True
    if s in ('false','0','no','n','off'):
        return False
    return None


def summarize_model(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    # Overall
    credits = [fnum(r.get('credited_final')) for r in rows if fnum(r.get('credited_final')) is not None]
    witness = [fnum(r.get('witness_final')) for r in rows if fnum(r.get('witness_final')) is not None]
    accs = [fnum(r.get('acc_final')) for r in rows if fnum(r.get('acc_final')) is not None]
    out: Dict[str, Any] = {
        'runs': len(rows),
        'credited_final_mean': (st.mean(credits) if credits else None),
        'witness_final_mean': (st.mean(witness) if witness else None),
        'acc_final_mean': (st.mean(accs) if accs else None),
    }
    # Effects: budget, self-consistency, idk
    def grp_mean(key: str) -> Dict[str, Any]:
        by: Dict[str, List[float]] = {}
        for r in rows:
            k = r.get(key)
            v = fnum(r.get('credited_final'))
            if k is None or v is None: continue
            by.setdefault(str(k), []).append(v)
        vals = {k: (st.mean(v) if v else None, len(v)) for k, v in by.items()}
        return vals
    budget = grp_mean('fact_cards_budget')
    sc = grp_mean('self_consistency_n')
    idk_vals: Dict[str, Any] = {}
    _idk = {}
    for r in rows:
        k = as_bool(r.get('idk_enabled'))
        v = fnum(r.get('credited_final'))
        if k is None or v is None: continue
        _idk.setdefault(str(k), []).append(v)
    if _idk:
        idk_vals = {k: (st.mean(v) if v else None, len(v)) for k, v in _idk.items()}
    # Correlations vs tokens/latency
    def corr(xs: List[float], ys: List[float]) -> float | None:
        n = min(len(xs), len(ys))
        if n < 2: return None
        x = xs[:n]; y = ys[:n]
        mx = st.mean(x); my = st.mean(y)
        num = sum((x[i]-mx)*(y[i]-my) for i in range(n))
        denx = (sum((xi-mx)**2 for xi in x))**0.5
        deny = (sum((yi-my)**2 for yi in y))**0.5
        if denx == 0 or deny == 0: return 0.0
        return num/(denx*deny)
    toks = [fnum(r.get('tokens_total')) for r in rows if fnum(r.get('tokens_total')) is not None]
    toks_c = [fnum(r.get('credited_final')) for r in rows if fnum(r.get('tokens_total')) is not None and fnum(r.get('credited_final')) is not None]
    secs = [fnum(r.get('mean_step_seconds')) for r in rows if fnum(r.get('mean_step_seconds')) is not None]
    secs_c = [fnum(r.get('credited_final')) for r in rows if fnum(r.get('mean_step_seconds')) is not None and fnum(r.get('credited_final')) is not None]
    out['effects'] = {
        'budget_mean_by_value': budget,
        'self_consistency_mean_by_value': sc,
        'idk_mean_by_value': idk_vals,
        'corr_tokens_total_vs_credited_final': corr([fnum(r.get('tokens_total')) for r in rows if fnum(r.get('tokens_total')) is not None and fnum(r.get('credited_final')) is not None], toks_c),
        'corr_mean_step_seconds_vs_credited_final': corr([fnum(r.get('mean_step_seconds')) for r in rows if fnum(r.get('mean_step_seconds')) is not None and fnum(r.get('credited_final')) is not None], secs_c),
    }
    # Best configs
    ranked = sorted([
        {
            'path': r.get('source_path'),
            'credited_final': fnum(r.get('credited_final')),
            'witness_final': fnum(r.get('witness_final')),
            'acc_final': fnum(r.get('acc_final')),
            'budget': r.get('fact_cards_budget'),
            'sc': r.get('self_consistency_n'),
            'steps': r.get('steps_n'),
            'tokens_total': r.get('tokens_total'),
            'mean_step_seconds': r.get('mean_step_seconds'),
        }
        for r in rows if fnum(r.get('credited_final')) is not None
    ], key=lambda d: d['credited_final'], reverse=True)
    out['top_runs'] = ranked[:10]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Component-style analysis of parameters contributing to performance by model')
    p.add_argument('--agg-dir', default='runs/_aggregated', help='where runs_summary.csv.gz is located')
    p.add_argument('--out-dir', default='runs/_aggregated/analysis', help='output directory for JSON/MD')
    p.add_argument('--filter-apples', action='store_true', help='restrict to mcq + closed_book + use_fact_cards + require_citations')
    args = p.parse_args(argv)

    src = os.path.join(args.agg_dir, 'runs_summary.csv.gz')
    rows = load_csv_gz(src)
    if args.filter_apples:
        rows = [r for r in rows if (r.get('task') == 'mcq' and str(r.get('closed_book')).lower() in ('true','1') and str(r.get('use_fact_cards')).lower() in ('true','1') and str(r.get('require_citations')).lower() in ('true','1'))]

    # Group by model_name (fallback to model_slug/family)
    by_model: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        k = r.get('model_name') or r.get('model_slug') or r.get('model_family') or 'unknown'
        by_model.setdefault(k, []).append(r)

    out: Dict[str, Any] = {}
    for model, lst in by_model.items():
        if not lst:
            continue
        out[model] = summarize_model(lst)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, 'component_analysis.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Write a simple markdown summary
    md_path = os.path.join(args.out_dir, 'component_analysis.md')
    lines: List[str] = []
    lines.append('# Component Analysis — Parameters by Model')
    lines.append('')
    for model in sorted(out.keys()):
        m = out[model]
        lines.append(f'## {model}')
        lines.append(f"- runs: {m.get('runs')}")
        lines.append(f"- credited_final_mean: {m.get('credited_final_mean')}")
        lines.append(f"- witness_final_mean: {m.get('witness_final_mean')}")
        lines.append(f"- acc_final_mean: {m.get('acc_final_mean')}")
        eff = m.get('effects') or {}
        bud = eff.get('budget_mean_by_value') or {}
        sc = eff.get('self_consistency_mean_by_value') or {}
        idk = eff.get('idk_mean_by_value') or {}
        lines.append('- budget means: ' + ', '.join([f"{k}:{v[0]} (n={v[1]})" for k,v in bud.items()]))
        lines.append('- self_consistency means: ' + ', '.join([f"{k}:{v[0]} (n={v[1]})" for k,v in sc.items()]))
        if idk:
            lines.append('- idk means: ' + ', '.join([f"{k}:{v[0]} (n={v[1]})" for k,v in idk.items()]))
        lines.append(f"- corr(tokens_total, credited_final): {eff.get('corr_tokens_total_vs_credited_final')}")
        lines.append(f"- corr(mean_step_seconds, credited_final): {eff.get('corr_mean_step_seconds_vs_credited_final')}")
        lines.append('')
        lines.append('Top runs:')
        for tr in m.get('top_runs') or []:
            lines.append(f"- {tr.get('path')}: credited={tr.get('credited_final')} witness={tr.get('witness_final')} acc={tr.get('acc_final')} budget={tr.get('budget')} sc={tr.get('sc')} steps={tr.get('steps')}")
        lines.append('')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(json.dumps({'json': json_path, 'md': md_path, 'models': len(out)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

