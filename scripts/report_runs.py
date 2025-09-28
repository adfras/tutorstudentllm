#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, os
from typing import Any, Dict, List, Tuple

from scripts.analyze import parse_log, metrics_for_run


def load_reasons(path: str) -> Dict[str, Any]:
    reasons: Dict[str, int] = {}
    credited = 0
    witness = 0
    coverages: List[float] = []
    steps_n = 0
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                rec = json.loads(ln)
                if rec.get('run_header'):
                    continue
                steps_n += 1
                ce = (rec.get('evaluation') or {}).get('citations_evidence') or {}
                if ce.get('credited'):
                    credited += 1
                if ce.get('witness_pass'):
                    witness += 1
                if isinstance(ce.get('coverage'), (int, float)):
                    coverages.append(float(ce['coverage']))
                r = ce.get('reasons')
                if isinstance(r, list):
                    for k in r:
                        reasons[k] = reasons.get(k, 0) + 1
    except Exception:
        pass
    cov_mean = (sum(coverages) / len(coverages)) if coverages else None
    return {
        'steps': steps_n,
        'credited_count': credited,
        'witness_count': witness,
        'coverage_mean': cov_mean,
        'reasons': reasons,
    }


def summarize(paths: List[str]) -> str:
    lines: List[str] = []
    lines.append('# ICL Simulator — Recent Runs Report')
    lines.append('')
    rows: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for p in paths:
        run = parse_log(p)
        met = metrics_for_run(run)
        rs = load_reasons(p)
        rows.append((p, met, rs))
    # Sort by mtime (newest first)
    rows.sort(key=lambda r: os.path.getmtime(r[0]) if os.path.exists(r[0]) else 0, reverse=True)
    for p, met, rs in rows:
        mcq = met.get('mcq') or {}
        timing = met.get('timing') or {}
        usage = met.get('usage') or {}
        lines.append(f'## {os.path.basename(p)}')
        lines.append('- path: ' + p)
        lines.append(f"- steps: {int(met.get('n') or 0)}")
        lines.append(f"- credited_final: {mcq.get('credited_final')}")
        lines.append(f"- raw_final: {mcq.get('acc_final')}")
        lines.append(f"- attribution_gap_final: {mcq.get('attribution_gap_final')}")
        lines.append(f"- witness_final: {mcq.get('witness_final')}")
        lines.append(f"- credited_auc: {mcq.get('credited_auc')} | acc_auc: {mcq.get('acc_auc')}")
        if timing:
            lines.append(f"- mean_step_seconds: {timing.get('mean_step_seconds')} | median: {timing.get('median_step_seconds')}")
        if usage:
            lines.append(f"- tokens_per_step_mean: {usage.get('tokens_per_step_mean')} | tokens_total: {usage.get('tokens_total')}")
        # Reasons
        R = rs.get('reasons') or {}
        if R:
            top = sorted(R.items(), key=lambda kv: kv[1], reverse=True)[:6]
            top_s = ", ".join([f"{k}:{v}" for k, v in top])
            lines.append(f"- coverage_mean: {rs.get('coverage_mean')}")
            lines.append(f"- top_failure_reasons: {top_s}")
        lines.append('')
    # Simple comparison: rank by credited_final
    try:
        comp = []
        for p, met, _ in rows:
            mcq = met.get('mcq') or {}
            comp.append((float(mcq.get('credited_final') or 0.0), os.path.basename(p)))
        comp.sort(reverse=True)
        lines.append('## Ranking by credited_final')
        for v, name in comp:
            lines.append(f'- {name}: {v}')
    except Exception:
        pass
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Summarize recent simulator runs into Markdown')
    p.add_argument('--log', action='append', default=[], help='path to JSONL log (repeatable)')
    p.add_argument('--glob', default=None, help='glob for JSONL logs')
    p.add_argument('--dir', default=None, help='directory containing *.jsonl logs')
    p.add_argument('--out', default=None, help='write Markdown to this path (default: stdout)')
    args = p.parse_args(argv)

    paths: List[str] = []
    paths += args.log
    if args.glob:
        paths += glob.glob(args.glob)
    if args.dir:
        paths += [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith('.jsonl')]
    # De-dup while preserving order
    seen = set()
    uniq: List[str] = []
    for x in paths:
        if x and x not in seen and os.path.exists(x):
            seen.add(x)
            uniq.append(x)
    if not uniq:
        md = '# ICL Simulator — Recent Runs Report\n\n(no logs found)\n'
        if args.out:
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            with open(args.out, 'w', encoding='utf-8') as f:
                f.write(md)
        else:
            print(md)
        return 1
    md = summarize(uniq)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(md)
    else:
        print(md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

