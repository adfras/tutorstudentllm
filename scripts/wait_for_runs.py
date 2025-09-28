#!/usr/bin/env python3
from __future__ import annotations
import argparse, gzip, csv, os, sys, time, subprocess, json
from typing import List, Dict


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Cannot signal but likely alive; fall back to ps
        try:
            out = subprocess.check_output(["bash","-lc", f"ps -p {pid} -o pid= || true"], text=True)
            return str(pid) in out
        except Exception:
            return False


def wait_for_pids(pids: List[int], timeout_min: int = 90, poll_sec: int = 5) -> None:
    t0 = time.time()
    alive = set(pids)
    while alive:
        now = time.time()
        if (now - t0) > timeout_min * 60:
            break
        done = []
        for pid in list(alive):
            if not pid_alive(pid):
                done.append(pid)
                alive.remove(pid)
        if not alive:
            break
        time.sleep(poll_sec)


def load_runs_summary(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as gz:
        r = csv.DictReader(gz)
        for row in r:
            rows.append(row)
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Wait for run PIDs, then aggregate and print a brief summary')
    p.add_argument('--pids', nargs='+', type=int, required=True, help='process IDs to wait for')
    p.add_argument('--logs', nargs='*', default=[], help='expected log files to summarize (optional)')
    p.add_argument('--timeout-min', type=int, default=90)
    p.add_argument('--poll-sec', type=int, default=5)
    p.add_argument('--runs-dir', default='runs')
    p.add_argument('--out-dir', default='runs/_aggregated')
    args = p.parse_args(argv)

    wait_for_pids(args.pids, timeout_min=args.timeout_min, poll_sec=args.poll_sec)

    # Aggregate
    try:
        subprocess.check_call([sys.executable, '-m', 'scripts.aggregate_runs', '--runs-dir', args.runs_dir, '--out-dir', args.out_dir])
    except subprocess.CalledProcessError as e:
        print(json.dumps({'error': f'aggregate_runs failed: {e}'}))
        return 1
    # Insights (optional)
    try:
        subprocess.check_call([sys.executable, '-m', 'scripts.insights', '--agg-dir', args.out_dir, '--out-dir', os.path.join(args.out_dir, 'plots')])
    except subprocess.CalledProcessError:
        pass

    # Summarize the provided logs (if any) from runs_summary
    summ_path = os.path.join(args.out_dir, 'runs_summary.csv.gz')
    rows = load_runs_summary(summ_path) if os.path.exists(summ_path) else []
    picked: List[Dict[str, str]] = []
    if args.logs:
        logset = set(os.path.relpath(p) for p in args.logs)
        for r in rows:
            sp = r.get('source_path')
            if sp in logset:
                picked.append(r)
    else:
        # fallback: show latest 5 runs
        rows_sorted = sorted(rows, key=lambda r: int(r.get('run_ts') or 0), reverse=True)
        picked = rows_sorted[:5]

    brief = []
    for r in picked:
        brief.append({
            'path': r.get('source_path'),
            'model': r.get('model_name') or r.get('model_slug') or r.get('model_family'),
            'budget': r.get('fact_cards_budget'),
            'sc': r.get('self_consistency_n'),
            'credited_final': r.get('credited_final'),
            'witness_final': r.get('witness_final'),
            'acc_final': r.get('acc_final'),
            'steps': r.get('steps_n'),
        })

    print(json.dumps({'aggregated': True, 'summary': brief}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

