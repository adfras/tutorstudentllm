#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, os
from typing import Any, Dict, List, Optional

# Reuse analyzer utilities for consistent parsing/metrics
from scripts.analyze import parse_log, metrics_for_run


def clamp01(x: Optional[float]) -> float:
    if x is None:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def moving_avg(vals: List[float], k: int = 5) -> List[float]:
    if not vals:
        return []
    k = max(1, int(k))
    out: List[float] = []
    s = 0.0
    from collections import deque
    q: deque = deque()
    for v in vals:
        q.append(v)
        s += v
        if len(q) > k:
            s -= q.popleft()
        out.append(s / len(q))
    return out


def stability_from_inst(inst: List[float]) -> float:
    if not inst:
        return 0.0
    n = len(inst)
    a = int(n * 0.2)
    b = int(n * 0.4)
    c = int(n * 0.9)
    plateau = inst[a:b] if b > a else inst[: max(1, n // 5)]
    late = inst[c:] if c < n else inst[-max(1, n // 10) :]
    if not plateau or not late:
        return 0.0
    return (sum(late) / len(late)) - (sum(plateau) / len(plateau))


def forget_onset(inst: List[float], delta: float = 0.10, k: int = 5) -> Optional[int]:
    if not inst:
        return None
    n = len(inst)
    ma = moving_avg(inst, k)
    a = int(n * 0.2)
    b = int(n * 0.4)
    plateau = inst[a:b] if b > a else inst[: max(1, n // 5)]
    if not plateau:
        return None
    base = sum(plateau) / len(plateau)
    # search after 40% of the run
    for i in range(max(b, k), n - k):
        if ma[i] <= base - delta and all(ma[j] <= base - delta for j in range(i, min(n, i + k))):
            return i
    return None


def compute_cls_for_run(path: str, delta: float = 0.10, include_forget: bool = True) -> Dict[str, Any]:
    parsed = parse_log(path)
    met = metrics_for_run(parsed)
    mcq = met.get("mcq") or {}
    # LAUC: prefer credited_auc (cumulative AUC); fallback to mean of cumulative; fallback to mean of instant
    lAUC = None
    if "credited_auc" in mcq and isinstance(mcq.get("credited_auc"), (int, float)):
        lAUC = float(mcq["credited_auc"])  # already normalized 0..1
    else:
        cc = mcq.get("credited_cumulative") or []
        if cc:
            lAUC = sum(cc) / len(cc)
        else:
            inst = mcq.get("credited_instant") or []
            lAUC = (sum(inst) / len(inst)) if inst else 0.0
    inst_vals = [float(x) for x in (mcq.get("credited_instant") or [])]
    STAB = stability_from_inst(inst_vals)
    WIT = mcq.get("witness_final")
    CLS = 0.6 * float(lAUC) + 0.4 * clamp01((WIT or 0.0) + STAB)
    out = {
        "run": os.path.basename(path),
        "steps": int(met.get("n") or 0),
        "LAUC": round(float(lAUC), 4),
        "WIT": (None if WIT is None else round(float(WIT), 4)),
        "STAB": round(float(STAB), 4),
        "CLS": round(float(CLS), 4),
    }
    if include_forget:
        onset = forget_onset(inst_vals, delta=delta, k=5)
        out["forget_onset_step"] = ("" if onset is None else int(onset))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compute CL&S (Context Learning & Stability) from simulator logs")
    p.add_argument("--log", action="append", default=[], help="path to JSONL (repeatable)")
    p.add_argument("--glob", default=None, help="glob for JSONL logs")
    p.add_argument("--dir", default=None, help="directory containing *.jsonl logs")
    p.add_argument("--delta", type=float, default=0.10, help="forgetting threshold Δ for onset detection")
    p.add_argument("--out", default=None, help="write CSV to this path (default: stdout)")
    args = p.parse_args(argv)

    paths: List[str] = []
    paths += args.log
    if args.glob:
        paths += glob.glob(args.glob)
    if args.dir:
        paths += [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith('.jsonl')]
    paths = [p for p in paths if p]
    if not paths:
        print("run,steps,LAUC,WIT,STAB,CLS,forget_onset_step")
        return 1
    rows = [compute_cls_for_run(p, delta=args.delta, include_forget=True) for p in paths]
    rows.sort(key=lambda r: r["CLS"], reverse=True)
    header = ["run","steps","LAUC","WIT","STAB","CLS","forget_onset_step"]
    out_lines = [",".join(header)]
    for r in rows:
        out_lines.append(
            ",".join(str(r.get(k, "")) for k in header)
        )
    csv = "\n".join(out_lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(csv)
    else:
        print(csv, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

