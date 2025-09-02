from __future__ import annotations
import argparse, glob, json, os, statistics
from typing import Any, Dict, List


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_log(path: str) -> Dict[str, Any]:
    header = None
    steps: List[Dict[str, Any]] = []
    for rec in load_jsonl(path):
        if rec.get("run_header"):
            header = rec
            continue
        steps.append(rec)
    return {"path": path, "header": header, "steps": steps}


def metrics_for_run(run: Dict[str, Any]) -> Dict[str, Any]:
    steps = run.get("steps", [])
    out: Dict[str, Any] = {"n": len(steps)}
    mcq = [s for s in steps if s.get("task", {}).get("type") == "mcq"]
    saq = [s for s in steps if s.get("task", {}).get("type") == "saq"]
    if mcq:
        inst = [1 if s.get("evaluation", {}).get("correct") else 0 for s in mcq]
        cum = []
        c = 0
        for i, v in enumerate(inst, 1):
            c += v
            cum.append(c / i)
        out.update({
            "mcq": {
                "n": len(mcq),
                "acc_final": cum[-1],
                "acc_auc": statistics.mean(cum),
                "instant": inst,
                "cumulative": cum,
            }
        })
    if saq:
        scores = [float(s.get("grading", {}).get("score") or 0.0) for s in saq]
        out.update({
            "saq": {
                "n": len(saq),
                "score_mean": statistics.mean(scores) if scores else 0.0,
                "scores": scores,
            }
        })
    # Alias-swap metrics
    fam_stats: Dict[str, Dict[str, Any]] = {}
    for s in steps:
        alias = s.get("alias") or {}
        fam = alias.get("family_id")
        if not fam:
            continue
        st = fam_stats.setdefault(fam, {"A": [], "B": [], "B_credited": []})
        if alias.get("phase") == "A":
            st["A"].append(1 if (s.get("evaluation", {}).get("correct")) else 0)
        elif alias.get("phase") == "B":
            st["B"].append(1 if (s.get("evaluation", {}).get("correct")) else 0)
            ev = (s.get("alias_evidence") or {})
            st["B_credited"].append(1 if ev.get("credited") else 0)
    if fam_stats:
        alias_out = {}
        for fam, st in fam_stats.items():
            acc_a = sum(st["A"]) / max(1, len(st["A"]))
            acc_b = sum(st["B"]) / max(1, len(st["B"]))
            credited_b = sum(st["B_credited"]) / max(1, len(st["B_credited"])) if st["B_credited"] else None
            alias_out[fam] = {
                "acc_A": acc_a,
                "acc_B": acc_b,
                "delta": (acc_a - acc_b),
                "credited_B": credited_b,
            }
        out["alias"] = alias_out
    return out


def group_key(header: Dict[str, Any]) -> str:
    cfg = (header or {}).get("config", {})
    d = cfg.get("dials", {})
    keys = [
        f"closed={d.get('closed_book')}",
        f"anon={d.get('anonymize')}",
        f"ctxpos={d.get('context_position')}",
        f"verify={d.get('verify')}",
        f"sc={d.get('self_consistency_n')}",
        f"rich={d.get('rich')}",
        f"task={cfg.get('task')}",
        f"diff={cfg.get('difficulty')}",
    ]
    return ";".join(keys)


def aggregate_group(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Aggregate MCQ and SAQ metrics across runs (aligned per-step averages)
    stats: Dict[str, Any] = {}
    mcq_runs = [r for r in runs if r.get("metrics", {}).get("mcq")]
    saq_runs = [r for r in runs if r.get("metrics", {}).get("saq")]
    if mcq_runs:
        finals = [r["metrics"]["mcq"]["acc_final"] for r in mcq_runs]
        aucs = [r["metrics"]["mcq"]["acc_auc"] for r in mcq_runs]
        # step-wise average of cumulative acc
        max_len = max(len(r["metrics"]["mcq"]["cumulative"]) for r in mcq_runs)
        step_avgs = []
        for i in range(max_len):
            vals = []
            for r in mcq_runs:
                cum = r["metrics"]["mcq"]["cumulative"]
                if i < len(cum):
                    vals.append(cum[i])
            if vals:
                step_avgs.append(sum(vals)/len(vals))
        stats["mcq"] = {
            "runs": len(mcq_runs),
            "acc_final_mean": statistics.mean(finals),
            "acc_auc_mean": statistics.mean(aucs),
            "cumulative_mean": step_avgs,
        }
    if saq_runs:
        means = [r["metrics"]["saq"]["score_mean"] for r in saq_runs]
        stats["saq"] = {
            "runs": len(saq_runs),
            "score_mean": statistics.mean(means),
        }
    # Alias aggregation: average per-family delta and credited_B
    alias_coll: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        alias = r.get("metrics", {}).get("alias") or {}
        for fam, m in alias.items():
            alias_coll.setdefault(fam, []).append(m)
    if alias_coll:
        alias_out = {}
        for fam, lst in alias_coll.items():
            deltas = [x.get("delta") for x in lst if x.get("delta") is not None]
            creds = [x.get("credited_B") for x in lst if x.get("credited_B") is not None]
            alias_out[fam] = {
                "runs": len(lst),
                "delta_mean": (statistics.mean(deltas) if deltas else None),
                "credited_B_mean": (statistics.mean(creds) if creds else None),
            }
        stats["alias"] = alias_out
    return stats


def analyze_many(paths: List[str]) -> Dict[str, Any]:
    runs = []
    for p in paths:
        run = parse_log(p)
        run_metrics = metrics_for_run(run)
        run["metrics"] = run_metrics
        runs.append(run)
    # Group by dials
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        k = group_key(r.get("header") or {})
        grouped.setdefault(k, []).append(r)
    groups_out = {k: aggregate_group(v) for k, v in grouped.items()}
    overall = aggregate_group(runs)
    return {"runs": runs, "groups": groups_out, "overall": overall}


def main(argv=None):
    p = argparse.ArgumentParser(description="Analyze simulator JSONL logs")
    p.add_argument("--log", action="append", default=[], help="path to JSONL log (can repeat)")
    p.add_argument("--glob", default=None, help="glob pattern for JSONL logs")
    p.add_argument("--dir", default=None, help="directory with *.jsonl logs")
    args = p.parse_args(argv)
    paths = []
    paths += args.log
    if args.glob:
        paths += glob.glob(args.glob)
    if args.dir:
        paths += [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith(".jsonl")]
    paths = [p for p in paths if p]
    if not paths:
        print(json.dumps({"error": "no logs"}))
        return 1
    res = analyze_many(paths)
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
