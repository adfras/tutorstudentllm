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
