from __future__ import annotations
import argparse, json, statistics


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def analyze(path: str):
    header = None
    steps = []
    for rec in load_jsonl(path):
        if rec.get("run_header"):
            header = rec
            continue
        steps.append(rec)
    out = {"n": len(steps), "mcq_acc": None, "saq_score_mean": None}
    if steps:
        mcq = [s for s in steps if s.get("task", {}).get("type") == "mcq"]
        saq = [s for s in steps if s.get("task", {}).get("type") == "saq"]
        if mcq:
            acc = sum(1 for s in mcq if s.get("evaluation", {}).get("correct")) / len(mcq)
            out["mcq_acc"] = acc
        if saq:
            scores = [float(s.get("grading", {}).get("score") or 0.0) for s in saq]
            out["saq_score_mean"] = statistics.mean(scores) if scores else 0.0
    return {"header": header, "metrics": out}


def main(argv=None):
    p = argparse.ArgumentParser(description="Analyze simulator JSONL logs")
    p.add_argument("--log", required=True)
    args = p.parse_args(argv)
    res = analyze(args.log)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

