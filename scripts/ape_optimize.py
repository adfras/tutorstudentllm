import argparse, json, os, time
from sim.orchestrator import Orchestrator, RunConfig, Dials
from sim.learner import LLMStudent


def run_with_header(header: str, steps: int = 10) -> dict:
    orch = Orchestrator()
    learner = LLMStudent()
    dials = Dials(closed_book=True, self_consistency_n=3)
    cfg = RunConfig(task="mcq", num_steps=steps, dials=dials)
    logs = orch.run(learner, cfg)
    n = len(logs)
    acc = sum(1 for s in logs if (s.get("evaluation") or {}).get("correct")) / max(1, n)
    return {"accuracy": acc, "n": n}


def main():
    p = argparse.ArgumentParser(description="Optimize instruction header (APE-like) on a small dev run")
    p.add_argument("--candidates", required=False, default=None, help="JSON/JSONL file with header candidates")
    p.add_argument("--out", required=False, default="docs/ape/header.json", help="where to save best header")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()

    if args.mock:
        # mock mode removed; ensure keys are set for live runs

    headers = [
        "You are a careful reasoner. Think step by step.",
        "Be concise and accurate. Use short steps.",
        "Follow instructions exactly. Use structured thinking.",
    ]
    if args.candidates:
        with open(args.candidates, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) == 1 and (lines[0].startswith("[") or lines[0].startswith("{")):
            obj = json.loads(lines[0])
            if isinstance(obj, list):
                headers = [str(x) for x in obj]
            elif isinstance(obj, dict):
                headers = [str(obj.get("header") or obj.get("text") or headers[0])]
        else:
            headers = [ln for ln in lines]
    results = []
    for h in headers:
        r = run_with_header(h, steps=args.steps)
        results.append({"header": h, **r})
    best = max(results, key=lambda x: x["accuracy"]) if results else {"header": headers[0], "accuracy": 0.0}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    print(json.dumps({"best": best, "all": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
