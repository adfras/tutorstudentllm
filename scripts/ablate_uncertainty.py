import argparse
import json
import os
from sim.orchestrator import Orchestrator, RunConfig, Dials
from sim.learner import LLMStudent


def run_once(steps: int, gate: bool) -> dict:
    orch = Orchestrator()
    learner = LLMStudent()
    dials = Dials(
        closed_book=True,
        self_consistency_n=3,
        adaptive_sc=True,
        temp_answer=0.3,
        temp_sc=0.7,
        uncertainty_gate=gate,
        conf_threshold=0.6,
        entropy_threshold=0.85,
        max_k_escalated=10,
        escalate_reasoning=True,
    )
    cfg = RunConfig(task="mcq", num_steps=steps, dials=dials)
    logs = orch.run(learner, cfg)
    # summarize
    n = len(logs)
    acc = sum(1 for s in logs if (s.get("evaluation") or {}).get("correct")) / max(1, n)
    avg_votes = sum(len((s.get("answer") or {}).get("votes") or []) for s in logs) / max(1, n)
    return {"n": n, "accuracy": acc, "avg_votes": avg_votes}


def main():
    p = argparse.ArgumentParser(description="Ablate uncertainty gating vs baseline")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()
    if args.mock:
        # mock mode removed; ensure keys are set for live runs or use oracle/algo
    base = run_once(args.steps, gate=False)
    gate = run_once(args.steps, gate=True)
    print(json.dumps({"baseline": base, "gated": gate}, indent=2))


if __name__ == "__main__":
    main()
