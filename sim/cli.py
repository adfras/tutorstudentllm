from __future__ import annotations
import argparse
import json
from typing import Any

from sim.orchestrator import Orchestrator, RunConfig, Dials
from sim.learner import LLMStudent, AlgoStudent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sim", description="General-Purpose ICL Simulator")
    p.add_argument("--skill-id", default=None)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--options", type=int, default=5)
    p.add_argument("--difficulty", default="medium", choices=["easy","medium","hard"])
    p.add_argument("--task", default="mcq", choices=["mcq","saq","code","proof","table_qa"], help="task type to run")
    p.add_argument("--closed-book", action="store_true")
    p.add_argument("--no-anon", action="store_true", help="disable anonymization")
    p.add_argument("--rich", action="store_true")
    p.add_argument("--self-consistency", type=int, default=1, help="N votes for MCQ")
    p.add_argument("--accumulate-notes", action="store_true")
    p.add_argument("--rare", dest="rare_emphasis", action="store_true")
    p.add_argument("--student", default="llm", choices=["llm","algo","stateful-llm"])
    p.add_argument("--notes-file", default=None)
    p.add_argument("--log", dest="log_path", default=None, help="path to JSONL log file")
    p.add_argument("--use-tools", action="store_true")
    p.add_argument("--tools", default="retriever", help="comma-separated tool names")
    p.add_argument("--domain", default="psych")
    p.add_argument("--use-examples", action="store_true")
    args = p.parse_args(argv)

    dials = Dials(
        closed_book=args.closed_book,
        anonymize=(not args.no_anon),
        rich=args.rich,
        self_consistency_n=args.self_consistency,
        accumulate_notes=args.accumulate_notes,
        rare_emphasis=args.rare_emphasis,
        use_tools=args.use_tools,
        tools=[t.strip() for t in (args.tools or "").split(",") if t.strip()],
    )
    cfg = RunConfig(skill_id=args.skill_id, task=args.task, num_steps=args.steps, num_options=args.options, difficulty=args.difficulty, dials=dials, domain=args.domain)
    orch = Orchestrator()
    if args.student == "llm":
        learner = LLMStudent()
    elif args.student == "stateful-llm":
        from sim.learner import StatefulLLMStudent
        learner = StatefulLLMStudent()
    else:
        learner = AlgoStudent()
    notes = ""
    if args.notes_file:
        try:
            with open(args.notes_file, "r", encoding="utf-8") as f:
                notes = f.read()
        except Exception:
            notes = ""
    logs = orch.run(learner, cfg, notes_text=notes, log_path=args.log_path)
    print(json.dumps({"config": cfg.__dict__, "n": len(logs), "results": logs}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
