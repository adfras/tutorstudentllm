from __future__ import annotations
import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from sim.orchestrator import Orchestrator, RunConfig, Dials
from sim.learner import LLMStudent, AlgoStudent
from scripts.analyze import analyze_many


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML not installed")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_job(job: Dict[str, Any], out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    # Build dials
    dials = Dials(
        closed_book=bool(job.get("dials", {}).get("closed_book", True)),
        anonymize=bool(job.get("dials", {}).get("anonymize", True)),
        rich=bool(job.get("dials", {}).get("rich", False)),
        verify=bool(job.get("dials", {}).get("verify", False)),
        self_consistency_n=int(job.get("dials", {}).get("self_consistency_n", 1)),
        accumulate_notes=bool(job.get("dials", {}).get("accumulate_notes", False)),
        context_position=str(job.get("dials", {}).get("context_position", "pre")),
        use_tools=bool(job.get("dials", {}).get("use_tools", False)),
        tools=list(job.get("dials", {}).get("tools", ["retriever"])),
    )
    cfg = RunConfig(
        skill_id=job.get("skill_id"),
        task=str(job.get("task", "mcq")),
        num_steps=int(job.get("steps", 10)),
        num_options=int(job.get("options", 5)),
        difficulty=str(job.get("difficulty", "medium")),
        domain=str(job.get("domain", "psych")),
        dials=dials,
    )
    # alias-swap extras
    if cfg.task == "alias_swap":
        cfg.alias_family_id = job.get("alias_family_id")
        if job.get("coverage_tau") is not None:
            cfg.coverage_tau = float(job.get("coverage_tau"))
    orch = Orchestrator()
    student = str(job.get("student", "llm"))
    if student == "llm":
        learner = LLMStudent()
    elif student == "stateful-llm":
        from sim.learner import StatefulLLMStudent
        learner = StatefulLLMStudent()
    else:
        learner = AlgoStudent()
    notes = ""
    notes_file = job.get("notes_file")
    if notes_file and os.path.exists(notes_file):
        notes = open(notes_file, "r", encoding="utf-8").read()
    repeats = int(job.get("repeats", 1))
    logs: List[str] = []
    for i in range(repeats):
        log_path = os.path.join(out_dir, f"{job.get('name','job')}_{i+1}.jsonl")
        orch.run(learner, cfg, notes_text=notes, log_path=log_path)
        logs.append(log_path)
    return logs


def run_experiment(cfg_path: str, out_dir: str) -> Dict[str, Any]:
    cfg = load_yaml(cfg_path)
    jobs = cfg.get("runs") or []
    all_logs: List[str] = []
    for job in jobs:
        job_out = os.path.join(out_dir, job.get("name", "job"))
        all_logs.extend(run_job(job, job_out))
    summary = analyze_many(all_logs)
    # write summary json and simple markdown
    os.makedirs(out_dir, exist_ok=True)
    summ_path = os.path.join(out_dir, "summary.json")
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False)
    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown_summary(summary))
    # Write a simple HTML too
    html_path = os.path.join(out_dir, "summary.html")
    html = "<html><body><pre>" + render_markdown_summary(summary).replace("&", "&amp;").replace("<", "&lt;") + "</pre></body></html>"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return {"logs": all_logs, "summary_json": summ_path, "summary_md": md_path, "summary_html": html_path}


def render_markdown_summary(summary: Dict[str, Any]) -> str:
    lines = ["# Experiment Summary"]
    overall = summary.get("overall", {})
    if "mcq" in overall:
        mcq = overall["mcq"]
        lines.append(f"- MCQ acc_final_mean: {mcq.get('acc_final_mean'):.3f}")
        lines.append(f"- MCQ acc_auc_mean: {mcq.get('acc_auc_mean'):.3f}")
    if "saq" in overall:
        saq = overall["saq"]
        lines.append(f"- SAQ score_mean: {saq.get('score_mean'):.3f}")
    lines.append("")
    lines.append("## Groups by Dials")
    for k, v in (summary.get("groups", {}) or {}).items():
        lines.append(f"### {k}")
        if "mcq" in v:
            m = v["mcq"]
            lines.append(f"- Runs: {m.get('runs')} | acc_final_mean: {m.get('acc_final_mean'):.3f} | acc_auc_mean: {m.get('acc_auc_mean'):.3f}")
        if "saq" in v:
            s = v["saq"]
            lines.append(f"- Runs: {s.get('runs')} | score_mean: {s.get('score_mean'):.3f}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run an ICL simulator experiment from YAML config")
    p.add_argument("--config", required=True, help="YAML file describing runs")
    p.add_argument("--out", required=True, help="output directory for logs and summary")
    args = p.parse_args(argv)
    res = run_experiment(args.config, args.out)
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
