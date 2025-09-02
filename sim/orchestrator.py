from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tutor.skill_map import load_skill_map
from tutor.llm_openai import OpenAILLM
from sim.tasks import MCQTask, SAQTask, Task, evaluate_mcq
from sim.anonymize import build_vocab, compile_codebook, anonymize_mcq, anonymize_text
from sim.tools import build_tools


@dataclass
class Dials:
    closed_book: bool = True
    anonymize: bool = True
    rich: bool = False  # rationales, etc.
    verify: bool = False
    reflection_every: int = 0  # not implemented yet
    context_position: str = "pre"  # pre|post|none
    self_consistency_n: int = 1  # >1 to enable majority-vote for MCQ
    accumulate_notes: bool = False  # accumulate correct info into notes across steps
    rare_emphasis: bool = False  # placeholder for rare-example emphasis
    use_tools: bool = False
    tools: List[str] = field(default_factory=lambda: ["retriever"])  # default toolset


@dataclass
class RunConfig:
    skill_id: Optional[str] = None
    task: str = "mcq"  # mcq|saq
    num_steps: int = 10
    num_options: int = 5
    difficulty: str = "medium"
    dials: Dials = field(default_factory=Dials)


class Orchestrator:
    def __init__(self, model: Optional[OpenAILLM] = None):
        self.llm = model or OpenAILLM()
        self.smap = load_skill_map()
        self.vocab = build_vocab(self.smap)

    def _make_mcq_task(self, skill_id: str, cfg: RunConfig, codebook: Optional[Dict[str, str]] = None) -> MCQTask:
        skill = self.smap["skills"].get(skill_id) or next(iter(self.smap["skills"].values()))
        q = self.llm.generate_mcq(skill, difficulty=cfg.difficulty, num_options=cfg.num_options, minimal=not cfg.dials.rich)
        stem = q.get("stem") or ""
        options = q.get("options") or []
        if cfg.dials.anonymize and codebook:
            stem, options = anonymize_mcq(stem, options, codebook)
        task = MCQTask(
            id=q.get("question_id") or f"mcq-{skill_id}",
            prompt={"skill_id": skill_id, "difficulty": cfg.dials.rich and q.get("difficulty") or cfg.difficulty},
            stem=stem,
            options=options,
            correct_index=int(q.get("correct_index", 0)),
            rationales=q.get("rationales"),
            misconception_tags=q.get("misconception_tags"),
            metadata={"source": "openai", "template_id": q.get("template_id")},
        )
        return task

    def _make_saq_task(self, skill_id: str, cfg: RunConfig, codebook: Optional[Dict[str, str]] = None) -> SAQTask:
        skill = self.smap["skills"].get(skill_id) or next(iter(self.smap["skills"].values()))
        q = self.llm.generate_saq(skill, difficulty=cfg.difficulty)
        stem = q.get("stem") or ""
        if cfg.dials.anonymize and codebook:
            stem = anonymize_text(stem, codebook)
        task = SAQTask(
            id=q.get("question_id") or f"saq-{skill_id}",
            prompt={"skill_id": skill_id, "difficulty": q.get("difficulty") or cfg.difficulty},
            stem=stem,
            expected_points=list(q.get("expected_points") or []),
            model_answer=q.get("model_answer") or "",
            difficulty=q.get("difficulty") or cfg.difficulty,
            metadata={"source": "openai"},
        )
        return task

    def run(self, learner, cfg: RunConfig, notes_text: str = "", log_path: Optional[str] = None) -> List[Dict[str, Any]]:
        # Per-run anonymization keys
        codebook = compile_codebook(self.vocab, seed=12345) if cfg.dials.anonymize else None
        logs: List[Dict[str, Any]] = []
        log_f = open(log_path, "a", encoding="utf-8") if log_path else None
        import uuid
        run_id = str(uuid.uuid4())
        # Write header line with run metadata
        if log_f:
            import json, time
            header = {
                "run_header": True,
                "run_id": run_id,
                "ts": int(time.time()),
                "config": {
                    "skill_id": cfg.skill_id,
                    "task": cfg.task,
                    "num_steps": cfg.num_steps,
                    "num_options": cfg.num_options,
                    "difficulty": cfg.difficulty,
                    "dials": vars(cfg.dials),
                },
            }
            log_f.write(json.dumps(header, ensure_ascii=False) + "\n")
            log_f.flush()
        skill_id = cfg.skill_id or next(iter(self.smap["skills"].keys()))
        notes_buf = notes_text or ""
        import time
        for step in range(cfg.num_steps):
            if (cfg.task or "mcq") == "saq":
                task = self._make_saq_task(skill_id, cfg, codebook)
                is_saq = True
            else:
                task = self._make_mcq_task(skill_id, cfg, codebook)
                is_saq = False
            # Prepare context shown to the learner
            ctx_text = notes_buf if cfg.dials.closed_book else ""
            if codebook and cfg.dials.anonymize and ctx_text:
                ctx_text = anonymize_text(ctx_text, codebook)
            context = {"notes_text": notes_buf, "context_text": ctx_text} if cfg.dials.closed_book else {}
            # Optional tools
            tool_outputs = []
            if cfg.dials.use_tools:
                tools = build_tools(cfg.dials.tools or [])
                # task view for tools
                task_view = {"stem": task.stem}
                if not is_saq:
                    task_view["options"] = task.options
                for tool in tools:
                    try:
                        out = tool.run(task=task_view, context={"notes_text": notes_buf})
                    except Exception:
                        out = {"name": getattr(tool, "name", "unknown"), "error": True}
                    tool_outputs.append(out)
                # inject tool outputs into context text
                tool_text_parts = []
                for out in tool_outputs:
                    if out.get("name") == "retriever" and out.get("snippets"):
                        tool_text_parts.append("retriever:\n- " + "\n- ".join(out["snippets"]))
                if tool_text_parts:
                    tool_text = "TOOLS:\n" + "\n".join(tool_text_parts)
                    ctx_text = (ctx_text + "\n\n" + tool_text).strip()
            # Answer
            if is_saq:
                # SAQ self-consistency: generate N drafts, grade each, keep best
                n = max(1, int(cfg.dials.self_consistency_n))
                saq_drafts = []
                best = None
                for _ in range(n):
                    a = learner.answer_saq(task, context=context)
                    g = self.llm.grade_saq(task.stem, task.expected_points, task.model_answer, a.get("student_answer") or "")
                    item = {"answer": a.get("student_answer"), "grading": g}
                    saq_drafts.append(item)
                    if best is None or float(g.get("score") or 0.0) > float(best["grading"].get("score") or 0.0):
                        best = item
                ans = {"student_answer": (best or saq_drafts[0]).get("answer")}
                grading = (best or saq_drafts[0])["grading"]
            else:
                # Self-consistency (majority vote) if enabled
                votes = []
                n = max(1, int(cfg.dials.self_consistency_n))
                for _ in range(n):
                    v = learner.answer_mcq(task, context=context)
                    votes.append(v.get("chosen_index"))
                from collections import Counter
                final_choice = None
                if votes:
                    counts = Counter(votes)
                    final_choice = counts.most_common(1)[0][0]
                ans = {"chosen_index": final_choice, "votes": votes}
                if cfg.dials.verify:
                    ans2 = learner.answer_mcq(task, context=context)
                    agree = (ans.get("chosen_index") == ans2.get("chosen_index"))
                    ans = {**ans, "verify_second": ans2, "verify_agree": agree}
                evaluation = evaluate_mcq(ans.get("chosen_index"), task)
            # record with presented stem for proof of context usage
            presented_stem = task.stem
            if context.get("context_text") and cfg.dials.context_position != "none":
                if cfg.dials.context_position == "pre":
                    presented_stem = f"CONTEXT:\n{context['context_text']}\n\nQUESTION: {task.stem}"
                elif cfg.dials.context_position == "post":
                    presented_stem = f"QUESTION: {task.stem}\n\nCONTEXT:\n{context['context_text']}"
            rec = {
                "run_id": run_id,
                "step": step,
                "ts": int(time.time()),
                "task": (
                    {"type": "saq", "stem": task.stem, "expected_points": task.expected_points}
                    if is_saq else
                    {"type": "mcq", "stem": task.stem, "options": task.options, "correct_index": task.correct_index}
                ),
                "presented_stem": presented_stem,
                "answer": ans,
                **({"evaluation": evaluation} if not is_saq else {"grading": grading, "saq_drafts": saq_drafts}),
                **({} if not tool_outputs else {"tool_outputs": tool_outputs, "tools_used": [o.get("name") for o in tool_outputs]}),
            }
            logs.append(rec)
            if log_f:
                import json
                log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log_f.flush()
            # Optional notes accumulation
            if cfg.dials.accumulate_notes:
                if not is_saq:
                    try:
                        ci = task.correct_index
                        correct_text = task.options[ci] if 0 <= ci < len(task.options) else ""
                        notes_buf += f"\nCorrect: {correct_text}"
                        if getattr(task, "rationales", None) and ci < len(task.rationales):
                            notes_buf += f"\nWhy: {task.rationales[ci]}"
                    except Exception:
                        pass
                else:
                    try:
                        # Append model answer as reference
                        notes_buf += f"\nModel: {task.model_answer}"
                    except Exception:
                        pass
            # Optional learner memory update
            try:
                if hasattr(learner, "update_memory") and callable(getattr(learner, "update_memory")):
                    learner.update_memory(task, rec)
            except Exception:
                pass
        if log_f:
            log_f.close()
        return logs
