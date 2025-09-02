from __future__ import annotations
from typing import Any, Dict, List, Optional

from sim.tasks import MCQTask


class Learner:
    def answer_mcq(self, task: MCQTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError


class LLMStudent(Learner):
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        if provider == "openai":
            from tutor.llm_openai import OpenAILLM
            self.model = OpenAILLM()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def answer_mcq(self, task: MCQTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Include provided context in the stem (closed-book enforcement via prompt)
        stem = task.stem
        if context and context.get("context_text"):
            stem = f"CONTEXT:\n{context['context_text']}\n\nQUESTION: {task.stem}"
        ans = self.model.answer_mcq(stem, task.options)
        return {"chosen_index": ans.get("chosen_index"), "raw": ans}


class AlgoStudent(Learner):
    """Closed-book algorithmic baseline: choose option with maximum overlap with NOTES text.
    context may contain {'notes_text': str}.
    """

    def answer_mcq(self, task: MCQTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        import re
        notes = (context or {}).get("notes_text") or ""
        def tok(s: str) -> set[str]:
            return set([t for t in re.findall(r"[a-zA-Z0-9]+", (s or "").lower()) if len(t) >= 3])
        nt = tok(notes)
        scores = []
        for i, opt in enumerate(task.options):
            ot = tok(opt)
            score = len(ot & nt)
            scores.append((score, i))
        scores.sort(reverse=True)
        chosen = scores[0][1] if scores else 0
        return {"chosen_index": chosen, "scores": scores}
