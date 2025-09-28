import os
import json

from sim.learner import LLMStudent
from sim.tasks import MCQTask


def test_pot_mcq_maps_numeric_to_option(monkeypatch):
    # Force mock mode
    os.environ["TUTOR_MOCK_LLM"] = "1"
    stu = LLMStudent(provider="openai")

    # Monkeypatch model._chat_json_opts to return a small program for the PoT step
    prog = {"program": "result = 8"}

    def fake_chat_json_opts(system: str, user: str, **kwargs):
        try:
            u = json.loads(user)
        except Exception:
            u = {}
        # If we asked for a program, return it; else return a basic MCQ answer
        if "program" in system.lower() or "program" in (system or "").lower():
            return prog
        return {"chosen_index": 0, "confidence": 0.5}

    monkeypatch.setattr(stu.model, "_chat_json_opts", fake_chat_json_opts)

    task = MCQTask(
        id="mcq-pot-1",
        prompt={},
        stem="Compute 2^3",
        options=["6", "7", "8", "9"],
        correct_index=2,
        rationales=None,
        misconception_tags=None,
        metadata={},
    )
    ctx = {"reasoning": "pot", "use_pyexec": True}
    ans = stu.answer_mcq(task, context=ctx)
    assert ans["chosen_index"] == 2

