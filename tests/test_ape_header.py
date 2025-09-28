import json, os

from sim.orchestrator import Orchestrator, RunConfig, Dials
from sim.learner import LLMStudent


def test_ape_header_injected_into_system(monkeypatch):
    from sim.tasks import MCQTask
    orch = Orchestrator(model=type("_T", (), {})())
    type(orch)._make_mcq_task = lambda self, sid, cfg, codebook=None: MCQTask(id="t1", prompt={}, stem="Q", options=["A","B"], correct_index=0)  # type: ignore
    stu = LLMStudent('openai')

    captured = {}

    def fake(system, user, **kw):
        captured['system'] = system
        return {"chosen_index": 0, "confidence": 0.5}

    # We intercept the model call to capture the 'system' message for assertion
    monkeypatch.setattr(stu.model, "_chat_json_opts", fake)

    header = "You are a careful reasoner."
    dials = Dials(closed_book=True, instruction_header=header)
    cfg = RunConfig(task='mcq', num_steps=1, dials=dials)
    orch.run(stu, cfg)
    assert header in captured.get('system','')
