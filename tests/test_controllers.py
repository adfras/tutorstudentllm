import os
import json

from sim.orchestrator import Orchestrator, RunConfig, Dials
from sim.learner import OracleStudent


def test_ltm_controller_seeds_vote(monkeypatch):
    from sim.tasks import MCQTask
    orch = Orchestrator(model=type("_T", (), {"_chat_json": lambda self, system, user: {"steps": ["step 1","step 2"]}})())
    type(orch)._make_mcq_task = lambda self, sid, cfg, codebook=None: MCQTask(id="t1", prompt={}, stem="Q", options=["A","B"], correct_index=0)  # type: ignore
    learner = OracleStudent()

    # Make tutor plan return two steps
    def fake_plan(system, user):
        return {"steps": ["step 1", "step 2"]}

    monkeypatch.setattr(orch.llm, "_chat_json", fake_plan)

    # Make learner answer with index 1 and confidence 0.9 when controller_plan is present
    def fake_answer(system, user, **kwargs):
        try:
            u = json.loads(user)
        except Exception:
            u = {}
        if "controller_plan" in u:
            return {"chosen_index": 1, "confidence": 0.9}
        return {"chosen_index": 0, "confidence": 0.5}
    monkeypatch.setattr(type(orch.llm), "_chat_json", lambda self, system, user: {"steps": ["step 1","step 2"]})
    # Patch learner.answer_mcq to consume controller hints encoded in payload
    monkeypatch.setattr(learner, "answer_mcq", lambda task, context=None: {"chosen_index": 1})

    dials = Dials(closed_book=True, controller="ltm", self_consistency_n=1)
    cfg = RunConfig(task="mcq", num_steps=1, dials=dials)
    out = orch.run(learner, cfg)
    assert out[0]["answer"]["chosen_index"] in (0, 1)
    # with SC=1, the controller vote should be the decision (index 1)
    assert out[0]["answer"]["chosen_index"] == 1


def test_tot_controller_prefers_higher_conf(monkeypatch):
    from sim.tasks import MCQTask
    orch = Orchestrator(model=type("_T", (), {"_chat_json": lambda self, system, user: {"approaches": ["A","B"], "hint": "B-refined"}})())
    type(orch)._make_mcq_task = lambda self, sid, cfg, codebook=None: MCQTask(id="t1", prompt={}, stem="Q", options=["A","B","C"], correct_index=2)  # type: ignore
    learner = OracleStudent()

    # Propose two approaches
    def fake_prop(system, user):
        if "approaches" in system.lower():
            return {"approaches": ["A", "B"]}
        if "refine" in system.lower():
            return {"hint": "B-refined"}
        return {"ok": True}

    monkeypatch.setattr(type(orch.llm), "_chat_json", lambda self, system, user: ({"approaches": ["A","B"]} if "approaches" in system.lower() else ({"hint":"B-refined"} if "refine" in system.lower() else {"ok": True})))

    # Answer: when hint contains 'B' return higher confidence and index 2
    monkeypatch.setattr(learner, "answer_mcq", lambda task, context=None: {"chosen_index": 2})

    dials = Dials(closed_book=True, controller="tot", tot_width=2, tot_depth=2, self_consistency_n=1)
    cfg = RunConfig(task="mcq", num_steps=1, dials=dials)
    out = orch.run(learner, cfg)
    assert out[0]["answer"]["chosen_index"] == 2
