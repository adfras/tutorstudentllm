import os

from sim.orchestrator import Orchestrator, RunConfig, Dials
from sim.learner import OracleStudent


def test_uncertainty_gate_escalates_votes(monkeypatch):
    from sim.tasks import MCQTask
    orch = Orchestrator(model=type("_T", (), {})())
    type(orch)._make_mcq_task = lambda self, sid, cfg, codebook=None: MCQTask(id="t1", prompt={}, stem="Q", options=["A","B","C","D"], correct_index=0)  # type: ignore
    # learner returns confidence=0.5; set threshold higher to trigger escalation
    learner = OracleStudent()
    import types
    call_count = {"n": 0}
    def fake_ans(self, task, context=None):
        call_count["n"] += 1
        return {"chosen_index": 0, "raw": {"confidence": 0.5}}
    learner.answer_mcq = types.MethodType(fake_ans, learner)
    dials = Dials(
        closed_book=True,
        self_consistency_n=1,
        adaptive_sc=False,
        uncertainty_gate=True,
        conf_threshold=0.6,
        entropy_threshold=0.99,
        max_k_escalated=3,
    )
    cfg = RunConfig(task="mcq", num_steps=1, num_options=4, dials=dials)
    logs = orch.run(learner, cfg)
    assert isinstance(logs, list) and len(logs) == 1
    step = logs[0]
    # Votes should include escalated samples (>1)
    votes = (step.get("answer") or {}).get("votes") or []
    assert len(votes) >= 2
