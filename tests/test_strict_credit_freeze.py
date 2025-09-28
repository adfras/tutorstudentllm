from sim.orchestrator import Orchestrator, RunConfig, Dials
from sim.learner import OracleStudent


def test_strict_credit_with_fact_cards_learn_passes(monkeypatch):
    # Use a fake tutor to avoid network
    fake_tutor = type("_T", (), {})()
    orch = Orchestrator(model=fake_tutor)
    # Simple MCQ
    from sim.tasks import MCQTask
    type(orch)._make_mcq_task = lambda self, sid, cfg, codebook=None: MCQTask(id="t1", prompt={}, stem="Q", options=["X1 ZQ-9", "Y2"], correct_index=0)  # type: ignore

    # Configure strict citations with Fact-Cards LEARN path
    dials = Dials(
        closed_book=True,
        anonymize=True,
        use_fact_cards=True,
        require_citations=True,
        freeze_cards=False,
        q_min=1,
    )
    cfg = RunConfig(task="mcq", num_steps=1, num_options=2, dials=dials)
    # Provide fixed Fact-Cards in notes to seed LEARN
    notes_cards = {
        "cards": [
            {"id": "c1", "quote": "X1 ZQ-9", "where": {"scope": "option", "option_index": 0, "start": 0, "end": 7}, "tags": ["general-concepts"]},
            {"id": "c2", "quote": "Y2", "where": {"scope": "option", "option_index": 1, "start": 0, "end": 2}, "tags": ["general-concepts"]},
        ]
    }
    import json
    logs = orch.run(OracleStudent(), cfg, notes_text=json.dumps(notes_cards))
    assert len(logs) == 1
    rec = logs[0]
    ev = rec.get("evaluation") or {}
    # Credited should be true: citations present, option-linked quote exists, coverage >= tau, witness passes
    assert ev.get("correct") is True
    ce = ev.get("citations_evidence") or {}
    assert ce.get("credited") is True
    # Card validation should see one option quote per option (injected)
    cv = rec.get("card_validation") or {}
    counts = (cv.get("counts") or {})
    assert counts.get("option", 0) >= 2
    modes = (cv.get("validator_modes") or {})
    assert (modes.get("offset", 0) + modes.get("canon-substring", 0)) >= 2
