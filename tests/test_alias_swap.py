from sim.orchestrator import Orchestrator, RunConfig, Dials
from sim.learner import OracleStudent


def test_alias_swap_flow_mock_llm(tmp_path, monkeypatch):
    # Stub tutor and task generation to avoid network
    fake_tutor = type("_T", (), {})()
    orch = Orchestrator(model=fake_tutor)
    from sim.tasks import MCQTask
    # Provide two fixed alias variants
    families = {"families": [{"id": "fam1", "alias_a": {"stem": "A?", "options": ["X","Y"], "correct_index": 0}, "alias_b": {"stem": "B?", "options": ["X","Y"], "correct_index": 1}}], "index": {"fam1": {"id": "fam1"}}}
    # Monkeypatch alias family loader inside orchestrator if needed (simplify by injecting a dummy loader)
    # Two steps: A then B; set coverage_tau low to accept overlap
    cfg = RunConfig(task="alias_swap", num_steps=2, dials=Dials(closed_book=True, anonymize=True, accumulate_notes=True), coverage_tau=0.0)
    logs = orch.run(OracleStudent(), cfg, notes_text="")
    assert len(logs) == 2
    a, b = logs
    assert a.get("alias", {}).get("phase") == "A"
    assert b.get("alias", {}).get("phase") == "B"
    # B record includes alias_evidence
    ev = b.get("alias_evidence") or {}
    assert "coverage" in ev and "witness_pass" in ev and "credited" in ev
