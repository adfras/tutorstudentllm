from sim.orchestrator import Orchestrator, RunConfig, Dials
from sim.learner import LLMStudent


def test_alias_swap_flow_mock_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("TUTOR_MOCK_LLM", "1")
    orch = Orchestrator()
    # Two steps: A then B; set coverage_tau low to accept overlap
    cfg = RunConfig(task="alias_swap", num_steps=2, dials=Dials(closed_book=True, anonymize=True, accumulate_notes=True), coverage_tau=0.0)
    logs = orch.run(LLMStudent(), cfg, notes_text="")
    assert len(logs) == 2
    a, b = logs
    assert a.get("alias", {}).get("phase") == "A"
    assert b.get("alias", {}).get("phase") == "B"
    # B record includes alias_evidence
    ev = b.get("alias_evidence") or {}
    assert "coverage" in ev and "witness_pass" in ev and "credited" in ev

