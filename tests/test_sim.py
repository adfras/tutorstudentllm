from sim.orchestrator import Orchestrator, RunConfig, Dials
from sim.learner import LLMStudent, AlgoStudent


def test_orchestrator_with_mock_llm():
    orch = Orchestrator()
    cfg = RunConfig(num_steps=3, num_options=5, dials=Dials(closed_book=True, anonymize=True, rich=False))
    logs = orch.run(LLMStudent(), cfg)
    assert len(logs) == 3
    for rec in logs:
        assert rec["task"]["type"] == "mcq"
        assert isinstance(rec["evaluation"]["correct"], bool)


def test_algo_student_with_notes(sample_notes):
    orch = Orchestrator()
    cfg = RunConfig(num_steps=1, num_options=4)
    logs = orch.run(AlgoStudent(), cfg, notes_text=sample_notes)
    assert len(logs) == 1
    rec = logs[0]
    assert "chosen_index" in rec["answer"]
    assert rec["task"]["type"] == "mcq"


def test_presented_stem_includes_context_when_closed_book(sample_notes, tmp_path):
    orch = Orchestrator()
    cfg = RunConfig(num_steps=1, dials=Dials(closed_book=True, anonymize=True))
    log_path = tmp_path / "run.jsonl"
    logs = orch.run(LLMStudent(), cfg, notes_text=sample_notes, log_path=str(log_path))
    rec = logs[0]
    assert "CONTEXT:" in rec["presented_stem"]
    # file should have one json line
    text = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(text) == 1 and "presented_stem" in text[0]
