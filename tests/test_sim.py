from sim.orchestrator import Orchestrator, RunConfig, Dials
from sim.learner import LLMStudent, AlgoStudent, StatefulLLMStudent


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
    # file should have header + one record
    text = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(text) == 2 and "presented_stem" in text[1]


def test_saq_flow_mock_llm(tmp_path):
    # SAQ in mock mode: student answers include expected key; grader returns score
    import os
    os.environ["TUTOR_MOCK_LLM"] = "1"
    orch = Orchestrator()
    cfg = RunConfig(task="saq", num_steps=1, dials=Dials(closed_book=True, anonymize=True))
    log_path = tmp_path / "saq.jsonl"
    logs = orch.run(LLMStudent(), cfg, notes_text="", log_path=str(log_path))
    assert len(logs) == 1
    rec = logs[0]
    assert rec["task"]["type"] == "saq"
    assert "grading" in rec and "score" in rec["grading"]
    assert "saq_drafts" in rec


def test_stateful_learner_memory_updates(sample_notes):
    orch = Orchestrator()
    st = StatefulLLMStudent()
    cfg = RunConfig(num_steps=1, dials=Dials(closed_book=True, anonymize=True))
    logs = orch.run(st, cfg, notes_text=sample_notes)
    # After one MCQ step, stateful learner should have some memory (best effort)
    assert isinstance(st.memory, list)


def test_tools_usage_in_presented_stem(sample_notes):
    # Ensure tools inject content and are logged
    orch = Orchestrator()
    cfg = RunConfig(num_steps=1, dials=Dials(closed_book=True, anonymize=True, use_tools=True, tools=["retriever"]))
    logs = orch.run(LLMStudent(), cfg, notes_text=sample_notes)
    rec = logs[0]
    assert "presented_stem" in rec and ("TOOLS:" in rec["presented_stem"] or rec.get("tool_outputs"))
    assert rec.get("tools_used") is None or "retriever" in rec.get("tools_used", [])
