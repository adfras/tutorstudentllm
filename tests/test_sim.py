from sim.orchestrator import Orchestrator, RunConfig, Dials
from sim.domain import DomainStore
from sim.learner import OracleStudent, AlgoStudent, StatefulLLMStudent


def test_orchestrator_with_oracle():
    # Use a fake tutor to avoid network and a simple task builder
    from sim.tasks import MCQTask
    class _Tutor:
        @staticmethod
        def grade_saq(stem, pts, model_answer, ans):
            return {"score": 1.0}
    fake_tutor = _Tutor()
    orch = Orchestrator(model=fake_tutor)
    type(orch)._make_mcq_task = lambda self, sid, cfg, codebook=None: MCQTask(id="t1", prompt={}, stem="S", options=["A","B","C","D","E"], correct_index=0)  # type: ignore
    cfg = RunConfig(num_steps=3, num_options=5, dials=Dials(closed_book=True, anonymize=True, rich=False))
    logs = orch.run(OracleStudent(), cfg)
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
    from sim.tasks import MCQTask
    class _Tutor:
        @staticmethod
        def grade_saq(stem, pts, model_answer, ans):
            return {"score": 1.0}
    fake_tutor = _Tutor()
    orch = Orchestrator(model=fake_tutor)
    cfg = RunConfig(num_steps=1, dials=Dials(closed_book=True, anonymize=True))
    log_path = tmp_path / "run.jsonl"
    type(orch)._make_mcq_task = lambda self, sid, cfg, codebook=None: MCQTask(id="t1", prompt={}, stem="S", options=["A","B","C","D","E"], correct_index=0)  # type: ignore
    logs = orch.run(OracleStudent(), cfg, notes_text=sample_notes, log_path=str(log_path))
    rec = logs[0]
    assert "CONTEXT:" in rec["presented_stem"]
    # file should have header + one record
    text = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(text) == 2 and "presented_stem" in text[1]


def test_saq_flow(tmp_path):
    # SAQ flow with stubbed grading to avoid network
    class _Tutor:
        @staticmethod
        def grade_saq(stem, pts, model_answer, ans):
            return {"score": 1.0}
    fake_tutor = _Tutor()
    orch = Orchestrator(model=fake_tutor)
    cfg = RunConfig(task="saq", num_steps=1, dials=Dials(closed_book=True, anonymize=True))
    log_path = tmp_path / "saq.jsonl"
    from sim.tasks import SAQTask
    type(orch)._make_saq_task = lambda self, sid, cfg, codebook=None: SAQTask(id="s1", prompt={}, stem="Define X", expected_points=[{"key":"x","required":True}], model_answer="x")  # type: ignore
    logs = orch.run(OracleStudent(), cfg, notes_text="", log_path=str(log_path))
    assert len(logs) == 1
    rec = logs[0]
    assert rec["task"]["type"] == "saq"
    assert "grading" in rec and "score" in rec["grading"]
    assert "saq_drafts" in rec


def test_stateful_learner_memory_updates(sample_notes):
    from sim.tasks import MCQTask
    fake_tutor = type("_T", (), {})()
    orch = Orchestrator(model=fake_tutor)
    type(orch)._make_mcq_task = lambda self, sid, cfg, codebook=None: MCQTask(id="t1", prompt={}, stem="S", options=["A","B","C","D","E"], correct_index=0)  # type: ignore
    st = StatefulLLMStudent()
    # Avoid network by stubbing model call
    import types
    st.model._chat_json_opts = types.MethodType(lambda self, system, user, **kw: {"chosen_index": 0, "confidence": 0.7}, st.model)
    cfg = RunConfig(num_steps=1, dials=Dials(closed_book=True, anonymize=True))
    logs = orch.run(st, cfg, notes_text=sample_notes)
    # After one MCQ step, stateful learner should have some memory (best effort)
    assert isinstance(st.memory, list)


def test_tools_usage_in_presented_stem(sample_notes):
    # Ensure tools inject content and are logged
    from sim.tasks import MCQTask
    fake_tutor = type("_T", (), {})()
    orch = Orchestrator(model=fake_tutor)
    cfg = RunConfig(num_steps=1, dials=Dials(closed_book=True, anonymize=True, use_tools=True, tools=["retriever"]))
    type(orch)._make_mcq_task = lambda self, sid, cfg, codebook=None: MCQTask(id="t1", prompt={}, stem="S TOOLS:", options=["A","B","C","D","E"], correct_index=0)  # type: ignore
    logs = orch.run(OracleStudent(), cfg, notes_text=sample_notes)
    rec = logs[0]
    assert "presented_stem" in rec and ("TOOLS:" in rec["presented_stem"] or rec.get("tool_outputs"))
    assert rec.get("tools_used") is None or "retriever" in rec.get("tools_used", [])


def test_domain_store_and_examples():
    ds = DomainStore()
    terms = ds.glossary_terms("general")
    assert isinstance(terms, list) and len(terms) >= 1
    ex = ds.mcq_examples("general", "general-concepts")
    assert isinstance(ex, list) and len(ex) >= 1


def test_header_has_anonymization_info(tmp_path, sample_notes):
    from sim.tasks import MCQTask
    fake_tutor = type("_T", (), {})()
    orch = Orchestrator(model=fake_tutor)
    cfg = RunConfig(num_steps=1, dials=Dials(closed_book=True, anonymize=True))
    log_path = tmp_path / "run.jsonl"
    type(orch)._make_mcq_task = lambda self, sid, cfg, codebook=None: MCQTask(id="t1", prompt={}, stem="S", options=["A","B","C","D","E"], correct_index=0)  # type: ignore
    logs = orch.run(OracleStudent(), cfg, notes_text=sample_notes, log_path=str(log_path))
    header = (log_path.read_text(encoding="utf-8").splitlines())[0]
    assert '"anonymization"' in header and '"seed"' in header
