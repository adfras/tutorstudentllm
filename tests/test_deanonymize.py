import os, json


def test_deanonymize_presented_stem(tmp_path, monkeypatch):
    monkeypatch.setenv("TUTOR_MOCK_LLM", "1")
    from sim.orchestrator import Orchestrator, RunConfig, Dials
    from sim.learner import LLMStudent

    orch = Orchestrator()
    cfg = RunConfig(skill_id='cog-learning-theories', task='mcq', num_steps=1, dials=Dials(closed_book=True, anonymize=True))
    log_path = tmp_path / "run.jsonl"
    orch.run(LLMStudent(), cfg, notes_text='operant conditioning increases behavior', log_path=str(log_path))

    # De-anonymize
    out_path = tmp_path / "run_deanon.jsonl"
    from scripts.deanonymize import process
    process(str(log_path), str(out_path))
    de_lines = out_path.read_text(encoding='utf-8').splitlines()
    # Search for an expected domain term in presented_stem after de-anonymization
    rec = json.loads(de_lines[1])
    assert 'EXAMPLE:' in rec.get('presented_stem', '') or 'operant' in rec.get('presented_stem', '').lower()

