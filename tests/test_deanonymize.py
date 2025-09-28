import json
from sim.orchestrator import Orchestrator, RunConfig, Dials
from sim.learner import OracleStudent
from sim.tasks import MCQTask


def test_deanonymize_presented_stem(tmp_path, monkeypatch):
    orch = Orchestrator(model=type("_T", (), {})())
    cfg = RunConfig(skill_id='general-concepts', task='mcq', num_steps=1, dials=Dials(closed_book=True, anonymize=True))
    log_path = tmp_path / "run.jsonl"
    type(orch)._make_mcq_task = lambda self, sid, cfg, codebook=None: MCQTask(id="t1", prompt={}, stem="EXAMPLE: recognize patterns", options=["A","B"], correct_index=0)  # type: ignore
    orch.run(OracleStudent(), cfg, notes_text='recognizing patterns can improve decisions', log_path=str(log_path))

    # De-anonymize
    out_path = tmp_path / "run_deanon.jsonl"
    from scripts.deanonymize import process
    process(str(log_path), str(out_path))
    de_lines = out_path.read_text(encoding='utf-8').splitlines()
    # Search for an expected domain term in presented_stem after de-anonymization
    rec = json.loads(de_lines[1])
    assert 'EXAMPLE:' in rec.get('presented_stem', '') or 'pattern' in rec.get('presented_stem', '').lower()
