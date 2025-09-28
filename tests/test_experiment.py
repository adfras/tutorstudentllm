import os, json, textwrap


def test_experiment_runner(tmp_path, monkeypatch):
    cfg_yaml = tmp_path / "exp.yaml"
    cfg_yaml.write_text(textwrap.dedent(
        """
        version: 1
        runs:
          - name: mcq_closed
            task: mcq
            steps: 2
            domain: general
            student: oracle
            dials:
              closed_book: true
              anonymize: true
          - name: saq_open
            task: saq
            steps: 1
            student: oracle
            dials:
              closed_book: true
              anonymize: true
        """
    ), encoding="utf-8")
    out = tmp_path / "out"
    from scripts.experiment import run_experiment
    # Avoid tutor calls by stubbing out task builders and SAQ grading
    from sim.orchestrator import Orchestrator
    from sim.tasks import MCQTask, SAQTask
    def fake_mcq(self, skill_id, cfg, codebook=None):
        return MCQTask(id="t1", prompt={}, stem="S", options=["A","B","C","D","E"], correct_index=0)
    def fake_saq(self, skill_id, cfg, codebook=None):
        return SAQTask(id="s1", prompt={}, stem="Define X", expected_points=[{"key":"x","required":True}], model_answer="x")
    monkeypatch.setattr(Orchestrator, "_make_mcq_task", fake_mcq)
    monkeypatch.setattr(Orchestrator, "_make_saq_task", fake_saq)
    import tutor.llm_openai as tlo
    monkeypatch.setattr(tlo.OpenAILLM, "grade_saq", staticmethod(lambda stem, pts, model_answer, ans: {"score": 1.0}))
    res = run_experiment(str(cfg_yaml), str(out))
    # Summary JSON exists and is valid
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert "overall" in summ and ("mcq" in summ.get("overall", {}) or "saq" in summ.get("overall", {}))
    # Markdown summary exists
    assert (out / "summary.md").exists()
