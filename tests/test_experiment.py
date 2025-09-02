import os, json, textwrap


def test_experiment_runner(tmp_path, monkeypatch):
    # Use mock LLM
    monkeypatch.setenv("TUTOR_MOCK_LLM", "1")
    cfg_yaml = tmp_path / "exp.yaml"
    cfg_yaml.write_text(textwrap.dedent(
        """
        version: 1
        runs:
          - name: mcq_closed
            task: mcq
            steps: 2
            domain: psych
            student: llm
            dials:
              closed_book: true
              anonymize: true
          - name: saq_open
            task: saq
            steps: 1
            student: llm
            dials:
              closed_book: true
              anonymize: true
        """
    ), encoding="utf-8")
    out = tmp_path / "out"
    from scripts.experiment import run_experiment
    res = run_experiment(str(cfg_yaml), str(out))
    # Summary JSON exists and is valid
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert "overall" in summ and ("mcq" in summ.get("overall", {}) or "saq" in summ.get("overall", {}))
    # Markdown summary exists
    assert (out / "summary.md").exists()

