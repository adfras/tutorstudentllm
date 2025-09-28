import json


def test_cli_summary_and_health(monkeypatch, capsys):
    # Fake orchestrator to avoid constructing real tutor wrappers
    class _FakeLLM:
        def verify_key_and_model(self):
            return {"ok": True, "model": "fake"}

    class _FakeOrch:
        def __init__(self, *args, **kwargs):
            self.llm = _FakeLLM()

        def run(self, learner, cfg, notes_text="", log_path=None, progress_cb=None):
            # Two records: one correct, one incorrect; attach simple usage and duration
            return [
                {
                    "task": {"type": "mcq"},
                    "evaluation": {"correct": True, "citations_evidence": {"credited": True}},
                    "tutor_usage": {"total_tokens": 10},
                    "student_usage": {"total_tokens": 20},
                    "duration_ms": 5,
                    "answer": {"chosen_index": 0},
                },
                {
                    "task": {"type": "mcq"},
                    "evaluation": {"correct": False, "citations_evidence": {"credited": False}},
                    "tutor_usage": {"total_tokens": 3},
                    "student_usage": {"total_tokens": 7},
                    "duration_ms": 4,
                    "answer": {"chosen_index": 1},
                },
            ]

    monkeypatch.setattr("sim.cli.Orchestrator", _FakeOrch)

    from sim import cli

    # Summary path
    rc = cli.main([
        "--student", "algo",
        "--steps", "2",
        "--summary",
    ])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    js = json.loads(out)
    assert js["summary"]["steps"] == 2
    assert js["summary"]["correct"] == 1
    assert js["summary"]["credited"] == 1
    assert js["summary"]["tutor_tokens"] == 13
    assert js["summary"]["student_tokens"] == 27

    # Health path
    capsys.readouterr()  # clear
    rc2 = cli.main(["--student", "algo", "--health"])
    assert rc2 == 0
    out2 = capsys.readouterr().out.strip()
    h = json.loads(out2)["health"]
    assert "tutor" in h and h["tutor"]["ok"] is True
    # For algo student, CLI reports ok by default
    assert h["student"]["ok"] is True

