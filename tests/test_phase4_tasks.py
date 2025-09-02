from sim.tasks import CodeTask, ProofTask, TableQATask
from sim.evaluators import evaluate_code_python, evaluate_proof_step, evaluate_table_qa
from sim.orchestrator import Orchestrator, RunConfig, Dials
from sim.learner import LLMStudent


def test_evaluate_code_python_pass():
    code = "def add(a,b):\n    return a+b\n"
    tests = [
        {"args": [1, 2], "kwargs": {}, "expected": 3},
        {"args": [-1, 3], "kwargs": {}, "expected": 2},
    ]
    res = evaluate_code_python("add", code, tests)
    assert res["ok"] and res["passed"] == 2 and res["total"] == 2


def test_evaluate_proof_step_keywords():
    res = evaluate_proof_step("By commutativity, a+b=b+a.", ["commutativity"])
    assert res["ok"] is True and res["missing_keywords"] == []


def test_evaluate_table_qa_exact():
    res = evaluate_table_qa("Bob", "bob")
    assert res["ok"] is True


def test_orchestrator_code_task_mock_llm():
    # Orchestrator makes a built-in code task; LLMStudent mock returns correct add
    cfg = RunConfig(task="code", num_steps=1)
    orch = Orchestrator()
    logs = orch.run(LLMStudent(), cfg)
    rec = logs[0]
    assert rec["task"]["type"] == "code"
    assert rec["evaluation"]["ok"] and rec["evaluation"]["passed"] == rec["evaluation"]["total"]

