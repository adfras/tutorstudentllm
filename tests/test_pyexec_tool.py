import os
import json

from sim.tools import SafePythonExecutor


def test_pyexec_addition():
    exe = SafePythonExecutor()
    code = "result = 2 + 3 * 4"
    out = exe.run(code=code)
    assert out["ok"] is True
    assert out["result"] == 14


def test_pyexec_table_rows():
    exe = SafePythonExecutor()
    rows = [{"name": "ann", "score": 3}, {"name": "bob", "score": 5}]
    code = (
        "best = None\n"
        "for r in rows:\n"
        "    if (best is None) or (r['score'] > best['score']):\n"
        "        best = r\n"
        "result = best['name']\n"
    )
    out = exe.run(code=code, inputs={"rows": rows})
    assert out["ok"] is True
    assert out["result"] == "bob"


def test_pyexec_blocks_imports_and_attrs():
    exe = SafePythonExecutor()
    bad = "import os\nresult = 1"
    out = exe.run(code=bad)
    assert out["ok"] is False
    bad2 = "result = (__import__('os').getcwd())"
    out2 = exe.run(code=bad2)
    assert out2["ok"] is False

