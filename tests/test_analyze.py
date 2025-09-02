import json


def write_log(tmp_path, steps):
    p = tmp_path / "run.jsonl"
    header = {"run_header": True, "run_id": "r1", "ts": 0, "config": {"task": "mcq", "difficulty": "medium", "dials": {"closed_book": True, "anonymize": True, "context_position": "pre", "verify": False, "self_consistency_n": 1, "rich": False}}}
    lines = [json.dumps(header)]
    lines += [json.dumps(s) for s in steps]
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def test_analyze_single_run_mcq(tmp_path):
    steps = [
        {"run_id":"r1","step":0,"ts":1,"task":{"type":"mcq"},"evaluation":{"correct": True}},
        {"run_id":"r1","step":1,"ts":2,"task":{"type":"mcq"},"evaluation":{"correct": False}},
        {"run_id":"r1","step":2,"ts":3,"task":{"type":"mcq"},"evaluation":{"correct": True}},
    ]
    path = write_log(tmp_path, steps)
    from scripts.analyze import parse_log, metrics_for_run, analyze_many
    run = parse_log(path)
    m = metrics_for_run(run)
    assert m["mcq"]["n"] == 3
    assert m["mcq"]["acc_final"] == 2/3
    assert len(m["mcq"]["cumulative"]) == 3
    res = analyze_many([path])
    assert res["overall"]["mcq"]["acc_final_mean"] == 2/3


def test_analyze_many_groups(tmp_path):
    # two runs, different verify dials
    steps = [{"run_id":"r1","step":0,"ts":1,"task":{"type":"mcq"},"evaluation":{"correct": True}}]
    p1 = tmp_path / "a.jsonl"
    h1 = {"run_header": True, "run_id": "r1", "ts": 0, "config": {"task": "mcq", "difficulty": "medium", "dials": {"closed_book": True, "anonymize": True, "context_position": "pre", "verify": False, "self_consistency_n": 1, "rich": False}}}
    p1.write_text("\n".join([json.dumps(h1)] + [json.dumps(s) for s in steps]), encoding="utf-8")
    p2 = tmp_path / "b.jsonl"
    h2 = {"run_header": True, "run_id": "r2", "ts": 0, "config": {"task": "mcq", "difficulty": "medium", "dials": {"closed_book": True, "anonymize": True, "context_position": "pre", "verify": True, "self_consistency_n": 1, "rich": False}}}
    p2.write_text("\n".join([json.dumps(h2)] + [json.dumps(s) for s in steps]), encoding="utf-8")
    from scripts.analyze import analyze_many
    res = analyze_many([str(p1), str(p2)])
    assert len(res["groups"]) == 2
    for v in res["groups"].values():
        assert v["mcq"]["acc_final_mean"] == 1.0

