import json


def test_evidence_pipeline_post_use_basic(monkeypatch):
    # Build a simple MCQ task with two options
    from sim.tasks import MCQTask
    task = MCQTask(id="t1", prompt={}, stem="S", options=["alpha beta", "gamma delta"], correct_index=0)

    # Notes buffer with two option-linked PRO cards for option A
    cards = {
        "cards": [
            {"id": "pa1", "quote": "alpha", "where": {"scope": "option", "option_index": 0, "source_id": "option:0"}, "tags": ["skill"]},
            {"id": "pa2", "quote": "beta",  "where": {"scope": "option", "option_index": 0, "source_id": "option:0"}, "tags": ["skill"]},
        ]
    }
    notes_buf = json.dumps(cards)

    from sim.evidence_pipeline import post_use_checks
    rep = post_use_checks(
        task=task,
        notes_buf=notes_buf,
        skill_id="skill",
        citations=["pa1", "pa2"],
        option_card_ids={"A": "pa1"},
        chosen_index=0,
        coverage_tau=0.1,
    )
    assert rep.post_pass is True
    assert rep.abstain_reason is None
    assert rep.witness_pass is True


def test_quote_ok_and_truncate():
    from sim.utils.cards import quote_ok
    from sim.utils.text import truncate_quote
    assert quote_ok("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen") is True
    assert quote_ok("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen") is False
    s = truncate_quote("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen", 15)
    assert len(s.split()) == 15

