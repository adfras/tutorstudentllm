from sim.validation import validate_option_quote, first_n_tokens_span


def test_dash_and_nbsp_pass():
    opt = "ZQ-49"
    card = {"quote": "ZQ 49", "where": {"scope": "option", "option_index": 0}}
    res = validate_option_quote(opt, card)
    assert res["ok"], "Canonicalized space/dash should match"


def test_offset_slice_wins():
    opt = "X1 equals X1"
    # Offsets point to "X1"
    card = {"quote": "X1 equals X1", "where": {"scope": "option", "option_index": 0, "start": 0, "end": 2}}
    res = validate_option_quote(opt, card)
    assert res["ok"]
    assert card["quote"] == "X1"


def test_wrong_option_fails():
    opt2 = "A-13"
    card = {"quote": "ZQ-49", "where": {"scope": "option", "option_index": 1}}
    res = validate_option_quote(opt2, card)
    assert not res["ok"], "Quote should not validate against wrong option"


def test_first_n_tokens_span_preserves_punct():
    text = "ZQ-49 maps to foo bar"
    s, e, sl = first_n_tokens_span(text, 2)
    assert sl == "ZQ-49", "Should slice original text including hyphen"

