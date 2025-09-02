from tutor.utils import extract_json


def test_extract_json_plain():
    assert extract_json('{"a":1}') == {"a": 1}


def test_extract_json_fenced():
    text = """
```json
{"x": true, "n": 5}
```
"""
    assert extract_json(text) == {"x": True, "n": 5}


def test_extract_json_with_extras():
    text = "Here you go: {\n \"ok\": true, \n \"v\": 2\n } Thanks!"
    obj = extract_json(text)
    assert obj["ok"] is True and obj["v"] == 2

