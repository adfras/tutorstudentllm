import os


def test_openai_mock_generate_mcq_minimal(monkeypatch):
    monkeypatch.setenv("TUTOR_MOCK_LLM", "1")
    from tutor.llm_openai import OpenAILLM
    from tutor.skill_map import load_skill_map

    smap = load_skill_map()
    skill = smap["skills"]["cog-learning-theories"]
    llm = OpenAILLM()
    q = llm.generate_mcq(skill, difficulty="easy", num_options=5, minimal=True)
    assert set(q.keys()) == {"stem", "options", "correct_index"}
    assert isinstance(q["stem"], str) and len(q["options"]) == 5
    assert 0 <= q["correct_index"] < len(q["options"])


def test_openai_mock_generate_mcq_rich(monkeypatch):
    monkeypatch.setenv("TUTOR_MOCK_LLM", "1")
    from tutor.llm_openai import OpenAILLM
    from tutor.skill_map import load_skill_map

    smap = load_skill_map()
    skill = smap["skills"]["cog-learning-theories"]
    llm = OpenAILLM()
    q = llm.generate_mcq(skill, difficulty="medium", num_options=4, minimal=False)
    assert q["difficulty"] in ("easy", "medium", "hard")
    n = len(q["options"])
    assert n == 4
    assert len(q["rationales"]) == n and len(q["misconception_tags"]) == n
    assert 0 <= q["correct_index"] < n


def test_answer_and_grade_mock(monkeypatch):
    monkeypatch.setenv("TUTOR_MOCK_LLM", "1")
    from tutor.llm_openai import OpenAILLM
    llm = OpenAILLM()
    ans = llm.answer_mcq("stem", ["a", "b", "c"])
    assert isinstance(ans["chosen_index"], int)
    grading = llm.grade_saq(
        stem="Define memory",
        expected_points=[{"key": "memory", "required": True}],
        model_answer="A definition.",
        student_answer="memory is something",
    )
    assert "score" in grading and 0.0 <= grading["score"] <= 1.0

