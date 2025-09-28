from tutor.skill_map import load_skill_map


def test_load_skill_map_has_known_skill():
    smap = load_skill_map()
    assert "skills" in smap and isinstance(smap["skills"], dict)
    assert "general-concepts" in smap["skills"], "known skill id should exist"
