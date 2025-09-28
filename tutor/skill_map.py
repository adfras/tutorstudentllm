from __future__ import annotations
import os
from typing import Dict, Any, Optional

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None


def load_skill_map(path: Optional[str] = None) -> Dict[str, Any]:
    # Allow override via env var; default to a domain-agnostic skill map
    default_path = "docs/iclsim/skill_map.general.yaml"
    path = path or os.getenv("SKILL_MAP_PATH", default_path)
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Please `pip install -r requirements.txt`.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Skill map not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    skills = {s["id"]: s for s in data.get("skills", [])}
    return {"meta": data, "skills": skills}


def skill_summary(sk: Dict[str, Any]) -> str:
    return f"{sk['name']} (id={sk['id']}, bloom={sk.get('bloom','')})"
