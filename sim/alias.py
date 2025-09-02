from __future__ import annotations
import os
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def load_alias_families(path: str = os.path.join("docs", "iclsim", "alias_families.yaml")) -> Dict[str, Any]:
    if yaml is None or not os.path.exists(path):
        return {"families": []}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    fams = data.get("families") or []
    idx = {f.get("id"): f for f in fams if f.get("id")}
    return {"families": fams, "index": idx}

