from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


class DomainStore:
    def __init__(self, root: Optional[str] = None):
        self.root = root or os.path.join("docs", "iclsim", "domains")
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        if yaml is None:
            return {}
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def get(self, domain_id: str) -> Dict[str, Any]:
        if domain_id in self._cache:
            return self._cache[domain_id]
        path = os.path.join(self.root, f"{domain_id}.yaml")
        data = self._load_yaml(path)
        self._cache[domain_id] = data
        return data

    def glossary_terms(self, domain_id: str) -> List[str]:
        data = self.get(domain_id)
        gl = (data.get("glossary") or {})
        return [k for k in gl.keys()]

    def mcq_examples(self, domain_id: str, skill_id: str) -> List[Dict[str, Any]]:
        data = self.get(domain_id)
        ex = (((data.get("examples") or {}).get("mcq") or {}).get(skill_id)) or []
        return list(ex)


# Note: invert_codebook was unused across the repository and removed to reduce footprint.
