from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, runtime_checkable
import re


@runtime_checkable
class Tool(Protocol):
    name: str
    def run(self, *, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        ...


@dataclass
class SnippetRetriever:
    name: str = "retriever"
    k: int = 3

    def _tok(self, s: str) -> set[str]:
        return set([t for t in re.findall(r"[a-zA-Z0-9]+", (s or "").lower()) if len(t) >= 3])

    def run(self, *, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        notes = (context or {}).get("notes_text") or ""
        if not notes:
            return {"name": self.name, "snippets": []}
        # Split notes into lines and score by token overlap with task stem/options
        lines = [ln.strip() for ln in notes.splitlines() if ln.strip()]
        qtext = (task.get("stem") or "") + "\n" + "\n".join(task.get("options", []))
        qtok = self._tok(qtext)
        scored: List[tuple[int, str]] = []
        for ln in lines:
            lt = self._tok(ln)
            score = len(lt & qtok)
            if score > 0:
                scored.append((score, ln))
        scored.sort(key=lambda x: x[0], reverse=True)
        snippets = [ln for _, ln in scored[: self.k]]
        return {"name": self.name, "snippets": snippets}


REGISTRY = {
    "retriever": SnippetRetriever,
}


def build_tools(names: List[str]) -> List[Tool]:
    tools: List[Tool] = []
    for n in names:
        cls = REGISTRY.get(n)
        if cls:
            tools.append(cls())
    return tools

