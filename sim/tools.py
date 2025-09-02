from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, runtime_checkable
import math
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


# ----------------- Optional TF-IDF Retriever (no external deps) -----------------

@dataclass
class TFIDFRetriever:
    name: str = "tfidf_retriever"
    k: int = 3

    def _tok(self, s: str) -> List[str]:
        import re
        return [t for t in re.findall(r"[a-zA-Z0-9]+", (s or "").lower()) if len(t) >= 3]

    def run(self, *, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        notes = (context or {}).get("notes_text") or ""
        if not notes:
            return {"name": self.name, "snippets": []}
        lines = [ln.strip() for ln in notes.splitlines() if ln.strip()]
        if not lines:
            return {"name": self.name, "snippets": []}
        # Build IDF across lines
        N = len(lines)
        df: Dict[str, int] = {}
        line_toks: List[List[str]] = []
        for ln in lines:
            toks = self._tok(ln)
            line_toks.append(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        idf = {t: math.log(1.0 + N / (1.0 + dfc)) for t, dfc in df.items()}
        # Query tokens from stem+options
        qtext = (task.get("stem") or "")
        for opt in task.get("options", []) or []:
            qtext += "\n" + (opt or "")
        qtoks = self._tok(qtext)
        if not qtoks:
            return {"name": self.name, "snippets": []}
        # Compute cosine similarity with simple TF-IDF (per line)
        from collections import Counter
        qtf = Counter(qtoks)
        qvec = {t: qtf[t] * idf.get(t, 0.0) for t in qtf}
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        scored: List[tuple[float, str]] = []
        for ln, toks in zip(lines, line_toks):
            tf = Counter(toks)
            vec = {t: tf[t] * idf.get(t, 0.0) for t in tf}
            dot = 0.0
            for t, w in qvec.items():
                dot += w * vec.get(t, 0.0)
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            sim = dot / (qnorm * norm)
            if sim > 0:
                scored.append((sim, ln))
        scored.sort(key=lambda x: x[0], reverse=True)
        return {"name": self.name, "snippets": [ln for _, ln in scored[: self.k]]}

# register
REGISTRY["tfidf_retriever"] = TFIDFRetriever
