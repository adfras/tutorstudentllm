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
    k: int = 1

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
    k: int = 1

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

# ----------------- Option-Conditioned Retriever -----------------

from dataclasses import dataclass
from typing import Tuple


def _split_paragraphs(text: str) -> list[str]:
    # Split on blank lines; fallback to non-empty lines
    parts = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if parts:
        return parts
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


@dataclass
class OptionRetriever:
    name: str = "option_retriever"
    k: int = 2

    def _tok(self, s: str) -> set[str]:
        return set([t for t in re.findall(r"[A-Za-z0-9]+", (s or "").lower()) if len(t) >= 3])

    def _mmr_select(self, query_toks: set[str], passages: list[tuple[int, str]], k: int, lam: float) -> list[int]:
        # passages: list of (pid, text)
        if not passages:
            return []
        sims = []
        for pid, txt in passages:
            pt = self._tok(txt)
            inter = len(query_toks & pt)
            denom = (len(query_toks) + len(pt) - inter) or 1
            sims.append((inter / denom, pid))
        sims.sort(key=lambda x: x[0], reverse=True)
        order = [pid for _, pid in sims]
        if k <= 1 or len(order) <= 1:
            return order[:k]
        selected: list[int] = []
        cand = order.copy()
        while cand and len(selected) < k:
            if not selected:
                selected.append(cand.pop(0))
                continue
            best_pid = None
            best_score = -1e9
            for pid in cand:
                # relevance
                rel = next((s for s, p in sims if p == pid), 0.0)
                # maximal similarity to already selected
                rep = 0.0
                txt = next((t for p, t in passages if p == pid), "")
                pt = self._tok(txt)
                for spid in selected:
                    stxt = next((t for p, t in passages if p == spid), "")
                    spt = self._tok(stxt)
                    inter = len(pt & spt)
                    denom = (len(pt) + len(spt) - inter) or 1
                    rep = max(rep, inter / denom)
                score = lam * rel - (1.0 - lam) * rep
                if score > best_score:
                    best_score, best_pid = score, pid
            selected.append(best_pid)  # type: ignore
            cand = [x for x in cand if x != best_pid]
        return selected[:k]

    def run(self, *, task: dict, context: dict) -> dict:
        notes = (context or {}).get("notes_text") or ""
        if not notes:
            return {"name": self.name, "snippets": []}
        # Config from context
        rcfg = (context or {}).get("retriever_config") or {}
        k = int(rcfg.get("k") or self.k)
        lam = float(rcfg.get("mmr_lambda") or 0.4)
        parts = _split_paragraphs(notes)
        passages = list(enumerate(parts))  # (pid, text)
        stem = str(task.get("stem") or "")
        options = list(task.get("options") or [])
        out_snips: list[dict] = []
        for oi, opt in enumerate(options):
            qtext = f"{stem}\n{opt}"
            toks = self._tok(qtext)
            chosen = self._mmr_select(toks, passages, k=k, lam=lam)
            for pid in chosen:
                txt = passages[pid][1]
                src_id = f"notes:para:{pid}"
                out_snips.append({"text": txt, "source_id": src_id, "option_index": oi})
        return {"name": self.name, "snippets": out_snips}


REGISTRY["option_retriever"] = OptionRetriever

# ----------------- Safe Python Executor (Program-of-Thought) -----------------

import ast
from dataclasses import dataclass


class _SafeVisitor(ast.NodeVisitor):
    ALLOWED_NODES = (
        ast.Module, ast.Expr, ast.Expression, ast.Assign, ast.AnnAssign,
        ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load, ast.Store,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd, ast.Call, ast.Name, ast.Tuple, ast.List, ast.Dict,
        ast.Subscript, ast.Slice, ast.Index, ast.Compare, ast.Eq, ast.NotEq,
        ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.And, ast.Or, ast.BoolOp,
        ast.Is, ast.IsNot, ast.Not,
        ast.Return, ast.If, ast.For, ast.While, ast.Break, ast.Continue,
        ast.FunctionDef, ast.arguments, ast.arg, ast.IfExp,
    )
    ALLOWED_NAMES = {
        # math-like safe functions
        "abs", "round", "min", "max", "sum", "len",
        "pow",
        # expose a small math shim in exec env (see executor)
        "sqrt", "floor", "ceil",
        # built-in types
        "int", "float", "bool", "range", "enumerate",
    }

    def visit(self, node):
        if not isinstance(node, self.ALLOWED_NODES):
            raise ValueError(f"Disallowed AST node: {type(node).__name__}")
        return super().visit(node)

    def visit_Import(self, node):
        raise ValueError("Import not allowed")

    def visit_ImportFrom(self, node):
        raise ValueError("Import not allowed")

    def visit_Attribute(self, node):
        raise ValueError("Attribute access not allowed")

    def visit_Call(self, node: ast.Call):
        # Only allow calls to whitelisted simple names
        if not isinstance(node.func, ast.Name) or node.func.id not in self.ALLOWED_NAMES:
            raise ValueError("Only whitelisted function calls allowed")
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords or []:
            self.visit(kw.value)


@dataclass
class SafePythonExecutor:
    name: str = "pyexec"
    timeout_ms: int = 500

    def run(self, *, task: Dict[str, Any] | None = None, context: Dict[str, Any] | None = None, code: str | None = None, inputs: dict | None = None) -> Dict[str, Any]:
        """Execute small numeric/table programs safely.
        - Validates AST against an allowlist; no imports or attribute access.
        - Provides a tiny math environment; no builtins.
        - Returns {'ok': bool, 'result': any, 'error': str?}.
        """
        # If called from the generic tools loop without code, return a no-op
        if code is None and (task is not None or context is not None):
            return {"name": self.name, "ok": True, "skipped": True}
        try:
            # Parse and validate
            tree = ast.parse(code or "", mode="exec")
            _SafeVisitor().visit(tree)
        except Exception as e:
            return {"ok": False, "error": f"parse/validate: {e}"}
        # Prepare sandbox env
        safe_globals: Dict[str, Any] = {
            "__builtins__": {},
            # safe math-like surface without module attrs
            "sqrt": __import__("math").sqrt,
            "floor": __import__("math").floor,
            "ceil": __import__("math").ceil,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "pow": pow,
            "int": int,
            "float": float,
            "bool": bool,
            "range": range,
            "enumerate": enumerate,
        }
        safe_locals: Dict[str, Any] = {}
        if inputs:
            # Shallow copy of inputs, clipped to primitives/containers
            safe_locals.update({k: v for k, v in inputs.items()})
        # Execute with a simple timeout guard using signal only on Unix; fallback otherwise
        try:
            import signal
            class Timeout(Exception):
                pass
            def handler(signum, frame):
                raise Timeout()
            old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(max(1, int(self.timeout_ms/1000)))
            try:
                exec(compile(tree, filename="<safe>", mode="exec"), safe_globals, safe_locals)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
        except Exception as e:
            return {"ok": False, "error": f"exec: {e}"}
        # Retrieve result: prefer 'result', else 'solve()' if defined
        try:
            if "result" in safe_locals:
                return {"ok": True, "result": safe_locals["result"]}
            if "solve" in safe_locals and callable(safe_locals["solve"]):
                res = safe_locals["solve"]()
                return {"ok": True, "result": res}
            # If last expression was assigned, expose any simple name
            for k in ("ans","out","value"):
                if k in safe_locals:
                    return {"ok": True, "result": safe_locals[k]}
            return {"ok": True, "result": None}
        except Exception as e:
            return {"ok": False, "error": f"post: {e}"}


REGISTRY["pyexec"] = SafePythonExecutor
