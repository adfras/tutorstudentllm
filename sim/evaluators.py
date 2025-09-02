from __future__ import annotations
from typing import Any, Dict, List
import ast


def _safe_ast(code: str) -> ast.Module:
    tree = ast.parse(code)
    allowed = (
        ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Load, ast.Store,
        ast.Return, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
        ast.Pow, ast.USub, ast.UAdd, ast.Num, ast.Constant, ast.Expr, ast.Compare,
        ast.Eq, ast.NotEq, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.If, ast.IfExp,
        ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Name, ast.Assign, ast.AnnAssign,
        ast.Call, ast.Tuple, ast.List, ast.Dict, ast.keyword, ast.Pass,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"Disallowed AST node: {type(node).__name__}")
        if isinstance(node, ast.Call):
            # disallow __import__ and attribute calls; only Name calls allowed from whitelist later
            if not isinstance(node.func, ast.Name):
                raise ValueError("Disallowed call target")
            if node.func.id in {"__import__"}:
                raise ValueError("Disallowed builtin")
    return tree


def evaluate_code_python(function_name: str, student_code: str, tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        tree = _safe_ast(student_code)
    except Exception as e:
        return {"ok": False, "error": f"ast:{e}"}
    env: Dict[str, Any] = {"__builtins__": {}}
    try:
        compiled = compile(tree, filename="<student>", mode="exec")
        exec(compiled, env, env)
        fn = env.get(function_name)
        if not callable(fn):
            return {"ok": False, "error": "function_not_found"}
    except Exception as e:
        return {"ok": False, "error": f"exec:{e}"}
    results = []
    passed = 0
    for t in tests:
        args = t.get("args") or []
        kwargs = t.get("kwargs") or {}
        expected = t.get("expected")
        try:
            got = fn(*args, **kwargs)
            ok = (got == expected)
        except Exception as e:
            got = None
            ok = False
            results.append({"args": args, "kwargs": kwargs, "ok": ok, "error": str(e)})
            continue
        if ok:
            passed += 1
        results.append({"args": args, "kwargs": kwargs, "ok": ok, "expected": expected, "got": got})
    return {"ok": True, "passed": passed, "total": len(tests), "results": results}


def evaluate_proof_step(answer: str, expected_keywords: List[str]) -> Dict[str, Any]:
    text = (answer or "").lower()
    miss = [k for k in expected_keywords if k and k.lower() not in text]
    ok = (len(miss) == 0)
    return {"ok": ok, "missing_keywords": miss}


def evaluate_table_qa(answer: str, expected: str) -> Dict[str, Any]:
    def norm(s: str) -> str:
        return (s or "").strip().lower()
    ok = (norm(answer) == norm(expected))
    return {"ok": ok, "expected": expected, "got": answer}

