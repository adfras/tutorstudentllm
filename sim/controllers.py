from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple


def _ensure_list(x):
    return x if isinstance(x, list) else ([] if x is None else [x])


def ltm_controller(llm, learner, task, base_ctx: Dict[str, Any], *, max_steps: int = 4) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Least-to-Most: PLAN (steps) -> SOLVE using learner with reasoning='ltm'.
    Returns (votes, meta). votes is a list[int|None] of chosen_index, meta has confidence.
    """
    import json
    system = (
        "Decompose the problem into the minimal numbered substeps. "
        "Return only JSON with steps (array of short strings, length 1..{})."
    ).format(max_steps)
    user = json.dumps({
        "stem": getattr(task, 'stem', ''),
        "options": getattr(task, 'options', []),
    }, ensure_ascii=False)
    plan = llm._chat_json(system, user)
    steps = plan.get("steps") or []
    steps = list(steps)[: max_steps]
    ctx2 = dict(base_ctx or {})
    ctx2["reasoning"] = "ltm"
    if steps:
        ctx2["controller_plan"] = steps
    ans = learner.answer_mcq(task, context=ctx2)
    return [ans.get("chosen_index")], [{"confidence": (ans.get("raw") or {}).get("confidence"), "controller": "ltm"}]


def tot_controller(llm, learner, task, base_ctx: Dict[str, Any], *, width: int = 2, depth: int = 2, judge: str = "self", budget: int = 6) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Tree-of-Thought (cheap): propose W approaches, evaluate each path up to depth D, pick best by confidence.
    Returns one vote (best) and meta.
    """
    import json
    width = max(1, int(width)); depth = max(1, int(depth)); budget = max(1, int(budget))
    # Propose root approaches
    sys_prop = (
        "Propose {} distinct high-level approaches as short phrases. "
        "Return only JSON {{approaches:[string,...]}}"
    ).format(width)
    prop = llm._chat_json(sys_prop, json.dumps({
        "stem": getattr(task, 'stem', ''),
        "options": getattr(task, 'options', []),
    }, ensure_ascii=False))
    approaches = list(prop.get("approaches") or [])[: width]
    nodes: List[Tuple[List[int|None], float, List[str]]] = []  # (vote_path, conf, hints)
    evals: List[Tuple[int|None, float, List[str]]] = []
    calls = 0
    for a in approaches:
        if calls >= budget: break
        ctx2 = dict(base_ctx or {}); ctx2["reasoning"] = "tot"; ctx2["controller_hint"] = str(a)
        r = learner.answer_mcq(task, context=ctx2)
        calls += 1
        conf = float((r.get("raw") or {}).get("confidence") or 0.0)
        evals.append((r.get("chosen_index"), conf, [str(a)]))
    # Optional one-step deepening for top-1
    if depth > 1 and evals:
        evals.sort(key=lambda t: t[1], reverse=True)
        top = evals[0]
        if calls < budget:
            sys_refine = "Refine the following approach for one additional step. Return JSON {{hint:string}}."
            refine = llm._chat_json(sys_refine, json.dumps({"approach": top[2][-1]}, ensure_ascii=False))
            hint = refine.get("hint") or top[2][-1]
            ctx3 = dict(base_ctx or {}); ctx3["reasoning"] = "tot"; ctx3["controller_hint"] = " | ".join(top[2] + [str(hint)])
            r2 = learner.answer_mcq(task, context=ctx3)
            calls += 1
            conf2 = float((r2.get("raw") or {}).get("confidence") or 0.0)
            evals.append((r2.get("chosen_index"), conf2, top[2] + [str(hint)]))
    if not evals:
        return [], []
    best = max(evals, key=lambda t: t[1])
    return [best[0]], [{"confidence": best[1], "controller": "tot", "hints": best[2]}]
