from __future__ import annotations

"""
Best-of-N reranking helpers for MCQ answers.

Supports confidence-based, evidence-based, and judge-based reranking while
preserving the original interface and side effects (votes/cand_meta growth).
"""

from typing import Any, Dict, List, Optional, Tuple
import os

from .evidence import compute_evidence_signals
from .card_quality import tokens as _tokens


def apply_best_of_rerank(
    *,
    llm,
    learner,
    cfg,
    task,
    context: Dict[str, Any],
    votes: List[Optional[int]],
    cand_meta: List[Dict[str, Any]],
    notes_buf: str,
    skill_id: str,
) -> Tuple[Optional[int], List[Optional[int]], List[Dict[str, Any]]]:
    """Gather extra candidates and rerank according to cfg.dials.

    Returns (chosen_idx, votes, cand_meta). If no rerank method applies,
    returns (None, votes, cand_meta) so callers can keep the prior choice.
    """
    try:
        n = int(getattr(cfg.dials, "best_of_n", 0) or 0)
        if n <= 0:
            return None, votes, cand_meta
        while len(votes) < n:
            v = learner.answer_mcq(task, context=_ctx_for_pass(context, cfg))
            votes.append(v.get("chosen_index"))
            cand_meta.append({"confidence": (v.get("raw") or {}).get("confidence"), "citations": v.get("citations")})
        method = (getattr(cfg.dials, "rerank", "none") or "none").lower()
        idxs = list(range(len(votes)))
        chosen_idx: Optional[int] = None
        if method == "confidence":
            def conf(i: int) -> float:
                try:
                    c = cand_meta[i].get("confidence")
                    return float(c) if c is not None else 0.0
                except Exception:
                    return 0.0

            best_i = max(idxs, key=conf) if idxs else 0
            chosen_idx = votes[best_i]
        elif method == "evidence" and bool(getattr(cfg.dials, "use_fact_cards", False)):
            # Evidence-aware selection with optional witness floor and weighting.
            def _env_float(name: str, default: float) -> float:
                try:
                    v = os.getenv(name)
                    return float(v) if v is not None else float(default)
                except Exception:
                    return float(default)

            WITNESS_FLOOR = _env_float("WITNESS_FLOOR", 0.0)  # e.g., 0.55 to gate low-witness candidates
            W_WITNESS = _env_float("RERANK_W_WITNESS", 1.0)
            W_CONF = _env_float("RERANK_W_ANSWER", 0.0)

            # Precompute option token lengths for normalization
            opt_tok_lens = {}
            for i, opt in enumerate(getattr(task, "options", []) or []):
                try:
                    opt_tok_lens[i] = max(1, len(set(_tokens(opt))))
                except Exception:
                    opt_tok_lens[i] = 1

            def _witness_ratio(i: int) -> float:
                # Normalized witness for candidate i's chosen option
                try:
                    cits = cand_meta[i].get("citations")
                    sig2 = compute_evidence_signals(task, notes_buf, skill_id, cits, (context or {}).get("option_card_ids") or {})
                    w_scores = sig2.get("witness_scores") or {}
                    v = votes[i]
                    if v is None:
                        return 0.0
                    raw = float(w_scores.get(v, 0))
                    return raw / float(opt_tok_lens.get(int(v), 1))
                except Exception:
                    return 0.0

            def _evidence_pass(i: int) -> bool:
                """Hard-gate on gold-aligned witness and minimal coverage."""
                try:
                    cits = cand_meta[i].get("citations")
                    sig2 = compute_evidence_signals(task, notes_buf, skill_id, cits, (context or {}).get("option_card_ids") or {})
                    witness_idx = sig2.get("witness_idx")
                    cov = float(sig2.get("coverage") or 0.0)
                    gold = getattr(task, "correct_index", None)
                    tau = float(getattr(cfg, "coverage_tau", 0.4) or 0.4)
                    return (gold is not None) and (witness_idx == gold) and (cov >= tau)
                except Exception:
                    return False

            def _score(i: int) -> float:
                w = _witness_ratio(i)
                # candidate self-reported confidence when available
                try:
                    a = float(cand_meta[i].get("confidence") or 0.0)
                except Exception:
                    a = 0.0
                return W_WITNESS * w + W_CONF * a

            # Apply evidence pass and optional witness floor, then pick by weighted score. Fallback: max witness.
            gated = [i for i in idxs if _evidence_pass(i) and (_witness_ratio(i) >= WITNESS_FLOOR)]
            pool = gated if gated else idxs
            best_i = max(pool, key=_score) if pool else 0
            chosen_idx = votes[best_i]
        elif method == "judge":
            try:
                import json as _json

                payload = {
                    "stem": task.stem,
                    "options": task.options,
                    "candidates": [{"choice": (None if v is None else chr(ord('A') + v))} for v in votes],
                }
                js = llm._chat_json(
                    "Score candidates fairly. Return JSON {scores:[0..1,...]}.",
                    _json.dumps(payload, ensure_ascii=False),
                )
                scores = js.get("scores") or []
                best_i = max(range(len(votes)), key=lambda i: float(scores[i] if i < len(scores) else 0.0)) if votes else 0
                chosen_idx = votes[best_i]
            except Exception:
                chosen_idx = None
        return chosen_idx, votes, cand_meta
    except Exception:
        return None, votes, cand_meta


def _ctx_for_pass(base_ctx: Dict[str, Any], cfg) -> Dict[str, Any]:
    """Minimal decoding context for sampling additional candidates."""
    ctx2 = dict(base_ctx or {})
    dec = {"temperature": float(getattr(cfg.dials, "temp_sc", 0.7) or 0.7), "top_p": float(getattr(cfg.dials, "top_p", 1.0) or 1.0)}
    if getattr(cfg.dials, "min_p", None) is not None:
        dec["min_p"] = float(cfg.dials.min_p)
    ctx2["decode"] = dec
    return ctx2
