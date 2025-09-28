from __future__ import annotations

"""
Helpers for Fact-Cards post-processing used by the orchestrator.
"""

from typing import Any, Dict, List
from .card_quality import tokens as _tokens
from .utils.cards import quote_ok as _quote_ok



def build_option_card_ids(new_cards: List[Dict[str, Any]], task) -> Dict[str, str]:
    """Return mapping like {'A': id, 'B': id, ...} for the first PRO card per option.

    Enforces the ≤15-token quote rule used by the evaluator.
    """
    option_card_ids: Dict[str, str] = {}
    try:
        for oi in range(len(getattr(task, "options", []) or [])):
            letter = chr(ord('A') + oi)
            cid = None
            for c in new_cards:
                w = c.get("where") or {}
                if (w.get("scope") == "option") and (w.get("option_index") == oi):
                    if _quote_ok(c.get("quote") or ""):
                        cid = c.get("id")
                        break
            if cid:
                option_card_ids[letter] = str(cid)
    except Exception:
        option_card_ids = {}
    return option_card_ids


def trim_card(c: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only minimal fields required by the student."""
    return {
        "id": c.get("id"),
        "quote": c.get("quote"),
        "where": c.get("where") or {},
        "tags": c.get("tags") or [],
    }


def clamp_cards(cards: list[dict], max_tokens: int = 15) -> list[dict]:
    """Clamp all card quotes to ≤ max_tokens using utils.truncate_quote.

    Idempotent; returns a new list with quotes trimmed where necessary.
    """
    from sim.utils.cards import clamp_card_quote
    out: list[dict] = []
    for c in (cards or []):
        try:
            out.append(clamp_card_quote(c, max_tokens=max_tokens))
        except Exception:
            out.append(c)
    return out


def dedup_per_option(cards: List[Dict[str, Any]], *, sim_threshold: float = 0.88) -> List[Dict[str, Any]]:
    """Remove near-duplicate option-scoped cards while preserving order.

    Two cards are considered duplicates if Jaccard(tokens(quote_a), tokens(quote_b))
    >= sim_threshold. Context-scoped cards are left as-is.
    """
    keep: List[Dict[str, Any]] = []
    # Group by option_index for option-scoped cards
    by_opt: Dict[int, List[Dict[str, Any]]] = {}
    others: List[Dict[str, Any]] = []
    for c in cards:
        w = c.get("where") or {}
        if (w.get("scope") == "option") and isinstance(w.get("option_index"), int):
            by_opt.setdefault(int(w.get("option_index")), []).append(c)
        else:
            others.append(c)
    def _jacc(a: List[str], b: List[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 0.0
        inter = len(sa & sb)
        denom = (len(sa | sb)) or 1
        return inter / denom
    for oi in sorted(by_opt.keys()):
        seen: List[List[str]] = []
        for c in by_opt[oi]:
            qt = _tokens(c.get("quote") or "")
            drop = False
            for st in seen:
                if _jacc(qt, st) >= sim_threshold:
                    drop = True
                    break
            if not drop:
                keep.append(c)
                seen.append(qt)
    keep.extend(others)
    return keep
