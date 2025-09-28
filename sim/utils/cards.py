from __future__ import annotations

"""
Card and quote utilities.

Centralizes common operations around Fact-Cards and quotes: length checks,
tag normalization, and simple de-duplication.
"""

from typing import Any, Dict, Iterable, List, Tuple
from .text import tokens, truncate_quote
from sim.card_quality import jaccard as _jaccard


def quote_ok(q: str | None, max_tokens: int = 15) -> bool:
    return len(tokens(q)) <= int(max_tokens)


def ensure_skill_tag(card: Dict[str, Any], skill_id: str | None) -> Dict[str, Any]:
    if not skill_id:
        return card
    tags = card.get("tags") or []
    if skill_id not in tags:
        card = dict(card)
        card["tags"] = list(tags) + [skill_id]
    return card


def clamp_card_quote(card: Dict[str, Any], max_tokens: int = 15) -> Dict[str, Any]:
    q = card.get("quote") or ""
    tq = truncate_quote(q, max_tokens=max_tokens)
    if tq != q:
        c2 = dict(card)
        c2["quote"] = tq
        return c2
    return card


def dedup_quotes(quotes: Iterable[str], *, jaccard_thresh: float = 0.88) -> List[str]:
    """Return quotes with near-duplicates removed using Jaccard over tokens."""
    out: List[str] = []
    for q in quotes:
        qt = set(tokens(q))
        if not qt:
            continue
        keep = True
        for r in out:
            if _jaccard(qt, set(tokens(r))) >= float(jaccard_thresh):
                keep = False
                break
        if keep:
            out.append(q)
    return out
