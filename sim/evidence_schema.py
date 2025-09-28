from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Quote:
    option: str           # "A" | "B" | "C" | ...
    source_id: str        # URL/doc-id or synthetic like "option:0"
    text: str             # verbatim quote
    start: Optional[int] = None
    end: Optional[int] = None
    card_id: Optional[str] = None


def adapt_card_to_quote(card: Dict[str, Any]) -> Quote | None:
    """Adapt a Fact-Card (internal schema) to a canonical Quote.
    Accepts cards with keys: id, quote, where:{scope, option_index, start, end, source_id}.
    """
    try:
        where = card.get("where") or {}
        scope = (where.get("scope") or "").strip().lower()
        if scope != "option":
            return None
        oi = where.get("option_index")
        if not isinstance(oi, int) or oi < 0:
            return None
        letter = chr(ord('A') + oi)
        src = where.get("source_id") or f"option:{oi}"
        txt = card.get("quote") or ""
        if not txt:
            return None
        s = where.get("start") if isinstance(where.get("start"), int) else None
        e = where.get("end") if isinstance(where.get("end"), int) else None
        return Quote(option=letter, source_id=str(src), text=str(txt), start=s, end=e, card_id=str(card.get("id") or ""))
    except Exception:
        return None


def adapt_cards_to_quotes(cards: List[Dict[str, Any]] | None) -> List[Quote]:
    out: List[Quote] = []
    for c in (cards or []):
        q = adapt_card_to_quote(c)
        if q:
            out.append(q)
    return out

