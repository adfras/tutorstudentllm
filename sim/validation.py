from __future__ import annotations
import re
import unicodedata
from typing import Dict, Any, Optional, Tuple

# Unicode canonicalization helpers
DASH_EQUIV = r"[\u002D\u2010\u2011\u2012\u2013\u2212]"  # -, hyphen, non-breaking hyphen, figure dash, en-dash, minus
SPACE_EQUIV = r"[\u00A0\s]"  # NBSP + whitespace


def canon(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.casefold()
    s = re.sub(SPACE_EQUIV, " ", s)
    s = re.sub(DASH_EQUIV, "-", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_count(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", s or ""))


def first_n_tokens_span(text: str, n: int = 15) -> Tuple[int, int, str]:
    """Return (start, end, slice) covering the first n alnum tokens in text.
    Preserves original punctuation/spacing by slicing the original string between
    the start of the first token and the end of the n-th token.
    If no tokens found, returns (0, 0, '').
    """
    if not text:
        return (0, 0, "")
    it = list(re.finditer(r"[A-Za-z0-9]+", text))
    if not it:
        return (0, 0, "")
    n = max(1, int(n))
    n = min(n, len(it))
    start = it[0].start()
    end = it[n - 1].end()
    return (start, end, text[start:end])


def validate_option_quote(presented_option: str, card: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that card's quote points inside presented_option.
    Trust offsets first (when provided and valid), otherwise perform a canonicalized
    substring check (NFKC + casefold + dash/space unification).
    Returns {ok: bool, mode: str, reason?: str} and may update card['quote'] when
    offsets are authoritative.
    """
    opt = presented_option or ""
    w = card.get("where") or {}
    q = card.get("quote") or ""
    # A) Offsets win if valid
    s = w.get("start")
    e = w.get("end")
    if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(opt):
        span = opt[s:e]
        if canon(span) != canon(q):
            # rewrite quote to exact slice; still strict later
            card["quote"] = span
        return {"ok": True, "mode": "offset"}
    # B) Canonicalized substring (strict)
    cq = canon(q); co = canon(opt)
    if cq and (cq in co):
        return {"ok": True, "mode": "canon-substring"}
    # C) Loose dash/space equivalence: treat '-' and ' ' as equal when matching
    if cq and (cq.replace('-', ' ') in co.replace('-', ' ')):
        return {"ok": True, "mode": "canon-substring"}
    return {"ok": False, "reason": "quote_not_in_option"}
