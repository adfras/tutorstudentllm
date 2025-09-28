from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable
import math
import re

# Lightweight quote/card quality scoring utilities.
# Works with sim.evidence_schema.Quote and raw option texts.

_WORD = re.compile(r"[A-Za-z0-9]+")


def tokens(s: str) -> List[str]:
    return _WORD.findall((s or "").lower())


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def has_mid_ellipsis(s: str) -> bool:
    # Detect Unicode ellipsis or manual ... between word chars
    return ("…" in (s or "")) or bool(re.search(r"\w\s?\.\.\.\s?\w", s or ""))


def build_idf(option_texts: List[str]) -> Dict[str, float]:
    # Very small IDF over option texts to reward option-specific terms
    docs = [set(tokens(t)) for t in option_texts]
    N = len(docs) or 1
    df: Dict[str, int] = {}
    for d in docs:
        for t in d:
            df[t] = df.get(t, 0) + 1
    return {t: math.log(1.0 + N / (1.0 + c)) for t, c in df.items()}


def rel_score(quote: str, option_text: str, idf: Dict[str, float] | None = None) -> float:
    qt = set(tokens(quote))
    ot = set(tokens(option_text))
    if not qt or not ot:
        return 0.0
    idf = idf or {}
    return sum(idf.get(t, 1.0) for t in (qt & ot)) / math.sqrt(len(qt) + 1.0)


def specificity(quote: str, option_text: str) -> float:
    qt = tokens(quote)
    ot = set(tokens(option_text))
    if not qt or not ot:
        return 0.0
    return sum(1 for t in qt if t in ot) / max(1, len(qt))


_NEG = {"not", "no", "none", "never", "without", "except", "unless", "however", "but"}


def negation_density(quote: str) -> float:
    qt = tokens(quote)
    return sum(1 for t in qt if t in _NEG) / max(1, len(qt))


def authority(source_id: str) -> float:
    s = (source_id or "").lower()
    if ".gov" in s or ".edu" in s:
        return 1.0
    if any(x in s for x in (".org", "nature.com", "nejm.org", "who.int")):
        return 0.75
    return 0.5


@dataclass
class QuoteScore:
    cqs: float
    reasons: List[str]
    features: Dict[str, float]


def assess_quote(quote_text: str,
                 option_text: str,
                 other_option_texts: List[str],
                 source_id: str,
                 *,
                 idf: Dict[str, float] | None = None,
                 min_len: int = 40,
                 max_len: int = 300) -> QuoteScore:
    L = len(quote_text or "")
    length_ok = (L >= int(min_len) and L <= int(max_len))
    ellipses = has_mid_ellipsis(quote_text or "")
    rel_self = rel_score(quote_text, option_text, idf)
    rel_others = max((rel_score(quote_text, o, idf) for o in other_option_texts), default=0.0)
    spec_self = specificity(quote_text, option_text)
    overlap_ratio = 0.0 if rel_self <= 0 else min(1.0, rel_others / (rel_self + 1e-6))
    neg = negation_density(quote_text)
    auth = authority(source_id)

    cqs = (
        0.35 * min(1.0, rel_self)
        + 0.20 * spec_self
        + 0.15 * auth
        + 0.15 * (1.0 - overlap_ratio)
        + 0.10 * (1.0 if length_ok else 0.0)
        + 0.05 * (1.0 - min(1.0, neg * 5))
    )
    if ellipses:
        cqs *= 0.85

    reasons: List[str] = []
    if not length_ok:
        reasons.append(f"bad_length:{L}")
    if ellipses:
        reasons.append("mid_ellipses")
    if overlap_ratio > 0.5:
        reasons.append("overlaps_other_options")
    if rel_self < 0.2:
        reasons.append("low_relevance")
    if spec_self < 0.02:
        reasons.append("low_specificity")

    feats = dict(
        length=float(L),
        length_ok=float(length_ok),
        rel_self=float(rel_self),
        rel_others=float(rel_others),
        specificity=float(spec_self),
        overlap_ratio=float(overlap_ratio),
        negation=float(neg),
        authority=float(auth),
    )
    return QuoteScore(cqs=float(cqs), reasons=reasons, features=feats)

