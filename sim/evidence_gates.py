from __future__ import annotations
from typing import Dict, List, Tuple
import re

from .evidence_schema import Quote


_TOK = re.compile(r"[A-Za-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(t.lower() for t in _TOK.findall(s or ""))


def per_option(quotes: List[Quote]) -> Dict[str, List[Quote]]:
    d: Dict[str, List[Quote]] = {}
    for q in quotes:
        d.setdefault(q.option, []).append(q)
    return d


def distinct_sources(quotes: List[Quote]) -> int:
    return len({q.source_id for q in quotes})


def coverage_for_option(quotes: List[Quote], option_text: str) -> float:
    """Token overlap coverage between concatenated quotes and the option text."""
    ot = _tokens(option_text)
    if not ot:
        return 0.0
    qt = set()
    for q in quotes:
        qt |= _tokens(q.text)
    return (len(qt & ot) / max(1, len(ot)))


def witness_overlap_ratio(chosen_quotes: List[Quote], other_quotes: List[Quote]) -> float:
    """Simple overlap: fraction of chosen (source_id,text) pairs that appear in other options."""
    if not chosen_quotes:
        return 0.0
    ch = {(q.source_id, q.text.strip()) for q in chosen_quotes}
    ot = {(q.source_id, q.text.strip()) for q in other_quotes}
    return len(ch & ot) / max(1, len(ch))


def evidence_health(quotes: List[Quote], options: List[str], chosen_letter: str | None = None, option_texts: Dict[str, str] | None = None) -> Dict[str, float | int]:
    by_opt = per_option(quotes)
    min_q = min((len(by_opt.get(letter, [])) for letter in options), default=0)
    cov = 0.0
    olap = 0.0
    srcs = 0
    if chosen_letter:
        ch = by_opt.get(chosen_letter, [])
        other = [q for ltr, lst in by_opt.items() if ltr != chosen_letter for q in lst]
        srcs = distinct_sources(ch)
        if option_texts and chosen_letter in option_texts:
            cov = coverage_for_option(ch, option_texts[chosen_letter])
        olap = witness_overlap_ratio(ch, other)
    return {
        "min_quotes_per_option": int(min_q),
        "coverage_chosen": float(cov),
        "witness_overlap_ratio": float(olap),
        "sources_chosen": int(srcs),
    }


def pass_pre_gates(quotes: List[Quote], options: List[str], *, q_min: int = 2, min_sources: int = 1) -> bool:
    by_opt = per_option(quotes)
    for ltr in options:
        qs = by_opt.get(ltr, [])
        if len(qs) < q_min:
            return False
        if distinct_sources(qs) < min_sources:
            return False
    return True


def pass_post_gates(quotes: List[Quote], chosen_letter: str, option_texts: Dict[str, str], *, q_min: int = 2, tau: float = 0.6, min_sources: int = 2) -> Tuple[bool, Dict[str, float | int]]:
    by_opt = per_option(quotes)
    ch = by_opt.get(chosen_letter, [])
    ok = (len(ch) >= q_min) and (distinct_sources(ch) >= min_sources) and (coverage_for_option(ch, option_texts.get(chosen_letter, "")) >= float(tau))
    return ok, evidence_health(quotes, list(option_texts.keys()), chosen_letter, option_texts)

