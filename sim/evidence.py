from __future__ import annotations

"""
Evidence helpers.

This module isolates small, testable utilities for computing evidence signals
from fact-cards and citations. Keeping this logic here reduces clutter in the
orchestrator and makes unit testing simpler.
"""

from typing import Any, Dict
from .card_quality import tokens as _tokens
from .utils.cards import quote_ok as _quote_ok


def compute_evidence_signals(task, notes_buf: str, skill_id: str, citations: list | None, option_card_ids: dict | None) -> Dict[str, Any]:
    """Compute evidence signals from cited card ids.

    Returns a dict with keys:
      - coverage: float
      - cited_ids: set[str]
      - witness_idx: int|None
      - witness_scores: dict[int,int]
      - has_option_quote: dict[int,bool]
      - required_ok: dict[int,bool]
    """
    import json as _json

    cited_ids: list[str] = []
    for x in (citations or []):
        try:
            if isinstance(x, dict) and x.get("id"):
                cited_ids.append(str(x.get("id")))
            elif isinstance(x, (str, int)):
                cited_ids.append(str(x))
        except Exception:
            continue
    idset = set(cited_ids)

    try:
        mem = _json.loads(notes_buf) if notes_buf else {"cards": []}
        cards = mem.get("cards") or []
    except Exception:
        cards = []

    # filter by ids, skill tag, short quotes (≤15 tokens)
    cited_cards = [
        c
        for c in cards
        if (not idset or (c.get("id") in idset))
        and (skill_id in (c.get("tags") or []))
        and _quote_ok(c.get("quote") or "")
    ]

    # token bag from cited quotes (all)
    ct_all = set(_tokens("\n".join([(c.get("quote") or "") for c in cited_cards])))

    # coverage against gold option text (use all cited quotes)
    gold = task.options[task.correct_index] if 0 <= task.correct_index < len(task.options) else ""
    gold_t = set(_tokens(gold))
    coverage = (len(gold_t & ct_all) / max(1, len(gold_t))) if gold_t else 0.0

    # witness scores per option: prioritize option-linked cited quotes for each option
    witness_scores: Dict[int, int] = {}
    # Build per-option token bags from cited option-linked cards
    per_opt_tokens: Dict[int, set[str]] = {i: set() for i in range(len(task.options))}
    for c in cited_cards:
        w = c.get("where") or {}
        if w.get("scope") == "option" and isinstance(w.get("option_index"), int):
            try:
                oi = int(w.get("option_index"))
            except Exception:
                continue
            # Guard against stale/mismatched cards pointing outside current options
            if 0 <= oi < len(task.options):
                per_opt_tokens[oi] |= set(_tokens(c.get("quote") or ""))
    # If no option-linked cites exist at all, fall back to all tokens
    use_fallback = all((len(per_opt_tokens.get(i, set())) == 0) for i in range(len(task.options)))
    for i, opt in enumerate(task.options):
        ot = set(_tokens(opt))
        src = (ct_all if use_fallback else per_opt_tokens.get(i, set()))
        witness_scores[i] = len(ot & src)
    witness_idx = max(witness_scores, key=witness_scores.get) if witness_scores else None

    # has option-linked quote per option (from cited cards)
    has_option_quote: Dict[int, bool] = {}
    for i in range(len(task.options)):
        has_option_quote[i] = any(
            (
                ((c.get("where") or {}).get("scope") == "option")
                and (((c.get("where") or {}).get("option_index") == i))
                for c in cited_cards
            )
        ) if cited_cards else False

    # required_ok: chosen option cited its required PRO id
    required_ok: Dict[int, bool] = {}
    opt_map = option_card_ids or {}
    for i in range(len(task.options)):
        letter = chr(ord('A') + i)
        req = str(opt_map.get(letter)) if isinstance(opt_map, dict) and letter in opt_map else None
        required_ok[i] = (not req) or (req in idset)

    return {
        "coverage": coverage,
        "cited_ids": idset,
        "witness_idx": witness_idx,
        "witness_scores": witness_scores,
        "has_option_quote": has_option_quote,
        "required_ok": required_ok,
    }


def score_options_from_cards(cards: list[dict], options: list[str]) -> Dict[str, Any]:
    """Compute coverage and witness scores per option from provided cards.

    Returns a dict with coverage_by_option, witness_score, quotes_sources.
    - coverage_by_option[i]: token overlap between option i and union of quoted tokens for i
    - witness_score[i]: size of token intersection (unnormalized)
    - quotes_sources[i]: set of distinct source_id strings contributing to option i
    """
    import re as _re
    coverage_by_option: Dict[int, float] = {}
    witness_score: Dict[int, int] = {}
    quotes_sources: Dict[int, set[str]] = {}
    for i, opt in enumerate(options or []):
        ot = set(_tokens(opt))
        qtoks: set[str] = set()
        srcs: set[str] = set()
        for c in cards or []:
            w = c.get("where") or {}
            if (w.get("scope") == "option") and (w.get("option_index") == i):
                q = c.get("quote") or ""
                qtoks |= set(_tokens(q))
                srcs.add(str(w.get("source_id") or f"option:{i}"))
        quotes_sources[i] = srcs
        coverage_by_option[i] = (len(ot & qtoks) / max(1, len(ot))) if ot else 0.0
        witness_score[i] = len(ot & qtoks)
    return {
        "coverage_by_option": coverage_by_option,
        "witness_score": witness_score,
        "quotes_sources": quotes_sources,
    }
