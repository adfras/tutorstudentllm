from __future__ import annotations

from typing import Any, Dict, List, Iterable

from .evidence import compute_evidence_signals
from .utils.cards import quote_ok as _quote_ok


def check(
    *,
    task,
    chosen_index: int | None,
    citations: List[Any] | None,
    notes_buf: str,
    skill_id: str,
    option_card_ids: Dict[str, Any] | None,
    retrieved_snippets: List[Any] | None,
    witness: Any | None = None,
    coverage_tau: float,
    q_min: int,
    tools_used: bool,
    cards_override: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Single-source credit check.

    Returns dict with: coverage, witness_pass, credited, reasons[list[str]].
    Policy:
      - require at least q_min cited cards that link to chosen option (when chosen_index is not None)
      - coverage >= tau vs gold option tokens
      - witness option matches gold (via overlap with cited quotes)
      - if tools are ON and snippets present: at least one cited quote must be a substring of any retrieved snippet
    """
    citations = citations or []
    # Signals from cited IDs
    sig = compute_evidence_signals(task, notes_buf, skill_id, citations, option_card_ids or {})
    coverage = float(sig.get("coverage") or 0.0)
    witness_idx = sig.get("witness_idx")
    witness_pass = bool(witness_idx is not None and witness_idx == getattr(task, "correct_index", None))
    cited_ids = set(sig.get("cited_ids") or [])
    has_option_quote = sig.get("has_option_quote") or {}

    # Rehydrate cited cards (≤15 tokens + skill tag) to check snippet provenance
    import json as _json
    try:
        mem = _json.loads(notes_buf) if notes_buf else {"cards": []}
        cards = (mem.get("cards") or [])
        if (not cards) and cards_override:
            cards = cards_override
    except Exception:
        cards = cards_override or []
    cited_cards = [
        c for c in (cards or [])
        if ((not cited_ids) or (str(c.get("id")) in cited_ids))
        and (skill_id in (c.get("tags") or [])) and _quote_ok(c.get("quote") or "")
    ]

    # Snippet policy (normalized substring match to avoid Unicode/spacing false negatives)
    snippet_ok = True
    if tools_used and retrieved_snippets:
        def _snip_text(x):
            if isinstance(x, dict):
                return x.get("text") or x.get("snippet") or ""
            return str(x)
        texts = [_snip_text(s) for s in retrieved_snippets]
        # Normalization helper
        import re as _re, unicodedata as _ud
        def _norm(s: str) -> str:
            try:
                s = _ud.normalize("NFKC", s or "")
                s = s.replace("“","\"").replace("”","\"").replace("’","'").replace("‘","'")
                s = s.replace("–","-").replace("—","-")
                s = _re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
                s = _re.sub(r"\s+", " ", s).strip().lower()
                return s
            except Exception:
                return (s or "").strip().lower()
        texts_n = [_norm(t) for t in texts]
        def _has_norm_substr(q: str, pool: list[str]) -> bool:
            qn = _norm(q)
            if not qn:
                return False
            return any(qn in tn for tn in pool)
        snippet_ok = any(_has_norm_substr((c.get("quote") or ""), texts_n) for c in cited_cards)
        # Allow structured WITNESS quotes (when present) to satisfy provenance if any witness quote
        # is a normalized substring of any retrieved snippet text.
        if not snippet_ok and witness is not None:
            def _iter_witness_quotes(w: Any) -> Iterable[str]:
                try:
                    if isinstance(w, dict):
                        for k in ("rule", "choice"):
                            v = w.get(k)
                            if isinstance(v, dict):
                                q = v.get("quote")
                                if isinstance(q, str) and q.strip():
                                    yield q
                            elif isinstance(v, list):
                                for it in v:
                                    if isinstance(it, dict):
                                        q = it.get("quote")
                                        if isinstance(q, str) and q.strip():
                                            yield q
                    elif isinstance(w, list):
                        for it in w:
                            if isinstance(it, dict):
                                q = it.get("quote")
                                if isinstance(q, str) and q.strip():
                                    yield q
                except Exception:
                    return
            w_quotes = list(_iter_witness_quotes(witness))
            def _norm(s: str) -> str:
                try:
                    s = _ud.normalize("NFKC", s or "")
                    s = s.replace("“","\"").replace("”","\"").replace("’","'").replace("‘","'")
                    s = s.replace("–","-").replace("—","-")
                    s = _re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
                    s = _re.sub(r"\s+", " ", s).strip().lower()
                    return s
                except Exception:
                    return (s or "").strip().lower()
            wq_norm = [_norm(q) for q in w_quotes if isinstance(q, str)]
            snippet_ok = any(any(qn and (qn in tn) for tn in texts_n) for qn in wq_norm)

    # Reasons
    reasons: List[str] = []
    if not citations:
        reasons.append("no_citations")
    if citations and not cited_cards:
        reasons.append("tag_or_quote_filter")
    # Chosen option must have ≥ q_min cited option-linked cards
    has_link = False
    if chosen_index is not None:
        has_link = bool(has_option_quote.get(int(chosen_index), False))
        # Count cited cards that link to chosen option
        link_count = 0
        for c in cited_cards:
            w = c.get("where") or {}
            if (w.get("scope") == "option") and (int(w.get("option_index") or -1) == int(chosen_index)):
                link_count += 1
        if not has_link:
            reasons.append("no_option_quote_for_choice")
            if link_count > 0 and link_count < int(q_min):
                reasons.append("link_below_q_min")
    if tools_used and retrieved_snippets and not snippet_ok:
        reasons.append("no_snippet_quote")
    if coverage < float(coverage_tau):
        reasons.append("coverage_below_tau")
    # Witness tie policy (optional): pass when gold is among top-tied options
    try:
        import os as _os
        tie_pass = (_os.getenv("TUTOR_WITNESS_TIE_PASS", "0").strip().lower() in ("1","true","yes","on"))
    except Exception:
        tie_pass = False
    if not witness_pass:
        # Detect tie among top witness scores
        try:
            w_scores = sig.get("witness_scores") or {}
            if w_scores:
                mx = max(w_scores.values())
                tied = [i for i, sc in w_scores.items() if sc == mx]
                gold = getattr(task, "correct_index", None)
                if tie_pass and (gold in tied) and (len(tied) > 1):
                    witness_pass = True
        except Exception:
            pass
    if not witness_pass:
        reasons.append("witness_mismatch")

    credited = bool(
        (chosen_index is not None)
        and not ("no_citations" in reasons)
        and not ("no_option_quote_for_choice" in reasons)
        and not (tools_used and retrieved_snippets and ("no_snippet_quote" in reasons))
        and (coverage >= float(coverage_tau))
        and witness_pass
    )
    return {
        "coverage": coverage,
        "witness_pass": witness_pass,
        "credited": credited,
        "reasons": reasons,
    }
