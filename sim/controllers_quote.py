from __future__ import annotations

"""
Evidence-first "quote-then-vote" controller.

Moves the dedicated controller branch out of the orchestrator to simplify
control flow. This controller selects option-linked quotes, ensures per-option
coverage and diversity, then performs self-consistency voting with early-stop
quorum. It returns votes, candidate meta, and any citations surfaced by the
student model.
"""

from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

from .evidence import compute_evidence_signals
from .evidence_schema import adapt_cards_to_quotes
from .card_quality import assess_quote, build_idf, tokens as _cq_tokens, jaccard as _cq_jaccard
from .evidence_gates import pass_pre_gates, pass_post_gates, evidence_health


def quote_then_vote_controller(
    *,
    llm,
    learner,
    task,
    context: Dict[str, Any],
    cfg,
    fact_cards_after: Optional[Dict[str, Any]],
    notes_buf: str,
    skill_id: str,
) -> Tuple[List[Optional[int]], List[Dict[str, Any]], Optional[List[Any]], Optional[int], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Run the evidence-first controller.

    Returns (votes, cand_meta, citations, chosen_idx, evi_telemetry, card_metrics).
    """
    # Normalize existing Fact-Cards to quotes (option-linked only)
    fc = (fact_cards_after or {}).get("cards") if isinstance(fact_cards_after, dict) else None
    quotes_all = adapt_cards_to_quotes(fc or [])
    letters = [chr(ord('A') + i) for i in range(len(task.options))]
    opt_texts = {chr(ord('A') + i): task.options[i] for i in range(len(task.options))}
    try:
        idf = build_idf(list(opt_texts.values()))
    except Exception:
        idf = {}
    min_cqs = float(getattr(cfg.dials, 'min_cqs', 0.0) or 0.0)
    dedup_sim = float(getattr(cfg.dials, 'dedup_sim', 0.88) or 0.88)
    top_k = int(getattr(cfg.dials, 'per_option_top_k', 3) or 3)
    min_len = int(getattr(cfg.dials, 'min_card_len', 40) or 40)
    max_len = int(getattr(cfg.dials, 'max_card_len', 300) or 300)
    # Score quotes
    scored_quotes = []  # (quote, QuoteScore)
    for q in (quotes_all or []):
        try:
            other_texts = [opt_texts[l] for l in letters if l != q.option]
            s = assess_quote(q.text, opt_texts.get(q.option, ''), other_texts, q.source_id,
                             idf=idf, min_len=min_len, max_len=max_len)
            if s.cqs >= min_cqs:
                scored_quotes.append((q, s))
        except Exception:
            continue
    # Per-option pick: distinct sources + de-dup near-identical
    by_opt: dict[str, list[tuple[Any, Any]]] = {l: [] for l in letters}
    for q, s in scored_quotes:
        by_opt.setdefault(q.option, []).append((q, s))
    quotes: list[Any] = []
    for ltr in letters:
        cand = sorted(by_opt.get(ltr, []), key=lambda qs: qs[1].cqs, reverse=True)
        seen_src: set[str] = set()
        seen_texts: list[str] = []
        picked = 0
        for q, s in cand:
            if picked >= top_k:
                break
            if q.source_id in seen_src:
                continue
            t = q.text or ""
            if any(_cq_jaccard(_cq_tokens(t), _cq_tokens(t2)) >= dedup_sim for t2 in seen_texts):
                continue
            quotes.append(q)
            seen_src.add(q.source_id)
            seen_texts.append(t)
            picked += 1
    # Pre-selection gates and optional boosts
    boosts = 0
    while not pass_pre_gates(quotes, letters, q_min=int(cfg.dials.q_min or 2), min_sources=1) and boosts < int(cfg.dials.max_learn_boosts or 0):
        try:
            jsb = learner.extract_fact_cards(task, source_text=(context or {}).get("context_text") or task.stem, context=context)
            more_cards = (jsb or {}).get("cards") or []
            more_q_all = adapt_cards_to_quotes(more_cards)
            # Score new quotes too
            more_q: list[Any] = []
            for q in (more_q_all or []):
                other_texts = [opt_texts[l] for l in letters if l != q.option]
                s = assess_quote(q.text, opt_texts.get(q.option, ''), other_texts, q.source_id,
                                 idf=idf, min_len=min_len, max_len=max_len)
                if s.cqs >= min_cqs:
                    more_q.append(q)
            quotes.extend(more_q)
        except Exception:
            pass
        boosts += 1
    # Post gates
    ok, gate_health = pass_post_gates(
        quotes,
        None,
        opt_texts,
        q_min=int(cfg.dials.q_min or 2),
        tau=float(getattr(cfg, 'coverage_tau', 0.4) or 0.4),
        min_sources=int(getattr(cfg.dials, 'min_sources_chosen', 2) or 2),
    )
    # Compute overall evidence health snapshot (min per-option quotes, coverage, etc.).
    # Use the correct signature: (quotes, options_letters, chosen_letter=None, option_texts=None)
    _ = evidence_health(quotes, letters)
    # Self-consistency with optional early-stop quorum
    citations: Optional[List[Any]] = None
    votes: List[Optional[int]] = []
    cand_meta: List[Dict[str, Any]] = []
    n_max = max(1, int(getattr(cfg.dials, 'self_consistency_n', 1) or 1))
    if bool(getattr(cfg.dials, 'adaptive_sc', False)) and n_max > 1:
        # early stopping
        v0 = learner.answer_mcq(task, context=_ctx_for_pass(context, cfg, temp=float(getattr(cfg.dials, 'temp_answer', 0.3) or 0.3)))
        votes.append(v0.get("chosen_index"))
        cand_meta.append({"confidence": (v0.get("raw") or {}).get("confidence"), "citations": v0.get("citations")})
        if v0.get("citations"):
            citations = v0.get("citations")
        quorum = int(getattr(cfg.dials, 'sc_quorum', 0) or 0) or (n_max // 2 + 1)
        while len(votes) < n_max:
            counts = Counter(votes)
            top = counts.most_common(1)[0]
            if top[1] >= quorum:
                break
            vi = learner.answer_mcq(task, context=_ctx_for_pass(context, cfg, temp=float(getattr(cfg.dials, 'temp_sc', 0.7) or 0.7)))
            votes.append(vi.get("chosen_index"))
            cand_meta.append({"confidence": (vi.get("raw") or {}).get("confidence"), "citations": vi.get("citations")})
            if (citations is None) and vi.get("citations"):
                citations = vi.get("citations")
    else:
        for _ in range(n_max):
            v = learner.answer_mcq(task, context=_ctx_for_pass(context, cfg, temp=float(getattr(cfg.dials, 'temp_sc', 0.7) or 0.7)))
            votes.append(v.get("chosen_index"))
            cand_meta.append({"confidence": (v.get("raw") or {}).get("confidence"), "citations": v.get("citations")})
            if v.get("citations"):
                citations = v.get("citations")
    # Majority vote + evidence tie-break (ignore None if any concrete votes exist)
    final_choice = None
    if votes:
        concrete = [v for v in votes if v is not None]
        if concrete:
            counts = Counter(concrete)
        else:
            counts = Counter(votes)
        final_choice = counts.most_common(1)[0][0]
    chosen_idx = final_choice
    if bool(getattr(cfg.dials, 'use_fact_cards', False)) and bool(getattr(cfg.dials, 'require_citations', False)) and citations:
        sig = compute_evidence_signals(task, notes_buf, skill_id, citations, (context or {}).get("option_card_ids") or {})
        w_scores = sig.get("witness_scores") or {}
        if w_scores and (chosen_idx is not None):
            witness_idx = sig.get("witness_idx")
            chosen_score = w_scores.get(chosen_idx, -1)
            top_score = max(w_scores.values()) if w_scores else -1
            if (witness_idx is not None) and (witness_idx != chosen_idx) and (top_score - chosen_score >= 1):
                chosen_idx = witness_idx
    # Build telemetry and simple card metrics for observability
    try:
        evi_telemetry = {
            "min_quotes_per_option": gate_health.get("min_quotes_per_option"),
            "coverage_chosen": gate_health.get("coverage_chosen"),
            "witness_overlap_ratio": gate_health.get("witness_overlap_ratio"),
            "num_distinct_sources": gate_health.get("sources_chosen"),
        }
        per_opt = {}
        for ltr in letters:
            opt_quotes = [q for q in quotes if getattr(q, 'option', None) == ltr]
            per_opt[ltr] = {
                "count": len(opt_quotes),
                "distinct_sources": len({getattr(q, 'source_id', None) for q in opt_quotes}),
            }
        card_metrics = {"per_option": per_opt}
    except Exception:
        evi_telemetry = None
        card_metrics = None
    return votes, cand_meta, citations, chosen_idx, evi_telemetry, card_metrics


def _ctx_for_pass(base_ctx: Dict[str, Any], cfg, *, temp: float) -> Dict[str, Any]:
    ctx2 = dict(base_ctx or {})
    dec = {"temperature": float(temp), "top_p": float(getattr(cfg.dials, "top_p", 1.0) or 1.0)}
    if getattr(cfg.dials, "min_p", None) is not None:
        dec["min_p"] = float(cfg.dials.min_p)
    ctx2["decode"] = dec
    rmode = getattr(cfg.dials, "reasoning", "none")
    if (rmode or "none") != "none":
        ctx2["reasoning"] = rmode
    return ctx2
