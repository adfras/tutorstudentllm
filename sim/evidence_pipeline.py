from __future__ import annotations

"""
Unified evidence pipeline (scaffold).

Wraps pre-/post- gating and evidence signal computation behind a simple
interface so the orchestrator can stay lean. This module intentionally
delegates to existing helpers (evidence_gates, evidence) to preserve current
behavior while providing one surface for future consolidation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .evidence_schema import Quote, adapt_cards_to_quotes
from .evidence_gates import pass_pre_gates, pass_post_gates, evidence_health
from .evidence import compute_evidence_signals
from .utils.cards import quote_ok


@dataclass
class EvidenceReport:
    pre_pass: Optional[bool] = None
    post_pass: Optional[bool] = None
    abstain_reason: Optional[str] = None
    coverage: Optional[float] = None
    witness_idx: Optional[int] = None
    witness_pass: Optional[bool] = None
    has_option_quote_chosen: Optional[bool] = None
    required_ok_chosen: Optional[bool] = None
    cited_ids: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def _normalize_citations(citations: Any) -> List[str]:
    out: List[str] = []
    for x in (citations or []):
        try:
            if isinstance(x, dict) and x.get("id"):
                out.append(str(x.get("id")))
            elif isinstance(x, (str, int)):
                out.append(str(x))
        except Exception:
            continue
    return out


def pre_learn_checks(cards: List[Dict[str, Any]], option_letters: List[str], *, q_min: int = 1, min_sources: int = 1) -> EvidenceReport:
    quotes: List[Quote] = adapt_cards_to_quotes(cards)
    ok = pass_pre_gates(quotes, option_letters, q_min=q_min, min_sources=min_sources)
    return EvidenceReport(pre_pass=ok, metrics=evidence_health(quotes, option_letters))


def post_use_checks(
    *,
    task,
    notes_buf: str,
    skill_id: str,
    citations: List[Any] | None,
    option_card_ids: Dict[str, Any] | None,
    chosen_index: Optional[int],
    coverage_tau: float = 0.6,
    min_sources: int = 2,
    q_min: int = 2,
) -> EvidenceReport:
    """Compute evidence signals and post-answer gating.

    Abstention reasons follow the current simulator policy:
      - no_citations
      - no_option_quote_for_choice
      - coverage_below_tau
      - required_card_not_cited
    """
    import json as _json

    # Normalize citations and parse cards from notes
    cited_ids = _normalize_citations(citations)
    try:
        mem = _json.loads(notes_buf) if notes_buf else {"cards": []}
        cards = mem.get("cards") or []
    except Exception:
        cards = []

    # Filter cited cards by skill tag and quote length (≤15 tokens)
    filtered = [
        c for c in cards
        if ((not cited_ids) or (str(c.get("id")) in cited_ids))
        and (skill_id in (c.get("tags") or []))
        and quote_ok(c.get("quote") or "")
    ]

    # Compute aggregate signals using existing helper
    sig = compute_evidence_signals(task, notes_buf, skill_id, citations or [], option_card_ids or {})

    # Coverage against gold option; witness index per options
    coverage = float(sig.get("coverage") or 0.0)
    witness_idx = sig.get("witness_idx")
    has_opt = sig.get("has_option_quote") or {}
    req_ok = sig.get("required_ok") or {}
    chosen_ok = None if chosen_index is None else bool(has_opt.get(int(chosen_index), False))
    chosen_req = None if chosen_index is None else bool(req_ok.get(int(chosen_index), False))

    # Determine abstention reason (hard-evidence only)
    abstain_reason = None
    if not cited_ids:
        abstain_reason = "no_citations"
    elif (chosen_index is not None) and not chosen_ok:
        abstain_reason = "no_option_quote_for_choice"
    elif coverage < float(coverage_tau):
        abstain_reason = "coverage_below_tau"
    elif (chosen_index is not None) and (not chosen_req):
        abstain_reason = "required_card_not_cited"

    # witness pass: does evidence align with the gold option?
    witness_pass = (witness_idx == getattr(task, "correct_index", None))

    # Build metrics summary
    metrics = {
        "num_cited_cards": len(filtered),
        "coverage": coverage,
        "witness_idx": witness_idx,
        "witness_pass": bool(witness_pass),
    }

    return EvidenceReport(
        post_pass=(abstain_reason is None),
        abstain_reason=abstain_reason,
        coverage=coverage,
        witness_idx=witness_idx,
        witness_pass=bool(witness_pass),
        has_option_quote_chosen=chosen_ok,
        required_ok_chosen=chosen_req,
        cited_ids=cited_ids,
        metrics=metrics,
    )

