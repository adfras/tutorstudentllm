from __future__ import annotations

"""
Fact-Card Manager (scaffold)

Encapsulates common LEARN/USE preparation steps:
 - normalize ids, tags and quotes (≤15 tokens)
 - enforce verbatim substring by scope (option/context) using validation helpers
 - synthesize per-option PRO cards (offset-first slice)
 - reserve within budget (required → top context → rest)
 - optional per-option deduplication

This mirrors the existing orchestrator logic but centralizes the operations so
they can be reused and tested in isolation. Meant to be adopted gradually
behind a feature flag to avoid behavior drift.
"""

from typing import Any, Dict, List, Tuple

from .utils.cards import clamp_card_quote, ensure_skill_tag
from .factcards import dedup_per_option
from .validation import first_n_tokens_span, validate_option_quote


class FactCardManager:
    def __init__(self) -> None:
        return

    def normalize_and_budget(
        self,
        *,
        merged: Dict[str, Dict[str, Any]],
        task,
        skill_id: str | None,
        retrieved_snippets: List[str] | None,
        budget: int,
        dedup_sim: float = 0.88,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return (new_cards, required_cards) within budget after normalization.

        merged: key→card map (as built in orchestrator)
        """
        src_snips = retrieved_snippets or []
        opt_texts = list(getattr(task, "options", []) or [])
        # Step 0: id uniqueness, tag ensure, quote clamp and scope enforcement
        seen_ids: set[str] = set()
        idx_counter = 1
        for k in list(merged.keys()):
            c = merged[k]
            cid = str(c.get("id") or "").strip()
            if not cid or cid in seen_ids:
                cid = f"f{idx_counter}"
                idx_counter += 1
                c["id"] = cid
            seen_ids.add(cid)
            # tag ensure
            c = ensure_skill_tag(c, skill_id)
            merged[k] = c
            # clamp quote length
            c = clamp_card_quote(c, max_tokens=15)
            merged[k] = c
            w = c.get("where") or {}
            scope = w.get("scope")
            quote = c.get("quote") or ""
            if scope == "option":
                try:
                    oi = int(w.get("option_index"))
                except Exception:
                    oi = 0
                opt = opt_texts[oi] if 0 <= oi < len(opt_texts) else (opt_texts[0] if opt_texts else "")
                # validate and adjust quote vs option
                vr = validate_option_quote(opt, c)
                if not vr.get("ok"):
                    # fallback: slice first n tokens from option
                    s, e, q0 = first_n_tokens_span(opt, 15)
                    c.setdefault("where", {})["start"] = s
                    c.setdefault("where", {})["end"] = e
                    c["quote"] = q0
                    merged[k] = c
            elif scope == "context":
                if not quote:
                    src = src_snips[0] if src_snips else ""
                    s, e, q0 = first_n_tokens_span(src, 15)
                    c.setdefault("where", {})["start"] = s
                    c.setdefault("where", {})["end"] = e
                    c["quote"] = q0
                    merged[k] = c
            else:
                # unknown scope
                if src_snips:
                    c.setdefault("where", {})["scope"] = "context"
                    s, e, q0 = first_n_tokens_span(src_snips[0], 15)
                    c.setdefault("where", {})["start"] = s
                    c.setdefault("where", {})["end"] = e
                    c["quote"] = q0
                else:
                    c.setdefault("where", {})["scope"] = "option"
                    c.setdefault("where", {})["option_index"] = 0
                    s0, e0, q0 = first_n_tokens_span(opt_texts[0] if opt_texts else "", 15)
                    c.setdefault("where", {})["start"] = s0
                    c.setdefault("where", {})["end"] = e0
                    c["quote"] = q0
                merged[k] = c
        # Step 1: Build required per-option PRO cards (if missing)
        required_cards: List[Dict[str, Any]] = []
        for oi, opt in enumerate(getattr(task, "options", []) or []):
            exists = False
            for c in merged.values():
                w = c.get("where") or {}
                if (w.get("scope") == "option") and (w.get("option_index") == oi):
                    # after clamp, this should be ≤15 tokens
                    exists = True
                    break
            if not exists:
                # slice directly from option
                s, e, quote = first_n_tokens_span(opt, 15)
                where = {"scope": "option", "option_index": oi, "start": s, "end": e, "source_id": f"option:{oi}"}
                rc = {
                    "id": f"opt{oi+1}",
                    "claim": quote,
                    "quote": quote,
                    "where": where,
                    "tags": ([] if not skill_id else [skill_id]),
                    "hypothesis": f"Option {oi} quoted",
                    "polarity": "pro",
                }
                mk = f"{quote.strip().lower()}|option:{oi}"
                if mk not in merged:
                    merged[mk] = rc
                required_cards.append(rc)
        # Step 2: Reserve budget (required → top-2 context → others)
        others_all = [c for _, c in merged.items() if c.get("id") not in {rc.get("id") for rc in required_cards}]
        context_cards = [c for c in others_all if (c.get("where") or {}).get("scope") == "context"]
        non_context = [c for c in others_all if (c.get("where") or {}).get("scope") != "context"]
        new_cards: List[Dict[str, Any]] = []
        # 1) Required
        for rc in required_cards:
            if len(new_cards) >= int(budget):
                break
            new_cards.append(rc)
        # 2) Up to 2 context
        for c in context_cards[:2]:
            if len(new_cards) >= int(budget):
                break
            if c.get("id") in {x.get("id") for x in new_cards}:
                continue
            new_cards.append(c)
        # 3) Fill the rest
        for c in non_context:
            if len(new_cards) >= int(budget):
                break
            if c.get("id") in {x.get("id") for x in new_cards}:
                continue
            new_cards.append(c)
        # Step 3: Per-option deduplication
        try:
            new_cards = dedup_per_option(new_cards, sim_threshold=float(dedup_sim))
        except Exception:
            pass
        return new_cards, required_cards

    def prepare_cards(
        self,
        *,
        merged: Dict[str, Dict[str, Any]],
        task,
        skill_id: str | None,
        retrieved_snippets: List[str] | None,
        budget: int,
        dedup_sim: float,
        q_min: int,
        max_boosts: int,
        extract_fn,
        learn_source: str,
        context: Dict[str, Any] | None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Full LEARN preparation with optional targeted boosts to satisfy q_min.

        extract_fn: callable(task, source_text, context) -> {cards:[...]}
        """
        def _counts(cards: List[Dict[str, Any]]) -> Dict[int, int]:
            cc: Dict[int, int] = {}
            for c in cards:
                w = c.get("where") or {}
                if (w.get("scope") == "option") and isinstance(w.get("option_index"), int):
                    oi = int(w.get("option_index"))
                    cc[oi] = cc.get(oi, 0) + 1
            return cc

        # Initial normalization/budgeting
        new_cards, required_cards = self.normalize_and_budget(
            merged=merged,
            task=task,
            skill_id=skill_id,
            retrieved_snippets=retrieved_snippets,
            budget=budget,
            dedup_sim=dedup_sim,
        )
        cnts = _counts(new_cards)
        boosts = 0
        # Loop until q_min satisfied or boosts exhausted
        while (min([cnts.get(i, 0) for i in range(len(getattr(task, 'options', []) or []))] or [0]) < int(q_min)) and (boosts < int(max_boosts)):
            try:
                js = extract_fn(task=task, source_text=learn_source, context=context)
                if isinstance(js, dict) and isinstance(js.get("cards"), list):
                    for c in js["cards"]:
                        # reuse orchestrator's merge key logic (claim/quote + scope + option_index)
                        w = c.get("where") or {}
                        scope = (w.get("scope") or "").strip().lower()
                        oi = w.get("option_index")
                        oi_s = str(int(oi)) if isinstance(oi, int) else "NA"
                        base = (str(c.get("claim") or "").strip().lower()) or (str(c.get("quote") or "").strip().lower())
                        mk = f"{base}|{scope}:{oi_s}"
                        if mk and mk not in merged:
                            merged[mk] = {
                                "id": c.get("id") or f"fc{len(merged)+1}",
                                "claim": c.get("claim") or "",
                                "quote": c.get("quote") or "",
                                "where": c.get("where") or {},
                                "tags": c.get("tags") or [],
                                "hypothesis": c.get("hypothesis") or "",
                                "polarity": c.get("polarity") or "pro",
                            }
            except Exception:
                pass
            new_cards, required_cards = self.normalize_and_budget(
                merged=merged,
                task=task,
                skill_id=skill_id,
                retrieved_snippets=retrieved_snippets,
                budget=budget,
                dedup_sim=dedup_sim,
            )
            cnts = _counts(new_cards)
            boosts += 1
        return new_cards, required_cards
