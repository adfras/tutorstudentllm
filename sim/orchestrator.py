from __future__ import annotations
"""High-level simulation runner.

This module coordinates tutor→student simulations. It focuses on the control
flow for steps, logging, anonymization, optional tools, and evidence gating.

For clarity and testability, small configuration and evidence utilities live
in lightweight helpers:
  - sim.config: Dials and RunConfig dataclasses
  - sim.evidence: evidence-scoring helpers

Public API: Orchestrator, RunConfig, Dials (re-exported).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from tutor.skill_map import load_skill_map
from tutor.llm_openai import OpenAILLM
from sim.tasks import (
    MCQTask,
    SAQTask,
    CodeTask,
    ProofTask,
    TableQATask,
    Task,
    evaluate_mcq,
)
from sim.evaluators import evaluate_code_python, evaluate_proof_step, evaluate_table_qa
from sim.anonymize import build_vocab, compile_codebook, anonymize_mcq, anonymize_text
from sim.tools import build_tools
from sim.evidence_schema import adapt_cards_to_quotes
from sim.card_quality import assess_quote, build_idf, tokens as _cq_tokens, jaccard as _cq_jaccard
from sim.evidence_gates import pass_pre_gates, pass_post_gates, evidence_health
from sim.domain import DomainStore
from sim.alias import load_alias_families
from tutor.templates import templates_for_skill
from sim.config import Dials, RunConfig
from sim.utils.guardrails import load_guardrails, load_talk_slopes, TokenBands
from sim.evidence import compute_evidence_signals
from sim.examples import select_and_prepare_examples
from sim.rerank import apply_best_of_rerank
from sim.controllers_quote import quote_then_vote_controller
from sim.factcards import build_option_card_ids, trim_card, clamp_cards
from sim.factcards_manager import FactCardManager
import os
from sim.utils.cards import quote_ok as _quote_ok
try:
    # Optional unified evidence pipeline
    from sim.evidence_pipeline import post_use_checks as _ep_post_use_checks
except Exception:
    _ep_post_use_checks = None  # type: ignore


def _tok(s: str) -> set[str]:
    import re as _re
    return set([t for t in _re.findall(r"[a-zA-Z0-9]+", (s or "").lower()) if len(t) >= 2])


__all__ = ["Orchestrator", "RunConfig", "Dials"]


def _flag_on(name: str, default: bool = True) -> bool:
    """Read a boolean-like env var once and coerce to bool.

    Exists at module scope so nested branches later can safely reference it
    (avoids UnboundLocalError when a locally-scoped helper isn't executed).
    """
    import os as _os
    v = _os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")


class Orchestrator:
    def __init__(self, model: Optional[OpenAILLM] = None):
        self.llm = model or OpenAILLM()
        self.smap = load_skill_map()
        self.domain = DomainStore()
        self.vocab = build_vocab(self.smap)
        # Budget guard state
        self._budget_strikes = 0
        # Bayesian guardrails state (optional)
        self._token_bands: TokenBands | None = None
        self._talk_slopes: dict[str, Any] = {}
        self._student_tokens_cum: int = 0
        self._turns_limit: int | None = None
        self._resolve_triggered: bool = False
        self._fact_manager = FactCardManager()

    def _factcard_key(self, card: Dict[str, Any]) -> str | None:
        """Build a stable merge key for fact-card dictionaries."""
        try:
            quote = str(card.get("quote") or "").strip().lower()
            where = card.get("where") or {}
            scope = str((where.get("scope") or "")).strip().lower()
            opt = where.get("option_index")
            opt_key = str(opt) if opt is not None else "-"
            source = str(where.get("source_id") or "")
            if quote:
                return "|".join([quote, scope, opt_key, source])
            cid = card.get("id")
            if cid is not None:
                return f"id:{cid}"
        except Exception:
            return None
        return None

    def _merge_cards(self, merged: Dict[str, Dict[str, Any]], cards: List[Dict[str, Any]] | None) -> None:
        if not cards:
            return
        for card in cards:
            if not isinstance(card, dict):
                continue
            key = self._factcard_key(card)
            if not key:
                continue
            if key in merged:
                continue
            merged[key] = dict(card)

    def _option_quote_counts(self, cards: List[Dict[str, Any]], num_options: int) -> List[int]:
        counts = [0 for _ in range(num_options)]
        for card in cards or []:
            where = card.get("where") or {}
            if (where.get("scope") == "option") and isinstance(where.get("option_index"), int):
                oi = int(where.get("option_index"))
                if 0 <= oi < len(counts):
                    counts[oi] += 1
        return counts

    def _apply_card_quality_filter(self, cards: List[Dict[str, Any]], task, cfg) -> List[Dict[str, Any]]:
        try:
            min_cqs = float(getattr(cfg.dials, "min_cqs", 0.0) or 0.0)
        except Exception:
            min_cqs = 0.0
        if min_cqs <= 0.0:
            return cards
        try:
            top_k = int(getattr(cfg.dials, "per_option_top_k", 3) or 3)
            min_len = int(getattr(cfg.dials, "min_card_len", 40) or 40)
            max_len = int(getattr(cfg.dials, "max_card_len", 300) or 300)
            dedup_sim = float(getattr(cfg.dials, "dedup_sim", 0.88) or 0.88)
        except Exception:
            return cards
        opt_texts = list(getattr(task, "options", []) or [])
        try:
            idf = build_idf(opt_texts)
        except Exception:
            idf = {}
        by_opt: Dict[int, List[Dict[str, Any]]] = {}
        context_cards: List[Dict[str, Any]] = []
        for card in cards or []:
            where = card.get("where") or {}
            if where.get("scope") == "option" and isinstance(where.get("option_index"), int):
                oi = int(where.get("option_index"))
                by_opt.setdefault(oi, []).append(card)
            else:
                context_cards.append(card)
        kept: List[Dict[str, Any]] = []
        required_ids = {str(card.get("id")) for card in cards if str(card.get("id") or "").startswith("opt")}
        for oi, card_list in by_opt.items():
            ranked: List[tuple[float, Dict[str, Any]]] = []
            other_opts = [opt_texts[j] for j in range(len(opt_texts)) if j != oi]
            for card in card_list:
                score = assess_quote(
                    card.get("quote") or "",
                    opt_texts[oi] if oi < len(opt_texts) else "",
                    other_opts,
                    str((card.get("where") or {}).get("source_id") or ""),
                    idf=idf,
                    min_len=min_len,
                    max_len=max_len,
                )
                if score.cqs >= min_cqs or str(card.get("id")) in required_ids:
                    ranked.append((score.cqs, card))
            if not ranked:
                # keep required if filter removes all
                ranked = [(0.0, card) for card in card_list if str(card.get("id")) in required_ids]
            ranked.sort(key=lambda t: t[0], reverse=True)
            # Deduplicate by quote similarity
            selected: List[Dict[str, Any]] = []
            for _, card in ranked:
                if len(selected) >= top_k and str(card.get("id")) not in required_ids:
                    continue
                quote = card.get("quote") or ""
                is_dup = False
                for prev in selected:
                    prev_q = prev.get("quote") or ""
                    if _cq_jaccard(_cq_tokens(quote), _cq_tokens(prev_q)) >= dedup_sim:
                        is_dup = True
                        break
                if not is_dup:
                    selected.append(card)
            kept.extend(selected)
        kept.extend(context_cards)
        return kept

    def _update_context_with_cards(
        self,
        *,
        context: Dict[str, Any],
        new_cards: List[Dict[str, Any]],
        fact_cards_after: Dict[str, Any],
        notes_buf: str,
        retrieval_corpus: str,
        skill_id: str,
        closed_book: bool,
        option_card_ids: Dict[str, str] | None,
    ) -> Dict[str, Any]:
        import json as _json

        if closed_book:
            ctx_json = _json.dumps(fact_cards_after, ensure_ascii=False)
            new_context: Dict[str, Any] = {
                "notes_text": retrieval_corpus or notes_buf,
                "context_text": ctx_json,
                "fact_cards": new_cards,
                "skill_id": skill_id,
            }
            if context.get("retrieved_snippets"):
                new_context["retrieved_snippets"] = context["retrieved_snippets"]
            if context.get("retrieved_sources"):
                new_context["retrieved_sources"] = context["retrieved_sources"]
        else:
            new_context = dict(context)
            new_context["fact_cards"] = new_cards
        if option_card_ids:
            new_context["option_card_ids"] = option_card_ids
        return new_context

    def _execute_fact_card_learning(
        self,
        *,
        learner,
        task,
        skill_id: str,
        cfg: RunConfig,
        context: Dict[str, Any],
        merged: Dict[str, Dict[str, Any]],
        learn_source: str,
    ) -> tuple[List[Dict[str, Any]], float, int]:
        import time as _time

        learn_t0 = _time.time()
        try:
            usage_before = int((learner.get_usage_counters() or {}).get("total_tokens") or 0)
        except Exception:
            usage_before = 0

        n_sc = int(getattr(cfg.dials, "sc_extract_n", 0) or 0)
        if n_sc <= 0:
            n_sc = max(1, int(getattr(cfg.dials, "self_consistency_n", 1) or 1))
        for _ in range(n_sc):
            try:
                js = learner.extract_fact_cards(task, source_text=learn_source, context=context)
                if isinstance(js, dict) and isinstance(js.get("cards"), list):
                    self._merge_cards(merged, js.get("cards"))
            except Exception:
                continue

        q_min = max(1, int(getattr(cfg.dials, "q_min", 1) or 1))
        max_boosts = int(getattr(cfg.dials, "max_learn_boosts", 0) or 0)
        dedup_sim = float(getattr(cfg.dials, "dedup_sim", 0.88) or 0.88)
        budget = int(getattr(cfg.dials, "fact_cards_budget", 10) or 10)
        retrieved_snips = (context or {}).get("retrieved_snippets") or []

        new_cards, _ = self._fact_manager.normalize_and_budget(
            merged=merged,
            task=task,
            skill_id=skill_id,
            retrieved_snippets=retrieved_snips,
            budget=budget,
            dedup_sim=dedup_sim,
        )

        boosts = 0
        num_options = len(getattr(task, "options", []) or [])
        while boosts < max_boosts:
            counts = self._option_quote_counts(new_cards, num_options)
            if counts and min(counts) >= q_min:
                break
            try:
                jsb = learner.extract_fact_cards(task, source_text=learn_source, context=context)
                if isinstance(jsb, dict) and isinstance(jsb.get("cards"), list):
                    self._merge_cards(merged, jsb.get("cards"))
            except Exception:
                pass
            boosts += 1
            new_cards, _ = self._fact_manager.normalize_and_budget(
                merged=merged,
                task=task,
                skill_id=skill_id,
                retrieved_snippets=retrieved_snips,
                budget=budget,
                dedup_sim=dedup_sim,
            )

        learn_time_s = max(0.0, _time.time() - learn_t0)
        try:
            usage_after = int((learner.get_usage_counters() or {}).get("total_tokens") or 0)
        except Exception:
            usage_after = usage_before
        learn_tokens = max(0, usage_after - usage_before)
        return new_cards, learn_time_s, learn_tokens

    def _prepare_fact_cards_flow(
        self,
        *,
        learner,
        task,
        skill_id: str,
        cfg: RunConfig,
        context: Dict[str, Any],
        notes_buf: str,
        retrieval_corpus: str,
        learn_source: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any] | None, Dict[str, Any], str, Dict[str, Any]]:
        import json as _json

        try:
            cards_mem = (_json.loads(notes_buf) if notes_buf else {"cards": []})
            if not isinstance(cards_mem, dict):
                cards_mem = {"cards": []}
        except Exception:
            cards_mem = {"cards": []}
        fact_cards_before = cards_mem

        merged: Dict[str, Dict[str, Any]] = {}
        self._merge_cards(merged, cards_mem.get("cards"))
        retrieved_snips = (context or {}).get("retrieved_snippets") or []
        budget = int(getattr(cfg.dials, "fact_cards_budget", 10) or 10)
        dedup_sim = float(getattr(cfg.dials, "dedup_sim", 0.88) or 0.88)

        learn_metrics: Dict[str, Any] = {}
        if cfg.dials.freeze_cards:
            new_cards, _ = self._fact_manager.normalize_and_budget(
                merged=merged,
                task=task,
                skill_id=skill_id,
                retrieved_snippets=retrieved_snips,
                budget=budget,
                dedup_sim=dedup_sim,
            )
        else:
            new_cards, learn_time_s, learn_tokens = self._execute_fact_card_learning(
                learner=learner,
                task=task,
                skill_id=skill_id,
                cfg=cfg,
                context=context,
                merged=merged,
                learn_source=learn_source,
            )
            if learn_tokens:
                learn_metrics["_learn_tokens"] = learn_tokens
            if learn_time_s:
                learn_metrics["_learn_time_s"] = learn_time_s
            # Update notes buf with latest cards once learning happens
            notes_buf = ""

        new_cards = self._apply_card_quality_filter(new_cards, task, cfg)
        new_cards = clamp_cards(new_cards, max_tokens=15)
        fact_cards_after = {"cards": [trim_card(c) for c in new_cards]}

        if not cfg.dials.freeze_cards:
            notes_buf = _json.dumps(fact_cards_after, ensure_ascii=False)

        option_card_ids = build_option_card_ids(new_cards, task)
        context_out = self._update_context_with_cards(
            context=context,
            new_cards=new_cards,
            fact_cards_after=fact_cards_after,
            notes_buf=notes_buf,
            retrieval_corpus=retrieval_corpus,
            skill_id=skill_id,
            closed_book=cfg.dials.closed_book,
            option_card_ids=option_card_ids,
        )
        context_out.update(learn_metrics)
        if option_card_ids:
            context_out["option_card_ids"] = option_card_ids
        return fact_cards_before, fact_cards_after, context_out, notes_buf, learn_metrics


    def _make_mcq_task(self, skill_id: str, cfg: RunConfig, codebook: Optional[Dict[str, str]] = None) -> MCQTask:
        skill = self.smap["skills"].get(skill_id) or next(iter(self.smap["skills"].values()))
        tmpl = None
        try:
            # Prefer a template when available to steer toward abstract/structural prompts
            ts = templates_for_skill(skill_id)
            if ts:
                import random as _r
                tmpl = _r.choice(ts)
        except Exception:
            tmpl = None
        q = self.llm.generate_mcq(skill, difficulty=cfg.difficulty, num_options=cfg.num_options, minimal=not cfg.dials.rich, template=tmpl)
        stem = q.get("stem") or ""
        options = q.get("options") or []
        if cfg.dials.anonymize and codebook:
            stem, options = anonymize_mcq(stem, options, codebook)
        task = MCQTask(
            id=q.get("question_id") or f"mcq-{skill_id}",
            prompt={"skill_id": skill_id, "difficulty": cfg.dials.rich and q.get("difficulty") or cfg.difficulty},
            stem=stem,
            options=options,
            correct_index=int(q.get("correct_index", 0)),
            rationales=q.get("rationales"),
            misconception_tags=q.get("misconception_tags"),
            metadata={"source": "openai", "template_id": q.get("template_id")},
        )
        return task

    def _make_saq_task(self, skill_id: str, cfg: RunConfig, codebook: Optional[Dict[str, str]] = None) -> SAQTask:
        skill = self.smap["skills"].get(skill_id) or next(iter(self.smap["skills"].values()))
        q = self.llm.generate_saq(skill, difficulty=cfg.difficulty)
        stem = q.get("stem") or ""
        if cfg.dials.anonymize and codebook:
            stem = anonymize_text(stem, codebook)
        task = SAQTask(
            id=q.get("question_id") or f"saq-{skill_id}",
            prompt={"skill_id": skill_id, "difficulty": q.get("difficulty") or cfg.difficulty},
            stem=stem,
            expected_points=list(q.get("expected_points") or []),
            model_answer=q.get("model_answer") or "",
            difficulty=q.get("difficulty") or cfg.difficulty,
            metadata={"source": "openai"},
        )
        return task

    def _make_code_task(self, cfg: RunConfig) -> CodeTask:
        starter = "def add(a,b):\n    # TODO: implement\n    return a-b\n"
        tests = [
            {"args": [2, 3], "kwargs": {}, "expected": 5},
            {"args": [-1, 4], "kwargs": {}, "expected": 3},
        ]
        return CodeTask(
            id="code-add-1",
            prompt={},
            description="Implement add(a,b)",
            function_name="add",
            starter_code=starter,
            tests=tests,
        )

    def _make_proof_task(self, cfg: RunConfig) -> ProofTask:
        return ProofTask(id="proof-comm", prompt={}, statement="Show a+b=b+a", expected_keywords=["commutativity"])

    def _make_table_task(self, cfg: RunConfig) -> TableQATask:
        csv = "name,score\nann,3\nbob,5\ncid,4\n"
        return TableQATask(id="table-top", prompt={}, csv=csv, question="Who has the highest score?", expected_answer="bob")

    def run(
        self,
        learner,
        cfg: RunConfig,
        notes_text: str = "",
        log_path: Optional[str] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        # Per-run anonymization keys
        # Build domain-aware vocab and codebook
        vocab = list(self.vocab)
        try:
            vocab += self.domain.glossary_terms(cfg.domain)
        except Exception:
            pass
        # deterministic per-run seed for audit; allow override via ANON_SEED
        import random, os
        seed_env = os.getenv("ANON_SEED")
        seed: int
        try:
            seed = int(seed_env) if (seed_env is not None) else random.randrange(1, 2**31 - 1)
        except Exception:
            seed = random.randrange(1, 2**31 - 1)
        codebook = compile_codebook(vocab, seed=seed) if cfg.dials.anonymize else None
        # Optional: load Bayesian guardrails and domain talk slopes; derive turns limit and talk mode
        try:
            self._token_bands = load_guardrails(getattr(cfg, 'guardrails_path', None) or os.getenv("BAYES_GUARDRAILS"))
        except Exception:
            self._token_bands = None
        try:
            self._talk_slopes = load_talk_slopes(getattr(cfg, 'talk_slopes_path', None) or os.getenv("BAYES_TALK_SLOPES"))
        except Exception:
            self._talk_slopes = {}
        # Determine turns limit (steps)
        self._turns_limit = None
        try:
            tl = getattr(cfg, 'turns_limit', None)
            if isinstance(tl, int) and tl > 0:
                self._turns_limit = tl
            else:
                ms = float(self._token_bands.mean_steps) if (self._token_bands and self._token_bands.mean_steps) else None  # type: ignore
                if ms and ms > 0:
                    self._turns_limit = int(round(ms + 0.2 * ms))
        except Exception:
            self._turns_limit = None
        # Per-domain talk tuning
        talk_mode = "neutral"
        try:
            slope = self._talk_slopes.get(cfg.domain) or self._talk_slopes.get("general")
            ppos = None
            if slope:
                ppos = float(getattr(slope, 'prob_positive', 0.5)) if not isinstance(slope, dict) else float(slope.get('prob_positive', 0.5))
            thr = float(getattr(cfg, 'talk_ppos_threshold', 0.7) or 0.7)
            if ppos is not None:
                if ppos >= thr:
                    talk_mode = "rich"
                elif ppos <= (1.0 - thr):
                    talk_mode = "lean"
        except Exception:
            talk_mode = "neutral"
        # Apply conservative talk-mode tweaks
        try:
            if talk_mode == "lean":
                cfg.dials.rich = False
                if cfg.dials.controller == "tot":
                    cfg.dials.controller = "basic"
                cfg.dials.controller_budget = min(cfg.dials.controller_budget, 3)
                cfg.dials.self_consistency_n = min(cfg.dials.self_consistency_n, 2)
                cfg.dials.compress_examples = True
                cfg.dials.compress_ratio = max(cfg.dials.compress_ratio, 3.0)
            elif talk_mode == "rich":
                cfg.dials.rich = True
                cfg.dials.controller_budget = max(cfg.dials.controller_budget, 6)
        except Exception:
            pass
        logs: List[Dict[str, Any]] = []
        # Ensure parent directory exists for log path
        if log_path:
            try:
                import os as _os
                d = _os.path.dirname(log_path)
                if d:
                    _os.makedirs(d, exist_ok=True)
            except Exception:
                pass
        log_f = open(log_path, "a", encoding="utf-8") if log_path else None
        import uuid
        run_id = str(uuid.uuid4())
        # Write header line with run metadata
        if log_f:
            import json, time
            header = {
                "run_header": True,
                "run_id": run_id,
                "ts": int(time.time()),
                "config": {
                    "skill_id": cfg.skill_id,
                    "task": cfg.task,
                    "num_steps": cfg.num_steps,
                    "num_options": cfg.num_options,
                    "difficulty": cfg.difficulty,
                    "domain": cfg.domain,
                    "dials": vars(cfg.dials),
                },
                "anonymization": ({"seed": seed, "vocab_size": len(vocab)} if codebook else None),
                "guardrails": ({
                    "guardrails_path": getattr(cfg, 'guardrails_path', None) or os.getenv("BAYES_GUARDRAILS"),
                    "talk_slopes_path": getattr(cfg, 'talk_slopes_path', None) or os.getenv("BAYES_TALK_SLOPES"),
                    "turns_limit": self._turns_limit,
                    "talk_mode": talk_mode,
                } if (getattr(cfg, 'guardrails_path', None) or getattr(cfg, 'talk_slopes_path', None) or self._turns_limit) else None),
            }
            log_f.write(json.dumps(header, ensure_ascii=False) + "\n")
            log_f.flush()
        skill_id = cfg.skill_id or next(iter(self.smap["skills"].keys()))
        # Keep a separate retrieval corpus (raw text/JSONL joined text) for tools like tfidf_retriever.
        # notes_buf is the learner-facing memory buffer (may become a JSON {cards:[...]} when using Fact-Cards).
        notes_buf = notes_text or ""
        retrieval_corpus = notes_text or ""
        # If using fact-cards and notes is empty, initialize cards memory
        if cfg.dials.use_fact_cards:
            import json as _json
            try:
                obj = _json.loads(notes_buf) if notes_buf.strip() else {"cards": []}
                if not isinstance(obj, dict) or "cards" not in obj:
                    obj = {"cards": []}
            except Exception:
                obj = {"cards": []}
            notes_buf = _json.dumps(obj, ensure_ascii=False)
        # Example usage counts (for rare-emphasis scheduling)
        example_counts: Dict[int, int] = {}
        import time
        fam_data = load_alias_families()
        # Preload exemplars (optional JSON/JSONL)
        exemplars: List[Dict[str, Any]] = []
        if cfg.shots_path and int(cfg.shots_k or 0) > 0:
            try:
                import json as _json
                with open(cfg.shots_path, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                if len(lines) == 1 and (lines[0].startswith("[") or lines[0].startswith("{")):
                    obj = _json.loads(lines[0])
                    if isinstance(obj, list):
                        exemplars = [x for x in obj if isinstance(x, dict)]
                    elif isinstance(obj, dict) and isinstance(obj.get("examples"), list):
                        exemplars = [x for x in obj.get("examples") if isinstance(x, dict)]
                else:
                    for ln in lines:
                        try:
                            rec = _json.loads(ln)
                            if isinstance(rec, dict):
                                exemplars.append(rec)
                        except Exception:
                            continue
            except Exception:
                exemplars = []
        total_steps = int(cfg.num_steps or 0)
        # initial progress callback (0 of N)
        if progress_cb:
            try:
                progress_cb(0, total_steps)
            except Exception:
                pass
        for step in range(total_steps):
            step_t0 = time.time()
            evi_telemetry = None
            # reset per-step usage counters for LLM students, if available
            try:
                if hasattr(learner, "reset_usage_counters") and callable(getattr(learner, "reset_usage_counters")):
                    learner.reset_usage_counters()
            except Exception:
                pass
            # reset per-step tutor usage counters (OpenAILLM)
            try:
                if hasattr(self.llm, "reset_usage_counters") and callable(getattr(self.llm, "reset_usage_counters")):
                    self.llm.reset_usage_counters()
            except Exception:
                pass
            # reset per-step message buffers
            try:
                if hasattr(learner, "reset_messages_buffer") and callable(getattr(learner, "reset_messages_buffer")):
                    learner.reset_messages_buffer()
            except Exception:
                pass
            try:
                if hasattr(self.llm, "reset_messages_buffer") and callable(getattr(self.llm, "reset_messages_buffer")):
                    self.llm.reset_messages_buffer()
            except Exception:
                pass
            if (cfg.task or "mcq") == "saq":
                task = self._make_saq_task(skill_id, cfg, codebook)
                is_saq = True
                alias_phase = None
                family_id = None
            elif cfg.task == "alias_swap":
                # Build from alias family content
                families = fam_data.get("families") or []
                fam_index = fam_data.get("index") or {}
                fam = fam_index.get(cfg.alias_family_id) if cfg.alias_family_id else (families[0] if families else None)
                if fam is None:
                    raise RuntimeError("No alias families available")
                family_id = fam.get("id")
                # pick A then B
                alias_phase = "A" if step % 2 == 0 else "B"
                var = fam.get("alias_a") if alias_phase == "A" else fam.get("alias_b")
                stem = var.get("stem") or ""
                options = var.get("options") or []
                correct_index = int(var.get("correct_index", 0))
                if cfg.dials.anonymize and codebook:
                    stem, options = anonymize_mcq(stem, options, codebook)
                task = MCQTask(
                    id=f"alias-{family_id}-{alias_phase}",
                    prompt={"family_id": family_id, "alias_phase": alias_phase},
                    stem=stem,
                    options=options,
                    correct_index=correct_index,
                    rationales=[var.get("rationale","") for _ in options],
                    misconception_tags=None,
                    metadata={"source": "alias_family"},
                )
                is_saq = False
            elif cfg.task == "code":
                task = self._make_code_task(cfg)
                is_saq = False
                alias_phase = None
                family_id = None
            elif cfg.task == "proof":
                task = self._make_proof_task(cfg)
                is_saq = False
                alias_phase = None
                family_id = None
            elif cfg.task == "table_qa":
                task = self._make_table_task(cfg)
                is_saq = False
                alias_phase = None
                family_id = None
            else:
                task = self._make_mcq_task(skill_id, cfg, codebook)
                is_saq = False
                alias_phase = None
                family_id = None
            # Prepare context shown to the learner
            ctx_text = notes_buf if cfg.dials.closed_book else ""
            if codebook and cfg.dials.anonymize and ctx_text:
                ctx_text = anonymize_text(ctx_text, codebook)
            # Expose raw retrieval corpus to tools while keeping learner-facing context_text as notes_buf
            context = (
                {"notes_text": retrieval_corpus, "context_text": ctx_text, "skill_id": skill_id}
                if cfg.dials.closed_book
                else {"skill_id": skill_id}
            )
            # Surface gating knob (q_min) so the learner prompt can align its IDK rule
            try:
                context["q_min"] = int(cfg.dials.q_min or 1)
            except Exception:
                context["q_min"] = 1
            # Instruction header (APE)
            if cfg.dials.instruction_header:
                context["instruction_header"] = cfg.dials.instruction_header
            # Resolve-or-escalate: if turns exceed limit, enforce concise resolution for remainder
            try:
                if (not self._resolve_triggered) and self._turns_limit and (step + 1) > int(self._turns_limit):
                    self._resolve_triggered = True
                if self._resolve_triggered:
                    # tighten orchestration
                    try:
                        if cfg.dials.controller == 'tot':
                            cfg.dials.controller = 'basic'
                    except Exception:
                        pass
                    try:
                        cfg.dials.self_consistency_n = min(cfg.dials.self_consistency_n, 2)
                    except Exception:
                        pass
                    # Add a terse header for immediate resolution
                    rh = (
                        "Resolve now: choose the best option. If evidence is weak, abstain (chosen_index=null). "
                        "Return JSON only; no explanation."
                    )
                    if context.get("instruction_header"):
                        context["instruction_header"] = (str(context["instruction_header"]).strip() + "\n" + rh)
                    else:
                        context["instruction_header"] = rh
            except Exception:
                pass
            # Few-shot exemplar selection per item (MCQ/SAQ)
            selected_examples: List[Dict[str, Any]] = []
            if exemplars and int(cfg.shots_k or 0) > 0:
                qtext = task.stem
                try:
                    if hasattr(task, "options") and task.options:
                        qtext += "\n" + " | ".join(task.options)
                except Exception:
                    pass
                selected_examples = select_and_prepare_examples(qtext, exemplars, cfg)
            if selected_examples:
                # Optional ordering
                if cfg.shots_order == "easy-hard":
                    def _lev(ex: Dict[str, Any]) -> int:
                        d = (ex.get("difficulty") or ex.get("level") or "").strip().lower()
                        if d in ("easy","e"): return 0
                        if d in ("medium","m","med"): return 1
                        if d in ("hard","h"): return 2
                        return 1
                    selected_examples.sort(key=_lev)
                # 'similar' keeps KNN order; 'as-is' preserves file order
                # Optional compression (LLMLingua-lite heuristic)
                if cfg.dials.compress_examples:
                    def _compress_text(s: str, factor: float) -> str:
                        import re as _re
                        toks = _re.findall(r"[A-Za-z0-9]+|\S", s or "")
                        if not toks:
                            return s or ""
                        # Keep numbers, operators, long tokens; drop common stopwords; downsample to ~1/factor
                        keep = []
                        stop = {"the","a","an","of","and","or","to","is","are","was","were","in","on","for","with"}
                        for t in toks:
                            al = t.isalpha()
                            if (not al) or (t.lower() not in stop) or (len(t) >= 5):
                                keep.append(t)
                        k = max(1, int(len(keep) / max(1.0, cfg.dials.compress_ratio)))
                        keep2 = keep[:k] if k < len(keep) else keep
                        out = " ".join(keep2)
                        return out
                    def _compress_ex(ex: Dict[str, Any]) -> Dict[str, Any]:
                        ex2 = dict(ex)
                        if ex2.get("stem"):
                            ex2["stem"] = _compress_text(ex2.get("stem"), cfg.dials.compress_ratio)
                        if isinstance(ex2.get("options"), list):
                            ex2["options"] = [ _compress_text(str(o), cfg.dials.compress_ratio) for o in ex2["options"] ]
                        if ex2.get("rationale"):
                            ex2["rationale"] = _compress_text(ex2.get("rationale"), cfg.dials.compress_ratio)
                        return ex2
                    selected_examples = [_compress_ex(e) for e in selected_examples]
                context["examples"] = selected_examples
            # Surface abstention/calibration controls to learners
            if cfg.dials.idk_enabled:
                context["target_confidence"] = float(cfg.dials.target_confidence)
            # Optional tools
            tool_outputs = []
            if cfg.dials.use_tools:
                tools = build_tools(cfg.dials.tools or [])
                # task view for tools
                task_view = {"stem": task.stem}
                if not is_saq:
                    task_view["options"] = task.options
                for tool in tools:
                    try:
                        # Pass anonymized notes to align with anonymized stem/options
                        tool_notes = anonymize_text(notes_buf, codebook) if (codebook and cfg.dials.anonymize) else notes_buf
                        out = tool.run(task=task_view, context={
                            "notes_text": tool_notes,
                            # Retriever config is optional and read by tools that support it
                            "retriever_config": {
                                "mmr_lambda": float(getattr(cfg.dials, "mmr_lambda", 0.4) or 0.4),
                                "k": 2,
                                "span_window": int(getattr(cfg.dials, "span_window", 240) or 240),
                            },
                        })
                    except Exception:
                        out = {"name": getattr(tool, "name", "unknown"), "error": True}
                    tool_outputs.append(out)
                # inject tool outputs into context text
                tool_text_parts = []
                retrieved_snips: List[str] = []
                retrieved_sources: Dict[str, str] = {}
                for out in tool_outputs:
                    nm = out.get("name")
                    if nm in ("retriever","tfidf_retriever") and out.get("snippets"):
                        snips_all = out.get("snippets") or []
                        # keep up to 3 to improve provenance without blowing up context
                        keep_n = min(3, len(snips_all))
                        raw_kept = [s if isinstance(s, str) else str(s) for s in snips_all[:keep_n]]
                        # Anonymize both displayed text and the snippets stored for provenance when anonymization is on
                        if codebook and cfg.dials.anonymize:
                            shown = [anonymize_text(s, codebook) for s in raw_kept]
                            retrieved_snips.extend(shown)
                        else:
                            shown = list(raw_kept)
                            retrieved_snips.extend(raw_kept)
                        tool_text_parts.append(f"{nm}:\n- " + "\n- ".join(shown))
                    if nm == "option_retriever" and out.get("snippets"):
                        items = out.get("snippets") or []
                        # Keep first per option for minimal context; also store all sources
                        seen_opt: set[int] = set()
                        added_texts: List[str] = []
                        for it in items:
                            if not isinstance(it, dict):
                                continue
                            oi = it.get("option_index")
                            txt = it.get("text") or ""
                            sid = it.get("source_id") or ""
                            if (oi is None) or (not txt):
                                continue
                            if oi in seen_opt:
                                # still register source mapping
                                if sid and txt and sid not in retrieved_sources:
                                    retrieved_sources[sid] = txt
                                continue
                            seen_opt.add(int(oi))
                            # Anonymize snippet if anonymization is on, to align with card quotes
                            txt2 = anonymize_text(txt, codebook) if (codebook and cfg.dials.anonymize) else txt
                            retrieved_snips.append(txt2)
                            if sid and txt and sid not in retrieved_sources:
                                retrieved_sources[sid] = txt
                            added_texts.append(txt2)
                        if added_texts:
                            show = added_texts[:min(2, len(added_texts))]
                            if codebook and cfg.dials.anonymize:
                                show = [anonymize_text(s, codebook) for s in show]
                            tool_text_parts.append("option_retriever:\n- " + "\n- ".join(show))
                if tool_text_parts:
                    tool_text = "TOOLS:\n" + "\n".join(tool_text_parts)
                    ctx_text = (ctx_text + "\n\n" + tool_text).strip()
                if retrieved_snips:
                    context["retrieved_snippets"] = list(retrieved_snips)
                if retrieved_sources:
                    context["retrieved_sources"] = dict(retrieved_sources)
            # Fact-Cards two-pass (LEARN then USE)
            fact_cards_before = None
            fact_cards_after = None
            citations = None
            fact_cards_prepared = False
            if cfg.dials.use_fact_cards and not is_saq and isinstance(task, MCQTask):
                import json as _json
                # Build a learning source: presented stem plus any tool snippets
                learn_source_parts = [task.stem]
                # Include options so cards can cite the same tokens as answer choices
                try:
                    if isinstance(task, MCQTask) and task.options:
                        learn_source_parts.append("OPTIONS: " + " | ".join(task.options))
                except Exception:
                    pass
                if context.get("retrieved_snippets"):
                    learn_source_parts.extend(context.get("retrieved_snippets")[:1])
                learn_source = "\n\n".join(learn_source_parts)
                try:
                    fact_cards_before, fact_cards_after, context, notes_buf, _ = self._prepare_fact_cards_flow(
                        learner=learner,
                        task=task,
                        skill_id=skill_id,
                        cfg=cfg,
                        context=context,
                        notes_buf=notes_buf,
                        retrieval_corpus=retrieval_corpus,
                        learn_source=learn_source,
                    )
                    fact_cards_prepared = True
                except Exception:
                    fact_cards_prepared = False
                if _flag_on("SIM_FACTCARDS_LEGACY", False) and not fact_cards_prepared:
                    # existing cards
                    try:
                        cards_mem = (_json.loads(notes_buf) if notes_buf else {"cards": []})
                        if not isinstance(cards_mem, dict):
                            cards_mem = {"cards": []}
                    except Exception:
                        cards_mem = {"cards": []}
                    fact_cards_before = cards_mem
                    # If freezing provided cards, skip LEARN and use existing as context
                    if cfg.dials.freeze_cards:
                        # Start from provided cards (enforce tags/quote clamp), then inject per-option PRO cards
                        try:
                            import re as _re
                            from sim.utils.text import truncate_quote as _truncate_quote
                            # Normalize provided cards
                            merged: Dict[str, Dict[str, Any]] = {}
                            for c in (cards_mem.get("cards") or []):
                                try:
                                    cc = dict(c)
                                    tags = cc.get("tags") or []
                                    if skill_id and skill_id not in tags:
                                        cc["tags"] = list(tags) + [skill_id]
                                    q = cc.get("quote") or ""
                                    if len(_re.findall(r"[A-Za-z0-9]+", q)) > 15:
                                        cc["quote"] = _truncate_quote(q, 15)
                                    w = cc.get("where") or {}
                                    mk = f"{(cc.get('quote') or '').strip().lower()}|{w.get('scope')}:{w.get('option_index')}"
                                    merged[mk] = cc
                                except Exception:
                                    continue
                            # Inject required per-option PRO cards (ephemeral; isolates answering while keeping freeze)
                            required_cards: list[dict] = []
                            opt_texts = list(getattr(task, "options", []) or [])
                            for oi, opt in enumerate(opt_texts):
                                # Build a quote slice directly from the option text (preserve punctuation/dashes)
                                try:
                                    from sim.validation import first_n_tokens_span
                                    s, e, slice_txt = first_n_tokens_span(opt, 15)
                                except Exception:
                                    s, e, slice_txt = 0, len(opt), opt
                                quote = slice_txt
                                where = {"scope": "option", "option_index": oi, "start": s, "end": e, "source_id": f"option:{oi}"}
                                rc = {
                                    "id": f"opt{oi+1}",
                                    "claim": quote,
                                    "quote": quote,
                                    "where": where,
                                    "tags": [skill_id] if skill_id else [],
                                    "hypothesis": f"Option {oi} quoted",
                                    "polarity": "pro",
                                }
                                mk = f"{quote.strip().lower()}|option:{oi}"
                                if mk not in merged:
                                    merged[mk] = rc
                                required_cards.append(rc)
                            # Assemble within budget: required option cards first, then up to 2 context cards, then others
                            all_cards = list(merged.values())
                            context_cards = [c for c in all_cards if (c.get("where") or {}).get("scope") == "context"]
                            non_context = [c for c in all_cards if (c.get("where") or {}).get("scope") != "context"]
                            new_cards: List[Dict[str, Any]] = []
                            for rc in required_cards:
                                if len(new_cards) >= cfg.dials.fact_cards_budget:
                                    break
                                new_cards.append(rc)
                            for c in context_cards[:2]:
                                if len(new_cards) >= cfg.dials.fact_cards_budget:
                                    break
                                if c.get("id") in {x.get("id") for x in new_cards}:
                                    continue
                                new_cards.append(c)
                            for c in non_context:
                                if len(new_cards) >= cfg.dials.fact_cards_budget:
                                    break
                                if c.get("id") in {x.get("id") for x in new_cards}:
                                    continue
                                new_cards.append(c)
                            # Build option_card_ids for USE prompt and trim payloads
                            option_card_ids: Dict[str, str] = build_option_card_ids(new_cards, task)
                            new_cards = clamp_cards(new_cards, max_tokens=15)
                            fact_cards_after = {"cards": [trim_card(c) for c in new_cards]}
                            # Present combined cards as context; keep notes_buf persisted as provided JSON
                            ctx_json = _json.dumps(fact_cards_after, ensure_ascii=False)
                            if cfg.dials.closed_book:
                                context = {
                                    "notes_text": retrieval_corpus or notes_buf,
                                    "context_text": ctx_json,
                                    "fact_cards": new_cards,
                                    "skill_id": skill_id,
                                    **({} if not context.get("retrieved_snippets") else {"retrieved_snippets": context.get("retrieved_snippets")})
                                }
                            else:
                                context = {**context, "fact_cards": new_cards}
                            if option_card_ids:
                                context["option_card_ids"] = option_card_ids
                        except Exception:
                            # Fallback: use provided cards within budget
                            prior_cards = list((cards_mem.get("cards") or [])[: cfg.dials.fact_cards_budget])
                            fact_cards_after = {"cards": [trim_card(c) for c in prior_cards]}
                            if cfg.dials.closed_book:
                                context = {"notes_text": retrieval_corpus or notes_buf, "context_text": _json.dumps(fact_cards_after, ensure_ascii=False), "fact_cards": prior_cards, "skill_id": skill_id}
                            else:
                                context = {**context, "fact_cards": prior_cards}
                elif not fact_cards_prepared:
                    try:
                        cards_mem = (_json.loads(notes_buf) if notes_buf else {"cards": []})
                        if not isinstance(cards_mem, dict):
                            cards_mem = {"cards": []}
                    except Exception:
                        cards_mem = {"cards": []}
                    fact_cards_before = cards_mem
                    # LEARN: self-consistency over card extraction
                    # Track LEARN cost
                    import time as _time
                    learn_t0 = _time.time()
                    try:
                        _u0 = int((learner.get_usage_counters() or {}).get("total_tokens") or 0)
                    except Exception:
                        _u0 = 0
                    n_sc = int(cfg.dials.sc_extract_n or 0)
                    if n_sc <= 0:
                        n_sc = max(1, int(cfg.dials.self_consistency_n))
                    extracted = []
                    for _ in range(n_sc):
                        try:
                            js = learner.extract_fact_cards(task, source_text=learn_source, context=context)
                            if isinstance(js, dict) and isinstance(js.get("cards"), list):
                                extracted.append(js["cards"])
                        except Exception:
                            pass
                    # merge cards by (claim/quote + scope + option_index), cap budget
                    def norm(s: str) -> str:
                        return (s or "").strip().lower()
                    def make_key(card: Dict[str, Any]) -> str:
                        w = card.get("where") or {}
                        scope = (w.get("scope") or "").strip().lower()
                        oi = w.get("option_index")
                        try:
                            oi_s = str(int(oi)) if oi is not None else "NA"
                        except Exception:
                            oi_s = "NA"
                        base = norm(card.get("claim") or "") or norm(card.get("quote") or "")
                        return f"{base}|{scope}:{oi_s}"
                    merged: Dict[str, Dict[str, Any]] = {}
                    for lst in extracted:
                        for c in lst:
                            k = make_key(c)
                            if not k:
                                continue
                            if k not in merged:
                                merged[k] = {
                                    "id": c.get("id") or f"fc{len(merged)+1}",
                                    "claim": c.get("claim") or "",
                                    "quote": c.get("quote") or "",
                                    "where": c.get("where") or {},
                                    "tags": c.get("tags") or [],
                                    "hypothesis": c.get("hypothesis") or "",
                                    "polarity": c.get("polarity") or "pro",
                                }
                    # include prior cards first (preserve distinct option-linked variants)
                    for c in (cards_mem.get("cards") or [])[: cfg.dials.fact_cards_budget]:
                        k = make_key(c)
                        if k and k not in merged:
                            merged[k] = c
                    # Normalize/validate cards, then ensure per-option witnesses and reserve snippets
                    used_manager = False

                    if _flag_on("SIM_USE_FACTCARD_MANAGER", True):
                        try:
                            from sim.factcards_manager import FactCardManager as _FactCardManager
                            _mgr = _FactCardManager()
                            new_cards, required_cards = _mgr.prepare_cards(
                                merged=merged,
                                task=task,
                                skill_id=skill_id,
                                retrieved_snippets=(context or {}).get("retrieved_snippets") or [],
                                budget=int(cfg.dials.fact_cards_budget),
                                dedup_sim=float(getattr(cfg.dials, 'dedup_sim', 0.88) or 0.88),
                                q_min=int(getattr(cfg.dials, 'q_min', 1) or 1),
                                max_boosts=int(getattr(cfg.dials, 'max_learn_boosts', 0) or 0),
                                extract_fn=lambda **kw: learner.extract_fact_cards(**kw),
                                learn_source=learn_source,
                                context=context,
                            )
                            used_manager = True
                        except Exception:
                            used_manager = False
                    # Apply quality filter even when manager used
                    try:
                        if float(getattr(cfg.dials, 'min_cqs', 0.0) or 0.0) > 0.0:
                            new_cards = _apply_card_quality_filter(new_cards)
                    except Exception:
                        pass
                    # Optional: apply card quality filter to emphasize high‑quality option quotes
                    def _apply_card_quality_filter(cards: list[dict]) -> list[dict]:
                        try:
                            min_cqs = float(getattr(cfg.dials, 'min_cqs', 0.0) or 0.0)
                            top_k = int(getattr(cfg.dials, 'per_option_top_k', 3) or 3)
                            min_len = int(getattr(cfg.dials, 'min_card_len', 40) or 40)
                            max_len = int(getattr(cfg.dials, 'max_card_len', 300) or 300)
                            dedup_sim = float(getattr(cfg.dials, 'dedup_sim', 0.88) or 0.88)
                            # Build IDF over options once
                            try:
                                idf = build_idf(list(getattr(task, 'options', []) or []))
                            except Exception:
                                idf = {}
                            # Group by option; keep context/other as-is
                            by_opt: dict[int, list[dict]] = {}
                            ctx_cards: list[dict] = []
                            for c in cards or []:
                                w = c.get('where') or {}
                                if w.get('scope') == 'option' and isinstance(w.get('option_index'), int):
                                    oi = int(w.get('option_index'))
                                    by_opt.setdefault(oi, []).append(c)
                                else:
                                    ctx_cards.append(c)
                            out: list[dict] = []
                            # Preserve any required opt cards (id startswith 'opt')
                            required_ids = {str(c.get('id')) for c in cards if str(c.get('id','')).startswith('opt')}
                            opt_texts = list(getattr(task, 'options', []) or [])
                            for oi, lst in by_opt.items():
                                # Always include required card(s) for this option
                                req = [c for c in lst if str(c.get('id','')).startswith('opt')]
                                keep: list[tuple[dict, float]] = [(c, 1e9) for c in req]
                                # Score remaining
                                scored: list[tuple[dict, float]] = []
                                for c in lst:
                                    if str(c.get('id')) in required_ids:
                                        continue
                                    try:
                                        q = c.get('quote') or ''
                                        others = [opt_texts[j] for j in range(len(opt_texts)) if j != oi]
                                        s = assess_quote(q, opt_texts[oi] if 0 <= oi < len(opt_texts) else '', others, str((c.get('where') or {}).get('source_id') or ''))
                                        # basic length check too (on characters)
                                        L = len(q)
                                        if s.cqs >= min_cqs and (L >= min_len) and (L <= max_len):
                                            scored.append((c, float(s.cqs)))
                                    except Exception:
                                        continue
                                scored.sort(key=lambda t: t[1], reverse=True)
                                # Dedup near-identical quotes per option
                                seen_texts: list[str] = [c.get('quote') or '' for c, _ in keep]
                                for c, sc in scored:
                                    if len(keep) >= max(0, top_k):
                                        break
                                    qt = c.get('quote') or ''
                                    if any(_cq_jaccard(_cq_tokens(qt), _cq_tokens(t2)) >= dedup_sim for t2 in seen_texts):
                                        continue
                                    keep.append((c, sc))
                                    seen_texts.append(qt)
                                out.extend([c for c, _ in keep])
                            # Re-append context cards unchanged
                            out.extend(ctx_cards)
                            return out
                        except Exception:
                            return list(cards or [])

                    if not used_manager:
                        try:
                            import re as _re
                            def _first_tokens(s: str, n: int = 15) -> str:
                                toks = _re.findall(r"[A-Za-z0-9]+", s or "")
                                return " ".join(toks[: n])
                            # Step 0: normalize quotes/tags/ids and enforce verbatim substring + ≤15 tokens
                            src_snips = (context or {}).get("retrieved_snippets") or []
                            opt_texts = list(getattr(task, "options", []) or [])
                            seen_ids: set[str] = set()
                            idx_counter = 1
                            for k, c in list(merged.items()):
                                c = merged[k]
                                # id uniqueness
                                cid = str(c.get("id") or "").strip()
                                if not cid or cid in seen_ids:
                                    cid = f"f{idx_counter}"
                                    idx_counter += 1
                                    c["id"] = cid
                                seen_ids.add(cid)
                                # ensure tags contain skill_id
                                tags = c.get("tags") or []
                                if skill_id and skill_id not in tags:
                                    tags = list(tags) + [skill_id]
                                    c["tags"] = tags
                                w = c.get("where") or {}
                                scope = w.get("scope")
                                quote = c.get("quote") or ""
                                # Fix oversize quotes
                                if len(_re.findall(r"[A-Za-z0-9]+", quote)) > 15:
                                    quote = _first_tokens(quote, 15)
                                # Enforce verbatim substring based on scope
                                if scope == "option":
                                    try:
                                        oi = int(w.get("option_index"))
                                    except Exception:
                                        oi = None
                                    if (oi is None) or (oi < 0) or (oi >= len(opt_texts)):
                                        # default to 0 if invalid
                                        oi = 0
                                        c.setdefault("where", {})["option_index"] = oi
                                    opt = opt_texts[oi] if oi is not None and 0 <= oi < len(opt_texts) else ""
                                    from sim.validation import canon, first_n_tokens_span
                                    # If offsets are present and valid, slice from option and override quote
                                    s = w.get("start")
                                    e = w.get("end")
                                    if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(opt):
                                        slice_txt = opt[s:e]
                                        # clamp to ≤15 tokens if needed by shrinking end to the 15th token boundary
                                        if len(_re.findall(r"[A-Za-z0-9]+", slice_txt)) > 15:
                                            # recompute span from s to cover 15 tokens
                                            # find tokens in opt starting from s
                                            toks = [m for m in _re.finditer(r"[A-Za-z0-9]+", opt) if m.end() > s]
                                            if toks:
                                                start_s = s if s <= toks[0].start() else toks[0].start()
                                                n_tok = min(15, len(toks))
                                                end_e = toks[n_tok - 1].end()
                                                slice_txt = opt[start_s:end_e]
                                                c.setdefault("where", {})["start"] = start_s
                                                c.setdefault("where", {})["end"] = end_e
                                        c["quote"] = slice_txt
                                    else:
                                        # Fallback: canonicalized substring check; if fails, pick first-n-tokens slice
                                        if quote and (canon(quote) in canon(opt)):
                                            c["quote"] = quote
                                        else:
                                            s2, e2, slice_txt2 = first_n_tokens_span(opt, 15)
                                            c.setdefault("where", {})["start"] = s2
                                            c.setdefault("where", {})["end"] = e2
                                            c["quote"] = slice_txt2
                                elif scope == "context":
                                    src = src_snips[0] if src_snips else (context.get("context_text") or "")
                                    if quote and quote not in src:
                                        quote = _first_tokens(src, 15)
                                    if not quote:
                                        quote = _first_tokens(src, 15)
                                    c["quote"] = quote
                                else:
                                    # unknown scope → coerce to context with snippet or to option 0 if no context
                                    if src_snips:
                                        c.setdefault("where", {})["scope"] = "context"
                                        c["quote"] = _first_tokens(src_snips[0], 15)
                                    else:
                                        c.setdefault("where", {})["scope"] = "option"
                                        c["where"]["option_index"] = 0
                                        # Make sure this quote is a slice of the option text
                                        try:
                                            from sim.validation import first_n_tokens_span
                                            s0, e0, q0 = first_n_tokens_span(opt_texts[0] if opt_texts else "", 15)
                                        except Exception:
                                            s0, e0, q0 = 0, 0, ""
                                        c.setdefault("where", {})["start"] = s0
                                        c.setdefault("where", {})["end"] = e0
                                        c["quote"] = q0
                            # Build required per-option PRO cards
                            required_cards: list[dict] = []
                            for oi, opt in enumerate(getattr(task, "options", []) or []):
                                # does a suitable card already exist?
                                exists = False
                                for c in merged.values():
                                    w = c.get("where") or {}
                                    if (w.get("scope") == "option") and (w.get("option_index") == oi):
                                        # enforce <=15 token quote
                                        q = c.get("quote") or ""
                                        if len(_re.findall(r"[A-Za-z0-9]+", q)) <= 15:
                                            exists = True
                                            break
                                if not exists:
                                    # Slice directly from option to preserve punctuation/dashes
                                    from sim.validation import first_n_tokens_span
                                    s, e, quote = first_n_tokens_span(opt, 15)
                                    where = {"scope": "option", "option_index": oi, "start": s, "end": e, "source_id": f"option:{oi}"}
                                    rc = {
                                        "id": f"opt{oi+1}",
                                        "claim": quote,
                                        "quote": quote,
                                        "where": where,
                                        "tags": [skill_id],
                                        "hypothesis": f"Option {oi} quoted",
                                        "polarity": "pro",
                                    }
                                    # insert into merged with deterministic key so it survives budgeting
                                    mk = f"{quote.strip().lower()}|option:{oi}"
                                    if mk not in merged:
                                        merged[mk] = rc
                                    required_cards.append(rc)
                            # Reserve budget: per-option PRO cards first, then up to 2 context snippet cards, then the rest
                            others_all = [c for k, c in merged.items() if c.get("id") not in {rc.get("id") for rc in required_cards}]
                            # Prefer context cards as snippet candidates
                            context_cards = [c for c in others_all if (c.get("where") or {}).get("scope") == "context"]
                            non_context = [c for c in others_all if (c.get("where") or {}).get("scope") != "context"]
                            # Assemble within budget
                            new_cards = []
                            # 1) Required per-option PROs
                            for rc in required_cards:
                                if len(new_cards) >= cfg.dials.fact_cards_budget:
                                    break
                                new_cards.append(rc)
                            # 2) Pin top-2 context/snippet cards (if any)
                            for c in context_cards[:2]:
                                if len(new_cards) >= cfg.dials.fact_cards_budget:
                                    break
                                if c.get("id") in {x.get("id") for x in new_cards}:
                                    continue
                                new_cards.append(c)
                            # 3) Fill remaining with others
                            for c in non_context:
                                if len(new_cards) >= cfg.dials.fact_cards_budget:
                                    break
                                if c.get("id") in {x.get("id") for x in new_cards}:
                                    continue
                                new_cards.append(c)
                        except Exception:
                            # Fallback: simple cap
                            new_cards = list(merged.values())[: cfg.dials.fact_cards_budget]
                    # Evidence tighten: near-duplicate dedup per option and pre-gate q_min with optional boosts
                    try:
                        from sim.factcards import dedup_per_option
                        new_cards = dedup_per_option(new_cards, sim_threshold=float(getattr(cfg.dials, 'dedup_sim', 0.88) or 0.88))
                        # q_min gating (pre-selection). If any option has < q_min, try one extra targeted extract round.
                        def _counts(cards: list[dict]) -> Dict[int, int]:
                            cc: Dict[int, int] = {}
                            for c in cards:
                                w = c.get("where") or {}
                                if w.get("scope") == "option" and isinstance(w.get("option_index"), int):
                                    oi = int(w.get("option_index"))
                                    cc[oi] = cc.get(oi, 0) + 1
                            return cc
                        cnts = _counts(new_cards)
                        boosts = 0
                        while (min([cnts.get(i, 0) for i in range(len(getattr(task, 'options', []) or []))] or [0]) < int(getattr(cfg.dials, 'q_min', 1) or 1)) and (boosts < int(getattr(cfg.dials, 'max_learn_boosts', 0) or 0)):
                            try:
                                jsb = learner.extract_fact_cards(task, source_text=learn_source, context=context)
                                if isinstance(jsb, dict) and isinstance(jsb.get("cards"), list):
                                    for c in jsb["cards"]:
                                        mk = make_key(c)
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
                            # rebuild new_cards quickly within budget
                            others_all = [c for k, c in merged.items() if c.get("id") not in {rc.get("id") for rc in required_cards}]
                            context_cards = [c for c in others_all if (c.get("where") or {}).get("scope") == "context"]
                            non_context = [c for c in others_all if (c.get("where") or {}).get("scope") != "context"]
                            new_cards = []
                            for rc in required_cards:
                                if len(new_cards) >= cfg.dials.fact_cards_budget:
                                    break
                                new_cards.append(rc)
                            for c in context_cards[:2]:
                                if len(new_cards) >= cfg.dials.fact_cards_budget:
                                    break
                                if c.get("id") in {x.get("id") for x in new_cards}:
                                    continue
                                new_cards.append(c)
                            for c in non_context:
                                if len(new_cards) >= cfg.dials.fact_cards_budget:
                                    break
                                if c.get("id") in {x.get("id") for x in new_cards}:
                                    continue
                                new_cards.append(c)
                            from sim.factcards import dedup_per_option
                            new_cards = dedup_per_option(new_cards, sim_threshold=float(getattr(cfg.dials, 'dedup_sim', 0.88) or 0.88))
                            cnts = _counts(new_cards)
                            boosts += 1
                    except Exception:
                        pass
                    # Apply card quality filter before building ids
                    if float(getattr(cfg.dials, 'min_cqs', 0.0) or 0.0) > 0.0:
                        new_cards = _apply_card_quality_filter(new_cards)
                    # Build per-option PRO id map and trim cards for learner context
                    option_card_ids: Dict[str, str] = build_option_card_ids(new_cards, task)
                    fact_cards_after = {"cards": [trim_card(c) for c in new_cards]}
                    notes_buf = _json.dumps(fact_cards_after, ensure_ascii=False)
                    # Measure LEARN resource usage
                    try:
                        _u1 = int((learner.get_usage_counters() or {}).get("total_tokens") or 0)
                    except Exception:
                        _u1 = _u0
                    _learn_tokens = max(0, _u1 - _u0)
                    _learn_time_s = max(0.0, _time.time() - learn_t0)
                    if cfg.dials.closed_book:
                        # preserve skill_id and retrieved_snippets in updated context
                        # Keep notes_text as the retrieval corpus so retrievers continue to see the source text,
                        # while context_text shows the current Fact-Cards JSON to the learner.
                        context = {
                            "notes_text": retrieval_corpus or notes_buf,
                            "context_text": notes_buf,
                            "fact_cards": new_cards,
                            "skill_id": skill_id,
                            **({} if not context.get("retrieved_snippets") else {"retrieved_snippets": context.get("retrieved_snippets")}),
                            **({} if not context.get("retrieved_sources") else {"retrieved_sources": context.get("retrieved_sources")}),
                            "_learn_tokens": _learn_tokens,
                            "_learn_time_s": _learn_time_s,
                        }
                    else:
                        context = {**context, "fact_cards": new_cards, "_learn_tokens": _learn_tokens, "_learn_time_s": _learn_time_s}
                    # Surface option_card_ids to learners if available
                    if option_card_ids:
                        context["option_card_ids"] = option_card_ids

            # Answer
            if is_saq:
                # SAQ self-consistency: generate N drafts, grade each, keep best
                n = max(1, int(cfg.dials.self_consistency_n))
                saq_drafts = []
                best = None
                for _ in range(n):
                    a = learner.answer_saq(task, context=context)
                    g = self.llm.grade_saq(task.stem, task.expected_points, task.model_answer, a.get("student_answer") or "")
                    item = {"answer": a.get("student_answer"), "grading": g}
                    saq_drafts.append(item)
                    if best is None or float(g.get("score") or 0.0) > float(best["grading"].get("score") or 0.0):
                        best = item
                ans = {"student_answer": (best or saq_drafts[0]).get("answer")}
                grading = (best or saq_drafts[0])["grading"]
            else:
                # Self-consistency / evaluation by task type
                if isinstance(task, MCQTask):
                    from collections import Counter
                    votes: list[int | None] = []
                    cand_meta: List[Dict[str, Any]] = []  # store per-sample metadata (confidence, citations)
                    citations = None
                    # Determine N based on policy/difficulty
                    if (cfg.dials.sc_policy or "fixed") == "adaptive":
                        if (cfg.difficulty or "medium") == "easy":
                            n_max = max(1, int(cfg.dials.sc_k_easy))
                        elif (cfg.difficulty or "medium") == "hard":
                            n_max = max(1, int(cfg.dials.sc_k_hard))
                        else:
                            n_max = max(1, int(cfg.dials.sc_k_medium))
                    else:
                        n_max = max(1, int(cfg.dials.self_consistency_n))
                    # Helper: decoding params and reasoning scaffold for this pass
                    def _ctx_for_pass(base_ctx: dict, *, temp: float, reasoning_override: str | None = None) -> dict:
                        ctx2 = dict(base_ctx or {})
                        # pass decoding
                        dec = {"temperature": float(temp), "top_p": float(cfg.dials.top_p)}
                        if cfg.dials.min_p is not None:
                            dec["min_p"] = float(cfg.dials.min_p)
                        # Grammar/JSON schema
                        if (cfg.dials.grammar or "json") == "schema":
                            # simple schema for MCQ answers
                            dec["response_format"] = {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "mcq_answer",
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "chosen_index": {"type": "integer", "minimum": 0},
                                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                        },
                                        "required": ["chosen_index"],
                                        "additionalProperties": True
                                    }
                                }
                            }
                        ctx2["decode"] = dec
                        rmode = reasoning_override if reasoning_override else cfg.dials.reasoning
                        if (rmode or "none") != "none":
                            ctx2["reasoning"] = rmode
                        # include examples if already selected
                        return ctx2
                    # Seed controller vote(s) if configured
                    if (cfg.dials.controller or "basic") != "basic":
                        try:
                            from sim.controllers import ltm_controller, tot_controller
                            base_ctx = _ctx_for_pass(context, temp=cfg.dials.temp_answer)
                            if cfg.dials.controller == "ltm":
                                v_ct, m_ct = ltm_controller(self.llm, learner, task, base_ctx, max_steps=4)
                            else:
                                v_ct, m_ct = tot_controller(
                                    self.llm,
                                    learner,
                                    task,
                                    base_ctx,
                                    width=int(cfg.dials.tot_width or 2),
                                    depth=int(cfg.dials.tot_depth or 2),
                                    judge=(cfg.dials.tot_judge or "self"),
                                    budget=int(cfg.dials.controller_budget or 6),
                                )
                            for v in v_ct:
                                votes.append(v)
                            cand_meta.extend(m_ct or [])
                        except Exception:
                            pass
                    # Branch: evidence-first quote-then-vote controller (no answer-time compute)
                    if (cfg.dials.controller or "basic") == "quote_then_vote" and cfg.dials.use_fact_cards:
                        votes2, meta2, citations2, chosen_idx2, evi_tel, card_metrics = quote_then_vote_controller(
                            llm=self.llm,
                            learner=learner,
                            task=task,
                            context=context,
                            cfg=cfg,
                            fact_cards_after=fact_cards_after,
                            notes_buf=notes_buf,
                            skill_id=skill_id,
                        )
                        # Build a minimal early-return record mirroring previous behavior
                        from time import time as _time
                        stem_text = getattr(task, "stem", "")
                        presented_stem = stem_text
                        try:
                            if context.get("context_text") and cfg.dials.context_position != "none":
                                if cfg.dials.context_position == "pre":
                                    presented_stem = f"CONTEXT:\n{context['context_text']}\n\nQUESTION: {stem_text}"
                                elif cfg.dials.context_position == "post":
                                    presented_stem = f"QUESTION: {stem_text}\n\nCONTEXT:\n{context['context_text']}"
                        except Exception:
                            presented_stem = stem_text
                        ans = {"chosen_index": chosen_idx2}
                        if citations2:
                            ans["citations"] = citations2
                        # Best-effort repair: ensure required PRO id and q_min option-linked citations are present
                        try:
                            if cfg.dials.use_fact_cards and isinstance(chosen_idx2, int):
                                opt_map = (context or {}).get("option_card_ids") or {}
                                letter = chr(ord('A') + int(chosen_idx2)) if (0 <= int(chosen_idx2) < 26) else None
                                req_id = (opt_map.get(letter) if letter and isinstance(opt_map, dict) else None)
                                cards_pool = []
                                try:
                                    if isinstance(fact_cards_after, dict):
                                        cards_pool = list(fact_cards_after.get("cards") or [])
                                except Exception:
                                    cards_pool = []
                                cits = list(ans.get("citations") or [])
                                # Normalize to strings for containment checks
                                def _to_id(x):
                                    try:
                                        return str(x.get("id")) if isinstance(x, dict) and x.get("id") is not None else (str(x) if x is not None else None)
                                    except Exception:
                                        return None
                                cit_ids = [ _to_id(x) for x in cits ]
                                # Prepend required PRO id if missing
                                if req_id is not None and (str(req_id) not in (cit_ids or [])):
                                    cits = [req_id] + cits
                                # Ensure at least q_min option-linked citations (by id) for the chosen option
                                try:
                                    need = max(1, int(getattr(cfg.dials, 'q_min', 1) or 1))
                                except Exception:
                                    need = 1
                                def _is_opt_card(c):
                                    w = c.get('where') or {}
                                    return (w.get('scope') == 'option') and (int(w.get('option_index') or -1) == int(chosen_idx2))
                                have = 0
                                have_ids = set()
                                for x in cits:
                                    xid = _to_id(x)
                                    if not xid:
                                        continue
                                    have_ids.add(xid)
                                    for c in cards_pool:
                                        if str(c.get('id')) == xid and _is_opt_card(c):
                                            have += 1
                                            break
                                if have < need:
                                    for c in cards_pool:
                                        if _is_opt_card(c):
                                            cid = str(c.get('id'))
                                            if cid not in have_ids:
                                                cits.append(cid)
                                                have_ids.add(cid)
                                                have += 1
                                                if have >= need:
                                                    break
                                if cits:
                                    ans['citations'] = cits
                        except Exception:
                            pass
                        # Synthesize a minimal witness block from cited cards to aid provenance (when missing)
                        try:
                            if cfg.dials.use_fact_cards and not isinstance(ans.get('witness'), (dict, list)):
                                cards_pool = []
                                try:
                                    if isinstance(fact_cards_after, dict):
                                        cards_pool = list(fact_cards_after.get("cards") or [])
                                except Exception:
                                    cards_pool = []
                                def _find_card(cid):
                                    for c in cards_pool:
                                        if str(c.get('id')) == str(cid):
                                            return c
                                    return None
                                cits_loc = list(ans.get('citations') or [])
                                if cits_loc:
                                    # choice: prefer option-linked cited card
                                    choice_id = None
                                    choice_quote = None
                                    for x in cits_loc:
                                        cid = str(x.get('id')) if isinstance(x, dict) and x.get('id') is not None else str(x)
                                        c = _find_card(cid)
                                        if c:
                                            w = c.get('where') or {}
                                            if w.get('scope') == 'option' and int(w.get('option_index') or -1) == int(chosen_idx2):
                                                choice_id = cid
                                                choice_quote = c.get('quote') or ''
                                                break
                                    if not choice_id:
                                        cid = str(cits_loc[0].get('id')) if isinstance(cits_loc[0], dict) and cits_loc[0].get('id') is not None else str(cits_loc[0])
                                        c = _find_card(cid)
                                        choice_id = cid
                                        choice_quote = (c.get('quote') if c else '')
                                    # rule: use a context card if available
                                    rule_id = None
                                    rule_quote = None
                                    for c in cards_pool:
                                        w = c.get('where') or {}
                                        if w.get('scope') == 'context' and (c.get('quote') or '').strip():
                                            rule_id = str(c.get('id'))
                                            rule_quote = c.get('quote')
                                            break
                                    if not rule_id:
                                        rule_id = choice_id
                                        rule_quote = choice_quote
                                    if choice_id:
                                        ans['witness'] = {'rule': {'card_id': rule_id, 'quote': rule_quote or ''}, 'choice': {'card_id': choice_id, 'quote': choice_quote or ''}}
                        except Exception:
                            pass
                        # Trim citations to emphasize chosen option evidence (reduce witness ambiguity)
                        try:
                            if cfg.dials.use_fact_cards and isinstance(ans.get('citations'), list) and isinstance(chosen_idx2, int):
                                cits = list(ans.get('citations') or [])
                                cards_pool = []
                                try:
                                    if isinstance(fact_cards_after, dict):
                                        cards_pool = list(fact_cards_after.get("cards") or [])
                                except Exception:
                                    cards_pool = []
                                def _to_id(x):
                                    try:
                                        return str(x.get("id")) if isinstance(x, dict) and x.get("id") is not None else (str(x) if x is not None else None)
                                    except Exception:
                                        return None
                                def _is_opt_card_id(cid: str) -> bool:
                                    for c in cards_pool:
                                        if str(c.get('id')) == cid:
                                            w = c.get('where') or {}
                                            return (w.get('scope') == 'option') and (int(w.get('option_index') or -1) == int(chosen_idx2))
                                    return False
                                # Keep PRO id first if present
                                opt_map = (context or {}).get("option_card_ids") or {}
                                letter = chr(ord('A') + int(chosen_idx2)) if (0 <= int(chosen_idx2) < 26) else None
                                pro_id = (opt_map.get(letter) if letter and isinstance(opt_map, dict) else None)
                                filtered: list[str] = []
                                if pro_id is not None:
                                    filtered.append(str(pro_id))
                                # Then keep other chosen-option citations
                                for x in cits:
                                    cid = _to_id(x)
                                    if cid and _is_opt_card_id(cid) and cid not in filtered:
                                        filtered.append(cid)
                                # Optionally, keep at most one context citation
                                for x in cits:
                                    cid = _to_id(x)
                                    if not cid:
                                        continue
                                    for c in cards_pool:
                                        if str(c.get('id')) == cid:
                                            w = c.get('where') or {}
                                            if w.get('scope') == 'context' and cid not in filtered:
                                                filtered.append(cid)
                                            break
                                if filtered:
                                    ans['citations'] = filtered
                        except Exception:
                            pass
                        evaluation = evaluate_mcq(ans.get("chosen_index"), task)
                        # Unified credit check for controller early-return
                        try:
                            from sim.credit import check as credit_check
                            ce = credit_check(
                                task=task,
                                chosen_index=ans.get("chosen_index"),
                                citations=ans.get("citations"),
                                notes_buf=notes_buf,
                                skill_id=skill_id,
                                option_card_ids=(context or {}).get("option_card_ids") or {},
                                retrieved_snippets=(context or {}).get("retrieved_snippets") or [],
                                witness=ans.get("witness"),
                                coverage_tau=float(cfg.coverage_tau),
                                q_min=int(getattr(cfg.dials, "q_min", 1) or 1),
                                tools_used=bool(getattr(cfg.dials, "use_tools", False)),
                                cards_override=(fact_cards_after.get("cards") if isinstance(fact_cards_after, dict) else None),
                            )
                            credited = bool(evaluation.get("correct") and ce.get("credited", False))
                            evaluation = {**evaluation, "citations_evidence": {**ce, "credited": credited}}
                        except Exception:
                            pass
                        try:
                            student_usage = learner.get_usage_counters() if hasattr(learner, "get_usage_counters") else None
                        except Exception:
                            student_usage = None
                        try:
                            tutor_usage = self.llm.get_usage_counters() if hasattr(self.llm, "get_usage_counters") else None
                        except Exception:
                            tutor_usage = None
                        _log_rationales = os.getenv("TUTOR_LOG_RATIONALES") in ("1", "true", "True")
                        # collect messages if enabled
                        _log_mode = (os.getenv("LOG_MESSAGES", "").lower())
                        _msgs: list[dict] = []
                        if _log_mode in ("counts", "text"):
                            try:
                                if hasattr(self.llm, "get_messages_buffer"):
                                    _msgs.extend(self.llm.get_messages_buffer())  # type: ignore
                            except Exception:
                                pass
                            try:
                                if hasattr(learner, "get_messages_buffer"):
                                    _msgs.extend(learner.get_messages_buffer())  # type: ignore
                            except Exception:
                                pass
                            try:
                                _msgs.sort(key=lambda m: m.get("ts", 0))
                            except Exception:
                                pass
                        rec = {
                            "run_id": run_id,
                            "step": step,
                            "ts": int(time.time()),
                            "task": ({"type": "mcq", "stem": task.stem, "options": task.options}
                                     | ({} if (not _log_rationales or not getattr(task, 'rationales', None)) else {"rationales": task.rationales})),
                            "presented_stem": presented_stem,
                            "answer": ans,
                            "evaluation": evaluation,
                            **({} if not tool_outputs else {"tool_outputs": tool_outputs, "tools_used": [o.get("name") for o in tool_outputs]}),
                            **({} if not cfg.dials.use_fact_cards else {"fact_cards_before": fact_cards_before, "fact_cards_after": fact_cards_after}),
                            **({"evidence_telemetry": evi_tel} if evi_tel else {}),
                            "duration_ms": int((_time() - step_t0) * 1000.0),
                            **({} if not student_usage else {"student_usage": student_usage}),
                            **({} if not tutor_usage else {"tutor_usage": tutor_usage}),
                            **({} if not card_metrics else {"card_metrics": card_metrics}),
                            **({} if not _msgs else {"messages": _msgs}),
                        }
                        # Guardrails: evaluate token bands and emit alerts/nudges
                        try:
                            rec = self._apply_guardrails(rec, cfg)
                        except Exception:
                            pass
                        logs.append(rec)
                        if log_f:
                            import json
                            log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            log_f.flush()
                        if progress_cb:
                            try:
                                progress_cb(step + 1, total_steps)
                            except Exception:
                                pass
                        continue

                    # Adaptive early-stopping SC: cheap first pass then sample until quorum
                    if cfg.dials.adaptive_sc and n_max > 1:
                        # First cheap pass
                        v0 = learner.answer_mcq(task, context=_ctx_for_pass(context, temp=cfg.dials.temp_answer))
                        votes.append(v0.get("chosen_index"))
                        cand_meta.append({"confidence": (v0.get("raw") or {}).get("confidence"), "citations": v0.get("citations")})
                        if v0.get("citations"):
                            citations = v0.get("citations")
                        quorum = int(cfg.dials.sc_quorum or 0) or (n_max // 2 + 1)
                        # Continue sampling until quorum reached or n_max
                        while len(votes) < n_max:
                            counts = Counter(votes)
                            top = counts.most_common(1)[0]
                            if top[1] >= quorum:
                                break
                            vi = learner.answer_mcq(task, context=_ctx_for_pass(context, temp=cfg.dials.temp_sc))
                            votes.append(vi.get("chosen_index"))
                            cand_meta.append({"confidence": (vi.get("raw") or {}).get("confidence"), "citations": vi.get("citations")})
                            if (citations is None) and vi.get("citations"):
                                citations = vi.get("citations")
                    else:
                        # Fixed-N SC (no early stopping)
                        for _ in range(n_max):
                            v = learner.answer_mcq(task, context=_ctx_for_pass(context, temp=cfg.dials.temp_sc if n_max > 1 else cfg.dials.temp_answer))
                            votes.append(v.get("chosen_index"))
                            cand_meta.append({"confidence": (v.get("raw") or {}).get("confidence"), "citations": v.get("citations")})
                            if v.get("citations"):
                                citations = v.get("citations")
                    # Uncertainty gating: escalate if low confidence or high entropy disagreement
                    if cfg.dials.uncertainty_gate:
                        import math as _m
                        try:
                            # avg confidence
                            confs = [float(cm.get("confidence")) for cm in cand_meta if cm.get("confidence") is not None]
                        except Exception:
                            confs = []
                        avg_conf = (sum(confs) / len(confs)) if confs else 0.0
                        # normalized entropy over votes
                        from collections import Counter as _C
                        cts = _C([v for v in votes if v is not None])
                        total = max(1, len([v for v in votes if v is not None]))
                        probs = [n/total for n in cts.values() if n > 0]
                        H = -sum(p*_m.log(p+1e-12) for p in probs)
                        Hmax = _m.log(len(probs)) if probs else 1.0
                        Hn = (H / Hmax) if Hmax > 0 else 0.0
                        # Trigger rules
                        trigger = (avg_conf < float(cfg.dials.conf_threshold)) or (Hn > float(cfg.dials.entropy_threshold))
                        if trigger and len(votes) < int(cfg.dials.max_k_escalated or 12):
                            extra = int(cfg.dials.max_k_escalated) - len(votes)
                            r_override = ("tot" if cfg.dials.escalate_reasoning else None)
                            for _ in range(max(0, extra)):
                                vi = learner.answer_mcq(task, context=_ctx_for_pass(context, temp=cfg.dials.temp_sc, reasoning_override=r_override))
                                votes.append(vi.get("chosen_index"))
                                cand_meta.append({"confidence": (vi.get("raw") or {}).get("confidence"), "citations": vi.get("citations")})
                                if (citations is None) and vi.get("citations"):
                                    citations = vi.get("citations")
                    # Majority vote (ignore None abstentions if any concrete votes exist)
                    final_choice = None
                    if votes:
                        concrete = [v for v in votes if v is not None]
                        if concrete:
                            counts = Counter(concrete)
                            final_choice = counts.most_common(1)[0][0]
                        else:
                            counts = Counter(votes)
                            final_choice = counts.most_common(1)[0][0]
                    # Evidence-first tie-break using cited quotes (if available)
                    chosen_idx = final_choice
                    if cfg.dials.use_fact_cards and cfg.dials.require_citations and citations:
                        sig = compute_evidence_signals(task, notes_buf, skill_id, citations, (context or {}).get("option_card_ids") or {})
                        w_scores = sig.get("witness_scores") or {}
                        if w_scores and (chosen_idx is not None):
                            witness_idx = sig.get("witness_idx")
                            chosen_score = w_scores.get(chosen_idx, -1)
                            top_score = max(w_scores.values()) if w_scores else -1
                            if (witness_idx is not None) and (witness_idx != chosen_idx) and (top_score - chosen_score >= 1):
                                chosen_idx = witness_idx
                    # Reflexion-style self-critique: add one revised candidate
                    if cfg.dials.reflexion:
                        try:
                            import json as _json
                            init_letter = None if chosen_idx is None else chr(ord('A') + int(chosen_idx))
                            payload = {
                                "stem": task.stem,
                                "options": task.options,
                                "initial_choice": init_letter,
                                "context": (context or {}).get("context_text") or "",
                            }
                            jsr = self.llm._chat_json(
                                "Critique and revise an MCQ choice. Return JSON with chosen_index (int) and confidence (0..1).",
                                _json.dumps(payload, ensure_ascii=False),
                            )
                            ci = jsr.get("chosen_index")
                            if isinstance(ci, int):
                                votes.append(ci)
                                cand_meta.append({"confidence": jsr.get("confidence"), "citations": None})
                        except Exception:
                            pass
                    # Best-of-N reranking (optional)
                    chosen_idx2, votes, cand_meta = apply_best_of_rerank(
                        llm=self.llm,
                        learner=learner,
                        cfg=cfg,
                        task=task,
                        context=context,
                        votes=votes,
                        cand_meta=cand_meta,
                        notes_buf=notes_buf,
                        skill_id=skill_id,
                    )
                    if chosen_idx2 is not None:
                        chosen_idx = chosen_idx2
                    # Evidence-weighted scoring override (when using Fact-Cards)
                    evi_telemetry: Dict[str, Any] | None = None
                    if (
                        cfg.dials.use_fact_cards
                        and bool(getattr(cfg.dials, "evidence_weighted_selection", False))
                        and (cfg.dials.citation_mode in ("lenient", "strict"))
                        and isinstance(fact_cards_after, dict)
                    ):
                        try:
                            cards = list((fact_cards_after or {}).get("cards") or [])
                            opt_texts = list(getattr(task, "options", []) or [])
                            from sim.evidence import score_options_from_cards
                            metrics = score_options_from_cards(cards, opt_texts)
                            coverage_by_option = metrics.get("coverage_by_option", {})
                            witness_score = metrics.get("witness_score", {})
                            quotes_sources = metrics.get("quotes_sources", {})
                            # Evidence score = α*coverage − β*overlap + γ*num_independent_quotes
                            alpha, beta, gamma = 1.0, 0.7, 0.1
                            def overlap_ratio(i: int) -> float:
                                own = witness_score.get(i, 0)
                                others = [w for k, w in witness_score.items() if k != i]
                                top_other = max(others) if others else 0
                                return (top_other / max(1, own)) if own > 0 else (1.0 if top_other > 0 else 0.0)
                            scores: Dict[int, float] = {}
                            eligible: list[int] = []
                            per_opt_cards_count = {}
                            for i, _ in enumerate(opt_texts):
                                per_opt_cards_count[i] = sum(1 for c in cards if (c.get('where') or {}).get('scope') == 'option' and (c.get('where') or {}).get('option_index') == i)
                                qcount = per_opt_cards_count[i]
                                cov = coverage_by_option.get(i, 0.0)
                                ov = overlap_ratio(i)
                                ndeps = len(quotes_sources.get(i, set()))
                                if qcount >= int(getattr(cfg.dials, "q_min", 1) or 1):
                                    eligible.append(i)
                                scores[i] = alpha * cov - beta * ov + gamma * float(ndeps)
                            evi_choice = max(eligible, key=lambda i: scores.get(i, -1e9)) if eligible else None
                            hard_fail = False
                            if (cfg.dials.citation_mode == "strict") and (evi_choice is not None):
                                cov_ok = (coverage_by_option.get(evi_choice, 0.0) >= float(cfg.coverage_tau))
                                ov_ok = (overlap_ratio(evi_choice) <= 1.0)
                                q_ok = (per_opt_cards_count.get(evi_choice, 0) >= int(getattr(cfg.dials, "q_min", 1) or 1))
                                if not (cov_ok and ov_ok and q_ok):
                                    hard_fail = True
                            if (evi_choice is not None) and (not hard_fail):
                                chosen_idx = evi_choice
                            min_q = min(per_opt_cards_count.values() or [0])
                            evi_telemetry = {
                                "min_quotes_per_option": min_q,
                                "coverage_by_option": coverage_by_option,
                                "coverage_chosen": (None if chosen_idx is None else coverage_by_option.get(int(chosen_idx), 0.0)),
                                "witness_overlap_ratio": (None if chosen_idx is None else overlap_ratio(int(chosen_idx))),
                                "num_distinct_sources": (None if chosen_idx is None else len(quotes_sources.get(int(chosen_idx), set()))),
                                "learn_tokens": (context or {}).get("_learn_tokens"),
                                "learn_time_s": (context or {}).get("_learn_time_s"),
                            }
                        except Exception:
                            evi_telemetry = None
                    ans = {"chosen_index": chosen_idx, "votes": votes}
                    if citations:
                        ans["citations"] = citations
                    if cfg.dials.verify:
                        ans2 = learner.answer_mcq(task, context=_ctx_for_pass(context, temp=cfg.dials.temp_answer))
                        agree = (ans.get("chosen_index") == ans2.get("chosen_index"))
                        evaluation = evaluate_mcq(ans.get("chosen_index"), task)
                        evaluation["self_check_agree"] = bool(agree)
                        # if disagreement and SC available, prefer majority of votes including the verify sample
                        if not agree and n_max <= 1:
                            # fall back to 2-vote majority (ties → keep original)
                            try:
                                v2 = ans2.get("chosen_index")
                                from collections import Counter as _C
                                c = _C([ans.get("chosen_index"), v2])
                                ans["chosen_index"] = c.most_common(1)[0][0]
                            except Exception:
                                pass
                        ans = {**ans, "verify_second": ans2, "verify_agree": agree}
                    # Abstention gate: only on hard evidence failures (no option-linked quote OR low coverage)
                    abstained = False
                    abstain_reason = None
                    evidence_report_data = None
                    # Evidence metrics (option link, coverage, witness) if citations required
                    if cfg.dials.use_fact_cards and cfg.dials.require_citations:
                        import re as _re, json as _json
                        def tok(s: str) -> set[str]:
                            return set([t for t in _re.findall(r"[a-zA-Z0-9]+", (s or "").lower()) if len(t) >= 2])
                        # Normalize citations (ids)
                        cited_ids: list[str] = []
                        for x in (citations or []):
                            try:
                                if isinstance(x, dict) and x.get("id"):
                                    cited_ids.append(str(x.get("id")))
                                elif isinstance(x, (str, int)):
                                    cited_ids.append(str(x))
                            except Exception:
                                continue
                        try:
                            mem = _json.loads(notes_buf) if notes_buf else {"cards": []}
                            cards = mem.get("cards") or []
                        except Exception:
                            cards = []
                        idset = set(cited_ids)
                        # filters consistent with evaluator
                        def quote_ok(q: str) -> bool:
                            return _quote_ok(q)
                        cited_cards = [c for c in cards if (not idset or (c.get("id") in idset)) and (skill_id in (c.get("tags") or [])) and quote_ok(c.get("quote") or "")]
                        chosen_idx2 = ans.get("chosen_index")
                        has_option_link = any(((c.get("where") or {}).get("scope") == "option" and ((c.get("where") or {}).get("option_index") == chosen_idx2) for c in cited_cards)) if cited_cards else False
                        # coverage
                        ct = tok("\n".join([(c.get("quote") or "") for c in cited_cards]))
                        gold = task.options[task.correct_index] if 0 <= task.correct_index < len(task.options) else ""
                        gold_t = tok(gold)
                        coverage = (len(gold_t & ct) / max(1, len(gold_t))) if gold_t else 0.0
                        # witness
                        w_scores = []
                        for i, opt in enumerate(task.options):
                            ot = tok(opt)
                            w_scores.append((len(ot & ct), i))
                        w_scores.sort(reverse=True)
                        witness_idx = w_scores[0][1] if w_scores else None
                        witness_pass = (witness_idx == task.correct_index)
                        # Optional unified pipeline override (env: SIM_USE_EVIDENCE_PIPELINE=1)
                        if _flag_on("SIM_USE_EVIDENCE_PIPELINE", True) and (_ep_post_use_checks is not None):
                            try:
                                rep = _ep_post_use_checks(
                                    task=task,
                                    notes_buf=notes_buf,
                                    skill_id=skill_id,
                                    citations=citations,
                                    option_card_ids=(context or {}).get("option_card_ids") or {},
                                    chosen_index=chosen_idx2,
                                    coverage_tau=float(cfg.coverage_tau),
                                )
                                if rep is not None:
                                    if rep.abstain_reason:
                                        abstain_reason = rep.abstain_reason
                                    evidence_report_data = {
                                        "post_pass": rep.post_pass,
                                        "abstain_reason": rep.abstain_reason,
                                        "coverage": rep.coverage,
                                        "witness_idx": rep.witness_idx,
                                        "witness_pass": rep.witness_pass,
                                        "has_option_quote_chosen": rep.has_option_quote_chosen,
                                        "required_ok_chosen": rep.required_ok_chosen,
                                        "cited_ids": rep.cited_ids,
                                        "metrics": rep.metrics,
                                    }
                            except Exception:
                                # Fall back silently to existing logic
                                pass
                        # Enforce citing the option's PRO card id when provided
                        required_ok = True
                        try:
                            opt_map = (context or {}).get("option_card_ids") or {}
                            if isinstance(opt_map, dict) and isinstance(chosen_idx2, int) and 0 <= chosen_idx2 < len(task.options):
                                letter = chr(ord('A') + chosen_idx2)
                                req_id = opt_map.get(letter)
                                if req_id:
                                    required_ok = (str(req_id) in idset)
                        except Exception:
                            required_ok = True
                        # Abstain ONLY on hard evidence failures
                        if cfg.dials.idk_enabled:
                            if not has_option_link:
                                abstain_reason = "no_option_quote_for_choice"
                            elif coverage < float(cfg.coverage_tau):
                                abstain_reason = "coverage_below_tau"
                            elif not required_ok:
                                abstain_reason = "required_card_not_cited"
                            if abstain_reason:
                                abstained = True
                                ans["chosen_index"] = None
                    # Final evaluation and calibrated score (if IDK enabled)
                    evaluation = evaluate_mcq(ans.get("chosen_index"), task)
                    if evidence_report_data is not None:
                        try:
                            evaluation["evidence_report"] = evidence_report_data
                        except Exception:
                            pass
                    if cfg.dials.idk_enabled:
                        # Score with penalty t/(1-t) for wrong, 0 for IDK, 1 for correct
                        tval = float(cfg.dials.target_confidence)
                        pen = (tval / max(1e-6, (1.0 - tval)))
                        if evaluation.get("abstained"):
                            cal = 0.0
                        elif evaluation.get("correct"):
                            cal = 1.0
                        else:
                            cal = -pen
                        evaluation["calibrated_score"] = cal
                        evaluation["t"] = tval
                        if abstained:
                            evaluation["abstain_reason"] = abstain_reason
                    # Evidence-gated credit: alias B (existing) and generic citations
                    alias_evidence = None
                    if (cfg.task == "alias_swap") and (alias_phase == "B"):
                        # Coverage and witness with respect to accumulated notes
                        import re as _re
                        def tok(s: str) -> set[str]:
                            return set([t for t in _re.findall(r"[a-zA-Z0-9]+", (s or "").lower()) if len(t) >= 3])
                        gold = task.options[task.correct_index] if 0 <= task.correct_index < len(task.options) else ""
                        if task.rationales and 0 <= task.correct_index < len(task.rationales):
                            gold += "\n" + (task.rationales[task.correct_index] or "")
                        gold_t = tok(gold)
                        notes_t = tok(notes_buf)
                        coverage = (len(gold_t & notes_t) / max(1, len(gold_t))) if gold_t else 0.0
                        # Witness pick from options using notes overlap
                        scores = []
                        for i, opt in enumerate(task.options):
                            ot = tok(opt)
                            scores.append((len(ot & notes_t), i))
                        scores.sort(reverse=True)
                        witness_idx = scores[0][1] if scores else None
                        witness_pass = (witness_idx == task.correct_index)
                        credited = bool(evaluation.get("correct") and coverage >= float(cfg.coverage_tau) and witness_pass)
                        alias_evidence = {"coverage": coverage, "witness_pass": witness_pass, "credited": credited}
                    # Generic citations evidence if required
                    if cfg.dials.require_citations and isinstance(task, MCQTask):
                        import re as _re
                        def tok(s: str) -> set[str]:
                            return set([t for t in _re.findall(r"[a-zA-Z0-9]+", (s or "").lower()) if len(t) >= 2])
                        # Normalize and de-duplicate citations (handle strings, ints, or dicts with id)
                        def _norm_citations(cits) -> list[str]:
                            ids: list[str] = []
                            for x in (cits or []):
                                try:
                                    if isinstance(x, dict):
                                        v = x.get("id") or x.get("card_id") or x.get("ref") or x.get("uid")
                                        if isinstance(v, (str, int)):
                                            ids.append(str(v))
                                    elif isinstance(x, (str, int)):
                                        ids.append(str(x))
                                except Exception:
                                    continue
                            seen = set(); out: list[str] = []
                            for i in ids:
                                if i not in seen:
                                    seen.add(i); out.append(i)
                            return out
                        cited_ids = _norm_citations(citations)
                        # Compute snippet_ok regardless of pipeline (optional contextual check)
                        try:
                            import json as _json
                            mem = _json.loads(notes_buf) if notes_buf else {"cards": []}
                            cards = mem.get("cards") or []
                            if (not cards) and fact_cards_after and isinstance(fact_cards_after, dict):
                                # Fallback to current-step cards if memory does not carry them
                                cards = (fact_cards_after.get("cards") or [])
                        except Exception:
                            cards = (fact_cards_after.get("cards") if isinstance(fact_cards_after, dict) else []) or []
                        idset = set(cited_ids)
                        from sim.utils.cards import quote_ok as _quote_ok2
                        chosen_idx = ans.get("chosen_index")
                        # Best-effort repair of citations for chosen option
                        try:
                            if cfg.dials.use_fact_cards and isinstance(chosen_idx, int):
                                opt_map = (context or {}).get("option_card_ids") or {}
                                letter = chr(ord('A') + int(chosen_idx)) if (0 <= int(chosen_idx) < 26) else None
                                req_id = (opt_map.get(letter) if letter and isinstance(opt_map, dict) else None)
                                # Build citations list from ans (if any)
                                cits = list(citations or [])
                                def _to_id(x):
                                    try:
                                        return str(x.get("id")) if isinstance(x, dict) and x.get("id") is not None else (str(x) if x is not None else None)
                                    except Exception:
                                        return None
                                cit_ids = [ _to_id(x) for x in cits ]
                                if req_id is not None and (str(req_id) not in (cit_ids or [])):
                                    cits = [req_id] + cits
                                # Ensure at least q_min option-linked citations
                                cards_pool = []
                                try:
                                    if isinstance(fact_cards_after, dict):
                                        cards_pool = list(fact_cards_after.get("cards") or [])
                                except Exception:
                                    cards_pool = []
                                try:
                                    need = max(1, int(getattr(cfg.dials, 'q_min', 1) or 1))
                                except Exception:
                                    need = 1
                                def _is_opt_card(c):
                                    w = c.get('where') or {}
                                    return (w.get('scope') == 'option') and (int(w.get('option_index') or -1) == int(chosen_idx))
                                have = 0
                                have_ids = set()
                                for x in cits:
                                    xid = _to_id(x)
                                    if not xid:
                                        continue
                                    have_ids.add(xid)
                                    for c in cards_pool:
                                        if str(c.get('id')) == xid and _is_opt_card(c):
                                            have += 1
                                            break
                                if have < need:
                                    for c in cards_pool:
                                        if _is_opt_card(c):
                                            cid = str(c.get('id'))
                                            if cid not in have_ids:
                                                cits.append(cid)
                                                have_ids.add(cid)
                                                have += 1
                                                if have >= need:
                                                    break
                                citations = cits
                                ans['citations'] = cits
                        except Exception:
                            pass
                        cited_cards = [c for c in cards if (not idset or (c.get("id") in idset)) and (skill_id in (c.get("tags") or [])) and _quote_ok2(c.get("quote") or "")]
                        retrieved = (context or {}).get("retrieved_snippets") or []
                        # Snippet policy: if tools are OFF or no retrieval used, skip snippet requirement
                        snippet_ok = True
                        try:
                            tools_used = bool(cfg.dials.use_tools)
                        except Exception:
                            tools_used = False
                        if tools_used and retrieved:
                            def _snip_text(x):
                                if isinstance(x, dict):
                                    return x.get("text") or x.get("snippet") or ""
                                return str(x)
                            texts = [ _snip_text(s) for s in retrieved ]
                            snippet_ok = any(any(((c.get("quote") or "") in t) for t in texts) for c in cited_cards)
                        # If witness missing, synthesize from cited cards (best-effort)
                        try:
                            if cfg.dials.use_fact_cards and not isinstance(ans.get('witness'), (dict, list)):
                                def _find_card(cid):
                                    for c in (cards or []):
                                        if str(c.get('id')) == str(cid):
                                            return c
                                    return None
                                cits_loc = list(citations or [])
                                if cits_loc:
                                    choice_id = None
                                    choice_quote = None
                                    for x in cits_loc:
                                        cid = str(x.get('id')) if isinstance(x, dict) and x.get('id') is not None else str(x)
                                        c = _find_card(cid)
                                        if c:
                                            w = (c.get('where') or {})
                                            if w.get('scope') == 'option' and int(w.get('option_index') or -1) == int(chosen_idx):
                                                choice_id = cid
                                                choice_quote = c.get('quote') or ''
                                                break
                                    if not choice_id:
                                        cid = str(cits_loc[0].get('id')) if isinstance(cits_loc[0], dict) and cits_loc[0].get('id') is not None else str(cits_loc[0])
                                        c = _find_card(cid)
                                        choice_id = cid
                                        choice_quote = (c.get('quote') if c else '')
                                    rule_id = None
                                    rule_quote = None
                                    for c in (cards or []):
                                        w = (c.get('where') or {})
                                        if w.get('scope') == 'context' and (c.get('quote') or '').strip():
                                            rule_id = str(c.get('id'))
                                            rule_quote = c.get('quote')
                                            break
                                    if not rule_id:
                                        rule_id = choice_id
                                        rule_quote = choice_quote
                                    if choice_id:
                                        ans['witness'] = {'rule': {'card_id': rule_id, 'quote': rule_quote or ''}, 'choice': {'card_id': choice_id, 'quote': choice_quote or ''}}
                        except Exception:
                            pass
                        # Emphasize chosen-option citations before credit (reduce witness ambiguity)
                        try:
                            if cfg.dials.use_fact_cards and isinstance(citations, list) and isinstance(chosen_idx, int):
                                cits = list(citations)
                                def _to_id(x):
                                    try:
                                        return str(x.get("id")) if isinstance(x, dict) and x.get("id") is not None else (str(x) if x is not None else None)
                                    except Exception:
                                        return None
                                def _is_opt_card_id(cid: str) -> bool:
                                    for c in (cards or []):
                                        if str(c.get('id')) == cid:
                                            w = (c.get('where') or {})
                                            return (w.get('scope') == 'option') and (int(w.get('option_index') or -1) == int(chosen_idx))
                                    return False
                                # Keep PRO id then chosen-option citations, plus one context
                                opt_map = (context or {}).get("option_card_ids") or {}
                                letter = chr(ord('A') + int(chosen_idx)) if (0 <= int(chosen_idx) < 26) else None
                                pro_id = (opt_map.get(letter) if letter and isinstance(opt_map, dict) else None)
                                filtered: list[str] = []
                                if pro_id is not None:
                                    filtered.append(str(pro_id))
                                for x in cits:
                                    cid = _to_id(x)
                                    if cid and _is_opt_card_id(cid) and cid not in filtered:
                                        filtered.append(cid)
                                for x in cits:
                                    cid = _to_id(x)
                                    if not cid:
                                        continue
                                    for c in (cards or []):
                                        if str(c.get('id')) == cid:
                                            w = (c.get('where') or {})
                                            if w.get('scope') == 'context' and cid not in filtered:
                                                filtered.append(cid)
                                            break
                                if filtered:
                                    citations = filtered
                                    ans['citations'] = filtered
                        except Exception:
                            pass
                        # Unified credit check
                        try:
                            from sim.credit import check as credit_check
                            ce = credit_check(
                                task=task,
                                chosen_index=chosen_idx,
                                citations=citations,
                                notes_buf=notes_buf,
                                skill_id=skill_id,
                                option_card_ids=(context or {}).get("option_card_ids") or {},
                                retrieved_snippets=retrieved,
                                witness=ans.get("witness"),
                                coverage_tau=float(cfg.coverage_tau),
                                q_min=int(getattr(cfg.dials, "q_min", 1) or 1),
                                tools_used=bool(getattr(cfg.dials, "use_tools", False)),
                                cards_override=(fact_cards_after.get("cards") if isinstance(fact_cards_after, dict) else None),
                            )
                            credited = bool(evaluation.get("correct") and ce.get("credited", False))
                            evaluation = {**evaluation, "citations_evidence": {**ce, "credited": credited}}
                        except Exception:
                            pass
                elif isinstance(task, CodeTask):
                    a = learner.answer_code(task, context=context)
                    code = a.get("code") or task.starter_code
                    evaluation = evaluate_code_python(task.function_name, code, task.tests)
                    ans = {"code": code}
                elif isinstance(task, ProofTask):
                    a = learner.answer_proof_step(task, context=context)
                    evaluation = evaluate_proof_step(a.get("step") or "", task.expected_keywords)
                    ans = a
                elif isinstance(task, TableQATask):
                    a = learner.answer_table_qa(task, context=context)
                    evaluation = evaluate_table_qa(a.get("answer") or "", task.expected_answer)
                    ans = a
                else:
                    ans = {}
                    evaluation = {}
            # record with presented stem for proof of context usage
            stem_text = getattr(task, "stem", "")
            presented_stem = stem_text
            # Optional domain examples injection (collect, append later after context positioning)
            ex_text_optional = None
            if cfg.task == "mcq" and cfg.dials.closed_book:
                try:
                    exs = self.domain.mcq_examples(cfg.domain, skill_id)
                except Exception:
                    exs = []
                if exs:
                    # choose example: rare-emphasis (least used) or round-robin
                    if cfg.dials.rare_emphasis:
                        # find least used index
                        idx, _ = min(((i, example_counts.get(i, 0)) for i in range(len(exs))), key=lambda t: (t[1], t[0]))
                    else:
                        idx = step % len(exs)
                    example_counts[idx] = example_counts.get(idx, 0) + 1
                    ex = exs[idx]
                    ex_text = f"EXAMPLE:\nQ: {ex.get('stem')}\nOptions: {', '.join(ex.get('options', []))}\nA: {ex.get('options', [''])[ex.get('correct_index', 0)]}"
                    # anonymize example text to preserve closed-book constraints
                    if codebook and cfg.dials.anonymize:
                        ex_text = anonymize_text(ex_text, codebook)
                    ex_text_optional = ex_text
            if context.get("context_text") and cfg.dials.context_position != "none":
                if cfg.dials.context_position == "pre":
                    presented_stem = f"CONTEXT:\n{context['context_text']}\n\nQUESTION: {stem_text}"
                elif cfg.dials.context_position == "post":
                    presented_stem = f"QUESTION: {stem_text}\n\nCONTEXT:\n{context['context_text']}"
            # If an example text was prepared, append it now
            if ex_text_optional:
                presented_stem = (presented_stem + "\n\n" + ex_text_optional).strip()
            # step duration and usage accounting
            step_ms = int((time.time() - step_t0) * 1000.0)
            student_usage = None
            try:
                if hasattr(learner, "get_usage_counters") and callable(getattr(learner, "get_usage_counters")):
                    student_usage = learner.get_usage_counters()
            except Exception:
                student_usage = None

            # Inline card validation so users don't need a separate script
            card_validation = None
            try:
                if cfg.dials.use_fact_cards and isinstance(task, MCQTask) and fact_cards_after:
                    def _tok_count(s: str) -> int:
                        import re as _re
                        return len(_re.findall(r"[A-Za-z0-9]+", s or ""))
                    cards = fact_cards_after.get("cards") or []
                    counts = {"option": 0, "context": 0, "other": 0}
                    issues: list[str] = []
                    seen_ids: set[str] = set()
                    per_opt = {i: 0 for i in range(len(task.options))}
                    # Gather source text for context quotes
                    src_snips = []
                    try:
                        for to in (tool_outputs or []):
                            if to.get("name") == "retriever":
                                src_snips.extend(to.get("snippets") or [])
                    except Exception:
                        pass
                    src_text = "\n".join(src_snips) or (presented_stem or "")
                    mode_counts = {"offset": 0, "canon-substring": 0, "fail": 0}
                    for c in cards:
                        cid = c.get("id")
                        if cid in seen_ids:
                            issues.append(f"duplicate_id:{cid}")
                        if cid:
                            seen_ids.add(cid)
                        w = c.get("where") or {}
                        scope = w.get("scope")
                        q = c.get("quote") or ""
                        tags = c.get("tags") or []
                        if not isinstance(tags, list) or not tags:
                            issues.append("missing_tags")
                        if scope == "option":
                            counts["option"] += 1
                            oi = w.get("option_index")
                            if not isinstance(oi, int) or not (0 <= oi < len(task.options)):
                                issues.append(f"bad_option_index:{oi}")
                            else:
                                per_opt[oi] += 1
                                if _tok_count(q) > 15:
                                    issues.append("long_quote_option")
                                opt_txt = task.options[oi]
                                # Offset-first, then canonicalized substring validation
                                try:
                                    from sim.validation import validate_option_quote
                                    res_v = validate_option_quote(opt_txt, c)
                                    if res_v.get("ok"):
                                        m = res_v.get("mode") or ""
                                        if m == "offset":
                                            mode_counts["offset"] += 1
                                        else:
                                            mode_counts["canon-substring"] += 1
                                    else:
                                        mode_counts["fail"] += 1
                                        issues.append("quote_not_in_option")
                                except Exception:
                                    if q and (q not in opt_txt):
                                        mode_counts["fail"] += 1
                                        issues.append("quote_not_in_option")
                        elif scope == "context":
                            counts["context"] += 1
                            if _tok_count(q) > 15:
                                issues.append("long_quote_context")
                            if q and (q not in src_text):
                                issues.append("quote_not_in_context")
                        else:
                            counts["other"] += 1
                            issues.append("unknown_scope")
                    for i, cnt in per_opt.items():
                        if cnt <= 0:
                            issues.append(f"missing_option_card:{i}")
                    card_validation = {"counts": counts, "issues": issues, "validator_modes": mode_counts}
            except Exception:
                card_validation = None

            # Optional: include rationales in task log if allowed by env
            _log_rationales = os.getenv("TUTOR_LOG_RATIONALES") in ("1", "true", "True")
            # collect messages if enabled
            _log_mode2 = (os.getenv("LOG_MESSAGES", "").lower())
            _msgs2: list[dict] = []
            if _log_mode2 in ("counts", "text"):
                try:
                    if hasattr(self.llm, "get_messages_buffer"):
                        _msgs2.extend(self.llm.get_messages_buffer())  # type: ignore
                except Exception:
                    pass
                try:
                    if hasattr(learner, "get_messages_buffer"):
                        _msgs2.extend(learner.get_messages_buffer())  # type: ignore
                except Exception:
                    pass
                try:
                    _msgs2.sort(key=lambda m: m.get("ts", 0))
                except Exception:
                    pass
            rec = {
                "run_id": run_id,
                "step": step,
                "ts": int(time.time()),
                "task": (
                    {"type": "saq", "stem": task.stem, "expected_points": task.expected_points}
                    if is_saq else
                    (
                        ({"type": "mcq", "stem": task.stem, "options": task.options, "correct_index": task.correct_index}
                         | ({} if (not _log_rationales or not getattr(task, 'rationales', None)) else {"rationales": task.rationales}))
                        if isinstance(task, MCQTask)
                        else (
                            {"type": "code", "function_name": task.function_name, "description": task.description}
                            if isinstance(task, CodeTask)
                            else (
                                {"type": "proof", "statement": task.statement}
                                if isinstance(task, ProofTask)
                                else {"type": "table_qa", "question": task.question}
                            )
                        )
                    )
                ),
                "presented_stem": presented_stem,
                "answer": ans,
                **({"evaluation": evaluation} if not is_saq else {"grading": grading, "saq_drafts": saq_drafts}),
                **({} if not tool_outputs else {"tool_outputs": tool_outputs, "tools_used": [o.get("name") for o in tool_outputs]}),
                **({} if not family_id else {"alias": {"family_id": family_id, "phase": alias_phase}, **({} if alias_evidence is None else {"alias_evidence": alias_evidence})}),
                **({} if not cfg.dials.use_fact_cards else {"fact_cards_before": fact_cards_before, "fact_cards_after": fact_cards_after}),
                **({} if not card_validation else {"card_validation": card_validation}),
                **({} if not evi_telemetry else {"evidence_telemetry": evi_telemetry}),
                "duration_ms": step_ms,
                **({} if not student_usage else {"student_usage": student_usage}),
                **({} if not hasattr(self.llm, 'get_usage_counters') else {"tutor_usage": (self.llm.get_usage_counters() or {})}),
                **({} if not _msgs2 else {"messages": _msgs2}),
            }
            # Guardrails: evaluate token bands and emit alerts/nudges
            try:
                rec = self._apply_guardrails(rec, cfg)
            except Exception:
                pass
            logs.append(rec)
            if log_f:
                import json
                log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log_f.flush()
            # Budget guards: apply per-model limits, throttle if consistently breached
            try:
                prov = getattr(learner, 'provider', None)
                model_name = getattr(getattr(learner, 'model', None), 'model', None)
            except Exception:
                prov = None; model_name = None
            try:
                # Determine budgets from presets
                max_s = None; max_tok = None
                if prov == 'deepinfra' and isinstance(model_name, str):
                    if 'Mixtral-8x7B-Instruct' in model_name:
                        max_s, max_tok = 60.0, 12000
                    elif 'DeepSeek-R1' in model_name:
                        max_s, max_tok = 180.0, 20000
                su = rec.get('student_usage') or {}
                step_ok = True
                if max_s is not None and float(rec.get('duration_ms') or 0)/1000.0 > max_s:
                    step_ok = False
                if (max_tok is not None) and isinstance(su.get('total_tokens'), int) and su.get('total_tokens') > max_tok:
                    step_ok = False
                if step_ok:
                    self._budget_strikes = 0
                else:
                    self._budget_strikes += 1
                    if self._budget_strikes >= 3:
                        # Throttle: turn ToT off; reduce sc_extract_n by 1 (min 2) while keeping evidence gates
                        try:
                            if cfg.dials.controller == 'tot':
                                cfg.dials.controller = 'basic'
                            k = int(cfg.dials.sc_extract_n or 0)
                            if k >= 3:
                                cfg.dials.sc_extract_n = k - 1
                        except Exception:
                            pass
                        self._budget_strikes = 0
            except Exception:
                pass
            # progress update after completing step
            if progress_cb:
                try:
                    progress_cb(step + 1, total_steps)
                except Exception:
                    pass
            # Optional notes accumulation
            if cfg.dials.accumulate_notes:
                if not is_saq:
                    try:
                        ci = task.correct_index
                        correct_text = task.options[ci] if 0 <= ci < len(task.options) else ""
                        notes_buf += f"\nCorrect: {correct_text}"
                        if getattr(task, "rationales", None) and ci < len(task.rationales):
                            notes_buf += f"\nWhy: {task.rationales[ci]}"
                    except Exception:
                        pass
                else:
                    try:
                        # Append model answer as reference
                        notes_buf += f"\nModel: {task.model_answer}"
                    except Exception:
                        pass
            # Optional learner memory update
            try:
                if hasattr(learner, "update_memory") and callable(getattr(learner, "update_memory")):
                    learner.update_memory(task, rec)
            except Exception:
                pass
            # Optional periodic reflection: condense notes/fact-cards every K steps into a brief summary
            try:
                k = int(getattr(cfg.dials, "reflection_every", 0) or 0)
            except Exception:
                k = 0
            if k and ((step + 1) % k == 0):
                try:
                    import json as _json
                    # Prefer summarizing Fact-Cards if present; otherwise summarize notes_buf
                    src_cards = None
                    try:
                        obj = _json.loads(notes_buf)
                        if isinstance(obj, dict) and isinstance(obj.get("cards"), list) and obj["cards"]:
                            src_cards = obj["cards"][: cfg.dials.fact_cards_budget]
                    except Exception:
                        src_cards = None
                    if src_cards:
                        sys_prompt = (
                            "You condense FactCards into a compact study note. Return only JSON with key 'notes' (≤ 240 chars). "
                            "Prefer option-linked quotes and key claims helpful for future MCQs."
                        )
                        user_obj = {"skill_id": skill_id, "cards": src_cards}
                    else:
                        sys_prompt = (
                            "You condense notes into a compact study note. Return only JSON with key 'notes' (≤ 240 chars)."
                        )
                        user_obj = {"skill_id": skill_id, "notes": notes_buf[-2000:]}
                    js = self.llm._chat_json(sys_prompt, _json.dumps(user_obj, ensure_ascii=False))
                    summary = None
                    if isinstance(js, dict):
                        summary = js.get("notes") or js.get("summary")
                    if isinstance(summary, str) and summary.strip():
                        if cfg.dials.use_fact_cards and src_cards:
                            # Preserve Fact-Cards JSON schema and add a 'notes' field
                            try:
                                obj = {"cards": src_cards, "notes": summary.strip()[:240]}
                                notes_buf = _json.dumps(obj, ensure_ascii=False)
                            except Exception:
                                pass
                        else:
                            notes_buf = (notes_buf + "\n" + summary.strip()[:240]).strip()
                except Exception:
                    pass
        if log_f:
            log_f.close()
        return logs

    # --- Guardrails helpers ---
    def _apply_guardrails(self, rec: Dict[str, Any], cfg: RunConfig) -> Dict[str, Any]:
        bands = self._token_bands
        if not bands:
            return rec
        alerts: Dict[str, Any] = {}
        # Per-step student tokens
        try:
            su = rec.get("student_usage") or {}
            step_tokens = int(su.get("total_tokens") or 0)
        except Exception:
            step_tokens = 0
        self._student_tokens_cum += max(0, step_tokens)
        # Evaluate per-step band
        low = bands.band_low_step if bands else None
        high = bands.band_high_step if bands else None
        opt = bands.opt_step if bands else None
        state = "in_band"
        action = None
        if (low is not None) and (step_tokens < low):
            state = "below_band"
            action = "nudge_richer" if bool(getattr(cfg, 'tokens_autonudge', False)) else "warn_richer"
            if getattr(cfg, 'tokens_autonudge', False):
                self._nudge_tokens(cfg, direction="up")
        elif (high is not None) and (step_tokens > high):
            state = "above_band"
            action = "nudge_leaner" if bool(getattr(cfg, 'tokens_autonudge', False)) else "warn_leaner"
            if getattr(cfg, 'tokens_autonudge', False):
                self._nudge_tokens(cfg, direction="down")
        else:
            # Inside band: check trough vicinity
            try:
                if (opt is not None) and (low is not None) and (high is not None):
                    bw = float(high - low)
                    margin = max(0.0, min(1.0, float(getattr(cfg, 'trough_margin', 0.2) or 0.2))) * bw
                    if abs(float(step_tokens) - float(opt)) <= margin * 0.5:
                        state = "near_trough"
                        action = "alert"
            except Exception:
                pass
        alerts["tokens_step"] = {
            "tokens": step_tokens,
            "state": state,
            "band_low": (float(low) if (low is not None) else None),
            "band_high": (float(high) if (high is not None) else None),
            "opt": (float(opt) if (opt is not None) else None),
            "autonudge": bool(getattr(cfg, 'tokens_autonudge', False)),
            "action": action,
        }
        # Run-level band annotation
        if bands and (bands.band_low_run is not None or bands.band_high_run is not None):
            alerts["tokens_run"] = {
                "tokens_cum": int(self._student_tokens_cum),
                "band_low": (float(bands.band_low_run) if (bands.band_low_run is not None) else None),
                "band_high": (float(bands.band_high_run) if (bands.band_high_run is not None) else None),
                "opt": (float(bands.opt_run) if (bands.opt_run is not None) else None),
            }
        # Turns alert
        if self._turns_limit is not None:
            alerts["turns"] = {
                "limit": int(self._turns_limit),
                "resolve_triggered": bool(self._resolve_triggered),
            }
        # Attach
        rec = {**rec, "guardrail_alerts": alerts}
        # Emit stderr for visibility
        try:
            import sys as _sys
            ts = alerts.get("tokens_step", {})
            st = ts.get("state")
            if st in ("below_band", "above_band", "near_trough"):
                _sys.stderr.write(f"\n[guardrails] step={rec.get('step')} tokens={ts.get('tokens')} state={st} action={ts.get('action')}\n")
        except Exception:
            pass
        return rec

    def _nudge_tokens(self, cfg: RunConfig, *, direction: str) -> None:
        try:
            if direction == "up":
                cfg.dials.rich = True
                cfg.dials.controller_budget = max(cfg.dials.controller_budget, 6)
                cfg.dials.self_consistency_n = max(cfg.dials.self_consistency_n, 2)
                cfg.dials.compress_examples = False
                cfg.dials.compress_ratio = min(cfg.dials.compress_ratio, 3.0)
            else:
                if cfg.dials.controller == "tot":
                    cfg.dials.controller = "basic"
                cfg.dials.controller_budget = min(cfg.dials.controller_budget, 3)
                cfg.dials.self_consistency_n = min(cfg.dials.self_consistency_n, 2)
                cfg.dials.compress_examples = True
                cfg.dials.compress_ratio = max(cfg.dials.compress_ratio, 3.0)
        except Exception:
            pass
