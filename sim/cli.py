from __future__ import annotations
"""Command-line entry for the ICL simulator.

Defaults are conservative (closed-book + anonymized). The CLI maps flags to a
`RunConfig` and `Dials`, then delegates to `Orchestrator.run`. Outputs are
written to JSONL when `--log` is provided and a minimal progress bar is
available via `--progress`.
"""
import argparse
import json
import sys
import time
import shutil
import os

from sim.orchestrator import Orchestrator, RunConfig, Dials
from sim.learner import LLMStudent, AlgoStudent, OracleStudent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sim", description="General-Purpose ICL Simulator")
    p.add_argument("--skill-id", default=None)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--options", type=int, default=5)
    p.add_argument("--difficulty", default="medium", choices=["easy","medium","hard"])
    p.add_argument("--task", default="mcq", choices=["mcq","saq","code","proof","table_qa"], help="task type to run")
    p.add_argument("--closed-book", action="store_true")
    p.add_argument("--no-anon", action="store_true", help="disable anonymization")
    # Accept --anonymize as a no-op flag (anonymization is on by default)
    p.add_argument("--anonymize", action="store_true", help="enable anonymization (default; use --no-anon to disable)")
    p.add_argument("--context-position", default="pre", choices=["pre","post","none"], help="place CONTEXT before or after QUESTION (lost-in-the-middle hygiene)")
    p.add_argument("--verify", action="store_true", help="self-check: second independent answer for agreement")
    p.add_argument("--rich", action="store_true")
    p.add_argument("--self-consistency", type=int, default=1, help="N votes for MCQ (max)")
    p.add_argument("--adaptive-sc", action="store_true", help="enable early-stopping self-consistency (stop when quorum reached)")
    p.add_argument("--sc-quorum", type=int, default=0, help="quorum to stop SC early (0=auto majority)")
    # Reasoning scaffolds + decoding controls
    p.add_argument("--reasoning", default="none", choices=["none","cot","ltm","tot","sot","selfdisco","got","pot"], help="reasoning scaffold")
    p.add_argument("--temperature", type=float, default=0.3, help="temperature for first-pass answers")
    p.add_argument("--temperature-sc", type=float, default=0.7, help="temperature used during self-consistency sampling")
    p.add_argument("--top-p", dest="top_p", type=float, default=1.0, help="top-p nucleus sampling (1.0 = off)")
    p.add_argument("--min-p", dest="min_p", type=float, default=None, help="min-p sampling (optional, engine-dependent)")
    # Controllers
    p.add_argument("--controller", default="basic", choices=["basic","ltm","tot","quote_then_vote"], help="reasoning controller for multi-step orchestration")
    p.add_argument("--controller-budget", type=int, default=6, help="max extra calls a controller may use")
    p.add_argument("--tot-width", type=int, default=2, help="ToT breadth")
    p.add_argument("--tot-depth", type=int, default=2, help="ToT depth")
    p.add_argument("--tot-judge", default="self", choices=["self","tutor"], help="judge for ToT (currently 'self' only)")
    # Grammar/JSON-schema decoding
    p.add_argument("--grammar", default="json", choices=["none","json","strict_json","schema"], help="constrained decoding mode")
    # SC policy (fixed or difficulty-adaptive)
    p.add_argument("--sc-policy", default="fixed", choices=["fixed","adaptive"], help="self-consistency policy")
    p.add_argument("--sc-k-easy", type=int, default=3, help="k for easy difficulty when --sc-policy=adaptive")
    p.add_argument("--sc-k-medium", type=int, default=5, help="k for medium difficulty when --sc-policy=adaptive")
    p.add_argument("--sc-k-hard", type=int, default=7, help="k for hard difficulty when --sc-policy=adaptive")
    # Best-of-N / Rerank
    p.add_argument("--best-of", dest="best_of_n", type=int, default=0, help="generate N candidates then rerank (0=off)")
    p.add_argument("--rerank", default="none", choices=["none","confidence","evidence","judge"], help="reranker for best-of")
    # Reflexion
    p.add_argument("--reflexion", action="store_true", help="enable one-pass self-critique and revision")
    # Few-shot exemplars
    p.add_argument("--shots", dest="shots_path", default=None, help="Path to exemplars JSON/JSONL for few-shot CoT/MCQ")
    p.add_argument("--shots-k", dest="shots_k", type=int, default=0, help="Max exemplars to include per item (0=off)")
    p.add_argument("--shots-selector", dest="shots_selector", default="knn", choices=["knn","random","as-is"], help="How to pick exemplars per item")
    p.add_argument("--shots-order", dest="shots_order", default="similar", choices=["similar","easy-hard","as-is"], help="How to order selected exemplars")
    p.add_argument("--shots-embed-backend", default="lexical", choices=["lexical","st","openai"], help="embedding backend for KNN")
    p.add_argument("--shots-diverse", action="store_true", help="use MMR diversification for selected exemplars")
    p.add_argument("--shots-mmr", type=float, default=0.5, help="MMR lambda (0..1) for diversity vs relevance")
    p.add_argument("--shots-reranker", default="none", choices=["none","ce"], help="optional reranker for selected exemplars")
    p.add_argument("--shots-reranker-model", default="BAAI/bge-reranker-base", help="cross-encoder model for --shots-reranker ce")
    # APE header
    p.add_argument("--ape-header", default=None, help="Path to a JSON or text file with an instruction header to prepend")
    # Compression (LLMLingua-lite)
    p.add_argument("--compress-examples", action="store_true", help="compress few-shot exemplars before sending to the model")
    p.add_argument("--compress-ratio", type=float, default=3.0, help="target compression ratio for exemplars (e.g., 3.0 ≈ keep ~1/3 tokens)")
    # Uncertainty gating
    p.add_argument("--uncertainty-gate", action="store_true", help="enable entropy/confidence gating to escalate compute")
    p.add_argument("--conf-threshold", type=float, default=0.45, help="avg confidence threshold to trigger escalation")
    p.add_argument("--entropy-threshold", type=float, default=0.90, help="normalized vote-entropy threshold (0..1) to trigger escalation")
    p.add_argument("--max-k-escalated", type=int, default=12, help="max total samples after escalation")
    p.add_argument("--escalate-reasoning", action="store_true", help="when gated, escalate reasoning to ToT for extra samples")
    p.add_argument("--sc-extract", type=int, default=0, help="N votes for Fact-Cards extraction (0=use --self-consistency)")
    p.add_argument("--reflection-every", type=int, default=0, help="every K steps, condense notes (Fact-Cards or accumulated) into a brief summary for memory")
    p.add_argument("--accumulate-notes", action="store_true")
    p.add_argument("--rare", dest="rare_emphasis", action="store_true")
    p.add_argument("--student", default="llm", choices=["llm","algo","stateful-llm","oracle"], help="student type: llm|algo|stateful-llm|oracle")
    p.add_argument("--provider", default="openai", choices=["openai","deepinfra","deepseek"], help="LLM provider for LLM students")
    p.add_argument("--model", default=None, help="Model name for provider (e.g., deepseek via DeepInfra)")
    p.add_argument("--notes-file", default=None)
    p.add_argument("--cards", dest="cards_path", default=None, help="Path to Fact-Cards JSON/JSONL to use as persistent context")
    p.add_argument("--cards-freeze", action="store_true", help="Use provided Fact-Cards as-is (disable LEARN updates)")
    p.add_argument("--log", dest="log_path", default=None, help="path to JSONL log file")
    p.add_argument("--use-tools", action="store_true")
    p.add_argument("--tools", default="retriever", help="comma-separated tool names (e.g., retriever,tfidf_retriever,option_retriever)")
    p.add_argument("--domain", default="general")
    # Bayesian guardrails (optional)
    p.add_argument("--guardrails", dest="guardrails_path", default=os.getenv("BAYES_GUARDRAILS"), help="Path to guardrails.json (token bands)")
    p.add_argument("--talk-slopes", dest="talk_slopes_path", default=os.getenv("BAYES_TALK_SLOPES"), help="Path to talk_slopes_by_domain CSV")
    p.add_argument("--turns-limit", dest="turns_limit", type=int, default=int(os.getenv("SESSION_TURNS_LIMIT", "0") or 0), help="Max turns before resolve-or-escalate (0=auto from guardrails)")
    p.add_argument("--tokens-autonudge", dest="tokens_autonudge", action="store_true", help="Auto-adjust dials to push tokens into band")
    p.add_argument("--trough-margin", dest="trough_margin", type=float, default=float(os.getenv("TROUGH_MARGIN", "0.2") or 0.2), help="Fraction of band range around opt treated as trough vicinity for alerts")
    p.add_argument("--talk-ppos-threshold", dest="talk_ppos_threshold", type=float, default=float(os.getenv("TALK_PPOS_THRESHOLD", "0.7") or 0.7), help="Probability threshold for positive talk slope to pick rich vs lean talk")
    p.add_argument("--use-examples", action="store_true", help="[experimental] may be ignored; use --shots/--shots-k for examples")
    p.add_argument("--progress", action="store_true", help="show a live progress bar (stderr)")
    # Fact-Cards ICL controls
    p.add_argument("--fact-cards", action="store_true", help="enable Fact-Cards two-pass (LEARN/USE)")
    # Evidence gating & retrieval params
    p.add_argument("--q-min", dest="q_min", type=int, default=1, help="min required quotes per option (option-linked)")
    p.add_argument("--coverage-tau", dest="coverage_tau", type=float, default=None, help="coverage threshold τ for evidence gating (overrides env)")
    p.add_argument("--max-learn-boosts", type=int, default=0, help="max LEARN escalation rounds when gates fail")
    p.add_argument("--mmr", dest="mmr_lambda", type=float, default=0.4, help="MMR lambda (0..1) for retrieval diversity (option retriever)")
    # --min-sources-chosen removed from CLI; may be controlled via code presets later
    p.add_argument("--dedup-sim", dest="dedup_sim", type=float, default=0.88, help="near-duplicate threshold for quotes/cards (Jaccard over tokens)")
    # Card quality & selection dials removed from CLI until fully wired:
    # --min-cqs, --min-card-len, --max-card-len, --per-option-top-k
    p.add_argument("--evidence-weighted-selection", action="store_true", help="[experimental] opt-in evidence rerank; may reduce accuracy")
    p.add_argument("--citations", default=None, choices=[None, "off", "lenient", "strict"], help="citation gating mode (overrides --require-citations)")
    p.add_argument("--cards-budget", type=int, default=10, help="max number of fact cards to keep")
    p.add_argument("--require-citations", action="store_true", help="require citations for credit (MCQ)")
    # Abstention + calibration
    p.add_argument("--idk", dest="idk_enabled", action="store_true", help="enable abstention: allow 'IDK' when below confidence/evidence")
    p.add_argument("--target-confidence", type=float, default=0.75, help="confidence threshold t for answering (0..1)")
    # Misc utility flags
    p.add_argument("--health", action="store_true", help="verify API connectivity for tutor/student and exit")
    p.add_argument("--summary", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--print-results", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    # Env overrides for commonly toggled dials
    env_anonymize = os.getenv("TUTOR_ANONYMIZE")
    anonymize = (not args.no_anon)
    if env_anonymize is not None:
        anonymize = (env_anonymize == "1" or env_anonymize.lower() in ("true","yes","on"))

    env_require_citations = os.getenv("TUTOR_REQUIRE_CITATIONS")
    # Determine citation mode
    citation_mode = None
    if args.citations is not None:
        citation_mode = args.citations
    elif env_require_citations is not None:
        citation_mode = "strict" if (env_require_citations == "1" or env_require_citations.lower() in ("true","yes","on")) else "off"
    # Back-compat: --require-citations toggles on when citations mode unspecified
    require_citations = args.require_citations or (citation_mode in ("lenient","strict"))

    # Load APE header (path or inline text)
    instruction_header = None
    if args.ape_header:
        if os.path.isfile(args.ape_header):
            try:
                with open(args.ape_header, "r", encoding="utf-8") as f:
                    txt = f.read().strip()
                import json as _json
                try:
                    obj = _json.loads(txt)
                    if isinstance(obj, dict) and obj.get("header"):
                        instruction_header = str(obj.get("header")).strip()
                    else:
                        instruction_header = txt
                except Exception:
                    instruction_header = txt
            except Exception:
                instruction_header = None
        else:
            instruction_header = args.ape_header.strip()

    dials = Dials(
        closed_book=args.closed_book,
        anonymize=anonymize,
        rich=args.rich,
        verify=args.verify,
        context_position=args.context_position,
        self_consistency_n=args.self_consistency,
        adaptive_sc=bool(args.adaptive_sc),
        sc_quorum=int(args.sc_quorum or 0),
        reasoning=args.reasoning,
        temp_answer=float(args.temperature or 0.3),
        temp_sc=float(args.temperature_sc or 0.7),
        top_p=float(args.top_p if args.top_p is not None else 1.0),
        min_p=(float(args.min_p) if args.min_p is not None else None),
        grammar=args.grammar,
        controller=args.controller,
        controller_budget=int(args.controller_budget or 6),
        tot_width=int(args.tot_width or 2),
        tot_depth=int(args.tot_depth or 2),
        tot_judge=args.tot_judge,
        instruction_header=instruction_header,
        sc_policy=args.sc_policy,
        sc_k_easy=int(args.sc_k_easy or 3),
        sc_k_medium=int(args.sc_k_medium or 5),
        sc_k_hard=int(args.sc_k_hard or 7),
        best_of_n=int(args.best_of_n or 0),
        rerank=args.rerank,
        reflexion=bool(args.reflexion),
        compress_examples=bool(args.compress_examples),
        compress_ratio=float(args.compress_ratio or 3.0),
        uncertainty_gate=bool(args.uncertainty_gate),
        conf_threshold=float(args.conf_threshold or 0.45),
        entropy_threshold=float(args.entropy_threshold or 0.90),
        max_k_escalated=int(args.max_k_escalated or 12),
        escalate_reasoning=bool(args.escalate_reasoning),
        sc_extract_n=max(0, int(args.sc_extract or 0)),
        q_min=max(1, int(args.q_min or 1)),
        max_learn_boosts=max(0, int(args.max_learn_boosts or 0)),
        mmr_lambda=float(args.mmr_lambda or 0.4),
        # span_window deprecated in CLI; keep default in Dials
        citation_mode=(citation_mode or ("strict" if require_citations else "off")),
        min_sources_chosen=max(1, int(getattr(args, 'min_sources_chosen', 2) or 2)),
        dedup_sim=float(getattr(args, 'dedup_sim', 0.88) or 0.88),
        # Card quality dials
        min_cqs=float(getattr(args, 'min_cqs', 0.0) or 0.0),
        min_card_len=int(getattr(args, 'min_card_len', 40) or 40),
        max_card_len=int(getattr(args, 'max_card_len', 300) or 300),
        per_option_top_k=int(getattr(args, 'per_option_top_k', 3) or 3),
        evidence_weighted_selection=bool(getattr(args, 'evidence_weighted_selection', False)),
        reflection_every=max(0, int(args.reflection_every or 0)),
        accumulate_notes=args.accumulate_notes,
        rare_emphasis=args.rare_emphasis,
        use_tools=args.use_tools,
        tools=[t.strip() for t in (args.tools or "").split(",") if t.strip()],
        use_fact_cards=args.fact_cards,
        fact_cards_budget=args.cards_budget,
        require_citations=require_citations,
        freeze_cards=args.cards_freeze,
        idk_enabled=args.idk_enabled,
        target_confidence=float(args.target_confidence or 0.75),
    )
    cfg = RunConfig(
        skill_id=args.skill_id,
        task=args.task,
        num_steps=args.steps,
        num_options=args.options,
        difficulty=args.difficulty,
        dials=dials,
        domain=args.domain,
        guardrails_path=args.guardrails_path,
        talk_slopes_path=args.talk_slopes_path,
        turns_limit=(None if int(args.turns_limit or 0) <= 0 else int(args.turns_limit)),
        tokens_autonudge=bool(args.tokens_autonudge),
        trough_margin=float(args.trough_margin or 0.2),
        talk_ppos_threshold=float(args.talk_ppos_threshold or 0.7),
        shots_path=args.shots_path,
        shots_k=int(args.shots_k or 0),
        shots_selector=args.shots_selector,
        shots_order=args.shots_order,
        shots_embed_backend=args.shots_embed_backend,
        shots_diverse=bool(args.shots_diverse),
        shots_mmr=float(args.shots_mmr or 0.5),
        # Reranker config (carried in RunConfig via dials or used in orchestrator branch)
    )
    # Optional coverage threshold via CLI first, then env var (TUTOR_COVERAGE_TAU)
    if args.coverage_tau is not None:
        try:
            cfg.coverage_tau = float(args.coverage_tau)
        except Exception:
            pass
    else:
        env_tau = os.getenv("TUTOR_COVERAGE_TAU")
        if env_tau:
            try:
                cfg.coverage_tau = float(env_tau)
            except Exception:
                pass
    orch = Orchestrator()
    if args.student == "llm":
        learner = LLMStudent(provider=args.provider, model=args.model)
    elif args.student == "stateful-llm":
        from sim.learner import StatefulLLMStudent
        learner = StatefulLLMStudent(provider=args.provider, model=args.model)
    elif args.student == "oracle":
        learner = OracleStudent()
    else:
        learner = AlgoStudent()

    # Optional: quick health check for keys/connectivity
    argv2 = argv if argv is not None else sys.argv[1:]
    if "--health" in (argv2 or []):
        try:
            tutor_health = orch.llm.verify_key_and_model() if hasattr(orch.llm, "verify_key_and_model") else {"ok": True}
        except Exception as e:
            tutor_health = {"ok": False, "error": str(e)}
        try:
            m = getattr(learner, "model", None)
            student_health = m.verify_key_and_model() if hasattr(m, "verify_key_and_model") else {"ok": True}
        except Exception as e:
            student_health = {"ok": False, "error": str(e)}
        print(json.dumps({"health": {"tutor": tutor_health, "student": student_health}}, ensure_ascii=False))
        return 0
    notes = ""
    if args.notes_file:
        try:
            with open(args.notes_file, "r", encoding="utf-8") as f:
                notes = f.read()
        except Exception:
            notes = ""
    # Optional Fact-Cards dataset ingestion
    if args.cards_path:
        def _load_cards(path: str) -> str:
            import json
            try:
                # Try JSONL first: aggregate cards/text across lines
                cards: list[dict] = []
                texts: list[str] = []
                with open(path, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                # Detect if single JSON object file
                if len(lines) == 1 and (lines[0].startswith("{") or lines[0].startswith("[")):
                    obj = json.loads(lines[0])
                    if isinstance(obj, dict) and isinstance(obj.get("cards"), list):
                        return json.dumps({"cards": obj.get("cards")}, ensure_ascii=False)
                    if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                        return obj.get("text")
                    if isinstance(obj, list):
                        # list of cards
                        return json.dumps({"cards": obj}, ensure_ascii=False)
                # JSONL or mixed
                for ln in lines:
                    try:
                        rec = json.loads(ln)
                        if isinstance(rec, dict):
                            if isinstance(rec.get("cards"), list):
                                cards.extend(rec.get("cards") or [])
                            elif isinstance(rec.get("text"), str):
                                texts.append(rec.get("text") or "")
                            elif isinstance(rec.get("notes"), str):
                                texts.append(rec.get("notes") or "")
                    except Exception:
                        # fallback: treat as plain text line
                        texts.append(ln)
                if cards:
                    return json.dumps({"cards": cards}, ensure_ascii=False)
                return "\n".join(texts)
            except Exception:
                # Final fallback: raw file contents
                try:
                    return open(path, "r", encoding="utf-8").read()
                except Exception:
                    return ""
        notes = _load_cards(args.cards_path) or notes
        # If user freezes cards, ensure we actually have a cards array
        if args.cards_freeze:
            import json as _json
            try:
                obj = _json.loads(notes) if notes.strip() else {}
            except Exception:
                obj = {}
            if not (isinstance(obj, dict) and isinstance(obj.get("cards"), list) and obj["cards"]):
                sys.stderr.write("[warn] --cards-freeze: no cards provided; using per-option PRO stubs only.\n")
    progress_cb = None
    if args.progress:
        # lightweight stderr progress bar without extra deps
        def _make_progress_cb():
            start = time.time()
            last_len = 0

            def _cb(done: int, total: int) -> None:
                nonlocal last_len
                try:
                    total = int(total or 0)
                    done = int(done or 0)
                except Exception:
                    return
                total = max(0, total)
                done = max(0, min(done, total))
                cols = shutil.get_terminal_size((80, 24)).columns
                bar_width = max(10, min(40, cols - 30))
                ratio = 0.0 if total == 0 else (done / total)
                ratio = max(0.0, min(1.0, ratio))
                filled = int(bar_width * ratio)
                bar = "[" + ("#" * filled) + ("-" * (bar_width - filled)) + "]"
                pct = int(ratio * 100)
                msg = f"{bar} {done}/{total} {pct}%"
                sys.stderr.write("\r" + msg + (" " * max(0, last_len - len(msg))))
                sys.stderr.flush()
                last_len = len(msg)
                if total > 0 and done >= total:
                    dur = time.time() - start
                    sys.stderr.write(f"\r{msg} in {dur:.1f}s\n")
                    sys.stderr.flush()

            return _cb

        progress_cb = _make_progress_cb()

    # Normalize grammar alias
    if dials.grammar == "strict_json":
        dials.grammar = "json"
    # For MCQ with Fact-Cards, force JSON grammar (schema cannot carry citations)
    try:
        if (args.task == "mcq") and dials.use_fact_cards and dials.grammar != "json":
            dials.grammar = "json"
    except Exception:
        pass
    logs = orch.run(learner, cfg, notes_text=notes, log_path=args.log_path, progress_cb=progress_cb)
    # Optional compact stdout JSON summary
    def _emit_summary(records: list[dict]) -> dict:
        total = len(records)
        correct = 0
        credited = 0
        abstained = 0
        tutor_tok = 0
        student_tok = 0
        dur_ms = 0
        for r in records:
            ev = r.get("evaluation") or {}
            if isinstance(ev, dict):
                correct += 1 if ev.get("correct") else 0
                abstained += 1 if ev.get("abstained") else 0
                ce = ev.get("citations_evidence") or {}
                credited += 1 if ce.get("credited") else 0
            tutor_tok += int(((r.get("tutor_usage") or {}).get("total_tokens") or 0) or 0)
            student_tok += int(((r.get("student_usage") or {}).get("total_tokens") or 0) or 0)
            dur_ms += int(r.get("duration_ms") or 0)
        return {
            "steps": total,
            "correct": correct,
            "credited": credited,
            "abstained": abstained,
            "tutor_tokens": tutor_tok,
            "student_tokens": student_tok,
            "duration_ms": dur_ms,
        }

    emit_mode = os.getenv("SIM_EMIT_JSON", "auto").lower()
    want_summary = ("--summary" in (argv2 or []))
    want_results = ("--print-results" in (argv2 or []))
    if want_results:
        print(json.dumps({"results": logs}, ensure_ascii=False))
    elif want_summary or (emit_mode == "summary") or (not args.log_path and emit_mode in ("auto", "on")):
        print(json.dumps({"summary": _emit_summary(logs)}, ensure_ascii=False))
    # Else: stay quiet when logging to file and not asked explicitly
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
