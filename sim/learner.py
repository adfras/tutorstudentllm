from __future__ import annotations
from typing import Any, Dict, List, Optional

from sim.tasks import MCQTask, SAQTask
from sim.tasks import CodeTask, ProofTask, TableQATask
from sim.prompts_mcq import fact_card_prompt_components


class Learner:
    def answer_mcq(self, task: MCQTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError
    def answer_saq(self, task: SAQTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError
    def answer_code(self, task: CodeTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError
    def answer_proof_step(self, task: ProofTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError
    def answer_table_qa(self, task: TableQATask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError
    def update_memory(self, *args, **kwargs) -> None:
        # Optional hook for stateful learners
        return None
    # Optional: extract fact-cards from a source text
    def extract_fact_cards(self, task, source_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError


class LLMStudent(Learner):
    def __init__(self, provider: str = "openai", model: str | None = None):
        self.provider = provider
        if provider == "openai":
            # Always use the student wrapper (strips unsupported decode args on gpt-4.1, etc.)
            from tutor.llm_openai import OpenAIStudentLLM
            self.model = OpenAIStudentLLM(model=model)
        elif provider in ("deepinfra", "deepseek"):
            # DeepSeek via DeepInfra OpenAI-compatible endpoint
            from tutor.llm_deepinfra import DeepInfraLLM
            self.model = DeepInfraLLM(model=model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        # Usage accounting per step
        self._calls_step = 0
        self._prompt_tokens_step = 0
        self._completion_tokens_step = 0
        self._total_tokens_step = 0
        self._latency_ms_step = 0.0

    # ---- Usage helpers ----
    def reset_usage_counters(self):  # called per step
        self._calls_step = 0
        self._prompt_tokens_step = 0
        self._completion_tokens_step = 0
        self._total_tokens_step = 0
        self._latency_ms_step = 0.0

    def _bump_usage(self, js: dict | None):
        try:
            if not isinstance(js, dict):
                return
            u = js.get("_usage") or {}
            pt = int(u.get("prompt_tokens") or 0)
            ct = int(u.get("completion_tokens") or 0)
            tt = int(u.get("total_tokens") or (pt + ct))
            ms = float(js.get("_request_ms") or 0.0)
            self._calls_step += 1
            self._prompt_tokens_step += pt
            self._completion_tokens_step += ct
            self._total_tokens_step += tt
            self._latency_ms_step += ms
        except Exception:
            return

    def get_usage_counters(self) -> Dict[str, float | int]:
        return {
            "calls": int(self._calls_step),
            "prompt_tokens": int(self._prompt_tokens_step),
            "completion_tokens": int(self._completion_tokens_step),
            "total_tokens": int(self._total_tokens_step),
            "request_ms_sum": float(self._latency_ms_step),
        }

    # Optional messages buffer (delegates to underlying model if available)
    def reset_messages_buffer(self) -> None:
        try:
            if hasattr(self.model, "reset_messages_buffer"):
                self.model.reset_messages_buffer()
        except Exception:
            return

    def get_messages_buffer(self) -> List[dict]:
        try:
            if hasattr(self.model, "get_messages_buffer"):
                return list(self.model.get_messages_buffer())  # type: ignore
        except Exception:
            return []
        return []

    def answer_mcq(self, task: MCQTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Include provided context in the stem (closed-book enforcement via prompt)
        stem = task.stem
        if context and context.get("fact_cards"):
            system, payload_str, decode_override = fact_card_prompt_components(task, context or {})
            decode = dict((context or {}).get("decode") or {})
            if decode_override:
                decode.update(decode_override)
            js = self.model._chat_json_opts(system, payload_str, **decode)
            self._bump_usage(js)
            # Parse choice and citations robustly
            def _to_letter(x: Any) -> str:
                try:
                    if isinstance(x, str):
                        return x.strip().upper()
                    if isinstance(x, int) and 0 <= x < 26:
                        return chr(ord('A') + x)
                    if isinstance(x, dict):
                        for k in ("id", "letter", "choice", "value"):
                            v = x.get(k)
                            if isinstance(v, str):
                                return v.strip().upper()
                        idx = x.get("index") or x.get("chosen_index")
                        if isinstance(idx, int) and 0 <= idx < 26:
                            return chr(ord('A') + idx)
                except Exception:
                    pass
                return ""
            # Detect abstention via explicit 'IDK'
            if isinstance(js.get("choice"), str) and js.get("choice").strip().upper() == "IDK":
                return {"chosen_index": None, "citations": []}
            letter_to_idx = {ch: i for i, ch in enumerate(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))}
            choice_letter = _to_letter(js.get("choice")) or _to_letter(js.get("letter"))
            chosen = letter_to_idx.get(choice_letter)
            if chosen is None:
                for k in ("chosen_index", "index"):
                    v = js.get(k)
                    if isinstance(v, int):
                        chosen = v
                        break
            if chosen is None:
                chosen = 0 if task.options else None
            # Attempt to collect citations for the chosen option
            citations: List[Any] = []
            try:
                opts = js.get("options") or []
                # If we don't have a letter yet, derive from chosen index
                if not choice_letter and isinstance(chosen, int) and 0 <= chosen < 26:
                    choice_letter = chr(ord('A') + chosen)
                for opt in opts:
                    oid = _to_letter(opt.get("id"))
                    if oid and oid == (choice_letter or "") and isinstance(opt.get("citations"), list):
                        citations = opt.get("citations") or []
                        break
                if not citations:
                    # fallback to top-level citations
                    top = js.get("citations")
                    if isinstance(top, list):
                        citations = top
            except Exception:
                top = js.get("citations")
                if isinstance(top, list):
                    citations = top
            out: Dict[str, Any] = {"chosen_index": chosen, "citations": citations}
            # Surface witness block if present for downstream evaluators
            try:
                if isinstance(js.get("witness"), dict) or isinstance(js.get("witness"), list):
                    out["witness"] = js.get("witness")
            except Exception:
                pass
            return out
        if context and context.get("context_text"):
            stem = f"CONTEXT:\n{context['context_text']}\n\nQUESTION: {task.stem}"
        # Program-of-Thought with safe execution: if requested, ask for a small program first
        if (context or {}).get("reasoning") == "pot" and (context or {}).get("use_pyexec", True):
            import json
            header = (context or {}).get("instruction_header")
            system = (
                "Write a tiny Python program to solve the multiple-choice question. "
                "Return only JSON {program:string}. Program must be pure Python, no imports/attributes; keep it short. "
                "The program should set a variable named result to the final numeric/string answer."
            )
            if header:
                system = header.strip() + " " + system
            payload = {"stem": stem, "options": task.options, **({"examples": (context or {}).get("examples")} if (context or {}).get("examples") else {})}
            decode = (context or {}).get("decode") or {}
            js_prog = self.model._chat_json_opts(system, json.dumps(payload, ensure_ascii=False), **decode)
            code = (js_prog or {}).get("program") or (js_prog or {}).get("code") or ""
            if code:
                try:
                    from sim.tools import SafePythonExecutor
                    exe = SafePythonExecutor()
                    out = exe.run(code=code, inputs={})
                    if out.get("ok"):
                        val = out.get("result")
                        # Map result to an option if possible
                        def _num(x):
                            try:
                                return float(str(x).strip())
                            except Exception:
                                return None
                        vnum = _num(val)
                        choice = None
                        if vnum is not None:
                            best = (1e18, None)
                            for i, opt in enumerate(task.options):
                                on = _num(opt)
                                if on is not None:
                                    err = abs(on - vnum)
                                    if err < best[0]:
                                        best = (err, i)
                            if best[1] is not None:
                                choice = best[1]
                        else:
                            # exact string match after stripping
                            sval = str(val).strip().lower()
                            for i, opt in enumerate(task.options):
                                if str(opt).strip().lower() == sval:
                                    choice = i; break
                        if choice is not None:
                            return {"chosen_index": choice, "raw": {"chosen_index": choice, "confidence": 0.8, "pot": True}}
                except Exception:
                    pass
        # Build JSON-only system with optional scaffolds, few-shot examples, and decoding controls
        reasoning = (context or {}).get("reasoning")
        scaffold = ""
        if reasoning == "cot":
            scaffold = " Think step-by-step internally."
        elif reasoning == "ltm":
            scaffold = " Internally break into minimal substeps and solve in order."
        elif reasoning == "tot":
            scaffold = " Propose two brief approaches and simulate 1–2 steps internally; continue with the better."
        elif reasoning == "sot":
            scaffold = " Outline a short internal skeleton, then fill it."
        elif reasoning == "selfdisco":
            scaffold = " Compose an internal plan by composing reasoning modules (e.g., step-by-step, compare-contrast)."
        elif reasoning == "got":
            scaffold = " Consider an internal DAG of subproblems and resolve dependencies succinctly."
        elif reasoning == "pot":
            scaffold = " Sketch internal pseudo-code steps (no execution), then map them to a choice."
        header = (context or {}).get("instruction_header")
        system = (
            "You are answering a multiple-choice question. Use provided examples if any as demonstrations. "
            "Return only JSON with keys: chosen_index (int), confidence (0..1)." + scaffold + " Do not include rationales in the output."
        )
        if header:
            system = header.strip() + " " + system
        import json
        payload = {"stem": stem, "options": task.options}
        if (context or {}).get("examples"):
            payload["examples"] = (context or {}).get("examples")
        if (context or {}).get("controller_hint"):
            payload["controller_hint"] = (context or {}).get("controller_hint")
        if (context or {}).get("controller_plan"):
            payload["controller_plan"] = (context or {}).get("controller_plan")
        decode = (context or {}).get("decode") or {}
        ans = self.model._chat_json_opts(system, json.dumps(payload, ensure_ascii=False), **decode)
        self._bump_usage(ans)
        chosen = ans.get("chosen_index")
        # Robust fallback: accept letter keys or default to 0
        if chosen is None:
            # Attempt to decode letter/choice from various fields, including raw text
            def _to_letter2(x: Any) -> str:
                if isinstance(x, str):
                    return x.strip().upper()
                if isinstance(x, int) and 0 <= x < 26:
                    return chr(ord('A') + x)
                if isinstance(x, dict):
                    v = x.get("id") or x.get("letter") or x.get("choice")
                    if isinstance(v, str):
                        return v.strip().upper()
                    idx = x.get("index") or x.get("chosen_index")
                    if isinstance(idx, int) and 0 <= idx < 26:
                        return chr(ord('A') + idx)
                return ""
            letter = _to_letter2(ans.get("letter")) or _to_letter2(ans.get("choice"))
            if (not letter) and ans.get("text"):
                import re as _re
                m = _re.search(r"\b([A-H])\b", ans.get("text").upper())
                if m:
                    letter = m.group(1)
            if letter in ("A","B","C","D","E","F","G","H"):
                chosen = ord(letter) - ord('A')
            else:
                chosen = 0 if task.options else None
        return {"chosen_index": chosen, "raw": ans}

    def answer_saq(self, task: SAQTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # If mocked, emit an answer that includes a key term to satisfy mock grading
        if getattr(self.model, "_mock", False):
            key = task.expected_points[0]["key"] if task.expected_points else "answer"
            return {"student_answer": f"This references {key}."}
        # Otherwise call the model in JSON mode for a concise answer
        import json
        reasoning = (context or {}).get("reasoning")
        scaffold = ""
        if reasoning == "cot":
            scaffold = " Think step-by-step internally."
        elif reasoning == "ltm":
            scaffold = " Break into minimal numbered subproblems internally, then solve them in order."
        elif reasoning == "tot":
            scaffold = " Consider two brief approaches internally and pick the better."
        elif reasoning == "sot":
            scaffold = " Draft a short internal outline, then write the answer."
        elif reasoning == "selfdisco":
            scaffold = " Compose an internal plan combining reasoning modules."
        elif reasoning == "got":
            scaffold = " Consider an internal DAG of subproblems."
        elif reasoning == "pot":
            scaffold = " Sketch internal pseudo-code steps before writing the answer."
        header = (context or {}).get("instruction_header")
        system = "You are answering a short-answer question. Return only JSON with: student_answer (string)." + scaffold
        if header:
            system = header.strip() + " " + system
        payload = {
            "stem": task.stem,
            "context": (context or {}).get("context_text") or "",
            **({"examples": (context or {}).get("examples")} if (context or {}).get("examples") else {}),
        }
        js = self.model._chat_json(system, json.dumps(payload, ensure_ascii=False))
        return {"student_answer": js.get("student_answer") or ""}

    def answer_code(self, task: CodeTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Mock path: if function is 'add', return a correct implementation
        if getattr(self.model, "_mock", False):
            if task.function_name == "add":
                return {"code": "def add(a,b):\n    return a+b\n"}
            return {"code": task.starter_code or ""}
        # Non-mock: ask model to produce code in JSON
        import json
        header = (context or {}).get("instruction_header")
        system = "Return only JSON with: code (string of Python function)."
        if header:
            system = header.strip() + " " + system
        payload = {
            "description": task.description,
            "function_name": task.function_name,
            "starter_code": task.starter_code,
            "context": (context or {}).get("context_text") or "",
        }
        js = self.model._chat_json(system, json.dumps(payload, ensure_ascii=False))
        return {"code": js.get("code") or (task.starter_code or "")}

    def answer_proof_step(self, task: ProofTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if getattr(self.model, "_mock", False):
            # Include expected keyword to satisfy evaluator
            kw = task.expected_keywords[0] if task.expected_keywords else ""
            return {"step": f"By {kw}, it follows."}
        import json
        header = (context or {}).get("instruction_header")
        system = "Return only JSON with: step (string) as a single proof step." \
                 "Keep short and focus on the required theorem/keyword."
        if header:
            system = header.strip() + " " + system
        payload = {"statement": task.statement, "context": (context or {}).get("context_text") or ""}
        js = self.model._chat_json(system, json.dumps(payload, ensure_ascii=False))
        return {"step": js.get("step") or ""}

    def answer_table_qa(self, task: TableQATask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if getattr(self.model, "_mock", False):
            return {"answer": task.expected_answer}
        import json, csv, io
        # Optional PoT with safe execution: provide 'rows' to the program
        if (context or {}).get("reasoning") == "pot" and (context or {}).get("use_pyexec", True):
            rows = list(csv.DictReader(io.StringIO(task.csv))) if task.csv else []
            system = (
                "Write a tiny Python program to answer the table question using the provided variable 'rows' (list of dicts). "
                "Return only JSON {program:string}. Program must be pure Python, no imports/attributes; keep it short. "
                "Set a variable named result to the final string answer."
            )
            payload = {"question": task.question}
            decode = (context or {}).get("decode") or {}
            js_prog = self.model._chat_json_opts(system, json.dumps(payload, ensure_ascii=False), **decode)
            code = (js_prog or {}).get("program") or (js_prog or {}).get("code") or ""
            if code:
                try:
                    from sim.tools import SafePythonExecutor
                    exe = SafePythonExecutor()
                    out = exe.run(code=code, inputs={"rows": rows})
                    if out.get("ok"):
                        val = out.get("result")
                        return {"answer": str(val) if val is not None else ""}
                except Exception:
                    pass
        header = (context or {}).get("instruction_header")
        system = "Return only JSON with: answer (string) to the table question."
        if header:
            system = header.strip() + " " + system
        payload = {"csv": task.csv, "question": task.question, "context": (context or {}).get("context_text") or ""}
        js = self.model._chat_json(system, json.dumps(payload, ensure_ascii=False))
        self._bump_usage(js)
        return {"answer": js.get("answer") or ""}

    def extract_fact_cards(self, task, source_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Fact-Cards v2 LEARN: option-linked, discriminative cards with quotes
        import json
        header = (context or {}).get("instruction_header")
        try:
            qmin = max(1, int((context or {}).get("q_min") or 1))
        except Exception:
            qmin = 1
        system = (
            "Extract discriminative FactCards. Return strictly JSON with key 'cards' as an array of objects: "
            "{id, claim, quote, where:{scope:'option'|'context', option_index?, start, end, source_id}, tags, hypothesis, polarity}. "
            "Rules: (1) quote is a verbatim substring (≤ 15 tokens) from an OPTION or CONTEXT; (2) include provided skill_id in tags; "
            f"(3) include at least {qmin} PRO card(s) for EACH option: for every option i, add ≥{qmin} cards with where.scope='option', where.option_index=i, and quotes taken from that option’s text; "
            "(4) if retrieved_snippets is non-empty, include at least one card that quotes from it and set where.source_id to the provided source id; for option quotes set where.source_id='option:<i>'; "
            "(5) Add 1–2 CON cards for the most confusable distractors (short quotes from those option texts showing why they don’t fit)."
        )
        if header:
            system = header.strip() + " " + system
        payload = {
            "skill_id": (context or {}).get("skill_id"),
            "stem": getattr(task, 'stem', ''),
            "options": getattr(task, 'options', []),
            "context": source_text,
            "retrieved_snippets": (context or {}).get("retrieved_snippets") or [],
            "retrieved_sources": (context or {}).get("retrieved_sources") or {},
        }
        js = self.model._chat_json(system, json.dumps(payload, ensure_ascii=False))
        self._bump_usage(js)
        out = {"cards": []}
        if isinstance(js, dict) and isinstance(js.get("cards"), list):
            cards = []
            for i, c in enumerate(js["cards"], 1):
                try:
                    card = {
                        "id": c.get("id") or f"f{i}",
                        "claim": c.get("claim") or "",
                        "quote": c.get("quote") or "",
                        "where": c.get("where") or {},
                        "tags": c.get("tags") or [],
                        "hypothesis": c.get("hypothesis") or "",
                        "polarity": c.get("polarity") or "pro",
                    }
                    # ensure skill tag present
                    sid = (context or {}).get("skill_id")
                    if sid and sid not in card["tags"]:
                        card["tags"].append(sid)
                    cards.append(card)
                except Exception:
                    continue
            out["cards"] = cards
        return out


class AlgoStudent(Learner):
    """Closed-book algorithmic baseline: choose option with maximum overlap with NOTES text.
    context may contain {'notes_text': str}.
    """

    def answer_mcq(self, task: MCQTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        import re
        notes = (context or {}).get("notes_text") or ""
        def tok(s: str) -> set[str]:
            return set([t for t in re.findall(r"[a-zA-Z0-9]+", (s or "").lower()) if len(t) >= 3])
        nt = tok(notes)
        scores = []
        for i, opt in enumerate(task.options):
            ot = tok(opt)
            score = len(ot & nt)
            scores.append((score, i))
        scores.sort(reverse=True)
        chosen = scores[0][1] if scores else 0
        return {"chosen_index": chosen, "scores": scores}

    def answer_saq(self, task: SAQTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # simple overlap heuristic: concatenate tokens from notes that match expected points
        import re
        notes = (context or {}).get("notes_text") or ""
        keys = [p.get("key", "") for p in (task.expected_points or [])]
        # build an answer that mentions as many keys as possible
        picks = []
        for k in keys:
            if k and re.search(re.escape(k), notes, re.I):
                picks.append(k)
        if not picks and keys:
            picks = [keys[0]]
        ans = "; ".join(picks) if picks else notes[:120]
        return {"student_answer": ans}

    def answer_code(self, task: CodeTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # naive template: map add to trivial implement; otherwise echo starter
        if task.function_name == "add":
            return {"code": "def add(a,b):\n    return a+b\n"}
        return {"code": task.starter_code or ""}

    def answer_proof_step(self, task: ProofTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        kw = task.expected_keywords[0] if task.expected_keywords else "reason"
        return {"step": f"Uses {kw}."}

    def answer_table_qa(self, task: TableQATask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"answer": task.expected_answer}

    def extract_fact_cards(self, task, source_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Deterministic v2 cards for baseline: one per option, one from snippet (if any)
        import re
        skill = (context or {}).get("skill_id")
        cards: List[Dict[str, Any]] = []
        idx_counter = 1
        for idx, opt in enumerate(getattr(task, "options", []) or []):
            toks = re.findall(r"[A-Za-z0-9]+", opt)
            quote = " ".join(toks[:15]) if toks else opt[:60]
            where = {"scope": "option", "option_index": idx, "start": 0, "end": len(quote)}
            cards.append({
                "id": f"f{idx_counter}",
                "claim": quote,
                "quote": quote,
                "where": where,
                "tags": ([] if not skill else [skill]),
                "hypothesis": f"Option {idx} matches quoted span",
                "polarity": "pro",
            })
            idx_counter += 1
        for snip in ((context or {}).get("retrieved_snippets") or [])[:1]:
            toks = re.findall(r"[A-Za-z0-9]+", snip)
            quote = " ".join(toks[:15]) if toks else snip[:80]
            where = {"scope": "context", "start": 0, "end": len(quote)}
            cards.append({
                "id": f"f{idx_counter}",
                "claim": quote,
                "quote": quote,
                "where": where,
                "tags": ([] if not skill else [skill]),
                "hypothesis": "Context supports one option",
                "polarity": "pro",
            })
            idx_counter += 1
        return {"cards": cards}


class StatefulLLMStudent(LLMStudent):
    def __init__(self, provider: str = "openai", model: str | None = None, memory_max_items: int = 20):
        super().__init__(provider=provider, model=model)
        self.memory: list[str] = []
        self.memory_max_items = memory_max_items

    def _memory_text(self) -> str:
        if not self.memory:
            return ""
        return "\n".join(self.memory[-self.memory_max_items :])

    def answer_mcq(self, task: MCQTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # If Fact-Cards are provided in context, use the same evidence-aware JSON path as LLMStudent,
        # but include stateful memory to assist selection.
        if context and context.get("fact_cards"):
            import json
            t = None
            try:
                t = float((context or {}).get("target_confidence"))
            except Exception:
                t = None
            req = 1
            try:
                req = max(1, int((context or {}).get("q_min") or 1))
            except Exception:
                req = 1
            system = (
                "You are answering a multiple-choice question ONLY using the provided FactCards. "
                "Return strictly JSON with keys: options (array), choice, witness (object). "
                "Each element of options must be {id:'A'|'B'|'C'|'D'|'E', hypothesis:string, score:number[0,1], citations:[card_id,...]}. "
                f"Rules: scores sum ≈ 1; For the selected choice, include ≥{req} citations that quote verbatim from that option (where.scope='option' with option_index). "
                "The FIRST citation for the chosen option MUST be its PRO card id from option_card_ids. If you cannot cite at least the required number of option-linked cards, output choice:'IDK'. "
                "If no such card exists for an option, you must NOT choose it. Always include citations for the chosen option. "
                "All cited cards must include the provided skill_id in tags; quotes must be verbatim substrings ≤ 15 tokens. "
                "You are also given option_card_ids mapping letters (A,B,...) to the PRO card id for each option. "
                "For any option you consider, include its own PRO card id in its citations; for the CHOSEN option, the FIRST citation MUST be its PRO card id. "
                "Additionally, include witness as JSON: {rule:{card_id:string, quote:string}, choice:{card_id:string, quote:string}}. "
                "Quotes in witness must be verbatim (≤30 words) and card_id must reference cited Fact-Cards; for choice, card_id MUST be the chosen option's PRO id. "
                "Verbatim only. No paraphrase in citation text."
            )
            payload = {
                "skill_id": (context or {}).get("skill_id"),
                "cards": context.get("fact_cards"),
                "question": task.stem,
                "options": task.options,
                "memory": self._memory_text(),
                **({"option_card_ids": (context or {}).get("option_card_ids")} if (context or {}).get("option_card_ids") else {}),
                **({"t": t} if t is not None else {}),
            }
            # Optional internal reasoning hint and decoding controls
            if (context or {}).get("reasoning") == "cot":
                system = system + " Think step-by-step internally, but output only the required JSON."
            decode = (context or {}).get("decode") or {}
            try:
                decode = {**decode, "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "mcq_evidence_with_witness",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "options": {"type": "array", "items": {"type": "object", "properties": {
                                    "id": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                                    "hypothesis": {"type": "string"},
                                    "score": {"type": "number"},
                                    "citations": {"type": "array", "items": {"anyOf": [
                                        {"type": "string"}, {"type": "integer"},
                                        {"type": "object", "properties": {"id": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}, "required": ["id"]}
                                    ]}}
                                } }},
                                "choice": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                                "chosen_index": {"type": "integer"},
                                "witness": {"type": "object", "properties": {
                                    "rule": {"type": "object", "properties": {"card_id": {"anyOf": [{"type": "string"}, {"type": "integer"}]}, "quote": {"type": "string"}}, "required": ["card_id", "quote"]},
                                    "choice": {"type": "object", "properties": {"card_id": {"anyOf": [{"type": "string"}, {"type": "integer"}]}, "quote": {"type": "string"}}, "required": ["card_id", "quote"]}
                                }, "required": ["rule", "choice"]}
                            },
                            "required": ["options", "witness"]
                        }
                    }
                }}
            except Exception:
                pass
            js = self.model._chat_json_opts(system, json.dumps(payload, ensure_ascii=False), **decode)
            self._bump_usage(js)
            # decode choice robustly (letter or index), and collect citations
            def _to_letter(x: Any) -> str:
                try:
                    if isinstance(x, str):
                        return x.strip().upper()
                    if isinstance(x, int) and 0 <= x < 26:
                        return chr(ord('A') + x)
                    if isinstance(x, dict):
                        v = x.get("id") or x.get("letter") or x.get("choice")
                        if isinstance(v, str):
                            return v.strip().upper()
                        idx = x.get("index") or x.get("chosen_index")
                        if isinstance(idx, int) and 0 <= idx < 26:
                            return chr(ord('A') + idx)
                except Exception:
                    return ""
                return ""
            # Detect abstention via explicit 'IDK'
            if isinstance(js.get("choice"), str) and js.get("choice").strip().upper() == "IDK":
                out: Dict[str, Any] = {"chosen_index": None, "citations": [], "raw": js}
                try:
                    if isinstance(js.get("witness"), dict) or isinstance(js.get("witness"), list):
                        out["witness"] = js.get("witness")
                except Exception:
                    pass
                return out
            letter = _to_letter(js.get("choice")) or _to_letter(js.get("selected"))
            chosen = None
            if letter in ("A","B","C","D","E","F","G","H"):
                chosen = ord(letter) - ord('A')
            if chosen is None:
                try:
                    ci = int(js.get("chosen_index"))
                    if 0 <= ci < len(task.options):
                        chosen = ci
                except Exception:
                    pass
            if chosen is None:
                chosen = 0 if task.options else None
            # collect citations if present
            citations = []
            try:
                opts = js.get("options")
                if isinstance(opts, list) and 0 <= chosen < len(opts):
                    opt = opts[chosen]
                    top = opt.get("citations")
                    if isinstance(top, list):
                        citations = top
            except Exception:
                top = js.get("citations")
                if isinstance(top, list):
                    citations = top
            out2: Dict[str, Any] = {"chosen_index": chosen, "citations": citations, "raw": js}
            try:
                if isinstance(js.get("witness"), dict) or isinstance(js.get("witness"), list):
                    out2["witness"] = js.get("witness")
            except Exception:
                pass
            return out2
        # Otherwise, use stateful memory + MCQ answering with optional scaffolds/examples
        stem = task.stem
        mem = self._memory_text()
        if mem:
            stem = f"MEMORY:\n{mem}\n\nQUESTION: {stem}"
        if context and context.get("context_text"):
            stem = f"CONTEXT:\n{context['context_text']}\n\n{stem}"
        # Program-of-Thought path with safe execution (optional)
        if (context or {}).get("reasoning") == "pot" and (context or {}).get("use_pyexec", True):
            import json
            system = (
                "Write a tiny Python program to solve the multiple-choice question. "
                "Return only JSON {program:string}. Program must be pure Python, no imports/attributes; keep it short. "
                "The program should set a variable named result to the final numeric/string answer."
            )
            payload = {"stem": stem, "options": task.options, **({"examples": (context or {}).get("examples")} if (context or {}).get("examples") else {})}
            decode = (context or {}).get("decode") or {}
            js_prog = self.model._chat_json_opts(system, json.dumps(payload, ensure_ascii=False), **decode)
            code = (js_prog or {}).get("program") or (js_prog or {}).get("code") or ""
            if code:
                try:
                    from sim.tools import SafePythonExecutor
                    exe = SafePythonExecutor()
                    out = exe.run(code=code, inputs={})
                    if out.get("ok"):
                        val = out.get("result")
                        def _num(x):
                            try:
                                return float(str(x).strip())
                            except Exception:
                                return None
                        vnum = _num(val)
                        choice = None
                        if vnum is not None:
                            best = (1e18, None)
                            for i, opt in enumerate(task.options):
                                on = _num(opt)
                                if on is not None:
                                    err = abs(on - vnum)
                                    if err < best[0]:
                                        best = (err, i)
                            if best[1] is not None:
                                choice = best[1]
                        else:
                            sval = str(val).strip().lower()
                            for i, opt in enumerate(task.options):
                                if str(opt).strip().lower() == sval:
                                    choice = i; break
                        if choice is not None:
                            return {"chosen_index": choice, "raw": {"chosen_index": choice, "confidence": 0.85, "pot": True}}
                except Exception:
                    pass
        reasoning = (context or {}).get("reasoning")
        scaffold = ""
        if reasoning == "cot":
            scaffold = " Think step-by-step internally."
        elif reasoning == "ltm":
            scaffold = " Internally break into minimal substeps and solve in order."
        elif reasoning == "tot":
            scaffold = " Consider two approaches internally and continue with the better."
        elif reasoning == "sot":
            scaffold = " Outline an internal skeleton, then fill it."
        elif reasoning == "selfdisco":
            scaffold = " Compose an internal reasoning plan from modules."
        elif reasoning == "got":
            scaffold = " Consider an internal DAG of subproblems."
        elif reasoning == "pot":
            scaffold = " Sketch internal pseudo-code steps (no execution)."
        import json
        header = (context or {}).get("instruction_header")
        system = (
            "You are answering a multiple-choice question. Use provided examples if any as demonstrations. "
            "Return only JSON with keys: chosen_index (int), confidence (0..1)." + scaffold
        )
        if header:
            system = header.strip() + " " + system
        payload = {"stem": stem, "options": task.options}
        if (context or {}).get("examples"):
            payload["examples"] = (context or {}).get("examples")
        decode = (context or {}).get("decode") or {}
        js = self.model._chat_json_opts(system, json.dumps(payload, ensure_ascii=False), **decode)
        return {"chosen_index": js.get("chosen_index"), "raw": js}


class OracleStudent(Learner):
    """Deterministic, evidence-compliant baseline for offline checks.

    - Chooses the correct option (gold index) when available.
    - When Fact-Cards are provided in context, cites the option's PRO card id
      twice to satisfy strict evidence gating and witness requirements.
    """

    def answer_mcq(self, task: MCQTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        chosen = int(getattr(task, "correct_index", 0) or 0)
        citations: List[str] = []
        try:
            opt_map = (context or {}).get("option_card_ids") or {}
            # Map A,B,... to chosen option
            letter = chr(ord('A') + chosen)
            pro_id = opt_map.get(letter)
            # Try to add a second, distinct option-linked card id for the chosen option
            second_id = None
            try:
                cards = (context or {}).get("fact_cards") or []
                for c in cards:
                    w = (c.get("where") or {})
                    if (w.get("scope") == "option") and (int(w.get("option_index") or -1) == chosen):
                        cid = str(c.get("id") or "")
                        if cid and (pro_id is None or cid != str(pro_id)):
                            second_id = cid
                            break
            except Exception:
                second_id = None
            if pro_id:
                citations.append(str(pro_id))
                citations.append(str(second_id) if second_id else str(pro_id))
        except Exception:
            citations = []
        return {"chosen_index": chosen, **({} if not citations else {"citations": citations})}

    def answer_saq(self, task: SAQTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Minimal SAQ answer touching expected key
        key = (task.expected_points[0].get("key") if task.expected_points else "answer")
        return {"student_answer": f"{key}: definition"}

    def answer_code(self, task: CodeTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Implement known simple functions for tests; else return starter
        if (task.function_name or "").strip() == "add":
            return {"code": "def add(a,b):\n    return a+b\n"}
        return {"code": task.starter_code}

    def answer_proof_step(self, task: ProofTask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"step": "By commutativity, a+b=b+a."}

    def answer_table_qa(self, task: TableQATask, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"answer": task.expected_answer or ""}

    def extract_fact_cards(self, task, source_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Deterministic option-linked cards for oracle runs.

        Produces at least two PRO cards per option by quoting verbatim substrings
        (≤15 tokens) from each option's text so strict evidence gates can pass.
        """
        cards: List[Dict[str, Any]] = []
        try:
            from re import findall
            skill = (context or {}).get("skill_id")
            idx = 1
            options = list(getattr(task, "options", []) or [])
            for oi, opt in enumerate(options):
                toks = findall(r"[A-Za-z0-9]+", opt or "")
                q1 = " ".join(toks[:15]) if toks else (opt or "")[:60]
                q2 = " ".join(toks[1:16]) if len(toks) > 1 else q1
                seen: set[str] = set()
                for q in (q1, q2):
                    if not q or q in seen:
                        continue
                    seen.add(q)
                    cards.append({
                        "id": f"f{idx}",
                        "claim": q,
                        "quote": q,
                        "where": {"scope": "option", "option_index": oi, "start": 0, "end": len(q), "source_id": f"option:{oi}"},
                        "tags": ([] if not skill else [skill]),
                        "hypothesis": f"Option {oi} matches quoted span",
                        "polarity": "pro",
                    })
                    idx += 1
        except Exception:
            pass
        return {"cards": cards}
