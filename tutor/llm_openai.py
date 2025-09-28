from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

from tutor.utils import extract_json


MODEL_NAME = "gpt-5-nano-2025-08-07"


class OpenAILLM:
    def __init__(self):
        # Load env
        from tutor.utils import load_env_dotenv_fallback
        load_env_dotenv_fallback()
        self.model = MODEL_NAME
        self._last_usage: dict | None = None
        self._last_ms: float | None = None
        # Mock toggle for offline dev/tests
        self._mock = str(os.getenv("TUTOR_MOCK_LLM", "")).lower() in ("1", "true", "yes", "on")
        if not self._mock:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set in environment or .env")
            # Lazy import to avoid hard dependency when listing help
            from openai import OpenAI  # type: ignore
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None  # type: ignore
        # Cumulative usage (per-step if Orchestrator resets)
        self._calls_step = 0
        self._prompt_tokens_step = 0
        self._completion_tokens_step = 0
        self._total_tokens_step = 0
        self._latency_ms_step = 0.0
        # Message buffer (optional; controlled by LOG_MESSAGES)
        self._messages_step: list[dict] = []
        self.role_group = "tutor"

    def _chat_json(self, system: str, user: str) -> Dict[str, Any]:
        return self._chat_json_opts(system, user)

    def _chat_json_opts(self, system: str, user: str, **decode_opts) -> Dict[str, Any]:
        # Prefer JSON mode when available
        import time
        t0 = time.time()
        kwargs = dict(decode_opts or {})
        # sanitize unsupported keys to avoid common API errors
        # (OpenAI ignores unknown args, but we keep it tidy.)
        # Pull response_format from kwargs when provided; default to JSON mode
        rf = kwargs.pop("response_format", {"type": "json_object"})
        if self._mock:
            # Default deterministic JSON for mock path: echo a minimal object
            try:
                from tutor.utils import extract_json as _extract
                _ = _extract(user)
            except Exception:
                pass
            return {"ok": True}
        resp = self.client.chat.completions.create(
            model=self.model,
            response_format=rf,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        ms = (time.time() - t0) * 1000.0
        content = resp.choices[0].message.content or "{}"
        try:
            js = extract_json(content)
        except Exception:
            # Fallback to plain JSON parse
            js = extract_json(content)
        # attach usage/timing
        try:
            usage = getattr(resp, "usage", None)
            if usage is not None:
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            else:
                usage_dict = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        except Exception:
            usage_dict = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        self._last_usage = usage_dict
        self._last_ms = ms
        # bump cumulative
        try:
            pt = int(usage_dict.get("prompt_tokens") or 0)
            ct = int(usage_dict.get("completion_tokens") or 0)
            tt = int(usage_dict.get("total_tokens") or (pt + ct))
            self._calls_step += 1
            self._prompt_tokens_step += pt
            self._completion_tokens_step += ct
            self._total_tokens_step += tt
            self._latency_ms_step += float(ms)
        except Exception:
            pass
        if isinstance(js, dict):
            js["_usage"] = usage_dict
            js["_request_ms"] = ms
        # Optionally log a message event (counts-only by default)
        try:
            import time as _t
            mode = os.getenv("LOG_MESSAGES", "").lower()
            if mode in ("counts", "text"):
                ev = {
                    "ts": int(_t.time()),
                    "role_group": self.role_group,
                    "api": "openai.chat",
                    "model": self.model,
                    "prompt_tokens": usage_dict.get("prompt_tokens"),
                    "completion_tokens": usage_dict.get("completion_tokens"),
                    "total_tokens": usage_dict.get("total_tokens"),
                    "request_ms": ms,
                    "system_len": len(system or ""),
                    "user_len": len(user or ""),
                }
                if mode == "text":
                    # Clamp to keep logs manageable
                    def _clip(s: str, n: int = 4000) -> str:
                        try:
                            return (s or "")[:n]
                        except Exception:
                            return ""
                    ev["system"] = _clip(system)
                    ev["user"] = _clip(user)
                self._messages_step.append(ev)
        except Exception:
            pass
        return js

    # ---- Usage helpers to mirror student-side accounting ----
    def reset_usage_counters(self) -> None:
        self._calls_step = 0
        self._prompt_tokens_step = 0
        self._completion_tokens_step = 0
        self._total_tokens_step = 0
        self._latency_ms_step = 0.0
        # Also reset messages buffer per step
        self._messages_step = []

    def get_usage_counters(self) -> Dict[str, float | int]:
        return {
            "calls": int(self._calls_step),
            "prompt_tokens": int(self._prompt_tokens_step),
            "completion_tokens": int(self._completion_tokens_step),
            "total_tokens": int(self._total_tokens_step),
            "request_ms_sum": float(self._latency_ms_step),
        }

    # Message buffer helpers
    def reset_messages_buffer(self) -> None:
        self._messages_step = []

    def get_messages_buffer(self) -> list[dict]:
        return list(self._messages_step)

    def verify_key_and_model(self) -> Dict[str, Any]:
        """Verify API key works and model is callable."""
        info: Dict[str, Any] = {"model": self.model}
        if self._mock:
            info.update({"mock": True, "models_list_ok": True, "chat_ok": True})
            return info
        # Check models endpoint
        try:
            _ = self.client.models.list()
            info["models_list_ok"] = True
        except Exception as e:
            info["models_list_ok"] = False
            info["models_error"] = str(e)
        # Try a minimal chat call
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return exactly the provided JSON."},
                    {"role": "user", "content": '{"ok": true}'}
                ],
            )
            text = resp.choices[0].message.content or "{}"
            extract_json(text)  # validate JSON parsable
            info["chat_ok"] = True
        except Exception as e:
            info["chat_ok"] = False
            info["chat_error"] = str(e)
        return info

    # MCQ generation
    def generate_mcq(
        self,
        skill: Dict[str, Any],
        difficulty: str = "medium",
        avoid_repeats: List[str] | None = None,
        num_options: int = 5,
        minimal: bool = True,
        template: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Mock: return deterministic abstract MCQ
        if self._mock:
            opts = ["A1", "B2", "C3", "D4", "E5"][: max(2, int(num_options or 5))]
            return {"question_id": "mock-1", "stem": "Pick the first token.", "options": opts, "correct_index": 0, "difficulty": difficulty}
        import json, random
        avoid_repeats = avoid_repeats or []
        if minimal:
            system = (
                "You are a domain-agnostic tutor generating a concise multiple-choice question. "
                "Target the given skill using purely structural, abstract content. Absolutely avoid discipline-specific terms and real-world entities. "
                "Use placeholder tokens (e.g., X1, ZQ-14) and neutral wording like pattern, sequence, mapping, rule. "
                "Return only a JSON object with keys: stem (string), options (array, length = constraints.num_options), correct_index (int). "
                "Keep stem under 25 words. Keep each option under 8 words. No explanations, no rationales, no extra fields."
            )
        else:
            system = (
                "You are a domain-agnostic tutor generating exam-quality multiple-choice questions. "
                "Use only abstract, structural content (patterns, rules, mappings) with placeholder tokens; avoid any subject-specific or real-world references. "
                "Include plausible distractors that reflect structural misconceptions (e.g., off-by-one, wrong mapping). "
                "Return only a JSON object with keys: question_id (string), stem (string), options (array of strings with length equal to constraints.num_options), correct_index (int), rationales (array of strings matching options length), misconception_tags (array of strings, same length as options), difficulty (string)."
            )
        payload = {
            "skill_id": skill["id"],
            "skill_name": skill["name"],
            "bloom": skill.get("bloom", ""),
            "difficulty": difficulty,
            "constraints": {"num_options": num_options, "avoid_repeats": avoid_repeats},
        }
        if template:
            payload["template_id"] = template.get("id")
            payload["template"] = template.get("description")
            if template.get("misconception_tags"):
                payload["misconception_tags"] = template["misconception_tags"]
        user = json.dumps(payload, ensure_ascii=False)
        return self._chat_json(system, user)

    # SAQ generation
    def generate_saq(self, skill: Dict[str, Any], difficulty: str = "medium") -> Dict[str, Any]:
        import json
        if self._mock:
            return {"question_id": "saq-1", "stem": "Define token.", "expected_points": [{"key": "token", "required": True}], "model_answer": "token: unit", "difficulty": difficulty}
        system = (
            "You are a domain-agnostic tutor generating a short-answer question. "
            "Expect a concise 1–3 sentence answer. Provide expected points and a reference answer. "
            "Return only a JSON object with keys: question_id (string), stem (string), expected_points (array of {key, required}), model_answer (string), difficulty (string)."
        )
        payload = {
            "skill_id": skill["id"],
            "skill_name": skill["name"],
            "bloom": skill.get("bloom", ""),
            "difficulty": difficulty,
        }
        user = json.dumps(payload, ensure_ascii=False)
        return self._chat_json(system, user)

    # Grade SAQ
    def grade_saq(self, stem: str, expected_points: List[Dict[str, Any]], model_answer: str, student_answer: str) -> Dict[str, Any]:
        if self._mock:
            return {"score": 1.0, "correct": True, "found_points": [p.get("key") for p in (expected_points or [])], "missing_points": [], "misconception_tags": [], "feedback": "Looks good."}
        system = (
            "You grade short answers fairly and rigorously. "
            "Return a JSON object with numeric score 0..1, boolean correct, found_points, missing_points, misconception_tags, and 2–3 sentence feedback."
        )
        import json

        user_obj = {
            "stem": stem,
            "expected_points": expected_points,
            "model_answer": model_answer,
            "student_answer": student_answer,
        }
        user = json.dumps(user_obj, ensure_ascii=False)
        return self._chat_json(system, user)

    # Self-verify MCQ: model answers the item independently
    def answer_mcq(self, stem: str, options: List[str]) -> Dict[str, Any]:
        import json
        if self._mock:
            return {"chosen_index": 0, "confidence": 0.5}
        system = (
            "You are answering a multiple-choice question. Return only JSON with keys: chosen_index (int), confidence (0..1)."
        )
        payload = {"stem": stem, "options": options}
        user = json.dumps(payload, ensure_ascii=False)
        return self._chat_json_opts(system, user)


class OpenAIStudentLLM:
    """OpenAI Chat API client for student models (selectable by --model).
    Uses the same JSON helpers as the tutor wrapper but does not hard-lock the model.
    """

    def __init__(self, model: str | None = None):
        from tutor.utils import load_env_dotenv_fallback
        load_env_dotenv_fallback()
        self.model = model or os.getenv("OPENAI_STUDENT_MODEL") or "gpt-4.1"
        self._last_usage: dict | None = None
        self._last_ms: float | None = None
        self._mock = str(os.getenv("TUTOR_MOCK_LLM", "")).lower() in ("1", "true", "yes", "on")
        if not self._mock:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set in environment or .env")
            from openai import OpenAI  # type: ignore
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None  # type: ignore
        self.role_group = "student"
        self._messages_step: list[dict] = []

    def _chat_json_opts(self, system: str, user: str, **decode_opts) -> Dict[str, Any]:
        import time
        t0 = time.time()
        kwargs = dict(decode_opts or {})
        # Some OpenAI models (e.g., gpt-4.1) only support default decoding; strip unsupported keys
        if (self.model or "").startswith("gpt-4.1"):
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
            kwargs.pop("min_p", None)
        rf = kwargs.pop("response_format", {"type": "json_object"})
        if self._mock:
            return {"chosen_index": 0, "confidence": 0.5}
        resp = self.client.chat.completions.create(
            model=self.model,
            response_format=rf,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        ms = (time.time() - t0) * 1000.0
        content = resp.choices[0].message.content or "{}"
        try:
            js = extract_json(content)
        except Exception:
            js = extract_json(content)
        try:
            usage = getattr(resp, "usage", None)
            if usage is not None:
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            else:
                usage_dict = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        except Exception:
            usage_dict = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        self._last_usage = usage_dict
        self._last_ms = ms
        if isinstance(js, dict):
            js["_usage"] = usage_dict
            js["_request_ms"] = ms
        # Optional message logging
        try:
            import time as _t
            mode = os.getenv("LOG_MESSAGES", "").lower()
            if mode in ("counts", "text"):
                ev = {
                    "ts": int(_t.time()),
                    "role_group": self.role_group,
                    "api": "openai.chat",
                    "model": self.model,
                    "prompt_tokens": usage_dict.get("prompt_tokens"),
                    "completion_tokens": usage_dict.get("completion_tokens"),
                    "total_tokens": usage_dict.get("total_tokens"),
                    "request_ms": ms,
                    "system_len": len(system or ""),
                    "user_len": len(user or ""),
                }
                if mode == "text":
                    def _clip(s: str, n: int = 4000) -> str:
                        try:
                            return (s or "")[:n]
                        except Exception:
                            return ""
                    ev["system"] = _clip(system)
                    ev["user"] = _clip(user)
                self._messages_step.append(ev)
        except Exception:
            pass
        return js

    # Messages buffer API
    def reset_messages_buffer(self) -> None:
        self._messages_step = []

    def get_messages_buffer(self) -> list[dict]:
        return list(self._messages_step)

    def answer_mcq(self, stem: str, options: List[str]) -> Dict[str, Any]:
        system = "You are answering a multiple-choice question. Return only JSON with keys: chosen_index (int), confidence (0..1)."
        import json
        payload = {"stem": stem, "options": options}
        return self._chat_json_opts(system, json.dumps(payload, ensure_ascii=False))

    def verify_key_and_model(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"model": self.model}
        if self._mock:
            info.update({"mock": True, "chat_ok": True})
            return info
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return exactly the provided JSON."},
                    {"role": "user", "content": '{"ok": true}'}
                ],
            )
            text = resp.choices[0].message.content or "{}"
            extract_json(text)
            info["chat_ok"] = True
        except Exception as e:
            info["chat_ok"] = False
            info["chat_error"] = str(e)
        return info
