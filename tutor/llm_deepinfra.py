from __future__ import annotations
import os
from typing import Any, Dict, List

from tutor.utils import extract_json, load_env_dotenv_fallback


class DeepInfraLLM:
    """OpenAI-compatible Chat API client for DeepInfra-hosted models (incl. DeepSeek).

    Notes
    - Reads `DEEPINFRA_API_KEY` and optional `DEEPINFRA_BASE_URL` (defaults to
      https://api.deepinfra.com/v1/openai).
    - Model name is taken from `DEEPINFRA_MODEL` unless explicitly provided.
    - OpenAI-compatible endpoint; no offline mock mode.
    - Returns strict JSON via response_format when supported; otherwise extracts JSON from content.
    """

    def __init__(self, model: str | None = None):
        load_env_dotenv_fallback()
        self.base_url = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
        self.model = model or os.getenv("DEEPINFRA_MODEL") or ""
        self._last_usage: dict | None = None
        self._last_ms: float | None = None
        # Mock toggle for offline dev/tests
        self._mock = str(os.getenv("TUTOR_MOCK_LLM", "")).lower() in ("1", "true", "yes", "on")
        if not self._mock:
            api_key = os.getenv("DEEPINFRA_API_KEY")
            if not api_key:
                raise RuntimeError("DEEPINFRA_API_KEY not set in environment or .env")
            if not self.model:
                raise RuntimeError("DEEPINFRA_MODEL not set (choose a DeepInfra model, e.g., 'deepseek-ai/DeepSeek-R1')")
            # Use OpenAI SDK pointed at DeepInfra's OpenAI-compatible endpoint
            from openai import OpenAI  # type: ignore
            self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        else:
            # In mock mode, no network client is constructed
            self.client = None  # type: ignore
        # Per-step usage + messages
        self._calls_step = 0
        self._prompt_tokens_step = 0
        self._completion_tokens_step = 0
        self._total_tokens_step = 0
        self._latency_ms_step = 0.0
        self._messages_step: list[dict] = []
        self.role_group = "student"

    def _chat_json(self, system: str, user: str) -> Dict[str, Any]:
        return self._chat_json_opts(system, user)

    def _chat_json_opts(self, system: str, user: str, **decode_opts) -> Dict[str, Any]:
        if self._mock:
            # Default deterministic shape for MCQ answer; callers that need other shapes
            # (e.g., extraction) should provide overrides or rely on higher-level mocks.
            try:
                return {"chosen_index": 0, "confidence": 0.5}
            except Exception:
                return {"chosen_index": 0, "confidence": 0.5}
        # Some OSS models may not honor response_format; we still request it and fallback to parsing
        import time
        t0 = time.time()
        rf = (decode_opts or {}).pop("response_format", {"type": "json_object"})
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                response_format=rf,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **(decode_opts or {}),
            )
        except Exception:
            # Fallback: some OSS models reject response_format; retry without it
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **(decode_opts or {}),
            )
        ms = (time.time() - t0) * 1000.0
        content = resp.choices[0].message.content or "{}"
        try:
            js = extract_json(content)
        except Exception:
            # Final fallback: return raw text for higher-level heuristics
            js = {"text": content, "_parse_error": True}
        # attach usage/timing if available
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
        # bump cumulative usage
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
        # Optional message logging
        try:
            import time as _t
            mode = os.getenv("LOG_MESSAGES", "").lower()
            if mode in ("counts", "text"):
                ev = {
                    "ts": int(_t.time()),
                    "role_group": self.role_group,
                    "api": "deepinfra.chat",
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

    # Usage/messages helpers for orchestrator
    def reset_usage_counters(self) -> None:
        self._calls_step = 0
        self._prompt_tokens_step = 0
        self._completion_tokens_step = 0
        self._total_tokens_step = 0
        self._latency_ms_step = 0.0
        self._messages_step = []

    def get_usage_counters(self) -> Dict[str, float | int]:
        return {
            "calls": int(self._calls_step),
            "prompt_tokens": int(self._prompt_tokens_step),
            "completion_tokens": int(self._completion_tokens_step),
            "total_tokens": int(self._total_tokens_step),
            "request_ms_sum": float(self._latency_ms_step),
        }

    def reset_messages_buffer(self) -> None:
        self._messages_step = []

    def get_messages_buffer(self) -> list[dict]:
        return list(self._messages_step)

    def answer_mcq(self, stem: str, options: List[str]) -> Dict[str, Any]:
        if self._mock:
            return {"chosen_index": 0 if options else 0, "confidence": 0.5}
        system = "You are answering a multiple-choice question. Return only JSON with keys: chosen_index (int), confidence (0..1)."
        import json
        payload = {"stem": stem, "options": options}
        return self._chat_json_opts(system, json.dumps(payload, ensure_ascii=False))

    def verify_key_and_model(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"base_url": self.base_url, "model": self.model}
        if self._mock:
            info.update({"mock": True, "chat_ok": True})
            return info
        try:
            _ = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return exactly the provided JSON."},
                    {"role": "user", "content": '{"ok": true}'}
                ],
            )
            info["chat_ok"] = True
        except Exception as e:
            info["chat_ok"] = False
            info["chat_error"] = str(e)
        return info
