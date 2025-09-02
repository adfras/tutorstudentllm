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
    - Honors TUTOR_MOCK_LLM=1 for offline runs.
    - Returns strict JSON via response_format when supported; otherwise extracts JSON from content.
    """

    def __init__(self, model: str | None = None):
        load_env_dotenv_fallback()
        self._mock = os.getenv("TUTOR_MOCK_LLM") == "1"
        self.base_url = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
        self.model = model or os.getenv("DEEPINFRA_MODEL") or ""
        if self._mock:
            self.client = None  # type: ignore
        else:
            api_key = os.getenv("DEEPINFRA_API_KEY")
            if not api_key:
                raise RuntimeError("DEEPINFRA_API_KEY not set in environment or .env")
            if not self.model:
                raise RuntimeError("DEEPINFRA_MODEL not set (choose a DeepInfra model, e.g., 'deepseek-ai/DeepSeek-R1')")
            # Use OpenAI SDK pointed at DeepInfra's OpenAI-compatible endpoint
            from openai import OpenAI  # type: ignore
            self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    def _chat_json(self, system: str, user: str) -> Dict[str, Any]:
        if self._mock:
            try:
                return extract_json(user)
            except Exception:
                return {"ok": True}
        # Some OSS models may not honor response_format; we still request it and fallback to parsing
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception:
            # Fallback: some OSS models reject response_format; retry without it
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        content = resp.choices[0].message.content or "{}"
        try:
            return extract_json(content)
        except Exception:
            return extract_json(content)

    def answer_mcq(self, stem: str, options: List[str]) -> Dict[str, Any]:
        if self._mock:
            return {"chosen_index": 0 if options else 0, "confidence": 0.5}
        system = "You are answering a multiple-choice question. Return only JSON with keys: chosen_index (int), confidence (0..1)."
        import json
        payload = {"stem": stem, "options": options}
        return self._chat_json(system, json.dumps(payload, ensure_ascii=False))

    def verify_key_and_model(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"base_url": self.base_url, "model": self.model}
        if self._mock:
            return {**info, "mock": True, "chat_ok": True}
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
