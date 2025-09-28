from __future__ import annotations

"""Prompt-building utilities for MCQ learners."""

from typing import Any, Dict, Tuple


def fact_card_prompt_components(task, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Return (system, payload, decode_overrides) for fact-card grounded MCQ prompts."""
    import json

    t = None
    try:
        t = float((context or {}).get("target_confidence"))
    except Exception:
        t = None
    header = (context or {}).get("instruction_header")
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
    if header:
        system = header.strip() + " " + system
    payload = {
        "skill_id": (context or {}).get("skill_id"),
        "cards": context.get("fact_cards"),
        "question": task.stem,
        "options": task.options,
    }
    if (context or {}).get("option_card_ids"):
        payload["option_card_ids"] = (context or {}).get("option_card_ids")
    if t is not None:
        payload["t"] = t
    if (context or {}).get("reasoning") == "cot":
        system = system + " Think step-by-step internally, but output only the required JSON."
    decode = {}
    try:
        decode = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "mcq_evidence_with_witness",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "options": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                                        "hypothesis": {"type": "string"},
                                        "score": {"type": "number"},
                                        "citations": {
                                            "type": "array",
                                            "items": {
                                                "anyOf": [
                                                    {"type": "string"},
                                                    {"type": "integer"},
                                                    {
                                                        "type": "object",
                                                        "properties": {"id": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
                                                        "required": ["id"],
                                                    },
                                                ]
                                            },
                                        },
                                    },
                                },
                            },
                            "choice": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                            "chosen_index": {"type": "integer"},
                            "witness": {
                                "type": "object",
                                "properties": {
                                    "rule": {
                                        "type": "object",
                                        "properties": {
                                            "card_id": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                                            "quote": {"type": "string"},
                                        },
                                        "required": ["card_id", "quote"],
                                    },
                                    "choice": {
                                        "type": "object",
                                        "properties": {
                                            "card_id": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                                            "quote": {"type": "string"},
                                        },
                                        "required": ["card_id", "quote"],
                                    },
                                },
                                "required": ["rule", "choice"],
                            },
                        },
                        "required": ["options", "witness"],
                    },
                },
            }
        }
    except Exception:
        decode = {}
    return system, json.dumps(payload, ensure_ascii=False), decode
