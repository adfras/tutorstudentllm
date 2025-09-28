from __future__ import annotations

"""
Text utilities shared across the simulator.

Provides a single source of truth for tokenization and quote truncation
rules so evidence handling stays consistent.
"""

import re
from typing import List

_TOK = re.compile(r"[A-Za-z0-9]+")


def tokens(s: str | None) -> List[str]:
    """Lightweight alnum tokenization used across evidence utilities."""
    return [t.lower() for t in _TOK.findall(s or "")] 


def truncate_quote(s: str | None, max_tokens: int = 15) -> str:
    """Clamp a quote to at most `max_tokens` tokens.

    Keeps original spacing by joining tokens with a single space.
    """
    toks = tokens(s)
    if not toks:
        return (s or "")
    return " ".join(toks[: max(1, int(max_tokens))])

