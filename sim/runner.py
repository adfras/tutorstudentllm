from __future__ import annotations

"""Straightforward one-function runner for simple use cases.

This provides a minimal, readable entry point that maps directly to the
Orchestrator with sensible defaults. It’s useful for small scripts and quick
experiments where the full CLI isn’t needed.
"""

from typing import Optional, List, Dict, Any

from .config import RunConfig, Dials
from .orchestrator import Orchestrator
from .learner import LLMStudent, AlgoStudent


def run_simple(
    *,
    steps: int = 3,
    closed_book: bool = True,
    anonymize: bool = True,
    notes: str = "",
    log_path: Optional[str] = None,
    student: str = "llm",  # llm|algo
    provider: str = "openai",
    model: Optional[str] = None,
    skill_id: Optional[str] = None,
    num_options: int = 5,
    difficulty: str = "medium",
) -> List[Dict[str, Any]]:
    dials = Dials(
        closed_book=closed_book,
        anonymize=anonymize,
    )
    cfg = RunConfig(
        skill_id=skill_id,
        num_steps=int(steps),
        num_options=int(num_options),
        difficulty=difficulty,
        dials=dials,
    )
    orch = Orchestrator()
    if student == "algo":
        learner = AlgoStudent()
    else:
        learner = LLMStudent(provider=provider, model=model)
    return orch.run(learner, cfg, notes_text=notes, log_path=log_path)

