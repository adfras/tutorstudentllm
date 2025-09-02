from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Tuple


TaskType = Literal["mcq", "saq", "code", "proof", "table_qa"]


@dataclass
class Task:
    type: TaskType
    id: str
    prompt: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCQTask(Task):
    type: TaskType = field(default="mcq", init=False)  # type: ignore[assignment]
    stem: str = ""
    options: List[str] = field(default_factory=list)
    correct_index: int = 0
    rationales: Optional[List[str]] = None
    misconception_tags: Optional[List[str]] = None


def evaluate_mcq(chosen_index: Optional[int], task: MCQTask) -> Dict[str, Any]:
    correct = (isinstance(chosen_index, int) and 0 <= chosen_index < len(task.options) and chosen_index == task.correct_index)
    return {
        "correct": bool(correct),
        "chosen_index": chosen_index,
        "correct_index": task.correct_index,
    }
