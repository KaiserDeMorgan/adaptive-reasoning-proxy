import re
from enum import Enum


class TaskType(Enum):
    FACTUAL   = "factual"
    REASONING = "reasoning"
    CREATIVE  = "creative"
    CODE      = "code"


# Entropy thresholds calibrated empirically — tune with your own data
THRESHOLDS = {
    TaskType.FACTUAL:   0.35,
    TaskType.CREATIVE:  0.55,
    TaskType.CODE:      0.60,
    TaskType.REASONING: 0.75,
}

_FACTUAL_RE = re.compile(r"\b(what is|who is|when|where|define|list)\b", re.I)
_REASON_RE  = re.compile(r"\b(why|explain|prove|analyze|compare|reason)\b", re.I)
_CODE_RE    = re.compile(r"\b(write|implement|code|function|class|script)\b", re.I)


def classify(prompt: str) -> TaskType:
    """
    Route a prompt to a task type using regex heuristics.
    Replace with an embedding-based classifier for production use.
    """
    if _CODE_RE.search(prompt):    return TaskType.CODE
    if _REASON_RE.search(prompt):  return TaskType.REASONING
    if _FACTUAL_RE.search(prompt): return TaskType.FACTUAL
    return TaskType.CREATIVE
