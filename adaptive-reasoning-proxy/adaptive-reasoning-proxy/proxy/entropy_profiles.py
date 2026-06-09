"""Entropy profile matching via Dynamic Time Warping (Priority 4).

Rather than reducing each task to a single scalar threshold, this module keeps
the *shape* of the entropy curve H(t) for past requests and matches a new
curve to known profiles with DTW distance — closer to the EDRM paper's
manifold framing.

It is an enrichment layer: profiles are learned online from completed requests
and exposed for analysis / nearest-profile classification. It does not (yet)
drive the stop decision, so it can never destabilise generation.
"""

import logging
from collections import defaultdict, deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Curves are resampled to a fixed length so DTW stays cheap and comparable.
PROFILE_LEN = 40
# How many recent reference curves to retain per task type.
MAX_PER_TASK = 25

_profiles: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_PER_TASK))


def _resample(trace: list[float], length: int = PROFILE_LEN) -> np.ndarray:
    """Linearly resample an entropy trace to a fixed length."""
    arr = np.asarray(trace, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(length)
    if arr.size == 1:
        return np.full(length, arr[0])
    xp = np.linspace(0.0, 1.0, arr.size)
    x = np.linspace(0.0, 1.0, length)
    return np.interp(x, xp, arr)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Classic O(n*m) DTW distance between two 1-D sequences."""
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            d = abs(ai - b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m])


def record_profile(task_value: str, trace: list[float]) -> None:
    """Store a completed request's entropy curve as a reference profile."""
    if not trace:
        return
    _profiles[task_value].append(_resample(trace))


def best_match(trace: list[float]) -> Optional[tuple[str, float]]:
    """Return ``(task_value, distance)`` of the nearest stored profile, or None
    if no profiles have been learned yet."""
    if not any(_profiles.values()):
        return None
    query = _resample(trace)
    best_task: Optional[str] = None
    best_dist = np.inf
    for task_value, curves in _profiles.items():
        for curve in curves:
            d = dtw_distance(query, curve)
            if d < best_dist:
                best_dist = d
                best_task = task_value
    if best_task is None:
        return None
    return best_task, best_dist


def classify_by_profile(trace: list[float]) -> Optional[str]:
    """Nearest-profile task classification for a (possibly partial) curve."""
    match = best_match(trace)
    return match[0] if match else None


def profile_summary() -> dict:
    """Per-task summary for the dashboard: count + mean reference curve."""
    summary = {}
    for task_value, curves in _profiles.items():
        if not curves:
            continue
        stacked = np.vstack(list(curves))
        summary[task_value] = {
            "count": len(curves),
            "mean_curve": [round(float(v), 4) for v in stacked.mean(axis=0)],
            "profile_len": PROFILE_LEN,
        }
    return summary
