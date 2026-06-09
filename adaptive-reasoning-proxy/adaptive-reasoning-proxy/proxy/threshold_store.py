import redis.asyncio as redis
import json
import logging
import os

logger = logging.getLogger(__name__)

# Honour REDIS_URL so the same code works locally and inside docker-compose
# (where Redis is reachable at redis://redis:6379, not localhost).
r = redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True
)

# In-memory fallback so threshold reads/writes (e.g. dashboard sliders) keep
# working when Redis is not running. Redis remains the source of truth when
# available; this mirror is what survives a missing Redis within one process.
_overrides: dict[str, float] = {}


async def get_threshold(task_type: str, default: float) -> float:
    """Retrieve the current entropy threshold for a task type.

    Precedence: Redis value -> in-memory override -> ``default``. Falls back
    gracefully if Redis is unreachable so the proxy keeps working without it.
    """
    try:
        val = await r.get(f"threshold:{task_type}")
    except redis.RedisError as exc:
        logger.warning("Redis unavailable, using override/default: %s", exc)
        return _overrides.get(task_type, default)
    if val is not None:
        return float(val)
    return _overrides.get(task_type, default)


async def set_threshold(task_type: str, value: float) -> float:
    """Set a threshold explicitly (used by the dashboard calibration sliders).

    Always updates the in-memory mirror; also writes to Redis when reachable.
    Returns the clamped value that was stored.
    """
    value = max(0.1, min(0.9, float(value)))
    _overrides[task_type] = value
    try:
        await r.set(f"threshold:{task_type}", value)
    except redis.RedisError as exc:
        logger.warning("Redis unavailable, threshold kept in memory only: %s", exc)
    return value


async def all_thresholds(defaults: dict[str, float]) -> dict[str, float]:
    """Return the current threshold for every task type in ``defaults``."""
    return {task: await get_threshold(task, default) for task, default in defaults.items()}


async def record_outcome(
    task_type: str,
    entropy_at_stop: float,
    user_rating: int,  # 1 = good, -1 = bad (truncated too early)
) -> float:
    """
    Nudge threshold based on user feedback.
    Bad rating (truncated too early) → lower threshold.
    Good rating → inch it up.

    Returns the new threshold value.
    """
    key = f"threshold:{task_type}"
    current = _overrides.get(task_type, 0.5)
    try:
        redis_val = await r.get(key)
        if redis_val is not None:
            current = float(redis_val)
    except redis.RedisError as exc:
        logger.warning("Redis unavailable during record_outcome: %s", exc)

    delta = 0.02 * user_rating
    new_val = max(0.1, min(0.9, current + delta))
    _overrides[task_type] = new_val
    try:
        await r.set(key, new_val)
        await r.lpush("outcome_log", json.dumps({
            "task": task_type,
            "entropy": entropy_at_stop,
            "rating": user_rating,
            "new_threshold": new_val,
        }))
    except redis.RedisError as exc:
        logger.warning("Redis unavailable, outcome not persisted: %s", exc)
    return new_val
