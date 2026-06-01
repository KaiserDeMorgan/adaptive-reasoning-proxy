import redis.asyncio as redis
import json

r = redis.Redis(decode_responses=True)


async def get_threshold(task_type: str, default: float) -> float:
    """Retrieve the current entropy threshold for a task type from Redis."""
    val = await r.get(f"threshold:{task_type}")
    return float(val) if val else default


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
    current = float(await r.get(key) or 0.5)
    delta = 0.02 * user_rating
    new_val = max(0.1, min(0.9, current + delta))
    await r.set(key, new_val)
    await r.lpush("outcome_log", json.dumps({
        "task": task_type,
        "entropy": entropy_at_stop,
        "rating": user_rating,
        "new_threshold": new_val,
    }))
    return new_val
