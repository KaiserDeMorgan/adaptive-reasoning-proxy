"""Async SQLite query log + aggregate stats for the dashboard.

SQLite is used as a zero-setup stand-in for the Postgres log table described
in the project plan: it needs no server, runs anywhere, and keeps the proxy
self-contained. All writes are best-effort and never raise into the hot path.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("QUERY_LOG_DB", str(Path(__file__).parent / "query_log.db"))

# These are tunable estimate knobs
# BASELINE_TOKENS is the assumed completion length had we not stopped early
# savings are measured against it. COST_PER_1K is a representative price.
BASELINE_TOKENS = int(os.getenv("BASELINE_TOKENS", "256"))
COST_PER_1K = float(os.getenv("COST_PER_1K_TOKENS", "0.60"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                REAL    NOT NULL,
    task_type         TEXT    NOT NULL,
    tokens_generated  INTEGER NOT NULL,
    early_stop        INTEGER NOT NULL,
    latency_ms        INTEGER NOT NULL,
    final_entropy_ema REAL    NOT NULL
);
"""

# Latest per-token entropy trace, held in memory for the live chart.
_latest_trace: dict[str, Any] = {
    "task_type": None,
    "entropy": [],   # H(t) per token
    "ema": [],       # smoothed EMA per token
    "stop_index": None,
    "threshold": None,
}


async def init_db() -> None:
    """Create the log table if it does not yet exist."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(_SCHEMA)
            await db.commit()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not initialise query log DB: %s", exc)


async def log_query(
    task_type: str,
    tokens_generated: int,
    early_stop: bool,
    latency_ms: int,
    final_entropy_ema: float,
) -> None:
    """Persist one completed request. Best-effort: never raises."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO query_log "
                "(ts, task_type, tokens_generated, early_stop, latency_ms, final_entropy_ema) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), task_type, tokens_generated, int(early_stop),
                 latency_ms, final_entropy_ema),
            )
            await db.commit()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not write query log: %s", exc)


async def recent(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent queries, newest first."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT ts, task_type, tokens_generated, early_stop, "
                "latency_ms, final_entropy_ema "
                "FROM query_log ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
        return [
            {
                "ts": r["ts"],
                "task_type": r["task_type"],
                "tokens_generated": r["tokens_generated"],
                "early_stop": bool(r["early_stop"]),
                "latency_ms": r["latency_ms"],
                "final_entropy_ema": r["final_entropy_ema"],
            }
            for r in rows
        ]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not read query log: %s", exc)
        return []


async def stats() -> dict[str, Any]:
    """Aggregate dashboard metrics across all logged queries.

    ``tokens_saved`` is measured against ``BASELINE_TOKENS`` for early-stopped
    requests only, and is an estimate (see module docstring).
    """
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT "
                "  COUNT(*) AS total, "
                "  COALESCE(SUM(early_stop), 0) AS stops, "
                "  COALESCE(AVG(latency_ms), 0) AS avg_latency, "
                "  COALESCE(SUM(CASE WHEN early_stop = 1 "
                "    THEN MAX(? - tokens_generated, 0) ELSE 0 END), 0) AS tokens_saved "
                "FROM query_log",
                (BASELINE_TOKENS,),
            )
            row = await cursor.fetchone()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not aggregate stats: %s", exc)
        row = None

    total = row["total"] if row else 0
    stops = row["stops"] if row else 0
    avg_latency = row["avg_latency"] if row else 0
    tokens_saved = row["tokens_saved"] if row else 0
    stop_rate = (stops / total) if total else 0.0
    cost_saved = (tokens_saved / 1000.0) * COST_PER_1K

    return {
        "total_queries": total,
        "early_stops": stops,
        "stop_rate": round(stop_rate, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "tokens_saved": tokens_saved,
        "cost_saved_usd": round(cost_saved, 4),
        "baseline_tokens": BASELINE_TOKENS,
    }


async def entropy_by_task() -> list[dict[str, Any]]:
    """Average final entropy at stop, grouped by task type.

    Scoped to early-stopped requests only, since those are the ones whose
    final entropy was actually driven by the per-task threshold decision —
    this is what lets a threshold be checked against where stops are
    actually landing, instead of eyeballing individual rows.
    """
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT task_type, AVG(final_entropy_ema) AS avg_entropy_at_stop, "
                "COUNT(*) AS n "
                "FROM query_log WHERE early_stop = 1 GROUP BY task_type"
            )
            rows = await cursor.fetchall()
        return [
            {
                "task_type": r["task_type"],
                "avg_entropy_at_stop": round(r["avg_entropy_at_stop"], 4),
                "n": r["n"],
            }
            for r in rows
        ]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not aggregate entropy by task: %s", exc)
        return []


def set_latest_trace(
    task_type: str,
    entropy: list[float],
    ema: list[float],
    stop_index: Optional[int],
    threshold: float,
) -> None:
    """Record the most recent request's entropy trace for the live chart."""
    _latest_trace.update(
        task_type=task_type,
        entropy=entropy,
        ema=ema,
        stop_index=stop_index,
        threshold=threshold,
    )


def get_latest_trace() -> dict[str, Any]:
    return dict(_latest_trace)
