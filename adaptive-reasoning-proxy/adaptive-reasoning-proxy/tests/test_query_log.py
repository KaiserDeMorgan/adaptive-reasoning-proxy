"""Tests for the async SQLite query log and stats aggregation."""

import os

import pytest

import query_log


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "ql.db"
    monkeypatch.setattr(query_log, "DB_PATH", str(db))
    return str(db)


async def test_init_creates_table(temp_db):
    await query_log.init_db()
    assert os.path.exists(temp_db)


async def test_log_and_recent_roundtrip(temp_db):
    await query_log.init_db()
    await query_log.log_query("factual", 42, True, 80, 0.21)
    await query_log.log_query("reasoning", 120, False, 200, 0.60)

    rows = await query_log.recent(10)
    assert len(rows) == 2
    # Newest first.
    assert rows[0]["task_type"] == "reasoning"
    assert rows[0]["early_stop"] is False
    assert rows[1]["task_type"] == "factual"
    assert rows[1]["early_stop"] is True


async def test_stats_aggregates_savings(temp_db, monkeypatch):
    monkeypatch.setattr(query_log, "BASELINE_TOKENS", 256)
    await query_log.init_db()
    # Two early stops (savings) and one full run (no savings).
    await query_log.log_query("factual", 56, True, 80, 0.2)   # saves 200
    await query_log.log_query("code", 156, True, 90, 0.3)     # saves 100
    await query_log.log_query("reasoning", 300, False, 250, 0.7)  # saves 0

    s = await query_log.stats()
    assert s["total_queries"] == 3
    assert s["early_stops"] == 2
    assert s["stop_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["tokens_saved"] == 300  # 200 + 100
    assert s["cost_saved_usd"] > 0


async def test_stats_empty_db_is_safe(temp_db):
    await query_log.init_db()
    s = await query_log.stats()
    assert s["total_queries"] == 0
    assert s["stop_rate"] == 0.0
    assert s["tokens_saved"] == 0


def test_latest_trace_roundtrip():
    query_log.set_latest_trace("factual", [0.5, 0.3], [0.5, 0.4], stop_index=2, threshold=0.35)
    t = query_log.get_latest_trace()
    assert t["task_type"] == "factual"
    assert t["entropy"] == [0.5, 0.3]
    assert t["stop_index"] == 2
    assert t["threshold"] == 0.35
