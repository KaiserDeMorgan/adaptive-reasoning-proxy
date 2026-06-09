"""Tests for the Shannon-entropy engine that drives early stopping."""

import math

import numpy as np
import pytest

from entropy_engine import EntropyState


def _peaked_logprobs() -> dict[str, float]:
    """One token carries almost all probability mass -> entropy near 0."""
    return {"a": math.log(0.97), "b": math.log(0.01), "c": math.log(0.01), "d": math.log(0.01)}


def _uniform_logprobs() -> dict[str, float]:
    """Four equally likely tokens -> entropy near ln(4) ~= 1.386."""
    p = math.log(0.25)
    return {"a": p, "b": p, "c": p, "d": p}


class TestUpdate:
    def test_uniform_distribution_returns_max_entropy(self):
        state = EntropyState()
        h = state.update(_uniform_logprobs())
        # H for 4 uniform outcomes is ln(4).
        assert h == pytest.approx(math.log(4), abs=1e-3)

    def test_peaked_distribution_returns_low_entropy(self):
        state = EntropyState()
        h = state.update(_peaked_logprobs())
        assert h < 0.25

    def test_update_increments_token_count(self):
        state = EntropyState()
        for _ in range(5):
            state.update(_uniform_logprobs())
        assert state.tokens_generated == 5

    def test_window_is_capped_at_20(self):
        state = EntropyState()
        for _ in range(50):
            state.update(_uniform_logprobs())
        assert len(state.window) == 20

    def test_ema_tracks_toward_recent_values(self):
        state = EntropyState()
        # Start high, then feed many low-entropy tokens; EMA should fall.
        state.update(_uniform_logprobs())
        high_ema = state.ema
        for _ in range(30):
            state.update(_peaked_logprobs())
        assert state.ema < high_ema

    def test_entropy_is_non_negative(self):
        state = EntropyState()
        for lp in (_uniform_logprobs(), _peaked_logprobs()):
            assert state.update(lp) >= 0.0


class TestShouldStop:
    def test_does_not_stop_before_min_tokens(self):
        state = EntropyState()
        for _ in range(20):  # below default min_tokens=40
            state.update(_peaked_logprobs())
        assert state.should_stop(threshold=0.35) is False

    def test_does_not_stop_with_tiny_window(self):
        state = EntropyState()
        # Force token count high but window short by inspecting guard directly.
        state.tokens_generated = 100
        for _ in range(5):  # window has only 5 entries (< 10)
            state.window.append(0.01)
        assert state.should_stop(threshold=0.35) is False

    def test_stops_on_sustained_low_entropy(self):
        state = EntropyState()
        for _ in range(50):
            state.update(_peaked_logprobs())
        assert state.should_stop(threshold=0.35) is True

    def test_does_not_stop_on_sustained_high_entropy(self):
        state = EntropyState()
        for _ in range(50):
            state.update(_uniform_logprobs())
        assert state.should_stop(threshold=0.35) is False

    def test_threshold_boundary_keeps_running_when_above(self):
        state = EntropyState()
        for _ in range(50):
            state.update(_peaked_logprobs())
        # Entropy is ~0.16; a threshold below that must not trigger a stop.
        assert state.should_stop(threshold=0.05) is False

    def test_returns_plain_bool_not_numpy(self):
        state = EntropyState()
        for _ in range(50):
            state.update(_peaked_logprobs())
        result = state.should_stop(threshold=0.35)
        assert isinstance(result, bool)
        assert not isinstance(result, np.bool_)
