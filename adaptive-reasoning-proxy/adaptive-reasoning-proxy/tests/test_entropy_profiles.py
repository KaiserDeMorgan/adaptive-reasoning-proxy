"""Tests for DTW entropy-profile matching."""

import numpy as np
import pytest

import entropy_profiles as ep


@pytest.fixture(autouse=True)
def _clear_profiles():
    ep._profiles.clear()
    yield
    ep._profiles.clear()


class TestResample:
    def test_resamples_to_fixed_length(self):
        out = ep._resample([0.1, 0.2, 0.3], length=10)
        assert len(out) == 10

    def test_empty_trace_returns_zeros(self):
        out = ep._resample([], length=5)
        assert np.allclose(out, 0.0)

    def test_single_point_is_broadcast(self):
        out = ep._resample([0.7], length=4)
        assert np.allclose(out, 0.7)


class TestDTW:
    def test_identical_sequences_have_zero_distance(self):
        a = np.array([0.1, 0.5, 0.9, 0.3])
        assert ep.dtw_distance(a, a) == pytest.approx(0.0)

    def test_distance_is_symmetric(self):
        a = np.array([0.1, 0.2, 0.3])
        b = np.array([0.3, 0.2, 0.1, 0.0])
        assert ep.dtw_distance(a, b) == pytest.approx(ep.dtw_distance(b, a))

    def test_distance_is_non_negative(self):
        a = np.array([0.1, 0.9, 0.2])
        b = np.array([0.8, 0.1, 0.7])
        assert ep.dtw_distance(a, b) >= 0.0

    def test_time_warped_curves_match_closely(self):
        # Same shape, stretched in time -> small DTW distance.
        base = np.array([0.0, 1.0, 0.0])
        stretched = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        warped = ep.dtw_distance(base, stretched)
        unrelated = ep.dtw_distance(base, np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        assert warped < unrelated


class TestProfileStore:
    def test_best_match_none_when_empty(self):
        assert ep.best_match([0.1, 0.2]) is None

    def test_record_and_match_nearest_task(self):
        # A clearly "factual" low curve and a "reasoning" high curve.
        for _ in range(3):
            ep.record_profile("factual", [0.1, 0.1, 0.05, 0.05])
            ep.record_profile("reasoning", [0.8, 0.85, 0.9, 0.82])

        assert ep.classify_by_profile([0.09, 0.08, 0.06]) == "factual"
        assert ep.classify_by_profile([0.83, 0.88, 0.86]) == "reasoning"

    def test_record_ignores_empty_trace(self):
        ep.record_profile("factual", [])
        assert ep.best_match([0.1]) is None

    def test_max_per_task_is_bounded(self):
        for i in range(ep.MAX_PER_TASK + 10):
            ep.record_profile("code", [0.5, 0.5, 0.5])
        assert len(ep._profiles["code"]) == ep.MAX_PER_TASK

    def test_profile_summary_shape(self):
        ep.record_profile("factual", [0.2, 0.3, 0.1])
        summary = ep.profile_summary()
        assert "factual" in summary
        assert summary["factual"]["count"] == 1
        assert len(summary["factual"]["mean_curve"]) == ep.PROFILE_LEN
