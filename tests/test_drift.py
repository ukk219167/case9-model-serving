"""
tests/test_drift.py
-------------------
Unit tests for the DriftMonitor.
All tests use an in-memory SQLite DB — no filesystem side effects.
"""

from __future__ import annotations

import pytest

from app.drift import DriftMonitor, _mean, _non_ascii_rate, _oov_rate, _std, _z_score

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def monitor() -> DriftMonitor:
    """Fresh in-memory monitor for each test."""
    return DriftMonitor(db_path=":memory:", window_size=10, alert_threshold=2.0)


# ---------------------------------------------------------------------------
# Signal extraction helpers
# ---------------------------------------------------------------------------


def test_non_ascii_rate_pure_ascii():
    assert _non_ascii_rate("hello world") == 0.0


def test_non_ascii_rate_mixed():
    rate = _non_ascii_rate("héllo")  # 'é' is non-ASCII
    assert 0.0 < rate < 1.0


def test_non_ascii_rate_empty():
    assert _non_ascii_rate("") == 0.0


def test_oov_rate_all_known(monkeypatch):
    """Words in the vocab produce 0.0 OOV rate."""
    from app import drift as drift_mod

    monkeypatch.setattr(drift_mod, "_load_vocab", lambda: frozenset(["hello", "world"]))
    mon = DriftMonitor(db_path=":memory:")
    # Direct call to helper
    assert _oov_rate("hello world", frozenset(["hello", "world"])) == 0.0


def test_oov_rate_all_unknown():
    rate = _oov_rate("xyzzy qwerty asdfgh", frozenset(["the", "a"]))
    assert rate == 1.0


def test_oov_rate_empty_text():
    assert _oov_rate("", frozenset(["the"])) == 0.0


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def test_mean_basic():
    assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_mean_empty():
    assert _mean([]) == 0.0


def test_std_basic():
    std = _std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], mean=5.0)
    assert std == pytest.approx(2.0, rel=0.01)


def test_std_single_value():
    assert _std([5.0], mean=5.0) == 0.0


def test_z_score_zero_baseline_std():
    """Zero baseline std should return 0.0 to avoid division by zero."""
    assert _z_score(window_mean=5.0, baseline_mean=5.0, baseline_std=0.0) == 0.0


def test_z_score_positive():
    z = _z_score(window_mean=10.0, baseline_mean=5.0, baseline_std=2.5)
    assert z == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# DriftMonitor.record()
# ---------------------------------------------------------------------------


def test_record_returns_bool(monitor: DriftMonitor):
    result = monitor.record("This is a normal English sentence.")
    assert isinstance(result, bool)


def test_record_increments_count(monitor: DriftMonitor):
    assert monitor._total_count() == 0
    monitor.record("First request.")
    monitor.record("Second request.")
    assert monitor._total_count() == 2


def test_record_no_alert_on_normal_text(monitor: DriftMonitor):
    """Normal English sentences should not trigger a drift alert."""
    for _ in range(10):
        flagged = monitor.record("The movie was really quite good actually.")
    assert not flagged


def test_record_alert_on_extreme_length(monitor: DriftMonitor):
    """
    Very long texts (10x the baseline mean) should eventually push
    the window mean far enough to trigger a length alert.
    """
    long_text = "word " * 2000  # ~10,000 chars vs baseline ~72
    for _ in range(10):
        flagged = monitor.record(long_text)
    assert flagged, "Expected drift alert for extreme text length"


def test_record_alert_on_non_ascii(monitor: DriftMonitor):
    """Text composed entirely of non-ASCII chars should trigger non_ascii_rate alert."""
    unicode_text = "αβγδεζηθ " * 50
    for _ in range(10):
        flagged = monitor.record(unicode_text)
    assert flagged, "Expected drift alert for high non-ASCII rate"


# ---------------------------------------------------------------------------
# DriftMonitor.report()
# ---------------------------------------------------------------------------


def test_report_structure_empty(monitor: DriftMonitor):
    """Report with no data should return zero-state without crashing."""
    report = monitor.report()
    assert "window_size" in report
    assert "requests_seen" in report
    assert "drift_detected" in report
    assert "signals" in report
    assert set(report["signals"].keys()) == {"text_length", "oov_rate", "non_ascii_rate"}


def test_report_requests_seen(monitor: DriftMonitor):
    monitor.record("One.")
    monitor.record("Two.")
    report = monitor.report()
    assert report["requests_seen"] == 2


def test_report_signal_fields(monitor: DriftMonitor):
    monitor.record("A perfectly ordinary sentence.")
    report = monitor.report()
    for signal in report["signals"].values():
        assert "baseline_mean" in signal
        assert "baseline_std" in signal
        assert "window_mean" in signal
        assert "window_std" in signal
        assert "z_score" in signal
        assert "alert" in signal


def test_report_window_respects_window_size():
    """Monitor with window_size=3 should only consider the last 3 requests."""
    mon = DriftMonitor(db_path=":memory:", window_size=3, alert_threshold=2.0)
    # Record 10 normal texts then 3 extreme ones
    for _ in range(10):
        mon.record("Short.")
    for _ in range(3):
        mon.record("word " * 2000)

    rows = mon._fetch_window()
    assert len(rows) == 3
    # All three rows should have very high text_length
    assert all(r[0] > 5000 for r in rows)
