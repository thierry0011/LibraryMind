"""
Tests for infrastructure.usage_tracker.UsageTracker.

datetime.utcnow is mocked throughout so tests are deterministic.
Records are sometimes inserted directly into tracker.records to keep
get_daily_cost tests independent of track() implementation details.
"""

from datetime import datetime as real_datetime
from unittest.mock import patch

import pytest

from infrastructure.usage_tracker import UsageTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TODAY = real_datetime(2026, 6, 19, 10, 0, 0)
YESTERDAY = real_datetime(2026, 6, 18, 23, 59, 59)
TOMORROW = real_datetime(2026, 6, 20, 0, 0, 0)


def _patch_now(dt: real_datetime):
    """Patch datetime in the usage_tracker module, keeping fromisoformat real."""
    p = patch("infrastructure.usage_tracker.datetime")
    mock_dt = p.start()
    mock_dt.utcnow.return_value = dt
    mock_dt.fromisoformat.side_effect = real_datetime.fromisoformat
    return p


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestUsageTrackerInit:
    def test_records_starts_empty(self) -> None:
        assert UsageTracker().records == []

    def test_pricing_contains_gpt_35_turbo(self) -> None:
        assert "gpt-3.5-turbo" in UsageTracker().pricing

    def test_pricing_contains_claude_2(self) -> None:
        assert "claude-2" in UsageTracker().pricing

    def test_pricing_contains_embedding_model(self) -> None:
        assert "text-embedding-3-small" in UsageTracker().pricing

    def test_gpt_35_turbo_input_rate(self) -> None:
        assert UsageTracker().pricing["gpt-3.5-turbo"]["input"] == pytest.approx(0.0005)

    def test_gpt_35_turbo_output_rate(self) -> None:
        assert UsageTracker().pricing["gpt-3.5-turbo"]["output"] == pytest.approx(
            0.0015
        )

    def test_claude_2_input_rate(self) -> None:
        assert UsageTracker().pricing["claude-2"]["input"] == pytest.approx(0.0080)

    def test_claude_2_output_rate(self) -> None:
        assert UsageTracker().pricing["claude-2"]["output"] == pytest.approx(0.0240)

    def test_embedding_input_rate(self) -> None:
        assert UsageTracker().pricing["text-embedding-3-small"][
            "input"
        ] == pytest.approx(0.00002)

    def test_embedding_output_rate_is_zero(self) -> None:
        assert UsageTracker().pricing["text-embedding-3-small"]["output"] == 0.0


# ---------------------------------------------------------------------------
# track — record structure
# ---------------------------------------------------------------------------


class TestTrackRecordStructure:
    @pytest.fixture
    def tracker(self):
        p = _patch_now(TODAY)
        yield UsageTracker()
        p.stop()

    def test_appends_one_record(self, tracker: UsageTracker) -> None:
        tracker.track("gpt-3.5-turbo", 100, 50)
        assert len(tracker.records) == 1

    def test_multiple_calls_append_multiple_records(
        self, tracker: UsageTracker
    ) -> None:
        tracker.track("gpt-3.5-turbo", 100, 50)
        tracker.track("claude-2", 200, 100)
        assert len(tracker.records) == 2

    def test_record_stores_model(self, tracker: UsageTracker) -> None:
        tracker.track("claude-2", 100, 50)
        assert tracker.records[0]["model"] == "claude-2"

    def test_record_stores_prompt_tokens(self, tracker: UsageTracker) -> None:
        tracker.track("gpt-3.5-turbo", 123, 0)
        assert tracker.records[0]["prompt_tokens"] == 123

    def test_record_stores_completion_tokens(self, tracker: UsageTracker) -> None:
        tracker.track("gpt-3.5-turbo", 0, 77)
        assert tracker.records[0]["completion_tokens"] == 77

    def test_record_contains_cost_key(self, tracker: UsageTracker) -> None:
        tracker.track("gpt-3.5-turbo", 100, 100)
        assert "cost" in tracker.records[0]

    def test_record_contains_timestamp_key(self, tracker: UsageTracker) -> None:
        tracker.track("gpt-3.5-turbo", 100, 100)
        assert "timestamp" in tracker.records[0]

    def test_timestamp_is_iso_format_string(self, tracker: UsageTracker) -> None:
        tracker.track("gpt-3.5-turbo", 100, 100)
        ts = tracker.records[0]["timestamp"]
        real_datetime.fromisoformat(ts)  # must not raise

    def test_timestamp_matches_mocked_now(self, tracker: UsageTracker) -> None:
        tracker.track("gpt-3.5-turbo", 100, 100)
        assert tracker.records[0]["timestamp"] == TODAY.isoformat()


# ---------------------------------------------------------------------------
# track — cost calculation
# ---------------------------------------------------------------------------


class TestTrackCostCalculation:
    @pytest.fixture(autouse=True)
    def _freeze(self):
        p = _patch_now(TODAY)
        yield
        p.stop()

    def test_gpt_35_turbo_cost(self) -> None:
        # input: (1000/1000)*0.0005=0.0005, output: (1000/1000)*0.0015=0.0015
        tracker = UsageTracker()
        tracker.track("gpt-3.5-turbo", 1000, 1000)
        assert tracker.records[0]["cost"] == pytest.approx(0.002)

    def test_claude_2_cost(self) -> None:
        # input: (100/1000)*0.008=0.0008, output: (200/1000)*0.024=0.0048
        tracker = UsageTracker()
        tracker.track("claude-2", 100, 200)
        assert tracker.records[0]["cost"] == pytest.approx(0.0056)

    def test_embedding_model_cost(self) -> None:
        # input: (500/1000)*0.00002=0.00001, output: 0
        tracker = UsageTracker()
        tracker.track("text-embedding-3-small", 500, 0)
        assert tracker.records[0]["cost"] == pytest.approx(0.00001)

    def test_unknown_model_cost_is_zero(self) -> None:
        tracker = UsageTracker()
        tracker.track("unknown-model-x", 1000, 1000)
        assert tracker.records[0]["cost"] == pytest.approx(0.0)

    def test_zero_tokens_cost_is_zero(self) -> None:
        tracker = UsageTracker()
        tracker.track("gpt-3.5-turbo", 0, 0)
        assert tracker.records[0]["cost"] == pytest.approx(0.0)

    def test_prompt_only_cost(self) -> None:
        # output=0 so only input cost counts
        tracker = UsageTracker()
        tracker.track("gpt-3.5-turbo", 2000, 0)
        assert tracker.records[0]["cost"] == pytest.approx((2000 / 1000) * 0.0005)

    def test_completion_only_cost(self) -> None:
        tracker = UsageTracker()
        tracker.track("gpt-3.5-turbo", 0, 2000)
        assert tracker.records[0]["cost"] == pytest.approx((2000 / 1000) * 0.0015)

    def test_cost_scales_linearly_with_tokens(self) -> None:
        t1 = UsageTracker()
        t2 = UsageTracker()
        tracker = UsageTracker()
        t1.track("gpt-3.5-turbo", 500, 500)
        t2.track("gpt-3.5-turbo", 1000, 1000)
        tracker.track("gpt-3.5-turbo", 500, 500)
        assert t2.records[0]["cost"] == pytest.approx(t1.records[0]["cost"] * 2)


# ---------------------------------------------------------------------------
# get_daily_cost
# ---------------------------------------------------------------------------


class TestGetDailyCost:
    def _make_record(self, dt: real_datetime, cost: float) -> dict:
        return {"cost": cost, "timestamp": dt.isoformat()}

    def test_returns_zero_when_no_records(self) -> None:
        tracker = UsageTracker()
        p = _patch_now(TODAY)
        try:
            assert tracker.get_daily_cost() == 0.0
        finally:
            p.stop()

    def test_returns_float(self) -> None:
        tracker = UsageTracker()
        p = _patch_now(TODAY)
        try:
            result = tracker.get_daily_cost()
            assert isinstance(result, float)
        finally:
            p.stop()

    def test_sums_single_todays_record(self) -> None:
        tracker = UsageTracker()
        tracker.records = [self._make_record(TODAY, 0.05)]
        p = _patch_now(TODAY)
        try:
            assert tracker.get_daily_cost() == pytest.approx(0.05)
        finally:
            p.stop()

    def test_sums_multiple_todays_records(self) -> None:
        tracker = UsageTracker()
        tracker.records = [
            self._make_record(TODAY.replace(hour=9), 0.01),
            self._make_record(TODAY.replace(hour=14), 0.03),
            self._make_record(TODAY.replace(hour=20), 0.02),
        ]
        p = _patch_now(TODAY)
        try:
            assert tracker.get_daily_cost() == pytest.approx(0.06)
        finally:
            p.stop()

    def test_excludes_yesterdays_records(self) -> None:
        tracker = UsageTracker()
        tracker.records = [self._make_record(YESTERDAY, 0.99)]
        p = _patch_now(TODAY)
        try:
            assert tracker.get_daily_cost() == pytest.approx(0.0)
        finally:
            p.stop()

    def test_excludes_future_records(self) -> None:
        tracker = UsageTracker()
        tracker.records = [self._make_record(TOMORROW, 0.99)]
        p = _patch_now(TODAY)
        try:
            assert tracker.get_daily_cost() == pytest.approx(0.0)
        finally:
            p.stop()

    def test_mixed_dates_returns_only_today(self) -> None:
        tracker = UsageTracker()
        tracker.records = [
            self._make_record(YESTERDAY, 1.0),
            self._make_record(TODAY, 0.07),
            self._make_record(TOMORROW, 2.0),
        ]
        p = _patch_now(TODAY)
        try:
            assert tracker.get_daily_cost() == pytest.approx(0.07)
        finally:
            p.stop()

    def test_zero_cost_records_included_without_error(self) -> None:
        tracker = UsageTracker()
        tracker.records = [self._make_record(TODAY, 0.0), self._make_record(TODAY, 0.0)]
        p = _patch_now(TODAY)
        try:
            assert tracker.get_daily_cost() == pytest.approx(0.0)
        finally:
            p.stop()


# ---------------------------------------------------------------------------
# track — logging side-effect
# ---------------------------------------------------------------------------


class TestTrackLogging:
    def test_logger_info_called_once_per_track(self) -> None:
        p = _patch_now(TODAY)
        tracker = UsageTracker()
        with patch("infrastructure.usage_tracker.logger") as mock_logger:
            tracker.track("gpt-3.5-turbo", 100, 50)
        mock_logger.info.assert_called_once()
        p.stop()

    def test_logger_message_contains_model_name(self) -> None:
        p = _patch_now(TODAY)
        tracker = UsageTracker()
        with patch("infrastructure.usage_tracker.logger") as mock_logger:
            tracker.track("claude-2", 100, 50)
        call_args = mock_logger.info.call_args[0][0]
        assert "claude-2" in call_args
        p.stop()

    def test_logger_message_contains_token_counts(self) -> None:
        p = _patch_now(TODAY)
        tracker = UsageTracker()
        with patch("infrastructure.usage_tracker.logger") as mock_logger:
            tracker.track("gpt-3.5-turbo", 111, 222)
        call_args = mock_logger.info.call_args[0][0]
        assert "111" in call_args
        assert "222" in call_args
        p.stop()

    def test_logger_called_with_timestamp_in_extra(self) -> None:
        p = _patch_now(TODAY)
        tracker = UsageTracker()
        with patch("infrastructure.usage_tracker.logger") as mock_logger:
            tracker.track("gpt-3.5-turbo", 100, 50)
        kwargs = mock_logger.info.call_args[1]
        assert "extra" in kwargs
        assert "timestamp" in kwargs["extra"]
        p.stop()
