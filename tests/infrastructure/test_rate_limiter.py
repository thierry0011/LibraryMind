"""
Tests for infrastructure.rate_limiter.RateLimiter.

time.monotonic is mocked throughout — no real sleeping required.
All settings dependencies are satisfied by conftest.py env-var setup.
"""

import threading
from unittest.mock import patch

import pytest

from infrastructure.rate_limiter import RateLimiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_limiter(max_tokens: int = 5, start_time: float = 1000.0) -> RateLimiter:
    """Return a RateLimiter with a controlled clock and known max_tokens."""
    with (
        patch("infrastructure.rate_limiter.settings") as mock_settings,
        patch("infrastructure.rate_limiter.time.monotonic", return_value=start_time),
    ):
        mock_settings.RATE_LIMIT_PER_MINUTE = max_tokens
        limiter = RateLimiter()
    return limiter


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestRateLimiterInit:
    def test_max_tokens_loaded_from_settings(self) -> None:
        limiter = make_limiter(max_tokens=10)
        assert limiter.max_tokens == 10

    def test_tokens_equal_max_tokens_at_start(self) -> None:
        limiter = make_limiter(max_tokens=15)
        assert limiter.tokens == 15

    def test_last_refill_set_to_monotonic_at_init(self) -> None:
        limiter = make_limiter(start_time=42.0)
        assert limiter.last_refill == 42.0

    def test_lock_is_created(self) -> None:
        limiter = make_limiter()
        assert limiter.lock is not None

    def test_max_tokens_cast_to_int(self) -> None:
        # settings value may be a string (env var raw value)
        with (
            patch("infrastructure.rate_limiter.settings") as ms,
            patch("infrastructure.rate_limiter.time.monotonic", return_value=0.0),
        ):
            ms.RATE_LIMIT_PER_MINUTE = "30"
            limiter = RateLimiter()
        assert limiter.max_tokens == 30
        assert isinstance(limiter.max_tokens, int)


# ---------------------------------------------------------------------------
# _refill — token replenishment logic
# ---------------------------------------------------------------------------


class TestRateLimiterRefill:
    def test_no_refill_when_no_time_elapsed(self) -> None:
        limiter = make_limiter(max_tokens=60, start_time=100.0)
        limiter.tokens = 0
        with patch("infrastructure.rate_limiter.time.monotonic", return_value=100.0):
            limiter._refill()
        assert limiter.tokens == 0

    def test_no_refill_when_elapsed_too_short(self) -> None:
        # 0.5 s with max_tokens=60 → int(0.5 * 1.0) = 0
        limiter = make_limiter(max_tokens=60, start_time=100.0)
        limiter.tokens = 0
        with patch("infrastructure.rate_limiter.time.monotonic", return_value=100.5):
            limiter._refill()
        assert limiter.tokens == 0

    def test_refills_proportionally_to_elapsed(self) -> None:
        # max_tokens=60 → rate = 1 token/s; after 10 s → +10 tokens
        limiter = make_limiter(max_tokens=60, start_time=100.0)
        limiter.tokens = 0
        with patch("infrastructure.rate_limiter.time.monotonic", return_value=110.0):
            limiter._refill()
        assert limiter.tokens == 10

    def test_tokens_capped_at_max(self) -> None:
        limiter = make_limiter(max_tokens=5, start_time=100.0)
        limiter.tokens = 4
        # 60 s elapsed → would add 5 tokens; capped at max_tokens=5
        with patch("infrastructure.rate_limiter.time.monotonic", return_value=160.0):
            limiter._refill()
        assert limiter.tokens == 5

    def test_last_refill_updated_after_successful_refill(self) -> None:
        limiter = make_limiter(max_tokens=60, start_time=100.0)
        limiter.tokens = 0
        with patch("infrastructure.rate_limiter.time.monotonic", return_value=110.0):
            limiter._refill()
        assert limiter.last_refill == 110.0

    def test_last_refill_not_updated_when_refill_amount_is_zero(self) -> None:
        limiter = make_limiter(max_tokens=60, start_time=100.0)
        limiter.tokens = 0
        with patch("infrastructure.rate_limiter.time.monotonic", return_value=100.4):
            limiter._refill()
        assert limiter.last_refill == 100.0  # unchanged


# ---------------------------------------------------------------------------
# acquire — happy path
# ---------------------------------------------------------------------------


class TestRateLimiterAcquireSuccess:
    def test_returns_true_when_tokens_available(self) -> None:
        limiter = make_limiter(max_tokens=5, start_time=0.0)
        with patch("infrastructure.rate_limiter.time.monotonic", return_value=0.0):
            assert limiter.acquire() is True

    def test_decrements_token_count_by_one(self) -> None:
        limiter = make_limiter(max_tokens=5, start_time=0.0)
        with patch("infrastructure.rate_limiter.time.monotonic", return_value=0.0):
            limiter.acquire()
        assert limiter.tokens == 4

    def test_can_acquire_all_available_tokens(self) -> None:
        limiter = make_limiter(max_tokens=3, start_time=0.0)
        with patch("infrastructure.rate_limiter.time.monotonic", return_value=0.0):
            for _ in range(3):
                assert limiter.acquire() is True
        assert limiter.tokens == 0

    def test_refill_called_before_consuming_token(self) -> None:
        # After enough time the bucket refills from 0 back to capacity
        limiter = make_limiter(max_tokens=60, start_time=0.0)
        limiter.tokens = 0
        # 60 s later → refill_amount = 60, capped at 60; then consume 1
        with patch("infrastructure.rate_limiter.time.monotonic", return_value=60.0):
            result = limiter.acquire()
        assert result is True
        assert limiter.tokens == 59


# ---------------------------------------------------------------------------
# acquire — rate-limited path
# ---------------------------------------------------------------------------


class TestRateLimiterAcquireRateLimited:
    def test_raises_when_no_tokens(self) -> None:
        limiter = make_limiter(max_tokens=5, start_time=0.0)
        limiter.tokens = 0
        with patch("infrastructure.rate_limiter.time.monotonic", return_value=0.0):
            with pytest.raises(Exception):
                limiter.acquire()

    def test_exception_message_contains_rate_limit(self) -> None:
        limiter = make_limiter(max_tokens=5, start_time=0.0)
        limiter.tokens = 0
        with patch("infrastructure.rate_limiter.time.monotonic", return_value=0.0):
            with pytest.raises(Exception, match="[Rr]ate limit"):
                limiter.acquire()

    def test_tokens_unchanged_when_rate_limited(self) -> None:
        limiter = make_limiter(max_tokens=5, start_time=0.0)
        limiter.tokens = 0
        with patch("infrastructure.rate_limiter.time.monotonic", return_value=0.0):
            with pytest.raises(Exception):
                limiter.acquire()
        assert limiter.tokens == 0

    def test_raises_after_exhausting_all_tokens(self) -> None:
        limiter = make_limiter(max_tokens=2, start_time=0.0)
        with patch("infrastructure.rate_limiter.time.monotonic", return_value=0.0):
            limiter.acquire()
            limiter.acquire()
            with pytest.raises(Exception):
                limiter.acquire()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestRateLimiterThreadSafety:
    def test_concurrent_acquires_never_exceed_token_budget(self) -> None:
        """N threads compete for tokens; successes must not exceed max_tokens."""
        max_tokens = 10
        limiter = make_limiter(max_tokens=max_tokens, start_time=0.0)

        successes = []
        errors = []
        lock = threading.Lock()

        def try_acquire():
            with patch("infrastructure.rate_limiter.time.monotonic", return_value=0.0):
                try:
                    limiter.acquire()
                    with lock:
                        successes.append(1)
                except Exception:
                    with lock:
                        errors.append(1)

        threads = [threading.Thread(target=try_acquire) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(successes) == max_tokens
        assert len(errors) == 30 - max_tokens

    def test_tokens_never_go_negative_under_concurrency(self) -> None:
        limiter = make_limiter(max_tokens=5, start_time=0.0)

        def try_acquire():
            with patch("infrastructure.rate_limiter.time.monotonic", return_value=0.0):
                try:
                    limiter.acquire()
                except Exception:
                    pass

        threads = [threading.Thread(target=try_acquire) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert limiter.tokens >= 0
