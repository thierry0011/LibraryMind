"""Tests for infrastructure.circuit_breaker.CircuitBreaker and the registry."""

from unittest.mock import patch

import pytest

from infrastructure.circuit_breaker import (
    CircuitBreaker,
    get_circuit_breaker,
    reset_circuit_breakers,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


# ---------------------------------------------------------------------------
# CircuitBreaker — closed state (happy path)
# ---------------------------------------------------------------------------


class TestCircuitBreakerClosedState:
    def test_starts_closed(self):
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        assert breaker.state == "closed"

    def test_allows_request_when_closed(self):
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        assert breaker.allow_request() is True

    def test_single_failure_does_not_open_below_threshold(self):
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        breaker.record_failure()
        assert breaker.state == "closed"
        assert breaker.allow_request() is True

    def test_success_resets_failure_count(self):
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        assert breaker.state == "closed"


# ---------------------------------------------------------------------------
# CircuitBreaker — opening on threshold
# ---------------------------------------------------------------------------


class TestCircuitBreakerOpensOnThreshold:
    def test_opens_after_reaching_failure_threshold(self):
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "open"

    def test_rejects_requests_once_open(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=30)
        breaker.record_failure()
        assert breaker.allow_request() is False

    def test_does_not_open_before_threshold_reached(self):
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)
        for _ in range(4):
            breaker.record_failure()
        assert breaker.state == "closed"


# ---------------------------------------------------------------------------
# CircuitBreaker — recovery (half-open)
# ---------------------------------------------------------------------------


class TestCircuitBreakerRecovery:
    def test_stays_open_before_recovery_timeout_elapses(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=30)
        breaker.record_failure()
        assert breaker.allow_request() is False

    def test_allows_one_trial_request_after_recovery_timeout(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        breaker.record_failure()
        assert breaker.allow_request() is True

    def test_second_concurrent_request_rejected_while_trial_in_flight(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        breaker.record_failure()
        assert breaker.allow_request() is True  # the one trial request
        assert breaker.allow_request() is False  # trial still unresolved

    def test_successful_trial_closes_the_breaker(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        breaker.record_failure()
        breaker.allow_request()  # enters half-open
        breaker.record_success()
        assert breaker.state == "closed"
        assert breaker.allow_request() is True

    def test_failed_trial_reopens_the_breaker(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        breaker.record_failure()
        breaker.allow_request()  # enters half-open
        breaker.record_failure()
        assert breaker.state == "open"


# ---------------------------------------------------------------------------
# get_circuit_breaker() — process-wide registry
# ---------------------------------------------------------------------------


class TestCircuitBreakerRegistry:
    def test_same_provider_name_returns_same_instance(self):
        assert get_circuit_breaker("openai") is get_circuit_breaker("openai")

    def test_different_provider_names_get_independent_breakers(self):
        openai_breaker = get_circuit_breaker("openai")
        anthropic_breaker = get_circuit_breaker("anthropic")
        openai_breaker.record_failure()
        openai_breaker.record_failure()
        openai_breaker.record_failure()
        assert openai_breaker.state == "open"
        assert anthropic_breaker.state == "closed"

    def test_uses_settings_for_failure_threshold(self):
        with patch("infrastructure.circuit_breaker.settings") as mock_settings:
            mock_settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD = 1
            mock_settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 30
            breaker = get_circuit_breaker("openai")
            breaker.record_failure()
        assert breaker.state == "open"

    def test_reset_circuit_breakers_clears_state(self):
        breaker = get_circuit_breaker("openai")
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "open"

        reset_circuit_breakers()

        fresh_breaker = get_circuit_breaker("openai")
        assert fresh_breaker.state == "closed"
