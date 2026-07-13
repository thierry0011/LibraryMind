"""Tests for ResilientAIService."""

import pytest
from unittest.mock import MagicMock, patch

from providers.resilient_ai_service import ResilientAIService
from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from app.infrastructure.circuit_breaker import (
    get_circuit_breaker,
    reset_circuit_breakers,
)
from config import settings as cfg_settings


@pytest.fixture(autouse=True)
def _clean_circuit_breakers():
    """The circuit breaker registry is a process-wide singleton by design
    (see infrastructure/circuit_breaker.py) so every ResilientAIService
    instance shares it — but that means it must be reset between tests,
    or one test's provider failures would leak into the next test's."""
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


@pytest.fixture
def service_with_mocks():
    """Instantiates ResilientAIService then replaces real providers with mocks."""
    svc = ResilientAIService()
    mock_primary = MagicMock()
    mock_primary.provider = "openai"
    mock_fallback = MagicMock()
    mock_fallback.provider = "anthropic"
    svc.providers = [mock_primary, mock_fallback]
    return svc, mock_primary, mock_fallback


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestResilientAIServiceInit:
    def test_creates_two_providers(self):
        svc = ResilientAIService()
        assert len(svc.providers) == 2

    def test_openai_is_first_when_primary_is_openai(self):
        with patch.object(cfg_settings, "PRIMARY_PROVIDER", "openai"):
            svc = ResilientAIService()
        assert isinstance(svc.providers[0], OpenAIProvider)

    def test_anthropic_is_first_when_primary_is_not_openai(self):
        with patch.object(cfg_settings, "PRIMARY_PROVIDER", "anthropic"):
            svc = ResilientAIService()
        assert isinstance(svc.providers[0], AnthropicProvider)

    def test_second_provider_is_different_from_first_when_openai_primary(self):
        with patch.object(cfg_settings, "PRIMARY_PROVIDER", "openai"):
            svc = ResilientAIService()
        assert isinstance(svc.providers[1], AnthropicProvider)

    def test_second_provider_is_different_from_first_when_anthropic_primary(self):
        with patch.object(cfg_settings, "PRIMARY_PROVIDER", "anthropic"):
            svc = ResilientAIService()
        assert isinstance(svc.providers[1], OpenAIProvider)


# ---------------------------------------------------------------------------
# generate() — happy path
# ---------------------------------------------------------------------------


class TestResilientAIServiceGenerate:
    def test_returns_result_from_first_provider(self, service_with_mocks):
        svc, mock_primary, _ = service_with_mocks
        mock_primary.generate.return_value = "Primary response"
        assert svc.generate("prompt", "system") == "Primary response"

    def test_does_not_call_second_provider_when_first_succeeds(
        self, service_with_mocks
    ):
        svc, mock_primary, mock_fallback = service_with_mocks
        mock_primary.generate.return_value = "ok"
        svc.generate("prompt", "system")
        mock_fallback.generate.assert_not_called()

    def test_calls_first_provider_with_correct_args(self, service_with_mocks):
        svc, mock_primary, _ = service_with_mocks
        mock_primary.generate.return_value = "ok"
        svc.generate("my prompt", "my system")
        mock_primary.generate.assert_called_once_with("my prompt", "my system")


# ---------------------------------------------------------------------------
# generate() — fallback behaviour
# ---------------------------------------------------------------------------


class TestResilientAIServiceFallback:
    def test_falls_back_to_second_when_first_raises(self, service_with_mocks):
        svc, mock_primary, mock_fallback = service_with_mocks
        mock_primary.generate.side_effect = RuntimeError("Primary down")
        mock_fallback.generate.return_value = "Fallback response"
        assert svc.generate("prompt", "system") == "Fallback response"

    def test_calls_second_provider_with_correct_args_on_fallback(
        self, service_with_mocks
    ):
        svc, mock_primary, mock_fallback = service_with_mocks
        mock_primary.generate.side_effect = Exception("fail")
        mock_fallback.generate.return_value = "ok"
        svc.generate("my prompt", "my system")
        mock_fallback.generate.assert_called_once_with("my prompt", "my system")

    def test_first_provider_is_still_tried_before_fallback(self, service_with_mocks):
        svc, mock_primary, mock_fallback = service_with_mocks
        mock_primary.generate.side_effect = Exception("fail")
        mock_fallback.generate.return_value = "ok"
        svc.generate("p", "s")
        mock_primary.generate.assert_called_once()

    def test_raises_runtime_error_when_all_providers_fail(self, service_with_mocks):
        svc, mock_primary, mock_fallback = service_with_mocks
        mock_primary.generate.side_effect = RuntimeError("Primary failed")
        mock_fallback.generate.side_effect = RuntimeError("Fallback failed")
        with pytest.raises(RuntimeError, match="All providers failed"):
            svc.generate("prompt", "system")

    def test_fallback_result_is_a_string(self, service_with_mocks):
        svc, mock_primary, mock_fallback = service_with_mocks
        mock_primary.generate.side_effect = Exception("fail")
        mock_fallback.generate.return_value = "string result"
        result = svc.generate("p", "s")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# generate() — circuit breaker integration
# ---------------------------------------------------------------------------


class TestResilientAIServiceCircuitBreaker:
    def test_open_breaker_skips_provider_without_calling_it(self, service_with_mocks):
        svc, mock_primary, mock_fallback = service_with_mocks
        breaker = get_circuit_breaker("openai")
        for _ in range(cfg_settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD):
            breaker.record_failure()
        mock_fallback.generate.return_value = "fallback response"

        result = svc.generate("prompt", "system")

        mock_primary.generate.assert_not_called()
        assert result == "fallback response"

    def test_repeated_failures_open_the_breaker(self, service_with_mocks):
        svc, mock_primary, mock_fallback = service_with_mocks
        mock_primary.generate.side_effect = Exception("down")
        mock_fallback.generate.return_value = "ok"

        for _ in range(cfg_settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD):
            svc.generate("prompt", "system")

        assert get_circuit_breaker("openai").state == "open"

    def test_success_keeps_breaker_closed(self, service_with_mocks):
        svc, mock_primary, _ = service_with_mocks
        mock_primary.generate.return_value = "ok"

        svc.generate("prompt", "system")

        assert get_circuit_breaker("openai").state == "closed"

    def test_success_resets_a_previously_degraded_breaker(self, service_with_mocks):
        svc, mock_primary, _ = service_with_mocks
        breaker = get_circuit_breaker("openai")
        breaker.record_failure()
        mock_primary.generate.return_value = "ok"

        svc.generate("prompt", "system")

        assert breaker.state == "closed"

    def test_both_providers_open_raises_without_calling_either(
        self, service_with_mocks
    ):
        svc, mock_primary, mock_fallback = service_with_mocks
        for name in ("openai", "anthropic"):
            breaker = get_circuit_breaker(name)
            for _ in range(cfg_settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD):
                breaker.record_failure()

        with pytest.raises(RuntimeError, match="All providers failed"):
            svc.generate("prompt", "system")

        mock_primary.generate.assert_not_called()
        mock_fallback.generate.assert_not_called()
