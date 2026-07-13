import threading
import time

from config import settings

_CLOSED = "closed"
_OPEN = "open"
_HALF_OPEN = "half_open"


class CircuitBreaker:
    """Per-provider circuit breaker so a provider already known to be down is
    skipped immediately instead of paying its full retry+backoff cost on
    every call before ResilientAIService falls through to the next one."""

    def __init__(self, failure_threshold: int, recovery_timeout: float):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = _CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        """Whether a call to the wrapped provider should be attempted now."""
        with self._lock:
            if self._state == _OPEN:
                if time.monotonic() - self._opened_at >= self.recovery_timeout:
                    # Let exactly one trial request through; everyone else
                    # is rejected until that trial resolves via
                    # record_success()/record_failure().
                    self._state = _HALF_OPEN
                    return True
                return False
            if self._state == _HALF_OPEN:
                return False
            return True

    def record_success(self):
        with self._lock:
            self._state = _CLOSED
            self._failure_count = 0
            self._opened_at = None

    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            if (
                self._state == _HALF_OPEN
                or self._failure_count >= self.failure_threshold
            ):
                self._state = _OPEN
                self._opened_at = time.monotonic()

    @property
    def state(self) -> str:
        return self._state


_registry: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(provider_name: str) -> CircuitBreaker:
    """Return the process-wide shared CircuitBreaker for this provider name.

    Shared across every ResilientAIService instance (RAGEngine, ChatbotService,
    ClassificationService, ...) so a provider failing under one service is
    immediately known to all the others, instead of each rediscovering the
    outage independently — the same class of bug already fixed once for the
    rate limiter by making it a process-wide singleton too."""
    if provider_name not in _registry:
        with _registry_lock:
            if provider_name not in _registry:
                _registry[provider_name] = CircuitBreaker(
                    failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                    recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
                )
    return _registry[provider_name]


def reset_circuit_breakers():
    """Clear all breaker state. Test-only utility for isolation between tests."""
    with _registry_lock:
        _registry.clear()
