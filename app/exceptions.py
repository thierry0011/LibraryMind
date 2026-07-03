"""
Application-level exception hierarchy for LibraryMind.

All custom exceptions inherit from LibraryMindException so callers can catch
the whole family with one clause when needed.  Select exceptions also inherit
from a matching stdlib type (ValueError, RuntimeError) so legacy code and
existing tests that catch those base types continue to work.
"""


class LibraryMindException(Exception):
    """Base class for all application-level exceptions."""


class RateLimitExceededException(LibraryMindException):
    """Raised by RateLimiter.acquire() when the token bucket is empty."""


class AIProviderException(LibraryMindException, RuntimeError):
    """Raised when every configured AI provider fails after retries.

    Inherits RuntimeError so existing callers that catch RuntimeError
    continue to work without modification.
    """


class InvalidAIResponseException(LibraryMindException, ValueError):
    """Raised when the AI returns output that cannot be parsed as expected.

    Inherits ValueError so existing callers that catch ValueError (and the
    tests that use pytest.raises(ValueError)) continue to work.
    """


class EmbeddingException(LibraryMindException):
    """Raised when the embedding API call fails or returns an unexpected format."""


class VectorStoreException(LibraryMindException):
    """Raised when a ChromaDB operation (upsert or query) fails."""
