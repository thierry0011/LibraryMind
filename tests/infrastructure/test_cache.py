"""
Tests for infrastructure.cache.Cache.

All Redis interaction is mocked — no live Redis instance is required.
The "unavailable" case is simulated by making redis.Redis raise
ConnectionError on construction, which matches the production failure mode.
"""

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest
import redis

from infrastructure.cache import Cache
from config import settings


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    """A mock Redis client whose ping succeeds."""
    client = MagicMock()
    client.ping.return_value = True
    return client


@pytest.fixture
def available(mock_client: MagicMock) -> Cache:
    """Cache whose Redis connection succeeded."""
    with patch("redis.Redis", return_value=mock_client):
        cache = Cache()
    return cache


@pytest.fixture
def unavailable() -> Cache:
    """Cache whose Redis constructor raised ConnectionError."""
    with patch("redis.Redis", side_effect=redis.ConnectionError("Connection refused")):
        cache = Cache()
    return cache


# ---------------------------------------------------------------------------
# __init__ — successful connection
# ---------------------------------------------------------------------------


class TestCacheInitAvailable:
    def test_available_is_true(self, available: Cache) -> None:
        assert available.available is True

    def test_client_attribute_is_set(
        self, available: Cache, mock_client: MagicMock
    ) -> None:
        assert available.client is mock_client

    def test_ttl_loaded_from_settings(self, available: Cache) -> None:
        assert available.ttl == settings.REDIS_TTL

    def test_ping_is_called_once(self, mock_client: MagicMock) -> None:
        with patch("redis.Redis", return_value=mock_client):
            Cache()
        mock_client.ping.assert_called_once()

    def test_redis_constructor_receives_correct_arguments(self) -> None:
        with patch("redis.Redis") as mock_cls:
            mock_cls.return_value.ping.return_value = True
            Cache()
        mock_cls.assert_called_once_with(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True,
        )


# ---------------------------------------------------------------------------
# __init__ — connection failure
# ---------------------------------------------------------------------------


class TestCacheInitUnavailable:
    def test_available_is_false_when_constructor_raises(self) -> None:
        with patch("redis.Redis", side_effect=redis.ConnectionError("refused")):
            cache = Cache()
        assert cache.available is False

    def test_no_exception_propagated_when_constructor_raises(self) -> None:
        with patch("redis.Redis", side_effect=redis.ConnectionError("refused")):
            Cache()  # must not propagate

    def test_available_is_false_when_ping_raises(self, mock_client: MagicMock) -> None:
        mock_client.ping.side_effect = redis.ConnectionError("ping failed")
        with patch("redis.Redis", return_value=mock_client):
            cache = Cache()
        assert cache.available is False

    def test_no_exception_propagated_when_ping_raises(
        self, mock_client: MagicMock
    ) -> None:
        mock_client.ping.side_effect = redis.ConnectionError("ping failed")
        with patch("redis.Redis", return_value=mock_client):
            Cache()  # must not propagate


# ---------------------------------------------------------------------------
# generate_key
# ---------------------------------------------------------------------------


class TestGenerateKey:
    def test_returns_a_string(self, available: Cache) -> None:
        assert isinstance(available.generate_key("prefix", "arg"), str)

    def test_returns_64_char_hex_digest(self, available: Cache) -> None:
        key = available.generate_key("prefix", "value")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_matches_expected_sha256(self, available: Cache) -> None:
        expected = hashlib.sha256("search:term1_term2".encode()).hexdigest()
        assert available.generate_key("search", "term1", "term2") == expected

    def test_same_inputs_produce_same_key(self, available: Cache) -> None:
        assert available.generate_key("p", "a", "b") == available.generate_key(
            "p", "a", "b"
        )

    def test_different_args_produce_different_keys(self, available: Cache) -> None:
        assert available.generate_key("p", "arg1") != available.generate_key(
            "p", "arg2"
        )

    def test_different_prefixes_produce_different_keys(self, available: Cache) -> None:
        assert available.generate_key("x", "arg") != available.generate_key("y", "arg")

    def test_multiple_args_joined_by_underscore(self, available: Cache) -> None:
        expected = hashlib.sha256("q:a_b_c".encode()).hexdigest()
        assert available.generate_key("q", "a", "b", "c") == expected

    def test_numeric_args_are_stringified(self, available: Cache) -> None:
        expected = hashlib.sha256("count:42_3.14".encode()).hexdigest()
        assert available.generate_key("count", 42, 3.14) == expected

    def test_single_arg(self, available: Cache) -> None:
        expected = hashlib.sha256("chat:hello".encode()).hexdigest()
        assert available.generate_key("chat", "hello") == expected


# ---------------------------------------------------------------------------
# get — cache available
# ---------------------------------------------------------------------------


class TestCacheGetAvailable:
    def test_returns_none_when_key_missing(
        self, available: Cache, mock_client: MagicMock
    ) -> None:
        mock_client.get.return_value = None
        assert available.get("missing") is None

    def test_returns_dict_for_json_object(
        self, available: Cache, mock_client: MagicMock
    ) -> None:
        data = {"title": "Dune", "author": "Herbert"}
        mock_client.get.return_value = json.dumps(data)
        assert available.get("book:1") == data

    def test_returns_list_for_json_array(
        self, available: Cache, mock_client: MagicMock
    ) -> None:
        data = ["Python", "FastAPI", "Redis"]
        mock_client.get.return_value = json.dumps(data)
        assert available.get("tags") == data

    def test_returns_integer_for_json_number(
        self, available: Cache, mock_client: MagicMock
    ) -> None:
        mock_client.get.return_value = json.dumps(99)
        assert available.get("count") == 99

    def test_calls_client_get_with_exact_key(
        self, available: Cache, mock_client: MagicMock
    ) -> None:
        mock_client.get.return_value = None
        available.get("exact_key")
        mock_client.get.assert_called_once_with("exact_key")

    def test_returns_none_when_stored_value_is_not_a_string(
        self, available: Cache, mock_client: MagicMock
    ) -> None:
        mock_client.get.return_value = (
            99999  # unexpected non-str (e.g., decode_responses off)
        )
        assert available.get("bad_key") is None

    def test_returns_none_for_empty_string_value(
        self, available: Cache, mock_client: MagicMock
    ) -> None:
        mock_client.get.return_value = ""  # falsy string — treated as cache miss
        assert available.get("empty_key") is None


# ---------------------------------------------------------------------------
# get — cache unavailable
# ---------------------------------------------------------------------------


class TestCacheGetUnavailable:
    def test_returns_none(self, unavailable: Cache) -> None:
        assert unavailable.get("any_key") is None

    def test_does_not_touch_redis(self, mock_client: MagicMock) -> None:
        mock_client.ping.side_effect = redis.ConnectionError()
        with patch("redis.Redis", return_value=mock_client):
            cache = Cache()
        cache.get("key")
        mock_client.get.assert_not_called()


# ---------------------------------------------------------------------------
# set — cache available
# ---------------------------------------------------------------------------


class TestCacheSetAvailable:
    def test_calls_setex_once(self, available: Cache, mock_client: MagicMock) -> None:
        available.set("k", {"v": 1})
        mock_client.setex.assert_called_once()

    def test_setex_receives_correct_key(
        self, available: Cache, mock_client: MagicMock
    ) -> None:
        available.set("my_key", "value")
        key_arg = mock_client.setex.call_args[0][0]
        assert key_arg == "my_key"

    def test_setex_receives_ttl_from_settings(
        self, available: Cache, mock_client: MagicMock
    ) -> None:
        available.set("k", "v")
        ttl_arg = mock_client.setex.call_args[0][1]
        assert ttl_arg == settings.REDIS_TTL

    def test_dict_value_is_json_serialised(
        self, available: Cache, mock_client: MagicMock
    ) -> None:
        data = {"title": "Dune"}
        available.set("book:1", data)
        stored = mock_client.setex.call_args[0][2]
        assert json.loads(stored) == data

    def test_list_value_is_json_serialised(
        self, available: Cache, mock_client: MagicMock
    ) -> None:
        data = [1, 2, 3]
        available.set("nums", data)
        stored = mock_client.setex.call_args[0][2]
        assert json.loads(stored) == data

    def test_string_value_is_json_serialised(
        self, available: Cache, mock_client: MagicMock
    ) -> None:
        available.set("greeting", "hello")
        stored = mock_client.setex.call_args[0][2]
        assert json.loads(stored) == "hello"


# ---------------------------------------------------------------------------
# set — cache unavailable
# ---------------------------------------------------------------------------


class TestCacheSetUnavailable:
    def test_does_not_call_setex(self, mock_client: MagicMock) -> None:
        mock_client.ping.side_effect = redis.ConnectionError()
        with patch("redis.Redis", return_value=mock_client):
            cache = Cache()
        cache.set("key", "value")
        mock_client.setex.assert_not_called()

    def test_no_exception_raised(self, unavailable: Cache) -> None:
        unavailable.set("key", {"data": 1})  # must not raise


# ---------------------------------------------------------------------------
# Round-trip (set → get)
# ---------------------------------------------------------------------------


class TestCacheRoundTrip:
    def test_get_after_set_returns_original_value(self) -> None:
        """Simulates a full cache cycle using an in-memory dict as the backing store."""
        store: dict = {}

        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.setex.side_effect = lambda key, ttl, value: store.__setitem__(
            key, value
        )
        mock_client.get.side_effect = lambda key: store.get(key)

        with patch("redis.Redis", return_value=mock_client):
            cache = Cache()

        payload = {"query": "machine learning", "results": [1, 2, 3]}
        key = cache.generate_key("search", "machine learning")
        cache.set(key, payload)

        assert cache.get(key) == payload
