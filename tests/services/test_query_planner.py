"""
Tests for services.query_planner.plan_query — the AI structured-extraction
call. All dependencies (provider, rate limiter, cache) are mocked; no live
services required. A failure here must always degrade to None, never raise,
so callers can safely fall back to plain semantic search.
"""

from unittest.mock import MagicMock

import pytest

from services.query_planner import plan_query

_KNOWN_GENRES = {"Fantasy", "Science Fiction", "Non-Fiction"}


@pytest.fixture
def deps():
    provider = MagicMock()
    rate_limiter = MagicMock()
    cache = MagicMock()
    cache.get.return_value = None
    cache.generate_key.return_value = "some-cache-key"
    return provider, rate_limiter, cache


class TestPlanQuerySuccess:
    def test_returns_parsed_dict_on_valid_json(self, deps):
        provider, rate_limiter, cache = deps
        provider.generate.return_value = (
            '{"title": null, "author": null, "genre": "Non-Fiction", '
            '"year_filter": {"operator": ">", "value": 2007}, "semantic_query": null}'
        )
        result = plan_query(
            provider, rate_limiter, cache, "some question", _KNOWN_GENRES
        )
        assert result["genre"] == "Non-Fiction"

    def test_strips_markdown_fences(self, deps):
        provider, rate_limiter, cache = deps
        provider.generate.return_value = '```json\n{"genre": "Fantasy"}\n```'
        result = plan_query(
            provider, rate_limiter, cache, "some question", _KNOWN_GENRES
        )
        assert result["genre"] == "Fantasy"

    def test_caches_the_result(self, deps):
        provider, rate_limiter, cache = deps
        provider.generate.return_value = '{"genre": "Fantasy"}'
        plan_query(provider, rate_limiter, cache, "some question", _KNOWN_GENRES)
        cache.set.assert_called_once()

    def test_acquires_rate_limit_token(self, deps):
        provider, rate_limiter, cache = deps
        provider.generate.return_value = '{"genre": "Fantasy"}'
        plan_query(provider, rate_limiter, cache, "some question", _KNOWN_GENRES)
        rate_limiter.acquire.assert_called_once()

    def test_does_not_release_token_on_success(self, deps):
        provider, rate_limiter, cache = deps
        provider.generate.return_value = '{"genre": "Fantasy"}'
        plan_query(provider, rate_limiter, cache, "some question", _KNOWN_GENRES)
        rate_limiter.release.assert_not_called()

    def test_system_prompt_includes_known_genres(self, deps):
        provider, rate_limiter, cache = deps
        provider.generate.return_value = '{"genre": null}'
        plan_query(provider, rate_limiter, cache, "some question", _KNOWN_GENRES)
        system = provider.generate.call_args.kwargs["system"]
        assert "Fantasy" in system
        assert "Non-Fiction" in system

    def test_uses_deterministic_temperature(self, deps):
        provider, rate_limiter, cache = deps
        provider.generate.return_value = '{"genre": null}'
        plan_query(provider, rate_limiter, cache, "some question", _KNOWN_GENRES)
        assert provider.generate.call_args.kwargs["temperature"] == 0.0

    def test_question_is_wrapped_in_tags(self, deps):
        provider, rate_limiter, cache = deps
        provider.generate.return_value = '{"genre": null}'
        plan_query(
            provider, rate_limiter, cache, "UNIQUE_QUESTION_MARKER", _KNOWN_GENRES
        )
        prompt = provider.generate.call_args.kwargs["prompt"]
        assert "<question>UNIQUE_QUESTION_MARKER</question>" == prompt


class TestPlanQueryCacheHit:
    def test_returns_cached_value_without_calling_provider(self, deps):
        provider, rate_limiter, cache = deps
        cache.get.return_value = {"genre": "Fantasy"}
        result = plan_query(
            provider, rate_limiter, cache, "some question", _KNOWN_GENRES
        )
        assert result == {"genre": "Fantasy"}
        provider.generate.assert_not_called()

    def test_cache_hit_does_not_acquire_rate_limit(self, deps):
        provider, rate_limiter, cache = deps
        cache.get.return_value = {"genre": "Fantasy"}
        plan_query(provider, rate_limiter, cache, "some question", _KNOWN_GENRES)
        rate_limiter.acquire.assert_not_called()


class TestPlanQueryFailureFallback:
    def test_returns_none_on_provider_exception(self, deps):
        provider, rate_limiter, cache = deps
        provider.generate.side_effect = Exception("all providers failed")
        result = plan_query(
            provider, rate_limiter, cache, "some question", _KNOWN_GENRES
        )
        assert result is None

    def test_releases_rate_limit_token_on_provider_exception(self, deps):
        provider, rate_limiter, cache = deps
        provider.generate.side_effect = Exception("all providers failed")
        plan_query(provider, rate_limiter, cache, "some question", _KNOWN_GENRES)
        rate_limiter.release.assert_called_once()

    def test_returns_none_on_malformed_json(self, deps):
        provider, rate_limiter, cache = deps
        provider.generate.return_value = "not valid json at all"
        result = plan_query(
            provider, rate_limiter, cache, "some question", _KNOWN_GENRES
        )
        assert result is None

    def test_returns_none_on_non_object_json(self, deps):
        provider, rate_limiter, cache = deps
        provider.generate.return_value = '["just", "a", "list"]'
        result = plan_query(
            provider, rate_limiter, cache, "some question", _KNOWN_GENRES
        )
        assert result is None

    def test_does_not_cache_a_failed_call(self, deps):
        provider, rate_limiter, cache = deps
        provider.generate.side_effect = Exception("fail")
        plan_query(provider, rate_limiter, cache, "some question", _KNOWN_GENRES)
        cache.set.assert_not_called()

    def test_malformed_json_does_not_raise(self, deps):
        provider, rate_limiter, cache = deps
        provider.generate.return_value = "garbage{{{"
        # Should not raise — must degrade to None for the caller to fall
        # back to plain semantic search.
        result = plan_query(
            provider, rate_limiter, cache, "some question", _KNOWN_GENRES
        )
        assert result is None
