"""
Tests for BaseProvider.

BaseProvider is abstract, so a minimal concrete subclass is defined here for each
provider path (openai / anthropic). All HTTP calls are replaced with MagicMock so
no real network requests are made.
"""

import pytest
import httpx
from unittest.mock import MagicMock

from providers.base import BaseProvider
from config import settings


# ---------------------------------------------------------------------------
# Minimal concrete subclasses used only in this test module
# ---------------------------------------------------------------------------


class _OpenAILikeProvider(BaseProvider):
    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return "gpt-3.5-turbo"

    def messages(self, prompt: str, system: str) -> list:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]


class _AnthropicLikeProvider(BaseProvider):
    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return "claude-2"

    def messages(self, prompt: str, system: str) -> list:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def openai_provider():
    p = _OpenAILikeProvider()
    p.client = MagicMock()
    return p


@pytest.fixture
def anthropic_provider():
    p = _AnthropicLikeProvider()
    p.client = MagicMock()
    return p


def _mock_response(json_data: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = json_data
    return resp


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestBaseProviderInit:
    def test_reads_api_key_from_settings(self):
        p = _OpenAILikeProvider()
        assert p.api_key == settings.AMALIAI_API_KEY
        p.close()

    def test_reads_base_url_from_settings(self):
        p = _OpenAILikeProvider()
        assert p.base_url == settings.AMALIAI_BASE_URL
        p.close()

    def test_reads_max_tokens_from_settings(self):
        p = _OpenAILikeProvider()
        assert p.max_tokens == settings.MAX_TOKENS
        p.close()

    def test_reads_temperature_from_settings(self):
        p = _OpenAILikeProvider()
        assert p.temperature == settings.TEMPERATURE
        p.close()

    def test_creates_httpx_client(self):
        p = _OpenAILikeProvider()
        assert isinstance(p.client, httpx.Client)
        p.close()


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestBaseProviderClose:
    def test_delegates_to_client_close(self):
        p = _OpenAILikeProvider()
        p.client = MagicMock()
        p.close()
        p.client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Abstract interface enforcement
# ---------------------------------------------------------------------------


class TestBaseProviderAbstract:
    def test_cannot_instantiate_base_directly(self):
        with pytest.raises(TypeError):
            BaseProvider()  # type: ignore[abstract]

    def test_subclass_missing_provider_cannot_be_instantiated(self):
        class Incomplete(BaseProvider):
            @property
            def model(self):
                return "m"

            def messages(self, p, s):
                return []

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_missing_model_cannot_be_instantiated(self):
        class Incomplete(BaseProvider):
            @property
            def provider(self):
                return "p"

            def messages(self, p, s):
                return []

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_missing_messages_cannot_be_instantiated(self):
        class Incomplete(BaseProvider):
            @property
            def provider(self):
                return "p"

            @property
            def model(self):
                return "m"

        with pytest.raises(TypeError):
            Incomplete()


# ---------------------------------------------------------------------------
# generate() — OpenAI response path
# ---------------------------------------------------------------------------


class TestGenerateOpenAI:
    def test_returns_content_from_choices(self, openai_provider):
        openai_provider.client.post.return_value = _mock_response(
            {"choices": [{"message": {"content": "Hello!"}}]}
        )
        assert openai_provider.generate("Hi", "Be helpful.") == "Hello!"

    def test_sends_provider_header(self, openai_provider):
        openai_provider.client.post.return_value = _mock_response(
            {"choices": [{"message": {"content": "ok"}}]}
        )
        openai_provider.generate("p", "s")
        headers = openai_provider.client.post.call_args[1]["headers"]
        assert headers["Provider"] == "openai"

    def test_sends_api_key_header(self, openai_provider):
        openai_provider.client.post.return_value = _mock_response(
            {"choices": [{"message": {"content": "ok"}}]}
        )
        openai_provider.generate("p", "s")
        headers = openai_provider.client.post.call_args[1]["headers"]
        assert headers["X-Api-Key"] == settings.AMALIAI_API_KEY

    def test_sends_content_type_header(self, openai_provider):
        openai_provider.client.post.return_value = _mock_response(
            {"choices": [{"message": {"content": "ok"}}]}
        )
        openai_provider.generate("p", "s")
        headers = openai_provider.client.post.call_args[1]["headers"]
        assert headers["Content-Type"] == "application/json"

    def test_sends_correct_model_in_payload(self, openai_provider):
        openai_provider.client.post.return_value = _mock_response(
            {"choices": [{"message": {"content": "ok"}}]}
        )
        openai_provider.generate("prompt", "system")
        payload = openai_provider.client.post.call_args[1]["json"]
        assert payload["model"] == "gpt-3.5-turbo"

    def test_sends_max_tokens_in_payload(self, openai_provider):
        openai_provider.client.post.return_value = _mock_response(
            {"choices": [{"message": {"content": "ok"}}]}
        )
        openai_provider.generate("prompt", "system")
        payload = openai_provider.client.post.call_args[1]["json"]
        assert payload["max_tokens"] == settings.MAX_TOKENS

    def test_sends_temperature_in_payload(self, openai_provider):
        openai_provider.client.post.return_value = _mock_response(
            {"choices": [{"message": {"content": "ok"}}]}
        )
        openai_provider.generate("prompt", "system")
        payload = openai_provider.client.post.call_args[1]["json"]
        assert payload["temperature"] == settings.TEMPERATURE

    def test_sends_stream_false_in_payload(self, openai_provider):
        openai_provider.client.post.return_value = _mock_response(
            {"choices": [{"message": {"content": "ok"}}]}
        )
        openai_provider.generate("prompt", "system")
        payload = openai_provider.client.post.call_args[1]["json"]
        assert payload["stream"] is False

    def test_posts_to_base_url(self, openai_provider):
        openai_provider.client.post.return_value = _mock_response(
            {"choices": [{"message": {"content": "ok"}}]}
        )
        openai_provider.generate("p", "s")
        call_url = openai_provider.client.post.call_args[1]["url"]
        assert call_url == settings.AMALIAI_BASE_URL

    def test_calls_raise_for_status(self, openai_provider):
        mock_resp = _mock_response({"choices": [{"message": {"content": "ok"}}]})
        openai_provider.client.post.return_value = mock_resp
        openai_provider.generate("p", "s")
        mock_resp.raise_for_status.assert_called_once()

    def test_raises_on_empty_choices(self, openai_provider):
        openai_provider.client.post.return_value = _mock_response({"choices": []})
        with pytest.raises((ValueError, IndexError, KeyError)):
            openai_provider.generate("p", "s")

    def test_includes_messages_in_payload(self, openai_provider):
        openai_provider.client.post.return_value = _mock_response(
            {"choices": [{"message": {"content": "ok"}}]}
        )
        openai_provider.generate("user text", "system text")
        payload = openai_provider.client.post.call_args[1]["json"]
        assert "messages" in payload
        assert isinstance(payload["messages"], list)


# ---------------------------------------------------------------------------
# generate() — Anthropic response path
# ---------------------------------------------------------------------------


class TestGenerateAnthropic:
    def test_returns_text_from_content(self, anthropic_provider):
        anthropic_provider.client.post.return_value = _mock_response(
            {"content": [{"text": "Anthropic reply"}]}
        )
        assert anthropic_provider.generate("Hi", "Be helpful.") == "Anthropic reply"

    def test_sends_anthropic_provider_header(self, anthropic_provider):
        anthropic_provider.client.post.return_value = _mock_response(
            {"content": [{"text": "ok"}]}
        )
        anthropic_provider.generate("p", "s")
        headers = anthropic_provider.client.post.call_args[1]["headers"]
        assert headers["Provider"] == "anthropic"

    def test_raises_on_empty_content(self, anthropic_provider):
        anthropic_provider.client.post.return_value = _mock_response({"content": []})
        with pytest.raises((ValueError, IndexError, KeyError)):
            anthropic_provider.generate("p", "s")


# ---------------------------------------------------------------------------
# generate() — error handling
# ---------------------------------------------------------------------------


class TestGenerateErrorHandling:
    def test_reraises_http_status_error(self):
        p = _OpenAILikeProvider()
        p.client = MagicMock()
        mock_http_resp = MagicMock()
        mock_http_resp.status_code = 401
        mock_http_resp.text = "Unauthorized"
        p.client.post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized", request=MagicMock(), response=mock_http_resp
        )
        with pytest.raises(httpx.HTTPStatusError):
            p.generate("p", "s")

    def test_reraises_connect_error(self):
        p = _OpenAILikeProvider()
        p.client = MagicMock()
        p.client.post.side_effect = httpx.ConnectError(
            "Connection refused", request=MagicMock()
        )
        with pytest.raises(httpx.RequestError):
            p.generate("p", "s")

    def test_reraises_timeout_error(self):
        p = _OpenAILikeProvider()
        p.client = MagicMock()
        p.client.post.side_effect = httpx.TimeoutException(
            "Request timed out", request=MagicMock()
        )
        with pytest.raises(httpx.RequestError):
            p.generate("p", "s")

    def test_raises_value_error_for_unknown_provider(self):
        class _UnknownProvider(BaseProvider):
            @property
            def provider(self):
                return "unknown_llm"

            @property
            def model(self):
                return "some-model"

            def messages(self, p, s):
                return []

        prov = _UnknownProvider()
        prov.client = MagicMock()
        prov.client.post.return_value = _mock_response({})
        with pytest.raises(ValueError, match="Unknown provider"):
            prov.generate("p", "s")
