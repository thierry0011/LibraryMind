"""Tests for OpenAIProvider."""

import pytest
from unittest.mock import MagicMock

from providers.openai_provider import OpenAIProvider
from providers.base import BaseProvider
from config import settings


@pytest.fixture
def provider():
    p = OpenAIProvider()
    p.client = MagicMock()
    return p


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


class TestOpenAIProviderProperties:
    def test_provider_returns_openai(self, provider):
        assert provider.provider == "openai"

    def test_model_returns_model_from_settings(self, provider):
        assert provider.model == settings.MODEL

    def test_is_subclass_of_base_provider(self):
        assert issubclass(OpenAIProvider, BaseProvider)

    def test_can_be_instantiated(self):
        p = OpenAIProvider()
        assert p is not None
        p.close()


class TestOpenAIProviderMessages:
    def test_returns_a_list(self, provider):
        assert isinstance(provider.messages("hello", "system"), list)

    def test_returns_two_messages(self, provider):
        assert len(provider.messages("hello", "system")) == 2

    def test_first_message_has_system_role(self, provider):
        result = provider.messages("user msg", "system instruction")
        assert result[0]["role"] == "system"

    def test_first_message_has_system_content(self, provider):
        result = provider.messages("user msg", "system instruction")
        assert result[0]["content"] == "system instruction"

    def test_second_message_has_user_role(self, provider):
        result = provider.messages("user msg", "system instruction")
        assert result[1]["role"] == "user"

    def test_second_message_has_user_content(self, provider):
        result = provider.messages("user msg", "system instruction")
        assert result[1]["content"] == "user msg"

    def test_preserves_prompt_verbatim(self, provider):
        prompt = "Search for Python programming books"
        assert provider.messages(prompt, "s")[1]["content"] == prompt

    def test_preserves_system_verbatim(self, provider):
        system = "You are a helpful library assistant."
        assert provider.messages("p", system)[0]["content"] == system

    def test_empty_prompt_is_allowed(self, provider):
        assert provider.messages("", "system")[1]["content"] == ""

    def test_empty_system_is_allowed(self, provider):
        assert provider.messages("prompt", "")[0]["content"] == ""


class TestOpenAIProviderGenerate:
    def test_generate_returns_content(self, provider):
        provider.client.post.return_value = _mock_response("Found 3 books.")
        assert (
            provider.generate("Find Python books", "You are a librarian.")
            == "Found 3 books."
        )

    def test_generate_payload_uses_correct_model(self, provider):
        provider.client.post.return_value = _mock_response("ok")
        provider.generate("p", "s")
        payload = provider.client.post.call_args[1]["json"]
        assert payload["model"] == settings.MODEL

    def test_generate_header_identifies_openai(self, provider):
        provider.client.post.return_value = _mock_response("ok")
        provider.generate("p", "s")
        headers = provider.client.post.call_args[1]["headers"]
        assert headers["Provider"] == "openai"

    def test_generate_payload_includes_messages(self, provider):
        provider.client.post.return_value = _mock_response("ok")
        provider.generate("user prompt", "system prompt")
        payload = provider.client.post.call_args[1]["json"]
        roles = [m["role"] for m in payload["messages"]]
        assert "system" in roles
        assert "user" in roles
