"""Tests for AnthropicProvider."""

import pytest
from unittest.mock import MagicMock

from providers.anthropic_provider import AnthropicProvider
from providers.base import BaseProvider
from config import settings


@pytest.fixture
def provider():
    p = AnthropicProvider()
    p.client = MagicMock()
    return p


def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"content": [{"text": text}]}
    return resp


class TestAnthropicProviderProperties:
    def test_provider_returns_anthropic(self, provider):
        assert provider.provider == "anthropic"

    def test_model_returns_model_claude_from_settings(self, provider):
        assert provider.model == settings.MODEL_CLAUDE

    def test_is_subclass_of_base_provider(self):
        assert issubclass(AnthropicProvider, BaseProvider)

    def test_can_be_instantiated(self):
        p = AnthropicProvider()
        assert p is not None
        p.close()


class TestAnthropicProviderMessages:
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
        prompt = "What is the capital of France?"
        assert provider.messages(prompt, "s")[1]["content"] == prompt

    def test_preserves_system_verbatim(self, provider):
        system = "You are a librarian assistant with expertise in fiction."
        assert provider.messages("p", system)[0]["content"] == system

    def test_empty_prompt_is_allowed(self, provider):
        result = provider.messages("", "system")
        assert result[1]["content"] == ""

    def test_empty_system_is_allowed(self, provider):
        result = provider.messages("prompt", "")
        assert result[0]["content"] == ""


class TestAnthropicProviderGenerate:
    def test_generate_returns_text(self, provider):
        provider.client.post.return_value = _mock_response("Book found!")
        assert provider.generate("Find a book", "You are a librarian.") == "Book found!"

    def test_generate_payload_includes_messages(self, provider):
        provider.client.post.return_value = _mock_response("ok")
        provider.generate("user prompt", "system prompt")
        payload = provider.client.post.call_args[1]["json"]
        messages = payload["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

    def test_generate_payload_uses_correct_model(self, provider):
        provider.client.post.return_value = _mock_response("ok")
        provider.generate("p", "s")
        payload = provider.client.post.call_args[1]["json"]
        assert payload["model"] == settings.MODEL_CLAUDE

    def test_generate_header_identifies_anthropic(self, provider):
        provider.client.post.return_value = _mock_response("ok")
        provider.generate("p", "s")
        headers = provider.client.post.call_args[1]["headers"]
        assert headers["Provider"] == "anthropic"
