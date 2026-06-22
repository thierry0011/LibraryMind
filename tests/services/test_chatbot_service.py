"""
Tests for services.chatbot_service.ChatbotService.

RAGEngine and ResilientAIService are fully mocked — no live services required.
chromadb is stubbed before the import chain pulls in the real C-extension.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("chromadb", MagicMock())

from services.chatbot_service import ChatbotService, _SYSTEM_PROMPT  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rag_result(answer="Some book context.", sources=None):
    return {"answer": answer, "sources": sources or [], "cached": False}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def svc():
    """ChatbotService with all external dependencies replaced by MagicMocks."""
    with (
        patch("services.chatbot_service.RAGEngine"),
        patch("services.chatbot_service.ResilientAIService"),
        patch("services.chatbot_service.RateLimiter"),
    ):
        service = ChatbotService()

    service.rag_engine = MagicMock()
    service.provider = MagicMock()
    service.rate_limiter = MagicMock()
    service.max_history = 10
    return service


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestChatbotServiceInit:
    def test_conversation_starts_empty(self, svc):
        assert svc.conversation == {}

    def test_max_history_loaded_from_settings(self):
        with (
            patch("services.chatbot_service.RAGEngine"),
            patch("services.chatbot_service.ResilientAIService"),
            patch("services.chatbot_service.RateLimiter"),
            patch("services.chatbot_service.settings") as mock_settings,
        ):
            mock_settings.MAX_CONVERSATION_HISTORY = 7
            service = ChatbotService()
        assert service.max_history == 7

    def test_rag_engine_instantiated(self):
        with (
            patch("services.chatbot_service.RAGEngine") as MockRAG,
            patch("services.chatbot_service.ResilientAIService"),
            patch("services.chatbot_service.RateLimiter"),
            patch("services.chatbot_service.settings") as ms,
        ):
            ms.MAX_CONVERSATION_HISTORY = 10
            ChatbotService()
        MockRAG.assert_called_once()

    def test_provider_instantiated(self):
        with (
            patch("services.chatbot_service.RAGEngine"),
            patch("services.chatbot_service.ResilientAIService") as MockProvider,
            patch("services.chatbot_service.RateLimiter"),
            patch("services.chatbot_service.settings") as ms,
        ):
            ms.MAX_CONVERSATION_HISTORY = 10
            ChatbotService()
        MockProvider.assert_called_once()


# ---------------------------------------------------------------------------
# TestGetConversationHistory
# ---------------------------------------------------------------------------


class TestGetConversationHistory:
    def test_returns_empty_list_for_unknown_id(self, svc):
        assert svc.get_conversation_history("unknown-id") == []

    def test_returns_stored_messages(self, svc):
        msgs = [{"role": "user", "content": "hello"}]
        svc.conversation["abc"] = msgs
        assert svc.get_conversation_history("abc") == msgs

    def test_different_ids_are_independent(self, svc):
        svc.conversation["id1"] = [{"role": "user", "content": "hi"}]
        assert svc.get_conversation_history("id2") == []

    def test_returns_all_stored_messages(self, svc):
        svc.conversation["xyz"] = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert len(svc.get_conversation_history("xyz")) == 2


# ---------------------------------------------------------------------------
# TestChatReturnValue
# ---------------------------------------------------------------------------


class TestChatReturnValue:
    def test_returns_a_dict(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        assert isinstance(svc.chat("id1", "question"), dict)

    def test_reply_key_present(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        assert "reply" in svc.chat("id1", "question")

    def test_sources_key_present(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        assert "sources" in svc.chat("id1", "question")

    def test_reply_matches_provider_output(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "The generated reply."
        result = svc.chat("id1", "question")
        assert result["reply"] == "The generated reply."

    def test_sources_come_from_rag_result(self, svc):
        sources = [{"title": "Dune", "author": "Frank Herbert", "similarity": 0.9}]
        svc.rag_engine.ask.return_value = _rag_result(sources=sources)
        svc.provider.generate.return_value = "reply"
        assert svc.chat("id1", "question")["sources"] == sources

    def test_sources_empty_when_rag_returns_none(self, svc):
        svc.rag_engine.ask.return_value = _rag_result(sources=[])
        svc.provider.generate.return_value = "reply"
        assert svc.chat("id1", "question")["sources"] == []


# ---------------------------------------------------------------------------
# TestChatRAGIntegration
# ---------------------------------------------------------------------------


class TestChatRAGIntegration:
    def test_rag_engine_called_with_user_message(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        svc.chat("id1", "Tell me about Dune.")
        svc.rag_engine.ask.assert_called_once_with("Tell me about Dune.")

    def test_rag_engine_called_exactly_once_per_chat(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        svc.chat("id1", "question")
        assert svc.rag_engine.ask.call_count == 1

    def test_provider_generate_called_exactly_once(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        svc.chat("id1", "question")
        assert svc.provider.generate.call_count == 1


# ---------------------------------------------------------------------------
# TestChatPromptConstruction
# ---------------------------------------------------------------------------


class TestChatPromptConstruction:
    def test_prompt_contains_rag_answer(self, svc):
        svc.rag_engine.ask.return_value = _rag_result(answer="RAG_ANSWER_MARKER")
        svc.provider.generate.return_value = "reply"
        svc.chat("id1", "question")
        prompt = svc.provider.generate.call_args.kwargs["prompt"]
        assert "RAG_ANSWER_MARKER" in prompt

    def test_prompt_contains_user_message(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        svc.chat("id1", "UNIQUE_USER_QUESTION")
        prompt = svc.provider.generate.call_args.kwargs["prompt"]
        assert "UNIQUE_USER_QUESTION" in prompt

    def test_correct_system_prompt_passed_to_provider(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        svc.chat("id1", "question")
        system = svc.provider.generate.call_args.kwargs["system"]
        assert system == _SYSTEM_PROMPT

    def test_first_message_has_no_prior_assistant_history(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        svc.chat("id1", "First ever message.")
        prompt = svc.provider.generate.call_args.kwargs["prompt"]
        assert "Assistant:" not in prompt

    def test_second_message_includes_first_turn_in_prompt(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "first reply"
        svc.chat("id1", "FIRST_QUESTION")

        svc.provider.generate.return_value = "second reply"
        svc.chat("id1", "Second question.")

        prompt = svc.provider.generate.call_args.kwargs["prompt"]
        assert "FIRST_QUESTION" in prompt

    def test_prior_assistant_reply_appears_in_next_prompt(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "ASSISTANT_REPLY_TEXT"
        svc.chat("id1", "First question.")

        svc.provider.generate.return_value = "second reply"
        svc.chat("id1", "Second question.")

        prompt = svc.provider.generate.call_args.kwargs["prompt"]
        assert "ASSISTANT_REPLY_TEXT" in prompt


# ---------------------------------------------------------------------------
# TestChatHistoryManagement
# ---------------------------------------------------------------------------


class TestChatHistoryManagement:
    def test_new_conversation_entry_created(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        svc.chat("brand-new-id", "hello")
        assert "brand-new-id" in svc.conversation

    def test_user_message_appended_to_history(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        svc.chat("id1", "What is Dune?")
        history = svc.get_conversation_history("id1")
        assert any(
            m["role"] == "user" and m["content"] == "What is Dune?" for m in history
        )

    def test_assistant_reply_appended_to_history(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "The assistant reply."
        svc.chat("id1", "question")
        history = svc.get_conversation_history("id1")
        assert any(
            m["role"] == "assistant" and m["content"] == "The assistant reply."
            for m in history
        )

    def test_user_message_before_assistant_in_history(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        svc.chat("id1", "question")
        history = svc.get_conversation_history("id1")
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_history_has_two_entries_after_one_turn(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        svc.chat("id1", "question")
        assert len(svc.get_conversation_history("id1")) == 2

    def test_history_grows_with_each_turn(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        svc.chat("id1", "First.")
        svc.chat("id1", "Second.")
        assert len(svc.get_conversation_history("id1")) == 4

    def test_history_trimmed_to_max_history(self, svc):
        svc.max_history = 4
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        for i in range(5):
            svc.chat("id1", f"Message {i}")
        assert len(svc.get_conversation_history("id1")) <= 4

    def test_oldest_messages_dropped_on_trim(self, svc):
        svc.max_history = 2
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        svc.chat("id1", "OLDEST_MESSAGE")
        svc.chat("id1", "Newer message.")
        contents = [m["content"] for m in svc.get_conversation_history("id1")]
        assert "OLDEST_MESSAGE" not in contents

    def test_separate_conversations_do_not_share_history(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        svc.chat("user-A", "Message from A.")
        svc.chat("user-B", "Message from B.")
        history_a = svc.get_conversation_history("user-A")
        history_b = svc.get_conversation_history("user-B")
        assert not any(m["content"] == "Message from B." for m in history_a)
        assert not any(m["content"] == "Message from A." for m in history_b)

    def test_subsequent_chats_on_same_id_accumulate_history(self, svc):
        svc.rag_engine.ask.return_value = _rag_result()
        svc.provider.generate.return_value = "reply"
        svc.chat("id1", "Turn one.")
        svc.chat("id1", "Turn two.")
        svc.chat("id1", "Turn three.")
        assert len(svc.get_conversation_history("id1")) == 6
