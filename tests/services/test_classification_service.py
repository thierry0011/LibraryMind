"""
Tests for services.classification_service.ClassificationService.

ResilientAIService is fully mocked — no live AI calls required.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from services.classification_service import ClassificationService, _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_CLASSIFICATION = {
    "category": "technical",
    "priority": "high",
    "sentiment": "negative",
    "department": "IT Support",
    "summary": "User cannot log into the library portal.",
}


def _json_response(data: dict = None) -> str:
    return json.dumps(data or _VALID_CLASSIFICATION)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def svc():
    """ClassificationService with ResilientAIService replaced by a MagicMock."""
    with patch("services.classification_service.ResilientAIService"):
        service = ClassificationService()
    service.provider = MagicMock()
    return service


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestClassificationServiceInit:
    def test_provider_instantiated(self):
        with patch(
            "services.classification_service.ResilientAIService"
        ) as MockProvider:
            ClassificationService()
        MockProvider.assert_called_once()

    def test_provider_attribute_set(self, svc):
        assert svc.provider is not None


# ---------------------------------------------------------------------------
# TestClassifyReturnValue
# ---------------------------------------------------------------------------


class TestClassifyReturnValue:
    def test_returns_a_dict(self, svc):
        svc.provider.generate.return_value = _json_response()
        assert isinstance(svc.classify("I cannot log in."), dict)

    def test_category_present(self, svc):
        svc.provider.generate.return_value = _json_response()
        assert "category" in svc.classify("I cannot log in.")

    def test_priority_present(self, svc):
        svc.provider.generate.return_value = _json_response()
        assert "priority" in svc.classify("I cannot log in.")

    def test_sentiment_present(self, svc):
        svc.provider.generate.return_value = _json_response()
        assert "sentiment" in svc.classify("I cannot log in.")

    def test_department_present(self, svc):
        svc.provider.generate.return_value = _json_response()
        assert "department" in svc.classify("I cannot log in.")

    def test_summary_present(self, svc):
        svc.provider.generate.return_value = _json_response()
        assert "summary" in svc.classify("I cannot log in.")

    def test_values_match_provider_output(self, svc):
        svc.provider.generate.return_value = _json_response()
        result = svc.classify("I cannot log in.")
        assert result["category"] == "technical"
        assert result["priority"] == "high"
        assert result["sentiment"] == "negative"
        assert result["department"] == "IT Support"


# ---------------------------------------------------------------------------
# TestClassifyProviderCall
# ---------------------------------------------------------------------------


class TestClassifyProviderCall:
    def test_provider_generate_called_once(self, svc):
        svc.provider.generate.return_value = _json_response()
        svc.classify("Some ticket text.")
        assert svc.provider.generate.call_count == 1

    def test_system_prompt_passed_to_provider(self, svc):
        svc.provider.generate.return_value = _json_response()
        svc.classify("Some ticket text.")
        system = svc.provider.generate.call_args.kwargs["system"]
        assert system == _SYSTEM_PROMPT

    def test_ticket_text_passed_as_prompt(self, svc):
        svc.provider.generate.return_value = _json_response()
        svc.classify("UNIQUE_TICKET_CONTENT")
        prompt = svc.provider.generate.call_args.kwargs["prompt"]
        assert prompt == "UNIQUE_TICKET_CONTENT"

    def test_different_tickets_pass_different_prompts(self, svc):
        svc.provider.generate.return_value = _json_response()
        svc.classify("Ticket A")
        first_prompt = svc.provider.generate.call_args.kwargs["prompt"]

        svc.classify("Ticket B")
        second_prompt = svc.provider.generate.call_args.kwargs["prompt"]

        assert first_prompt != second_prompt


# ---------------------------------------------------------------------------
# TestClassifyCategories
# ---------------------------------------------------------------------------


class TestClassifyCategories:
    @pytest.mark.parametrize(
        "category",
        ["account", "borrowing", "technical", "complaint", "suggestion", "general"],
    )
    def test_valid_category_values_are_accepted(self, svc, category):
        data = {**_VALID_CLASSIFICATION, "category": category}
        svc.provider.generate.return_value = json.dumps(data)
        result = svc.classify("Some ticket.")
        assert result["category"] == category

    @pytest.mark.parametrize("priority", ["low", "medium", "high", "urgent"])
    def test_valid_priority_values_are_accepted(self, svc, priority):
        data = {**_VALID_CLASSIFICATION, "priority": priority}
        svc.provider.generate.return_value = json.dumps(data)
        result = svc.classify("Some ticket.")
        assert result["priority"] == priority

    @pytest.mark.parametrize("sentiment", ["positive", "neutral", "negative"])
    def test_valid_sentiment_values_are_accepted(self, svc, sentiment):
        data = {**_VALID_CLASSIFICATION, "sentiment": sentiment}
        svc.provider.generate.return_value = json.dumps(data)
        result = svc.classify("Some ticket.")
        assert result["sentiment"] == sentiment


# ---------------------------------------------------------------------------
# TestClassifyErrorHandling
# ---------------------------------------------------------------------------


class TestClassifyErrorHandling:
    def test_raises_value_error_on_invalid_json(self, svc):
        svc.provider.generate.return_value = "not valid json at all"
        with pytest.raises(ValueError):
            svc.classify("Some ticket.")

    def test_error_message_contains_raw_response(self, svc):
        raw = "RAW_BROKEN_RESPONSE"
        svc.provider.generate.return_value = raw
        with pytest.raises(ValueError, match=raw):
            svc.classify("Some ticket.")

    def test_accepts_markdown_fenced_json(self, svc):
        fenced = f"```json\n{_json_response()}\n```"
        svc.provider.generate.return_value = fenced
        result = svc.classify("I cannot log in.")
        assert result["category"] == "technical"

    def test_empty_string_response_raises_value_error(self, svc):
        svc.provider.generate.return_value = ""
        with pytest.raises(ValueError):
            svc.classify("Some ticket.")
