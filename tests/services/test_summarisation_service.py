"""
Tests for services.summarisation_service.SummarizationService.

ResilientAIService is fully mocked — no live AI calls required.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from services.summarisation_service import SummarizationService, _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_SUMMARY = {
    "overall_sentiment": "positive",
    "average_rating": 4.2,
    "key_themes": ["adventure", "friendship"],
    "praise": ["great pacing", "vivid world-building"],
    "criticism": ["slow start"],
    "recommendation": "Great for fans of epic fantasy.",
}


def _json_response(data: dict = None) -> str:
    return json.dumps(data or _VALID_SUMMARY)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def svc():
    """SummarizationService with ResilientAIService replaced by a MagicMock."""
    with patch("services.summarisation_service.ResilientAIService"):
        service = SummarizationService()
    service.provider = MagicMock()
    return service


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestSummarizationServiceInit:
    def test_provider_instantiated(self):
        with patch("services.summarisation_service.ResilientAIService") as MockProvider:
            SummarizationService()
        MockProvider.assert_called_once()

    def test_provider_attribute_set(self, svc):
        assert svc.provider is not None


# ---------------------------------------------------------------------------
# TestSummarizeReturnValue
# ---------------------------------------------------------------------------


class TestSummarizeReturnValue:
    def test_returns_a_dict(self, svc):
        svc.provider.generate.return_value = _json_response()
        result = svc.summarize(["Great book!", "Loved it."])
        assert isinstance(result, dict)

    def test_overall_sentiment_present(self, svc):
        svc.provider.generate.return_value = _json_response()
        result = svc.summarize(["Good read."])
        assert "overall_sentiment" in result

    def test_average_rating_present(self, svc):
        svc.provider.generate.return_value = _json_response()
        result = svc.summarize(["Good read."])
        assert "average_rating" in result

    def test_key_themes_present(self, svc):
        svc.provider.generate.return_value = _json_response()
        result = svc.summarize(["Good read."])
        assert "key_themes" in result

    def test_praise_present(self, svc):
        svc.provider.generate.return_value = _json_response()
        result = svc.summarize(["Good read."])
        assert "praise" in result

    def test_criticism_present(self, svc):
        svc.provider.generate.return_value = _json_response()
        result = svc.summarize(["Good read."])
        assert "criticism" in result

    def test_recommendation_present(self, svc):
        svc.provider.generate.return_value = _json_response()
        result = svc.summarize(["Good read."])
        assert "recommendation" in result

    def test_values_match_provider_output(self, svc):
        svc.provider.generate.return_value = _json_response()
        result = svc.summarize(["Good read."])
        assert result["overall_sentiment"] == "positive"
        assert result["average_rating"] == 4.2


# ---------------------------------------------------------------------------
# TestSummarizeProviderCall
# ---------------------------------------------------------------------------


class TestSummarizeProviderCall:
    def test_provider_generate_called_once(self, svc):
        svc.provider.generate.return_value = _json_response()
        svc.summarize(["Review one.", "Review two."])
        assert svc.provider.generate.call_count == 1

    def test_system_prompt_passed_to_provider(self, svc):
        svc.provider.generate.return_value = _json_response()
        svc.summarize(["Review one."])
        system = svc.provider.generate.call_args.kwargs["system"]
        assert system == _SYSTEM_PROMPT

    def test_prompt_contains_review_text(self, svc):
        svc.provider.generate.return_value = _json_response()
        svc.summarize(["UNIQUE_REVIEW_TEXT"])
        prompt = svc.provider.generate.call_args.kwargs["prompt"]
        assert "UNIQUE_REVIEW_TEXT" in prompt

    def test_prompt_numbers_each_review(self, svc):
        svc.provider.generate.return_value = _json_response()
        svc.summarize(["First review.", "Second review."])
        prompt = svc.provider.generate.call_args.kwargs["prompt"]
        assert "Review 1:" in prompt
        assert "Review 2:" in prompt

    def test_all_reviews_included_in_prompt(self, svc):
        reviews = ["Alpha review.", "Beta review.", "Gamma review."]
        svc.provider.generate.return_value = _json_response()
        svc.summarize(reviews)
        prompt = svc.provider.generate.call_args.kwargs["prompt"]
        for review in reviews:
            assert review in prompt

    def test_single_review_still_numbered(self, svc):
        svc.provider.generate.return_value = _json_response()
        svc.summarize(["Only review."])
        prompt = svc.provider.generate.call_args.kwargs["prompt"]
        assert "Review 1:" in prompt

    def test_empty_reviews_sends_empty_prompt(self, svc):
        svc.provider.generate.return_value = _json_response()
        svc.summarize([])
        prompt = svc.provider.generate.call_args.kwargs["prompt"]
        assert prompt == ""


# ---------------------------------------------------------------------------
# TestSummarizeErrorHandling
# ---------------------------------------------------------------------------


class TestSummarizeErrorHandling:
    def test_raises_value_error_on_invalid_json(self, svc):
        svc.provider.generate.return_value = "not valid json at all"
        with pytest.raises(ValueError):
            svc.summarize(["Some review."])

    def test_error_message_contains_raw_response(self, svc):
        raw = "RAW_BROKEN_RESPONSE"
        svc.provider.generate.return_value = raw
        with pytest.raises(ValueError, match=raw):
            svc.summarize(["Some review."])

    def test_accepts_markdown_fenced_json(self, svc):
        fenced = f"```json\n{_json_response()}\n```"
        svc.provider.generate.return_value = fenced
        result = svc.summarize(["Good read."])
        assert result["overall_sentiment"] == "positive"

    def test_empty_string_response_raises_value_error(self, svc):
        svc.provider.generate.return_value = ""
        with pytest.raises(ValueError):
            svc.summarize(["Some review."])
