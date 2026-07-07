"""
Tests for services.rag_engine.RAGEngine.

All external dependencies (Redis, ChromaDB, AI providers, tiktoken) are
mocked — no live services are required.
"""

from unittest.mock import ANY, MagicMock, patch

import pytest

from services.rag_engine import RAGEngine, _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_book(
    title="Dune",
    author="Frank Herbert",
    similarity=0.9,
    doc="A sweeping sci-fi epic about politics, religion, and ecology.",
):
    return {
        "id": "book_001",
        "document": doc,
        "metadata": {"title": title, "author": author, "year": 1965, "genre": "Sci-Fi"},
        "similarity": similarity,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rag():
    """RAGEngine with all external dependencies replaced by MagicMocks."""
    with (
        patch("services.rag_engine.VectorStore"),
        patch("services.rag_engine.Cache"),
        patch("services.rag_engine.get_usage_tracker"),
        patch("services.rag_engine.get_rate_limiter"),
        patch("services.rag_engine.ResilientAIService"),
        patch(
            "services.rag_engine.EmbeddingsService"
        ),  # now at services.embedding_service
        patch("services.rag_engine.tiktoken"),
    ):
        engine = RAGEngine()

    engine.vector_store = MagicMock()
    engine.cache = MagicMock()
    engine.usage_tracker = MagicMock()
    engine.rate_limiter = MagicMock()
    engine.provider = MagicMock()
    engine.embedding_service = MagicMock()
    engine._tokenizer = MagicMock()
    engine._tokenizer.encode.return_value = list(range(10))  # 10 tokens for any text
    engine.relevance_threshold = 0.7
    engine.rag_top_k = 5
    return engine


# ---------------------------------------------------------------------------
# TestRAGEngineAsk — step-by-step flow
# ---------------------------------------------------------------------------


class TestRAGEngineAskCacheHit:
    def test_returns_cached_value_with_cached_true(self, rag):
        rag.cache.get.return_value = {"answer": "cached answer", "sources": []}

        result = rag.ask("What is Dune about?")

        assert result == {"answer": "cached answer", "sources": [], "cached": True}

    def test_skips_rate_limiter_on_cache_hit(self, rag):
        rag.cache.get.return_value = {"answer": "cached", "sources": []}

        rag.ask("What is Dune about?")

        rag.rate_limiter.acquire.assert_not_called()

    def test_skips_embedding_on_cache_hit(self, rag):
        rag.cache.get.return_value = {"answer": "cached", "sources": []}

        rag.ask("What is Dune about?")

        rag.embedding_service.embed.assert_not_called()

    def test_skips_ai_generation_on_cache_hit(self, rag):
        rag.cache.get.return_value = {"answer": "cached", "sources": []}

        rag.ask("What is Dune about?")

        rag.provider.generate.assert_not_called()


class TestRAGEngineAskRateLimit:
    def test_rate_limit_exception_propagates(self, rag):
        rag.cache.get.return_value = None
        rag.rate_limiter.acquire.side_effect = Exception(
            "Rate limit exceeded. Please try again later."
        )

        with pytest.raises(Exception, match="Rate limit exceeded"):
            rag.ask("Tell me about Dune.")

    def test_embedding_not_called_when_rate_limited(self, rag):
        rag.cache.get.return_value = None
        rag.rate_limiter.acquire.side_effect = Exception("Rate limit exceeded.")

        with pytest.raises(Exception):
            rag.ask("Tell me about Dune.")

        rag.embedding_service.embed.assert_not_called()


class TestRAGEngineAskFullFlow:
    def test_returns_answer_string(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1, 0.2, 0.3]
        rag.vector_store.search_books.return_value = [_make_book(similarity=0.85)]
        rag.provider.generate.return_value = "Dune is a sci-fi masterpiece."

        result = rag.ask("What is Dune about?")

        assert result["answer"] == "Dune is a sci-fi masterpiece."

    def test_returns_sources_list(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [_make_book(similarity=0.85)]
        rag.provider.generate.return_value = "Answer."

        result = rag.ask("What is Dune about?")

        assert len(result["sources"]) == 1
        assert result["sources"][0]["title"] == "Dune"
        assert result["sources"][0]["author"] == "Frank Herbert"

    def test_cached_flag_is_false_on_fresh_response(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [_make_book(similarity=0.85)]
        rag.provider.generate.return_value = "Answer."

        result = rag.ask("What is Dune about?")

        assert result["cached"] is False

    def test_search_called_with_embedding_and_top_k(self, rag):
        rag.cache.get.return_value = None
        embedding = [0.1, 0.2, 0.3]
        rag.embedding_service.embed.return_value = embedding
        rag.vector_store.search_books.return_value = []

        rag.ask("Any question.")

        rag.vector_store.search_books.assert_called_once_with(embedding, top_k=5)

    def test_generate_called_with_correct_system_prompt(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [_make_book(similarity=0.9)]
        rag.provider.generate.return_value = "Answer."

        rag.ask("A question.")

        rag.provider.generate.assert_called_once_with(prompt=ANY, system=_SYSTEM_PROMPT)

    def test_generate_prompt_contains_question(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [_make_book(similarity=0.9)]
        rag.provider.generate.return_value = "Answer."

        rag.ask("What makes Dune special?")

        prompt_arg = rag.provider.generate.call_args.kwargs["prompt"]
        assert "What makes Dune special?" in prompt_arg


class TestRAGEngineAskRelevanceFilter:
    def test_no_results_above_threshold_returns_refusal(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [
            _make_book(similarity=0.5),
            _make_book(similarity=0.3),
        ]

        result = rag.ask("Where can I buy groceries?")

        assert "couldn't find" in result["answer"].lower()
        assert result["sources"] == []
        assert result["cached"] is False

    def test_no_results_does_not_call_generate(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [_make_book(similarity=0.2)]

        rag.ask("Unrelated query.")

        rag.provider.generate.assert_not_called()

    def test_no_results_does_not_track_usage(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [_make_book(similarity=0.1)]

        rag.ask("Unrelated query.")

        rag.usage_tracker.track.assert_not_called()

    def test_only_books_at_or_above_threshold_appear_in_sources(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [
            _make_book(title="Above", similarity=0.85),
            _make_book(title="Below", similarity=0.4),
            _make_book(title="Exactly", similarity=0.7),  # == threshold, should pass
        ]
        rag.provider.generate.return_value = "Answer."

        result = rag.ask("space opera")

        titles = [s["title"] for s in result["sources"]]
        assert "Above" in titles
        assert "Exactly" in titles
        assert "Below" not in titles


class TestRAGEngineAskCaching:
    def test_cache_get_uses_correct_key(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [_make_book()]
        rag.provider.generate.return_value = "Answer."

        rag.ask("What is Dune?")

        expected_key = rag.cache.generate_key("rag", "What is Dune?")
        rag.cache.get.assert_called_once_with(expected_key)

    def test_cache_set_called_after_generation(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [_make_book()]
        rag.provider.generate.return_value = "The answer."

        rag.ask("Tell me about Dune.")

        rag.cache.set.assert_called_once()

    def test_cached_flag_not_stored_in_cache(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [_make_book()]
        rag.provider.generate.return_value = "Answer."

        rag.ask("Tell me about Dune.")

        _, stored_value = rag.cache.set.call_args[0]
        assert "cached" not in stored_value

    def test_cache_set_stores_correct_answer(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [_make_book()]
        rag.provider.generate.return_value = "The real answer."

        rag.ask("Tell me about Dune.")

        _, stored_value = rag.cache.set.call_args[0]
        assert stored_value["answer"] == "The real answer."

    def test_no_cache_set_on_refusal(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [_make_book(similarity=0.1)]

        rag.ask("Off-topic question.")

        rag.cache.set.assert_not_called()


class TestRAGEngineAskUsageTracking:
    def test_track_called_once_after_generation(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [_make_book(similarity=0.9)]
        rag.provider.generate.return_value = "Answer."

        rag.ask("A question.")

        rag.usage_tracker.track.assert_called_once()

    def test_track_receives_integer_token_counts(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [_make_book(similarity=0.9)]
        rag.provider.generate.return_value = "Answer."

        rag.ask("A question.")

        _, prompt_tokens, completion_tokens = rag.usage_tracker.track.call_args[0]
        assert isinstance(prompt_tokens, int)
        assert isinstance(completion_tokens, int)

    def test_track_receives_model_string(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [_make_book(similarity=0.9)]
        rag.provider.generate.return_value = "Answer."

        rag.ask("A question.")

        model, _, _ = rag.usage_tracker.track.call_args[0]
        assert isinstance(model, str)
        assert len(model) > 0


class TestRAGEngineAskSources:
    def test_sources_contain_title_author_similarity(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [_make_book(similarity=0.9)]
        rag.provider.generate.return_value = "Answer."

        result = rag.ask("Question.")

        source = result["sources"][0]
        assert "title" in source
        assert "author" in source
        assert "similarity" in source

    def test_similarity_rounded_to_four_decimal_places(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [
            _make_book(similarity=0.876543210)
        ]
        rag.provider.generate.return_value = "Answer."

        result = rag.ask("Question.")

        assert result["sources"][0]["similarity"] == 0.8765

    def test_multiple_relevant_books_all_in_sources(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [
            _make_book(title="Dune", similarity=0.9),
            _make_book(title="Foundation", similarity=0.8),
        ]
        rag.provider.generate.return_value = "Answer."

        result = rag.ask("Science fiction classics.")

        titles = [s["title"] for s in result["sources"]]
        assert "Dune" in titles
        assert "Foundation" in titles


# ---------------------------------------------------------------------------
# TestBuildContext
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_single_book_contains_title(self, rag):
        context = rag._build_context([_make_book(title="Dune")])
        assert "Dune" in context

    def test_single_book_contains_author(self, rag):
        context = rag._build_context([_make_book(author="Frank Herbert")])
        assert "Frank Herbert" in context

    def test_single_book_contains_description(self, rag):
        context = rag._build_context([_make_book(doc="Epic desert planet story.")])
        assert "Epic desert planet story." in context

    def test_single_book_numbered_one(self, rag):
        context = rag._build_context([_make_book()])
        assert "[1]" in context

    def test_multiple_books_numbered_sequentially(self, rag):
        books = [_make_book(title="Book A"), _make_book(title="Book B")]
        context = rag._build_context(books)
        assert "[1]" in context
        assert "[2]" in context

    def test_multiple_books_all_titles_present(self, rag):
        books = [_make_book(title="Book A"), _make_book(title="Book B")]
        context = rag._build_context(books)
        assert "Book A" in context
        assert "Book B" in context

    def test_missing_metadata_falls_back_to_unknown(self, rag):
        book = {"id": "x", "document": "Some text.", "metadata": {}, "similarity": 0.8}
        context = rag._build_context([book])
        assert "Unknown" in context

    def test_missing_year_genre_falls_back_to_na(self, rag):
        book = {"id": "x", "document": "Some text.", "metadata": {}, "similarity": 0.8}
        context = rag._build_context([book])
        assert "N/A" in context


# ---------------------------------------------------------------------------
# TestBuildPrompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_contains_the_question(self, rag):
        prompt = rag._build_prompt("What is Dune?", "context block")
        assert "What is Dune?" in prompt

    def test_contains_the_context(self, rag):
        prompt = rag._build_prompt("question", "CONTEXT_BLOCK_HERE")
        assert "CONTEXT_BLOCK_HERE" in prompt

    def test_has_separator_between_context_and_question(self, rag):
        prompt = rag._build_prompt("q", "ctx")
        assert "---" in prompt

    def test_question_appears_after_context(self, rag):
        prompt = rag._build_prompt("MY_QUESTION", "MY_CONTEXT")
        assert prompt.index("MY_CONTEXT") < prompt.index("MY_QUESTION")


# ---------------------------------------------------------------------------
# TestCountTokens
# ---------------------------------------------------------------------------


class TestRAGEngineMetadataFilterRouting:
    """
    Structured questions (year ranges, "how many X books") must be answered
    by filtering the full catalogue's metadata, not by semantic similarity
    over book descriptions — similarity search can't reliably answer these.
    """

    def _catalogue(self):
        return [
            {
                "id": "book_001",
                "document": "Desert planet epic.",
                "metadata": {
                    "title": "Dune",
                    "author": "Frank Herbert",
                    "year": 1965,
                    "genre": "Science Fiction",
                },
            },
            {
                "id": "book_002",
                "document": "Mars survival story.",
                "metadata": {
                    "title": "The Martian",
                    "author": "Andy Weir",
                    "year": 2011,
                    "genre": "Science Fiction",
                },
            },
        ]

    def test_year_filter_skips_embedding_call(self, rag):
        rag.cache.get.return_value = None
        rag.vector_store.get_all_books.return_value = self._catalogue()
        rag.provider.generate.return_value = "Only Dune is before 2000."

        rag.ask("What books came out before the year 2000?")

        rag.embedding_service.embed.assert_not_called()

    def test_year_filter_skips_semantic_search(self, rag):
        rag.cache.get.return_value = None
        rag.vector_store.get_all_books.return_value = self._catalogue()
        rag.provider.generate.return_value = "Only Dune is before 2000."

        rag.ask("What books came out before the year 2000?")

        rag.vector_store.search_books.assert_not_called()

    def test_year_filter_only_matching_book_in_sources(self, rag):
        rag.cache.get.return_value = None
        rag.vector_store.get_all_books.return_value = self._catalogue()
        rag.provider.generate.return_value = "Only Dune is before 2000."

        result = rag.ask("What books came out before the year 2000?")

        titles = [s["title"] for s in result["sources"]]
        assert titles == ["Dune"]

    def test_year_filter_no_matches_returns_refusal_without_generate_call(self, rag):
        rag.cache.get.return_value = None
        rag.vector_store.get_all_books.return_value = self._catalogue()

        result = rag.ask("What books came out before the year 1900?")

        assert result["sources"] == []
        assert "1900" in result["answer"]
        rag.provider.generate.assert_not_called()

    def test_year_filter_no_matches_does_not_track_usage(self, rag):
        rag.cache.get.return_value = None
        rag.vector_store.get_all_books.return_value = self._catalogue()

        rag.ask("What books came out before the year 1900?")

        rag.usage_tracker.track.assert_not_called()

    def test_enumeration_query_matches_all_books(self, rag):
        rag.cache.get.return_value = None
        rag.vector_store.get_all_books.return_value = self._catalogue()
        rag.provider.generate.return_value = "We have 2 books."

        result = rag.ask("How many books do you have?")

        titles = [s["title"] for s in result["sources"]]
        assert set(titles) == {"Dune", "The Martian"}

    def test_normal_question_still_uses_semantic_search(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [_make_book(similarity=0.9)]
        rag.provider.generate.return_value = "Dune is great."

        rag.ask("What is Dune about?")

        rag.vector_store.get_all_books.assert_not_called()
        rag.vector_store.search_books.assert_called_once()

    def test_year_filter_response_is_cached(self, rag):
        rag.cache.get.return_value = None
        rag.vector_store.get_all_books.return_value = self._catalogue()
        rag.provider.generate.return_value = "Only Dune is before 2000."

        rag.ask("What books came out before the year 2000?")

        rag.cache.set.assert_called_once()

    def test_rate_limiter_released_on_failure_during_metadata_path(self, rag):
        rag.cache.get.return_value = None
        rag.vector_store.get_all_books.side_effect = Exception("ChromaDB down")

        with pytest.raises(Exception, match="ChromaDB down"):
            rag.ask("What books came out before the year 2000?")

        rag.rate_limiter.release.assert_called_once()


class TestRAGEngineAskReturnsBooks:
    def test_result_includes_raw_books_used_for_context(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        book = _make_book(similarity=0.9)
        rag.vector_store.search_books.return_value = [book]
        rag.provider.generate.return_value = "Answer."

        result = rag.ask("What is Dune about?")

        assert result["books"] == [book]

    def test_no_match_result_has_no_books_key(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [_make_book(similarity=0.1)]

        result = rag.ask("Off-topic question.")

        assert "books" not in result


class TestRAGEngineVagueFollowupFallback:
    def test_falls_back_to_previous_books_when_nothing_found(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = []
        rag.provider.generate.return_value = "Rich Dad Poor Dad is about..."
        previous = [_make_book(title="Rich Dad Poor Dad")]

        result = rag.ask("tell me more about this book", previous_books=previous)

        assert result["answer"] == "Rich Dad Poor Dad is about..."
        assert result["sources"][0]["title"] == "Rich Dad Poor Dad"

    def test_no_fallback_without_previous_books(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = []

        result = rag.ask("tell me more about this book", previous_books=None)

        assert result["sources"] == []
        rag.provider.generate.assert_not_called()

    def test_no_fallback_when_question_is_not_a_followup(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = []
        previous = [_make_book(title="Rich Dad Poor Dad")]

        result = rag.ask("What is quantum computing?", previous_books=previous)

        assert result["sources"] == []
        rag.provider.generate.assert_not_called()

    def test_real_match_preferred_over_fallback(self, rag):
        rag.cache.get.return_value = None
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = [
            _make_book(title="Dune", similarity=0.9)
        ]
        rag.provider.generate.return_value = "Dune is..."
        previous = [_make_book(title="Rich Dad Poor Dad")]

        result = rag.ask("tell me more about this book", previous_books=previous)

        titles = [s["title"] for s in result["sources"]]
        assert titles == ["Dune"]

    def test_vague_followup_bypasses_cache_get(self, rag):
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = []
        previous = [_make_book(title="Rich Dad Poor Dad")]

        rag.ask("tell me more about this book", previous_books=previous)

        rag.cache.get.assert_not_called()

    def test_vague_followup_bypasses_cache_set(self, rag):
        rag.embedding_service.embed.return_value = [0.1]
        rag.vector_store.search_books.return_value = []
        rag.provider.generate.return_value = "Answer."
        previous = [_make_book(title="Rich Dad Poor Dad")]

        rag.ask("tell me more about this book", previous_books=previous)

        rag.cache.set.assert_not_called()


class TestCountTokens:
    def test_returns_length_of_encoded_tokens(self, rag):
        rag._tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        assert rag._count_tokens("hello world") == 5

    def test_delegates_to_tokenizer_encode(self, rag):
        rag._tokenizer.encode.return_value = []
        rag._count_tokens("test input")
        rag._tokenizer.encode.assert_called_once_with("test input")

    def test_empty_string_returns_zero_for_empty_encoding(self, rag):
        rag._tokenizer.encode.return_value = []
        assert rag._count_tokens("") == 0
