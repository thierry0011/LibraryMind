"""
Tests for services.query_signals — cheap detection of query characteristics
deterministic metadata parsing was never designed to handle at all.
"""

from services.query_signals import (
    has_comparative_phrasing,
    has_hard_signal,
    is_likely_non_english,
)


# ---------------------------------------------------------------------------
# is_likely_non_english
# ---------------------------------------------------------------------------


class TestIsLikelyNonEnglish:
    def test_ordinary_english_sentence_is_not_flagged(self):
        assert is_likely_non_english("tell me about Dune") is False

    def test_longer_english_sentence_is_not_flagged(self):
        assert (
            is_likely_non_english(
                "do you have any non fiction books published after 2007"
            )
            is False
        )

    def test_kinyarwanda_sentence_is_flagged(self):
        assert is_likely_non_english("Mbwira niba mufite igitabo kitwa Dune") is True

    def test_french_sentence_is_flagged(self):
        assert (
            is_likely_non_english(
                "Est-ce que vous avez des livres de science fiction publies"
            )
            is True
        )

    def test_bare_title_is_not_flagged(self):
        # Too short to have enough signal either way — left to semantic/
        # title search rather than guessed at.
        assert is_likely_non_english("Dune") is False

    def test_bare_foreign_word_is_not_flagged(self):
        assert is_likely_non_english("Emma") is False

    def test_short_foreign_phrase_is_not_flagged(self):
        # Below the minimum word count for the check to apply — an
        # accepted gap, since short proper-noun-heavy queries like this
        # are usually still findable via semantic search regardless.
        assert is_likely_non_english("Avez-vous Dune?") is False

    def test_comparative_english_sentence_is_not_flagged(self):
        assert (
            is_likely_non_english(
                "books similar to Atomic Habits but more philosophical"
            )
            is False
        )


# ---------------------------------------------------------------------------
# has_comparative_phrasing
# ---------------------------------------------------------------------------


class TestHasComparativePhrasing:
    def test_similar_to(self):
        assert has_comparative_phrasing("books similar to Dune") is True

    def test_books_like(self):
        assert has_comparative_phrasing("books like Atomic Habits") is True

    def test_in_the_style_of(self):
        assert has_comparative_phrasing("something in the style of Tolkien") is True

    def test_ordinary_question_not_flagged(self):
        assert has_comparative_phrasing("what is Dune about?") is False

    def test_genre_query_not_flagged(self):
        assert has_comparative_phrasing("show me science fiction books") is False


# ---------------------------------------------------------------------------
# has_hard_signal
# ---------------------------------------------------------------------------


class TestHasHardSignal:
    def test_true_for_non_english(self):
        assert has_hard_signal("Mbwira niba mufite igitabo kitwa Dune") is True

    def test_true_for_comparative(self):
        assert has_hard_signal("books similar to Dune") is True

    def test_false_for_ordinary_english_question(self):
        assert has_hard_signal("what is Dune about?") is False

    def test_false_for_structured_english_question(self):
        assert (
            has_hard_signal(
                "do you have any non fiction books published after 2007"
            )
            is False
        )
