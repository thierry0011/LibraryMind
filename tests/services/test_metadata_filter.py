"""
Tests for services.metadata_filter — pure parsing/filtering functions used to
route structured catalogue queries (year ranges, genre, author) away from
semantic vector search.
"""

from services.metadata_filter import (
    book_matches,
    build_filter_spec,
    describe_no_match,
    extract_known_term,
    has_enumeration_cue,
    has_temporal_cue,
    looks_like_metadata_query,
    looks_like_vague_followup,
    parse_year_filter,
)


# ---------------------------------------------------------------------------
# parse_year_filter
# ---------------------------------------------------------------------------


class TestParseYearFilter:
    def test_before_year(self):
        assert parse_year_filter("books published before 2000") == {
            "op": "lt",
            "year": 2000,
        }

    def test_prior_to_year(self):
        assert parse_year_filter("anything prior to 1990") == {"op": "lt", "year": 1990}

    def test_after_year(self):
        assert parse_year_filter("books after 2010") == {"op": "gt", "year": 2010}

    def test_since_year(self):
        assert parse_year_filter("what came out since 2015") == {
            "op": "gt",
            "year": 2015,
        }

    def test_up_to_year(self):
        assert parse_year_filter("books up to 1980") == {"op": "lte", "year": 1980}

    def test_from_year(self):
        assert parse_year_filter("books from 2005") == {"op": "gte", "year": 2005}

    def test_exact_year(self):
        assert parse_year_filter("what was published in 1965") == {
            "op": "eq",
            "year": 1965,
        }

    def test_between_years(self):
        assert parse_year_filter("books between 1980 and 1990") == {
            "op": "between",
            "start": 1980,
            "end": 1990,
        }

    def test_between_years_reversed_order_sorted(self):
        assert parse_year_filter("books between 1990 and 1980") == {
            "op": "between",
            "start": 1980,
            "end": 1990,
        }

    def test_decade(self):
        assert parse_year_filter("any books from the 1990s") == {
            "op": "between",
            "start": 1990,
            "end": 1999,
        }

    def test_no_year_returns_none(self):
        assert parse_year_filter("what is Dune about?") is None

    def test_unrelated_four_digit_number_without_keyword_returns_none(self):
        assert parse_year_filter("tell me about book 1234") is None


# ---------------------------------------------------------------------------
# has_enumeration_cue / looks_like_metadata_query
# ---------------------------------------------------------------------------


class TestEnumerationCue:
    def test_how_many(self):
        assert has_enumeration_cue("how many fantasy books do you have") is True

    def test_list_all(self):
        assert has_enumeration_cue("list all your thrillers") is True

    def test_no_cue(self):
        assert has_enumeration_cue("what is Dune about?") is False


class TestTemporalCue:
    def test_published_cue(self):
        assert has_temporal_cue("books published after two thousand seven") is True

    def test_came_out_cue(self):
        assert has_temporal_cue("what came out recently") is True

    def test_written_in_cue(self):
        assert has_temporal_cue("anything written in the nineties") is True

    def test_no_cue(self):
        assert has_temporal_cue("what is Dune about?") is False

    def test_fires_even_when_year_regex_would_also_match(self):
        # has_temporal_cue is deliberately independent of whether
        # parse_year_filter can resolve the year — callers combine the two.
        assert has_temporal_cue("published after 2007") is True


class TestLooksLikeMetadataQuery:
    def test_year_query_is_metadata_query(self):
        assert looks_like_metadata_query("books before 2000") is True

    def test_enumeration_query_is_metadata_query(self):
        assert looks_like_metadata_query("how many books do you have") is True

    def test_normal_question_is_not_metadata_query(self):
        assert looks_like_metadata_query("what is Dune about?") is False


# ---------------------------------------------------------------------------
# looks_like_vague_followup
# ---------------------------------------------------------------------------


class TestLooksLikeVagueFollowup:
    def test_tell_me_more(self):
        assert looks_like_vague_followup("tell me more") is True

    def test_tell_me_more_about_this_book(self):
        assert looks_like_vague_followup("tell me more about this book") is True

    def test_this_one(self):
        assert looks_like_vague_followup("what genre is this one") is True

    def test_that_book(self):
        assert looks_like_vague_followup("who wrote that book") is True

    def test_about_it(self):
        assert looks_like_vague_followup("can you tell me about it") is True

    def test_new_topic_naming_a_title_is_not_vague(self):
        assert looks_like_vague_followup("what is Dune about?") is False

    def test_year_query_is_not_vague(self):
        assert looks_like_vague_followup("books published before 2000") is False


# ---------------------------------------------------------------------------
# extract_known_term / build_filter_spec
# ---------------------------------------------------------------------------


class TestExtractKnownTerm:
    def test_matches_known_genre(self):
        result = extract_known_term("any fantasy books?", ["Fantasy", "Thriller"])
        assert result == "Fantasy"

    def test_prefers_longer_match(self):
        result = extract_known_term(
            "science fiction books please", ["Fiction", "Science Fiction"]
        )
        assert result == "Science Fiction"

    def test_no_match_returns_none(self):
        assert extract_known_term("romance novels", ["Fantasy", "Thriller"]) is None

    def test_ignores_falsy_candidates(self):
        assert extract_known_term("fantasy books", ["", None, "Fantasy"]) == "Fantasy"

    def test_fuzzy_matches_spacing_variant_of_hyphenated_genre(self):
        # The actual reported bug: "non fiction" (space) vs the catalogue's
        # "Non-Fiction" (hyphen) — a formatting difference, not a real typo.
        result = extract_known_term(
            "books about non fiction published after 2007",
            ["Non-Fiction", "Fantasy"],
        )
        assert result == "Non-Fiction"

    def test_fuzzy_matches_exact_after_normalizing_hyphen(self):
        result = extract_known_term("classic-fiction books", ["Classic Fiction"])
        assert result == "Classic Fiction"

    def test_fuzzy_match_does_not_fire_on_unrelated_text(self):
        assert (
            extract_known_term("tell me about Dune", ["Fantasy", "Non-Fiction"]) is None
        )

    def test_fuzzy_match_does_not_fire_on_abbreviation(self):
        # True abbreviations ("sci-fi" for "Science Fiction") need real-world
        # knowledge to resolve, not string similarity — left to the AI
        # planner or plain semantic search, not this fuzzy fallback.
        assert (
            extract_known_term("show me your sci-fi books", ["Science Fiction"]) is None
        )

    def test_exact_match_preferred_over_fuzzy(self):
        # Exact substring matching runs first; fuzzy is only a fallback.
        result = extract_known_term("fantasy books please", ["Fantasy", "Non-Fiction"])
        assert result == "Fantasy"


class TestBuildFilterSpec:
    def test_resolves_genre_from_catalogue(self):
        all_books = [
            {"metadata": {"genre": "Fantasy", "author": "Author A"}},
            {"metadata": {"genre": "Thriller", "author": "Author B"}},
        ]
        spec = build_filter_spec("fantasy books please", all_books)
        assert spec["genre"] == "Fantasy"
        assert spec["author"] is None
        assert spec["year"] is None

    def test_resolves_author_from_catalogue(self):
        all_books = [{"metadata": {"genre": "Sci-Fi", "author": "Frank Herbert"}}]
        spec = build_filter_spec("books by Frank Herbert", all_books)
        assert spec["author"] == "Frank Herbert"

    def test_resolves_year_independent_of_catalogue(self):
        spec = build_filter_spec("books before 2000", [])
        assert spec["year"] == {"op": "lt", "year": 2000}

    def test_combines_year_and_genre(self):
        all_books = [{"metadata": {"genre": "Fantasy", "author": "Author A"}}]
        spec = build_filter_spec("fantasy books before 2000", all_books)
        assert spec["genre"] == "Fantasy"
        assert spec["year"] == {"op": "lt", "year": 2000}


# ---------------------------------------------------------------------------
# book_matches
# ---------------------------------------------------------------------------


class TestBookMatches:
    def test_no_filters_matches_everything(self):
        assert book_matches(
            {"year": 1965}, {"year": None, "genre": None, "author": None}
        )

    def test_year_lt_matches(self):
        filters = {"year": {"op": "lt", "year": 2000}, "genre": None, "author": None}
        assert book_matches({"year": 1965}, filters) is True
        assert book_matches({"year": 2005}, filters) is False

    def test_year_between_matches(self):
        filters = {
            "year": {"op": "between", "start": 1980, "end": 1990},
            "genre": None,
            "author": None,
        }
        assert book_matches({"year": 1985}, filters) is True
        assert book_matches({"year": 1979}, filters) is False
        assert book_matches({"year": 1990}, filters) is True

    def test_string_year_metadata_is_coerced(self):
        filters = {"year": {"op": "eq", "year": 1965}, "genre": None, "author": None}
        assert book_matches({"year": "1965"}, filters) is True

    def test_unparseable_year_excludes_book(self):
        filters = {"year": {"op": "lt", "year": 2000}, "genre": None, "author": None}
        assert book_matches({"year": ""}, filters) is False
        assert book_matches({}, filters) is False

    def test_genre_filter_case_insensitive(self):
        filters = {"year": None, "genre": "fantasy", "author": None}
        assert book_matches({"genre": "Fantasy"}, filters) is True
        assert book_matches({"genre": "Thriller"}, filters) is False

    def test_author_filter_case_insensitive(self):
        filters = {"year": None, "genre": None, "author": "frank herbert"}
        assert book_matches({"author": "Frank Herbert"}, filters) is True
        assert book_matches({"author": "Andy Weir"}, filters) is False

    def test_combined_filters_require_all_to_match(self):
        filters = {
            "year": {"op": "lt", "year": 2000},
            "genre": "Fantasy",
            "author": None,
        }
        assert book_matches({"year": 1990, "genre": "Fantasy"}, filters) is True
        assert book_matches({"year": 1990, "genre": "Thriller"}, filters) is False
        assert book_matches({"year": 2005, "genre": "Fantasy"}, filters) is False


# ---------------------------------------------------------------------------
# describe_no_match
# ---------------------------------------------------------------------------


class TestDescribeNoMatch:
    def test_year_only(self):
        msg = describe_no_match(
            {"year": {"op": "lt", "year": 2000}, "genre": None, "author": None}
        )
        assert "before 2000" in msg

    def test_genre_only(self):
        msg = describe_no_match({"year": None, "genre": "Fantasy", "author": None})
        assert "Fantasy" in msg

    def test_author_only(self):
        msg = describe_no_match(
            {"year": None, "genre": None, "author": "Frank Herbert"}
        )
        assert "Frank Herbert" in msg

    def test_no_filters_generic_message(self):
        msg = describe_no_match({"year": None, "genre": None, "author": None})
        assert "couldn't find" in msg.lower()

    def test_combined_filters_all_mentioned(self):
        msg = describe_no_match(
            {"year": {"op": "gt", "year": 2010}, "genre": "Fantasy", "author": None}
        )
        assert "after 2010" in msg
        assert "Fantasy" in msg
