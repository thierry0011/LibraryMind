"""
Tests for services.query_validator.validate_query_plan — sanitizing raw AI
query-planner output against the real catalogue. Pure function, no I/O.
"""

from services.query_validator import validate_query_plan

_KNOWN_GENRES = {"Fantasy", "Science Fiction", "Non-Fiction"}
_KNOWN_AUTHORS = {"Frank Herbert", "Yuval Noah Harari"}


class TestValidateGenre:
    def test_valid_genre_is_kept(self):
        plan = validate_query_plan({"genre": "Non-Fiction"}, _KNOWN_GENRES, _KNOWN_AUTHORS)
        assert plan["filters"]["genre"] == "Non-Fiction"

    def test_case_insensitive_genre_resolves_to_canonical_form(self):
        plan = validate_query_plan({"genre": "non-fiction"}, _KNOWN_GENRES, _KNOWN_AUTHORS)
        assert plan["filters"]["genre"] == "Non-Fiction"

    def test_hallucinated_genre_is_rejected(self):
        plan = validate_query_plan({"genre": "Adventure"}, _KNOWN_GENRES, _KNOWN_AUTHORS)
        assert plan["filters"]["genre"] is None

    def test_missing_genre_is_none(self):
        plan = validate_query_plan({}, _KNOWN_GENRES, _KNOWN_AUTHORS)
        assert plan["filters"]["genre"] is None

    def test_non_string_genre_is_rejected(self):
        plan = validate_query_plan({"genre": 123}, _KNOWN_GENRES, _KNOWN_AUTHORS)
        assert plan["filters"]["genre"] is None


class TestValidateAuthor:
    def test_valid_author_is_kept(self):
        plan = validate_query_plan(
            {"author": "Frank Herbert"}, _KNOWN_GENRES, _KNOWN_AUTHORS
        )
        assert plan["filters"]["author"] == "Frank Herbert"

    def test_case_insensitive_author_resolves_to_canonical_form(self):
        plan = validate_query_plan(
            {"author": "frank herbert"}, _KNOWN_GENRES, _KNOWN_AUTHORS
        )
        assert plan["filters"]["author"] == "Frank Herbert"

    def test_hallucinated_author_is_rejected(self):
        plan = validate_query_plan(
            {"author": "Someone Made Up"}, _KNOWN_GENRES, _KNOWN_AUTHORS
        )
        assert plan["filters"]["author"] is None


class TestValidateYearFilter:
    def test_valid_gt_filter(self):
        plan = validate_query_plan(
            {"year_filter": {"operator": ">", "value": 2007}},
            _KNOWN_GENRES,
            _KNOWN_AUTHORS,
        )
        assert plan["filters"]["year"] == {"op": "gt", "year": 2007}

    def test_valid_lt_filter(self):
        plan = validate_query_plan(
            {"year_filter": {"operator": "<", "value": 2000}},
            _KNOWN_GENRES,
            _KNOWN_AUTHORS,
        )
        assert plan["filters"]["year"] == {"op": "lt", "year": 2000}

    def test_valid_between_filter(self):
        plan = validate_query_plan(
            {"year_filter": {"operator": "between", "value": 1980, "value2": 1990}},
            _KNOWN_GENRES,
            _KNOWN_AUTHORS,
        )
        assert plan["filters"]["year"] == {"op": "between", "start": 1980, "end": 1990}

    def test_between_filter_sorts_reversed_values(self):
        plan = validate_query_plan(
            {"year_filter": {"operator": "between", "value": 1990, "value2": 1980}},
            _KNOWN_GENRES,
            _KNOWN_AUTHORS,
        )
        assert plan["filters"]["year"] == {"op": "between", "start": 1980, "end": 1990}

    def test_missing_year_filter_is_none(self):
        plan = validate_query_plan({}, _KNOWN_GENRES, _KNOWN_AUTHORS)
        assert plan["filters"]["year"] is None

    def test_null_year_filter_is_none(self):
        plan = validate_query_plan({"year_filter": None}, _KNOWN_GENRES, _KNOWN_AUTHORS)
        assert plan["filters"]["year"] is None

    def test_unknown_operator_is_rejected(self):
        plan = validate_query_plan(
            {"year_filter": {"operator": "roughly", "value": 2000}},
            _KNOWN_GENRES,
            _KNOWN_AUTHORS,
        )
        assert plan["filters"]["year"] is None

    def test_non_integer_value_is_rejected(self):
        plan = validate_query_plan(
            {"year_filter": {"operator": ">", "value": "a long time ago"}},
            _KNOWN_GENRES,
            _KNOWN_AUTHORS,
        )
        assert plan["filters"]["year"] is None

    def test_between_missing_value2_is_rejected(self):
        plan = validate_query_plan(
            {"year_filter": {"operator": "between", "value": 1980}},
            _KNOWN_GENRES,
            _KNOWN_AUTHORS,
        )
        assert plan["filters"]["year"] is None

    def test_malformed_year_filter_shape_is_rejected(self):
        plan = validate_query_plan(
            {"year_filter": "sometime in the 2000s"}, _KNOWN_GENRES, _KNOWN_AUTHORS
        )
        assert plan["filters"]["year"] is None


class TestValidateTitleAndSemanticQuery:
    def test_title_is_kept_when_present(self):
        plan = validate_query_plan({"title": "Dune"}, _KNOWN_GENRES, _KNOWN_AUTHORS)
        assert plan["title"] == "Dune"

    def test_blank_title_is_none(self):
        plan = validate_query_plan({"title": "   "}, _KNOWN_GENRES, _KNOWN_AUTHORS)
        assert plan["title"] is None

    def test_semantic_query_is_kept_when_present(self):
        plan = validate_query_plan(
            {"semantic_query": "books about habit formation"},
            _KNOWN_GENRES,
            _KNOWN_AUTHORS,
        )
        assert plan["semantic_query"] == "books about habit formation"

    def test_non_string_semantic_query_is_none(self):
        plan = validate_query_plan({"semantic_query": 42}, _KNOWN_GENRES, _KNOWN_AUTHORS)
        assert plan["semantic_query"] is None


class TestHasStructuredFilter:
    def test_true_when_genre_resolved(self):
        plan = validate_query_plan({"genre": "Fantasy"}, _KNOWN_GENRES, _KNOWN_AUTHORS)
        assert plan["has_structured_filter"] is True

    def test_true_when_year_resolved(self):
        plan = validate_query_plan(
            {"year_filter": {"operator": ">", "value": 2000}},
            _KNOWN_GENRES,
            _KNOWN_AUTHORS,
        )
        assert plan["has_structured_filter"] is True

    def test_false_when_nothing_resolved(self):
        plan = validate_query_plan(
            {"title": None, "semantic_query": "space battles"},
            _KNOWN_GENRES,
            _KNOWN_AUTHORS,
        )
        assert plan["has_structured_filter"] is False

    def test_false_when_only_title_present(self):
        plan = validate_query_plan({"title": "Dune"}, _KNOWN_GENRES, _KNOWN_AUTHORS)
        assert plan["has_structured_filter"] is False

    def test_false_for_completely_empty_plan(self):
        plan = validate_query_plan({}, _KNOWN_GENRES, _KNOWN_AUTHORS)
        assert plan["has_structured_filter"] is False


class TestValidateQueryPlanNeverRaises:
    def test_missing_all_fields(self):
        plan = validate_query_plan({}, _KNOWN_GENRES, _KNOWN_AUTHORS)
        assert plan["filters"] == {"year": None, "genre": None, "author": None}
        assert plan["title"] is None
        assert plan["semantic_query"] is None

    def test_empty_known_sets(self):
        plan = validate_query_plan({"genre": "Fantasy"}, set(), set())
        assert plan["filters"]["genre"] is None
