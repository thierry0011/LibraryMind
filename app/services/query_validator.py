"""
Validates AI query-planner output against the catalogue's real values.
The planner suggests; this module decides. A field that doesn't check out
against real catalogue data is discarded rather than trusted or
auto-corrected — if the model ignored the grounded genre list and invented
"Adventure", the answer is null, not a guess at what it might have meant.
"""

_YEAR_OP_MAP = {"<": "lt", "<=": "lte", ">": "gt", ">=": "gte", "==": "eq"}


def _validate_year_filter(raw_year_filter) -> dict | None:
    """Convert the planner's {"operator": ..., "value": ...} shape into the
    {"op": ..., "year": ...} / {"op": "between", "start": ..., "end": ...}
    shape metadata_filter.book_matches() already understands, rejecting
    anything malformed rather than guessing at intent."""
    if not isinstance(raw_year_filter, dict):
        return None

    operator = raw_year_filter.get("operator")
    value = raw_year_filter.get("value")

    if operator == "between":
        value2 = raw_year_filter.get("value2")
        if not isinstance(value, int) or not isinstance(value2, int):
            return None
        start, end = sorted((value, value2))
        return {"op": "between", "start": start, "end": end}

    op = _YEAR_OP_MAP.get(operator)
    if op is None or not isinstance(value, int):
        return None
    return {"op": op, "year": value}


def _resolve_known_value(value, known_values) -> str | None:
    """Case-insensitively resolve `value` to its canonical catalogue form,
    or None if it isn't a real catalogue value at all."""
    if not isinstance(value, str) or not value.strip():
        return None
    lookup = {v.lower(): v for v in known_values if v}
    return lookup.get(value.strip().lower())


def validate_query_plan(raw_plan: dict, known_genres, known_authors) -> dict:
    """Sanitize a raw AI-produced plan into a filters dict shaped exactly
    like metadata_filter.build_filter_spec()'s output (so it flows into the
    existing book_matches()/describe_no_match() unchanged), plus a title/
    semantic_query residual for the semantic search fallback path.

    Always returns a dict — never raises, never returns None — so callers
    can rely on its shape even when every field validation fails and the
    plan simply carries no usable signal.
    """
    genre = _resolve_known_value(raw_plan.get("genre"), known_genres)
    author = _resolve_known_value(raw_plan.get("author"), known_authors)
    year_filter = _validate_year_filter(raw_plan.get("year_filter"))

    title = raw_plan.get("title")
    title = title.strip() if isinstance(title, str) and title.strip() else None

    semantic_query = raw_plan.get("semantic_query")
    semantic_query = (
        semantic_query.strip()
        if isinstance(semantic_query, str) and semantic_query.strip()
        else None
    )

    filters = {"year": year_filter, "genre": genre, "author": author}
    has_structured_filter = bool(year_filter or genre or author)

    return {
        "filters": filters,
        "title": title,
        "semantic_query": semantic_query,
        "has_structured_filter": has_structured_filter,
    }
