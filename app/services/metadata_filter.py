"""
Lightweight, dependency-free parsing for catalogue metadata queries
(publication-year ranges, genre, author) that dense vector similarity
search cannot answer reliably — e.g. "books published before 2000" or
"how many fantasy books do you have".

Pure functions only: no I/O, no AI calls. RAGEngine decides *when* to use
these (via `looks_like_metadata_query`) and supplies the catalogue data
needed to resolve genre/author names (via `build_filter_spec`).
"""

import re

_ENUMERATION_CUES = (
    "how many",
    "list all",
    "list every",
    "all books",
    "every book",
    "count of",
    "number of",
)

_YEAR = r"(\d{4})"
# Optional filler between a comparison keyword and the year itself, e.g.
# "before the year 2000" or "since year 2015".
_FILLER = r"(?:the\s+)?(?:year\s+)?"


def has_enumeration_cue(question: str) -> bool:
    q = question.lower()
    return any(cue in q for cue in _ENUMERATION_CUES)


def parse_year_filter(question: str) -> dict | None:
    """Parse a publication-year constraint out of a natural-language question."""
    q = question.lower()

    m = re.search(rf"between\s+{_FILLER}{_YEAR}\s+and\s+{_FILLER}{_YEAR}", q)
    if m:
        start, end = sorted((int(m.group(1)), int(m.group(2))))
        return {"op": "between", "start": start, "end": end}

    m = re.search(r"\b((?:19|20)\d)0s\b", q)
    if m:
        decade_start = int(m.group(1)) * 10
        return {"op": "between", "start": decade_start, "end": decade_start + 9}

    m = re.search(rf"\b(?:before|prior to|earlier than)\s+{_FILLER}{_YEAR}", q)
    if m:
        return {"op": "lt", "year": int(m.group(1))}

    m = re.search(rf"\b(?:on or before|up to|until|through)\s+{_FILLER}{_YEAR}", q)
    if m:
        return {"op": "lte", "year": int(m.group(1))}

    m = re.search(rf"\b(?:on or after|from)\s+{_FILLER}{_YEAR}", q)
    if m:
        return {"op": "gte", "year": int(m.group(1))}

    m = re.search(rf"\b(?:after|since|later than)\s+{_FILLER}{_YEAR}", q)
    if m:
        return {"op": "gt", "year": int(m.group(1))}

    m = re.search(rf"\b(?:in|published in|released in)\s+{_FILLER}{_YEAR}\b", q)
    if m:
        return {"op": "eq", "year": int(m.group(1))}

    return None


def looks_like_metadata_query(question: str) -> bool:
    """Cheap, catalogue-independent check for whether a query needs filtering
    over structured fields rather than semantic similarity over descriptions."""
    return parse_year_filter(question) is not None or has_enumeration_cue(question)


_VAGUE_FOLLOWUP_CUES = (
    "tell me more",
    "more about",
    "this book",
    "that book",
    "this one",
    "that one",
    "about it",
)


def looks_like_vague_followup(question: str) -> bool:
    """Cheap check for a follow-up that refers back to a book already
    discussed ("tell me more about this book") instead of naming one
    directly. Used only as a last resort when semantic search finds
    nothing, so a loose match is fine — it can only recover an answer,
    never displace a real one."""
    q = question.lower()
    return any(cue in q for cue in _VAGUE_FOLLOWUP_CUES)


def extract_known_term(question: str, candidates) -> str | None:
    """Return the longest known catalogue value (genre/author) mentioned in the
    question, so we only match real values rather than guessing from keywords."""
    q = question.lower()
    for term in sorted((c for c in candidates if c), key=len, reverse=True):
        if term.lower() in q:
            return term
    return None


def build_filter_spec(question: str, all_books: list) -> dict:
    """Build a filter spec from the question, resolving genre/author against
    the values that actually exist in the catalogue."""
    known_genres = {b["metadata"].get("genre") for b in all_books if b["metadata"].get("genre")}
    known_authors = {b["metadata"].get("author") for b in all_books if b["metadata"].get("author")}
    return {
        "year": parse_year_filter(question),
        "genre": extract_known_term(question, known_genres),
        "author": extract_known_term(question, known_authors),
    }


def _year_matches(raw_year, year_filter: dict) -> bool:
    try:
        year = int(raw_year)
    except (TypeError, ValueError):
        return False

    op = year_filter["op"]
    if op == "lt":
        return year < year_filter["year"]
    if op == "lte":
        return year <= year_filter["year"]
    if op == "gt":
        return year > year_filter["year"]
    if op == "gte":
        return year >= year_filter["year"]
    if op == "eq":
        return year == year_filter["year"]
    if op == "between":
        return year_filter["start"] <= year <= year_filter["end"]
    return False


def book_matches(metadata: dict, filters: dict) -> bool:
    year_filter = filters.get("year")
    if year_filter is not None and not _year_matches(metadata.get("year"), year_filter):
        return False

    genre_filter = filters.get("genre")
    if genre_filter is not None and str(metadata.get("genre", "")).lower() != genre_filter.lower():
        return False

    author_filter = filters.get("author")
    if author_filter is not None and str(metadata.get("author", "")).lower() != author_filter.lower():
        return False

    return True


def _describe_year_filter(year_filter: dict) -> str:
    op = year_filter["op"]
    if op == "lt":
        return f"published before {year_filter['year']}"
    if op == "lte":
        return f"published on or before {year_filter['year']}"
    if op == "gt":
        return f"published after {year_filter['year']}"
    if op == "gte":
        return f"published on or after {year_filter['year']}"
    if op == "eq":
        return f"published in {year_filter['year']}"
    if op == "between":
        return f"published between {year_filter['start']} and {year_filter['end']}"
    return ""


def describe_no_match(filters: dict) -> str:
    parts = []
    year_filter = filters.get("year")
    if year_filter:
        parts.append(_describe_year_filter(year_filter))
    if filters.get("genre"):
        parts.append(f"in the {filters['genre']} genre")
    if filters.get("author"):
        parts.append(f"by {filters['author']}")

    if not parts:
        return (
            "I couldn't find any books in our catalog matching that. "
            "Try rephrasing, or ask about a different topic."
        )
    return (
        f"I couldn't find any books in our catalog {' '.join(parts)}. "
        "Try rephrasing, or ask about a different topic."
    )
