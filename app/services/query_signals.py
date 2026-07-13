"""
Cheap, deterministic detection of "hard signals" — query characteristics
that the existing regex/substring metadata parsing was never designed to
handle at all, regardless of whether it happens to succeed or fail on a
given question. When one of these fires, the query should go straight to
the AI query planner rather than attempting (or retrying) deterministic
extraction first.

Pure functions only: no I/O, no AI calls.
"""

# A deliberately small set of very common English function words. Real
# English sentences of any length almost always contain several of these
# ("do", "you", "have", "the", "about", ...); genuine non-English text and
# bare title/name lookups ("Dune", "Emma") do not accumulate enough of them
# to register — the latter are short enough that len(words) < _MIN_WORDS
# skips the check entirely rather than misfiring on a lookup that's fine to
# leave to semantic/title search.
_ENGLISH_FUNCTION_WORDS = {
    "a",
    "about",
    "after",
    "all",
    "an",
    "and",
    "any",
    "are",
    "before",
    "book",
    "books",
    "by",
    "can",
    "could",
    "do",
    "find",
    "for",
    "give",
    "have",
    "i",
    "in",
    "is",
    "it",
    "me",
    "more",
    "my",
    "of",
    "on",
    "or",
    "published",
    "show",
    "some",
    "than",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "who",
    "with",
    "would",
    "written",
    "you",
    "your",
}

_MIN_WORDS_FOR_LANGUAGE_CHECK = 4
_MAX_ENGLISH_WORD_RATIO = 0.15

_COMPARATIVE_CUES = (
    "similar to",
    "similar books",
    "like ",
    "books like",
    "in the style of",
    "comparable to",
    "reminds me of",
    "something like",
    "in the vein of",
)


def is_likely_non_english(text: str) -> bool:
    """Whether this question likely isn't English, based on how few common
    English function words it contains. Skipped for very short questions
    (bare title/name lookups) since there's not enough signal either way,
    and those are already handled fine by semantic/title search."""
    words = [w.strip("?.,!¿¡") for w in text.lower().split()]
    words = [w for w in words if w]
    if len(words) < _MIN_WORDS_FOR_LANGUAGE_CHECK:
        return False

    hits = sum(1 for w in words if w in _ENGLISH_FUNCTION_WORDS)
    return (hits / len(words)) <= _MAX_ENGLISH_WORD_RATIO


def has_comparative_phrasing(text: str) -> bool:
    """Whether the question asks for books relative to a reference book
    ("similar to X", "like X but Y") — a recommendation-style intent the
    current retrieval (search using the query's own embedding) can't
    express at all, regardless of phrasing, so it always needs the planner."""
    q = text.lower()
    return any(cue in q for cue in _COMPARATIVE_CUES)


def has_hard_signal(text: str) -> bool:
    """Whether this question has a characteristic deterministic metadata
    parsing structurally cannot handle, so it should go straight to the AI
    query planner instead of attempting (or retrying) regex/substring
    extraction first."""
    return is_likely_non_english(text) or has_comparative_phrasing(text)
