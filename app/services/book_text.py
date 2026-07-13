def build_embedding_text(book: dict) -> str:
    """Rich, natural-language text for embedding a book — phrased close to how
    a patron would actually ask about it, so genre/year/author/location carry
    real semantic signal instead of living only in unembedded metadata."""
    title = book.get("title", "Unknown")
    author = book.get("author", "an unknown author")
    genre = book.get("genre") or "an unspecified genre"
    year = book.get("year") or "an unknown year"
    shelf_number = book.get("shelf_number")
    description = book.get("description", "")

    text = (
        f"{title} is a {genre} book written by {author} and published in {year}. "
        f"What is {title} about? {description} "
        f"Who wrote {title}? It was written by {author}. "
        f"What genre is {title}? It is {genre}. "
        f"When was {title} published? It came out in {year}."
    )
    if shelf_number:
        text += f" Where can I find {title} in the library? It is shelved at {shelf_number}."
    return text
