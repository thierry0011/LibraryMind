import sys
import os
import json
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.embedding_service import EmbeddingsService
from app.infrastructure.vector_store import VectorStore
from logger import get_logger

logger = get_logger(__name__)


def main():
    embedding_service = EmbeddingsService()
    vector_store = VectorStore()

    book_path = os.path.join(os.path.dirname(__file__), "..", "data", "books.json")

    with open(book_path, "r") as f:
        books = json.load(f)

    logger.info(f"Loaded {len(books)} books from catalogue")

    texts = [
        f"book title: {book['title']}, book author: {book['author']}, book description: {book['description']}"
        for book in books
    ]

    batch_size = 5
    all_embeddings = []

    logger.info("Generating embeddings for all books...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info(
            f"Embedding batch {i // batch_size + 1} of {math.ceil(len(texts) / batch_size)}"
        )
        batch_embeddings = embedding_service.embed_batch(batch)
        all_embeddings.extend(batch_embeddings)

    embeddings = all_embeddings
    logger.info("Embeddings generated successfully")

    logger.info("Upserting books into vector store...")
    for book, embedding in zip(books, embeddings):
        vector_store.upsert_books(
            id=book["id"],
            embedding=embedding,
            metadata={
                "title": book["title"],
                "author": book["author"],
                "year": book["year"],
                "genre": book["genre"],
            },
            document=book["description"],
        )

    logger.info(f"Seeding complete. {len(books)} books stored in ChromaDB.")


if __name__ == "__main__":
    main()
