import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.embedding_service import EmbeddingsService
from app.services.book_text import build_embedding_text
from app.infrastructure.vector_store import VectorStore
from app.exceptions import (
    EmbeddingException,
    VectorStoreException,
    RateLimitExceededException,
)

router = APIRouter()
_embedding_service = EmbeddingsService()
_vector_store = VectorStore()


class BookIngestRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    author: str = Field(..., min_length=1, max_length=200)
    year: str | None = None
    genre: str | None = None
    isbn: str | None = None
    shelf_number: str | None = None
    description: str = Field(..., min_length=1, max_length=5000)


class BookIngestResponse(BaseModel):
    id: str
    message: str


@router.post(
    "/",
    response_model=BookIngestResponse,
    status_code=201,
    summary="Add a book to the knowledge base",
)
def ingest_book(body: BookIngestRequest):
    """
    Embed a book description and store it in the vector knowledge base.
    Uses a deterministic ID (title + author) so re-submitting the same book
    performs an upsert rather than creating a duplicate.
    """
    try:
        book_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{body.title}:{body.author}"))
        embedding_text = build_embedding_text(
            {
                "title": body.title,
                "author": body.author,
                "year": body.year,
                "genre": body.genre,
                "shelf_number": body.shelf_number,
                "description": body.description,
            }
        )
        embedding = _embedding_service.embed(embedding_text)
        metadata = {
            "title": body.title,
            "author": body.author,
            "year": body.year or "",
            "genre": body.genre or "",
            "isbn": body.isbn or "",
            "shelf_number": body.shelf_number or "",
        }
        _vector_store.upsert_books(
            id=book_id,
            embedding=embedding,
            metadata=metadata,
            document=body.description,
        )
    except RateLimitExceededException as e:
        raise HTTPException(status_code=429, detail=str(e))
    except (EmbeddingException, VectorStoreException) as e:
        raise HTTPException(status_code=503, detail=f"Ingest error: {e}")

    return BookIngestResponse(id=book_id, message="Book ingested successfully.")
