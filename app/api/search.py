from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.embedding_service import EmbeddingsService
from app.infrastructure.vector_store import VectorStore
from app.services.rag_engine import RAGEngine
from app.exceptions import (
    RateLimitExceededException,
    EmbeddingException,
    VectorStoreException,
    AIProviderException,
    InvalidAIResponseException,
)

router = APIRouter()
_embedding_service = EmbeddingsService()
_vector_store = VectorStore()
_rag_engine = RAGEngine()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class BookResult(BaseModel):
    id: str
    title: str
    author: str
    year: str | None = None
    genre: str | None = None
    isbn: str | None = None
    shelf_number: str | None = None
    description: str
    similarity: float


class SearchResponse(BaseModel):
    results: list[BookResult]
    total: int


@router.post("/books", response_model=SearchResponse, summary="Semantic book search")
def search_books(body: SearchRequest):
    """
    Search the library catalogue using natural language.
    Returns books ranked by semantic similarity to the query.
    """
    try:
        embedding = _embedding_service.embed(body.query)
        candidates = _vector_store.search_books(embedding, top_k=body.top_k)
    except RateLimitExceededException as e:
        raise HTTPException(status_code=429, detail=str(e))
    except (EmbeddingException, VectorStoreException) as e:
        raise HTTPException(status_code=503, detail=f"Search error: {e}")

    results = [
        BookResult(
            id=b["id"],
            title=b["metadata"].get("title", "Unknown"),
            author=b["metadata"].get("author", "Unknown"),
            year=str(b["metadata"]["year"]) if b["metadata"].get("year") else None,
            genre=str(b["metadata"]["genre"]) if b["metadata"].get("genre") else None,
            isbn=str(b["metadata"]["isbn"]) if b["metadata"].get("isbn") else None,
            shelf_number=str(b["metadata"]["shelf_number"])
            if b["metadata"].get("shelf_number")
            else None,
            description=b["document"],
            similarity=round(b["similarity"], 4),
        )
        for b in candidates
    ]
    return SearchResponse(results=results, total=len(results))


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class SourceBook(BaseModel):
    title: str | None = None
    author: str | None = None
    year: str | None = None
    genre: str | None = None
    isbn: str | None = None
    shelf_number: str | None = None
    similarity: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceBook]
    cached: bool


@router.post("/ask", response_model=AskResponse, summary="RAG-grounded book Q&A")
def ask(body: AskRequest):
    """
    Ask a detailed question about the library collection.
    Returns an answer grounded solely in the catalogue with source citations.
    """
    try:
        result = _rag_engine.ask(body.question)
    except RateLimitExceededException as e:
        raise HTTPException(status_code=429, detail=str(e))
    except InvalidAIResponseException as e:
        raise HTTPException(status_code=422, detail=str(e))
    except (AIProviderException, EmbeddingException, VectorStoreException) as e:
        raise HTTPException(status_code=503, detail=f"AI provider error: {e}")

    return AskResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        cached=result.get("cached", False),
    )
