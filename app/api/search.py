from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.embedding_service import EmbeddingsService
from app.infrastructure.vector_store import VectorStore
from app.services.rag_engine import RAGEngine

router = APIRouter()
_embedding_service = EmbeddingsService()
_vector_store = VectorStore()
_rag_engine = RAGEngine()


# ---------------------------------------------------------------------------
# POST /search/books
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class BookResult(BaseModel):
    id: str
    title: str
    author: str
    year: str | None = None
    genre: str | None = None
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
    except Exception as e:
        msg = str(e)
        if "rate limit" in msg.lower():
            raise HTTPException(status_code=429, detail=msg)
        raise HTTPException(status_code=503, detail=f"Search error: {msg}")

    results = [
        BookResult(
            id=b["id"],
            title=b["metadata"].get("title", "Unknown"),
            author=b["metadata"].get("author", "Unknown"),
            year=b["metadata"].get("year") or None,
            genre=b["metadata"].get("genre") or None,
            description=b["document"],
            similarity=round(b["similarity"], 4),
        )
        for b in candidates
    ]
    return SearchResponse(results=results, total=len(results))


# ---------------------------------------------------------------------------
# POST /search/ask
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class SourceBook(BaseModel):
    title: str | None = None
    author: str | None = None
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
    except Exception as e:
        msg = str(e)
        if "rate limit" in msg.lower():
            raise HTTPException(status_code=429, detail=msg)
        raise HTTPException(status_code=503, detail=f"AI provider error: {msg}")

    return AskResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        cached=result.get("cached", False),
    )
