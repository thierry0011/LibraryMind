from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.embedding_service import EmbeddingsService
from app.infrastructure.vector_store import VectorStore

router = APIRouter()
_embedding_service = EmbeddingsService()
_vector_store = VectorStore()


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


@router.post("/", response_model=SearchResponse, summary="Semantic book search")
def search(body: SearchRequest):
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
