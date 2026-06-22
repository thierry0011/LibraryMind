from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.rag_engine import RAGEngine

router = APIRouter()
_engine = RAGEngine()


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


@router.post("/", response_model=AskResponse, summary="RAG-grounded book Q&A")
def ask(body: AskRequest):
    """
    Ask a detailed question about the library collection.
    Returns an answer grounded solely in the catalogue with source citations.
    """
    try:
        result = _engine.ask(body.question)
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
