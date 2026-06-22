from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.classification_service import ClassificationService

router = APIRouter()
_service = ClassificationService()


class ClassifyRequest(BaseModel):
    ticket: str = Field(..., min_length=1, max_length=2000)


class ClassifyResponse(BaseModel):
    category: str
    priority: str
    sentiment: str
    department: str
    summary: str


@router.post("/", response_model=ClassifyResponse, summary="Classify support ticket")
def classify(body: ClassifyRequest):
    """
    Classify a library support ticket by category, priority, and sentiment,
    and route it to the appropriate department.

    Categories: account | borrowing | technical | complaint | suggestion | general
    Priorities: low | medium | high | urgent
    Sentiments: positive | neutral | negative
    """
    try:
        result = _service.classify(body.ticket)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        msg = str(e)
        if "rate limit" in msg.lower():
            raise HTTPException(status_code=429, detail=msg)
        raise HTTPException(status_code=503, detail=f"AI provider error: {msg}")

    return ClassifyResponse(**result)
