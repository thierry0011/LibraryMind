from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.summarisation_service import SummarizationService

router = APIRouter()
_service = SummarizationService()


class SummariseRequest(BaseModel):
    reviews: list[str] = Field(..., min_length=1)


class SummariseResponse(BaseModel):
    overall_sentiment: str
    average_rating: float
    key_themes: list[str]
    praise: list[str]
    criticism: list[str]
    recommendation: str


@router.post(
    "/reviews", response_model=SummariseResponse, summary="Summarise book reviews"
)
def summarise(body: SummariseRequest):
    """
    Summarise a collection of book reviews.
    Extracts overall sentiment, average rating, key themes, praise points,
    criticism points, and a reading recommendation.
    """
    try:
        result = _service.summarize(body.reviews)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        msg = str(e)
        if "rate limit" in msg.lower():
            raise HTTPException(status_code=429, detail=msg)
        raise HTTPException(status_code=503, detail=f"AI provider error: {msg}")

    return SummariseResponse(**result)
