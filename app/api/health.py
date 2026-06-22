from fastapi import APIRouter
from pydantic import BaseModel

from app.infrastructure.usage_tracker import get_usage_tracker

router = APIRouter()
_request_count = 0


class HealthResponse(BaseModel):
    status: str
    daily_cost_usd: float
    total_requests: int


@router.get("/", response_model=HealthResponse, summary="Health check")
def health():
    """
    Returns service status, cumulative daily AI spend, and total request count
    since the last restart.
    """
    global _request_count
    _request_count += 1
    tracker = get_usage_tracker()
    return HealthResponse(
        status="ok",
        daily_cost_usd=round(tracker.get_daily_cost(), 6),
        total_requests=_request_count,
    )
