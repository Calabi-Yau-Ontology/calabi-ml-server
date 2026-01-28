from fastapi import APIRouter, HTTPException

from .dtos import (
    ProposeRequest,
    ProposeResponse,
    ClassifyRequest,
    ClassifyResponse,
)
from .service import ontology_service


def _status_from_errors(errors: list[dict]) -> int:
    stages = {e.get("stage") for e in errors}
    if "validation" in stages:
        return 422
    if "disabled" in stages:
        return 503
    if "llm_call" in stages:
        return 502
    return 500

router = APIRouter(
    prefix="/ontology",
    tags=["ontology"],
)


@router.post(
    "/propose",
    response_model=ProposeResponse,
    summary="Propose ontology elements from CQs",
)
async def propose(payload: ProposeRequest) -> ProposeResponse:
    """
    Propose taxonomy elements from CQs.
    """
    result = await ontology_service.propose(payload)
    if result.errors:
        raise HTTPException(status_code=_status_from_errors(result.errors), detail=result.errors)
    return result


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify concepts into existing taxonomy",
)
async def classify(payload: ClassifyRequest) -> ClassifyResponse:
    """
    Classify concepts into existing taxonomy.
    """
    if payload.scope in ("concept", "both") and not payload.concepts:
        raise HTTPException(
            status_code=422,
            detail="scope requires concepts but concepts is empty",
        )
    if payload.scope in ("event", "both"):
        if not payload.eventId:
            raise HTTPException(
                status_code=422,
                detail="scope requires eventId",
            )
        if not (payload.eventTitle or payload.eventNormalizedTextEn):
            raise HTTPException(
                status_code=422,
                detail="scope requires event context (eventTitle or eventNormalizedTextEn)",
            )
    result = await ontology_service.classify(payload)
    if result.errors:
        raise HTTPException(status_code=_status_from_errors(result.errors), detail=result.errors)
    return result
