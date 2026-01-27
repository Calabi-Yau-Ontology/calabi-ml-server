from fastapi import APIRouter, HTTPException

from .dtos import (
    ProposeRequest,
    ProposeResponse,
    ClassifyRequest,
    ClassifyResponse,
)

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
    Placeholder endpoint.
    Commit 1 only provides schema and routing.
    """
    raise HTTPException(status_code=501, detail="not_implemented")


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify concepts into existing taxonomy",
)
async def classify(payload: ClassifyRequest) -> ClassifyResponse:
    """
    Placeholder endpoint.
    Commit 1 only provides schema and routing.
    """
    raise HTTPException(status_code=501, detail="not_implemented")
