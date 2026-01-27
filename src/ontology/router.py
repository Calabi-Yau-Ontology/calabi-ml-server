from fastapi import APIRouter

from .dtos import (
    ProposeRequest,
    ProposeResponse,
    ClassifyRequest,
    ClassifyResponse,
)
from .service import ontology_service

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
    return await ontology_service.propose(payload)


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify concepts into existing taxonomy",
)
async def classify(payload: ClassifyRequest) -> ClassifyResponse:
    """
    Classify concepts into existing taxonomy.
    """
    return await ontology_service.classify(payload)
