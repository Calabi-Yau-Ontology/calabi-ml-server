from fastapi import APIRouter

from .dtos import (
    NERRequest,
    NERResponse
)
from .service import ner_service

router = APIRouter(
    prefix="/nlp",
    tags=["nlp"],
)

@router.post(
    "/ner",
    response_model=NERResponse,
    summary="Run multilingual NER",
    response_description="Normalized text, extracted mentions, and errors if any.",
)
async def ner(payload: NERRequest) -> NERResponse:
    """
    Execute the NER pipeline:
    1. 언어 감지
    2. OpenRouter 기반 canonicalization + labeling
    3. surface 기반 span 재탐색
    """
    out = await ner_service.run(text=payload.text, lang_hint=payload.lang_hint)
    return NERResponse(**out)
