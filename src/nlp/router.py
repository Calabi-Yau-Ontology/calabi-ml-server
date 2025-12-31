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

@router.post("/ner", response_model=NERResponse)
async def ner(payload: NERRequest) -> NERResponse:
    out = await ner_service.run(text=payload.text, lang_hint=payload.lang_hint)
    return NERResponse(**out)