from fastapi import APIRouter

from .dtos import (
    NERRequest,
    NERResponse,
    SuggestRequest,
    SuggestResponse,
)
from .service import ner_service, suggestion_service

router = APIRouter(
    prefix="/nlp",
    tags=["nlp"],
)

@router.post("/ner", response_model=NERResponse)
async def ner(payload: NERRequest) -> NERResponse:
    out = await ner_service.run(text=payload.text, lang_hint=payload.lang_hint)
    print("NER Response:", out)  # DEBUG
    return NERResponse(**out)

@router.post("/suggest", response_model=SuggestResponse)
async def suggest_terms(payload: SuggestRequest) -> SuggestResponse:
    """
    일정 입력 중 용어/문구 추천.
    - 내부적으로 NER 한 번 수행 후
    - NER 결과 + 텍스트 기반 추천 생성
    """
    entities = ner_service.extract_entities(payload.text)
    suggestions = suggestion_service.generate(payload, entities)
    return SuggestResponse(suggestions=suggestions, entities=entities)