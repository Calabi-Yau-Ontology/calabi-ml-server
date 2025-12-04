from typing import Optional
from pydantic import BaseModel, Field

from schemas import Entity
from schemas import SuggestContext, SuggestItem

class NERRequest(BaseModel):
    text: str = Field(..., description="분석할 원문 텍스트")

class NERResponse(BaseModel):
    entities: list[Entity]

class SuggestRequest(BaseModel):
    user_id: str
    text: str
    context: Optional[SuggestContext] = None

class SuggestResponse(BaseModel):
    suggestions: list[SuggestItem]
    entities: list[Entity]