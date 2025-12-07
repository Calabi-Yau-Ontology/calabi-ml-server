from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

# ---------- NER ----------

class NERRequest(BaseModel):
    text: str = Field(..., description="분석할 원문 텍스트")
    
class Entity(BaseModel):
    text: str
    label: str = Field(..., description="엔티티 타입")
    start: int = Field(..., description="원문 상 시작 인덱스")
    end: int = Field(..., description="원문 상 끝 인덱스 (exclusive)")

class NERResponse(BaseModel):
    entities: list[Entity]

# ---------- Suggest ----------

class SuggestContext(BaseModel):
    field: Optional[str] = Field(
        default=None,
        description="어느 필드인지 (예: title, description)",
    )
    cursor_position: Optional[int] = Field(
        default=None,
        description="커서 위치 (선택)",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="추가 컨텍스트",
    )

class SuggestRequest(BaseModel):
    user_id: str
    text: str
    context: Optional[SuggestContext] = None

class SuggestItem(BaseModel):
    type: Literal["completion", "tag", "entity"]
    text: str
    score: float

class SuggestResponse(BaseModel):
    suggestions: list[SuggestItem]
    entities: list[Entity]