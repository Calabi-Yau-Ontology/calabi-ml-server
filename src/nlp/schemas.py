from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

Lang = Literal["ko", "en", "unknown"]

# 넓게 허용
NERLabel = Literal[
    "Activity", "Location", "Person", "Project", "Topic", "Organization", "Food", "Movie", "TVShow", "Animal"
    "Date", "None" # "Particle", "Preposition", "Verb", "Adjective", "Adverb", "Conjunction"
]

class Span(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)

class NERInfo(BaseModel):
    label: NERLabel
    confidence: float = Field(ge=0.0, le=1.0)

class CanonicalInfo(BaseModel):
    en: str
    reason: Optional[str] = None

class Mention(BaseModel):
    surface: str
    span: Span
    ner: NERInfo
    canonical: CanonicalInfo

# for OpenAI normalizer output

class OutMention(BaseModel):
    surface: str
    span: Span
    canonical_en: str
    reason: Literal["abbr_expansion", "normalization", "unchanged", "unknown"]

# ---------- old schemas ----------

class Entity(BaseModel):
    text: str
    label: str = Field(..., description="엔티티 타입")
    start: int = Field(..., description="원문 상 시작 인덱스")
    end: int = Field(..., description="원문 상 끝 인덱스 (exclusive)")

## ---------- Suggest ----------

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

class SuggestItem(BaseModel):
    type: Literal["completion", "tag", "entity"]
    text: str
    score: float