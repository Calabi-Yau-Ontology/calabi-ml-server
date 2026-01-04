from typing import Literal, Optional
from pydantic import BaseModel, Field

Lang = Literal["ko", "en", "unknown"]

# 넓게 허용
NERLabel = Literal[
    "Activity", "Location", "Person", "Project", "Topic", "Organization", "Food", "Movie", "TVShow", "Animal",
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
    canonical_en: str
    anchor_en: str
    reason: Literal["abbr_expansion", "normalization", "unchanged", "unknown"]
