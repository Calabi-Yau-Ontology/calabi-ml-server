from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from .schemas import Lang, Mention, OutMention

# for OpenAI normalizer output

class CanonicalizeOut(BaseModel):
    normalized_text_en: str
    mentions: List[OutMention]

class NERRequest(BaseModel):
    text: str = Field(min_length=1)
    lang_hint: Optional[Literal["ko", "en"]] = None

class NERResponse(BaseModel):
    text: str
    lang: Lang
    normalized_text_en: Optional[str] = None
    mentions: List[Mention]
    errors: List[Dict[str, Any]] = Field(default_factory=list)
