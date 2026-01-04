from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict

from .schemas import Lang, Mention, OutMention

# for OpenAI normalizer output

class CanonicalizeOut(BaseModel):
    normalized_text_en: str
    mentions: List[OutMention]

class NERRequest(BaseModel):
    text: str = Field(min_length=1, description="원문 텍스트")
    lang_hint: Optional[Literal["ko", "en"]] = Field(
        default=None,
        description="언어 (선택)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "한강에서 친구들과 치맥하며 불꽃놀이 구경",
                "lang_hint": "ko",
            }
        }
    )

class NERResponse(BaseModel):
    text: str
    lang: Lang
    normalized_text_en: Optional[str] = None
    mentions: List[Mention]
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "한강에서 친구들과 치맥하며 불꽃놀이 구경",
                "lang": "ko",
                "normalized_text_en": "Watching fireworks while having chicken and beer with friends at Han River",
                "mentions": [
                    {
                        "surface": "한강",
                        "span": {"start": 0, "end": 2},
                        "ner": {"label": "Location"},
                        "canonical": {"en": "Han River", "reason": "normalization"},
                    },
                    {
                        "surface": "친구들",
                        "span": {"start": 5, "end": 8},
                        "ner": {"label": "Person"},
                        "canonical": {"en": "friend", "reason": "normalization"},
                    },
                    {
                        "surface":"치맥",
                        "span": {"start":10,"end":12},
                        "ner":{"label":"Food"}, 
                        "canonical": {"en":"chicken and beer","reason":"abbr_expansion"}
                    },
                    {
                        "surface":"불꽃놀이",
                        "span":{"start":15,"end":19},
                        "ner":{"label":"Activity"},
                        "canonical":{"en":"firework","reason":"normalization"}
                    },
                    {
                        "surface":"구경",
                        "span":{"start":20,"end":22},
                        "ner":{"label":"Activity"},
                        "canonical":{"en":"watching","reason":"normalization"}
                    }
                ],
                "errors": [],
            }
        }
    )
