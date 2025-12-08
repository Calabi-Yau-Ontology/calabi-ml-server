from typing import Final

DEFAULT_ENTITY_LABEL: Final[str] = "Concept"

# 추천 score 기본값
SCORE_COMPLETION: Final[float] = 0.5
SCORE_TAG: Final[float] = 0.6

# 추천 score 기본값 - new
SCORE_COMPLETION_HISTORY: Final[float] = 0.9
SCORE_COMPLETION_GENERIC: Final[float] = 0.5
SCORE_TAG_ENTITY: Final[float] = 0.65
SCORE_TAG_POPULAR: Final[float] = 0.6

# NER 기반 완성/추천 강도
SCORE_COMPLETION_ENTITY_ACTIVE: Final[float] = 0.95
SCORE_COMPLETION_NEXT_ENTITY: Final[float] = 0.85
SCORE_COMPLETION_NEXT_TAG: Final[float] = 0.75
SCORE_COMPLETION_NEXT_HISTORY: Final[float] = 0.7

# 전체 추천 수 상한
MAX_SUGGESTIONS: Final[int] = 20
