from typing import Final

DEFAULT_ENTITY_LABEL: Final[str] = "Concept"

# 추천 score 기본값
SCORE_COMPLETION: Final[float] = 0.7
SCORE_TAG: Final[float] = 0.6

# 추천 score 기본값 - new
SCORE_COMPLETION_HISTORY: Final[float] = 0.9
SCORE_COMPLETION_GENERIC: Final[float] = 0.7
SCORE_TAG_ENTITY: Final[float] = 0.65
SCORE_TAG_POPULAR: Final[float] = 0.6