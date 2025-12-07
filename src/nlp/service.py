from dataclasses import dataclass
from typing import List

from . import constants
from .exceptions import NERError, SuggestionError
from .schemas import Entity, SuggestItem
from .utils import simple_tokenize


@dataclass
class NERService:
    """
    현재는 rule-based 더미 구현.
    나중에 HF 모델 로딩해서 여기에 붙이면 됨.
    """

    def extract_entities(self, text: str) -> List[Entity]:
        if not text:
            return []

        entities: list[Entity] = []
        cursor = 0

        try:
            tokens = simple_tokenize(text)
            for token in tokens:
                start = text.find(token, cursor)
                if start == -1:
                    continue
                end = start + len(token)

                if len(token) >= 2:
                    entities.append(
                        Entity(
                            text=token,
                            label=constants.DEFAULT_ENTITY_LABEL,
                            start=start,
                            end=end,
                        )
                    )

                cursor = end
        except Exception as exc:  # noqa: BLE001
            raise NERError(str(exc)) from exc

        return entities


@dataclass
class SuggestionService:
    """
    Calabi용 용어/문구 추천 엔진 (MVP).
    """

    def generate(
        self,
        user_id: str,
        text: str,
        entities: list[Entity],
    ) -> List[SuggestItem]:
        if not text:
            return []

        try:
            suggestions: list[SuggestItem] = []

            last_token = text.strip().split()[-1] if text.strip() else ""

            base_candidates: list[str] = []
            if last_token:
                base_candidates.extend(
                    [
                        f"{last_token} 회의",
                        f"{last_token} 정리",
                        f"{last_token} 리뷰",
                    ]
                )

            tag_candidates = [e.text for e in entities]

            seen: set[str] = set()

            for cand in base_candidates:
                if cand in seen:
                    continue
                seen.add(cand)
                suggestions.append(
                    SuggestItem(
                        type="completion",
                        text=cand,
                        score=constants.SCORE_COMPLETION,
                    )
                )

            for cand in tag_candidates:
                if cand in seen:
                    continue
                seen.add(cand)
                suggestions.append(
                    SuggestItem(
                        type="tag",
                        text=cand,
                        score=constants.SCORE_TAG,
                    )
                )

            suggestions.sort(key=lambda x: x.score, reverse=True)
            return suggestions

        except Exception as exc:  # noqa: BLE001
            raise SuggestionError(str(exc)) from exc

# “의존성 주입”까지는 아니고, 그냥 모듈 단위 싱글톤처럼 사용
ner_service = NERService()
suggestion_service = SuggestionService()
