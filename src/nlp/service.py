from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

from . import constants
from .exceptions import NERError, SuggestionError
from .schemas import Entity, SuggestItem, SuggestRequest
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

    - 현재 입력 text에 대해:
      1) history(과거 이벤트 제목들)를 활용한 prefix 기반 completion
      2) generic한 suffix(회의/정리/리뷰) completion
      3) NER 엔티티 및 인기 태그 기반 tag 추천
    """
    def _history_completions(
        self,
        text: str,
        history: Sequence[str],
    ) -> list[SuggestItem]:
        text_norm = text.strip().lower()
        if not text_norm:
            return []

        seen: set[str] = set()
        completions: list[SuggestItem] = []

        for phrase in history:
            phrase_stripped = phrase.strip()
            if not phrase_stripped:
                continue

            phrase_norm = phrase_stripped.lower()
            # 아주 단순한 prefix 기반 자동완성
            if phrase_norm.startswith(text_norm) and phrase_norm != text_norm:
                if phrase_stripped in seen:
                    continue
                seen.add(phrase_stripped)
                completions.append(
                    SuggestItem(
                        type="completion",
                        text=phrase_stripped,
                        score=constants.SCORE_COMPLETION_HISTORY,
                    )
                )

        return completions

    def _generic_completions(self, text: str) -> list[SuggestItem]:
        base = text.strip()
        if not base:
            return []

        suffixes = [" 회의", " 정리", " 리뷰"]
        seen: set[str] = set()
        items: list[SuggestItem] = []

        for suf in suffixes:
            candidate = base + suf
            if candidate in seen:
                continue
            seen.add(candidate)
            items.append(
                SuggestItem(
                    type="completion",
                    text=candidate,
                    score=constants.SCORE_COMPLETION_GENERIC,
                )
            )
        return items

    def _tag_suggestions(
        self,
        entities: list[Entity],
        popular_tags: Sequence[str],
    ) -> list[SuggestItem]:
        items: list[SuggestItem] = []
        seen: set[str] = set()

        # 현재 텍스트에서 뽑힌 엔티티를 tag 후보로
        for e in entities:
            if e.text in seen:
                continue
            seen.add(e.text)
            items.append(
                SuggestItem(
                    type="tag",
                    text=e.text,
                    score=constants.SCORE_TAG_ENTITY,
                )
            )

        # 과거에서 뽑힌 인기 태그
        for tag in popular_tags:
            if tag in seen:
                continue
            seen.add(tag)
            items.append(
                SuggestItem(
                    type="tag",
                    text=tag,
                    score=constants.SCORE_TAG_POPULAR,
                )
            )

        return items

    def generate(
        self,
        request: SuggestRequest,
        entities: list[Entity],
    ) -> list[SuggestItem]:
        try:
            text = request.text
            ctx = request.context

            history: Sequence[str] = []
            popular_tags: Sequence[str] = []

            if ctx is not None and ctx.extra:
                extra: dict[str, Any] = ctx.extra
                history = extra.get("history", []) or []
                popular_tags = extra.get("popular_tags", []) or []

            suggestions: list[SuggestItem] = []

            # 1) 과거 history(이벤트 제목들) 기반 completion
            suggestions.extend(self._history_completions(text, history))

            # 2) generic suffix 기반 completion
            suggestions.extend(self._generic_completions(text))

            # 3) 엔티티 + 인기 태그 기반 tag 추천
            suggestions.extend(self._tag_suggestions(entities, popular_tags))

            # 중복 제거 + score 순 정렬
            dedup: dict[tuple[str, str], SuggestItem] = {}
            for s in suggestions:
                key = (s.type, s.text)
                # 더 높은 점수만 유지
                if key not in dedup or dedup[key].score < s.score:
                    dedup[key] = s

            result = sorted(dedup.values(), key=lambda x: x.score, reverse=True)
            return result

        except Exception as exc:  # noqa: BLE001
            raise SuggestionError(str(exc)) from exc


# “의존성 주입”까지는 아니고, 그냥 모듈 단위 싱글톤처럼 사용
ner_service = NERService()
suggestion_service = SuggestionService()
