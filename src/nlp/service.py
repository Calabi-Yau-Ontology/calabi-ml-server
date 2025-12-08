from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

from . import constants
from .exceptions import NERError, SuggestionError
from .schemas import Entity, SuggestItem, SuggestRequest
from .utils import simple_tokenize


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


@dataclass(frozen=True)
class CursorContext:
    cursor: int
    token_start: int
    token_end: int
    fragment: str
    token_text: str

    @property
    def has_fragment(self) -> bool:
        return bool(self.fragment)

    @property
    def at_token_boundary(self) -> bool:
        return not self.fragment


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

    def _cursor_context(self, text: str, cursor: int | None) -> CursorContext:
        if cursor is None:
            cursor = len(text)
        length = len(text)
        cursor = max(0, min(length, cursor))

        start = cursor
        while start > 0 and not text[start - 1].isspace():
            start -= 1

        end = cursor
        while end < length and not text[end].isspace():
            end += 1

        fragment = text[start:cursor]
        token_text = text[start:end]
        return CursorContext(
            cursor=cursor,
            token_start=start,
            token_end=end,
            fragment=fragment,
            token_text=token_text,
        )

    def _entity_completions_for_active_token(
        self,
        text: str,
        cursor_ctx: CursorContext,
        entities: Sequence[Entity],
        history_tokens: Sequence[tuple[str, int]],
        popular_tags: Sequence[str],
    ) -> list[SuggestItem]:
        if not cursor_ctx.has_fragment:
            return []

        fragment_norm = cursor_ctx.fragment.strip().lower()
        if not fragment_norm:
            return []

        before = text[:cursor_ctx.token_start]
        after = text[cursor_ctx.token_end:]
        seen: set[str] = set()
        items: list[SuggestItem] = []

        def append_candidate(candidate: str, score: float) -> None:
            candidate = candidate.strip()
            if not candidate:
                return
            candidate_norm = candidate.lower()
            if candidate_norm in seen:
                return
            if not candidate_norm.startswith(fragment_norm):
                return
            if (
                candidate_norm == fragment_norm
                and cursor_ctx.token_text.strip().lower() == candidate_norm
            ):
                return
            seen.add(candidate_norm)
            completed = f"{before}{candidate}{after}"
            items.append(
                SuggestItem(
                    type="completion",
                    text=completed,
                    score=score,
                )
            )

        for ent in entities:
            append_candidate(
                ent.text,
                constants.SCORE_COMPLETION_ENTITY_ACTIVE,
            )

        for token, _freq in history_tokens:
            append_candidate(
                token,
                constants.SCORE_COMPLETION_HISTORY_ACTIVE,
            )

        for tag in popular_tags:
            append_candidate(
                tag,
                constants.SCORE_COMPLETION_POPULAR_ACTIVE,
            )

        return items

    def _next_word_recommendations(
        self,
        text: str,
        cursor_ctx: CursorContext,
        entities: Sequence[Entity],
        popular_tags: Sequence[str],
        history: Sequence[str],
    ) -> list[SuggestItem]:
        if not cursor_ctx.at_token_boundary:
            return []

        before = text[:cursor_ctx.cursor]
        after = text[cursor_ctx.cursor:]

        def build_base() -> str:
            if not before:
                return ""
            if before[-1].isspace():
                return before
            return f"{before} "

        def append_candidate(
            items_list: list[SuggestItem],
            candidate: str,
            score: float,
            seen_texts: set[str],
        ) -> None:
            candidate = candidate.strip()
            if not candidate:
                return
            base = build_base()
            suggestion_text = f"{base}{candidate}"
            if after and not after[0].isspace():
                suggestion_text = f"{suggestion_text} {after}"
            else:
                suggestion_text = f"{suggestion_text}{after}"

            if suggestion_text in seen_texts:
                return
            seen_texts.add(suggestion_text)
            items_list.append(
                SuggestItem(
                    type="completion",
                    text=suggestion_text,
                    score=score,
                )
            )

        items: list[SuggestItem] = []
        seen: set[str] = set()

        for ent in entities:
            append_candidate(
                items,
                ent.text,
                constants.SCORE_COMPLETION_NEXT_ENTITY,
                seen,
            )

        for tag in popular_tags:
            append_candidate(
                items,
                tag,
                constants.SCORE_COMPLETION_NEXT_TAG,
                seen,
            )

        for phrase in history:
            for token in simple_tokenize(phrase):
                append_candidate(
                    items,
                    token,
                    constants.SCORE_COMPLETION_NEXT_HISTORY,
                    seen,
                )

        return items

    def _history_token_candidates(
        self,
        history: Sequence[str],
    ) -> list[tuple[str, int]]:
        tokens: dict[str, list[Any]] = {}
        for phrase in history:
            for token in simple_tokenize(phrase):
                token_clean = token.strip()
                if len(token_clean) < 2:
                    continue
                key = token_clean.lower()
                if key not in tokens:
                    tokens[key] = [token_clean, 1]
                else:
                    tokens[key][1] += 1
        sorted_tokens = sorted(
            ((value[0], value[1]) for value in tokens.values()),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return sorted_tokens

    def _popular_tag_candidates(
        self,
        popular_tags: Sequence[str],
    ) -> list[str]:
        return [
            tag.strip()
            for tag in popular_tags
            if isinstance(tag, str) and len(tag.strip()) >= 2
        ]

    def _deduplicate_and_rank(
        self,
        suggestions: Sequence[SuggestItem],
        current_text: str,
    ) -> list[SuggestItem]:
        current_norm = _normalize_text(current_text)
        best_by_text: dict[str, SuggestItem] = {}

        for suggestion in suggestions:
            normalized = _normalize_text(suggestion.text)
            if not normalized or normalized == current_norm:
                continue

            existing = best_by_text.get(normalized)
            if existing is None or existing.score < suggestion.score:
                best_by_text[normalized] = suggestion

        ranked = sorted(
            best_by_text.values(),
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked[: constants.MAX_SUGGESTIONS]

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

            cursor_ctx = self._cursor_context(
                text,
                ctx.cursor_position if ctx else None,
            )

            history_tokens = self._history_token_candidates(history)
            popular_tag_candidates = self._popular_tag_candidates(popular_tags)

            suggestions: list[SuggestItem] = []

            # 0) 현재 단어에 대해 NER 기반 자동완성
            suggestions.extend(
                self._entity_completions_for_active_token(
                    text,
                    cursor_ctx,
                    entities,
                    history_tokens,
                    popular_tag_candidates,
                )
            )

            # 0-1) 다음 단어 추천
            suggestions.extend(
                self._next_word_recommendations(
                    text,
                    cursor_ctx,
                    entities,
                    popular_tag_candidates,
                    history,
                )
            )

            # 1) 과거 history(이벤트 제목들) 기반 completion
            prefix_text = text[: cursor_ctx.cursor]
            suggestions.extend(self._history_completions(prefix_text, history))

            # 2) generic suffix 기반 completion
            suggestions.extend(self._generic_completions(prefix_text))

            # 3) 엔티티 + 인기 태그 기반 tag 추천
            suggestions.extend(
                self._tag_suggestions(entities, popular_tag_candidates)
            )

            # 중복 제거 + score 순 정렬
            return self._deduplicate_and_rank(suggestions, text)

        except Exception as exc:  # noqa: BLE001
            raise SuggestionError(str(exc)) from exc


# “의존성 주입”까지는 아니고, 그냥 모듈 단위 싱글톤처럼 사용
ner_service = NERService()
suggestion_service = SuggestionService()
