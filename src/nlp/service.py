from dataclasses import dataclass
from typing import Any, Dict, List, Sequence
from langdetect import detect

from src.config import settings
from .models.ner_engine import NEREngine
from .openai.normalizer import canonicalize_with_normalized_sentence
from .utils import clamp01, similarity

from . import constants
from .exceptions import NERError, SuggestionError
from .schemas import Entity, SuggestItem
from .dtos import SuggestRequest
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

def detect_lang(text: str, lang_hint: str | None) -> str:
    if lang_hint in ("ko", "en"):
        return lang_hint
    try:
        lang = detect(text)
        if lang.startswith("ko"):
            return "ko"
        if lang.startswith("en"):
            return "en"
        return "unknown"
    except Exception:
        return "unknown"
    
def _best_relabel(canonical_en: str, en_entities: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """
    Find best matching English NER entity for canonical_en via string similarity.
    Returns: {"label":..., "confidence":...} or None
    """
    best = None
    best_score = 0.0
    for e in en_entities:
        txt = str(e.get("text", ""))
        score = similarity(canonical_en, txt)
        if score > best_score:
            best_score = score
            best = e

    # threshold: similarity gate + entity confidence gate
    if not best:
        return None
    if best_score < 0.75:
        return None

    return best

class NERService:
    def __init__(self) -> None:
        self.engine = NEREngine(min_token_len=settings.NER_MIN_TOKEN_LEN)

    async def run(self, text: str, lang_hint: str | None) -> Dict[str, Any]:
        errors: List[Dict[str, Any]] = []
        lang = detect_lang(text, lang_hint)

        # ---- Pass 1: span candidates on original text (label is just a hint) ----
        try:
            raw_entities = self.engine.extract(text)
        except Exception as e:
            raw_entities = []
            errors.append({"stage": "ner_pass1", "message": str(e)})

        raw_entities = raw_entities[: settings.NER_MAX_MENTIONS]

        base_mentions: List[Dict[str, Any]] = []
        for e in raw_entities:
            surface = (e.text or "").strip()
            if not surface:
                continue
            base_mentions.append(
                {
                    "surface": surface,
                    "span": {"start": int(e.start), "end": int(e.end)},
                    "ner": {"label": e.label, "confidence": clamp01(float(e.score))},
                }
            )
        # ---- Pass 2: GPT produces (normalized_text_en + canonical_en per mention) ----
        try:
            canon_out = await canonicalize_with_normalized_sentence(
                text=text,
                lang=lang,
                mentions=[{"surface": m["surface"], "span": m["span"]} for m in base_mentions],
            )
        except Exception as e:
            canon_out = {"normalized_text_en": "", "mentions": []}
            errors.append({"stage": "canonicalize", "message": str(e)})

        normalized_text_en = str(canon_out.get("normalized_text_en", "")).strip() or None

        canon_index: Dict[tuple[int, int, str], Dict[str, Any]] = {}
        for cm in canon_out.get("mentions", []):
            try:
                k = (int(cm["span"]["start"]), int(cm["span"]["end"]), str(cm["surface"]))
                canon_index[k] = cm
            except Exception:
                continue

        # ---- Pass 3: English re-labeling using GLiNER on normalized_text_en ----
        en_entities: List[Dict[str, Any]] = []
        if normalized_text_en:
            try:
                en_raw = self.engine.extract(normalized_text_en)
                for e in en_raw:
                    en_entities.append({"text": e.text, "label": e.label, "confidence": clamp01(float(e.score))})
            except Exception as e:
                errors.append({"stage": "ner_pass3_en", "message": str(e)})

        mentions: List[Dict[str, Any]] = []
        for m in base_mentions:
            k = (int(m["span"]["start"]), int(m["span"]["end"]), str(m["surface"]))
            cm = canon_index.get(k)

            canon_en = (cm.get("canonical_en") if cm else m["surface"]) or m["surface"]
            reason = (cm.get("reason") if cm else "unchanged") if cm else "unchanged"

            # default label from pass1
            final_label = m["ner"]["label"]
            final_conf = float(m["ner"]["confidence"])

            # try relabel using english NER if we have it
            if en_entities and canon_en:
                best = _best_relabel(str(canon_en), en_entities)
                if best:
                    # override only if relabel confidence is strong
                    if float(best.get("confidence", 0.0)) >= 0.55:
                        final_label = str(best["label"])
                        final_conf = max(final_conf, float(best.get("confidence", 0.0)))

            mentions.append(
                {
                    "surface": m["surface"],
                    "span": m["span"],
                    "ner": {"label": final_label, "confidence": clamp01(final_conf)},
                    "canonical": {"en": str(canon_en), "reason": str(reason)},
                }
            )

        return {
            "text": text,
            "lang": lang,
            "normalized_text_en": normalized_text_en,
            "mentions": mentions,
            "errors": errors,
        }

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
