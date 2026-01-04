from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from langdetect import detect

from src.config import settings
from .models.ner_engine import NEREngine
from .openai.normalizer import canonicalize_with_anchors
from .utils import clamp01

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


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _span_len(s: Tuple[int, int]) -> int:
    return max(0, s[1] - s[0])


def _find_first_span(haystack: str, needle: str) -> Optional[Tuple[int, int]]:
    if not haystack or not needle:
        return None
    idx = haystack.find(needle)
    if idx < 0:
        return None
    return (idx, idx + len(needle))


def _match_en_entity_by_anchor(
    normalized_text_en: str,
    en_entities: List[Dict[str, Any]],
    anchor_en: str,
) -> Optional[Dict[str, Any]]:
    """
    Anchor-first matching
    Returns best en entity dict or None.
    """
    if not normalized_text_en or not en_entities:
        return None

    # substring-based
    if anchor_en:
        span = _find_first_span(normalized_text_en, anchor_en)
        if span:
            a = span
            best = None
            best_score = 0.0
            for e in en_entities:
                try:
                    es = (int(e["start"]), int(e["end"]))
                except Exception:
                    continue
                ov = _overlap(a, es)
                if ov <= 0:
                    continue
                coverage = ov / max(1, _span_len(a))
                conf = float(e.get("confidence", 0.0))
                score = 0.7 * coverage + 0.3 * conf
                if score > best_score:
                    best_score = score
                    best = e
            if best:
                return best

    return None


def override_label(
    base_label: str,
    base_conf: float,
    matched_en: Optional[Dict[str, Any]],
) -> Tuple[str, float]:
    """
    Override rule:
    - if anchor match exists, trust its label regardless of confidence
    - keep confidence as the better of base vs anchor match to avoid regressions
    """
    if not matched_en:
        return base_label, base_conf

    en_label = str(matched_en.get("label", base_label))
    en_conf = clamp01(float(matched_en.get("confidence", 0.0)))
    return en_label, max(base_conf, en_conf)


class NERService:
    def __init__(self) -> None:
        self.engine = NEREngine(min_token_len=settings.NER_MIN_TOKEN_LEN)

    async def run(self, text: str, lang_hint: str | None) -> Dict[str, Any]:
        import time

        full_start_time = time.time()
        print("NERService.run started at", full_start_time)
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
        surface_hint_map: Dict[str, deque[Dict[str, Any]]] = {}
        for e in raw_entities:
            surface = (e.text or "").strip()
            if not surface:
                continue
            mention = {
                "surface": surface,
                "span": {"start": int(e.start), "end": int(e.end)},
                "ner": {"label": str(e.label), "confidence": clamp01(float(e.score))},
            }
            base_mentions.append(mention)
            dq = surface_hint_map.setdefault(surface, deque())
            dq.append(mention)
        # ---- Pass 2: GPT produces (normalized_text_en + canonical_en + anchor_en) ----
        start_time = time.time()
        print("Starting canonicalization at", start_time)
        mention_hints = [{"surface": m["surface"], "span": m["span"]} for m in base_mentions]
        try:
            canon_out = await canonicalize_with_anchors(
                text=text,
                lang=lang,
                mention_hints=mention_hints,
            )
        except Exception as e:
            canon_out = {"normalized_text_en": "", "mentions": []}
            errors.append({"stage": "canonicalize", "message": str(e)})
        end_time = time.time()
        print("Finished canonicalization at", end_time, "took", end_time - start_time, "seconds")
        normalized_text_en = str(canon_out.get("normalized_text_en", "")).strip() or None
        canon_mentions = canon_out.get("mentions", []) or []
        
        # ---- Pass 3: English re-labeling using GLiNER on normalized_text_en ----
        en_entities: List[Dict[str, Any]] = []
        if normalized_text_en:
            try:
                en_raw = self.engine.extract(normalized_text_en)
                for e in en_raw:
                    en_entities.append(
                        {
                            "text": (e.text or ""),
                            "start": int(e.start),
                            "end": int(e.end),
                            "label": str(e.label),
                            "confidence": clamp01(float(e.score)),
                        }
                    )
            except Exception as e:
                errors.append({"stage": "ner_pass3_en", "message": str(e)})

        def _pop_surface_hint(surface: str) -> Optional[Dict[str, Any]]:
            dq = surface_hint_map.get(surface)
            if not dq:
                return None
            try:
                return dq.popleft()
            except IndexError:
                return None

        mentions: List[Dict[str, Any]] = []
        seen_surface_label: set[tuple[str, str]] = set()
        for cm in canon_mentions:
            surface = (cm.get("surface") or "").strip()
            if not surface:
                continue
            canon_en = (cm.get("canonical_en") or "").strip()
            if not canon_en:
                continue
            reason = (cm.get("reason") or "unknown")
            anchor_en = (cm.get("anchor_en") or "").strip()

            hint_cache: Optional[Dict[str, Any]] = None

            def _ensure_hint() -> Optional[Dict[str, Any]]:
                nonlocal hint_cache
                if hint_cache is None:
                    hint_cache = _pop_surface_hint(surface)
                return hint_cache

            hint = _ensure_hint()
            if hint:
                base_label = str(hint["ner"]["label"])
                base_conf = clamp01(float(hint["ner"]["confidence"]))
            else:
                base_label = "None"
                base_conf = 0.0

            matched = None
            if normalized_text_en and en_entities and anchor_en:
                matched = _match_en_entity_by_anchor(
                    normalized_text_en=normalized_text_en,
                    en_entities=en_entities,
                    anchor_en=anchor_en,
                )

            if not matched and base_label == "None":
                print(
                    "No anchor match and base label is None, trying single-word re-inference for surface:",
                    surface,
                    "canon_en:",
                    canon_en,
                )
                try:
                    single_preds = self.engine.extract(canon_en)
                    if single_preds:
                        best_p = max(single_preds, key=lambda x: x.score)
                        if best_p.score > 0.3:
                            base_label = str(best_p.label)
                            base_conf = clamp01(float(best_p.score))
                except Exception:
                    pass

            final_label, final_conf = override_label(base_label, base_conf, matched)

            dedup_key = (surface, final_label)
            if dedup_key in seen_surface_label:
                continue
            seen_surface_label.add(dedup_key)

            span_tuple = _find_first_span(text, surface)
            if span_tuple is None:
                hint = _ensure_hint()
                if hint:
                    try:
                        span_tuple = (
                            int(hint["span"]["start"]),
                            int(hint["span"]["end"]),
                        )
                    except Exception:
                        span_tuple = None
            if span_tuple is None:
                errors.append({"stage": "span_lookup", "surface": surface})
                span_tuple = (0, 0)

            mentions.append(
                {
                    "surface": surface,
                    "span": {"start": span_tuple[0], "end": span_tuple[1]},
                    "ner": {"label": final_label, "confidence": clamp01(final_conf)},
                    "canonical": {"en": str(canon_en), "reason": str(reason)},
                }
            )

        full_end_time = time.time()
        print("NERService.run finished at", full_end_time, "took", full_end_time - full_start_time, "seconds")
        return {
            "text": text,
            "lang": lang,
            "normalized_text_en": normalized_text_en,
            "mentions": mentions,
            "errors": errors,
        }

# module-level singletons
ner_service = NERService()
