from typing import Any, Dict, List, Optional, Tuple

from langdetect import detect

from .openai.normalizer import canonicalize_with_anchors
from .schemas import NER_LABELS


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


def _find_first_span(haystack: str, needle: str) -> Optional[Tuple[int, int]]:
    if not haystack or not needle:
        return None
    idx = haystack.find(needle)
    if idx < 0:
        return None
    return (idx, idx + len(needle))


class NERService:
    def __init__(self) -> None:
        pass

    async def run(self, text: str, lang_hint: str | None) -> Dict[str, Any]:
        import time

        full_start_time = time.time()
        print("NERService.run started at", full_start_time)

        errors: List[Dict[str, Any]] = []
        lang = detect_lang(text, lang_hint)

        # ---- OpenAI pass: normalization + labeling ----
        start_time = time.time()
        print("Starting canonicalization at", start_time)
        try:
            canon_out = await canonicalize_with_anchors(text=text, lang=lang)
        except Exception as e:
            canon_out = {"normalized_text_en": "", "mentions": []}
            errors.append({"stage": "canonicalize", "message": str(e)})
        end_time = time.time()
        print("Finished canonicalization at", end_time, "took", end_time - start_time, "seconds")

        normalized_text_en = str(canon_out.get("normalized_text_en", "")).strip() or None
        canon_mentions = canon_out.get("mentions", []) or []

        allowed_labels = set(NER_LABELS)
        mentions: List[Dict[str, Any]] = []
        seen_surface_label: set[tuple[str, str]] = set()

        for cm in canon_mentions:
            surface = (cm.get("surface") or "").strip()
            if not surface:
                continue
            canon_en = (cm.get("canonical_en") or "").strip()
            reason = str(cm.get("reason") or "unknown")
            label = (cm.get("label") or "").strip()
            if label not in allowed_labels:
                label = "None"

            anchor_en = (cm.get("anchor_en") or "").strip()
            if normalized_text_en and anchor_en and normalized_text_en.find(anchor_en) < 0:
                anchor_en = ""

            dedup_key = (surface, label)
            if dedup_key in seen_surface_label:
                continue
            seen_surface_label.add(dedup_key)

            span_tuple = _find_first_span(text, surface)
            if span_tuple is None:
                errors.append({"stage": "span_lookup", "surface": surface})
                span_tuple = (0, 0)

            mentions.append(
                {
                    "surface": surface,
                    "span": {"start": span_tuple[0], "end": span_tuple[1]},
                    "ner": {"label": label},
                    "canonical": {"en": canon_en, "reason": reason},
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
