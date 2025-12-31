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
        for e in raw_entities:
            surface = (e.text or "").strip()
            if not surface:
                continue
            base_mentions.append(
                {
                    "surface": surface,
                    "span": {"start": int(e.start), "end": int(e.end)},
                    "ner": {"label": str(e.label), "confidence": clamp01(float(e.score))},
                }
            )
        # ---- Pass 2: GPT produces (normalized_text_en + canonical_en + anchor_en) ----
        start_time = time.time()
        print("Starting canonicalization at", start_time)
        try:
            canon_out = await canonicalize_with_anchors(
                text=text,
                lang=lang,
                mentions=[{"surface": m["surface"], "span": m["span"]} for m in base_mentions],
            )
        except Exception as e:
            canon_out = {"normalized_text_en": "", "mentions": []}
            errors.append({"stage": "canonicalize", "message": str(e)})
        end_time = time.time()
        print("Finished canonicalization at", end_time, "took", end_time - start_time, "seconds")
        print("Canonicalization output:", canon_out)
        normalized_text_en = str(canon_out.get("normalized_text_en", "")).strip() or None
        # mentions 정렬은 (start,end,surface) 키로 매칭
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

        mentions: List[Dict[str, Any]] = []
        seen_surface_label: set[tuple[str, str]] = set()
        for m in base_mentions:
            k = (int(m["span"]["start"]), int(m["span"]["end"]), str(m["surface"]))
            cm = canon_index.get(k)

            if cm:
                canon_en = (cm.get("canonical_en") or "").strip()
                if not canon_en:
                    continue  # canonical key missing -> skip mention
                reason = (cm.get("reason") or "unknown")
                anchor_en = (cm.get("anchor_en") or "").strip()
            else:
                canon_en = m["surface"]
                reason = "fallback"
                anchor_en = ""

            # default label from pass1
            base_label = str(m["ner"]["label"])
            base_conf = clamp01(float(m["ner"]["confidence"]))

            matched = None
            if normalized_text_en and en_entities and anchor_en:
                matched = _match_en_entity_by_anchor(
                    normalized_text_en=normalized_text_en,
                    en_entities=en_entities,
                    anchor_en=anchor_en,
                )

            # --- [변경] Fallback 시 개별 단어 재추론 로직 추가 ---
            if not matched and base_label == "None":
                print("No anchor match and base label is None, trying single-word re-inference for surface:", m["surface"], "canon_en:", canon_en)
                # anchor 매칭도 안 되고, 1차 NER도 None인 경우 -> 단어 단위로 다시 추론
                try:
                    # GLiNER는 문맥 없이 단어만 넣어도 추론 가능
                    # extract() 결과 리스트 중 가장 score 높은 것 채택
                    single_preds = self.engine.extract(canon_en)#m["surface"])
                    if single_preds:
                        # score순 정렬되어 있다고 가정하거나 max score 찾기
                        best_p = max(single_preds, key=lambda x: x.score)
                        # 임계값(0.3 등) 이상일 때만 덮어쓰기
                        if best_p.score > 0.3: 
                            base_label = str(best_p.label)
                            base_conf = clamp01(float(best_p.score))
                except Exception:
                    pass
            # ----------------------------------------------------

            final_label, final_conf = override_label(base_label, base_conf, matched)

            dedup_key = (m["surface"], final_label)
            if dedup_key in seen_surface_label:
                continue
            seen_surface_label.add(dedup_key)

            mentions.append(
                {
                    "surface": m["surface"],
                    "span": m["span"],
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
