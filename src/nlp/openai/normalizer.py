# src/nlp/normalizer.py
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Literal, Optional, Tuple

from cachetools import LRUCache
from openai import OpenAI
from pydantic import BaseModel, Field

from src.config import settings
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class Span(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class OutMention(BaseModel):
    surface: str
    span: Span

    canonical_en: str
    anchor_en: str

    anchor_span_en: Optional[Span] = None
    reason: Literal["abbr_expansion", "normalization", "unchanged", "unknown"]


class CanonicalizeOut(BaseModel):
    normalized_text_en: str
    mentions: List[OutMention]


# cache: (lang, surface) -> (canonical_en, anchor_en, reason)
_CANON_CACHE: LRUCache = LRUCache(maxsize=2048)
_client: Optional[OpenAI] = None


def _enabled() -> bool:
    return bool(settings.CANONICALIZATION_ENABLED and settings.OPENAI_API_KEY)


def _client_get() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def _fallback(surface: str) -> Tuple[str, str]:
    s = (surface or "").strip()
    abbr_map = {
        "클밍": "climbing",
        "볼더": "bouldering",
        "mtg": "meeting",
        "eod": "end of day",
        "proj": "project",
    }
    canon = abbr_map.get(s.lower(), abbr_map.get(s, s))
    reason = "abbr_expansion" if canon != s else "unchanged"
    return canon, reason


def _call_openai_sync(system_prompt: str, user_prompt: str) -> CanonicalizeOut:
    client = _client_get()
    resp = client.responses.parse(
        model=settings.OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        text_format=CanonicalizeOut,
    )
    return resp.output_parsed


def _safe_find_span(haystack: str, needle: str) -> Optional[Dict[str, int]]:
    if not haystack or not needle:
        return None
    idx = haystack.find(needle)
    if idx < 0:
        return None
    return {"start": idx, "end": idx + len(needle)}


async def canonicalize_with_anchors(
    text: str,
    lang: str,
    mentions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Input mentions:
      [{ "surface": str, "span": {"start": int, "end": int} }, ...]

    Output:
      {
        "normalized_text_en": str,
        "mentions": [{surface, span, canonical_en, anchor_en, anchor_span_en?, reason}...]
      }

    Guarantees:
    - never throws (best-effort)
    - preserves input surface/span exactly in output mentions
    """
    if not mentions:
        return {"normalized_text_en": "", "mentions": []}

    # disabled -> fallback only (no anchors guaranteed)
    if not _enabled():
        out: List[Dict[str, Any]] = []
        for m in mentions:
            surface = str(m.get("surface", "")).strip()
            span = m.get("span") or {"start": 0, "end": 0}
            canon, reason = _fallback(surface)
            out.append(
                {
                    "surface": surface,
                    "span": span,
                    "canonical_en": canon,
                    "anchor_en": canon,  # best-effort
                    "anchor_span_en": None,
                    "reason": reason,
                }
            )
        return {"normalized_text_en": text if lang == "en" else "", "mentions": out}

    payload = {"text": text, "lang": lang, "mentions": mentions}
    user_prompt = USER_PROMPT_TEMPLATE.format(payload=payload)
    try:
        parsed: CanonicalizeOut = await asyncio.to_thread(_call_openai_sync, SYSTEM_PROMPT, user_prompt)
        normalized = (parsed.normalized_text_en or "").strip()

        # map by (orig_start, orig_end, surface)
        idx: Dict[Tuple[int, int, str], OutMention] = {}
        for om in parsed.mentions:
            idx[(om.span.start, om.span.end, om.surface)] = om

        out: List[Dict[str, Any]] = []

        for m in mentions:
            surface = str(m.get("surface", "")).strip()
            span = m.get("span") or {"start": 0, "end": 0}
            key = (int(span.get("start", 0)), int(span.get("end", 0)), surface)

            om = idx.get(key)
            if om is None:
                canon, reason = _fallback(surface)
                anchor_en = canon
                anchor_span_en = _safe_find_span(normalized, anchor_en)
            else:
                canon = (om.canonical_en or "").strip() or surface
                anchor_en = (om.anchor_en or "").strip() or canon
                reason = str(om.reason)

                # if model didn't provide anchor_span, derive if possible
                if om.anchor_span_en:
                    anchor_span_en = {"start": om.anchor_span_en.start, "end": om.anchor_span_en.end}
                else:
                    anchor_span_en = _safe_find_span(normalized, anchor_en)

                # hard rule: anchor_en must exist in normalized. If not, fallback safely.
                if normalized and normalized.find(anchor_en) < 0:
                    anchor_en = canon
                    anchor_span_en = _safe_find_span(normalized, anchor_en)

            _CANON_CACHE[(lang, surface)] = (canon, anchor_en, reason)

            out.append(
                {
                    "surface": surface,
                    "span": span,
                    "canonical_en": canon,
                    "anchor_en": anchor_en,
                    "anchor_span_en": anchor_span_en,
                    "reason": reason,
                }
            )

        return {"normalized_text_en": normalized, "mentions": out}

    except Exception:
        # total failure -> fallback
        out: List[Dict[str, Any]] = []
        for m in mentions:
            surface = str(m.get("surface", "")).strip()
            span = m.get("span") or {"start": 0, "end": 0}
            canon, reason = _fallback(surface)
            out.append(
                {
                    "surface": surface,
                    "span": span,
                    "canonical_en": canon,
                    "anchor_en": canon,
                    "anchor_span_en": None,
                    "reason": reason,
                }
            )
        return {"normalized_text_en": text if lang == "en" else "", "mentions": out}
