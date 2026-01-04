# src/nlp/normalizer.py
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from cachetools import LRUCache
from openai import OpenAI

from src.config import settings
from src.nlp.dtos import CanonicalizeOut
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# cache: (lang, surface) -> (canonical_en, anchor_en, reason)
_CANON_CACHE: LRUCache = LRUCache(maxsize=2048)
_client: Optional[OpenAI] = None


def _enabled() -> bool:
    return bool(settings.CANONICALIZATION_ENABLED and settings.OPENROUTER_API_KEY)


def _client_get() -> OpenAI:
    global _client
    if _client is None:
        # _client = OpenAI(api_key=settings.OPENAI_API_KEY)
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
        )
    return _client


def _fallback(surface: str) -> Tuple[str, str]:
    s = (surface or "").strip()
    abbr_map = {
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
        # model=settings.OPENAI_MODEL,
        model=settings.OPENROUTER_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        text_format=CanonicalizeOut,
    )
    return resp.output_parsed


async def canonicalize_with_anchors(
    text: str,
    lang: str,
    mention_hints: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Calls OpenAI API to produce:
      {
        "normalized_text_en": str,
        "mentions": [{surface, canonical_en, anchor_en, reason}...]
      }

    mention_hints are optional span candidates used only for fallback when the API is disabled or fails.
    """
    text = text or ""
    mention_hints = mention_hints or []
    if not text.strip():
        print("In canonicalize_with_anchors: empty text")
        return {"normalized_text_en": "", "mentions": []}

    def _fallback_from_hints() -> Dict[str, Any]:
        out: List[Dict[str, Any]] = []
        for m in mention_hints:
            surface = str(m.get("surface", "")).strip()
            if not surface:
                continue
            canon, reason = _fallback(surface)
            _CANON_CACHE[(lang, surface)] = (canon, canon, reason)
            out.append(
                {
                    "surface": surface,
                    "canonical_en": canon,
                    "anchor_en": canon,
                    "reason": reason,
                }
            )
        normalized = text if lang == "en" else ""
        return {"normalized_text_en": normalized, "mentions": out}

    # disabled -> fallback only
    if not _enabled():
        print("In canonicalize_with_anchors: disabled, using fallback")
        return _fallback_from_hints()

    payload = {"text": text, "lang": lang}
    user_prompt = USER_PROMPT_TEMPLATE.format(payload=payload)
    try:
        parsed: CanonicalizeOut = await asyncio.to_thread(
            _call_openai_sync,
            SYSTEM_PROMPT,
            user_prompt,
        )
        print("In canonicalize_with_anchors: API response:", parsed)
        normalized = (parsed.normalized_text_en or "").strip()

        out: List[Dict[str, Any]] = []
        for om in parsed.mentions:
            surface = (om.surface or "").strip()
            if not surface:
                continue
            canon = (om.canonical_en or "").strip()
            anchor_en = (om.anchor_en or "").strip()
            reason = str(om.reason)

            if normalized and anchor_en and normalized.find(anchor_en) < 0:
                anchor_en = ""

            _CANON_CACHE[(lang, surface)] = (canon, anchor_en or canon, reason)

            out.append(
                {
                    "surface": surface,
                    "canonical_en": canon,
                    "anchor_en": anchor_en,
                    "reason": reason,
                }
            )

        return {"normalized_text_en": normalized, "mentions": out}

    except Exception as e:
        print("Exception in canonicalize_with_anchors:", e)
        return _fallback_from_hints()
