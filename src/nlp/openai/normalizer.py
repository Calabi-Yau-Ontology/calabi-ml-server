import asyncio
from typing import Any, Dict, List, Literal, Optional, Tuple

from cachetools import LRUCache
from openai import OpenAI
from pydantic import BaseModel, Field

from src.config import settings
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


# ----------------------------
# Pydantic output schema (for responses.parse)
# ----------------------------

class Span(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class OutMention(BaseModel):
    surface: str
    span: Span
    canonical_en: str
    reason: Literal["abbr_expansion", "normalization", "unchanged", "unknown"]


class CanonicalizeOut(BaseModel):
    normalized_text_en: str
    mentions: List[OutMention]


# ----------------------------
# In-memory cache (per-process)
# key: (lang, surface) -> (canonical_en, reason)
# ----------------------------

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
    """
    Conservative fallback: keep as-is unless we have a small abbr map.
    """
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
    """
    Sync call using Responses API + parse() (returns Pydantic object).
    """
    client = _client_get()
    resp = client.responses.parse(
        model=settings.OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        text_format=CanonicalizeOut,
    )
    return resp.output_parsed  # CanonicalizeOut (Pydantic object)


async def canonicalize_with_normalized_sentence(
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
        "mentions": [{surface, span, canonical_en, reason}...]
      }

    Guarantees:
    - never throws (best-effort)
    - preserves input surface/span exactly in output
    - DOES NOT use json.loads (parse() returns typed object)
    """
    if not mentions:
        return {
            "normalized_text_en": text if lang == "en" else "",
            "mentions": [],
        }

    # Disabled => fallback only (still return normalized_text_en best-effort)
    if not _enabled():
        out: List[Dict[str, Any]] = []
        for m in mentions:
            surface = str(m.get("surface", "")).strip()
            span = m.get("span") or {"start": 0, "end": 0}
            canon, reason = _fallback(surface)
            _CANON_CACHE[(lang, surface)] = (canon, reason)
            out.append({"surface": surface, "span": span, "canonical_en": canon, "reason": reason})
        return {"normalized_text_en": text if lang == "en" else "", "mentions": out}

    # Build GPT input payload (includes full sentence context + mentions list)
    payload = {"text": text, "lang": lang, "mentions": mentions}
    user_prompt = USER_PROMPT_TEMPLATE.format(payload=payload)

    try:
        parsed: CanonicalizeOut = await asyncio.to_thread(_call_openai_sync, SYSTEM_PROMPT, user_prompt)

        normalized_text_en = (parsed.normalized_text_en or "").strip()

        # index by (start,end,surface) to preserve exact mapping
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
            else:
                canon = (om.canonical_en or "").strip() or surface
                reason = str(om.reason)

            _CANON_CACHE[(lang, surface)] = (canon, reason)
            out.append({"surface": surface, "span": span, "canonical_en": canon, "reason": reason})

        return {"normalized_text_en": normalized_text_en, "mentions": out}

    except Exception:
        # total failure => fallback
        out: List[Dict[str, Any]] = []
        for m in mentions:
            surface = str(m.get("surface", "")).strip()
            span = m.get("span") or {"start": 0, "end": 0}
            canon, reason = _fallback(surface)
            _CANON_CACHE[(lang, surface)] = (canon, reason)
            out.append({"surface": surface, "span": span, "canonical_en": canon, "reason": reason})
        return {"normalized_text_en": text if lang == "en" else "", "mentions": out}
