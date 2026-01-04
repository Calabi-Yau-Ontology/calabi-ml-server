# src/nlp/normalizer.py
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from openai import OpenAI
from src.config import settings
from src.nlp.dtos import CanonicalizeOut
from src.nlp.schemas import NER_LABELS
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

_client: Optional[OpenAI] = None


def _enabled() -> bool:
    has_key = bool(settings.OPENAI_API_KEY or settings.OPENROUTER_API_KEY)
    return bool(settings.CANONICALIZATION_ENABLED and has_key)


def _client_get() -> OpenAI:
    global _client
    if _client is None:
        # _client = OpenAI(api_key=settings.OPENAI_API_KEY)
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
        )
    return _client


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
) -> Dict[str, Any]:
    """
    Calls OpenAI (via OpenRouter) to produce normalized_text_en and mention data.
    """
    text = text or ""
    if not text.strip():
        print("In canonicalize_with_anchors: empty text")
        return {"normalized_text_en": "", "mentions": []}

    if not _enabled():
        print("In canonicalize_with_anchors: disabled, returning empty output")
        return {"normalized_text_en": "", "mentions": []}

    payload = {"text": text, "lang": lang, "labels": list(NER_LABELS)}
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
            label = (om.label or "").strip()

            if normalized and anchor_en and normalized.find(anchor_en) < 0:
                anchor_en = ""

            out.append(
                {
                    "surface": surface,
                    "label": label,
                    "canonical_en": canon,
                    "anchor_en": anchor_en,
                    "reason": reason,
                }
            )

        return {"normalized_text_en": normalized, "mentions": out}

    except Exception as e:
        print("Exception in canonicalize_with_anchors:", e)
        return {"normalized_text_en": "", "mentions": []}
