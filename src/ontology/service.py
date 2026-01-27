from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from openai import OpenAI
from src.config import settings

from .dtos import ProposeRequest, ProposeResponse, ClassifyRequest, ClassifyResponse
from .prompts import (
    PROPOSE_SYSTEM_PROMPT,
    PROPOSE_USER_PROMPT_TEMPLATE,
    CLASSIFY_SYSTEM_PROMPT,
    CLASSIFY_USER_PROMPT_TEMPLATE,
)

_client: Optional[OpenAI] = None


def _client_get() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
        )
    return _client


def _enabled() -> bool:
    return bool(settings.OPENROUTER_API_KEY)


def _call_openai_sync(system_prompt: str, user_prompt: str, response_model):
    client = _client_get()
    resp = client.responses.parse(
        model=settings.OPENROUTER_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        text_format=response_model,
    )
    return resp.output_parsed


class OntologyService:
    async def propose(self, payload: ProposeRequest) -> ProposeResponse:
        if not _enabled():
            return ProposeResponse(
                ok=False,
                errors=[{"stage": "disabled", "message": "openrouter_key_missing"}],
            )

        user_prompt = PROPOSE_USER_PROMPT_TEMPLATE.format(
            payload=json.dumps(payload.model_dump(), ensure_ascii=False)
        )
        try:
            parsed: ProposeResponse = await asyncio.to_thread(
                _call_openai_sync,
                PROPOSE_SYSTEM_PROMPT,
                user_prompt,
                ProposeResponse,
            )
            if parsed.errors:
                parsed.ok = False
            return parsed
        except Exception as e:
            return ProposeResponse(
                ok=False,
                errors=[{"stage": "llm_call", "message": str(e)}],
            )

    async def classify(self, payload: ClassifyRequest) -> ClassifyResponse:
        if not _enabled():
            return ClassifyResponse(
                ok=False,
                errors=[{"stage": "disabled", "message": "openrouter_key_missing"}],
            )

        user_prompt = CLASSIFY_USER_PROMPT_TEMPLATE.format(
            payload=json.dumps(payload.model_dump(), ensure_ascii=False)
        )
        try:
            parsed: ClassifyResponse = await asyncio.to_thread(
                _call_openai_sync,
                CLASSIFY_SYSTEM_PROMPT,
                user_prompt,
                ClassifyResponse,
            )
        except Exception as e:
            return ClassifyResponse(
                ok=False,
                errors=[{"stage": "llm_call", "message": str(e)}],
            )

        if payload.mode == "existing_only":
            if parsed.oClassesToAdd or parsed.subclassEdgesToAdd:
                parsed.oClassesToAdd = []
                parsed.subclassEdgesToAdd = []
                parsed.errors.append(
                    {
                        "stage": "validation",
                        "message": "existing_only mode: stripped class/edge additions",
                    }
                )

        if parsed.errors and payload.mode != "existing_only":
            parsed.ok = False

        return parsed


# module-level singleton
ontology_service = OntologyService()
