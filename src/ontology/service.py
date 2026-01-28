from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, List

from openai import OpenAI
from pydantic import BaseModel, Field
from src.config import settings

from .dtos import ProposeRequest, ProposeResponse, ClassifyRequest, ClassifyResponse
from .prompts import (
    PROPOSE_SYSTEM_PROMPT,
    PROPOSE_USER_PROMPT_TEMPLATE,
    CLASSIFY_SYSTEM_PROMPT,
    CLASSIFY_USER_PROMPT_TEMPLATE,
    CLASSIFY_ROOT_SYSTEM_PROMPT,
    CLASSIFY_ROOT_USER_PROMPT_TEMPLATE,
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


def _call_openai_sync(system_prompt: str, user_prompt: str, response_model, model: Optional[str] = None):
    client = _client_get()
    resp = client.responses.parse(
        model=model or settings.OPENROUTER_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        text_format=response_model,
    )
    return resp.output_parsed


def _lite_model() -> str:
    return settings.OPENROUTER_MODEL_LITE or settings.OPENROUTER_MODEL


class RootCandidate(BaseModel):
    oClassId: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: Optional[str] = None


class RootSelection(BaseModel):
    conceptKey: str
    roots: List[RootCandidate] = Field(default_factory=list)


class RootSelectionResponse(BaseModel):
    conceptRoots: List[RootSelection] = Field(default_factory=list)
    eventRoots: List[RootCandidate] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)


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
                ok=(payload.mode == "existing_only"),
                errors=[{"stage": "disabled", "message": "openrouter_key_missing"}],
            )

        roots = self._get_root_classes(payload)
        if not roots:
            return ClassifyResponse(
                ok=(payload.mode == "existing_only"),
                errors=[
                    {
                        "stage": "validation",
                        "message": "no root oclasses available in snapshot",
                    }
                ],
            )
        activity_roots = [c for c in roots if getattr(c, "facet", None) == "Activity"]
        root_payload = {
            "mode": payload.mode,
            "rootOClasses": [self._o_class_view(c) for c in roots],
            "conceptRootOClasses": [self._o_class_view(c) for c in roots],
            "eventRootOClasses": [self._o_class_view(c) for c in activity_roots],
            "concepts": [c.model_dump() for c in payload.concepts],
            "eventTitle": payload.eventTitle or self._infer_event_title(payload),
            "eventNormalizedTextEn": payload.eventNormalizedTextEn,
        }
        root_prompt = CLASSIFY_ROOT_USER_PROMPT_TEMPLATE.format(
            payload=json.dumps(root_payload, ensure_ascii=False)
        )
        try:
            root_selection: RootSelectionResponse = await asyncio.to_thread(
                _call_openai_sync,
                CLASSIFY_ROOT_SYSTEM_PROMPT,
                root_prompt,
                RootSelectionResponse,
                _lite_model(),
            )
        except Exception as e:
            return ClassifyResponse(
                ok=(payload.mode == "existing_only"),
                errors=[{"stage": "llm_call", "message": str(e)}],
            )

        allowed_root_ids = {c.id for c in roots}
        root_errors: List[Dict[str, Any]] = root_selection.errors or []
        concept_root_map: Dict[str, List[RootCandidate]] = {}
        for entry in root_selection.conceptRoots:
            concept_root_map[entry.conceptKey] = self._select_roots(
                entry.roots, allowed_root_ids
            )
        event_roots = self._select_roots(
            root_selection.eventRoots, {c.id for c in activity_roots}
        )

        missing_concepts = [
            c.conceptKey for c in payload.concepts if c.conceptKey not in concept_root_map
        ]
        if missing_concepts:
            root_errors.append(
                {
                    "stage": "validation",
                    "message": "root selection missing for some concepts",
                    "missingCount": len(missing_concepts),
                    "missing": missing_concepts[:50],
                }
            )

        stage2_payload = self._build_stage2_payload(
            payload,
            concept_root_map,
            event_roots,
            roots,
            activity_roots,
            root_errors,
        )
        if not stage2_payload.get("concepts") and not stage2_payload.get("event"):
            root_errors.append(
                {
                    "stage": "validation",
                    "message": "no candidates for stage2 classification",
                }
            )
            return ClassifyResponse(
                ok=(payload.mode == "existing_only"),
                errors=root_errors,
            )
        stage2_prompt = CLASSIFY_USER_PROMPT_TEMPLATE.format(
            payload=json.dumps(stage2_payload, ensure_ascii=False)
        )
        try:
            parsed: ClassifyResponse = await asyncio.to_thread(
                _call_openai_sync,
                CLASSIFY_SYSTEM_PROMPT,
                stage2_prompt,
                ClassifyResponse,
            )
        except Exception as e:
            return ClassifyResponse(
                ok=(payload.mode == "existing_only"),
                errors=[{"stage": "llm_call", "message": str(e)}],
            )

        if parsed.errors is None:
            parsed.errors = []
        if root_errors:
            parsed.errors.extend(root_errors)

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

            allowed_ids = {c.id for c in payload.snapshot.oClasses if c.id}
            if parsed.classifications:
                kept = []
                dropped = []
                for item in parsed.classifications:
                    if item.oClassId in allowed_ids:
                        kept.append(item)
                    else:
                        dropped.append(
                            {
                                "conceptKey": item.conceptKey,
                                "conceptType": item.conceptType,
                                "conceptName": item.conceptName,
                                "oClassId": item.oClassId,
                            }
                        )
                if dropped:
                    parsed.classifications = kept
                    parsed.errors.append(
                        {
                            "stage": "validation",
                            "message": "existing_only mode: stripped classifications not in snapshot",
                            "droppedCount": len(dropped),
                            "dropped": dropped[:50],
                        }
                    )
            if parsed.eventActivities:
                kept_events = []
                dropped_events = []
                for item in parsed.eventActivities:
                    if item.eventId is None:
                        item.eventId = (
                            payload.eventId
                            or payload.eventTitle
                            or self._infer_event_title(payload)
                        )
                    if item.oClassId in allowed_ids:
                        kept_events.append(item)
                    else:
                        dropped_events.append(
                            {
                                "eventId": item.eventId,
                                "oClassId": item.oClassId,
                            }
                        )
                if dropped_events:
                    parsed.eventActivities = kept_events
                    parsed.errors.append(
                        {
                            "stage": "validation",
                            "message": "existing_only mode: stripped event activities not in snapshot",
                            "droppedCount": len(dropped_events),
                            "dropped": dropped_events[:50],
                        }
                    )

        if parsed.errors and payload.mode != "existing_only":
            parsed.ok = False
        if payload.mode == "existing_only":
            parsed.ok = True

        return parsed

    def _infer_event_title(self, payload: ClassifyRequest) -> Optional[str]:
        for concept in payload.concepts:
            if concept.sourceText:
                return concept.sourceText
        return None

    def _get_root_classes(self, payload: ClassifyRequest):
        roots = [c for c in payload.snapshot.oClasses if getattr(c, "isRoot", False)]
        if roots:
            return roots
        child_ids = {e.childId for e in payload.snapshot.subclassEdges}
        return [c for c in payload.snapshot.oClasses if c.id not in child_ids]

    def _o_class_view(self, o_class):
        return {
            "id": o_class.id,
            "labelKo": o_class.labelKo,
            "labelEn": o_class.labelEn,
            "facet": o_class.facet,
            "description": o_class.description,
        }

    def _select_roots(
        self, candidates: List[RootCandidate], allowed_ids: set[str]
    ) -> List[RootCandidate]:
        valid = [c for c in candidates if c.oClassId in allowed_ids]
        if not valid:
            return []
        selected = [c for c in valid if (c.confidence or 0) >= 0.4]
        selected = sorted(selected or valid, key=lambda c: c.confidence or 0, reverse=True)[
            :5
        ]
        return selected

    def _build_stage2_payload(
        self,
        payload: ClassifyRequest,
        concept_root_map: Dict[str, List[RootCandidate]],
        event_roots: List[RootCandidate],
        all_roots: List[Any],
        activity_roots: List[Any],
        root_errors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        children_by_parent: Dict[str, List[str]] = {}
        for edge in payload.snapshot.subclassEdges:
            children_by_parent.setdefault(edge.parentId, []).append(edge.childId)
        class_index = {c.id: c for c in payload.snapshot.oClasses}

        def collect_leaves(root_id: str) -> List[str]:
            visited = set()
            stack = [root_id]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                for child in children_by_parent.get(node, []):
                    stack.append(child)
            leaves = [n for n in visited if not children_by_parent.get(n)]
            return leaves or [root_id]

        concept_items = []
        for concept in payload.concepts:
            roots = concept_root_map.get(concept.conceptKey, [])
            if not roots:
                root_errors.append(
                    {
                        "stage": "validation",
                        "message": "no roots selected for concept",
                        "conceptKey": concept.conceptKey,
                    }
                )
                continue
            concept_items.append(
                {
                    **concept.model_dump(),
                    "rootCandidates": [r.oClassId for r in roots],
                }
            )

        event_payload = None
        if not event_roots:
            if not (payload.eventTitle or payload.eventNormalizedTextEn):
                root_errors.append(
                    {
                        "stage": "validation",
                        "message": "event context missing; event classification skipped",
                    }
                )
            elif not activity_roots:
                root_errors.append(
                    {
                        "stage": "validation",
                        "message": "no Activity roots in snapshot; event classification skipped",
                    }
                )
            else:
                root_errors.append(
                    {
                        "stage": "validation",
                        "message": "event root selection empty; event classification skipped",
                    }
                )

        subtree_candidates: Dict[str, List[Dict[str, Any]]] = {}
        roots_for_subtrees = {c.oClassId for c in event_roots} | {
            r.oClassId for roots in concept_root_map.values() for r in roots
        } | {root_id for item in concept_items for root_id in item["rootCandidates"]}
        for root in roots_for_subtrees:
            leaves = collect_leaves(root)
            entries = []
            for leaf_id in leaves[:200]:
                o_class = class_index.get(leaf_id)
                if not o_class:
                    continue
                entries.append(self._o_class_view(o_class))
            subtree_candidates[root] = entries

        if event_roots:
            event_payload = {
                "title": payload.eventTitle or self._infer_event_title(payload),
                "normalizedTextEn": payload.eventNormalizedTextEn,
                "rootCandidates": [r.oClassId for r in event_roots],
            }

        return {
            "mode": payload.mode,
            "concepts": concept_items,
            "event": event_payload,
            "rootSubtrees": subtree_candidates,
        }


# module-level singleton
ontology_service = OntologyService()
