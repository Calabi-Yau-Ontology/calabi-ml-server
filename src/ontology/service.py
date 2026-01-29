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
                ok=False,
                errors=[{"stage": "disabled", "message": "openrouter_key_missing"}],
            )

        roots = self._get_root_classes(payload)
        if not roots:
            return ClassifyResponse(
                ok=False,
                errors=[
                    {
                        "stage": "validation",
                        "message": "no root oclasses available in snapshot",
                    }
                ],
            )
        include_concept = payload.scope in ("concept", "both")
        include_event = payload.scope in ("event", "both")
        concept_roots = (
            [c for c in roots if getattr(c, "facet", None) == "Entity"]
            if include_concept
            else []
        )
        activity_roots = (
            [c for c in roots if getattr(c, "facet", None) == "Activity"]
            if include_event
            else []
        )
        event_title = (
            payload.eventTitle or self._infer_event_title(payload)
            if include_event
            else None
        )
        root_candidates = []
        if include_concept:
            root_candidates.extend(concept_roots)
        if include_event:
            root_candidates.extend(activity_roots)
        root_candidates = list({c.id: c for c in root_candidates}.values())

        root_payload = {
            "mode": payload.mode,
            "rootOClasses": [self._o_class_view(c) for c in root_candidates],
            "conceptRootOClasses": [self._o_class_view(c) for c in concept_roots],
            "eventRootOClasses": [self._o_class_view(c) for c in activity_roots],
            "concepts": [c.model_dump() for c in payload.concepts] if include_concept else [],
            "eventTitle": event_title,
            "eventNormalizedTextEn": payload.eventNormalizedTextEn if include_event else None,
            "scope": payload.scope,
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
                ok=False,
                errors=[{"stage": "llm_call", "message": str(e)}],
            )

        concept_root_ids = {c.id for c in concept_roots}
        root_warnings: List[Dict[str, Any]] = root_selection.errors or []
        concept_root_map: Dict[str, List[RootCandidate]] = {}
        if include_concept:
            for entry in root_selection.conceptRoots:
                concept_root_map[entry.conceptKey] = self._select_roots(
                    entry.roots, concept_root_ids
                )
        event_roots = (
            self._select_roots(root_selection.eventRoots, {c.id for c in activity_roots})
            if include_event
            else []
        )

        if include_concept:
            missing_concepts = [
                c.conceptKey
                for c in payload.concepts
                if c.conceptKey not in concept_root_map
            ]
            if missing_concepts:
                root_warnings.append(
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
            root_warnings,
            include_concept,
            include_event,
        )
        if not stage2_payload.get("concepts") and not stage2_payload.get("event"):
            root_warnings.append(
                {
                    "stage": "validation",
                    "message": "no candidates for stage2 classification",
                }
            )
            return ClassifyResponse(
                ok=True,
                errors=[],
                warnings=root_warnings,
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
                ok=False,
                errors=[{"stage": "llm_call", "message": str(e)}],
            )

        if parsed.errors is None:
            parsed.errors = []
        if parsed.warnings is None:
            parsed.warnings = []
        if root_warnings:
            parsed.warnings.extend(root_warnings)

        resolved_event_id = (
            payload.eventId
            or payload.eventTitle
            or self._infer_event_title(payload)
        )
        if parsed.eventActivities is not None:
            if not include_event:
                parsed.eventActivities = []
            elif not resolved_event_id:
                if parsed.eventActivities:
                    parsed.warnings.append(
                        {
                            "stage": "validation",
                            "message": "event context missing; ignoring eventActivities",
                        }
                    )
                parsed.eventActivities = []
            else:
                for item in parsed.eventActivities:
                    item.eventId = resolved_event_id

        if not include_concept and parsed.classifications is not None:
            parsed.classifications = []

        stage2_concepts = stage2_payload.get("concepts") or []
        root_subtrees = stage2_payload.get("rootSubtrees") or {}
        root_leaf_map = {
            root_id: {entry["id"] for entry in entries}
            for root_id, entries in root_subtrees.items()
        }
        concept_leaf_map: Dict[str, set[str]] = {}
        if include_concept:
            for concept in stage2_concepts:
                allowed = set()
                for root_id in concept.get("rootCandidates", []):
                    allowed |= root_leaf_map.get(root_id, set())
                concept_leaf_map[concept.get("conceptKey")] = allowed

        event_leaf_ids: set[str] = set()
        if include_event and stage2_payload.get("event"):
            for root_id in stage2_payload["event"].get("rootCandidates", []):
                event_leaf_ids |= root_leaf_map.get(root_id, set())

        if parsed.classifications:
            kept = []
            dropped = []
            for item in parsed.classifications:
                allowed = concept_leaf_map.get(item.conceptKey)
                if not allowed or item.oClassId not in allowed:
                    dropped.append(
                        {
                            "conceptKey": item.conceptKey,
                            "conceptType": item.conceptType,
                            "conceptName": item.conceptName,
                            "oClassId": item.oClassId,
                        }
                    )
                else:
                    kept.append(item)
            if dropped:
                parsed.classifications = kept
                parsed.warnings.append(
                    {
                        "stage": "validation",
                        "message": "leaf-only: stripped classifications not in candidate leaves",
                        "droppedCount": len(dropped),
                        "dropped": dropped[:50],
                    }
                )
        if include_concept and stage2_concepts and not parsed.classifications:
            parsed.warnings.append(
                {
                    "stage": "classification",
                    "message": "no concept classifications returned; left unclassified",
                    "count": len(stage2_concepts),
                }
            )

        if parsed.eventActivities:
            kept_events = []
            dropped_events = []
            for item in parsed.eventActivities:
                if event_leaf_ids and item.oClassId in event_leaf_ids:
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
                parsed.warnings.append(
                    {
                        "stage": "validation",
                        "message": "leaf-only: stripped event activities not in candidate leaves",
                        "droppedCount": len(dropped_events),
                        "dropped": dropped_events[:50],
                    }
                )
        if include_event and stage2_payload.get("event") and not parsed.eventActivities:
            parsed.warnings.append(
                {
                    "stage": "classification",
                    "message": "no event activities returned; left unclassified",
                }
            )

        if payload.mode == "existing_only":
            if parsed.oClassesToAdd or parsed.subclassEdgesToAdd:
                parsed.oClassesToAdd = []
                parsed.subclassEdgesToAdd = []
                parsed.warnings.append(
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
                    parsed.warnings.append(
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
                    parsed.warnings.append(
                        {
                            "stage": "validation",
                            "message": "existing_only mode: stripped event activities not in snapshot",
                            "droppedCount": len(dropped_events),
                            "dropped": dropped_events[:50],
                        }
                    )

        if parsed.errors:
            parsed.ok = False

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
        include_concept: bool,
        include_event: bool,
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
            return leaves

        concept_items = []
        if include_concept:
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
        if include_event and not event_roots:
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
            if not leaves:
                root_errors.append(
                    {
                        "stage": "validation",
                        "message": "no leaf candidates for root",
                        "rootId": root,
                    }
                )
                continue
            entries = []
            for leaf_id in leaves[:200]:
                o_class = class_index.get(leaf_id)
                if not o_class:
                    continue
                entries.append(self._o_class_view(o_class))
            subtree_candidates[root] = entries

        if include_event and event_roots:
            event_payload = {
                "title": payload.eventTitle or self._infer_event_title(payload),
                "normalizedTextEn": payload.eventNormalizedTextEn,
                "rootCandidates": [r.oClassId for r in event_roots],
            }

        return {
            "mode": payload.mode,
            "scope": payload.scope,
            "concepts": concept_items,
            "event": event_payload,
            "rootSubtrees": subtree_candidates,
        }


# module-level singleton
ontology_service = OntologyService()
