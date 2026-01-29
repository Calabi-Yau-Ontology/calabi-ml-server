import os
import sys
from pathlib import Path

os.environ.setdefault("OPENROUTER_API_KEY", "test")

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.ontology import service  # noqa: E402
from src.ontology.dtos import (  # noqa: E402
    ClassifyRequest,
    Snapshot,
    SnapshotOClass,
    SnapshotEdge,
    ConceptSample,
    ClassifyResponse,
)


def _fake_call(system_prompt, user_prompt, response_model, model=None):
    if response_model is service.RootSelectionResponse:
        return service.RootSelectionResponse(
            conceptRoots=[
                service.RootSelection(
                    conceptKey="Place::교동짬뽕",
                    roots=[service.RootCandidate(oClassId="E_Place", confidence=0.9)],
                )
            ],
            eventRoots=[service.RootCandidate(oClassId="A_Activity", confidence=0.95)],
            errors=[],
        )
    if response_model is ClassifyResponse:
        return ClassifyResponse(
            ok=True,
            classifications=[
                {
                    "conceptKey": "Place::교동짬뽕",
                    "conceptType": "Place",
                    "conceptName": "교동짬뽕",
                    "oClassId": "E_Place",
                    "confidence": 0.8,
                }
            ],
            eventActivities=[
                {"eventId": "wrong-id", "oClassId": "A_Climbing", "confidence": 0.9},
                {"eventId": "wrong-id", "oClassId": "A_Activity", "confidence": 0.6},
            ],
            errors=[],
            warnings=[],
        )
    raise RuntimeError(f"Unhandled response model: {response_model}")


def main() -> None:
    service._call_openai_sync = _fake_call

    snapshot = Snapshot(
        oClasses=[
            SnapshotOClass(id="E_Place", facet="Entity", isRoot=True),
            SnapshotOClass(id="E_Restaurant", facet="Entity"),
            SnapshotOClass(id="A_Activity", facet="Activity", isRoot=True),
            SnapshotOClass(id="A_Climbing", facet="Activity"),
        ],
        subclassEdges=[
            SnapshotEdge(parentId="E_Place", childId="E_Restaurant"),
            SnapshotEdge(parentId="A_Activity", childId="A_Climbing"),
        ],
        seedVersion="test",
    )

    req = ClassifyRequest(
        concepts=[
            ConceptSample(
                conceptKey="Place::교동짬뽕",
                conceptType="Place",
                conceptName="교동짬뽕",
                sourceText="교동짬뽕에서 점심",
            )
        ],
        snapshot=snapshot,
        mode="existing_only",
        scope="both",
        eventId="evt-1",
        eventTitle="Climbing with Alice",
    )

    result = service.ontology_service.classify(req)
    if hasattr(result, "__await__"):
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(result)

    print("ok:", result.ok)
    print("classifications:", result.classifications)
    print("eventActivities:", result.eventActivities)
    print("warnings:", result.warnings)
    print("errors:", result.errors)


if __name__ == "__main__":
    main()
