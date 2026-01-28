from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


class SnapshotOClass(BaseModel):
    id: str
    labelKo: Optional[str] = None
    labelEn: Optional[str] = None
    facet: Optional[str] = None
    kind: Optional[str] = None
    isRoot: Optional[bool] = None
    description: Optional[str] = None
    seedVersion: Optional[str] = None


class SnapshotEdge(BaseModel):
    childId: str
    parentId: str


class Snapshot(BaseModel):
    oClasses: List[SnapshotOClass] = Field(default_factory=list)
    subclassEdges: List[SnapshotEdge] = Field(default_factory=list)
    seedVersion: Optional[str] = None


class ConceptSample(BaseModel):
    conceptKey: str
    conceptType: str
    conceptName: str
    sourceText: Optional[str] = None
    normalizedTextEn: Optional[str] = None
    surface: Optional[str] = None
    span: Optional[Dict[str, int]] = None
    occurrences: Optional[int] = None
    examples: Optional[List[str]] = None


class Classification(BaseModel):
    conceptKey: str
    conceptType: str
    conceptName: str
    oClassId: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: Optional[str] = None


class EventActivityClassification(BaseModel):
    eventId: Optional[str] = None
    oClassId: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: Optional[str] = None


class QueryTemplate(BaseModel):
    cqId: Optional[str] = None
    goal: Optional[str] = None
    querySketch: Optional[str] = None
    params: Optional[List[str]] = None
    outputSchema: Optional[List[Dict[str, Any]]] = None


class ProposeRequest(BaseModel):
    cqs: List[str] = Field(min_length=1)
    snapshot: Snapshot
    conceptSamples: Optional[List[ConceptSample]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cqs": [
                    "최근 30일 가장 많이 한 활동은?",
                    "최근 8주 운동 관련 활동 추세는?",
                ],
                "snapshot": {
                    "seedVersion": "0.1.1",
                    "oClasses": [
                        {"id": "PhysicalActivity", "labelKo": "운동", "facet": "Activity"},
                        {"id": "Work", "labelKo": "업무", "facet": "Activity"},
                    ],
                    "subclassEdges": [
                        {"childId": "Cardio", "parentId": "PhysicalActivity"}
                    ],
                },
            }
        }
    )


class ProposeResponse(BaseModel):
    ok: bool = True
    oClassesToAdd: List[SnapshotOClass] = Field(default_factory=list)
    subclassEdgesToAdd: List[SnapshotEdge] = Field(default_factory=list)
    classifications: List[Classification] = Field(default_factory=list)
    queryTemplates: List[QueryTemplate] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)


class ClassifyRequest(BaseModel):
    concepts: List[ConceptSample] = Field(default_factory=list)
    snapshot: Snapshot
    mode: Literal["existing_only", "allow_new_leaf"] = "existing_only"
    scope: Literal["concept", "event", "both"] = "both"
    eventId: Optional[str] = None
    eventTitle: Optional[str] = None
    eventNormalizedTextEn: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "concepts": [
                    {
                        "conceptKey": "Activity::climbing",
                        "conceptType": "Activity",
                        "conceptName": "climbing",
                        "occurrences": 3,
                    }
                ],
                "snapshot": {"seedVersion": "0.1.1", "oClasses": [], "subclassEdges": []},
                "mode": "existing_only",
                "scope": "both",
                "eventTitle": "Climbing with Alice",
            }
        }
    )


class ClassifyResponse(BaseModel):
    ok: bool = True
    classifications: List[Classification] = Field(default_factory=list)
    eventActivities: List[EventActivityClassification] = Field(default_factory=list)
    oClassesToAdd: List[SnapshotOClass] = Field(default_factory=list)
    subclassEdgesToAdd: List[SnapshotEdge] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
