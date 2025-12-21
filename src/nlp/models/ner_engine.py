from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.config import settings
from ..utils import dedup_by_span, clamp01
from ..schemas import NERLabel

try:
    from gliner import GLiNER  # type: ignore
except Exception:  # pragma: no cover
    GLiNER = None

@dataclass
class RawEntity:
    text: str
    start: int
    end: int
    label: str
    score: float


class NEREngine:
    def __init__(self, min_token_len: int) -> None:
        self.min_token_len = min_token_len
        self._gliner = None

        if GLiNER is not None:
            try:
                self._gliner = GLiNER.from_pretrained(settings.GLINER_MODEL)
            except Exception as e:
                print(f"Failed to load GLiNER model '{settings.GLINER_MODEL}': {e}")
                self._gliner = None

        # label set for GLiNER
        self._labels = list(NERLabel.__args__)

    def extract(self, text: str) -> List[RawEntity]:
        if self._gliner is not None:
            print("Using GLiNER for NER extraction", text)
            ents = self._extract_gliner(text)
            if ents:
                return ents
        print("Falling back to simple NER extraction", text, ents)
        return self._extract_fallback(text)

    def _extract_gliner(self, text: str) -> List[RawEntity]:
        preds = self._gliner.predict_entities(
            text,
            self._labels,
            threshold=float(settings.GLINER_THRESHOLD),
        )

        out: List[RawEntity] = []
        for p in preds:
            try:
                out.append(
                    RawEntity(
                        text=str(p["text"]),
                        start=int(p["start"]),
                        end=int(p["end"]),
                        label=str(p["label"]),
                        score=clamp01(float(p.get("score", 0.5))),
                    )
                )
            except Exception:
                continue

        # dedup by span (keep best score)
        as_dict = [
            {"span": {"start": e.start, "end": e.end}, "score": e.score, "_e": e} for e in out
        ]
        deduped = dedup_by_span(as_dict)
        return [d["_e"] for d in deduped]

    def _extract_fallback(self, text: str) -> List[RawEntity]:
        out: List[RawEntity] = []
        i, n = 0, len(text)

        while i < n:
            if text[i].isspace():
                i += 1
                continue

            j = i
            while j < n and not text[j].isspace():
                j += 1

            token = text[i:j].strip()
            if len(token) >= self.min_token_len:
                out.append(RawEntity(text=token, start=i, end=j, label="None", score=0.40))
            i = j

        # dedup
        as_dict = [{"span": {"start": e.start, "end": e.end}, "score": e.score, "_e": e} for e in out]
        deduped = dedup_by_span(as_dict)
        return [d["_e"] for d in deduped]
