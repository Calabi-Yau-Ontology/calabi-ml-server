from difflib import SequenceMatcher
from typing import Iterable, List, Tuple


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

def dedup_by_span(items: Iterable[dict]) -> List[dict]:
    best: dict[Tuple[int, int], dict] = {}
    for it in items:
        span = it.get("span") or {}
        s, e = int(span.get("start", 0)), int(span.get("end", 0))
        key = (s, e)
        score = float(it.get("score", it.get("confidence", 0.0)) or 0.0)
        if key not in best:
            best[key] = it
        else:
            prev = float(best[key].get("score", best[key].get("confidence", 0.0)) or 0.0)
            if score > prev:
                best[key] = it
    return sorted(
        best.values(),
        key=lambda x: (
            int((x.get("span") or {}).get("start", 0)),
            int((x.get("span") or {}).get("end", 0)),
        ),
    )

def norm_text(s: str) -> str:
    return " ".join((s or "").lower().strip().split())

def similarity(a: str, b: str) -> float:
    a2, b2 = norm_text(a), norm_text(b)
    if not a2 or not b2:
        return 0.0
    if a2 == b2:
        return 1.0
    # substring boost
    if a2 in b2 or b2 in a2:
        return 0.92
    return SequenceMatcher(None, a2, b2).ratio()
