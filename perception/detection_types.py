from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class Selection:
    found: bool
    target: str
    box: tuple[int, int, int, int] | None = None
    score: float = 0.0
    center: tuple[int, int] | None = None
    ambiguous: bool = False
    all_detections: list[Detection] | None = None


class BaseDetector:
    def detect(self, image, labels: list[str], threshold: float = 0.10) -> list[Detection]:
        raise NotImplementedError

    def select(
        self,
        image,
        target: str,
        candidates: list[str] | None = None,
        threshold: float = 0.10,
        ambiguity_ratio: float = 0.80,
    ) -> Selection:
        labels = candidates or [target]
        if target not in labels:
            labels = [target, *labels]
        dets = self.detect(image, labels, threshold=threshold)
        target_hits = [d for d in dets if d.label == target]
        if not target_hits:
            return Selection(found=False, target=target, all_detections=dets)
        best = target_hits[0]
        ambiguous = len(target_hits) > 1 and target_hits[1].score >= best.score * ambiguity_ratio
        return Selection(
            found=True, target=target, box=best.box, score=best.score,
            center=best.center, ambiguous=ambiguous, all_detections=dets,
        )
