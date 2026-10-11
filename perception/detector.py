from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

MODEL_ID = "google/owlv2-base-patch16-ensemble"


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


def _pick_device(device: str | None) -> str:
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ObjectDetector:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.device = _pick_device(device)
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @staticmethod
    def _to_pil(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(np.ascontiguousarray(image)).convert("RGB")
        raise TypeError(f"unsupported image type: {type(image)}")

    @torch.no_grad()
    def detect(self, image, labels: list[str], threshold: float = 0.10) -> list[Detection]:
        pil = self._to_pil(image)
        inputs = self.processor(text=[labels], images=pil, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([pil.size[::-1]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold
        )[0]
        dets: list[Detection] = []
        for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = (int(round(v)) for v in box.tolist())
            dets.append(Detection(labels[int(label_idx)], float(score), (x1, y1, x2, y2)))
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets

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
