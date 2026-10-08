#!/usr/bin/env python3
"""
Open-vocabulary object detector for the voice-controlled arm.

Why this exists
---------------
The SmolVLA grasp policy is reliable when there is ONE object in the scene, but
it dithers when there are two, because it was only trained on single-object
scenes and never had to use the words to choose between them. This module adds
the missing "which one did you mean" step: given the camera frame and the object
name from the (typed or spoken) command, it finds that specific object in the
image and returns where it is.

It uses OWLv2 (Open-World Localization), a zero-shot open-vocabulary detector:
you hand it free-text labels at run time ("pen", "teddy bear", "red block") and
it localizes them without ever being trained on your objects. That means adding
a new object costs zero retraining -- you just say its name.

Typical use
-----------
    det = ObjectDetector()                       # loads once, reuse it
    hit = det.select(frame, "pen",               # what the user asked for
                     candidates=["pen", "teddy bear", "red block"])
    if hit.found:
        print(hit.box, hit.score)                # pixel box + confidence
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

# OWLv2 ensemble checkpoint: best accuracy of the base OWLv2 variants, still
# small enough to run on a laptop. Downloads once (~ 1.5 GB) then caches.
MODEL_ID = "google/owlv2-base-patch16-ensemble"


@dataclass
class Detection:
    """One detected object: its label, confidence, and pixel box."""
    label: str
    score: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class Selection:
    """The result of asking 'where is the <target>?' in a scene."""
    found: bool                      # is the target confidently present?
    target: str                      # the object name we searched for
    box: tuple[int, int, int, int] | None = None
    score: float = 0.0
    center: tuple[int, int] | None = None
    ambiguous: bool = False          # two+ equally-good matches for the target
    all_detections: list[Detection] | None = None  # everything seen, for debug


def _pick_device(device: str | None) -> str:
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ObjectDetector:
    """Zero-shot open-vocabulary detector wrapping OWLv2."""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.device = _pick_device(device)
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @staticmethod
    def _to_pil(image) -> Image.Image:
        """Accept a PIL image, a path, or an HWC numpy/OpenCV array (BGR or RGB)."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            # OpenCV hands us BGR; the policy/camera stack is RGB. Callers pass
            # RGB by convention, so we don't flip here -- see detect()'s note.
            # ascontiguousarray is essential: a BGR->RGB view (frame[:, :, ::-1])
            # has negative strides, which silently produces empty/garbage
            # detections when handed to the processor.
            return Image.fromarray(np.ascontiguousarray(image)).convert("RGB")
        raise TypeError(f"unsupported image type: {type(image)}")

    @torch.no_grad()
    def detect(self, image, labels: list[str], threshold: float = 0.10) -> list[Detection]:
        """Return every detection of any of `labels` above `threshold`.

        `image` may be a PIL image, a file path, or an RGB numpy array. If you
        are reading frames with OpenCV (BGR), convert first: frame[:, :, ::-1].
        """
        pil = self._to_pil(image)
        inputs = self.processor(text=[labels], images=pil, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # target_sizes is (height, width) so boxes come back in pixel coords.
        target_sizes = torch.tensor([pil.size[::-1]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        dets: list[Detection] = []
        for score, label_idx, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            x1, y1, x2, y2 = (int(round(v)) for v in box.tolist())
            dets.append(
                Detection(label=labels[int(label_idx)], score=float(score),
                          box=(x1, y1, x2, y2))
            )
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
        """Find the `target` object in the scene and say where it is.

        Passing the full `candidates` list (all objects that might be present)
        rather than just the target makes OWLv2 far more decisive: it scores the
        labels against each other, so a real pen beats a "that might be a pen"
        on some other object. `candidates` defaults to just [target].

        `ambiguous` is set when a second detection of the SAME target scores
        within `ambiguity_ratio` of the best one -- i.e. two pens in frame and
        we can't tell which you meant.
        """
        labels = candidates or [target]
        if target not in labels:
            labels = [target, *labels]
        dets = self.detect(image, labels, threshold=threshold)

        target_hits = [d for d in dets if d.label == target]
        if not target_hits:
            return Selection(found=False, target=target, all_detections=dets)

        best = target_hits[0]
        ambiguous = (
            len(target_hits) > 1
            and target_hits[1].score >= best.score * ambiguity_ratio
        )
        return Selection(
            found=True,
            target=target,
            box=best.box,
            score=best.score,
            center=best.center,
            ambiguous=ambiguous,
            all_detections=dets,
        )
