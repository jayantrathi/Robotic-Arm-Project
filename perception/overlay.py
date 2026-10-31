from __future__ import annotations

import numpy as np

TARGET_COLOR = (40, 230, 70)
OTHER_COLOR = (235, 60, 60)


def annotate(frame_rgb, target=None, others=None):
    import cv2

    img = np.ascontiguousarray(frame_rgb).copy()
    for box, label in (others or []):
        _draw(cv2, img, box, OTHER_COLOR, label)
    if target is not None:
        box, label = target
        _draw(cv2, img, box, TARGET_COLOR, label)
    return img


def _draw(cv2, img, box, color, label):
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def log_overlay(annotated_rgb, entity="detector/view"):
    try:
        import rerun as rr
        rr.log(entity, rr.Image(annotated_rgb))
    except Exception:
        pass
