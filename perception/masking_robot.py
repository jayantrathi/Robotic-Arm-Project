from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from autostop import AutoStopRobot

Box = tuple[int, int, int, int]


class MaskingThreadSafeRobot(AutoStopRobot):
    def __init__(
        self,
        robot,
        *,
        detector,
        target: str,
        candidates: list[str],
        camera_key: str = "camera1",
        threshold: float = 0.15,
        detect_hz: float = 2.0,
        dilate: float = 0.25,
        hold_s: float = 1.2,
        fill_mode: str = "blur",
        blur_ksize: int = 61,
        feather_ksize: int = 41,
        protect_bottom: float = 0.35,
        protect_width: float = 0.60,
        initial_distractors: list[Box] | None = None,
        **autostop_cfg,
    ):
        super().__init__(robot, **autostop_cfg)
        self._detector = detector
        self._target = target
        self._candidates = candidates
        self._camera_key = camera_key
        self._threshold = threshold
        self._period = 1.0 / detect_hz
        self._dilate = dilate
        self._hold_s = hold_s
        self._fill_mode = fill_mode
        self._blur_ksize = blur_ksize | 1
        self._feather_ksize = feather_ksize | 1
        self._protect_bottom = protect_bottom
        self._protect_width = protect_width

        self._distractors: list[Box] = list(initial_distractors or [])
        self._ts = time.time() if initial_distractors else 0.0
        self._latest_frame: np.ndarray | None = None
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        frame = obs.get(self._camera_key)
        if frame is None:
            return obs
        with self._state_lock:
            self._latest_frame = frame.copy()
            boxes = list(self._distractors)
            ts = self._ts
        if boxes and (time.time() - ts) <= self._hold_s:
            obs[self._camera_key] = self._suppress(frame, boxes)
        return obs

    def _detect_loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            with self._state_lock:
                frame = self._latest_frame
            if frame is not None:
                try:
                    dets = self._detector.detect(frame, self._candidates, threshold=self._threshold)
                    best: dict[str, Box] = {}
                    for d in dets:
                        if d.label == self._target:
                            continue
                        best.setdefault(d.label, d.box)
                    with self._state_lock:
                        self._distractors = list(best.values())
                        self._ts = time.time()
                except Exception as e:
                    print(f"[masking_robot] detector error: {e}")
            sleep = self._period - (time.time() - t0)
            if sleep > 0:
                self._stop.wait(sleep)

    def _dilated_boxes(self, boxes: list[Box], w: int, h: int) -> list[Box]:
        out = []
        for (x1, y1, x2, y2) in boxes:
            bw, bh = x2 - x1, y2 - y1
            out.append((
                max(0, int(x1 - self._dilate * bw)),
                max(0, int(y1 - self._dilate * bh)),
                min(w, int(x2 + self._dilate * bw)),
                min(h, int(y2 + self._dilate * bh)),
            ))
        return out

    def _suppress(self, frame: np.ndarray, boxes: list[Box]) -> np.ndarray:
        import cv2

        h, w = frame.shape[:2]
        dboxes = self._dilated_boxes(boxes, w, h)
        py = int(h * (1 - self._protect_bottom))
        px1 = int(w * (0.5 - self._protect_width / 2))
        px2 = int(w * (0.5 + self._protect_width / 2))

        if self._fill_mode == "inpaint":
            mask8 = np.zeros((h, w), np.uint8)
            for (x1, y1, x2, y2) in dboxes:
                mask8[y1:y2, x1:x2] = 255
            mask8[py:h, px1:px2] = 0
            return cv2.inpaint(frame, mask8, 3, cv2.INPAINT_TELEA)

        alpha = np.zeros((h, w), np.float32)
        for (x1, y1, x2, y2) in dboxes:
            alpha[y1:y2, x1:x2] = 1.0
        alpha = cv2.GaussianBlur(alpha, (self._feather_ksize, self._feather_ksize), 0)
        alpha[py:h, px1:px2] = 0.0
        alpha = alpha[:, :, None]

        sub = np.zeros_like(frame) if self._fill_mode == "black" \
            else cv2.GaussianBlur(frame, (self._blur_ksize, self._blur_ksize), 0)
        out = frame.astype(np.float32) * (1 - alpha) + sub.astype(np.float32) * alpha
        return out.clip(0, 255).astype(frame.dtype)

    def stop(self) -> None:
        self._stop.set()


def make_masking_wrapper(**cfg):
    def factory(robot):
        return MaskingThreadSafeRobot(robot, **cfg)
    return factory
