#!/usr/bin/env python3
"""
A drop-in replacement for LeRobot's ThreadSafeRobot that hides *distractor*
objects from the camera feed the policy sees, leaving the target and everything
else untouched.

Why this shape
--------------
The SmolVLA policy is reliable with ONE object in frame but dithers with two.
An earlier version kept a window around the target and blurred everything else;
that globally altered the frame and hurt the grasp even when nothing needed
disambiguating. This version does the opposite and minimal thing: it blurs ONLY
the boxes of the *other* objects (the distractors), pixel-for-pixel leaving the
target and the rest of the scene exactly as the camera saw it.

Consequences that matter:
  - Pen alone (or once the bear leaves the wrist view mid-reach): nothing is
    detected to suppress, so the frame is returned UNCHANGED -> identical to the
    raw baseline that grasps well.
  - Two objects: the bear is blurred into the background so the policy commits
    to the pen, while the pen stays sharp for a precise grasp.

Real-time
---------
OWLv2 runs ~1.4 Hz on CPU (measured), in a background thread, so it never fights
the policy for the MPS GPU. ``get_observation`` only does a cheap numpy/cv2
blur with the most recent distractor boxes. Boxes are held for ``hold_s`` to
bridge the gaps between detections, and cleared as soon as a detection finds no
distractor -- so suppression stops the moment the bear leaves frame.

Injected by monkeypatching ``lerobot.rollout.context.ThreadSafeRobot``.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from autostop import AutoStopRobot

Box = tuple[int, int, int, int]


class MaskingThreadSafeRobot(AutoStopRobot):
    """ThreadSafeRobot that blurs out everything the policy was NOT asked for.

    Inherits return-to-home auto-stop from AutoStopRobot, so a masked run also
    ends itself after one pick-and-place.
    """

    def __init__(
        self,
        robot,
        *,
        detector,
        target: str,                 # detector label of the object we WANT (kept sharp)
        candidates: list[str],       # all object labels that might be present
        camera_key: str = "camera1",
        threshold: float = 0.15,
        detect_hz: float = 2.0,      # background detection rate (CPU OWLv2 ~1.4 Hz)
        dilate: float = 0.25,        # grow each distractor box so it's fully covered
        hold_s: float = 1.2,         # keep suppressing with the last boxes this long
        fill_mode: str = "blur",     # how to remove each distractor:
                                     #   "inpaint" = paint it out with surrounding table
                                     #               texture (cleanest; looks single-object)
                                     #   "blur"    = blur it into a soft blob
                                     #   "black"   = black its box out
        blur_ksize: int = 61,        # how hard to blur the distractor region
        feather_ksize: int = 41,     # soften the edge so there's no hard rectangle
        protect_bottom: float = 0.35,  # never suppress this fraction of the frame
        protect_width: float = 0.60,   # (bottom-centre) -- it's where the gripper is
        initial_distractors: list[Box] | None = None,  # seed from the gate
        **autostop_cfg,              # autostop/leave/ret/... forwarded to AutoStopRobot
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
        self._blur_ksize = blur_ksize | 1       # must be odd for cv2
        self._feather_ksize = feather_ksize | 1
        self._protect_bottom = protect_bottom
        self._protect_width = protect_width

        # Seed with the gate's distractor boxes so suppression is live from the
        # first frame -- no window where the raw two-object scene reaches the
        # policy and it lunges at the bear.
        self._distractors: list[Box] = list(initial_distractors or [])
        self._ts = time.time() if initial_distractors else 0.0
        self._latest_frame: np.ndarray | None = None
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()

    # -- the only overridden behaviour ---------------------------------------

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        frame = obs.get(self._camera_key)
        if frame is None:
            return obs
        with self._state_lock:
            self._latest_frame = frame.copy()     # for the detector thread
            boxes = list(self._distractors)
            ts = self._ts
        if boxes and (time.time() - ts) <= self._hold_s:
            obs[self._camera_key] = self._suppress(frame, boxes)
        return obs

    # -- background detection ------------------------------------------------

    def _detect_loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            with self._state_lock:
                frame = self._latest_frame
            if frame is not None:
                try:
                    dets = self._detector.detect(
                        frame, self._candidates, threshold=self._threshold
                    )
                    # best (highest-score) box per NON-target label == distractors.
                    # dets are score-sorted, so the first per label is its best.
                    best: dict[str, Box] = {}
                    for d in dets:
                        if d.label == self._target:
                            continue
                        best.setdefault(d.label, d.box)
                    with self._state_lock:
                        self._distractors = list(best.values())
                        self._ts = time.time()
                except Exception as e:  # never let the detector kill the run
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

        # The gripper sits at the bottom-centre of an eye-in-hand frame. Never
        # suppress there: blurring/erasing a distractor that strays near the
        # gripper would blind the policy to its own hand and wreck the approach.
        py = int(h * (1 - self._protect_bottom))
        px1 = int(w * (0.5 - self._protect_width / 2))
        px2 = int(w * (0.5 + self._protect_width / 2))

        # "inpaint": fill the distractor box with surrounding table texture so the
        # scene looks like the clean, single-object scene the policy trained on --
        # no smudge or blob left behind to perturb a delicate approach.
        if self._fill_mode == "inpaint":
            mask8 = np.zeros((h, w), np.uint8)
            for (x1, y1, x2, y2) in dboxes:
                mask8[y1:y2, x1:x2] = 255
            mask8[py:h, px1:px2] = 0          # protect the gripper zone
            return cv2.inpaint(frame, mask8, 3, cv2.INPAINT_TELEA)

        # "blur"/"black": composite a substitute over the distractor box, with a
        # feathered edge so the policy doesn't see an alien hard rectangle.
        alpha = np.zeros((h, w), np.float32)
        for (x1, y1, x2, y2) in dboxes:
            alpha[y1:y2, x1:x2] = 1.0
        alpha = cv2.GaussianBlur(alpha, (self._feather_ksize, self._feather_ksize), 0)
        alpha[py:h, px1:px2] = 0.0            # protect the gripper zone (after feather)
        alpha = alpha[:, :, None]

        sub = np.zeros_like(frame) if self._fill_mode == "black" \
            else cv2.GaussianBlur(frame, (self._blur_ksize, self._blur_ksize), 0)
        out = frame.astype(np.float32) * (1 - alpha) + sub.astype(np.float32) * alpha
        return out.clip(0, 255).astype(frame.dtype)

    # -- lifecycle -----------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()


def make_masking_wrapper(**cfg):
    """Return a ThreadSafeRobot-compatible factory preloaded with masking config.

    Monkeypatch usage:
        import lerobot.rollout.context as ctx
        ctx.ThreadSafeRobot = make_masking_wrapper(detector=..., target=..., ...)
    """
    def factory(robot):
        return MaskingThreadSafeRobot(robot, **cfg)
    return factory
