#!/usr/bin/env python3
"""
Off-frame search: pan the arm to find an object that starts outside the camera's
current view, then leave the arm facing it so the grasp policy can take over.

How it works
------------
The SO-101 start pose (read from the real dataset, very consistent) points the
wrist camera at the table. Holding that pose and sweeping only `shoulder_pan`
rotates the camera left/right across the workspace. At each pan step we grab a
frame and run the detector; when the object appears in view we stop, leaving the
arm in a valid grasp-start pose, just rotated to face the object.

Safety
------
- Every commanded joint is clamped to the safe range observed in the dataset.
- Motion is interpolated in small steps (slow slew), never a single big jump.
- Torque is held on disconnect, so the arm keeps its pose for the grasp handoff
  (the grasp process reconnects and only sets PID/mode -- it doesn't move the arm).

Standalone (search only, no grasp) -- use this to watch the motion first:
    ~/robotarm/env/bin/python ~/ARM/perception/search.py bear
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

FOLLOWER_PORT = "/dev/tty.usbmodem5B790815221"
FOLLOWER_ID = "my_follower_arm"
CAMERA_INDEX = 0
CAMERA_KEY = "camera1"

# Scan pose = the dataset's consistent start pose (wrist cam looking at the table).
# shoulder_pan is what we sweep; the rest are held here.
SCAN_POSE = {
    "shoulder_lift.pos": -103.0,
    "elbow_flex.pos": 96.0,
    "wrist_flex.pos": 53.0,
    "wrist_roll.pos": 5.0,
    "gripper.pos": 2.0,
}
# Safe joint ranges (from observed dataset min/max, kept a little inside).
SAFE = {
    "shoulder_pan.pos": (-60.0, 82.0),
    "shoulder_lift.pos": (-105.0, 70.0),
    "elbow_flex.pos": (-85.0, 97.0),
    "wrist_flex.pos": (-56.0, 102.0),
    "wrist_roll.pos": (-22.0, 91.0),
    "gripper.pos": (0.0, 90.0),
}

PAN_HOME = 39.0            # the start pose's pan angle
PAN_SWEEP = list(range(80, -56, -12))  # sweep high->low across the safe pan range
SETTLE_S = 0.4            # let the arm settle + autoexposure before detecting


def _clamp(key, val):
    lo, hi = SAFE[key]
    return float(np.clip(val, lo, hi))


def _present_pose(robot) -> dict:
    obs = robot.get_observation()
    return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}


def move_to(robot, target: dict, steps: int = 30, dt: float = 0.05) -> None:
    """Interpolate smoothly from the present pose to `target` (clamped)."""
    target = {k: _clamp(k, v) for k, v in target.items()}
    start = _present_pose(robot)
    for i in range(1, steps + 1):
        a = i / steps
        cmd = {k: (1 - a) * start.get(k, v) + a * v for k, v in target.items()}
        robot.send_action(cmd)
        time.sleep(dt)


def build_robot():
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    cams = {CAMERA_KEY: OpenCVCameraConfig(index_or_path=CAMERA_INDEX,
                                           width=640, height=480, fps=30)}
    cfg = SO101FollowerConfig(
        port=FOLLOWER_PORT, id=FOLLOWER_ID, cameras=cams,
        use_degrees=True,
        disable_torque_on_disconnect=False,   # hold pose for the grasp handoff
    )
    return SO101Follower(cfg)


def _detect_at(robot, detector, target, candidates, threshold):
    """Grab the current camera frame and look for the target."""
    time.sleep(SETTLE_S)
    obs = robot.get_observation()
    frame = obs.get(CAMERA_KEY)
    if frame is None:
        return None
    return detector.select(frame, target, candidates=candidates, threshold=threshold)


def _center_on(robot, detector, target, candidates, threshold, pan,
               tol_px=55, max_iter=5) -> float:
    """Nudge shoulder_pan until the object is centred in the frame.

    The sign of (image-x change)/(pan change) depends on camera mounting, so we
    probe once with a small nudge to learn it, then servo proportionally. If the
    object is ever lost during centring we revert to the last good pan and stop.
    """
    cx0 = 320  # image centre (640 wide)

    def err_at(p):
        move_to(robot, {"shoulder_pan.pos": _clamp("shoulder_pan.pos", p)}, steps=10, dt=0.05)
        sel = _detect_at(robot, detector, target, candidates, threshold)
        if sel is None or not sel.found:
            return None
        return sel.center[0] - cx0

    e0 = err_at(pan)
    if e0 is None or abs(e0) <= tol_px:
        return pan
    # probe direction
    probe = 5.0
    e1 = err_at(pan + probe)
    if e1 is None:                      # lost it -> revert, accept current
        move_to(robot, {"shoulder_pan.pos": pan}, steps=10, dt=0.05)
        return pan
    slope = (e1 - e0) / probe           # px per degree
    if abs(slope) < 1.0:                # pan barely moves the object; stop
        return pan + probe
    cur = pan + probe
    for _ in range(max_iter):
        e = err_at(cur)
        if e is None:
            move_to(robot, {"shoulder_pan.pos": cur}, steps=8, dt=0.05)
            return cur
        if abs(e) <= tol_px:
            break
        adj = float(np.clip(-e / slope, -12.0, 12.0))
        cur = _clamp("shoulder_pan.pos", cur + adj)
    print(f"[search] centred at pan={cur:+.0f} (err={e if e is not None else '?'}px)")
    return cur


def search_for(robot, detector, target, candidates, threshold=0.18) -> bool:
    """Pan to find `target`, centre on it, and leave the arm facing it. -> found?"""
    # Move to the scan pose at the home pan first (known-safe, camera on the table).
    print("[search] moving to scan pose...")
    move_to(robot, {**SCAN_POSE, "shoulder_pan.pos": PAN_HOME})

    # Sweep until the object appears, then centre on it.
    for pan in [PAN_HOME] + PAN_SWEEP:
        move_to(robot, {"shoulder_pan.pos": pan}, steps=18, dt=0.05)
        sel = _detect_at(robot, detector, target, candidates, threshold)
        found = sel is not None and sel.found
        cx = sel.center[0] if (found and sel.center) else None
        print(f"[search] pan={pan:+.0f}  " + (f"found@{cx} score={sel.score:.2f}" if found else "none"))
        if found:
            final = _center_on(robot, detector, target, candidates, threshold, pan)
            print(f"[search] object centred at pan={final:+.0f}. Arm left facing it.")
            return True

    # Not found: return to home pan so the arm is in a known pose.
    print("[search] object not found across the sweep. Returning to home pan.")
    move_to(robot, {**SCAN_POSE, "shoulder_pan.pos": PAN_HOME})
    return False


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Pan the arm to find an object (search only, no grasp).")
    ap.add_argument("object", help="object to find, e.g. 'bear'")
    ap.add_argument("--threshold", type=float, default=0.18)
    args = ap.parse_args()

    from detector import ObjectDetector
    from objects import REGISTRY, all_detect_labels, resolve

    obj = resolve(args.object) or REGISTRY.get(args.object.lower())
    if obj is None:
        print(f"Don't know '{args.object}'. Known: {', '.join(REGISTRY)}")
        return 2
    detect_target = obj.detect[0]
    candidates = all_detect_labels()

    print("Loading detector (cpu)...")
    detector = ObjectDetector(device="cpu")

    print("Connecting robot...")
    robot = build_robot()
    robot.connect()
    try:
        ok = search_for(robot, detector, detect_target, candidates, threshold=args.threshold)
        print("\nRESULT:", "FOUND" if ok else "NOT FOUND")
    finally:
        robot.disconnect()   # torque held (disable_torque_on_disconnect=False)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
