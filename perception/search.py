#!/usr/bin/env python3
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

SCAN_POSE = {
    "shoulder_lift.pos": -103.0,
    "elbow_flex.pos": 96.0,
    "wrist_flex.pos": 53.0,
    "wrist_roll.pos": 5.0,
    "gripper.pos": 2.0,
}
SAFE = {
    "shoulder_pan.pos": (-60.0, 82.0),
    "shoulder_lift.pos": (-105.0, 70.0),
    "elbow_flex.pos": (-85.0, 97.0),
    "wrist_flex.pos": (-56.0, 102.0),
    "wrist_roll.pos": (-22.0, 91.0),
    "gripper.pos": (0.0, 90.0),
}

PAN_HOME = 39.0
PAN_SWEEP = list(range(80, -56, -12))
SETTLE_S = 0.4


def _clamp(key, val):
    lo, hi = SAFE[key]
    return float(np.clip(val, lo, hi))


def _present_pose(robot) -> dict:
    obs = robot.get_observation()
    return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}


def move_to(robot, target: dict, steps: int = 30, dt: float = 0.05) -> None:
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
        disable_torque_on_disconnect=False,
    )
    return SO101Follower(cfg)


def _detect_at(robot, detector, target, candidates, threshold):
    time.sleep(SETTLE_S)
    obs = robot.get_observation()
    frame = obs.get(CAMERA_KEY)
    if frame is None:
        return None
    return detector.select(frame, target, candidates=candidates, threshold=threshold)


def _center_on(robot, detector, target, candidates, threshold, pan, tol_px=55, max_iter=5) -> float:
    cx0 = 320

    def err_at(p):
        move_to(robot, {"shoulder_pan.pos": _clamp("shoulder_pan.pos", p)}, steps=10, dt=0.05)
        sel = _detect_at(robot, detector, target, candidates, threshold)
        if sel is None or not sel.found:
            return None
        return sel.center[0] - cx0

    e0 = err_at(pan)
    if e0 is None or abs(e0) <= tol_px:
        return pan
    probe = 5.0
    e1 = err_at(pan + probe)
    if e1 is None:
        move_to(robot, {"shoulder_pan.pos": pan}, steps=10, dt=0.05)
        return pan
    slope = (e1 - e0) / probe
    if abs(slope) < 1.0:
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


def search_for(robot, detector, target, candidates, threshold=0.18):
    print("[search] moving to scan pose...")
    move_to(robot, {**SCAN_POSE, "shoulder_pan.pos": PAN_HOME})

    for pan in [PAN_HOME] + PAN_SWEEP:
        move_to(robot, {"shoulder_pan.pos": pan}, steps=18, dt=0.05)
        sel = _detect_at(robot, detector, target, candidates, threshold)
        found = sel is not None and sel.found
        cx = sel.center[0] if (found and sel.center) else None
        print(f"[search] pan={pan:+.0f}  " + (f"found@{cx} score={sel.score:.2f}" if found else "none"))
        if found:
            final = _center_on(robot, detector, target, candidates, threshold, pan)
            print(f"[search] object centred at pan={final:+.0f}. Arm left facing it.")
            return final

    print("[search] object not found across the sweep. Returning to home pan.")
    move_to(robot, {**SCAN_POSE, "shoulder_pan.pos": PAN_HOME})
    return None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("object")
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
        final_pan = search_for(robot, detector, detect_target, candidates, threshold=args.threshold)
        if final_pan is not None:
            print(f"\nRESULT: FOUND at pan={final_pan:+.0f} (offset from home {PAN_HOME:.0f} = {final_pan-PAN_HOME:+.0f})")
        else:
            print("\nRESULT: NOT FOUND")
    finally:
        robot.disconnect()
    return 0 if final_pan is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
