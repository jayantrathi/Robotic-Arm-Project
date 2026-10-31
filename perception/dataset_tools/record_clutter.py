#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_OUT = os.path.expanduser("~/ARM/rf_dataset/clutter")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--target", type=int, default=60)
    ap.add_argument("--port", default=None, help="follower port if different from search.py")
    ap.add_argument("--angles", default=None, help="comma-separated pan angles; default = search sweep")
    args = ap.parse_args()

    import search
    if args.port:
        search.FOLLOWER_PORT = args.port
    from search import build_robot, move_to, SCAN_POSE, PAN_HOME, PAN_SWEEP, CAMERA_KEY, _clamp

    if args.angles:
        angles = [_clamp("shoulder_pan.pos", float(a)) for a in args.angles.split(",")]
    else:
        angles = [PAN_HOME] + PAN_SWEEP

    os.makedirs(args.out, exist_ok=True)
    count = len([f for f in os.listdir(args.out) if f.endswith(".jpg")])

    print("Connecting robot...")
    robot = build_robot()
    robot.connect()
    try:
        print("Moving to scan pose...")
        move_to(robot, {**SCAN_POSE, "shoulder_pan.pos": PAN_HOME})
        print(f"\nReady. {count} frames already here, target {args.target}.")
        print(f"Each sweep pans across {len(angles)} angles: {[round(a) for a in angles]}")
        while count < args.target:
            cmd = input(f"\n[{count}/{args.target}] Spread objects across the desk, Enter to sweep (or 'q'): ")
            if cmd.strip().lower() == "q":
                break
            for a in angles:
                move_to(robot, {"shoulder_pan.pos": a}, steps=12, dt=0.05)
                time.sleep(0.4)
                frame = robot.get_observation().get(CAMERA_KEY)
                if frame is None:
                    print("  camera read failed")
                    continue
                path = os.path.join(args.out, f"clutter_{count:03d}.jpg")
                cv2.imwrite(path, frame[:, :, ::-1])
                count += 1
                print(f"  saved {os.path.basename(path)} @ pan={a:+.0f}")
            move_to(robot, {"shoulder_pan.pos": PAN_HOME}, steps=15, dt=0.05)
        print(f"\nDone. {count} frames in {args.out}")
    finally:
        move_to(robot, {**SCAN_POSE, "shoulder_pan.pos": PAN_HOME}, steps=20)
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
