#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_OUT = os.path.expanduser("~/ARM/rf_dataset/clutter")
DISPLAY_MODE = "rerun"


def start_preview():
    try:
        from lerobot.utils.visualization_utils import init_visualization
        init_visualization(DISPLAY_MODE, session_name="freehand")
        return True
    except Exception as e:
        print(f"[preview off: {e}] You can still capture, just without the live feed.")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--port", default=None, help="follower port if different from search.py")
    args = ap.parse_args()

    import search
    if args.port:
        search.FOLLOWER_PORT = args.port
    from search import build_robot, CAMERA_KEY

    os.makedirs(args.out, exist_ok=True)
    count = len([f for f in os.listdir(args.out) if f.endswith(".jpg")])

    print("Connecting robot...")
    robot = build_robot()
    robot.connect()

    input("\nHold the arm so it doesn't drop, then press Enter to go limp...")
    robot.bus.disable_torque()
    print("Torque OFF. Move the arm by hand to frame a shot.")

    preview = start_preview()
    latest = [None]
    lock = threading.Lock()
    stop = threading.Event()

    def stream():
        from lerobot.utils.visualization_utils import log_visualization_data
        while not stop.is_set():
            try:
                obs = robot.get_observation()
            except Exception:
                time.sleep(0.05)
                continue
            frame = obs.get(CAMERA_KEY)
            with lock:
                latest[0] = frame
            if preview:
                try:
                    log_visualization_data(DISPLAY_MODE, observation=obs)
                except Exception:
                    pass
            time.sleep(0.05)

    worker = threading.Thread(target=stream, daemon=True)
    worker.start()

    print(f"\n{count} frames already here. Enter = capture, 'q' = finish.")
    try:
        while True:
            cmd = input(f"[{count}] frame it, Enter to capture (or 'q'): ")
            if cmd.strip().lower() == "q":
                break
            with lock:
                frame = None if latest[0] is None else latest[0].copy()
            if frame is None:
                print("  no frame yet, give the camera a second")
                continue
            path = os.path.join(args.out, f"freehand_{count:03d}.jpg")
            cv2.imwrite(path, frame[:, :, ::-1])
            count += 1
            print(f"  saved {os.path.basename(path)}")
    finally:
        stop.set()
        worker.join(timeout=2.0)
        print("\nHold the arm, re-locking torque at its current pose...")
        robot.bus.enable_torque()
        robot.disconnect()
        print(f"Done. {count} frames in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
