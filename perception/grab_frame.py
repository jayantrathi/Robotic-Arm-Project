#!/usr/bin/env python3
"""
Grab one (or a few) frames from the wrist camera and save them to
perception/live/. Used for tuning the detector against the real scene.

Run it from a Terminal that has macOS Camera permission:
    ~/robotarm/env/bin/python ~/ARM/perception/grab_frame.py          # 1 frame
    ~/robotarm/env/bin/python ~/ARM/perception/grab_frame.py 5        # 5 frames, 1s apart

The FIRST time you run it, macOS will pop up a Camera permission prompt for
your Terminal app -> click Allow. After that it just works.
"""
import os
import sys
import time

import cv2

CAMERA_INDEX = 0
OUT_DIR = os.path.join(os.path.dirname(__file__), "live")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    os.makedirs(OUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: could not open camera index", CAMERA_INDEX)
        print("If macOS blocked it, grant this Terminal app Camera access in")
        print("System Settings > Privacy & Security > Camera, then rerun.")
        return 1

    # let autoexposure/white-balance settle before the first save
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    saved = []
    for i in range(n):
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"frame {i}: read FAILED")
            continue
        path = os.path.join(OUT_DIR, f"live_{i}.jpg")
        cv2.imwrite(path, frame)
        saved.append(path)
        print(f"frame {i}: saved {path}  shape={frame.shape}")
        if n > 1:
            time.sleep(1.0)
    cap.release()

    if saved:
        print(f"\nDone. {len(saved)} frame(s) in {OUT_DIR}/")
        return 0
    print("\nNo frames captured.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
