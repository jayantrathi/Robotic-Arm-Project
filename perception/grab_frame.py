#!/usr/bin/env python3
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
        print("Grant this terminal Camera access in System Settings > Privacy & Security > Camera.")
        return 1

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
