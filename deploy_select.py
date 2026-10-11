#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perception"))

FOLLOWER_PORT = "/dev/tty.usbmodem5B790815221"
FOLLOWER_ID = "my_follower_arm"
CAMERA_INDEX = 0
CAMERA_KEY = "camera1"
POLICY_PATH = "jayantrathi/smolvla_lang_grasp_v2"
POLICY_DEVICE = "mps"
DEFAULT_DURATION = 40


def gate(detector, detect_target: str, candidates: list[str], threshold: float):
    import cv2

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: could not open camera for the gate check.")
        return None
    for _ in range(10):
        cap.read(); time.sleep(0.05)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("ERROR: camera read failed during gate check.")
        return None
    rgb = frame[:, :, ::-1]
    return detector.select(rgb, detect_target, candidates=candidates, threshold=threshold)


def run_rollout(task: str, duration: int) -> None:
    cameras = (f"{{ {CAMERA_KEY}: {{type: opencv, index_or_path: {CAMERA_INDEX}, "
               f"width: 640, height: 480, fps: 30}} }}")
    argv = [
        "deploy_select",
        "--strategy.type=base",
        f"--policy.path={POLICY_PATH}",
        f"--policy.device={POLICY_DEVICE}",
        "--robot.type=so101_follower",
        f"--robot.port={FOLLOWER_PORT}",
        f"--robot.id={FOLLOWER_ID}",
        f"--robot.cameras={cameras}",
        f"--task={task}",
        "--inference.type=rtc",
        f"--duration={duration}",
        "--display_data=true",
    ]
    sys.argv = argv
    from lerobot.scripts import lerobot_rollout
    lerobot_rollout.main()


def main() -> int:
    ap = argparse.ArgumentParser(description="Grasp a named object, chosen by the detector.")
    ap.add_argument("object", help="object to grasp, e.g. 'pen', 'the teddy bear'")
    ap.add_argument("--mask", action="store_true", help="hide other objects from the camera feed")
    ap.add_argument("--search", action="store_true", help="pan to find the object if it's not in view")
    ap.add_argument("--duration", type=int, default=DEFAULT_DURATION)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--detector-device", default="cpu", help="cpu keeps it off the policy's GPU")
    ap.add_argument("--force", action="store_true", help="run even if the object isn't found")
    ap.add_argument("--auto-stop", action="store_true", help="experimental: stop when the arm returns home")
    ap.add_argument("--fill-mode", default="blur", choices=["inpaint", "blur", "black"])
    ap.add_argument("--blur", type=int, default=61)
    ap.add_argument("--dilate", type=float, default=0.25)
    args = ap.parse_args()

    from objects import REGISTRY, all_detect_labels, resolve

    obj = resolve(args.object) or REGISTRY.get(args.object.lower())
    if obj is None:
        print(f"Don't know '{args.object}'. Known: {', '.join(REGISTRY)}")
        return 2
    detect_target = obj.detect[0]
    candidates = all_detect_labels()
    print(f"Target: {obj.name!r}  (detect as {detect_target!r})  ->  task: {obj.task!r}")

    print(f"Loading detector on {args.detector_device}...")
    from detector import ObjectDetector
    detector = ObjectDetector(device=args.detector_device)

    found_pan = None
    if args.search:
        from search import build_robot, search_for
        print("Search mode: panning to find the object...")
        srobot = build_robot()
        srobot.connect()
        try:
            found_pan = search_for(srobot, detector, detect_target, candidates, threshold=args.threshold)
        finally:
            srobot.disconnect()
        if found_pan is None and not args.force:
            print("  Search didn't find the object. Aborting (use --force to run anyway).")
            return 1
        sel = None
    else:
        print("Gate check: is the object in view?")
        sel = gate(detector, detect_target, candidates, args.threshold)
        if sel is None:
            return 1
        if not sel.found:
            print(f"  NOT found. Detections: "
                  f"{[(d.label, round(d.score,2)) for d in (sel.all_detections or [])][:5]}")
            if not args.force:
                print("  Aborting (use --force, or --search to look for it).")
                return 1
            print("  --force set: continuing despite no detection.")
        else:
            print(f"  found {detect_target!r} at center={sel.center} "
                  f"score={sel.score:.2f}" + ("  [AMBIGUOUS - two candidates]" if sel.ambiguous else ""))

    import lerobot.rollout.context as ctx
    autostop = args.auto_stop

    if args.mask:
        initial_distractors: list[tuple] = []
        seen: set[str] = set()
        for d in (sel.all_detections if sel else []) or []:
            if d.label != detect_target and d.label not in seen:
                initial_distractors.append(d.box)
                seen.add(d.label)
        print(f"Masking ON: blurring {len(initial_distractors)} distractor(s) out of the feed."
              + ("  Auto-stop ON." if autostop else ""))
        from masking_robot import make_masking_wrapper
        ctx.ThreadSafeRobot = make_masking_wrapper(
            detector=detector,
            target=detect_target,
            candidates=candidates,
            camera_key=CAMERA_KEY,
            threshold=args.threshold,
            fill_mode=args.fill_mode,
            blur_ksize=args.blur,
            dilate=args.dilate,
            initial_distractors=initial_distractors,
            autostop=autostop,
        )
    elif autostop:
        print("Auto-stop ON: will end the run when the arm returns home.")
        from autostop import make_autostop_wrapper
        ctx.ThreadSafeRobot = make_autostop_wrapper(autostop=True)

    print(f"\nRunning policy for {args.duration}s...\n")
    run_rollout(obj.task, args.duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
