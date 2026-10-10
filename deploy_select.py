#!/usr/bin/env python3
"""
Deploy the grasp policy on a SPECIFIC named object, chosen by the detector.

This is deploy.sh + perception. Given an object name (typed here, or later
handed over by voice_grasp.py), it:

  1. GATE: grabs one camera frame and confirms the object is actually there
     before moving the arm. If it's not found, we stop -- no blind flailing.
  2. MASK (optional, --mask): blacks the other objects out of the camera feed so
     the single-object-reliable policy isn't confused by a distractor. No
     retraining; see perception/masking_robot.py.
  3. RUN: launches LeRobot's normal rollout with the trained instruction for
     that object.

Usage:
    python deploy_select.py pen                 # gate + grasp the pen
    python deploy_select.py "the teddy" --mask  # also mask out everything else
    python deploy_select.py bear --duration 30

Start with gate-only (no --mask) to confirm the pipeline, then add --mask for
true two-object disambiguation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perception"))

# --- Config: matches scripts/deploy.sh -------------------------------------
FOLLOWER_PORT = "/dev/tty.usbmodem5B790815221"
FOLLOWER_ID = "my_follower_arm"
CAMERA_INDEX = 0
CAMERA_KEY = "camera1"               # must match the policy's trained camera name
POLICY_PATH = "jayantrathi/smolvla_lang_grasp_v2"
POLICY_DEVICE = "mps"
DEFAULT_DURATION = 40            # safety cap only; auto-stop normally ends the run at ~home
# ---------------------------------------------------------------------------


def gate(detector, detect_target: str, candidates: list[str], threshold: float):
    """Grab one frame, confirm the object is present. Returns the Selection."""
    import cv2

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: could not open camera for the gate check.")
        return None
    for _ in range(10):          # let exposure settle
        cap.read(); time.sleep(0.05)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("ERROR: camera read failed during gate check.")
        return None

    rgb = frame[:, :, ::-1]       # BGR -> RGB (detector makes it contiguous)
    sel = detector.select(rgb, detect_target, candidates=candidates, threshold=threshold)
    return sel


def run_rollout(task: str, duration: int) -> None:
    """Invoke LeRobot's rollout CLI in-process with our deploy flags."""
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
    ap.add_argument("object", help="object to grasp (e.g. 'pen', 'the teddy bear')")
    ap.add_argument("--mask", action="store_true",
                    help="black out other objects in the camera feed (two-object disambiguation)")
    ap.add_argument("--duration", type=int, default=DEFAULT_DURATION)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--detector-device", default="cpu",
                    help="where the detector runs. Keep 'cpu' so it doesn't fight the "
                         "policy for the MPS GPU (that stalls the arm). Use 'mps' only "
                         "without --mask.")
    ap.add_argument("--search", action="store_true",
                    help="if the object isn't already in view, pan the arm to find it "
                         "and centre on it before grasping")
    ap.add_argument("--force", action="store_true",
                    help="run even if the gate check doesn't find the object")
    ap.add_argument("--auto-stop", action="store_true",
                    help="EXPERIMENTAL: auto-end the run when the arm returns home. "
                         "Off by default -- just press Ctrl-C when the grasp is done "
                         "(cleanly returns the arm home). Can mis-fire mid-grasp if the "
                         "policy swings near the home pose partway through.")
    # Masking tunables -- iterate on these at the arm without a code change.
    # Masking blurs only the DISTRACTOR objects; the target + rest of the frame
    # are left untouched.
    ap.add_argument("--fill-mode", default="blur", choices=["inpaint", "blur", "black"],
                    help="how to remove the distractor: inpaint (paint it out with table "
                         "texture, cleanest), blur (soft blob), or black")
    ap.add_argument("--blur", type=int, default=61,
                    help="blur strength on the distractor; raise to smear a stubborn "
                         "distractor more (e.g. 81, 101)")
    ap.add_argument("--dilate", type=float, default=0.25,
                    help="how much to grow each distractor box so it's fully covered")
    args = ap.parse_args()

    from objects import REGISTRY, all_detect_labels, resolve

    obj = resolve(args.object) or REGISTRY.get(args.object.lower())
    if obj is None:
        print(f"Don't know '{args.object}'. Known: {', '.join(REGISTRY)}")
        return 2
    detect_target = obj.detect[0]
    candidates = all_detect_labels()
    print(f"Target: {obj.name!r}  (detect as {detect_target!r})  ->  task: {obj.task!r}")

    # Load the detector once; reused by the gate and (if --mask) the mask thread.
    # Default device is CPU: during --mask the detector runs concurrently with
    # the policy, and sharing the MPS GPU stalls the policy so the arm freezes.
    print(f"Loading detector on {args.detector_device}...")
    from detector import ObjectDetector
    detector = ObjectDetector(device=args.detector_device)

    # 1. FIND THE OBJECT
    found_pan = None
    if args.search:
        # Pan the arm to find + centre on the object, leaving it facing the object
        # (torque held, so it keeps the pose while the rollout reconnects).
        from search import build_robot, search_for
        print("Search mode: panning to find the object...")
        srobot = build_robot()
        srobot.connect()
        try:
            found_pan = search_for(srobot, detector, detect_target, candidates,
                                   threshold=args.threshold)
        finally:
            srobot.disconnect()   # disable_torque_on_disconnect=False -> arm holds pose
        if found_pan is None and not args.force:
            print("  Search didn't find the object. Aborting (use --force to run anyway).")
            return 1
        sel = None  # no gate selection; the mask thread will find distractors itself
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

    # 2. Install the robot wrapper (monkeypatch before the rollout builds it):
    #    - auto-stop at return-home always (unless --no-auto-stop)
    #    - masking on top when --mask
    import lerobot.rollout.context as ctx
    autostop = args.auto_stop

    if args.mask:
        # distractor boxes the gate already saw = best box per non-target label
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
            # seed with the gate's distractor boxes so suppression is live from
            # the first frame (no unmasked startup window)
            initial_distractors=initial_distractors,
            autostop=autostop,
        )
    elif autostop:
        print("Auto-stop ON: will end the run when the arm returns home.")
        from autostop import make_autostop_wrapper
        ctx.ThreadSafeRobot = make_autostop_wrapper(autostop=True)

    # 3. RUN
    print(f"\nRunning policy for {args.duration}s...\n")
    run_rollout(obj.task, args.duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
