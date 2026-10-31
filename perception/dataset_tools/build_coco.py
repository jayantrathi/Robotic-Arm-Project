#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_REPO = "jayantrathi/lang_grasp_v2"
DEFAULT_OUT = os.path.expanduser("~/ARM/rf_dataset/labeled")
OFFSET_FRACS = [0.0, 0.08, 0.16, 0.25, 0.35]
CAMERA_KEY = "observation.images.front"
MIN_AREA_FRAC = 0.002
MAX_AREA_FRAC = 0.8


def tensor_to_rgb(t) -> np.ndarray:
    return (t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)


def box_ok(box, w, h) -> bool:
    x1, y1, x2, y2 = box
    frac = ((x2 - x1) * (y2 - y1)) / float(w * h)
    return MIN_AREA_FRAC <= frac <= MAX_AREA_FRAC


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--threshold", type=float, default=0.08)
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from detector import ObjectDetector
    from objects import all_detect_labels, resolve

    categories = all_detect_labels()
    cat_id = {name: i + 1 for i, name in enumerate(categories)}

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(args.repo)
    os.makedirs(args.out, exist_ok=True)
    detector = ObjectDetector(device=args.device)
    print(f"Detector on {detector.device}. {ds.meta.total_episodes} episodes.")

    records = []
    for ep in ds.meta.episodes:
        obj = resolve(ep["tasks"][0])
        if obj is None:
            continue
        label = obj.detect[0]
        start, length = ep["dataset_from_index"], ep["length"]
        for frac in OFFSET_FRACS:
            rgb = tensor_to_rgb(ds[start + int(frac * length)][CAMERA_KEY])
            h, w = rgb.shape[:2]
            sel = detector.select(rgb, label, candidates=[label], threshold=args.threshold)
            if not sel.found or not box_ok(sel.box, w, h):
                continue
            fname = f"ep{ep['episode_index']:03d}_{int(frac*100):02d}.jpg"
            cv2.imwrite(os.path.join(args.out, fname), rgb[:, :, ::-1])
            records.append({"fname": fname, "label": label, "box": sel.box, "w": w, "h": h})

    rng = random.Random(args.seed)
    kept = []
    for label in categories:
        hits = [r for r in records if r["label"] == label]
        rng.shuffle(hits)
        drop = hits[args.per_class:]
        for r in drop:
            os.remove(os.path.join(args.out, r["fname"]))
        kept.extend(hits[:args.per_class])

    coco = {
        "info": {"description": "so101 grasp objects, OWLv2 pre-labels"},
        "licenses": [],
        "categories": [{"id": cat_id[n], "name": n, "supercategory": "none"} for n in categories],
        "images": [],
        "annotations": [],
    }
    for i, r in enumerate(kept):
        coco["images"].append({"id": i, "file_name": r["fname"], "width": r["w"], "height": r["h"]})
        x1, y1, x2, y2 = r["box"]
        bw, bh = x2 - x1, y2 - y1
        coco["annotations"].append({
            "id": i, "image_id": i, "category_id": cat_id[r["label"]],
            "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0, "segmentation": [],
        })
    with open(os.path.join(args.out, "_annotations.coco.json"), "w") as f:
        json.dump(coco, f)

    print(f"\nWrote {len(kept)} images to {args.out}")
    for label in categories:
        print(f"  {label}: {sum(1 for r in kept if r['label'] == label)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
