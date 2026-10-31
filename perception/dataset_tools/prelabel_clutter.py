#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from glob import glob

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_IN = os.path.expanduser("~/ARM/rf_dataset/clutter")
DEFAULT_OUT = os.path.expanduser("~/ARM/rf_dataset/clutter_labeled")
MIN_AREA_FRAC = 0.001
MAX_AREA_FRAC = 0.8


def box_ok(box, w, h) -> bool:
    x1, y1, x2, y2 = box
    frac = ((x2 - x1) * (y2 - y1)) / float(w * h)
    return MIN_AREA_FRAC <= frac <= MAX_AREA_FRAC


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default=DEFAULT_IN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--threshold", type=float, default=0.12)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from detector import ObjectDetector
    from objects import all_detect_labels

    categories = all_detect_labels()
    cat_id = {name: i + 1 for i, name in enumerate(categories)}

    frames = sorted(glob(os.path.join(args.in_dir, "*.jpg")))
    if not frames:
        print(f"No frames in {args.in_dir}")
        return 1

    os.makedirs(args.out, exist_ok=True)
    detector = ObjectDetector(device=args.device)
    print(f"Detector on {detector.device}. {len(frames)} clutter frames.")

    coco = {
        "info": {"description": "so101 clutter frames, OWLv2 pre-labels"},
        "licenses": [],
        "categories": [{"id": cat_id[n], "name": n, "supercategory": "none"} for n in categories],
        "images": [],
        "annotations": [],
    }
    ann_id = 0
    per_class = {n: 0 for n in categories}
    empty = []

    for img_id, path in enumerate(frames):
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        rgb = bgr[:, :, ::-1]
        dets = detector.detect(rgb, categories, threshold=args.threshold)
        dets = [d for d in dets if box_ok(d.box, w, h)]

        fname = os.path.basename(path)
        shutil.copy2(path, os.path.join(args.out, fname))
        coco["images"].append({"id": img_id, "file_name": fname, "width": w, "height": h})
        if not dets:
            empty.append(fname)
        for d in dets:
            x1, y1, x2, y2 = d.box
            bw, bh = x2 - x1, y2 - y1
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cat_id[d.label],
                "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0, "segmentation": [],
            })
            ann_id += 1
            per_class[d.label] += 1

    with open(os.path.join(args.out, "_annotations.coco.json"), "w") as f:
        json.dump(coco, f)

    print(f"\nWrote {len(coco['images'])} images, {ann_id} boxes to {args.out}")
    for n in categories:
        print(f"  {n}: {per_class[n]}")
    print(f"\n{len(empty)} frames with NO detections (review/drop in Roboflow):")
    for e in empty:
        print(f"  {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
