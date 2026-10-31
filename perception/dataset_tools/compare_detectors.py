#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from glob import glob

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_COCO = os.path.expanduser("~/ARM/rf_dataset/clutter_labeled")
IOU_MATCH = 0.5


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def load_gt(coco_dir: str):
    with open(os.path.join(coco_dir, "_annotations.coco.json")) as f:
        coco = json.load(f)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_file: dict[str, list[tuple[str, tuple]]] = {}
    img_by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    for im in coco["images"]:
        by_file.setdefault(im["file_name"], [])
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        box = (x, y, x + w, y + h)
        fn = img_by_id[a["image_id"]]
        by_file[fn].append((cats[a["category_id"]], box))
    return by_file


def build_detector(kind: str):
    from objects import all_detect_labels
    labels = all_detect_labels()
    if kind == "roboflow":
        from roboflow_detector import RoboflowDetector
        return RoboflowDetector(), labels
    from detector import ObjectDetector
    return ObjectDetector(device="cpu"), labels


def run(kind: str, coco_dir: str, gt, threshold: float, writer):
    detector, candidates = build_detector(kind)
    canon = {c.lower(): c for c in candidates}
    totals = {"pairs": 0, "found": 0, "correct": 0}
    lat_ms: list[float] = []

    for fname, items in sorted(gt.items()):
        path = os.path.join(coco_dir, fname)
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        rgb = bgr[:, :, ::-1]

        t0 = time.perf_counter()
        dets = detector.detect(rgb, candidates, threshold=threshold)
        lat = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(lat)

        present = {}
        for cls, box in items:
            present.setdefault(canon.get(cls.lower(), cls), []).append(box)

        for target, gt_boxes in present.items():
            hits = [d for d in dets if d.label == target]
            found = bool(hits)
            best_iou = 0.0
            if found:
                top = hits[0]
                best_iou = max(iou(top.box, g) for g in gt_boxes)
            correct = found and best_iou >= IOU_MATCH
            totals["pairs"] += 1
            totals["found"] += int(found)
            totals["correct"] += int(correct)
            writer.writerow([kind, fname, target, int(found), round(best_iou, 3),
                             int(correct), round(lat, 1)])

    return totals, lat_ms


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def pctl(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def main() -> int:
    ap = argparse.ArgumentParser(description="OWLv2 vs Roboflow: selection accuracy + latency.")
    ap.add_argument("--coco-dir", default=DEFAULT_COCO)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--detectors", nargs="+", default=["owlv2", "roboflow"])
    ap.add_argument("--out", default=os.path.expanduser("~/ARM/rf_dataset/detector_compare.csv"))
    args = ap.parse_args()

    gt = load_gt(args.coco_dir)
    n_frames = len(gt)
    n_pairs = sum(len({c for c, _ in v}) for v in gt.values())
    print(f"GT: {n_frames} frames, {n_pairs} (frame, class-present) pairs from {args.coco_dir}")
    print(f"Match: IoU >= {IOU_MATCH}, threshold = {args.threshold}\n")

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["detector", "frame", "target", "found", "iou", "correct", "latency_ms"])
        summary = {}
        for kind in args.detectors:
            print(f"Running {kind}...")
            totals, lat = run(kind, args.coco_dir, gt, args.threshold, writer)
            summary[kind] = (totals, lat)

    print(f"\nWrote per-frame rows to {args.out}\n")
    print(f"{'detector':<10} {'recall':>8} {'accuracy':>9} {'lat p50':>9} {'lat p90':>9}")
    for kind, (t, lat) in summary.items():
        print(f"{kind:<10} {pct(t['found'], t['pairs']):>8} {pct(t['correct'], t['pairs']):>9} "
              f"{pctl(lat, 0.5):>7.0f}ms {pctl(lat, 0.9):>7.0f}ms")
    print("\nrecall   = target present and detector returned a box of that class")
    print("accuracy = that box overlaps the right instance (IoU >= 0.5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
