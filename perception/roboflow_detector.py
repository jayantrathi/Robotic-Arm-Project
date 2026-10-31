from __future__ import annotations

import base64
import os

import numpy as np

from detection_types import BaseDetector, Detection

DEFAULT_API_URL = "https://serverless.roboflow.com"

# Roboflow slugifies class names on export; map them back to the strings
# objects.py / the policy tasks use.
_CANON = {
    "pen": "pen",
    "teddybear": "teddy bear",
    "teddy_bear": "teddy bear",
    "teddy-bear": "teddy bear",
    "lipbalm": "lip balm",
    "lip_balm": "lip balm",
    "lip-balm": "lip balm",
    "pouch": "pouch",
}


def _canon(name: str) -> str:
    key = name.strip().lower().replace(" ", "")
    return _CANON.get(key, _CANON.get(name.strip().lower(), name.strip().lower()))


def _env(key: str) -> str | None:
    v = os.environ.get(key)
    if v:
        return v
    path = os.path.expanduser("~/ARM/.env")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _iter_preds(node):
    """Yield detection dicts anywhere in a workflow response."""
    if isinstance(node, list):
        for item in node:
            yield from _iter_preds(item)
    elif isinstance(node, dict):
        has_box = all(k in node for k in ("x", "y", "width", "height"))
        has_cls = "class" in node or "class_name" in node
        if has_box and has_cls and "confidence" in node:
            yield node
        else:
            for v in node.values():
                yield from _iter_preds(v)


class RoboflowDetector(BaseDetector):
    def __init__(
        self,
        model_id: str | None = None,
        version: str | None = None,
        api_key: str | None = None,
        api_url: str | None = None,
        workspace: str | None = None,
        workflow_id: str | None = None,
        device: str | None = None,  # ignored; hosted inference. kept for interface parity
    ):
        model_id = model_id or _env("ROBOFLOW_MODEL")
        version = version or _env("ROBOFLOW_VERSION")
        api_key = api_key or _env("ROBOFLOW_API_KEY")
        api_url = (api_url or _env("ROBOFLOW_API_URL") or DEFAULT_API_URL).rstrip("/")
        workspace = workspace or _env("ROBOFLOW_WORKSPACE")
        workflow_id = workflow_id or _env("ROBOFLOW_WORKFLOW") or model_id

        if not api_key:
            raise ValueError("No ROBOFLOW_API_KEY found (set it in ~/ARM/.env).")
        self.api_key = api_key

        self.use_workflow = bool(workspace)
        if self.use_workflow:
            if not workflow_id:
                raise ValueError("Workflow mode needs ROBOFLOW_WORKFLOW (or ROBOFLOW_MODEL).")
            self.endpoint = f"{api_url}/infer/workflows/{workspace}/{workflow_id}"
        else:
            if not model_id or not version:
                raise ValueError("Serverless mode needs ROBOFLOW_MODEL and ROBOFLOW_VERSION.")
            self.endpoint = f"{api_url}/{model_id}/{version}"

    def _jpeg_b64(self, image) -> str | None:
        import cv2

        arr = np.ascontiguousarray(image)
        ok, buf = cv2.imencode(".jpg", arr[:, :, ::-1])
        if not ok:
            return None
        return base64.b64encode(buf).decode("ascii")

    def detect(self, image, labels: list[str], threshold: float = 0.10) -> list[Detection]:
        import requests

        payload = self._jpeg_b64(image)
        if payload is None:
            return []

        if self.use_workflow:
            resp = requests.post(
                self.endpoint,
                json={"api_key": self.api_key,
                      "inputs": {"image": {"type": "base64", "value": payload}}},
                timeout=30,
            )
            resp.raise_for_status()
            preds = list(_iter_preds(resp.json().get("outputs", resp.json())))
        else:
            resp = requests.post(
                self.endpoint,
                params={"api_key": self.api_key},
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
            resp.raise_for_status()
            preds = resp.json().get("predictions", [])

        allowed = set(labels) if labels else None
        dets: list[Detection] = []
        for p in preds:
            score = float(p.get("confidence", 0.0))
            if score < threshold:
                continue
            label = _canon(str(p.get("class", p.get("class_name", ""))))
            if allowed is not None and label not in allowed:
                continue
            x, y, w, h = p["x"], p["y"], p["width"], p["height"]
            box = (int(round(x - w / 2)), int(round(y - h / 2)),
                   int(round(x + w / 2)), int(round(y + h / 2)))
            dets.append(Detection(label, score, box))
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets
