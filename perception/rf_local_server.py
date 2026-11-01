#!/usr/bin/env python3
# Local RF-DETR inference server. Run it in the isolated inference venv:
#   ~/rf_infer/bin/python perception/rf_local_server.py
# Then point the robot at it:  export ROBOFLOW_LOCAL_URL=http://localhost:9145
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _env(key: str, default: str | None = None) -> str | None:
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
    return default


API_KEY = _env("ROBOFLOW_API_KEY")
MODEL_ID = _env("ROBOFLOW_MODEL_ID",
                "jayants-workspace-mbbme/so101-objectdetection-2-rfdetr-nas-pecoret-t1--13f96f")
PORT = int(_env("ROBOFLOW_LOCAL_PORT", "9145"))

import cv2
import numpy as np
from inference import get_model

print(f"Loading {MODEL_ID} ...")
MODEL = get_model(MODEL_ID, api_key=API_KEY)
print("Model loaded.")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "model": MODEL_ID})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/detect":
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", "0"))
        arr = cv2.imdecode(np.frombuffer(self.rfile.read(n), np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            self._send(400, {"error": "bad image"})
            return
        res = MODEL.infer(arr)[0]
        preds = [{"x": float(p.x), "y": float(p.y), "width": float(p.width),
                  "height": float(p.height), "class": p.class_name,
                  "confidence": float(p.confidence)} for p in res.predictions]
        self._send(200, {"predictions": preds})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"RF-DETR local server on http://localhost:{PORT}  (POST /detect with a JPEG body)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
