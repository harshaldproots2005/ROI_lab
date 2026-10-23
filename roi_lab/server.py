"""Browser page for the crown ROI clone. Does not import the machine project.

Open http://127.0.0.1:8765
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from pipeline import demo_frame, draw_pink_on_rect, run_crown_roi, suggest_ring

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
PORT = 8765


def _decode_image(raw: bytes):
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("That file is not an image OpenCV can read.")
    return image


def _png(image) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode the result image.")
    return buf.tobytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[roi_lab]", fmt % args)

    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict):
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            page = (STATIC / "index.html").read_bytes()
            self._send(200, page, "text/html; charset=utf-8")
            return
        self._json(404, {"error": "Not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            if path == "/api/demo":
                image, star, girdle = demo_frame()
                overlay, progress = run_crown_roi(image, star, girdle)
                self._json(200, {
                    "image_png_base64": _b64(_png(overlay)),
                    "progress": {str(k): round(v, 2) for k, v in progress.items()},
                    "star_points": [[round(p[0], 1), round(p[1], 1)] for p in star],
                    "girdle_points": [[round(p[0], 1), round(p[1], 1)] for p in girdle],
                })
                return
            if path == "/api/place":
                payload = json.loads(raw.decode("utf-8"))
                image = _decode_image(_unb64(payload.get("image_base64", "")))
                ratio = float(payload.get("star_ratio", 0.63))
                star, girdle, info = suggest_ring(image, ratio)
                self._json(200, {
                    "star_points": [[round(p[0], 1), round(p[1], 1)] for p in star],
                    "girdle_points": [[round(p[0], 1), round(p[1], 1)] for p in girdle],
                    "info": info,
                })
                return
            if path == "/api/line":
                payload = json.loads(raw.decode("utf-8"))
                image = _decode_image(_unb64(payload.get("image_base64", "")))
                overlay, ends = draw_pink_on_rect(image)
                self._json(200, {
                    "image_png_base64": _b64(_png(overlay)),
                    "line_ends": ends,
                })
                return
            if path != "/api/run":
                self._json(404, {"error": "Not found"})
                return
            payload = json.loads(raw.decode("utf-8"))
            image = _decode_image(_unb64(payload.get("image_base64", "")))
            overlay, progress = run_crown_roi(
                image,
                payload.get("star_points"),
                payload.get("girdle_points"),
                payload.get("previous_progress"),
            )
            self._json(200, {
                "image_png_base64": _b64(_png(overlay)),
                "progress": {str(k): round(v, 2) for k, v in progress.items()},
            })
        except Exception as exc:
            self._json(400, {"error": str(exc)})


def _b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    import base64
    if "," in text and text.strip().startswith("data:"):
        text = text.split(",", 1)[1]
    return base64.b64decode(text)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Crown ROI lab: http://127.0.0.1:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
