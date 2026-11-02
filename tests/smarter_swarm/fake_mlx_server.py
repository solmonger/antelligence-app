"""A stand-in for ``mlx_lm.server`` in tests: same flags, same two routes, no model. Standard library only.

It refuses to start unless downloads are off (``HF_HUB_OFFLINE=1``) and the asked model has a snapshot in
``HF_HUB_CACHE``, which is how the real run is meant to start it. Like the real server, it reports the model as a
local path. ``FAKE_MLX_EXIT`` makes it exit at once with that code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    if os.environ.get("FAKE_MLX_EXIT"):
        return int(os.environ["FAKE_MLX_EXIT"])
    hub = Path(os.environ.get("HF_HUB_CACHE", ""))
    snaps = hub / ("models--" + a.model.replace("/", "--")) / "snapshots"
    if os.environ.get("HF_HUB_OFFLINE") != "1" or not snaps.is_dir():
        print(f"refusing: offline={os.environ.get('HF_HUB_OFFLINE')!r}, snapshots at {snaps}", flush=True)
        return 3
    served = str(next(snaps.iterdir()))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": a.model}]})
            return self._json(404, {})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if req.get("model") != a.model:
                return self._json(404, {"error": {"message": f"model {req.get('model')!r} not served"}})
            if req.get("chat_template_kwargs") != {"enable_thinking": False}:
                return self._json(400, {"error": {"message": "thinking was not turned off"}})
            seed = hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()
            p = round(random.Random(seed).random(), 2)
            content = json.dumps({"reason": "fake", "probability": p})
            return self._json(200, {"id": "x", "model": served,
                                    "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": 10, "completion_tokens": 10}})

    HTTPServer((a.host, a.port), H).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
