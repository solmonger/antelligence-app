#!/usr/bin/env python3
"""An Ollama-shaped front for ``mlx_lm.server``, so the shared-read harness can switch MLX models safely.

``mlx_lm.server`` serves one model and can't unload it, so the harness's memory guard would see the last model still
in memory at every model change. This stand-in sits where the harness expects Ollama (port 11434) and:

- ``GET /v1/models``: lists the models in the model file;
- ``POST /v1/chat/completions``: starts ``mlx_lm.server --model <the asked model>`` on a private port if it isn't
  already serving that model (stopping the one it started before), waits until it answers, and passes the call on;
- ``POST /api/generate`` with ``"keep_alive": 0`` (the harness's unload call): stops the server it started, so the
  memory really comes back before the guard checks for the next model.

It only ever stops the one server process it started itself. Standard library only. Run 1 used a stand-in like this
on the Mac (docs/plans/2026-10-30-shared-read.md, "How run 1 ran"); this one lives in the repo.

    python scripts/mlx_standin.py --models docs/research/fresh-20261031/models.json \\
        --mlx-server "$(which mlx_lm.server)"
"""

from __future__ import annotations

import argparse
import json
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional, Sequence

MAX_BODY = 8_000_000


class Backend:
    """The one ``mlx_lm.server`` this stand-in owns."""

    def __init__(self, cmd: Sequence[str], port: int, start_timeout_s: float = 1800.0, log: Optional[Path] = None):
        self.cmd, self.port, self.start_timeout_s, self.log = list(cmd), port, start_timeout_s, log
        self.proc: Optional[subprocess.Popen] = None
        self.model: Optional[str] = None
        self.lock = threading.Lock()
        self.inflight = 0
        self.idle = threading.Condition(self.lock)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _alive(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base}/v1/models", timeout=5) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def stop(self) -> None:
        """Stop the server this stand-in started, if any. Caller holds the lock."""
        while self.inflight:
            self.idle.wait()
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=30)
            print(f"[standin] stopped {self.model} (pid {self.proc.pid})", flush=True)
        self.proc, self.model = None, None

    def ensure(self, model: str) -> None:
        """Serve ``model``, switching from another one if needed. Caller holds the lock."""
        if self.model == model and self.proc is not None and self.proc.poll() is None:
            return
        self.stop()
        out = open(self.log, "a") if self.log else subprocess.DEVNULL
        cmd = self.cmd + ["--model", model, "--host", "127.0.0.1", "--port", str(self.port)]
        self.proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
        self.model = model
        print(f"[standin] started {model} (pid {self.proc.pid})", flush=True)
        deadline = time.monotonic() + self.start_timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                code = self.proc.returncode
                self.proc, self.model = None, None
                raise RuntimeError(f"mlx_lm.server for {model} exited with code {code}")
            if self._alive():
                return
            time.sleep(1.0)
        self.stop()
        raise RuntimeError(f"mlx_lm.server for {model} did not answer within {self.start_timeout_s:.0f}s")


def make_handler(backend: Backend, models: List[str]):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet: the harness prints its own progress
            pass

        def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj) -> None:
            self._send(status, json.dumps(obj).encode())

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") in ("/v1/models", "/api/tags"):
                return self._json(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in models]})
            return self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._json(413, {"error": "body too large"})
            raw = self.rfile.read(n)
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                return self._json(400, {"error": "bad json"})
            if self.path.rstrip("/") == "/api/generate":
                if body.get("keep_alive") == 0:
                    with backend.lock:
                        backend.stop()
                return self._json(200, {"done": True})
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            model = str(body.get("model") or "")
            if model not in models:
                return self._json(404, {"error": {"message": f"model '{model}' not found"}})
            with backend.lock:
                try:
                    backend.ensure(model)
                except RuntimeError as exc:
                    return self._json(503, {"error": {"message": str(exc)}})
                backend.inflight += 1
            try:
                req = urllib.request.Request(f"{backend.base}/v1/chat/completions", data=raw,
                                             headers={"Content-Type": "application/json"}, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=1800) as r:
                        return self._send(r.status, r.read())
                except urllib.error.HTTPError as exc:
                    return self._send(exc.code, exc.read())
                except (urllib.error.URLError, OSError) as exc:
                    return self._json(503, {"error": {"message": f"mlx_lm.server: {exc}"}})
            finally:
                with backend.lock:
                    backend.inflight -= 1
                    backend.idle.notify_all()

    return Handler


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", type=Path, required=True, help="the harness's model file (reader and ants)")
    ap.add_argument("--mlx-server", default="mlx_lm.server", help="the mlx_lm.server command (a path, or a command line)")
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--mlx-port", type=int, default=8090)
    ap.add_argument("--log", type=Path, default=None, help="where mlx_lm.server's own output goes")
    a = ap.parse_args(argv)
    cfg = json.loads(a.models.read_text())
    models = [cfg["reader"]["tag"]] + [m["tag"] for m in cfg["ants"]]
    backend = Backend(shlex.split(a.mlx_server), a.mlx_port, log=a.log)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(backend, models))

    def bye(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    print(f"[standin] on 127.0.0.1:{a.port} for {len(models)} models; mlx_lm.server on port {a.mlx_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        with backend.lock:
            backend.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
