#!/usr/bin/env python3
"""HTTP shim for Liquid d1-3B: ``POST /v1/systemone`` -> ``model.system_one_batch`` (goal 4.5, brief S3).

Written to run in its own ``uv`` venv (transformers 5.14+, torch, Pillow), NOT the repo venv, and NOT run in CI:
the model is loaded only by :func:`main`; :func:`handle_request` is pure given a loaded model, so tests drive it
with a fake. Standard library only for serving.

Remote code. ``trust_remote_code=True`` executes the repo's Python (``api.py``, ``runner.py``, ``prompt.py``,
``hybrid.py``, ``lfm2_vl.py``, ``modeling_d1.py``). All six were read at the pinned revision before this was
written (2026-10-12; logged in the S3 report). The revision is pinned below and verified against the loaded config.

    python scripts/v41_d1_server.py --host 127.0.0.1 --port 8520          # optional: V41_D1_TOKEN=... for bearer auth

Wire format: the body of ``DecisionRequest.payload()`` (``model``, ``state``, ``questions`` and, for local look,
``images``: base64 PNG strings). Reply: ``{"model", "identity", "answers", "usage", "timing"}``; ``answers`` is d1's
``system_one`` output untouched, ``identity`` is ``{served_model, revision, runtime, runtime_version}``.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Mapping, Optional

D1_MODEL = "LiquidAI/d1-3B"
D1_REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
MAX_BODY_BYTES = 4_000_000
MAX_IMAGES = 4
MAX_QUESTIONS = 8
QUESTION_TYPES = ("noul", "choice", "score")


class BadRequest(ValueError):
    """The request breaks the wire contract (HTTP 400)."""


def decode_images(images: Any) -> List[Any]:
    """base64 PNG strings -> PIL images (Pillow is imported here, not at module load)."""
    if not isinstance(images, list) or len(images) > MAX_IMAGES:
        raise BadRequest(f"images must be a list of at most {MAX_IMAGES} base64 strings")
    if not images:
        return []
    from PIL import Image

    out = []
    for item in images:
        try:
            out.append(Image.open(io.BytesIO(base64.b64decode(item, validate=True))).convert("RGB"))
        except (binascii.Error, TypeError, ValueError, OSError) as exc:
            raise BadRequest(f"image is not a base64 PNG: {type(exc).__name__}") from None
    return out


def validate(payload: Any, served_model: str) -> None:
    if not isinstance(payload, dict) or payload.get("model") != served_model:
        raise BadRequest(f"model must be {served_model!r}")
    if not isinstance(payload.get("state"), str) or not payload["state"]:
        raise BadRequest("state must be a non-empty string")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not 0 < len(questions) <= MAX_QUESTIONS:
        raise BadRequest(f"questions must be an object of 1..{MAX_QUESTIONS} questions")
    for qid, q in questions.items():
        if not isinstance(q, dict) or q.get("type") not in QUESTION_TYPES or not isinstance(q.get("instructions"), str):
            raise BadRequest(f"question {qid!r} is malformed")


def handle_request(model: Any, identity: Mapping[str, str], payload: Any) -> Dict[str, Any]:
    """One request -> one reply, through ``model.system_one_batch`` (one forward pass for all questions)."""
    validate(payload, identity["served_model"])
    images = decode_images(payload.get("images", []))
    start = time.perf_counter()
    result = model.system_one_batch([(payload["state"], payload["questions"], images)])[0]
    server_ms = int(round((time.perf_counter() - start) * 1000))
    return {"model": identity["served_model"], "identity": dict(identity), "answers": result["answers"],
            "usage": result["usage"], "timing": {"server_ms": server_ms, "queue_ms": 0}}


def make_handler(model: Any, identity: Mapping[str, str], lock: threading.Lock, token: Optional[str]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:  # no request text in logs
            return

        def _send(self, code: int, body: Mapping[str, Any]) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _authorised(self) -> bool:
            return token is None or hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}")

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True, "identity": dict(identity)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            if not self._authorised():
                return self._send(401, {"error": "unauthorised"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY_BYTES:
                    raise BadRequest("body size out of range")
                payload = json.loads(self.rfile.read(length))
                queued = time.perf_counter()
                with lock:  # one forward at a time; the model is not thread-safe
                    waited = int(round((time.perf_counter() - queued) * 1000))
                    reply = handle_request(model, identity, payload)
                reply["timing"]["queue_ms"] = waited
                self._send(200, reply)
            except (BadRequest, ValueError) as exc:
                self._send(400, {"error": str(exc)[:200]})
            except Exception as exc:  # the caller retries 5xx; never echo internals
                self._send(500, {"error": type(exc).__name__})

    return Handler


def load_model(revision: str = D1_REVISION):
    """Load d1-3B at the pinned revision (remote code enabled) and return (model, identity)."""
    import torch
    import transformers
    from transformers import AutoModel

    model = AutoModel.from_pretrained(D1_MODEL, revision=revision, trust_remote_code=True, dtype=torch.bfloat16)
    loaded = getattr(model.config, "_commit_hash", None)
    if loaded not in (None, revision):
        raise SystemExit(f"loaded revision {loaded} != pinned {revision}")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    identity = {"served_model": D1_MODEL, "revision": revision, "runtime": "transformers",
                "runtime_version": transformers.__version__}
    return model, identity


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8520)
    ap.add_argument("--revision", default=D1_REVISION)
    args = ap.parse_args(argv)
    model, identity = load_model(args.revision)
    token = os.environ.get("V41_D1_TOKEN") or None
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, identity, threading.Lock(), token))
    print(f"d1 shim on {args.host}:{args.port} identity={json.dumps(identity)}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
