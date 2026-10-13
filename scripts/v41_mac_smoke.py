#!/usr/bin/env python3
"""One-off Mac smoke for the local workers (docs/v4.1/MAC-MODELS.md). Synthetic state only; no MRI data.

    python scripts/v41_mac_smoke.py bonsai [--url http://127.0.0.1:8521]
    python scripts/v41_mac_smoke.py d1 [--url http://127.0.0.1:8522] [-n 5]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from antelligence.kernel.decision import DecisionRequest  # noqa: E402
from antelligence.providers import decision_backends as db  # noqa: E402
from antelligence.worlds.tumor_mri.questions import build_questions  # noqa: E402

BONSAI_HF = "prism-ml/Ternary-Bonsai-8B-gguf"
BONSAI_REV = "c2aefbeb4b24469cd11579c3384b990404c17a30"


def requests(model: str, n: int, with_images: bool):
    rng = np.random.default_rng(7)
    for i in range(n):
        images = db.patch_to_images(rng.random((64, 64, 2)).astype(np.float32)) if with_images else ()
        state = (f"Round {i + 1} of 24, tick 0. Payload left: {100 - 7 * i}%. Local estimate: uncertain "
                 f"{0.3 + 0.1 * i:.1f}. Nearest candidate A at 12 voxels.")
        yield DecisionRequest(model=model, state=state, seed=i, questions=build_questions(("A", "B", "C")),
                              images=tuple(images))


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("bonsai", "d1"))
    ap.add_argument("--url")
    ap.add_argument("-n", type=int, default=None)
    ap.add_argument("--api-version", default=None)
    a = ap.parse_args()
    if a.which == "bonsai":
        be = db.LlamaCppBackend(a.url or "http://127.0.0.1:8521", BONSAI_HF, BONSAI_REV, retries=0, timeout_s=300)
        reqs = list(requests(be.model, a.n or 1, False))
        schema = db.answer_schema(reqs[0])
    else:
        be = db.D1Backend(a.url or "http://127.0.0.1:8522", retries=0, timeout_s=300)
        reqs = list(requests(be.model, a.n or 5, True))
    for i, r in enumerate(reqs):
        t = time.monotonic()
        resp = await be.decide(r)
        print(json.dumps({"i": i, "wall_s": round(time.monotonic() - t, 3), "identity": resp.identity,
                          "prob_method": resp.prob_method, "degraded": resp.degraded, "timing": resp.timing,
                          "tokens": [resp.input_tokens, resp.output_tokens], "images": len(r.images),
                          "answers": {q: ans.to_dict() for q, ans in resp.answers.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
