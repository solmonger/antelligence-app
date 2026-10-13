#!/usr/bin/env python3
"""d1-3B local-look calibration on development patches (goal 5 step 6, 4.5). DEV CASES ONLY.

    python scripts/v41_local_look_calib.py --url http://127.0.0.1:8522 --cache-dir <ssd>/locallook-cache \
        [--n 2000] [--seed 41] [--concurrency 2]

2,000 patches (n/40 per dev case; half centred in truth WT, half brain-but-outside), questioned exactly as the
harness does (``look_request``), cached by request hash so a rerun is free. Writes ``summary.json`` (aggregates only)
and ``config/v4_1/look_temperature.json``. The temperature rule: ECE (15 bins) > 0.10 -> fit on dev_a, report on dev_b
before/after, then refit on all; otherwise freeze 1.0.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from antelligence.experiments import v41_local_look as ll  # noqa: E402
from antelligence.perception.store import load_case_arrays  # noqa: E402
from antelligence.providers import decision_backends as db  # noqa: E402
from antelligence.worlds.tumor_mri.tuning import load_split  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BINS = 15
SEED = 41


def auroc(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)  # average ranks: ties count half
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reliability(p, y, bins: int = BINS):
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        s = idx == b
        rows.append({"bin": f"{b / bins:.3f}-{(b + 1) / bins:.3f}", "n": int(s.sum()),
                     "mean_p": round(float(p[s].mean()), 4) if s.any() else None,
                     "frac_tumour": round(float(y[s].mean()), 4) if s.any() else None})
    return rows


def scores(p, y):
    return {"n": len(p), "ece": round(ll.ece(p, y, BINS), 5), "brier": round(ll.brier(p, y), 5),
            "auroc": round(auroc(p, y), 5)}


def wt_view(a):
    """The store keeps truth as (3, X, Y, Z) in the order WT, TC, ET (perception/msd.truth_channels); use WT."""
    t = a["truth"]
    return {**a, "truth": t[0] if t.ndim == 4 else t}


def draw(split, n: int, seed: int):
    """Deterministic balanced samples over the 40 dev cases, plus the arrays (dev cases only are ever loaded)."""
    cases = sorted(split["dev"])
    arrays = {c: wt_view(load_case_arrays(c)) for c in cases}
    per = n // len(cases)
    samples = ll.sample_cases(arrays, frozenset(cases), per, seed)
    return samples, arrays


async def run(samples, arrays, backend, concurrency: int):
    sem = asyncio.Semaphore(concurrency)
    out = [None] * len(samples)

    async def one(i, s):
        patch = ll.local_patch(s.case, s.xyz, arrays=dict(arrays[s.case]))
        async with sem:
            r = await backend.decide(ll.look_request(backend.inner.model, patch))
        out[i] = (r.answers[ll.LOOK_QUESTION].p_yes, r.elapsed_s, int(r.timing.get("server_ms", 0)), r.cached,
                  r.identity)

    await asyncio.gather(*(one(i, s) for i, s in enumerate(samples)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8522")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--raw-out", help="write raw per-patch p/label here (keep off the repo)")
    ap.add_argument("--out", default=str(REPO / "docs/research/v4.1-20261013/locallook/summary.json"))
    ap.add_argument("--temp-out", default=str(REPO / "config/v4_1/look_temperature.json"))
    ap.add_argument("--versions", default="", help="JSON of runtime versions of the d1 server environment")
    a = ap.parse_args()

    split_path = REPO / "config/v4_1/split.json"
    split = load_split(split_path)
    samples, arrays = draw(split, a.n, a.seed)
    be = db.CachedDecision(db.D1Backend(a.url, retries=1, timeout_s=300), a.cache_dir)
    t0 = time.monotonic()
    res = asyncio.run(run(samples, arrays, be, a.concurrency))
    wall = time.monotonic() - t0
    p = np.array([r[0] for r in res])
    y = np.array([s.label for s in samples])
    case = np.array([s.case for s in samples])
    ids = {json.dumps(r[4], sort_keys=True) for r in res}
    assert len(ids) == 1, f"mixed identities: {ids}"
    identity = json.loads(next(iter(ids)))
    lat = np.array([r[1] for r in res]) * 1000
    srv = np.array([r[2] for r in res])

    overall = scores(p, y)
    rel = reliability(p, y)
    temp, method, split_report = 1.0, "none (ECE <= 0.10): temperature frozen at 1.0", None
    if overall["ece"] > ll.ECE_THRESHOLD:
        ia = np.isin(case, sorted(split["dev_a"]))
        t_a = ll.fit_temperature(p[ia], y[ia])
        pb = ll.apply_temperature(list(p[~ia]), t_a)
        split_report = {"fit_on": "dev_a", "n_fit": int(ia.sum()), "temperature_dev_a": round(t_a, 5),
                        "held_out": "dev_b", "n_held_out": int((~ia).sum()),
                        "before": scores(p[~ia], y[~ia]), "after": scores(pb, y[~ia])}
        temp = ll.fit_temperature(p, y)
        method = ("single temperature T minimising NLL of sigmoid(logit(p)/T) (bounded search over log T in "
                  "[ln 0.05, ln 20]); validated fit-on-dev_a / test-on-dev_b, then refit on all dev patches")
    pt = ll.apply_temperature(list(p), temp)
    after = scores(pt, y)

    h = hashlib.sha256()
    for s, pi in zip(samples, p):
        h.update(f"{s.case}|{s.xyz}|{s.label}|{pi:.9f}\n".encode())
    data_sha = h.hexdigest()
    summary = {
        "study": "v4.1", "step": "local-look calibration (goal 5 step 6)", "split": "dev only (dev_a + dev_b)",
        "model": identity, "hardware": "Apple M4 Max, MPS, bf16 (NOT the L40S pod). Assumption: same weights and "
        "revision, so probabilities should match the pod's d1 up to bf16/kernel numerics; not verified here.",
        "versions": json.loads(a.versions) if a.versions else {},
        "sampling": {"seed": a.seed, "n": len(samples), "per_case": a.n // 40, "cases": sorted(split["dev"]),
                     "n_tumour": int(y.sum()), "n_outside": int((1 - y).sum()), "patch": "64x64 axial FLAIR+T1c",
                     "question": ll.LOOK_INSTRUCTIONS},
        "ece_bins": BINS, "raw": overall, "reliability_raw": rel,
        "latency_ms": {"mean_wall": round(float(lat.mean()), 1), "median_wall": round(float(np.median(lat)), 1),
                       "p95_wall": round(float(np.percentile(lat, 95)), 1),
                       "mean_server": round(float(srv.mean()), 1), "n_cached_this_run": int(sum(r[3] for r in res)),
                       "note": "elapsed_s as stored when each reply was first computed; concurrency "
                       f"{a.concurrency}, wall-clock includes queueing behind the shim lock"},
        "run_wall_s": round(wall, 1),
        "temperature": {"value": round(temp, 5), "ece_threshold": ll.ECE_THRESHOLD, "refit": temp != 1.0,
                        "dev_a_dev_b_check": split_report, "method": method, "after_all_dev": after},
        "calibration_data_sha256": data_sha,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1) + "\n")
    cfg = {"study": "v4.1", "temperature": round(temp, 5), "method": method,
           "applies_to": "d1-3B local-look P(tumour): sigmoid(logit(p)/T)", "ece_bins": BINS,
           "measured": {"raw": overall, "after": after}, "n_patches": len(samples), "seed": a.seed,
           "model": identity, "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
           "sha256": data_sha, "sha256_of": "case|xyz|label|p_raw over the calibration patches, in sample order"}
    Path(a.temp_out).write_text(json.dumps(ll.seal_temperature_doc(cfg), indent=1) + "\n")
    if a.raw_out:
        np.savez_compressed(a.raw_out, p=p, y=y, case=case)
    print(json.dumps({"raw": overall, "after": after, "T": temp, "check": split_report,
                      "latency": summary["latency_ms"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
