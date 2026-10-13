"""Local-look sampler (balanced, dev only) and calibration scores. Synthetic arrays only."""
from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import numpy as np
import pytest

from antelligence.experiments.v41_local_look import (SplitError, apply_temperature, brier, calibration_report, dev_cases,
                                                     ece, fit_temperature, look_request, measure, patches,
                                                     sample_cases, sample_centres)
from antelligence.providers.decision_backends import FakeBackend
from antelligence.kernel.decision import Answer

REPO = Path(__file__).resolve().parents[3]


def arrays(seed=0, n=24):
    rng = np.random.default_rng(seed)
    g = np.indices((n, n, n))
    brain = ((g[0] - n / 2) ** 2 + (g[1] - n / 2) ** 2 + (g[2] - n / 2) ** 2) <= (n / 2 - 1) ** 2
    truth = ((g[0] - 12) ** 2 + (g[1] - 12) ** 2 + (g[2] - 12) ** 2 <= 9) & brain
    return {"brain_mask": brain, "truth": truth, "flair": rng.random((n, n, n)).astype(np.float32),
            "t1c": rng.random((n, n, n)).astype(np.float32), "p_tumour": truth.astype(np.float32), "unc": truth * 0.5}


def test_local_look_balanced_sampler_dev_only():
    a = arrays()
    picks = sample_centres(a["truth"], a["brain_mask"], 20, seed=5)
    assert len(picks) == 20 and sum(l for _, l in picks) == 10  # half truth centres, half outside
    for (x, y, z), label in picks:
        assert bool(a["truth"][x, y, z]) == bool(label) and a["brain_mask"][x, y, z]
    assert picks == sample_centres(a["truth"], a["brain_mask"], 20, seed=5)  # seeded
    assert picks != sample_centres(a["truth"], a["brain_mask"], 20, seed=6)
    assert len({xyz for xyz, _ in picks}) == 20  # no repeats
    odd = sample_centres(a["truth"], a["brain_mask"], 7, seed=1)
    assert sum(l for _, l in odd) == 3 and len(odd) == 7  # the extra one is outside
    tiny = sample_centres(a["truth"], a["brain_mask"], 10_000, seed=1)  # a side with fewer voxels gives all it has
    assert sum(l for _, l in tiny) == int((a["truth"] & a["brain_mask"]).sum())

    # dev cases come from the committed split; eval/reserve ids are refused before any array is read
    dev = dev_cases(REPO / "config" / "v4_1" / "split.json")
    assert len(dev) == 40
    import json
    split = json.loads((REPO / "config" / "v4_1" / "split.json").read_text())
    dev_id, eval_id, reserve_id = split["dev_a"][0], split["eval"][0], split["reserve"][0]
    samples = sample_cases({dev_id: a}, dev, 10, seed=3)
    assert len(samples) == 10 and {s.case for s in samples} == {dev_id}
    for held in (eval_id, reserve_id):
        class Boom(dict):
            def __getitem__(self, k):
                raise AssertionError("arrays of a held-out case were read")

        with pytest.raises(SplitError):
            sample_cases({dev_id: a, held: Boom()}, dev, 10, seed=3)
    # patches come from perception.store.local_patch: (64, 64, 2)
    first = next(patches(samples, {dev_id: a}))
    assert first.shape == (64, 64, 2)


def test_look_request_and_measure():
    patch = np.random.default_rng(1).random((64, 64, 2)).astype(np.float32)
    req = look_request("LiquidAI/d1-3B", patch)
    assert len(req.images) == 2 and base64.b64decode(req.images[0])[:4] == b"\x89PNG"
    assert req.questions["look"].type == "noul" and "images" in req.payload()
    fake = FakeBackend(lambda r: {"look": Answer("noul", 0.8, {"yes": 0.8, "no": 0.2})}, model="LiquidAI/d1-3B")
    assert asyncio.run(measure(fake, "LiquidAI/d1-3B", iter([patch, patch]))) == [0.8, 0.8]


def test_scores_and_temperature_refit():
    y = np.array([1] * 50 + [0] * 50)
    perfect = np.where(y == 1, 0.999, 0.001)
    assert ece(perfect, y) < 0.01 and brier(perfect, y) < 1e-5
    # overconfident detector: says 0.95 / 0.05 but is right 70% of the time -> ECE > 0.10 -> refit softens (T > 1)
    rng = np.random.default_rng(0)
    right = rng.random(2000) < 0.7
    yy = rng.integers(0, 2, 2000)
    pred = np.where(right, yy, 1 - yy)
    p = np.where(pred == 1, 0.95, 0.05)
    rep = calibration_report(p, yy)
    assert rep["ece"] > 0.10 and rep["refit"] and rep["temperature"] > 1.5
    assert rep["ece_after"] < rep["ece"] and rep["brier_after"] < rep["brier"] and rep["ece_after"] < 0.05
    assert apply_temperature(0.95, rep["temperature"]) == pytest.approx(0.7, abs=0.05)
    # already calibrated: no refit, T = 1
    good = calibration_report(np.where(pred == 1, 0.7, 0.3), yy)
    assert not good["refit"] and good["temperature"] == 1.0 and good["ece_after"] == good["ece"]
    assert apply_temperature(0.5, 3.0) == pytest.approx(0.5)
    assert fit_temperature(p, yy) == pytest.approx(rep["temperature"])
    with pytest.raises(ValueError):
        ece([0.5], [1, 0])
    with pytest.raises(ValueError):
        apply_temperature(0.5, 0.0)
