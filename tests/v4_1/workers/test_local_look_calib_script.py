"""The calibration script's sampler: 2,000 patches, balanced, dev-only, deterministic. Synthetic arrays only."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from antelligence.experiments.v41_local_look import SplitError, dev_cases, sample_cases

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("v41_local_look_calib", REPO / "scripts" / "v41_local_look_calib.py")
calib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calib)

SPLIT = json.loads((REPO / "config" / "v4_1" / "split.json").read_text())


def fake(case_seed, n=20):
    rng = np.random.default_rng(case_seed)
    g = np.indices((n, n, n))
    brain = ((g[0] - n / 2) ** 2 + (g[1] - n / 2) ** 2 + (g[2] - n / 2) ** 2) <= (n / 2 - 1) ** 2
    wt = ((g[0] - 10) ** 2 + (g[1] - 10) ** 2 + (g[2] - 10) ** 2 <= 16) & brain
    truth3 = np.stack([wt, wt & (g[0] > 10), wt & (g[1] > 12)])  # store layout: WT, TC, ET
    return calib.wt_view({"brain_mask": brain, "truth": truth3, "flair": rng.random((n, n, n), dtype=np.float32),
                          "t1c": rng.random((n, n, n), dtype=np.float32)})


def test_sampler_balanced_dev_only_deterministic():
    cases = sorted(SPLIT["dev_a"] + SPLIT["dev_b"])
    assert len(cases) == 40
    arrays = {c: fake(i) for i, c in enumerate(cases)}
    dev = dev_cases(REPO / "config" / "v4_1" / "split.json")
    s1 = sample_cases(arrays, dev, 50, calib.SEED)  # 2000 / 40
    assert len(s1) == 2000
    assert sum(s.label for s in s1) == 1000  # balanced overall
    for c in cases:  # spread: every dev case gives 25 inside + 25 outside
        mine = [s for s in s1 if s.case == c]
        assert len(mine) == 50 and sum(s.label for s in mine) == 25
    for s in s1:  # labels match the WT channel (index 0), centres inside the brain
        assert bool(arrays[s.case]["truth"][s.xyz]) == bool(s.label) and arrays[s.case]["brain_mask"][s.xyz]
    assert s1 == sample_cases(arrays, dev, 50, calib.SEED)  # deterministic
    assert s1 != sample_cases(arrays, dev, 50, calib.SEED + 1)
    assert {s.case for s in s1} <= set(cases)  # dev only
    with pytest.raises(SplitError):
        sample_cases({**arrays, SPLIT["eval"][0]: fake(1)}, dev, 50, calib.SEED)
    with pytest.raises(SplitError):
        sample_cases({SPLIT["reserve"][0]: fake(1)}, dev, 50, calib.SEED)


def test_auroc_and_reliability():
    assert calib.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert calib.auroc([0.5, 0.5], [0, 1]) == 0.5
    rows = calib.reliability([0.0, 1.0, 0.5], [0, 1, 1])
    assert len(rows) == calib.BINS and sum(r["n"] for r in rows) == 3
