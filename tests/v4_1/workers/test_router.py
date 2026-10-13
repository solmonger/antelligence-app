"""kNN router: labels from S', ties to the cheaper model, frozen save/load."""
from __future__ import annotations

import numpy as np
import pytest

from antelligence.experiments.v41_router import (BONSAI_ROUTER_MODELS, FEATURE_NAMES, K, KNNRouter, RouterError,
                                                 TrainRow, best_model, fit_router, make_rows, match_count,
                                                 nearest_level, router_features, sprime_labels)
from antelligence.kernel.decision import Answer

CANDS = [{"label": "A", "centroid_vox": (10, 10, 10), "p_tumour": 0.9, "dist_mm": 20.0},
         {"label": "B", "centroid_vox": (30, 10, 10), "p_tumour": 0.8, "dist_mm": 44.0}]


def dec(target, release, amount):
    return {"target": target, "release": release, "amount": amount}


def test_router_labels_from_sprime():
    # S' walks to a voxel 2 voxels (4 mm) from candidate B: label B; release 0 -> no; amount from the fraction
    intent = {"target": "X", "release": 0, "amount": 3, "target_vox": [28, 10, 10]}
    assert sprime_labels(intent, CANDS) == {"target": "B", "release": "no", "amount": "3"}
    # far from every candidate (> 10 mm): explore
    assert sprime_labels({**intent, "target_vox": [60, 60, 60]}, CANDS)["target"] == "X"
    # S' stays and releases a fraction: nearest level (0.10, 0.25, 0.50, 0.75, 1.00)
    stay = {"target": "S", "release": 1, "amount_fraction": 0.6}
    assert sprime_labels(stay, CANDS) == {"target": "S", "release": "yes", "amount": "3"}
    assert nearest_level(0.65) == 4 and nearest_level(0.375) == 2 and nearest_level(0.0) == 1 and nearest_level(2.0) == 5
    labels = {"target": "B", "release": "no", "amount": "3"}
    assert match_count(dec("B", 0, 3), labels) == 3 and match_count(dec("A", 1, 3), labels) == 1
    # the router label is the cheapest model with the most matches
    decisions = {"d1": dec("A", 0, 3), "qwen3.5-4b": dec("B", 0, 3), "qwen3.5-9b": dec("B", 0, 3)}
    assert best_model(decisions, labels) == "qwen3.5-4b"  # 4B and 9B both score 3; the cheaper wins
    decisions["d1"] = dec("B", 0, 3)
    assert best_model(decisions, labels) == "d1"
    decisions = {"d1": dec("A", 1, 1), "qwen3.5-4b": dec("A", 1, 1), "qwen3.5-9b": dec("B", 1, 1)}
    assert best_model(decisions, labels) == "qwen3.5-9b"  # 9B alone scores 2 of 3


def test_router_features_shape():
    answers = {"target": Answer("choice", "A", {"A": 0.7, "B": 0.1, "C": 0.1, "S": 0.05, "X": 0.05}, 0.7),
               "release": Answer("noul", 0.9, {"yes": 0.9, "no": 0.1}),
               "amount": Answer("score", 2.0, {"0": 0.0, "1": 0.2, "2": 0.6, "3": 0.2, "4": 0.0}, 0.6)}
    fields = {"local_unc": 0.3, "payload_fraction": 0.5, "round": 4, "candidates": CANDS}
    f = router_features(fields, answers)
    assert len(f) == len(FEATURE_NAMES) == 10
    assert f[0] == 0.7 and f[2] == 0.9 and f[4] == 0.6 and f[6:] == [0.3, 0.5, 20.0, 4.0]
    assert f[1] > 0 and f[3] == pytest.approx(-(0.9 * np.log(0.9) + 0.1 * np.log(0.1)))
    stay = {**answers, "target": Answer("choice", "S", {"A": 0.1, "B": 0.1, "C": 0.1, "S": 0.6, "X": 0.1}, 0.6)}
    assert router_features(fields, stay)[8] == 0.0


def rows_with_labels(counts, spread=0.0):
    rng = np.random.default_rng(0)
    out = []
    for label, n in counts.items():
        for _ in range(n):
            out.append(TrainRow(list(rng.normal(0, spread, len(FEATURE_NAMES))), label))
    return out


def test_router_knn_ties_to_cheaper():
    # 15 identical points, 5 per model: every neighbour is at distance 0, votes tie 5/5/5 -> the cheapest (d1)
    rows = rows_with_labels({"qwen3.5-9b": 5, "qwen3.5-4b": 5, "d1": 5})
    r = KNNRouter.fit(rows, k=K)
    assert r.k == 15 and r.predict([0.0] * 10) == "d1"
    # tie between the two costlier models only -> 4B
    r2 = KNNRouter.fit(rows_with_labels({"qwen3.5-9b": 4, "qwen3.5-4b": 4, "d1": 1}), k=K)
    assert r2.predict([0.0] * 10) == "qwen3.5-4b"
    # plurality still wins when not tied
    r3 = KNNRouter.fit(rows_with_labels({"qwen3.5-9b": 8, "qwen3.5-4b": 4, "d1": 3}), k=K)
    assert r3.predict([0.0] * 10) == "qwen3.5-9b"
    # k is capped by the training size; features are standardised (a far feature does not dominate by scale)
    assert KNNRouter.fit(rows_with_labels({"d1": 3}), k=K).k == 3
    a = [TrainRow([0.0] * 9 + [0.0], "d1")] * 5 + [TrainRow([0.0] * 9 + [100.0], "qwen3.5-9b")] * 5
    r4 = KNNRouter.fit(a, k=5)
    assert r4.predict([0.0] * 9 + [90.0]) == "qwen3.5-9b" and r4.predict([0.0] * 9 + [5.0]) == "d1"
    # a tie in the TRAINING labels goes to the cheaper model as well (make_rows)
    d = {"d1": dec("A", 0, 1), "qwen3.5-4b": dec("A", 0, 1), "qwen3.5-9b": dec("A", 0, 1)}
    assert make_rows([([0.0] * 10, d, {"target": "A", "release": "no", "amount": "1"})])[0].label == "d1"


def test_router_freeze_roundtrip_and_bonsai_variant(tmp_path):
    rows = rows_with_labels({"d1": 6, "bonsai-8b": 6}, spread=1.0)
    r = fit_router(rows, bonsai=True)
    assert r.models == BONSAI_ROUTER_MODELS
    digest = r.save(tmp_path / "router.json")
    again = KNNRouter.load(tmp_path / "router.json", expect_sha256=digest)
    probes = np.random.default_rng(1).normal(0, 1, (20, 10))
    assert [again.predict(p) for p in probes] == [r.predict(p) for p in probes] and again.sha256 == digest
    (tmp_path / "router.json").write_text((tmp_path / "router.json").read_text().replace('"k":12', '"k":11'))
    with pytest.raises(RouterError):
        KNNRouter.load(tmp_path / "router.json")
    with pytest.raises(RouterError):
        fit_router(rows_with_labels({"qwen3.5-9b": 2}), bonsai=True)  # label outside this router's models
    with pytest.raises(RouterError):
        KNNRouter.fit([])
