"""Fixtures for the harness tests: a synthetic phantom case, fake backends, a fake queen, a frozen router and
thresholds, and a mock anchoring lab. No network, no keys, no paid calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pytest

from antelligence.anchoring.tamper import Lab
from antelligence.experiments.v41_conformal import ConformalThresholds
from antelligence.experiments.v41_harness import (BONSAI, D1, FW, Q4, Q9, KeyVault, V41Harness, Worker)
from antelligence.experiments.v41_router import (BONSAI_ROUTER_MODELS, FEATURE_NAMES, ROUTER_MODELS, KNNRouter,
                                                 TrainRow)
from antelligence.kernel.decision import Answer
from antelligence.providers.decision_backends import FakeBackend, hashed_answers
from antelligence.providers.wrappers import FakeProvider
from antelligence.worlds.tumor_mri import WorldConfig
from tests.v4_1.world.phantom import make_phantom, phantom_config

CASE_DEV = "BRATS_DEV_001"
CASE_EVAL = "BRATS_EVAL_001"
LOOK_P = {"value": 0.05}


def small_cfg(**kw) -> WorldConfig:
    return phantom_config(**kw)


def make_loader(calls: Optional[List[tuple]] = None, with_images: bool = True):
    """The injected ``load_case`` (stands in for L2c's adapter): the phantom crop plus its regions."""
    def load_case(case_id: str, variant: str = "full"):
        if calls is not None:
            calls.append((case_id, variant))
        crop = make_phantom(flair=with_images)
        if variant == "missing_flair" and crop.flair is not None:
            crop = type(crop)(**{**crop.__dict__, "flair": np.zeros_like(crop.flair)})
        return crop, None
    return load_case


def d1_responder(look_p: float = 0.05, other: Callable = hashed_answers):
    def respond(request):
        if "look" in request.questions:
            return {"look": Answer("noul", look_p, {"yes": look_p, "no": 1.0 - look_p})}
        return other(request)
    return respond


def fake_workers(calls: Optional[List[str]] = None, d1_look: float = 0.05, d1_answers=None,
                 q4_answers=None, latency_s: float = 0.0) -> Dict[str, Worker]:
    def tracked(name, responder):
        def fn(request):
            if calls is not None:
                calls.append(name)
            return responder(request)
        return fn
    d1 = FakeBackend(tracked(D1, d1_responder(d1_look, d1_answers or hashed_answers)), model="d1-fake", name=D1,
                     latency_s=latency_s)
    q4 = FakeBackend(tracked(Q4, q4_answers or hashed_answers), model="q4-fake", name=Q4, latency_s=latency_s)
    q9 = FakeBackend(tracked(Q9, hashed_answers), model="q9-fake", name=Q9, latency_s=latency_s)
    bonsai = FakeBackend(tracked(BONSAI, hashed_answers), model="bonsai-fake", name=BONSAI)
    fw = FakeBackend(tracked(FW, hashed_answers), model="fw-fake", name=FW)
    return {D1: Worker(d1), Q4: Worker(q4), Q9: Worker(q9), BONSAI: Worker(bonsai), FW: Worker(fw)}


def frozen_router(models=ROUTER_MODELS, k=5) -> KNNRouter:
    rng = np.random.default_rng(3)
    rows = [TrainRow(list(rng.random(len(FEATURE_NAMES))), models[i % len(models)]) for i in range(30)]
    return KNNRouter.fit(rows, models, k)


def thresholds(arm="SW-B", target=Q9, qhat=0.45) -> ConformalThresholds:
    per = {q: {"alpha": 0.2, "qhat": qhat, "n": 100, "dev_defer_rate": 0.2, "on_target": True}
           for q in ("target", "release", "amount")}
    return ConformalThresholds(arm, target, per, 0.2, 0.2, True)


def queen_reply(request) -> str:
    """A valid queen reply built from the regions and squads in the prompt (the fake queen reads what the real one sees)."""
    body = json.loads(request.messages[1]["content"])
    ids = [r["region_id"] for r in body["regions"]]
    routes = [{"squad": s["squad"], "region_ids": ids[s["squad"] % max(1, len(ids)):][:2] or ids[:1], "priority": 2}
              for s in body["squads"]]
    return json.dumps({"plan": "split the squads", "guesses": ["core is the largest region"], "routes": routes,
                       "candidates": ids[:6]})


def queen_provider(responder: Callable = queen_reply) -> FakeProvider:
    return FakeProvider(responder, model="fake-queen")


def allow_all(case: str) -> str:
    return "eval" if case == CASE_EVAL else "dev"


@dataclass
class Rig:
    harness: V41Harness
    lab: Lab
    calls: List[str] = field(default_factory=list)
    loads: List[tuple] = field(default_factory=list)
    gate_calls: List[str] = field(default_factory=list)


def build_rig(tmp_path: Path, *, gate=None, d1_look=0.05, d1_answers=None, q4_answers=None, vault=None, perception_stats=None, post_mode_factory=None,
              reputation_client=None, queen=None, workers=None, anchored=True, **kw) -> Rig:
    lab = Lab()
    calls: List[str] = []
    loads: List[tuple] = []
    gate_calls: List[str] = []

    def gate_fn(case: str) -> str:
        gate_calls.append(case)
        return (gate or allow_all)(case)

    h = V41Harness(
        load_case=make_loader(loads), gate=gate_fn, workers=workers or fake_workers(calls, d1_look, d1_answers, q4_answers),
        queen_provider=queen or queen_provider(), world_config=small_cfg(),
        router=frozen_router(), bonsai_router=frozen_router(BONSAI_ROUTER_MODELS), thresholds=thresholds(),
        bonsai_thresholds=thresholds("SW-Bonsai", BONSAI),
        anchorer_factory=(lambda board, akey, out_dir: lab.anchorer(board, akey, out_dir)) if anchored else None,
        allow_unanchored=not anchored, post_mode_factory=post_mode_factory, reputation_client=reputation_client,
        keys=vault or KeyVault(), out_dir=tmp_path / "out", perception_stats=perception_stats, **kw)
    return Rig(h, lab, calls, loads, gate_calls)


@pytest.fixture
def rig(tmp_path):
    return build_rig(tmp_path)
