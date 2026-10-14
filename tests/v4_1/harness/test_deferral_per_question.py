"""Per-question deferral in the harness, the logged state, and --log-counterfactuals all (step 9). No network, no keys."""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pytest

from antelligence.experiments import v41_fit
from antelligence.experiments.v41_conformal import ConformalThresholds
from antelligence.experiments.v41_harness import (D1, FW, Q4, Q9, BONSAI, HarnessError, KeyVault, V41Harness, Worker)
from antelligence.experiments.v41_router import ROUTER_MODELS, KNNRouter, TrainRow, FEATURE_NAMES
from antelligence.kernel.decision import Answer
from antelligence.providers.decision_backends import FakeBackend, hashed_answers
from tests.v4_1.harness.conftest import (CASE_DEV, CASE_EVAL, allow_all, d1_responder, make_loader, queen_provider,
                                         small_cfg)

N_DEC = 8 * small_cfg().rounds


def constant(target=None, release=0.9, level=3):
    """A responder whose answers are the same for every request: target letter (default: the last offered), P(yes), level."""
    def respond(request):
        opts = request.questions["target"].options
        t = target if target in opts else opts[0]
        out = {"target": Answer("choice", t, {o: (0.9 if o == t else 0.1 / (len(opts) - 1)) for o in opts}, 0.9),
               "release": Answer("noul", release, {"yes": release, "no": 1 - release}),
               "amount": Answer("score", level - 1, {str(i): (0.8 if i == level - 1 else 0.05) for i in range(5)}, 0.8)}
        return out
    return respond


def only_d1_router():
    rng = np.random.default_rng(1)
    return KNNRouter.fit([TrainRow(list(rng.random(len(FEATURE_NAMES))), D1) for _ in range(30)], ROUTER_MODELS, 5)


def all_defer(deferral, target_model=Q9) -> ConformalThresholds:
    """qhat = inf puts every label in every set, so every deferring question defers."""
    per = {q: {"alpha": 0.2, "qhat": math.inf, "n": 10, "dev_defer_rate": 1.0, "on_target": False}
           for q in ("target", "release", "amount")}
    info = {q: {"model": m, "defers": m is not None, "match": {}, "d1_match": 0.0, "margin": 0.05, "n": 10}
            for q, m in deferral.items()}
    for q in info:
        info[q]["model"] = info[q]["model"] or Q9
    return ConformalThresholds("SW-B", target_model, per, 1.0, 0.2, False, info)


class Rig:
    def __init__(self, tmp_path, thresholds, d1=None, q4=None, q9=None):
        self.backends = {
            D1: FakeBackend(d1_responder(0.05, d1 or constant("A", 0.7, 1)), model="d1-fake", name=D1),
            Q4: FakeBackend(q4 or constant("X", 0.2, 2), model="q4-fake", name=Q4),
            Q9: FakeBackend(q9 or constant("B", 0.8, 5), model="q9-fake", name=Q9),
            BONSAI: FakeBackend(hashed_answers, model="bonsai-fake", name=BONSAI),
            FW: FakeBackend(hashed_answers, model="fw-fake", name=FW)}
        router = only_d1_router()
        self.harness = V41Harness(
            load_case=make_loader(), gate=allow_all, workers={k: Worker(v) for k, v in self.backends.items()},
            queen_provider=queen_provider(), world_config=small_cfg(), router=router, bonsai_router=router,
            thresholds=thresholds, bonsai_thresholds=thresholds, allow_unanchored=True, keys=KeyVault(),
            out_dir=tmp_path / "out")

    def run(self, arm="SW-B", case=CASE_DEV, **kw):
        return asyncio.run(self.harness.run(arm, case, 1, **kw))

    def calls(self, name):
        return [r for r in self.backends[name].requests if "look" not in r.questions]


def test_only_deferred_questions_take_their_models_answer_and_shared_models_get_one_call(tmp_path):
    # target and release both go to the 4B, amount to the 9B; d1 keeps nothing deferred away
    th = all_defer({"target": Q4, "release": Q4, "amount": Q9})
    rig = Rig(tmp_path, th)
    res = rig.run()
    assert len(res.decision_log) == N_DEC and res.cell.get("error") is None
    assert len(rig.calls(Q4)) == N_DEC and len(rig.calls(Q9)) == N_DEC       # one call each, though 4B answers two questions
    assert len(rig.calls(D1)) == N_DEC
    for r in res.decision_log:
        d = r["deferral"]
        assert d["deferred_questions"] == ["target", "release", "amount"] and r["deferred"] == 1
        assert d["to"] == {"target": Q4, "release": Q4, "amount": Q9}
        assert d["d1"]["target"] == "A"                                      # the d1 answer is logged
        assert (r["target"], r["release"], r["amount"]) == ("X", 0, 5)       # 4B: target X, no release; 9B: level 5
        assert r["backend"] == Q4 and {c["worker"] for c in r["calls"]} == {D1, Q4, Q9}


def test_a_question_without_a_deferral_model_keeps_d1s_answer(tmp_path):
    th = all_defer({"target": Q9, "release": None, "amount": None})
    rig = Rig(tmp_path, th)
    res = rig.run()
    assert len(rig.calls(Q9)) == N_DEC and not rig.calls(Q4)
    for r in res.decision_log:
        d = r["deferral"]
        assert d["deferred_questions"] == ["target"] and d["to"] == {"target": Q9}
        assert d["answers_from"] == {"target": Q9, "release": D1, "amount": D1}
        assert (r["target"], r["release"], r["amount"]) == ("B", 1, 1)       # 9B target; d1 release 0.7 -> yes; d1 level 1
        assert set(d["sets"]) == {"target", "release", "amount"}             # all three sets are logged
    assert res.cell["defer_rate"] == 1.0


def test_no_deferral_question_means_d1_answers_and_the_sets_are_only_logged(tmp_path):
    th = all_defer({"target": None, "release": None, "amount": None})
    rig = Rig(tmp_path, th)
    res = rig.run()
    assert not rig.calls(Q4) and not rig.calls(Q9)
    for r in res.decision_log:
        assert r["deferred"] == 0 and r["deferral"]["deferred_questions"] == [] and r["backend"] == D1
        assert set(r["deferral"]["sets"]) == {"target", "release", "amount"}


def test_each_decision_logs_the_current_target_and_the_label_map(tmp_path):
    rig = Rig(tmp_path, all_defer({"target": Q9, "release": None, "amount": None}))
    res = rig.run()
    for r in res.decision_log:
        res_ = r["resolution"]
        assert res_["source"] == "exact" and set(res_["target_map"]) >= {"S", "X"}
        assert res_["target_map"]["X"] == "explore" and res_["target_map"]["S"] == res_["current"]
    first = [r for r in res.decision_log if r["round"] == 0]
    assert first and all(r["resolution"]["current"] in ("hold",) or r["resolution"]["current"].startswith("r") for r in first)


# --- step 9: --log-counterfactuals all --------------------------------------------------------------------------------

def test_log_counterfactuals_all_logs_every_model_and_changes_nothing_live(tmp_path):
    th = all_defer({"target": Q9, "release": None, "amount": None})
    plain = Rig(tmp_path / "a", th).run()
    rig = Rig(tmp_path / "b", th)
    res = rig.run(log_counterfactuals="all")
    assert [(r["target"], r["release"], r["amount"], r["backend"]) for r in res.decision_log] == \
           [(r["target"], r["release"], r["amount"], r["backend"]) for r in plain.decision_log]   # live decisions unchanged
    assert res.cell["coverage"] == plain.cell["coverage"]
    assert res.run_log["log_counterfactuals"] == "all" and plain.run_log["log_counterfactuals"] == "none"
    for r in res.decision_log:
        assert set(r["counterfactual"]) == {D1, Q4, Q9}
        assert r["counterfactual"][Q4]["target"] == "X" and r["counterfactual"][Q9]["target"] == "B"
        assert r["counterfactual"][D1]["probs"]["target"] and r["sprime"] and r["features"]
    assert len(rig.calls(Q4)) == N_DEC and len(rig.calls(Q9)) == N_DEC          # one call per model per decision, even the used 9B
    assert all(r["counterfactual"] is None for r in plain.decision_log)


def test_log_counterfactuals_is_dev_only_worker_arms_only_and_not_with_calibration(tmp_path):
    rig = Rig(tmp_path, all_defer({"target": Q9, "release": None, "amount": None}))
    with pytest.raises(HarnessError):
        rig.run(case=CASE_EVAL, log_counterfactuals="all")
    with pytest.raises(HarnessError):
        rig.run(arm="FW", log_counterfactuals="all")
    with pytest.raises(HarnessError):
        rig.run(calibration=True, log_counterfactuals="all")
    with pytest.raises(HarnessError):
        rig.run(log_counterfactuals="some")


def test_a_failed_counterfactual_never_fails_the_live_decision(tmp_path):
    from antelligence.kernel.decision import DecisionError

    class Broken(FakeBackend):
        async def decide(self, request):
            raise DecisionError("4b: down")

    th = all_defer({"target": Q9, "release": None, "amount": None})
    rig = Rig(tmp_path, th)
    rig.harness.workers[Q4] = Worker(Broken(None, model="q4-fake", name=Q4))
    res = rig.run(log_counterfactuals="all")
    assert res.cell["decision_errors"] == 0 and len(res.decision_log) == N_DEC
    assert all("error" in r["counterfactual"][Q4] and Q9 in r["counterfactual"] for r in res.decision_log)


def test_the_fit_reads_step9_logs_where_the_router_not_the_9b_drove(tmp_path):
    th = all_defer({"target": Q9, "release": None, "amount": None})
    rig = Rig(tmp_path, th)
    res = rig.run(log_counterfactuals="all")
    assert {r["backend"] for r in res.decision_log} == {Q9}                     # deferred target: the 9B led
    runs = [v41_fit.load_run(d, None) for d in v41_fit.discover([res.run_dir.parent.parent])]
    assert runs[0].mixed
    samples, skipped = v41_fit.samples_from(runs)
    assert len(samples) == N_DEC and not skipped
    s = samples[0]
    assert set(s.decisions) == {D1, Q4, Q9} and s.resolution == "exact" and s.target_map and s.sprime_resolved
    assert s.driver == Q9
    th2 = v41_fit.fit_conformal(samples)
    assert set(th2.deferral) == {"target", "release", "amount"}
    report = v41_fit.conformal_report(th2, samples)
    assert report["conformity_label"] == "sprime-resolved"
