"""S'-resolved labels (decision 2026-10-13): resolution, the match metric, conformity labels, the per-question deferral
target with the 5-point rule, the frozen JSON, and rebuilding the state of older logs."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from antelligence.experiments import v41_conformal as vc
from antelligence.experiments import v41_fit
from antelligence.experiments.v41_conformal import (ConformalDeferral, ConformalThresholds, DevRecord,
                                                    choose_deferral_models, freeze_thresholds)
from antelligence.experiments.v41_resolve import (EXPLORE_KEY, HOLD_KEY, ResolveError, compare_keys, current_key,
                                                  derive_resolutions, match_to_sprime, resolution_record,
                                                  resolve_distribution, resolve_label, state_from_fields)
from antelligence.kernel.types import Intent
from antelligence.worlds.tumor_mri import ACTION_DECIDE, TumorMRIWorld
from antelligence.worlds.tumor_mri.data import select_candidates
from tests.v4_1.world.phantom import make_phantom, phantom_config

LR = {"A": "r10", "B": "r20", "C": "r30"}


# --- 1. the one resolution function ------------------------------------------------------------------------------------

def test_region_label_maps_to_its_region_whatever_the_current_target():
    assert resolve_label("B", LR, "r10") == "r20" and resolve_label("B", LR, HOLD_KEY) == "r20"


def test_stay_resolves_to_the_current_target_region_explore_or_hold():
    assert resolve_label("S", LR, "r30") == "r30"
    assert resolve_label("S", LR, EXPLORE_KEY) == EXPLORE_KEY       # a squad that is exploring keeps exploring
    assert resolve_label("S", LR, HOLD_KEY) == HOLD_KEY             # no target: "stay put", documented as hold
    assert resolve_label("S", LR, None) == HOLD_KEY


def test_explore_resolves_to_explore_and_unoffered_labels_are_an_error():
    assert resolve_label("X", LR, "r10") == EXPLORE_KEY
    with pytest.raises(ResolveError):
        resolve_label("F", LR, "r10")


def test_s_matches_sprime_iff_the_current_target_is_sprimes_region():
    # S' heads for B (r20). S matches it only while the squad's current target is r20.
    assert resolve_label("S", LR, "r20") == resolve_label("B", LR, "r10")
    assert resolve_label("S", LR, "r10") != resolve_label("B", LR, "r10")


def test_current_key_from_the_observation():
    assert current_key("B", 20) == "r20" and current_key("X", 20) == EXPLORE_KEY and current_key("S", None) == HOLD_KEY


def test_resolve_distribution_sums_labels_that_point_at_the_same_region():
    tmap = {"A": "r10", "B": "r20", "S": "r10", "X": EXPLORE_KEY}
    out = resolve_distribution({"A": 0.2, "B": 0.3, "S": 0.4, "X": 0.1}, tmap)
    assert out == {"r10": pytest.approx(0.6), "r20": pytest.approx(0.3), EXPLORE_KEY: pytest.approx(0.1)}
    assert list(out) == ["r10", "r20", EXPLORE_KEY]                   # first-occurrence order, deterministic
    assert sum(out.values()) == pytest.approx(1.0)
    assert resolve_distribution({"A": 0.5, "B": 0.5}, {}) == {"A": 0.5, "B": 0.5}   # no map = raw


def test_compare_keys_knows_when_it_cannot_tell():
    assert compare_keys("r1", "r1") is True and compare_keys("r1", "r2") is False
    assert compare_keys("e30:C", "e30:D") is False and compare_keys("e30:C", "e60:C") is None
    assert compare_keys("e30:C", "r5") is None and compare_keys("e30:C", EXPLORE_KEY) is False


# --- 2. the world and the harness give the state ---------------------------------------------------------------------------

def _world():
    w = TumorMRIWorld(make_phantom(), phantom_config(), seed=1)
    w.set_candidates(select_candidates(w.regions))
    return w


def test_observation_carries_the_current_target_region():
    w = _world()
    obs = w.observe("squad-0", 0).data
    assert obs["target"] == "S" and obs["target_region"] is None
    st = state_from_fields(obs)
    assert st["current"] == HOLD_KEY and set(st["labels"]) >= {"S", "X", "A"}
    w.apply("squad-0", Intent(ACTION_DECIDE, {"target": "B", "release": 0, "amount": 1}), 0)
    o = w.observe("squad-0", 0).data
    rid_b = w.candidates_for(0)[1].region_id
    assert o["target"] == "B" and o["target_region"] == rid_b
    w.apply("squad-0", Intent(ACTION_DECIDE, {"target": "S", "release": 0, "amount": 1}), 5)
    assert w.observe("squad-0", 5).data["target_region"] == rid_b            # Stay keeps the region
    rec = resolution_record(w.observe("squad-0", 5).data, {"target": "B", "release": "no", "amount": "1"})
    assert rec["target_map"]["S"] == f"r{rid_b}" == rec["sprime_target"] and rec["current"] == f"r{rid_b}"
    w.apply("squad-0", Intent(ACTION_DECIDE, {"target": "X", "release": 0, "amount": 1}), 10)
    o = w.observe("squad-0", 10).data
    assert o["target"] == "X" and o["target_region"] is None
    assert resolution_record(o)["target_map"]["S"] == EXPLORE_KEY


def test_queen_assignment_sets_the_region_and_a_relabel_does_not_move_it():
    w = _world()
    rid = w.candidates_for(0)[1].region_id
    assert w.assign_route(0, rid)
    assert w.observe("squad-0", 0).data["target_region"] == rid
    # the queen offers the same regions under different letters later: the region the squad heads for is unchanged
    old = w.candidates_for(0)
    w.set_candidates([type(c)(**{**c.__dict__, "label": l}) for c, l in zip(reversed(old), "ABCDEF")], squad=0)
    assert w.candidates_for(0)[0].region_id == old[-1].region_id
    assert w.observe("squad-0", 0).data["target_region"] == rid


# --- 3. the metric ---------------------------------------------------------------------------------------------------

def _sample(decisions, sprime, tmap=None, sp_res=None, stale=False, src="exact"):
    return v41_fit.Sample("SW-B", "c", {"target": {}, "release": {}, "amount": {}}, {}, (), decisions, sprime,
                          tmap or {}, sp_res or {}, src, stale)


def _d(target, release=1, amount=3):
    return {"target": target, "release": release, "amount": amount}


def test_match_to_sprime_reports_raw_and_resolved():
    sp = {"target": "B", "release": "yes", "amount": "3"}
    on_b = {"A": "r10", "B": "r20", "S": "r20", "X": EXPLORE_KEY}      # squad is already heading for B
    off_b = {"A": "r10", "B": "r20", "S": "r10", "X": EXPLORE_KEY}     # squad is heading for A
    samples = [
        _sample({"m": _d("S"), "d1": _d("B")}, sp, on_b, {"target": "r20"}),     # S matches: current target is S''s region
        _sample({"m": _d("S"), "d1": _d("A")}, sp, off_b, {"target": "r20"}),    # S does not
        _sample({"m": _d("B"), "d1": _d("B")}, sp, off_b, {"target": "r20"}),
        _sample({"m": _d("X"), "d1": _d("X")}, sp, off_b, {"target": "r20"}),
    ]
    rep = match_to_sprime(samples)["models"]
    assert rep["m"]["target"]["raw"] == pytest.approx(1 / 4)               # only the explicit B
    assert rep["m"]["target"]["resolved"] == pytest.approx(2 / 4)          # + the S that kept B
    assert rep["m"]["target"]["resolved_upper"] == pytest.approx(2 / 4)    # exact state: no unknowns
    assert rep["d1"]["target"]["raw"] == pytest.approx(2 / 4) == rep["d1"]["target"]["resolved"]
    assert rep["m"]["release"]["raw"] == rep["m"]["release"]["resolved"] == 1.0
    assert rep["m"]["amount"]["raw"] == rep["m"]["amount"]["resolved"] == 1.0


def test_match_to_sprime_gives_a_range_when_a_derived_target_is_from_an_earlier_step():
    sp = {"target": "C", "release": "yes", "amount": "3"}
    s = _sample({"m": _d("S")}, sp, {"S": "e30:D", "C": "e60:C"}, {"target": "e60:C"}, stale=True, src="derived")
    t = match_to_sprime([s])["models"]["m"]["target"]
    assert t["resolved"] == 0.0 and t["resolved_upper"] == 1.0


# --- 4. rebuilding the state of older logs -------------------------------------------------------------------------------

def _row(squad, tick, target, sprime_t, offered="ABCD"):
    cf = {"d1": {"probs": {"target": {l: 0.1 for l in offered + "SX"}}}}
    return {"squad": squad, "tick": tick, "target": target, "counterfactual": cf,
            "sprime": {"target": sprime_t, "release": "no", "amount": "3"}}


def test_derive_resolutions_replays_assignment_choice_stay_and_relabel():
    routes = {0: [(0, [11, 12]), (30, [12])]}
    rows = [_row(0, 0, "S", "A"),      # queen gave r11: S keeps it
            _row(0, 5, "B", "B"),      # worker picks B = r12
            _row(0, 10, "S", "A"),     # S keeps r12; S' wants A = r11
            _row(0, 30, "S", "A"),     # new step, A = r12 now (route[0]); the squad still heads for r12
            _row(0, 35, "X", "B")]
    out = derive_resolutions(rows, routes)
    assert out[0]["current"] == "r11" and out[0]["sprime_target"] == "r11" and out[0]["target_map"]["S"] == "r11"
    assert out[1]["target_map"]["B"] == "r12"
    assert out[2]["current"] == "r12" and out[2]["target_map"]["S"] == "r12" and out[2]["sprime_target"] == "r11"
    assert out[3]["current"] == "r12" and out[3]["sprime_target"] == "r12" and out[3]["target_map"]["S"] == "r12"
    assert out[4]["target_map"]["B"] == "e30:B" and out[4]["sprime_target"] == "e30:B"   # an unlogged extra candidate
    assert not out[0]["stale"] and not out[2]["stale"] and out[3]["stale"] is True   # r12 was chosen in step 0
    # after X the squad explores
    rows.append(_row(0, 40, "S", "A"))
    assert derive_resolutions(rows, routes)[5]["current"] == EXPLORE_KEY


def test_derive_resolutions_marks_a_target_carried_from_an_earlier_step_as_stale():
    routes = {0: [(0, [11]), (30, [12])]}
    rows = [_row(0, 0, "S", "A"), _row(0, 30, "S", "A")]
    out = derive_resolutions(rows, routes)
    assert out[0]["stale"] is False and out[1]["stale"] is True and out[1]["current"] == "r11"
    assert out[1]["sprime_target"] == "r12"


def test_a_squad_with_no_route_entries_gets_no_record():
    assert derive_resolutions([_row(3, 0, "S", "A")], {0: [(0, [1])]}) == {}


# --- 5. conformity labels --------------------------------------------------------------------------------------------

def test_d1_probability_of_a_region_is_summed_over_the_labels_that_resolve_to_it():
    s = v41_fit.Sample("SW-B", "c", {"target": {"A": 0.1, "B": 0.2, "S": 0.5, "X": 0.2},
                                     "release": {"yes": 0.5, "no": 0.5}, "amount": {"1": 1.0}}, {}, (), {},
                       {"target": "A", "release": "yes", "amount": "1"},
                       {"A": "r10", "B": "r20", "S": "r10", "X": EXPLORE_KEY}, {"target": "r10"}, "exact")
    rp = s.resolved_probs()
    assert rp["target"] == {"r10": pytest.approx(0.6), "r20": pytest.approx(0.2), EXPLORE_KEY: pytest.approx(0.2)}
    assert s.sprime_labels() == {"target": "r10", "release": "yes", "amount": "1"}


def test_conformity_label_is_sprimes_resolved_label_not_the_driver_answer():
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(400):
        # d1 puts most mass on S, which resolves to region r10, which is also what S' wants; the 9B answers B
        probs = {"A": 0.03, "B": 0.05, "S": 0.82, "X": 0.1}
        samples.append(v41_fit.Sample(
            "SW-B", "c", {"target": probs, "release": {"yes": 0.9, "no": 0.1}, "amount": {"1": 0.8, "2": 0.2}},
            {"target": "B", "release": "yes", "amount": "1"}, (),
            {"d1": _d("S", 1, 1), "qwen3.5-4b": _d("S", 1, 1), "qwen3.5-9b": _d("B", 1, 1)},
            {"target": "A", "release": "yes", "amount": "1"},
            {"A": "r10", "B": "r20", "S": "r10", "X": EXPLORE_KEY}, {"target": "r10"}, "exact"))
    th = v41_fit.fit_conformal(samples)
    # d1's resolved argmax r10 equals S''s label on every record, so the conformity scores are 1 - 0.85, not 1 - 0.05
    assert th.per_question["target"]["qhat"] == pytest.approx(0.15, abs=1e-9)
    rep = v41_fit.conformal_report(th, samples)
    assert rep["d1_argmax_matches_sprime"]["target"] == 1.0 and rep["conformity_label"] == "sprime-resolved"
    assert th.deferral["target"]["match"]["qwen3.5-9b"] == 0.0 and th.deferral["target"]["match"]["d1"] == 1.0


# --- 6. per-question deferral target and the 5-point rule -------------------------------------------------------------------

def _models(n, match):
    """n decisions; ``match`` = {model: {q: hits}} out of n; the rest miss."""
    resolved, sprime = [], []
    for i in range(n):
        sp = {"target": "T", "release": "yes", "amount": "1"}
        sprime.append(sp)
        row = {}
        for m, hits in match.items():
            row[m] = {q: (sp[q] if i < hits.get(q, 0) else "miss") for q in vc.QTYPES}
        resolved.append(row)
    return resolved, sprime


def test_best_candidate_per_question_and_five_point_rule():
    # target: 9B best by 20 points; release: 4B best by exactly 5 points (defers); amount: 9B best by 4 points (does not)
    res, sp = _models(100, {"d1": {"target": 30, "release": 50, "amount": 50},
                            "qwen3.5-4b": {"target": 40, "release": 55, "amount": 52},
                            "qwen3.5-9b": {"target": 50, "release": 54, "amount": 54}})
    out = choose_deferral_models(res, sp, ("qwen3.5-4b", "qwen3.5-9b"))
    assert out["target"]["model"] == "qwen3.5-9b" and out["target"]["defers"]
    assert out["release"]["model"] == "qwen3.5-4b" and out["release"]["defers"]          # 55 - 50 = 5: not below the margin
    assert out["amount"]["model"] == "qwen3.5-9b" and not out["amount"]["defers"]        # 54 - 50 = 4 < 5
    assert out["target"]["match"] == {"d1": 0.3, "qwen3.5-4b": 0.4, "qwen3.5-9b": 0.5}


def test_ties_between_candidates_go_to_the_cheaper_and_no_candidate_beating_d1_means_no_deferral():
    res, sp = _models(100, {"d1": {"target": 60, "release": 60, "amount": 60},
                            "qwen3.5-4b": {"target": 80, "release": 40, "amount": 60},
                            "qwen3.5-9b": {"target": 80, "release": 30, "amount": 60}})
    out = choose_deferral_models(res, sp, ("qwen3.5-4b", "qwen3.5-9b"))
    assert out["target"]["model"] == "qwen3.5-4b" and out["target"]["defers"]
    assert not out["release"]["defers"] and not out["amount"]["defers"]


def test_bonsai_arm_has_one_candidate():
    res, sp = _models(100, {"d1": {"target": 10}, "bonsai-8b": {"target": 30}})
    out = choose_deferral_models(res, sp, ("bonsai-8b",))
    assert out["target"]["model"] == "bonsai-8b" and out["target"]["defers"] and not out["release"]["defers"]


def _records(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        probs, labels = {}, {}
        for q, names in (("target", ("r1", "r2", "r3", "explore")), ("release", ("yes", "no")), ("amount", tuple("12345"))):
            p = rng.dirichlet(np.ones(len(names)) * 0.6)
            probs[q] = dict(zip(names, p.tolist()))
            labels[q] = names[int(np.argmax(p))] if rng.random() < 0.8 else names[int(rng.integers(len(names)))]
        out.append(DevRecord(probs, labels))
    return out


def _deferral(**defers):
    return {q: {"model": "qwen3.5-9b", "defers": defers.get(q, False), "match": {}, "d1_match": 0.0, "margin": 0.05, "n": 1}
            for q in vc.QTYPES}


def test_questions_that_do_not_defer_stay_out_of_the_joint_rate_and_the_decision():
    recs = _records(1500, 3)
    th = freeze_thresholds(recs, arm="a", target_model="qwen3.5-9b", deferral=_deferral(target=True))
    assert th.deferring() == ("target",) and th.model_for("release") is None and th.model_for("target") == "qwen3.5-9b"
    # the joint rate is the target question's alone, and lands on 20%
    assert th.joint_dev_defer_rate == pytest.approx(th.per_question["target"]["dev_defer_rate"])
    assert abs(th.joint_dev_defer_rate - 0.2) <= 0.02 and th.joint_on_target
    dec = ConformalDeferral(th)
    flags = [dec.decide(r.probs) for r in recs]
    assert np.mean([f.deferred for f in flags]) == pytest.approx(th.joint_dev_defer_rate)
    assert all(f.deferred_questions in ((), ("target",)) for f in flags)
    assert all(set(f.sets) == set(vc.QTYPES) for f in flags)           # the other sets are still computed (logged)


def test_nothing_defers_when_no_question_beats_d1():
    th = freeze_thresholds(_records(300, 4), arm="a", target_model="qwen3.5-9b", deferral=_deferral())
    assert th.deferring() == () and th.joint_dev_defer_rate == 0.0 and not th.joint_on_target
    assert not any(ConformalDeferral(th).decide(r.probs).deferred for r in _records(100, 5))


def test_a_file_without_a_deferral_block_defers_every_question_to_its_target_model():
    th = ConformalThresholds("a", "qwen3.5-9b", {q: {"alpha": 0.2, "qhat": 0.4, "n": 1, "dev_defer_rate": 0.2,
                                                     "on_target": True} for q in vc.QTYPES}, 0.2, 0.2, True)
    assert th.deferring() == vc.QTYPES and all(th.model_for(q) == "qwen3.5-9b" for q in vc.QTYPES)
    assert "deferral" not in th.body()


def test_the_choice_and_match_numbers_are_frozen_under_the_sha256(tmp_path):
    defer = _deferral(target=True, release=True)
    defer["target"]["model"] = "qwen3.5-4b"
    defer["target"]["match"] = {"d1": 0.3, "qwen3.5-4b": 0.5}
    th = freeze_thresholds(_records(400, 6), arm="a", target_model="qwen3.5-9b", deferral=defer)
    path = tmp_path / "c.json"
    sha = th.save(path)
    back = ConformalThresholds.load(path, sha)
    assert back.sha256 == sha and back.model_for("target") == "qwen3.5-4b" and back.model_for("amount") is None
    assert back.deferral["target"]["match"] == {"d1": 0.3, "qwen3.5-4b": 0.5}
    doc = json.loads(path.read_text())
    assert doc["body"]["conformity_label"] == "sprime-resolved" and "deferral" in doc["body"]
    doc["body"]["deferral"]["target"]["model"] = "qwen3.5-9b"                       # tampering is caught
    path.write_text(json.dumps(doc))
    with pytest.raises(vc.ConformalError):
        ConformalThresholds.load(path)
    other = freeze_thresholds(_records(400, 6), arm="a", target_model="qwen3.5-9b", deferral=_deferral(target=True, release=True))
    assert other.sha256 != sha


def test_audit_coverage_counts_only_the_audited_questions():
    audits = [({"target": ("r1",)}, {"target": "r1"}), ({"target": ("r2",)}, {"target": "r1"})]
    cov = vc.audit_coverage(audits)
    assert cov["target"] == 0.5 and cov["all"] == 0.5 and "release" not in cov
