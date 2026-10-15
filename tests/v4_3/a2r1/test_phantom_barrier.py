"""A2r1 on the synthetic phantom: more drug released, fewer dose-OK idle turns, bit-identical reruns, speed, and the barrier.

Phantom numbers (S-solo, 120 ticks, 8 ants, noisy fake look, seeds 1 to 4), switch off -> on:
released share of total drug 0.852/0.836/0.859/0.863 -> 0.969 on every seed; turns with reason dose_ok (payload left)
152/153/152/132 -> 17/18/17/15 of 192 decision turns."""

import json
import time
from dataclasses import replace

import numpy as np
import pytest

from antelligence.experiments.v43_harness import bundle_of
from antelligence.worlds.tumor_mri import colony_policies as cp
from antelligence.worlds.tumor_mri.colony_obs import expected_keys
from antelligence.worlds.tumor_mri.colony_policies import AntState, ScanInfo, TurnInputs
from antelligence.worlds.tumor_mri.data import Region
from tests.v4_2.q1.helpers import phantom
from tests.v4_3.a2.helpers import R43, state, v43_harness

OFF = replace(R43, dose_move_on=False)
SEEDS = (1, 2, 3, 4)


def run(rules, arm="S-solo", seed=1, ticks=120):
    return v43_harness(rules, ticks).run_episode(arm, phantom(), seed)


def released_share(rec):
    return rec["metrics"]["released_total"] / rec["total_drug"]


def dose_ok_turns(rec):
    """Decision turns whose release reason was dose_ok: the ant had payload left and its own dose was already >= D_t."""
    return sum(1 for e in rec["colony"]["action_log"] if e["kind"] == "decision" and e["params"]["meta"]["reason"] == "dose_ok")


@pytest.mark.parametrize("seed", SEEDS)
def test_s_solo_releases_a_larger_share_of_the_drug_with_the_rule_on(seed):
    off, on = run(OFF, seed=seed), run(R43, seed=seed)
    assert released_share(on) > released_share(off)
    assert released_share(on) >= 0.9 > released_share(off)


@pytest.mark.parametrize("seed", SEEDS)
def test_s_solo_spends_fewer_turns_dose_ok_with_payload_left(seed):
    off, on = run(OFF, seed=seed), run(R43, seed=seed)
    assert dose_ok_turns(on) < dose_ok_turns(off)
    assert on["script_counters"].get("moved_on_dose_ok", 0) > 0
    assert "moved_on_dose_ok" not in off["script_counters"]


def test_two_runs_with_the_rule_on_are_bit_identical():
    a, b = run(R43, seed=2), run(R43, seed=2)
    assert json.dumps(bundle_of(a), sort_keys=True) == json.dumps(bundle_of(b), sort_keys=True)
    assert a["cards_final"] == b["cards_final"] and a["script_counters"] == b["script_counters"]


def test_the_rule_on_run_is_not_slower_than_twice_the_rule_off_run():
    def best(rules):
        times = []
        for _ in range(5):
            t = time.perf_counter()
            run(rules, seed=1)
            times.append(time.perf_counter() - t)
        return min(times)
    off, on = best(OFF), best(R43)
    assert on <= 2.0 * off, f"on {on:.3f}s vs off {off:.3f}s"


def test_the_helper_stays_fast_on_a_large_zone():
    """120 ants x 10 move-ons on a zone of ~13k candidate voxels (far above a real tumour) with 60 known spots each."""
    n = 80
    p = np.zeros((n, n, n))
    p[10:70, 10:70, 10:70] = 0.9
    region = Region(0, (40.0, 40.0, 40.0), 0.9, 0.1, 400_000)      # zone radius ~ 45.7 + 6 mm = 25.9 voxels
    sc = ScanInfo((region,), 2.0, p)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for ant in range(120):
        st = state(R43)
        st.released_at.extend(tuple(int(c) for c in rng.integers(20, 60, 3)) for _ in range(60))
        for _ in range(10):
            inp = TurnInputs(tick=0, round=0, pos=tuple(int(c) for c in rng.integers(25, 55, 3)), voxel_mm=2.0, payload_fraction=1.0,
                             scan_p=0.9, look=0.9, dose_rel=1.0, candidates=[], reading_log=[], uptake=None)
            assert cp.find_move_on_spot(inp, st, sc, 0) is not None
    assert len(sc.zone_voxels(0)) > 10_000
    assert time.perf_counter() - t0 < 3.0


# --- the information barrier -------------------------------------------------------------------------------------------

class Spy:
    def __init__(self, monkeypatch):
        self.obs, self.marks_seen = [], []
        decide, known = cp.ColonyScript.decide, cp.known_dosed_spots
        spy = self

        def decide_spy(this, view):
            spy.obs.append(dict(view.observation))
            return decide(this, view)

        def known_spy(inp, st):
            spy.marks_seen.append(inp.marks)
            return known(inp, st)

        monkeypatch.setattr(cp.ColonyScript, "decide", decide_spy)
        monkeypatch.setattr(cp, "known_dosed_spots", known_spy)


def test_s_solo_observation_gains_no_key_and_the_move_on_never_reads_marks(monkeypatch):
    spy = Spy(monkeypatch)
    rec = run(R43, seed=3)
    assert rec["script_counters"]["moved_on_dose_ok"] > 0 and spy.marks_seen             # the rule ran
    assert all(m is None for m in spy.marks_seen)                                       # S-solo: the marks input is absent
    assert spy.obs and all(set(o) == expected_keys("S-solo") for o in spy.obs)
    assert not any(k in o for o in spy.obs for k in ("p_tumour", "scan_map", "dose_goal", "marks_near"))


@pytest.mark.parametrize("arm", ("S-marks", "S-talk"))
def test_sharing_arms_read_marks_through_the_existing_input_only(arm, monkeypatch):
    spy = Spy(monkeypatch)
    rec = run(R43, arm=arm, seed=3)
    assert rec["script_counters"]["moved_on_dose_ok"] > 0
    assert all(m is not None for m in spy.marks_seen)
    assert spy.obs and all(set(o) == expected_keys(arm) for o in spy.obs)


def test_the_scan_map_reaches_the_policy_only_when_the_switch_is_on():
    h_on, h_off = v43_harness(R43, 60), v43_harness(OFF, 60)
    case = phantom()
    world = h_on.build_world(__import__("antelligence.experiments.v42_harness", fromlist=["resolve_arm"]).resolve_arm("S-solo"),
                             case, 1, None)
    assert h_on.build_scan(world, case).p_tumour is not None
    assert h_off.build_scan(world, case).p_tumour is None
