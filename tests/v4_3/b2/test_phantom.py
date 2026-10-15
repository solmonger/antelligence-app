"""B2 on the synthetic phantom (synthetic TC / ET channels, the B1 foraging field, fake noisy look). No real case is touched.

Numbers reported at build time (120 ticks, 8 ants, seeds 1 to 4, uptake level mid):
probes S-solo 19/18/23/16 = 76; S-marks 9/14/8/10 = 41 (probes saved by marks 2/3/2/4 = 11); S-talk 41 as S-marks.
Drug released on poor tissue (u < 0.4, level high so that the phantom has some: 27% of its tumour voxels), sum over 4 seeds:
S-solo 599.3 -> 160.7 with the rule on; S-marks 510.8 -> 92.3. (Level mid: S-solo 38.0 -> 7.6.)"""

import json
import time
from dataclasses import replace

import numpy as np
import pytest

from antelligence.experiments.v43_harness import V43Harness, bundle_of
from antelligence.worlds.tumor_mri.foraging import foraging_uptake
from tests.v4_2.q1.helpers import noisy_look_factory
from tests.v4_3.a2.helpers import R43, estimate
from tests.v4_3.b1.helpers import load_yaml_configs, phantom_with_channels
from tests.v4_3.b2.helpers import RF, run

ARMS = ("S-solo", "S-marks", "S-talk")
SEEDS = (1, 2, 3, 4)


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_probes_and_holds_resolve(arm):
    rec = run(RF, arm, 1)
    c = rec["script_counters"]
    assert c["probes"] > 0 and c["holds_resolved"] > 0 and c.get("hold_timeouts", 0) == 0
    assert c["holds_resolved"] <= c["probes"]
    assert rec["ant_rules"]["rules"]["foraging"] is True and rec["ant_rules"]["sha"] == RF.sha()
    assert abs(rec["metrics"]["mass_residual"]) < 1e-9


def test_s_solo_never_uses_a_foreign_mark():
    for seed in SEEDS:
        assert run(RF, "S-solo", seed)["script_counters"].get("probes_saved_by_marks", 0) == 0


@pytest.mark.parametrize("arm", ARMS)
def test_two_runs_are_bit_identical(arm):
    a, b = run(RF, arm, 2), run(RF, arm, 2)
    assert json.dumps(bundle_of(a), sort_keys=True) == json.dumps(bundle_of(b), sort_keys=True)
    assert a["cards_final"] == b["cards_final"] and a["script_counters"] == b["script_counters"]
    assert a["colony"]["action_log"] == b["colony"]["action_log"]


def test_s_marks_probes_no_more_than_s_solo_over_four_seeds():
    solo = sum(run(RF, "S-solo", s)["script_counters"].get("probes", 0) for s in SEEDS)
    marks_recs = [run(RF, "S-marks", s) for s in SEEDS]
    marks = sum(r["script_counters"].get("probes", 0) for r in marks_recs)
    saved = sum(r["script_counters"].get("probes_saved_by_marks", 0) for r in marks_recs)
    print(f"probes S-solo {solo}, S-marks {marks}, saved by marks {saved}")
    assert marks <= solo and saved > 0


def poor_drug(rec, u):
    """Drug (in squad-shares x total/n) released where the true uptake is below 0.4, from the ``dosed`` marks (one per release)."""
    share = rec["total_drug"] / 8
    return sum(m["amount_milli"] / 1000 * share for m in rec["colony"]["marks"]
               if m["mark_kind"] == "dosed" and u[tuple(m["pos_vox"])] < 0.4)


@pytest.mark.parametrize("arm", ("S-solo", "S-marks"))
def test_less_drug_is_released_on_poor_tissue_with_the_rule_on(arm):
    case = phantom_with_channels()
    u = foraging_uptake(case.crop.truth, case.scorer_channels["tc"], case.scorer_channels["et"], case.crop.case_id, "high")
    assert (u[case.crop.truth] < 0.4).mean() > 0.2
    w, c, _ = load_yaml_configs("config/v4_3/world.yaml", 120)
    w = w.with_(facts=replace(w.facts, uptake_level="high"))
    on = off = 0.0
    for s in SEEDS:
        on += poor_drug(V43Harness(w, c, RF, estimate(), look_fn_factory=noisy_look_factory).run_episode(arm, case, s), u)
        off += poor_drug(V43Harness(w, c, R43, estimate(), look_fn_factory=noisy_look_factory).run_episode(arm, case, s), u)
    print(f"{arm}: drug on poor tissue, rule on {on:.1f}, off {off:.1f}")
    assert on < off


def test_the_rule_on_run_is_within_twice_the_rule_off_run():
    def best(rules):
        times = []
        for _ in range(5):
            t = time.perf_counter()
            run(rules, "S-solo", 1)
            times.append(time.perf_counter() - t)
        return min(times)
    off, on = best(R43), best(RF)
    assert on <= 2.0 * off, f"on {on:.3f}s vs off {off:.3f}s"


def test_the_scan_carries_its_p_map_whenever_a_rule_needs_it():
    from tests.v4_3.a2.helpers import R42
    from tests.v4_3.b2.helpers import harness
    case = phantom_with_channels()
    arm = type("A", (), {"kind": "ant", "driver_arm": "S-solo"})()
    for rules, expect in ((RF, True), (R43, True), (R42, False)):
        h = harness(rules)
        world = h.build_world(arm, case, 1, noisy_look_factory(case))
        assert (h.build_scan(world, case).p_tumour is not None) is expect
