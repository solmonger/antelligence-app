"""The tumour wiring: colony off == v4.1, per-ant visit sets, the observation barrier on the real world (goal 2.8)."""

from pathlib import Path

import numpy as np
import pytest

from antelligence.kernel.colony.episode import ColonyConfig, run_colony_episode
from antelligence.kernel.signal import SignalDraft
from antelligence.kernel.types import Intent
from antelligence.worlds.tumor_mri import OBS_KEYS, Crop, ScriptS, TumorMRIWorld, run_episode
from antelligence.worlds.tumor_mri.colony_obs import ALLOWED, ANT_ARMS, COLONY_OBS_KEYS, arm_spec, expected_keys
from antelligence.worlds.tumor_mri.colony_world import ColonyTumorWorld
from tests.v4_1.world.phantom import phantom_config

FIX = Path(__file__).parents[2] / "v4_1" / "world" / "fixtures"


@pytest.fixture(scope="module")
def crop() -> Crop:
    return Crop.from_npz(FIX / "phantom.npz")


def cfg():
    return phantom_config(ticks=60)


def look_fn_for(crop):
    p = crop.p_tumour
    return lambda squad, pos, tick: float(p[pos])          # a perception-derived stand-in; real SimLook is W1's


class Chatty:
    """ScriptS plus colony behaviour: judgement mark where it stands, an ask when it can quote a look."""

    def __init__(self):
        self.inner = ScriptS()

    def decide(self, view):
        base = self.inner.decide(view)
        o, params, emit = view.observation, dict(base.params), []
        if o.get("local_look") is not None and o["local_look"] >= 0.5 and view.tick % 10 == 0:
            emit.append(SignalDraft(kind="tumour_here", pos=tuple(float(c) for c in o["position_vox"]), ttl=1))
        if view.agent_id == "squad-0" and view.tick == 10 and o["local_look"] is not None:
            params["ask"] = {"question": "is_tumour", "pos_vox": list(o["position_vox"]), "note": "hello"}
        return Intent(base.action, params, emit=tuple(emit), meta=base.meta)

    def describe(self):
        return {}


def run(crop, arm, seed=3, policy=Chatty, **kw):
    world = ColonyTumorWorld(crop, cfg(), seed=seed, arm=arm, look_fn=look_fn_for(crop))
    seen = []

    class Rec(policy):
        def decide(self, view):
            seen.append((view.agent_id, view.tick, dict(view.observation)))
            return super().decide(view)

    res = run_colony_episode(world, lambda a: Rec(), arm=arm_spec(arm), base_keys=OBS_KEYS, **kw)
    return world, res, seen


def test_colony_obs_tables_match_interfaces():
    assert COLONY_OBS_KEYS == ("look", "uptake_reading", "washout_reading", "reading_log", "marks_near", "messages_in",
                               "zone_need", "colony_note", "region_status", "visited_self", "card_self", "cards_public",
                               "notebook_self")
    assert len(OBS_KEYS) == 18 and not set(OBS_KEYS) & set(COLONY_OBS_KEYS) - {"look"}
    assert ALLOWED["S"] == () and ALLOWED["S-prime"] == ()
    assert set(ALLOWED["S-marks"]) == set(ALLOWED["S-solo"]) | {"marks_near", "region_status", "cards_public"}
    assert set(ALLOWED["S-talk"]) == set(ALLOWED["S-marks"]) | {"messages_in"}
    for a in ("S-marks-displaced", "S-marks-perfect", "S-marks-quiet"):
        assert ALLOWED[a] == ALLOWED["S-marks"]
    assert ALLOWED["S-talk-shuffled"] == ALLOWED["S-talk"]
    with pytest.raises(ValueError):
        arm_spec("S-prime")


def test_colony_off_is_v41_exactly(crop):
    plain_world = TumorMRIWorld(crop, cfg(), seed=3)
    base = run_episode(plain_world, lambda a: ScriptS())
    w = ColonyTumorWorld(crop, cfg(), seed=3, arm="S")
    out = run_colony_episode(w, lambda a: ScriptS(), arm=arm_spec("S"), base_keys=OBS_KEYS)
    assert out.metrics == base
    assert w.describe() == plain_world.describe()                       # same config hash input
    assert out.colony["marks"] == [] and out.colony["messages"] == []
    # the observation of a colony-off world is untouched
    assert set(w.observe("squad-0", 0).data) == set(OBS_KEYS)


def test_s_keeps_colony_wide_visit_sets_and_colony_arms_use_per_ant_sets(crop):
    w_s = ColonyTumorWorld(crop, cfg(), seed=3, arm="S")
    run_colony_episode(w_s, lambda a: ScriptS(), arm=arm_spec("S"))
    assert w_s._visited_regions and not hasattr(w_s, "_vis_regions")
    w, res, _ = run(crop, "S-solo")
    assert not w._visited_regions                                       # the colony-wide set is never written
    per = w._vis_regions
    assert len(per) == 8 and all(isinstance(s, set) for s in per)
    walked = [int(a.sum()) for a in w._vis_arrays]
    assert all(n > 1 for n in walked) and len(set(walked)) > 1
    assert w.visited_self(0) == {"regions": sorted(per[0]), "walked_count": walked[0]}


def test_candidate_visited_flag_is_the_asking_ants_own(crop):
    w = ColonyTumorWorld(crop, cfg(), seed=3, arm="S-marks")
    w.start_episode()
    cand = w.candidates_for(0)[0]
    w._vis_regions[0].add(cand.region_id)
    flags = lambda i: {c["label"]: c["visited"] for c in w.observe(f"squad-{i}", 0).data["candidates"]}
    assert flags(0)[cand.label] is True and flags(1)[cand.label] is False


def test_explore_uses_own_visits_only(crop):
    w = ColonyTumorWorld(crop, cfg(), seed=3, arm="S-solo")
    w.start_episode()
    far = max(w.regions, key=lambda r: r.volume_mm3)
    for i in range(1, 8):
        w._vis_regions[i].update(r.region_id for r in w.regions)           # everyone else has been everywhere
    goal0 = w._explore_goal(0, w._squads[0])
    w._vis_regions[0].update(r.region_id for r in w.regions)
    assert goal0 is not None                                               # ant 0 still has unvisited regions to explore
    assert far.region_id not in w._visited_regions


@pytest.mark.parametrize("arm", ANT_ARMS)
def test_barrier_on_the_tumour_world_every_arm(crop, arm):
    queen_in = []

    def queen(tick, inputs):
        queen_in.append(dict(inputs))
        return {"zone_need": {r["region_id"]: "medium" for r in inputs["regions"]}}

    w, res, seen = run(crop, arm, queen=queen)
    assert seen
    want = expected_keys(arm)
    for agent, tick, obs in seen:
        assert set(obs) == want, (arm, agent, tick, set(obs) ^ want)
    # runtime check is on too (base_keys=OBS_KEYS): reaching here means it never fired
    spec = arm_spec(arm)
    if spec.colony:
        keys = {frozenset(i) for i in queen_in}
        assert keys == {frozenset({"tick", "regions", "marks", "quorum_events"} if spec.reads_marks else {"tick", "regions"})}
        assert not any("zone_need" in o and arm == "S-solo" and o["zone_need"] != seen[0][2]["zone_need"] for _, _, o in seen)
    if arm == "S-solo":
        assert not {"marks_near", "messages_in", "region_status", "cards_public"} & set(seen[-1][2])
        assert res.colony["counters"]["marks"]                              # but it still writes marks


def test_colony_run_on_the_tumour_world_replays_exactly(crop):
    w, res, _ = run(crop, "S-talk")
    assert res.colony["counters"]["asks_sent"] + res.colony["counters"]["asks_refused"] >= 1
    assert res.colony["reading_logs"]["0"] and any(e["kind"] == "look" for e in res.colony["reading_logs"]["0"])
    w2 = ColonyTumorWorld(crop, cfg(), seed=3, arm="S-talk", look_fn=look_fn_for(crop))
    rep = run_colony_episode(w2, None, arm=arm_spec("S-talk"), base_keys=OBS_KEYS, replay=res.colony)
    assert rep.colony == res.colony and rep.metrics == res.metrics
    w3, res3, _ = run(crop, "S-talk")
    assert res3.colony == res.colony


def test_dosed_marks_carry_the_fraction_of_a_squads_share(crop):
    w, res, _ = run(crop, "S-marks")
    dosed = [m for m in res.colony["marks"] if m["mark_kind"] == "dosed"]
    assert dosed and all(0 < m["amount_milli"] <= 1000 for m in dosed)
    released = res.metrics["released_total"]
    share = w.total_drug / 8
    assert sum(m["amount_milli"] for m in dosed) / 1000 == pytest.approx(released / share, abs=0.01 * len(dosed))


def test_quiet_rule_pair_changes_turns_not_sensors(crop):
    q = ColonyConfig(quiet_rule=True, heartbeat_rounds=3)
    w, res, seen = run(crop, "S-marks", config=q)
    rounds = 60 // 5
    skipped = sum(res.colony["turns_skipped"].values())
    assert skipped > 0 and len(seen) + skipped == 8 * rounds
    for ant in range(8):
        assert len([e for e in res.colony["reading_logs"][str(ant)] if e["kind"] == "dose"]) == rounds
    off = run(crop, "S-marks")[1]
    assert sum(off.colony["turns_skipped"].values()) == 0
