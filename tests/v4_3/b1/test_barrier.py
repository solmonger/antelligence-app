"""B1 barrier (v4.2 goal 2.8, extended): the uptake field reaches an ant only through its own uptake reading at its own release
voxel, or through its own dose reading. Two worlds that differ ONLY in the uptake field (level low vs high) are driven with the
same recorded actions (a replay, so no policy reacts to the difference). Before any release every observation is identical.
After one, the only differences are the releasing ants' dose readings and uptake reading-log entries, and (sharing arms) the marks
those readings produce and the queen's scaling of the zone need. ``field.uptake`` never appears in an observation or a queen input."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from antelligence.experiments.v43_harness import V43Harness
from antelligence.kernel.colony import episode as ep
from antelligence.kernel.colony.episode import QUEEN_MARK_KINDS
from antelligence.kernel.types import Intent
from antelligence.worlds.tumor_mri import OBS_KEYS, TumorMRIWorld
from antelligence.worlds.tumor_mri.colony_obs import ALLOWED, expected_keys
from antelligence.worlds.tumor_mri.colony_queen import ScriptedQueen
from antelligence.worlds.tumor_mri.world import ACTION_DECIDE
from tests.v4_2.q1.helpers import noisy_look_factory
from tests.v4_3.a2.helpers import R43, estimate
from tests.v4_3.b1.helpers import load_yaml_configs, phantom_with_channels

ANT_ARMS = ("S-solo", "S-marks", "S-talk")
SOLO_DIFF = {"local_dose", "local_dose_rel", "reading_log", "uptake_reading"}
SHARING_DIFF = SOLO_DIFF | {"marks_near", "zone_need", "colony_note", "messages_in"}
UPTAKE_DERIVED_MARKS = {"good_uptake", "slow_uptake", "dose_ok"}      # marks placed from the uptake word / the dose reading


def harness(level: str) -> V43Harness:
    w, c, _ = load_yaml_configs("config/v4_3/world.yaml", 120)
    w = w.with_(facts=replace(w.facts, uptake_level=level))
    return V43Harness(w, c, R43, estimate(), look_fn_factory=noisy_look_factory)


def capture(h: V43Harness, arm: str, seed: int, replay=None):
    """(record, [(ant, tick, observation)], [(tick, queen inputs)])."""
    obs, queen_in = [], []
    orig_obs, orig_q = ep._Run._observation, ScriptedQueen.__call__

    def spy_obs(self, i, tick, readings):
        o = orig_obs(self, i, tick, readings)
        obs.append((i, tick, copy.deepcopy(o)))
        return o

    def spy_q(self, tick, inputs):
        queen_in.append((tick, copy.deepcopy(dict(inputs))))
        return orig_q(self, tick, inputs)

    ep._Run._observation, ScriptedQueen.__call__ = spy_obs, spy_q
    try:
        rec = h.run_episode(arm, phantom_with_channels(), seed, replay=replay)
    finally:
        ep._Run._observation, ScriptedQueen.__call__ = orig_obs, orig_q
    return rec, obs, queen_in


@pytest.fixture(scope="module", params=ANT_ARMS)
def twins(request):
    arm = request.param
    reference, _, _ = capture(harness("low"), arm, 3)
    lo = capture(harness("low"), arm, 3, replay=reference["colony"])
    hi = capture(harness("high"), arm, 3, replay=reference["colony"])
    return arm, reference, lo, hi


def releases(rec):
    return [e for e in rec["colony"]["action_log"] if e["kind"] == "decision" and e["params"]["accepted"] and e["params"]["world"].get("release")]


def walk(obj, banned, path=""):
    """Every (path, key) where a banned key appears."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in banned:
                hits.append((path, k))
            hits += walk(v, banned, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            hits += walk(v, banned, f"{path}[{i}]")
    return hits


def test_the_twins_took_the_same_actions(twins):
    arm, ref, (ra, _, _), (rb, _, _) = twins
    assert ra["colony"]["action_log"] == rb["colony"]["action_log"] == ref["colony"]["action_log"]
    assert releases(ra), "the test needs releases"
    assert ra["metrics"]["coverage"] != rb["metrics"]["coverage"]           # the field changed what the drug did


def test_observations_identical_up_to_the_first_release(twins):
    arm, ref, (ra, oa, _), (rb, ob, _) = twins
    first = min(e["tick"] for e in releases(ra))
    early = [(a, b) for a, b in zip(oa, ob) if a[1] <= first]
    assert early and len(oa) == len(ob)
    for (i, t, a), (j, u, b) in early:
        assert (i, t) == (j, u) and a == b, f"observation of ant {i} at tick {t} differs before any release"


def test_after_a_release_only_dose_and_uptake_readings_differ(twins):
    arm, ref, (ra, oa, qa), (rb, ob, qb) = twins
    sharing = "marks_near" in ALLOWED[arm]
    allowed_diff = SHARING_DIFF if sharing else SOLO_DIFF
    released_by = {}                                   # ant -> first tick it released
    for e in releases(ra):
        released_by.setdefault(e["ant"], e["tick"])
    differing = set()
    for (i, t, a), (j, u, b) in zip(oa, ob):
        assert set(a) == set(b) == set(expected_keys(arm))                   # no key is added, whatever the field
        diff = {k for k in a if a[k] != b[k]}
        assert diff <= allowed_diff, f"ant {i} tick {t}: {sorted(diff - allowed_diff)} differ"
        differing |= diff
        if "reading_log" in diff:                    # entry by entry: only dose and uptake readings differ, and uptake only for releasers
            assert len(a["reading_log"]) == len(b["reading_log"])
            for x, y in zip(a["reading_log"], b["reading_log"]):
                if x != y:
                    assert x["kind"] == y["kind"] and x["kind"] in ("dose", "uptake"), (i, t, x, y)
                    assert x["pos_vox"] == y["pos_vox"] and x["tick"] == y["tick"]
                    if x["kind"] == "uptake":
                        assert i in released_by and released_by[i] < t
        if "uptake_reading" in diff:
            assert i in released_by and released_by[i] < t
        if "marks_near" in diff:                     # rows of any other kind are identical
            ka = [r for r in a["marks_near"] if r["kind"] not in UPTAKE_DERIVED_MARKS]
            kb = [r for r in b["marks_near"] if r["kind"] not in UPTAKE_DERIVED_MARKS]
            assert ka == kb
    assert {"local_dose", "reading_log"} <= differing                        # the field is reachable through the dose reading
    assert "uptake_reading" in differing
    if not sharing:
        assert not (differing & {"marks_near", "zone_need", "colony_note", "messages_in"})


def test_uptake_readings_are_about_the_ants_own_release_voxels(twins):
    arm, ref, (ra, oa, _), _ = twins
    stood = {}                                       # (ant, tick) -> position at that turn, and the release voxels per ant
    for i, t, o in oa:
        stood[(i, t)] = tuple(o["position_vox"])
    own_release = {}
    for e in releases(ra):
        own_release.setdefault(e["ant"], set()).add(stood[(e["ant"], e["tick"])])
    seen = 0
    for i, t, o in oa:
        for e in o["reading_log"]:
            if e["kind"] == "uptake":
                seen += 1
                assert tuple(e["pos_vox"]) in own_release.get(i, set()), "an uptake reading about a voxel the ant never released at"
        up = o["uptake_reading"]
        if up is not None:
            assert tuple(up["about_vox"]) in own_release.get(i, set()) and up["word"] in ("good", "middling", "poor")
    assert seen


def test_the_field_itself_is_in_no_observation_and_no_queen_input(twins):
    arm, ref, (ra, oa, qa), _ = twins
    for i, t, o in oa:
        assert walk({k: v for k, v in o.items() if k not in ("reading_log", "uptake_reading")}, {"uptake", "u", "tc", "et", "base", "sigma"}) == []
        assert set(o) == expected_keys(arm)
    sharing = "marks_near" in ALLOWED[arm]
    for _, inputs in qa:
        assert set(inputs) == ({"tick", "regions", "marks", "quorum_events"} if sharing else {"tick", "regions"})
        assert walk(inputs, {"uptake", "u", "tc", "et", "reading_log", "local_dose"}) == []
        for m in inputs.get("marks", ()):
            assert m["mark_kind"] in QUEEN_MARK_KINDS


def test_queen_sees_uptake_marks_only_as_board_marks_in_sharing_arms(twins):
    arm, ref, (ra, oa, qa), _ = twins
    kinds = {m["mark_kind"] for _, q in qa for m in q.get("marks", ())}
    if "marks_near" in ALLOWED[arm]:
        assert kinds <= set(QUEEN_MARK_KINDS)
        assert all("uptake_factor_milli" in r for r in ra["colony"]["need_per_zone"])
        assert any(r["uptake_factor_milli"] != 1000 for r in ra["colony"]["need_per_zone"])    # she acted on the uptake marks
    else:
        assert kinds == set()                                                       # S-solo: no board
        assert all("uptake_factor_milli" in r and r["uptake_factor_milli"] == 1000 for r in ra["colony"]["need_per_zone"]) or \
            all("uptake_factor_milli" not in r for r in ra["colony"]["need_per_zone"])


# --- the bare world (no colony) ---------------------------------------------------------------------------------------------

def test_world_observations_differ_only_in_local_dose_after_a_release():
    from antelligence.worlds.tumor_mri.config import FactsConfig
    from tests.v4_1.world.phantom import oracle_config
    from tests.v4_2.q1.helpers import PHANTOM
    from antelligence.worlds.tumor_mri import Crop
    crop = Crop.from_npz(PHANTOM)
    ch = phantom_with_channels().scorer_channels
    worlds = [TumorMRIWorld(crop, oracle_config(facts=FactsConfig(f2=True, f2_model="foraging", uptake_level=lvl)), seed=4, scorer_channels=ch)
              for lvl in ("low", "high")]
    assert not np.array_equal(worlds[0].field.uptake, worlds[1].field.uptake)
    core = np.argwhere(ch["tc"] & ~ch["et"])                      # every squad walks to a core voxel (base 0.4 / 0.2) and releases
    centre = core[np.argmin(((core - core.mean(axis=0)) ** 2).sum(axis=1))].tolist()
    seen_diff = set()
    for tick in range(0, 100, 5):
        obs = [[w.observe(a, tick).data for a in w.agents()] for w in worlds]
        for oa, ob in zip(*obs):
            assert set(oa) == set(ob) == set(OBS_KEYS)
            d = {k for k in oa if oa[k] != ob[k]}
            seen_diff |= d
            if tick <= 0:
                assert d == set()
            assert d <= {"local_dose", "local_dose_rel"}, d
        for w in worlds:
            for i, name in enumerate(w.agents()):
                w.apply(name, Intent(ACTION_DECIDE, {"target": "S", "release": int(tick >= 20 and i % 2 == 0), "amount": 3,
                                                     "target_vox": centre}, meta={"scripted": True}), tick)
            for k in range(5):
                w.step_environment(tick + k)
    assert seen_diff == {"local_dose", "local_dose_rel"}
