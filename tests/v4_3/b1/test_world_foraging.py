"""B1 sections 2 and 3: the world builds the foraging field, mass balance, sensor words and counts, events, automatic marks."""

import numpy as np
import pytest

from antelligence.kernel.types import Intent
from antelligence.worlds.tumor_mri import ScriptS, TumorMRIWorld, run_episode
from antelligence.worlds.tumor_mri.colony_policies import automatic_marks
from antelligence.worlds.tumor_mri.config import FactsConfig
from antelligence.worlds.tumor_mri.data import Crop
from antelligence.worlds.tumor_mri.foraging import (UPTAKE_LEVELS, ForagingError, automatic_marks_v43, foraging_uptake,
                                                    queen_uptake_scaling, select_auto_mark_rule, uptake_word)
from antelligence.worlds.tumor_mri.world import ACTION_DECIDE
from tests.v4_1.world.phantom import oracle_config
from tests.v4_2.q1.helpers import PHANTOM
from tests.v4_3.b1.helpers import channels

FORAGING = dict(f2=True, f2_model="foraging", uptake_level="mid")


@pytest.fixture(scope="module")
def crop() -> Crop:
    return Crop.from_npz(PHANTOM)


def fcfg(**kw):
    return oracle_config(facts=FactsConfig(**{**FORAGING, **kw}))


def run(crop, cfg, seed=2, ch=True):
    w = TumorMRIWorld(crop, cfg, seed=seed, scorer_channels=channels(crop) if ch else None)
    return w, run_episode(w, lambda a: ScriptS())


# --- the field in the world -----------------------------------------------------------------------------------------------

def test_world_builds_the_foraging_field_from_the_scorer_channels(crop):
    w = TumorMRIWorld(crop, fcfg(), seed=1, scorer_channels=channels(crop))
    ch = channels(crop)
    want = foraging_uptake(crop.truth, ch["tc"], ch["et"], crop.case_id, "mid")
    assert np.array_equal(w.field.uptake, want) and w.field.dose_raw is not None
    assert len(np.unique(w.field.uptake)) > 10          # continuous, not two-level
    for lvl in ("low", "high"):
        w2 = TumorMRIWorld(crop, fcfg(uptake_level=lvl), seed=1, scorer_channels=channels(crop))
        assert np.array_equal(w2.field.uptake, foraging_uptake(crop.truth, ch["tc"], ch["et"], crop.case_id, lvl))


def test_foraging_without_channels_is_a_clear_error(crop):
    with pytest.raises(ForagingError, match="TC and ET"):
        TumorMRIWorld(crop, fcfg(), seed=1)
    with pytest.raises(ForagingError, match="TC and ET"):
        TumorMRIWorld(crop, fcfg(f2_source="B", f2_share=0.2), seed=1)       # no Source B for the foraging field


def test_v42_model_unchanged_by_the_flag_default(crop):
    ch = channels(crop)
    cfg = oracle_config(facts=FactsConfig(f2=True, f2_source="A", u_core=0.5))
    w = TumorMRIWorld(crop, cfg, seed=1, scorer_channels=ch)
    assert set(np.unique(w.field.uptake)) == {0.5, 1.0} and "f2_model" not in cfg.to_dict()["facts"]


# --- mass balance ---------------------------------------------------------------------------------------------------------

def test_mass_balance_closes_and_u_only_scales_dose(crop):
    base_w, base = run(crop, oracle_config())
    for lvl in ("low", "mid", "high"):
        w, m = run(crop, fcfg(uptake_level=lvl))
        assert abs(m["mass_residual"]) < 1e-9
        assert 0.0 <= m["off_target_share"] <= 1.0
        # mass accounting untouched: the concentration history is the same as with no uptake field
        assert m["released_total"] == base["released_total"] and m["decayed"] == base["decayed"]
        assert m["lost_boundary"] == base["lost_boundary"] and m["off_target_share"] == base["off_target_share"]
        assert np.array_equal(w.field.conc, base_w.field.conc)
        assert np.allclose(w.field.dose, w.field.uptake * w.field.dose_raw) and np.allclose(w.field.dose_raw, base_w.field.dose)
        assert m["coverage"] <= base["coverage"] + 1e-12 and float(w.field.dose.sum()) < float(base_w.field.dose.sum())
        assert "facts" in m


def test_higher_level_means_less_dose(crop):
    sums = [float(run(crop, fcfg(uptake_level=lvl))[0].field.dose.sum()) for lvl in ("low", "mid", "high")]
    assert sums[0] > sums[1] > sums[2]


# --- sensor words ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("val,word", [(1.2, "good"), (0.7, "good"), (0.6999999, "middling"), (0.55, "middling"), (0.4, "middling"),
                                      (0.3999999, "poor"), (0.0, "poor"), (-0.2, "poor")])
def test_word_boundaries(val, word):
    assert uptake_word(val) == word


def release_then_walk(w, i=0):
    v0 = tuple(w.squad_positions()[i])
    w.apply(f"squad-{i}", Intent(ACTION_DECIDE, {"target": "S", "release": 1, "amount": 3, "target_vox": [18, 16, 17]},
                                 meta={"scripted": True}), 0)
    for t in range(5):
        w.step_environment(t)
    return v0


def test_reading_in_foraging_mode_uses_the_new_words_and_counts(crop):
    w = TumorMRIWorld(crop, fcfg(), seed=5, scorer_channels=channels(crop))
    assert w.sensor_stats()["uptake_good"] == 0 and w.sensor_stats()["uptake_middling"] == 0 and w.sensor_stats()["uptake_poor"] == 0
    for target, word in ((1.0, "good"), (0.55, "middling"), (0.1, "poor")):
        w = TumorMRIWorld(crop, fcfg(), seed=5, scorer_channels=channels(crop))
        w.field.uptake[:] = target
        v0 = release_then_walk(w)
        r = w.sensor_readings("squad-0", 5)["uptake"]
        assert r["pos_vox"] == list(v0) and r["word"] == uptake_word(r["reading"])
        assert abs(r["reading"] - target) < 0.5                                   # ratio + N(0, 0.1)
        st = w.sensor_stats()
        assert st["uptake_readings"] == 1 and st["uptake_" + r["word"]] == 1
        assert sum(st[f"uptake_{k}"] for k in ("good", "middling", "poor")) == 1


def test_v42_mode_words_and_stats_unchanged(crop):
    cfg = oracle_config(facts=FactsConfig(f2=True, f2_source="A", u_core=0.5))
    for u, word in ((0.3, "slow"), (1.0, "as expected")):
        w = TumorMRIWorld(crop, cfg, seed=5, scorer_channels=channels(crop))
        w.field.uptake[:] = u
        release_then_walk(w)
        assert w.sensor_readings("squad-0", 5)["uptake"]["word"] == word
        assert set(w.sensor_stats()) == {"washout_turns", "washout_available", "washout_fast", "uptake_readings", "uptake_slow",
                                         "washout_available_share"}


class FixedWord:
    """Replace the sensor so an event test does not depend on noise."""

    def __init__(self, w, word):
        self.w = w
        w.sensor_readings = lambda agent, tick: {"uptake": {"pos_vox": [3, 4, 5], "reading": 0.5, "word": word}, "washout": None}


@pytest.mark.parametrize("word,kinds", [("good", ["good_uptake"]), ("middling", []), ("poor", ["slow_uptake"]),
                                        ("slow", ["slow_uptake"]), ("as expected", [])])
def test_pending_sensor_events_mapping(crop, word, kinds):
    w = TumorMRIWorld(crop, fcfg(), seed=1, scorer_channels=channels(crop))
    FixedWord(w, word)
    ev = w.pending_sensor_events("squad-2", 10)
    assert [e["kind"] for e in ev] == kinds
    for e in ev:
        assert e["ant"] == 2 and e["tick"] == 10 and e["pos_vox"] == [3, 4, 5] and e["reading"] == 0.5


# --- the automatic mark rule ------------------------------------------------------------------------------------------------

def ctx(readings, release_fraction=0.0, zone=1):
    return {"ant": 1, "tick": 20, "pos_vox": [9, 9, 9], "readings": readings, "release_fraction": release_fraction, "zone": zone}


def up(word, value=0.5, pos=(3, 4, 5)):
    return {"kind": "uptake", "value": value, "word": word, "pos_vox": list(pos)}


def test_rule_adds_good_and_poor_at_the_release_voxel():
    assert automatic_marks_v43(ctx([up("good", 0.8)])) == [("good_uptake", [3, 4, 5], 1.0, 800000)]
    assert automatic_marks_v43(ctx([up("poor", 0.2)])) == [("slow_uptake", [3, 4, 5], 1.0, 200000)]
    assert automatic_marks_v43(ctx([up("slow", 0.2)])) == [("slow_uptake", [3, 4, 5], 1.0, 200000)]
    assert automatic_marks_v43(ctx([up("middling")])) == [] and automatic_marks_v43(ctx([up("as expected")])) == []
    assert automatic_marks_v43(ctx([up("good", 1.3)]))[0][3] == 1_000_000 and automatic_marks_v43(ctx([up("poor", -0.1)]))[0][3] == 0


def test_rule_keeps_every_other_mark_of_the_base_rule():
    readings = [{"kind": "dose", "value": 1.2}, up("good", 0.9), {"kind": "washout", "value": 0.03, "word": "fast", "pos_vox": [7, 7, 7]},
                {"kind": "look", "value": 0.9, "band": 2}]
    got = automatic_marks_v43(ctx(readings, release_fraction=0.5))
    base = automatic_marks(ctx([r for r in readings if r["kind"] != "uptake"], release_fraction=0.5))
    assert got[:len(base)] == base and got[len(base):] == [("good_uptake", [3, 4, 5], 1.0, 900000)]
    assert [k for k, *_ in got] == ["dosed", "dose_ok", "washout", "good_uptake"]
    # a reading with no pos_vox falls back to the ant's own voxel, as in the base rule
    assert automatic_marks_v43(ctx([{"kind": "uptake", "value": 0.9, "word": "good"}]))[0][1] == [9, 9, 9]


def test_base_rule_is_untouched_and_ignores_the_new_words():
    assert automatic_marks(ctx([up("poor", 0.2)])) == [] and automatic_marks(ctx([up("good", 0.9)])) == []
    assert automatic_marks(ctx([up("slow", 0.2)])) == [("slow_uptake", [3, 4, 5], 1.0, 200000)]


def test_selection_by_config(crop):
    from antelligence.worlds.tumor_mri import WorldConfig
    v42 = WorldConfig(facts=FactsConfig(f1_simlook=True, f2=True, u_core=0.5))
    assert select_auto_mark_rule(v42) is automatic_marks
    assert select_auto_mark_rule(WorldConfig()) is automatic_marks
    fo = WorldConfig(facts=FactsConfig(f1_simlook=True, f2=True, f2_model="foraging"))
    assert select_auto_mark_rule(fo) is automatic_marks_v43
    assert select_auto_mark_rule(WorldConfig(facts=FactsConfig(f2=False, f2_model="foraging"))) is automatic_marks
    assert queen_uptake_scaling(fo, True) is True and queen_uptake_scaling(fo, False) is False
    assert queen_uptake_scaling(v42, True) is False and queen_uptake_scaling(WorldConfig(), True) is False
