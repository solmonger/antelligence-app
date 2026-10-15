"""The Q1 harness end to end on the synthetic phantom with a fake close look: arms, barrier, queen hook, controls, refusal."""

import json
import numpy as np
import pytest

from antelligence.experiments import v42_harness as hz
from antelligence.experiments.v42_harness import (ARMS, ArmDef, CaseData, EvalRefused, V42Harness, assert_dev_case, load_dev_case,
                                                  register_arm, resolve_arm)
from antelligence.kernel.colony.episode import ColonyConfig
from antelligence.worlds.tumor_mri import ScriptS, TumorMRIWorld, oracle_for, run_episode
from antelligence.worlds.tumor_mri.colony_obs import ALLOWED, expected_keys
from antelligence.worlds.tumor_mri.colony_policies import ColonyScript
from antelligence.worlds.tumor_mri.colony_queen import ScriptedQueen
from tests.v4_2.q1.helpers import fake_look_factory, harness, phantom, phantom_cfg

ANT_ARMS = ("S-solo", "S-marks", "S-talk", "S-marks-quiet")
ALL_ARMS = ("S", "S-solo", "S-marks", "S-talk", "S-marks-quiet", "S'")
SHARED = {"marks_near", "messages_in", "region_status", "cards_public"}
MARK_KEYS = {"subtype", "mark_kind", "ant", "pos_vox", "amount_milli", "reading_ppm", "tick"}


@pytest.fixture(scope="module")
def case():
    return phantom()


class Spy:
    """Records every observation a colony script is shown and every input the queen is given."""

    def __init__(self, monkeypatch):
        self.obs, self.queen_in = [], []
        orig, qorig = ColonyScript.decide, ScriptedQueen.__call__
        spy = self

        def decide(this, view):
            spy.obs.append((this.ant, view.tick, dict(view.observation)))
            return orig(this, view)

        def queen(this, tick, inputs):
            spy.queen_in.append((tick, dict(inputs)))
            return qorig(this, tick, inputs)

        monkeypatch.setattr(ColonyScript, "decide", decide)
        monkeypatch.setattr(ScriptedQueen, "__call__", queen)


def test_registered_arms_by_name():
    assert set(ALL_ARMS) <= set(ARMS)
    assert resolve_arm("S'").name == "S'"
    assert ARMS["S-marks-quiet"].quiet_rule and not any(a.quiet_rule for n, a in ARMS.items() if n != "S-marks-quiet")
    with pytest.raises(KeyError):
        resolve_arm("S-nope")


@pytest.mark.parametrize("arm", ANT_ARMS)
def test_information_barrier_end_to_end_per_ant_arm(arm, case, monkeypatch):
    """Every observation an ant sees has exactly OBS_KEYS | ALLOWED[arm]; the queen sees only what the arm allows."""
    spy = Spy(monkeypatch)
    rec = harness(120).run_episode(arm, case, 3)
    want = expected_keys(arm)
    assert spy.obs and all(set(o) == want for _, _, o in spy.obs)
    sharing = "marks_near" in ALLOWED[arm]
    ticks = {t for t, _ in spy.queen_in}
    if sharing:
        assert {frozenset(i) for _, i in spy.queen_in} == {frozenset({"tick", "regions", "marks", "quorum_events"})}
        assert ticks == {0, 30, 60, 90}
        for t, i in spy.queen_in:
            assert all(m["mark_kind"] in ("dosed", "dose_ok", "tumour_here", "not_tumour") and m["tick"] < t for m in i["marks"])
            assert all(set(m) == MARK_KEYS for m in i["marks"])                        # board payload only: no dose, no truth
    else:
        assert {frozenset(i) for _, i in spy.queen_in} == {frozenset({"tick", "regions"})} and ticks == {0}
    # own readings only: every reading-log row of an ant is at a voxel the ant stood on (uptake: a voxel it released at)
    stood = {}
    for ant, tick, o in spy.obs:
        stood.setdefault(ant, set()).add(tuple(o["position_vox"]))
    for ant, tick, o in spy.obs:
        if not ARMS[arm].quiet_rule:                  # a skipped turn still reads its sensors, but the script is not shown it
            assert all(tuple(e["pos_vox"]) in stood[ant] for e in o["reading_log"] if e["kind"] in ("look", "dose", "washout"))
    assert rec["queen_hook_unset"] is True


def test_solo_sees_nothing_shared_and_a_frozen_need(case, monkeypatch):
    spy = Spy(monkeypatch)
    harness(120).run_episode("S-solo", case, 3)
    assert not any(SHARED & set(o) for _, _, o in spy.obs)
    first = spy.obs[0][2]["zone_need"]
    assert first and all(o["zone_need"] == first for _, _, o in spy.obs)               # the tick-0 value for the whole run
    assert all(o["colony_note"] == spy.obs[0][2]["colony_note"] for _, _, o in spy.obs)
    assert all(o["card_self"]["in_run"]["confirmed"] == 0 and o["card_self"]["in_run"]["contradicted"] == 0 for _, _, o in spy.obs)
    assert all(o["visited_self"]["regions"] == sorted(set(o["visited_self"]["regions"])) for _, _, o in spy.obs)


def test_arm_s_and_sprime_observations_are_plain_v41(case):
    h = harness(60)
    rec = h.run_episode("S", case, 3)
    plain = TumorMRIWorld(case.crop, phantom_cfg(60), 3)
    assert rec["metrics"] == run_episode(plain, lambda a: ScriptS())
    rec = h.run_episode("S'", case, 3)
    w = TumorMRIWorld(case.crop, phantom_cfg(60), 3)
    o = oracle_for(w)
    assert rec["metrics"] == run_episode(w, lambda a: o)


@pytest.mark.parametrize("arm", ALL_ARMS)
def test_queen_hook_stays_unset_in_every_arm(arm, case):
    h = harness(30)
    for a in ALL_ARMS:
        assert h.build_world(resolve_arm(a), case, 1, fake_look_factory(case)).queen_hook is None
    rec = h.run_episode(arm, case, 1)
    assert rec["queen_hook_unset"] is True


def test_the_queen_runs_through_the_drivers_queen_argument(case, monkeypatch):
    spy = Spy(monkeypatch)
    harness(60).run_episode("S-marks", case, 1)
    assert [t for t, _ in spy.queen_in] == [0, 30]


def test_quiet_rule_only_in_the_quiet_arm(case):
    h = harness(60)
    skipped = {a: sum(h.run_episode(a, case, 3)["colony"]["turns_skipped"].values()) for a in ANT_ARMS}
    assert skipped["S-solo"] == skipped["S-marks"] == skipped["S-talk"] == 0 and skipped["S-marks-quiet"] > 0
    assert h.run_episode("S-marks-quiet", case, 3)["colony"]["params"]["quiet_rule"] is True
    assert h.run_episode("S-marks", case, 3)["colony"]["params"]["quiet_rule"] is False
    bad = V42Harness(phantom_cfg(30), ColonyConfig(quiet_rule=True), look_fn_factory=fake_look_factory)
    with pytest.raises(ValueError, match="quiet rule"):
        bad.run_episode("S-marks", case, 1)
    assert bad.run_episode("S-marks-quiet", case, 1)["quiet_rule"] is True


def test_determinism_and_replay(case):
    h = harness(60)
    a, b = h.run_episode("S-talk", case, 2), h.run_episode("S-talk", case, 2)
    assert a["colony"] == b["colony"] and a["metrics"] == b["metrics"]
    assert json.dumps(a["cards_final"], sort_keys=True) == json.dumps(b["cards_final"], sort_keys=True)
    assert h.replay(a, case)["metrics"] == a["metrics"]
    c = h.run_episode("S-talk", case, 5)
    assert c["colony"]["action_log"] != a["colony"]["action_log"]                    # the episode seed matters (entry sites)


def test_marks_are_countable_in_every_arm_and_on_the_board(case):
    h = harness(60)
    for arm in ANT_ARMS:
        rec = h.run_episode(arm, case, 3)
        n = sum(rec["colony"]["counters"]["marks"].values())
        assert n > 0
        assert rec["board_entries"] >= n


def test_talk_arm_asks_and_answers_while_marks_arm_does_not(case):
    h = harness(120, noisy=True)
    talk = h.run_episode("S-talk", case, 3)
    c = talk["colony"]["counters"]
    assert c["asks_sent"] > 0 and c["asks_answered"] > 0 and talk["script_counters"].get("asks", 0) == c["asks_sent"] + c["asks_refused"]
    assert h.run_episode("S-marks", case, 3)["colony"]["counters"]["asks_sent"] == 0


def test_cards_and_notebooks_are_wired(case, monkeypatch):
    spy = Spy(monkeypatch)
    rec = harness(120).run_episode("S-talk", case, 3)
    o = spy.obs[-1][2]
    assert o["card_self"]["policy"] == "S-talk" and o["card_self"]["thresholds"].keys() == {"dose", "explore", "verify"}
    assert len(o["cards_public"]) == 7 and all(set(c) == {"ant_id", "policy", "model", "in_run", "band"} for c in o["cards_public"])
    assert o["notebook_self"] == []
    cards = rec["cards_final"]
    marks = [m for m in rec["colony"]["marks"] if m["mark_kind"] in ("tumour_here", "not_tumour", "need_more")]
    answers = [m for m in rec["colony"]["messages"] if m["type"] == "answer" and m["answer"] in ("yes", "no")]
    assert sum(c["in_run"]["claims"] for c in cards.values()) == len(marks) + len(answers)
    assert all(c["cross_run"]["runs"] == 1 and c["cross_run"]["checked"] == c["in_run"]["claims"] for c in cards.values())
    assert all(c["cross_run"]["correct"] <= c["cross_run"]["checked"] for c in cards.values())
    assert rec["colony"]["cards_before"]["ant-00"]["in_run"] == {"claims": 0, "confirmed": 0, "contradicted": 0}


def test_cards_in_sharing_arms_see_other_ants_confirmations(case, monkeypatch):
    spy = Spy(monkeypatch)
    harness(120).run_episode("S-marks", case, 3)
    seen = [c["in_run"] for _, _, o in spy.obs for c in o["cards_public"]]
    assert any(i["confirmed"] + i["contradicted"] > 0 for i in seen)


# --- controls (M1 supplies the implementations; the hooks exist) -------------------------------------------------------------

def test_mark_position_hook_displaces_marks(case):
    h = harness(60)
    base = h.run_episode("S-solo", case, 3)["colony"]["marks"]          # S-solo never reads marks, so only their position changes
    moved = h.run_episode("S-solo", case, 3, mark_pos_fn=lambda idx, ant, kind, pos: [pos[0] + 1, pos[1], pos[2]])["colony"]["marks"]
    assert len(moved) == len(base) and moved[0]["pos_vox"][0] == base[0]["pos_vox"][0] + 1


def test_observation_provider_override_and_registered_control_arm(case, monkeypatch):
    spy = Spy(monkeypatch)
    register_arm(ArmDef("S-talk-shuffled", "ant", "S-talk-shuffled", verdict=False))
    try:
        rec = harness(30).run_episode("S-talk-shuffled", case, 3, obs_providers={"cards_public": lambda i, t: ["override"]})
        assert rec["colony"]["arm"] == "S-talk-shuffled"
        assert all(o["cards_public"] == ["override"] for _, _, o in spy.obs)
        assert set(spy.obs[0][2]) == expected_keys("S-talk-shuffled")
    finally:
        ARMS.pop("S-talk-shuffled")


# --- development cases only ---------------------------------------------------------------------------------------------------

def _split():
    return json.loads((hz.REPO_ROOT / "config" / "v4_1" / "split.json").read_text())


def test_eval_reserve_and_unknown_cases_are_refused_before_loading(tmp_path):
    doc = _split()
    for cid in (doc["eval"][0], doc["reserve"][0], "BRATS_nope"):
        with pytest.raises(EvalRefused):
            assert_dev_case(cid)
        with pytest.raises(EvalRefused):
            load_dev_case(cid, root=tmp_path / "nothing-here")                           # refused before any file is touched
    assert_dev_case(doc["dev_a"][0])


def test_harness_refuses_a_case_object_from_the_eval_split(case):
    ev = CaseData("x", case.crop, None, None, "eval")
    with pytest.raises(EvalRefused):
        harness(30).run_episode("S-solo", ev, 1)
    dev = CaseData(_split()["eval"][0], case.crop, None, None, "dev")                   # lies about its split: the id is checked too
    with pytest.raises(EvalRefused):
        harness(30).run_episode("S-solo", dev, 1)


def test_colony_arms_need_a_close_look(case):
    with pytest.raises(ValueError, match="close look"):
        V42Harness(phantom_cfg(30)).run_episode("S-solo", case, 1)


# --- hidden facts through the harness ------------------------------------------------------------------------------------------

FACTS = dict(f2=True, f2_source="B", f2_share=0.35, u_core=0.3, f3=True, f3_decay=0.08, f3_share=0.3)


def test_sensor_events_become_marks_one_for_one(case):
    """W1's slow-uptake and fast-washout events are placed as marks: counts agree with the world's own sensor stats."""
    rec = harness(120, **FACTS).run_episode("S-marks", case, 3)
    stats, marks = rec["metrics"]["facts"], rec["colony"]["counters"]["marks"]
    assert stats["uptake_readings"] > 0 and stats["washout_fast"] > 0
    assert marks.get("slow_uptake", 0) == stats["uptake_slow"]
    assert marks.get("washout", 0) == stats["washout_fast"]
    slow = [m for m in rec["colony"]["marks"] if m["mark_kind"] == "slow_uptake"]
    assert all(0 <= m["reading_ppm"] <= 1_000_000 for m in slow)
    logs = [e for log in rec["colony"]["reading_logs"].values() for e in log if e["kind"] == "uptake"]
    assert logs and {e["word"] for e in logs} <= {"slow", "as expected"}


def test_source_a_uses_scorer_side_channels_and_slows_the_dose(case):
    truth = case.crop.truth
    et = truth & (np.indices(truth.shape)[0] > truth.nonzero()[0].mean())
    with_ch = CaseData(case.case_id, case.crop, None, {"tc": truth, "et": et}, "synthetic")
    facts = dict(f2=True, f2_source="A", u_core=0.3)
    rec = harness(60, **facts).run_episode("S-solo", with_ch, 3)
    base = harness(60).run_episode("S-solo", with_ch, 3)
    assert rec["metrics"]["coverage"] <= base["metrics"]["coverage"]
    assert rec["metrics"]["facts"]["uptake_readings"] > 0
    with pytest.raises(ValueError, match="TC and ET"):
        harness(30, **facts).run_episode("S-solo", case, 3)                            # no channels: Source A refuses


def test_real_simlook_table_runs_on_the_phantom(case):
    h = harness(60, fake=False)
    rec = h.run_episode("S-marks", case, 3)
    assert rec["sensor"]["source"] == "boards" and rec["timing_s"]["simlook"] >= 0
    h2 = V42Harness(phantom_cfg(30, f1_simlook=True, f1_source="log2000"))
    with pytest.raises(ValueError, match="recorded, never inferred"):
        h2.run_episode("S-solo", case, 1)


def test_oracle_arm_is_s_prime_everywhere_and_s_prime_is_a_cli_alias_only(tmp_path):
    from antelligence.experiments.v42_harness import VERDICT_ARMS
    from antelligence.experiments.v42_report import ARM_ORDER
    from antelligence.worlds.tumor_mri.colony_obs import ALLOWED
    assert "S'" in ARMS and "S-prime" not in ARMS and "S'" in VERDICT_ARMS and "S'" in ARM_ORDER and "S'" in ALLOWED
    with pytest.raises(KeyError):
        resolve_arm("S-prime")
    from tests.v4_2.q1.test_cards_cli import run_cli
    code, _ = run_cli(tmp_path, "--arms", "S-prime", "--seeds", "1")
    assert code == 0 and (tmp_path / "S'" / "phantom" / "seed1" / "record.json").is_file()


def test_colony_world_forwards_scorer_channels_and_the_harness_world_has_no_workaround():
    import numpy as np
    from antelligence.experiments.v42_harness import HarnessWorld
    from antelligence.worlds.tumor_mri.colony_world import ColonyTumorWorld
    case = phantom()
    tc = np.zeros(case.crop.shape, dtype=bool)
    tc[5, 5, 5] = True
    et = np.zeros(case.crop.shape, dtype=bool)
    for cls, extra in ((ColonyTumorWorld, {}), (HarnessWorld, {"arm": "S-solo"})):
        w = cls(case.crop, phantom_cfg(10), 1, scorer_channels={"tc": tc, "et": et}, **extra)
        assert w._scorer.tc[5, 5, 5] and not w._scorer.tc.sum() > 1 and w._scorer.et is not None
    assert "_hidden_fields" not in vars(HarnessWorld) and "_channels" not in vars(HarnessWorld(case.crop, phantom_cfg(10), 1, arm="S-solo"))
