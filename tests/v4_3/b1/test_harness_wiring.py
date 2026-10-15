"""B1: the V43 harness picks the foraging mark rule and the queen's uptake scaling from the world config (sharing arms only)."""

import collections

import pytest

from antelligence.experiments.v43_harness import V43Harness
from antelligence.worlds.tumor_mri.ant_rules import AntRules
from antelligence.worlds.tumor_mri.colony_queen import ScriptedQueen
from antelligence.worlds.tumor_mri.foraging import automatic_marks_v43
from antelligence.worlds.tumor_mri.colony_policies import automatic_marks
from tests.v4_2.q1.helpers import noisy_look_factory
from tests.v4_3.a2.helpers import R42, R43, estimate
from tests.v4_3.b1.helpers import load_yaml_configs, phantom_with_channels, record_digest
from tests.v4_3.b1.test_v42_identity import V42


def v43(rel, rules=R43, ticks=120):
    w, c, _ = load_yaml_configs(rel, ticks)
    return V43Harness(w, c, rules, estimate() if rules.use_combined else None, look_fn_factory=noisy_look_factory)


@pytest.fixture
def queen_flags(monkeypatch):
    seen = []
    orig = ScriptedQueen.__call__

    def spy(self, tick, inputs):
        seen.append(self.uptake_scaling)
        return orig(self, tick, inputs)

    monkeypatch.setattr(ScriptedQueen, "__call__", spy)
    return seen


def mark_kinds(rec):
    return collections.Counter(m["mark_kind"] for m in rec["colony"]["marks"])


def test_world_yaml_is_the_foraging_config():
    w, c, doc = load_yaml_configs("config/v4_3/world.yaml")
    f = w.facts
    assert f.f1_simlook and f.f2 and f.f2_model == "foraging" and f.uptake_level == "mid" and f.foraging
    assert f.f3 and f.f3_decay == 0.04 and f.f3_share == 0.1                       # F3 unchanged from v4.2
    assert f.u_core == 1.0                                                          # u_core is out of use
    assert doc["colony"]["ant_rules"] == {"version": "v43"} and AntRules.from_dict(doc["colony"]["ant_rules"]).version == "v43"
    assert c.half_life["good_uptake"] == 60 and "good_uptake" not in c.to_dict()["half_life"]
    base, _, _ = load_yaml_configs("config/v4_2/world.yaml")
    assert {k: v for k, v in w.to_dict().items() if k != "facts"} == {k: v for k, v in base.to_dict().items() if k != "facts"}


@pytest.mark.parametrize("arm,scaled", [("S-solo", False), ("S-marks", True), ("S-talk", True)])
def test_foraging_config_switches_on_the_rule_and_the_queen_scaling(arm, scaled, queen_flags):
    h = v43("config/v4_3/world.yaml")
    assert h.auto_mark_rule() is automatic_marks_v43
    rec = h.run_episode(arm, phantom_with_channels(), 1)
    assert set(queen_flags) == {scaled}
    kinds = mark_kinds(rec)
    assert kinds["good_uptake"] > 0 and set(kinds) <= {"dosed", "dose_ok", "slow_uptake", "washout", "good_uptake", "tumour_here", "not_tumour", "need_more"}
    st = rec["metrics"]["facts"]
    assert st["uptake_good"] + st["uptake_middling"] + st["uptake_poor"] == st["uptake_readings"] > 0 and st["uptake_slow"] == 0
    rows = rec["colony"]["need_per_zone"]
    assert all(("uptake_factor_milli" in r) == scaled for r in rows)
    assert rec["total_drug"] > 0 and abs(rec["metrics"]["mass_residual"]) < 1e-9


def test_sharing_arm_queen_changes_zone_need_with_the_marks(queen_flags):
    rec = v43("config/v4_3/world.yaml").run_episode("S-marks", phantom_with_channels(), 1)
    factors = {r["uptake_factor_milli"] for r in rec["colony"]["need_per_zone"]}
    assert factors <= {500, 1000, 1250} and factors != {1000}


@pytest.mark.parametrize("rel", ["config/v4_2/world.yaml", "config/v4_3/world_gateA.yaml"])
def test_old_configs_keep_the_old_rule_and_no_scaling(rel, queen_flags):
    h = v43(rel)
    assert h.auto_mark_rule() is automatic_marks
    rec = h.run_episode("S-marks", phantom_with_channels(), 1)
    assert set(queen_flags) == {False} and "good_uptake" not in mark_kinds(rec)
    assert all("uptake_factor_milli" not in r for r in rec["colony"]["need_per_zone"])
    assert "uptake_good" not in (rec["metrics"].get("facts") or {})


@pytest.mark.parametrize("arm", ["S", "S-solo", "S-marks", "S-talk"])
def test_v43_harness_with_v42_rules_on_the_v42_config_matches_the_base_commit(arm):
    rec = v43("config/v4_2/world.yaml", R42).run_episode(arm, phantom_with_channels(), 1)
    # same digest as the v4.2 harness at the base commit, except the v4.3 harness adds its own record keys (not in the digest)
    assert record_digest(rec) == V42[(arm, 1)]
