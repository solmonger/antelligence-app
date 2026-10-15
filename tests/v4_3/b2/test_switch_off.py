"""B2 switch off: earlier behaviour stays byte-identical.

* the legacy golden (v4.2 rules) is pinned by ``tests/v4_3/a2/test_legacy_golden.py``; here the rules hash is pinned too;
* gate A: S-solo and S-marks on the phantom with ``AntRules.v43()`` give the SAME bundles, cards, counters and board entries as
  before B2 existed. The digests below were computed at integration HEAD 293e2d6 (before any B2 change), 120 ticks, seeds 1 and 2,
  with ``config/v4_3/world_gateA.yaml`` (the gate-A world) and ``config/v4_3/world.yaml`` (the foraging world, B2 rule off)."""

import pytest

from antelligence.worlds.tumor_mri.ant_rules import AntRules
from tests.v4_3.a2.helpers import R43
from tests.v4_3.b1.helpers import record_digest
from tests.v4_3.b2.helpers import run

GOLDEN = {
    ("config/v4_3/world_gateA.yaml", "S-solo", 1): "1c2c1a932d4a5a92c581f8edb38b61e394f658c7b93d7456b51b1996e19a3d25",
    ("config/v4_3/world_gateA.yaml", "S-solo", 2): "cc2be78100509fe8f16dc018aa49b1d3ff672fa9208db8dbdc254ea4c36375c6",
    ("config/v4_3/world_gateA.yaml", "S-marks", 1): "f660c289f3e45c0748c3edc4796ed5642085b466e6d6721d7f3dce69bfda9bed",
    ("config/v4_3/world_gateA.yaml", "S-marks", 2): "cd27c6701ac332af695f441e5d47bccc1109ada4d28bf176f8ff6ea612bbbc83",
    ("config/v4_3/world.yaml", "S-solo", 1): "97da61513e856c2c28a3ecd31ed2d0e8e7686d2f3eb4d0d808bf5e4b79dade88",
    ("config/v4_3/world.yaml", "S-solo", 2): "cdb792eaa1af9fd5253346e3664fec9334d8df34a387bdc15452a78ac7f3f0fa",
    ("config/v4_3/world.yaml", "S-marks", 1): "9cb2c29e6be6b5ea39c7d5529f406efb215e27ef7253bd3d77cb59c346ec0281",
    ("config/v4_3/world.yaml", "S-marks", 2): "937bfa4d37c0dbfcb93386c23781df81b0fd7067fead05f70f94168f7d9c0712",
}


@pytest.mark.parametrize("rel,arm,seed", sorted(GOLDEN))
def test_gate_a_reproduces_with_the_rule_off(rel, arm, seed):
    rec = run(R43, arm, seed, rel=rel)
    assert record_digest(rec) == GOLDEN[(rel, arm, seed)]
    assert "probes" not in rec["script_counters"] and rec["ant_rules"]["sha"] == R43.sha()
    assert "foraging" not in rec["ant_rules"]["rules"]


def test_the_gate_a_config_file_does_not_mention_the_rule():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[3] / "config" / "v4_3" / "world_gateA.yaml").read_text()
    assert "foraging: true" not in text and "ant_rules: {version: v43}" in text


def test_the_v42_and_v43_rule_hashes_are_unchanged():
    assert AntRules.v42().sha() == "9b9cbcadaa1af354e2b282f818493c72763eefdf5429cbba7ce85017619cefc2"
    assert AntRules.v43().sha() == "ddf0cd1fe7229f97db4f7747f221d7892fdf40b69af5d1b6a7677578ef617855"
