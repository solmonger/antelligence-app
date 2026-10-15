"""Legacy golden: V43Harness with AntRules.v42() and no estimate reproduces V42Harness byte for byte (design readings 1)."""

import json
from pathlib import Path

import pytest

from antelligence.experiments import v42_harness as hz
from antelligence.experiments.v42_harness import V42Harness, bundle_of, load_dev_case
from antelligence.kernel.colony.episode import ColonyConfig
from antelligence.worlds.tumor_mri import WorldConfig
from antelligence.worlds.tumor_mri.ant_rules import AntRules
from tests.v4_2.q1.helpers import noisy_look_factory, phantom, phantom_cfg
from tests.v4_3.a2.helpers import R42
from antelligence.experiments.v43_harness import V43Harness

REPO = Path(__file__).resolve().parents[3]
CASES_ROOT = Path("/Volumes/WD_BLACK/antelligence-v4.1/data_v4_1/cases")
ARMS = ("S-solo", "S-marks", "S-talk")


def same(a, b):
    """Records agree on everything the bundle, the cards and the scripts report."""
    assert json.dumps(bundle_of(a), sort_keys=True) == json.dumps(bundle_of(b), sort_keys=True)
    assert a["cards_final"] == b["cards_final"]
    assert a["script_counters"] == b["script_counters"]
    assert a["board_entries"] == b["board_entries"]
    assert a["metrics"] == b["metrics"]


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("seed", (1, 2))
def test_v42_rules_match_v42_harness_on_the_phantom(arm, seed):
    case = phantom()
    old = V42Harness(phantom_cfg(60), look_fn_factory=noisy_look_factory).run_episode(arm, case, seed)
    new = V43Harness(phantom_cfg(60), None, R42, None, look_fn_factory=noisy_look_factory).run_episode(arm, case, seed)
    same(old, new)
    assert new["ant_rules"]["sha"] == R42.sha() and new["ant_rules"]["rules"]["version"] == "v42"
    assert "answer_rule" not in new["colony"]["params"]              # the v4.2 board header / params are unchanged


def test_legacy_logs_carry_no_est_or_scan_p():
    rec = V43Harness(phantom_cfg(60), None, R42, None, look_fn_factory=noisy_look_factory).run_episode("S-solo", phantom(), 1)
    entries = [e for log in rec["colony"]["reading_logs"].values() for e in log]
    assert entries and all("est" not in e and "scan_p" not in e for e in entries)


def test_v42_cards_have_no_min_resolved_key():
    rec = V43Harness(phantom_cfg(60), None, R42, None, look_fn_factory=noisy_look_factory).run_episode("S-marks", phantom(), 1)
    assert all("min_resolved" not in c for c in rec["cards_final"].values())
    assert all(c["rev"] == "v4.2" for c in rec["cards_final"].values())


# Only the arms the freeze guard exempts (S, S-solo, S') may run on a real development case before the lead freezes the code. S-marks and
# S-talk on a real case would be refused by ``V43Harness.run_episode`` (v4.3 C-fix); the phantom tests above pin them byte for byte.
@pytest.mark.skipif(not (CASES_ROOT / "BRATS_016" / "case.npz").is_file(), reason="development case store not readable here")
@pytest.mark.parametrize("arm", ("S-solo",))
def test_v42_rules_match_v42_harness_on_a_real_dev_case(arm):
    world = WorldConfig.from_yaml(REPO / "config" / "v4_2" / "world.yaml").with_(ticks=60)
    case = load_dev_case("BRATS_016", root=CASES_ROOT)
    colony = ColonyConfig()
    old = V42Harness(world, colony).run_episode(arm, case, 3)
    new = V43Harness(world, colony, AntRules.v42(), None).run_episode(arm, case, 3)
    same(old, new)
