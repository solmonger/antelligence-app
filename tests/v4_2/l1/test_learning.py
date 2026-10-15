import importlib.util
import sys
from pathlib import Path

import pytest

from antelligence.experiments import v42_harness as hmod
from antelligence.kernel.colony.cards import Card
from antelligence.kernel.colony.notebook import Notebook
from tests.v4_2.q1.helpers import fake_look_factory, phantom, phantom_cfg
from antelligence.experiments.v42_harness import V42Harness
import antelligence.experiments.v42_arms  # noqa: F401

_spec = importlib.util.spec_from_file_location("v42_learning", Path(__file__).resolve().parents[3] / "scripts" / "v42_learning.py")
L = importlib.util.module_from_spec(_spec)
sys.modules["v42_learning"] = L
_spec.loader.exec_module(L)


def _h():
    return V42Harness(phantom_cfg(ticks=40), look_fn_factory=fake_look_factory, with_board=False)


def test_carry_changes_starting_thresholds_and_restores_harness(tmp_path):
    h, case = _h(), phantom()
    orig = hmod.ColonyKit
    carried = {f"ant-{i:02d}": {"dose": 0.2, "explore": 0.9, "verify": 0.3} for i in range(8)}
    with L.carry_thresholds(carried):
        kit = hmod.ColonyKit("S-marks", 8, None, sharing=True)
        assert kit.cards[3].thresholds == carried["ant-03"] and kit.scripts[3].state.thresholds.theta == carried["ant-03"]
    assert hmod.ColonyKit is orig
    assert hmod.ColonyKit("S-marks", 8, None, sharing=True).cards[0].thresholds == {"dose": .5, "explore": .5, "verify": .5}


def test_save_load_roundtrip_carries_only_thresholds(tmp_path):
    h, case = _h(), phantom()
    rec = h.run_episode("S-marks", case, 1)
    L.save_cards(rec, tmp_path)
    got = L.load_carried(tmp_path)
    assert got == {a: c["thresholds"] for a, c in rec["cards_final"].items()}
    assert (tmp_path / "cards" / "ant-00.json").is_file()


def test_carried_run_differs_only_via_thresholds_and_default_path_unchanged():
    h, case = _h(), phantom()
    base = h.run_episode("S-marks", case, 1)
    again = h.run_episode("S-marks", case, 1)
    assert base["metrics"] == again["metrics"]                     # no leakage after a carried block
    carried = {f"ant-{i:02d}": {"dose": 0.1, "explore": 1.0, "verify": 0.1} for i in range(8)}
    with L.carry_thresholds(carried):
        c = h.run_episode("S-marks", case, 1)
    assert c["cards_final"]["ant-00"]["cross_run"]["runs"] == 1    # nothing but thresholds carried
    assert h.run_episode("S-marks", case, 1)["metrics"] == base["metrics"]


def test_notebook_round_trip(tmp_path):
    r = L.notebook_round_trip(tmp_path)
    assert r["ok"]
    names = [s["step"] for s in r["steps"]]
    assert names == ["write", "cap", "archive", "freeze", "reload"]
    assert r["steps"][1]["refused_with"] == "cap exceeded"
    assert Notebook.load(tmp_path / "notebook_ant-00.json").frozen


def test_eval_case_refused(tmp_path):
    from antelligence.experiments.v42_harness import EvalRefused
    with pytest.raises(EvalRefused):
        L.drive.guard_dev_cases(["BRATS_NOT_A_DEV_CASE"])
