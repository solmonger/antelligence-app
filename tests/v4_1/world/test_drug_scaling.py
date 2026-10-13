"""drug_mode per_perceived_volume: the budget is k x V_perceived (perception only); widened band matches goal 4.2."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from antelligence.worlds.tumor_mri import ScriptS, TumorMRIWorld, WorldConfig, oracle_for, run_episode
from antelligence.worlds.tumor_mri.config import perceived_volume_vox
from antelligence.worlds.tumor_mri.tuning import STRICT, WIDENED, drug_for_coverage, gap_by_truth_bin
from tests.v4_1.world.phantom import make_phantom, phantom_config


def _cfg(**kw):
    return phantom_config(drug_mode="per_perceived_volume", drug_scale_k=0.5, **kw)


def test_drug_scales_with_perceived_volume_not_truth():
    crop = make_phantom()
    v = float(crop.p_tumour.sum())
    assert perceived_volume_vox(crop.p_tumour) == pytest.approx(v)
    w = TumorMRIWorld(crop, _cfg(), seed=1)
    assert w.total_drug == pytest.approx(0.5 * v)
    assert sum(w.squad_payloads()) == pytest.approx(0.5 * v)
    # perturb the truth (scorer side only): the budget, the observations and the run are unchanged
    rng = np.random.default_rng(0)
    truth2 = crop.truth ^ (rng.random(crop.truth.shape) < 0.2)
    crop2 = dataclasses.replace(crop, truth=truth2, outside_crop_truth_voxels=crop.outside_crop_truth_voxels + 500)
    w2 = TumorMRIWorld(crop2, _cfg(), seed=1)
    assert w2.total_drug == w.total_drug
    assert w2.squad_payloads() == w.squad_payloads()
    assert w2.observe("squad-0", 0) == w.observe("squad-0", 0)
    # perturb perception: the budget follows it, linearly in the scale
    crop3 = dataclasses.replace(crop, p_tumour=np.clip(crop.p_tumour * 0.5, 0, 1))
    assert TumorMRIWorld(crop3, _cfg(), seed=1).total_drug == pytest.approx(0.5 * float(crop3.p_tumour.sum()))
    # the oracle plans with the resolved budget
    o = oracle_for(w)
    assert o.cfg.total_drug == pytest.approx(w.total_drug) and o.cfg.drug_mode == "fixed"
    run_episode(w, lambda a: ScriptS())


def test_fixed_mode_unchanged_and_validation():
    crop = make_phantom()
    cfg = phantom_config(total_drug=400.0)
    assert cfg.drug_mode == "fixed"
    assert TumorMRIWorld(crop, cfg, seed=0).total_drug == 400.0
    with pytest.raises(ValueError):
        WorldConfig(drug_mode="per_perceived_volume")  # k missing
    with pytest.raises(ValueError):
        WorldConfig(drug_mode="bogus")
    assert WorldConfig.from_dict(_cfg().to_dict()).to_dict() == _cfg().to_dict()
    assert "total_drug" not in _cfg().to_dict()
    with pytest.raises(ValueError, match="total_drug"):
        WorldConfig.from_dict({**_cfg().to_dict(), "total_drug": 3100.0})
    import yaml
    from pathlib import Path
    shipped = yaml.safe_load((Path(__file__).parents[3] / "config/v4_1/world.yaml").read_text())["world"]
    assert shipped["drug_mode"] == "per_perceived_volume" and "total_drug" not in shipped


def test_widened_band_matches_goal():
    assert STRICT["sprime"] == (0.70, 0.80) and STRICT["s"] == (0.30, 0.50)
    assert WIDENED["sprime"] == (0.60, 0.85) and WIDENED["s"] == (0.20, 0.55)
    assert STRICT["gap"] == WIDENED["gap"] == 0.15 and STRICT["gap_frac"] == WIDENED["gap_frac"] == 0.80


def test_report_helpers():
    assert drug_for_coverage([(1.0, 0.2), (2.0, 0.6)], 0.5) == pytest.approx(2 ** 0.75)
    assert drug_for_coverage([(1.0, 0.2)], 0.5) is None
    rows = [{"truth_vox": 1000, "s": 0.5, "sprime": 0.8}, {"truth_vox": 20000, "s": 0.5, "sprime": 0.55}]
    out = {b["bin"]: b for b in gap_by_truth_bin(rows)}
    assert out["<3k"]["gap_ge_0.15_fraction"] == 1.0 and out[">15k"]["gap_ge_0.15_fraction"] == 0.0 and out["3-8k"]["n"] == 0
