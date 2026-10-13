"""tumor_mri world: determinism, information barrier, mass balance, golden S and S', mapping, entry sites,
labels, decision rule, tuning guard."""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from antelligence.kernel.decision import Answer
from antelligence.kernel.types import NOOP, Intent, World
from antelligence.worlds.tumor_mri import (ALL_LABELS, OBS_KEYS, Crop, ScriptS, TumorMRIWorld, WorldConfig,
                                           build_request, candidates_from_perception, decide_from_answers, oracle_for,
                                           run_episode, sample_entry_sites, sprime_candidate_label)
from antelligence.worlds.tumor_mri.tuning import (SplitError, assert_dev_only, grid_search, load_split)
from tests.v4_1.world.phantom import make_phantom, phantom_config

FIX = Path(__file__).parent / "fixtures"
GOLD = json.loads((FIX / "golden.json").read_text())


@pytest.fixture(scope="module")
def crop() -> Crop:
    return Crop.from_npz(FIX / "phantom.npz")


def run_s(crop, seed, cfg=None):
    w = TumorMRIWorld(crop, cfg or phantom_config(), seed=seed)
    return w, run_episode(w, lambda a: ScriptS())


def run_sp(crop, seed, cfg=None):
    w = TumorMRIWorld(crop, cfg or phantom_config(), seed=seed)
    o = oracle_for(w)
    return w, run_episode(w, lambda a: o)


def test_fixture_matches_generator(crop):
    ref = make_phantom(36)
    assert np.array_equal(crop.brain, ref.brain) and np.array_equal(crop.truth, ref.truth)
    assert np.allclose(crop.p_tumour, ref.p_tumour, atol=1e-6)


def test_is_a_kernel_world(crop):
    w = TumorMRIWorld(crop, phantom_config(), seed=0)
    assert isinstance(w, World)
    assert w.agents() == [f"squad-{i}" for i in range(8)]
    assert w.apply("squad-0", Intent(NOOP), 0).accepted
    assert not w.apply("squad-0", Intent("decide", {"target": "S", "release": 0, "amount": 1}), 3).accepted
    assert not w.apply("squad-0", Intent("decide", {"target": "A", "release": 0, "amount": 1}), 0).accepted  # none offered


def test_determinism_same_seed(crop):
    w1, m1 = run_s(crop, 5)
    w2, m2 = run_s(crop, 5)
    assert m1 == m2
    assert np.array_equal(w1.field.dose, w2.field.dose)
    assert w1.entry_sites == w2.entry_sites and w1.describe() == w2.describe()
    w3, _ = run_s(crop, 6)
    assert w3.entry_sites != w1.entry_sites
    p1, p2 = run_sp(crop, 5)[1], run_sp(crop, 5)[1]
    assert p1 == p2


def _walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}/{k}"
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


def test_no_observation_key_carries_truth(crop):
    # (1) keys are the whitelist, none mentions truth
    w = TumorMRIWorld(crop, phantom_config(), seed=3)
    w.set_candidates(candidates_from_perception(crop.p_tumour, crop.unc, crop.brain))
    obs = w.observe("squad-0", 0).data
    assert set(obs) == set(OBS_KEYS)
    for key in _walk(obs):
        assert not any(t in key.lower() for t in ("truth", "gt", "ground", "label_mask", "wt"))
    # (2) behavioural: replace the truth with its complement; every observation of a whole S episode is identical
    flipped = Crop(brain=crop.brain, p_tumour=crop.p_tumour, unc=crop.unc, truth=~crop.truth,
                   outside_crop_truth_voxels=7, case_id="flipped")
    logs = []
    for c in (crop, flipped):
        world = TumorMRIWorld(c, phantom_config(), seed=3)
        seen = []
        pol = ScriptS()

        class Spy:
            def decide(self, view):
                seen.append(json.dumps(view.observation, sort_keys=True))
                return pol.decide(view)

        run_episode(world, lambda a: Spy())
        logs.append(seen)
    assert logs[0] == logs[1] and len(logs[0]) == 8 * 24
    # (3) the world object exposes truth only through the scorer accessor
    assert not hasattr(w, "truth") and not hasattr(w, "crop")
    assert set(vars(w._percept)) == {"brain", "p_tumour", "unc", "flair", "t1c"}


def test_state_text_has_no_truth_and_fits(crop):
    w = TumorMRIWorld(crop, phantom_config(), seed=3)
    w.set_candidates(candidates_from_perception(crop.p_tumour, crop.unc, crop.brain))
    req = build_request("m", w.observe("squad-1", 0).data, brief="x" * 5000)
    assert len(req.state) <= 1200
    assert tuple(req.questions) == ("target", "release", "amount")
    assert req.questions["target"].options[-2:] == ("S", "X")
    assert req.questions["amount"].options == ("1", "2", "3", "4", "5")


def test_mass_conservation_minus_decay_and_boundary(crop):
    for runner in (run_s, run_sp):
        w, m = runner(crop, 1)
        f = w.field
        assert f.released > 0 and f.decayed > 0 and f.lost_boundary > 0
        assert abs(f.released - (f.mass + f.decayed + f.lost_boundary)) < 1e-9 * f.released
        assert abs(m["mass_residual"]) < 1e-9 * f.released
        assert f.released <= w.config.total_drug + 1e-9


def test_single_puff_loses_mass_at_boundary_only_near_edge():
    from antelligence.worlds.tumor_mri.physics import DrugField
    f = DrugField((21, 21, 21), release_sigma_vox=1.0, diffuse_sigma_vox=1.0, decay=0.02)
    f.deposit((10, 10, 10), 100.0)
    assert f.lost_boundary == pytest.approx(0.0, abs=1e-12)
    f.step()
    assert f.mass == pytest.approx(100.0 * 0.98, rel=1e-3)
    g = DrugField((21, 21, 21), release_sigma_vox=1.0, diffuse_sigma_vox=1.0, decay=0.0)
    g.deposit((0, 10, 10), 100.0)
    assert 25.0 < g.lost_boundary < 35.0  # kernel voxels -1..-3 (~30%) fall outside the crop face
    g.step()
    assert abs(g.mass_balance_residual()) < 1e-10


def _check_golden(m, gold):
    for k in ("coverage", "off_target_share", "lost_boundary", "decayed", "released_total"):
        assert m[k] == pytest.approx(gold[k], rel=1e-6, abs=1e-9), k
    assert m["time_to_50_coverage"] == gold["time_to_50_coverage"]
    assert {str(k): v for k, v in m["coverage_at_queen"].items()} == pytest.approx(gold["coverage_at_queen"], rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("seed", [1, 2])
def test_script_S_golden_phantom(crop, seed):
    w, m = run_s(crop, seed)
    assert [list(s) for s in w.entry_sites] == GOLD[str(seed)]["entry_sites"]
    _check_golden(m, GOLD[str(seed)]["S"])
    assert set(m["coverage_at_queen"]) == {0, 30, 60, 90, 120}


@pytest.mark.parametrize("seed", [1, 2])
def test_oracle_Sprime_golden_phantom(crop, seed):
    w, m = run_sp(crop, seed)
    _check_golden(m, GOLD[str(seed)]["Sprime"])
    assert m["coverage"] > run_s(crop, seed)[1]["coverage"]  # the oracle beats the script


def test_sprime_candidate_mapping():
    cands = [{"label": "A", "centroid_vox": [10.0, 10.0, 10.0]}, {"label": "B", "centroid_vox": [20.0, 10.0, 10.0]}]
    assert sprime_candidate_label((10, 10, 10), cands) == "A"
    assert sprime_candidate_label((15, 10, 10), cands) == "A"  # 10 mm from A, 10 mm from B: tie -> earlier label
    assert sprime_candidate_label((19, 10, 10), cands) == "B"
    assert sprime_candidate_label((15, 16, 10), cands) == "X"  # > 10 mm from both
    assert sprime_candidate_label((16, 10, 10), cands) == "B"  # 8 mm from B
    assert sprime_candidate_label((10, 10, 15.1), cands) == "X"  # just over 10 mm
    assert sprime_candidate_label((0, 0, 0), []) == "X"
    # within-radius boundary is inclusive
    assert sprime_candidate_label((10, 10, 15), cands) == "A"


def test_entry_sites_constraints(crop):
    cfg = phantom_config()
    for seed in range(6):
        sites = sample_entry_sites(crop.brain, crop.p_tumour, seed, voxel_mm=2.0)
        assert len(sites) == len(set(sites)) == 8
        perceived = (crop.p_tumour >= 0.3) & crop.brain
        dist_mm = ndimage.distance_transform_edt(~perceived) * 2.0
        padded = np.pad(crop.brain, 1)
        depth = ndimage.distance_transform_edt(padded)[1:-1, 1:-1, 1:-1]
        for s in sites:
            assert crop.brain[s]
            assert dist_mm[s] >= 10.0
            assert depth[s] <= 2.0 + 1e-9  # outer shell (4 mm)
        assert sites == sample_entry_sites(crop.brain, crop.p_tumour, seed)
    assert sample_entry_sites(crop.brain, crop.p_tumour, 0) != sample_entry_sites(crop.brain, crop.p_tumour, 1)
    w = TumorMRIWorld(crop, cfg, seed=2)
    assert w.entry_sites == sample_entry_sites(crop.brain, crop.p_tumour, 2, shell_mm=cfg.entry_shell_mm)
    with pytest.raises(Exception):
        sample_entry_sites(np.zeros((5, 5, 5), bool), np.zeros((5, 5, 5)), 0)


def test_single_token_labels():
    assert set(ALL_LABELS) == set("ABCDEFSX12345") | {"yes", "no"}
    for lab in ALL_LABELS:
        if lab not in ("yes", "no"):
            assert len(lab) == 1
    tok_files = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--*/snapshots/*/tokenizer.json"))
    try:
        from tokenizers import Tokenizer
    except ImportError:  # pragma: no cover
        pytest.skip("tokenizers package not installed")
    if not tok_files:
        pytest.skip("no tokenizer.json in the offline HF cache")
    checked = 0
    for path in sorted(tok_files):
        try:
            tok = Tokenizer.from_file(path)
        except Exception:
            continue
        if "gemma" not in path.lower():
            continue
        for lab in ALL_LABELS:
            assert len(tok.encode(lab, add_special_tokens=False).ids) == 1, (path, lab)
        checked += 1
    if not checked:
        pytest.skip("no usable gemma tokenizer in the offline HF cache")


def test_decision_rule(crop):
    w = TumorMRIWorld(crop, phantom_config(), seed=0)
    w.set_candidates(candidates_from_perception(crop.p_tumour, crop.unc, crop.brain))
    req = build_request("m", w.observe("squad-0", 0).data)
    opts = req.questions["target"].options
    tied = {o: 1.0 / len(opts) for o in opts}
    ans = {"target": Answer("choice", opts[0], tied), "release": Answer("noul", 0.5, {"yes": 0.5, "no": 0.5}),
           "amount": Answer("score", 2.0, {"0": 0.1, "1": 0.2, "2": 0.5, "3": 0.1, "4": 0.1})}
    d = decide_from_answers(ans, req)
    assert d["target"] == opts[0] and d["release"] == 1 and d["amount"] == 3  # tie -> earlier; P(yes)>=0.5 releases
    ans["release"] = Answer("noul", 0.49, {"yes": 0.49, "no": 0.51})
    assert decide_from_answers(ans, req)["release"] == 0


def test_stay_keeps_target_and_explore_goes_to_unvisited_region(crop):
    cands = candidates_from_perception(crop.p_tumour, crop.unc, crop.brain)
    assert len(cands) >= 2
    w = TumorMRIWorld(crop, phantom_config(), seed=0)
    w.set_candidates(cands)
    start = w.squad_positions()[0]
    w.apply("squad-0", Intent("decide", {"target": "S", "release": 0, "amount": 1}), 0)
    for t in range(5):
        w.step_environment(t)
    assert w.squad_positions()[0] == start  # no target yet: stays put
    w.apply("squad-0", Intent("decide", {"target": "A", "release": 0, "amount": 1}), 5)
    for t in range(5, 10):
        w.step_environment(t)
    mid = w.squad_positions()[0]
    assert mid != start and max(abs(a - b) for a, b in zip(mid, start)) <= 5  # <= one voxel per tick
    w.apply("squad-0", Intent("decide", {"target": "S", "release": 0, "amount": 1}), 10)
    assert w._squads[0].target == "A"  # Stay keeps the current target
    for t in range(10, 15):
        w.step_environment(t)
    assert w.squad_positions()[0] != mid  # ... and keeps walking toward it
    # Explore: nearest unvisited perceived region
    w2 = TumorMRIWorld(crop, phantom_config(), seed=0)
    w2.set_candidates(cands)
    pos = w2.squad_positions()[0]
    near = min(cands, key=lambda c: (np.linalg.norm(np.subtract(pos, c.centroid_vox)), c.label))
    w2.apply("squad-0", Intent("decide", {"target": "X", "release": 0, "amount": 1}), 0)
    assert w2._squads[0].goal == w2._paths.nearest_brain_voxel(near.centroid_vox)
    w2._visited_regions.add(near.region_id)  # visited -> skipped
    w2.apply("squad-0", Intent("decide", {"target": "X", "release": 0, "amount": 1}), 5)
    other = [c for c in cands if c.region_id != near.region_id]
    pos = w2.squad_positions()[0]
    near2 = min(other, key=lambda c: (np.linalg.norm(np.subtract(pos, c.centroid_vox)), c.label))
    assert w2._squads[0].goal == w2._paths.nearest_brain_voxel(near2.centroid_vox)
    # no unvisited region -> highest-uncertainty cluster
    w2._visited_regions.update(c.region_id for c in cands)
    w2.apply("squad-0", Intent("decide", {"target": "X", "release": 0, "amount": 1}), 10)
    assert w2._squads[0].goal is not None
    allpts = np.concatenate(w2._clusters)
    assert np.linalg.norm(allpts - np.array(w2._squads[0].goal), axis=1).min() <= 6.0  # at a high-uncertainty cluster


def test_script_S_rule(crop):
    obs = dict(local_p_tumour=0.6, payload=10.0, candidates=[
        {"label": "A", "p_tumour": 0.9, "volume_mm3": 1000, "dist_mm": 50.0, "visited": False},
        {"label": "B", "p_tumour": 0.6, "volume_mm3": 1000, "dist_mm": 10.0, "visited": False},
        {"label": "C", "p_tumour": 0.9, "volume_mm3": 9000, "dist_mm": 5.0, "visited": True}])
    from antelligence.kernel.types import LocalView
    it = ScriptS().decide(LocalView("squad-0", 0, 0, 0, obs))
    assert it.params == {"target": "B", "release": 1, "amount": 3}  # C visited; B beats A on p*vol/dist
    obs["local_p_tumour"] = 0.49
    assert ScriptS().decide(LocalView("squad-0", 0, 0, 0, obs)).params["release"] == 0


def test_off_target_share_definition(crop):
    w = TumorMRIWorld(crop, phantom_config(), seed=0)
    assert w.metrics()["off_target_share"] == 0.0
    # a puff in the middle of the tumour: most mass stays/decays in WT early on, so off-target is small
    centre = tuple(int(c) for c in np.argwhere(crop.truth).mean(axis=0).round())
    w.field.deposit(centre, 100.0)
    w.step_environment(0)
    early = w.metrics()["off_target_share"]
    for t in range(1, 60):
        w.step_environment(t)
    m = w.metrics()
    assert 0.0 < early < m["off_target_share"] <= 1.0
    sc = w.scorer_truth()
    on = w.field.decayed_in_region + w.field.mass_in(sc.truth)
    assert m["off_target_share"] == pytest.approx(1.0 - on / 100.0)
    assert "lost_boundary_share" in m


def test_release_payload_accounting(crop):
    w = TumorMRIWorld(crop, phantom_config(total_drug=800.0), seed=0)
    share = 100.0
    assert w.apply("squad-0", Intent("decide", {"target": "S", "release": 1, "amount": 3}), 0).accepted
    assert w._squads[0].payload == pytest.approx(share * 0.5)
    w.apply("squad-0", Intent("decide", {"target": "S", "release": 1, "amount": 2}), 5)
    assert w._squads[0].payload == pytest.approx(share * 0.5 * 0.75)


def test_local_patches(crop):
    c = make_phantom(36, flair=True)
    w = TumorMRIWorld(c, phantom_config(), seed=0)
    p = w.local_patches("squad-0")
    assert p["flair"].shape == (64, 64) and p["t1c"].shape == (64, 64)


def test_coverage_counts_outside_crop_truth(crop):
    a = TumorMRIWorld(crop, phantom_config(), seed=0).scorer_truth()
    b = TumorMRIWorld(Crop(brain=crop.brain, p_tumour=crop.p_tumour, unc=crop.unc, truth=crop.truth,
                           outside_crop_truth_voxels=0), phantom_config(), seed=0).scorer_truth()
    dose = np.full(crop.brain.shape, 1e9)
    assert b.coverage(dose) == pytest.approx(1.0)
    assert a.coverage(dose) == pytest.approx(int(crop.truth.sum()) / (int(crop.truth.sum()) + 40))
    assert a.truth_share_outside_crop == pytest.approx(40 / (int(crop.truth.sum()) + 40))


def test_speed_100_cube():
    n = 100
    g = np.indices((n, n, n)).astype(np.float32)
    c = (n - 1) / 2
    brain = ((g[0] - c) ** 2 + (g[1] - c) ** 2 + (g[2] - c) ** 2) <= 48 ** 2
    truth = ((g[0] - 55) ** 2 + (g[1] - 50) ** 2 + (g[2] - 45) ** 2) <= 10 ** 2
    p = ndimage.gaussian_filter(truth.astype(np.float64), 1.5) * 1.8
    big = Crop.from_arrays(dict(brain=brain, p_tumour=np.clip(p, 0, 1), unc=np.clip(4 * p * (1 - p), 0, 1), truth=truth))
    t0 = time.perf_counter()
    w = TumorMRIWorld(big, phantom_config(), seed=0)
    run_episode(w, lambda a: ScriptS())
    assert time.perf_counter() - t0 < 4.0  # target < 2 s; margin for loaded CI machines


# --- tuning guard ---------------------------------------------------------------------------------------------


def test_tune_refuses_eval_case(tmp_path, capsys):
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"dev_a": ["BRATS_001"], "dev_b": ["BRATS_003"], "eval": ["BRATS_002"],
                                      "reserve": ["BRATS_004"]}))
    split = load_split(split_path)
    assert_dev_only(["BRATS_001", "BRATS_003"], split)  # dev = dev_a + dev_b
    with pytest.raises(SplitError):
        assert_dev_only(["BRATS_004"], split)  # reserve
    with pytest.raises(SplitError):
        assert_dev_only(["BRATS_001", "BRATS_002"], split)
    with pytest.raises(SplitError):
        assert_dev_only(["BRATS_999"], split)  # unknown cases are refused too
    import importlib.util
    spec = importlib.util.spec_from_file_location("tune_script", Path(__file__).parents[3] / "scripts/v41_tune_world.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main(["--split", str(split_path), "--cases", "BRATS_002", "--cases-dir", str(tmp_path),
                   "--config", str(tmp_path / "w.yaml")])
    assert rc == 3 and "REFUSED" in capsys.readouterr().err
    rc = mod.main(["--split", str(split_path), "--cases", "BRATS_004", "--cases-dir", str(tmp_path),
                   "--config", str(tmp_path / "w.yaml")])
    assert rc == 3
    assert not (tmp_path / "w.yaml").exists()


def test_grid_search_bands_and_cache():
    calls = []

    def fake(crop_, cfg, seed):
        calls.append(cfg.dose_threshold / cfg.total_drug)
        r = cfg.dose_threshold / cfg.total_drug
        return min(0.9, 0.2 / (r * 10)), min(0.95, 0.5 / (r * 10))

    crops = {"a": None, "b": None, "c": None}
    res = grid_search(crops, WorldConfig(), [100.0, 200.0, 400.0, 800.0], [5.0, 10.0], evaluate=fake)
    assert res["status"] in ("ok", "fallback_widened_band", "no_feasible_point")
    assert len(set(round(c, 9) for c in calls)) * 3 == len(calls)  # each unique ratio evaluated once per case
    sat = grid_search(crops, WorldConfig(), [100.0], [5.0], evaluate=lambda c, cfg, s: (0.9, 0.9))
    assert sat["status"] == "no_feasible_point"
