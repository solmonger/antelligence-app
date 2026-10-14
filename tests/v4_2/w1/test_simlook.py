"""W1: SimLook (goal 5.1) and the cell-table builder / board-look filter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from antelligence.worlds.tumor_mri import Crop, TumorMRIWorld
from antelligence.worlds.tumor_mri import simlook as sl
from tests.v4_1.world.phantom import oracle_config

PHANTOM = Path(__file__).resolve().parents[2] / "v4_1" / "world" / "fixtures" / "phantom.npz"
REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def crop() -> Crop:
    return Crop.from_npz(PHANTOM)


def _looks(seed=0, big=400, small=30):
    """Synthetic labelled looks: every cell has `big` looks, except the (in_0_4, scan 0) and (out_4_10, all) cells."""
    r = np.random.default_rng(seed)
    b, s, v = [], [], []
    for band in range(sl.N_BANDS):
        for scan in range(sl.N_SCAN):
            n = small if (band, scan) == (3, 0) else big
            if band == 1:
                n = 12  # out_4_10 is short everywhere
            centre = 0.15 + 0.7 * (band >= 3) + 0.05 * scan
            v += list(np.clip(r.normal(centre, 0.1, n), 0, 1) * 1e6)
            b += [band] * n
            s += [scan] * n
    return b, s, v


def test_bands_and_scan_bands():
    t = np.zeros((20, 20, 20), bool)
    t[5:15, 5:15, 5:15] = True
    band = sl.distance_band_index(t, 2.0)
    assert band[10, 10, 10] == 4 and band[5, 10, 10] == 3 and band[4, 10, 10] == 2  # boundary voxel = 2 mm
    assert band[0, 10, 10] == 1 and band[10, 10, 10] == 4
    assert list(sl.scan_band_index(np.array([0.0, 0.19, 0.2, 0.5, 0.79, 0.8, 1.0]))) == [0, 0, 1, 2, 2, 3, 3]
    far = np.zeros((40, 4, 4), bool)
    far[0] = True
    assert sl.distance_band_index(far, 2.0)[39, 0, 0] == 0   # > 10 mm outside


def test_cell_table_merges_are_logged_and_resolve():
    t = sl.build_cell_table(*_looks())
    assert t["n_looks"] == sum(c["n_raw"] for c in t["cells"].values())
    log = {(m["merge"], m.get("cell")) for m in t["merge_log"]}
    assert ("scan_band_dropped", "d3s0") in log                       # 30 looks: drop the scan-p band
    assert t["cells"]["d3s0"]["entry"] == "d3" and t["entries"]["d3"]["n"] == 30 + 3 * 400
    assert t["cells"]["d3s1"]["entry"] == "d3s1"                       # big cells keep their own entry
    merged = [m for m in t["merge_log"] if m["merge"] == "distance_bands"]
    assert merged and "out_4_10" in merged[0]["bands"]                  # 48 looks in the band: merged with a neighbour
    for c in t["cells"].values():
        assert c["entry"] in t["entries"] and t["entries"][c["entry"]]["n"] >= sl.MIN_CELL
    assert all(len(e["quantiles_ppm"]) == sl.N_LEVELS for e in t["entries"].values())
    assert t == sl.build_cell_table(*_looks())                          # deterministic


def test_simlook_marginals_follow_the_cell_quantiles(crop):
    table = sl.build_cell_table(*_looks())
    s = sl.SimLook("phantom", crop.truth, crop.p_tumour, 2.0, table)
    band = sl.distance_band_index(crop.truth, 2.0)
    vals = s._field
    assert 0.0 <= vals.min() and vals.max() <= 1.0
    inside, outside = vals[band == 4], vals[band == 0]
    assert inside.mean() > outside.mean() + 0.3  # inside reads high, far outside low, as in the synthetic looks
    ent = table["entries"][table["cells"][sl.cell_key(4, 3)]["entry"]]
    sel = (band == 4) & (sl.scan_band_index(crop.p_tumour) == 3)
    assert sel.sum() > 100
    assert np.median(vals[sel]) == pytest.approx(ent["quantiles_ppm"][100] / 1e6, abs=0.03)


def test_simlook_identical_across_ants_ticks_seeds_arms(crop):
    table = sl.build_cell_table(*_looks())
    s = sl.SimLook("phantom", crop.truth, crop.p_tumour, 2.0, table)
    pos = (18, 16, 17)
    vals = {s(a, pos, t) for a in range(8) for t in (0, 5, 60, 115)}
    assert len(vals) == 1
    s2 = sl.SimLook("phantom", crop.truth, crop.p_tumour, 2.0, table)   # rebuilt: same field
    assert np.array_equal(s._field, s2._field)
    other = sl.SimLook("other_case", crop.truth, crop.p_tumour, 2.0, table)
    assert not np.array_equal(s._field, other._field)                   # per case
    # injected as look_fn: every squad, tick and seed observes the same value at a voxel
    seen = set()
    for seed in (1, 2):
        w = TumorMRIWorld(crop, oracle_config(), seed=seed, look_fn=s)
        for sq in w.agents():
            w._squads[int(sq.split("-")[1])].pos = pos
            for t in (0, 25):
                seen.add(w.observe(sq, t).data["local_look"])
    assert seen == {s(0, pos, 0)}


def test_simlook_identical_across_processes(crop, tmp_path):
    table = sl.build_cell_table(*_looks())
    tp = tmp_path / "t.json"
    sl.save_table(table, tp)
    code = (
        "import hashlib, sys; import numpy as np\n"
        "from pathlib import Path\n"
        "from antelligence.worlds.tumor_mri import Crop, simlook as sl\n"
        f"c = Crop.from_npz(Path({str(PHANTOM)!r}))\n"
        f"s = sl.SimLook('phantom', c.truth, c.p_tumour, 2.0, sl.load_table({str(tp)!r}))\n"
        "print(hashlib.sha256(s._field.tobytes()).hexdigest())\n")
    outs = {subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True,
                           env={"PYTHONPATH": str(REPO), "HF_HUB_OFFLINE": "1", "PATH": "/usr/bin"}).stdout.strip()
            for _ in range(2)}
    here = sl.SimLook("phantom", crop.truth, crop.p_tumour, 2.0, sl.load_table(tp))
    assert outs == {hashlib.sha256(here._field.tobytes()).hexdigest()}


def test_uniform_field_is_uniform_and_smooth():
    q = sl.uniform_field((30, 30, 30), "c1")
    assert q.min() > 0 and q.max() < 1
    assert np.allclose(np.sort(q.ravel()), (np.arange(q.size) + 0.5) / q.size)  # exact rank re-uniformisation
    rough = np.abs(np.diff(sl.rng("f1_field", "c1").random((30, 30, 30)), axis=0)).mean()
    assert np.abs(np.diff(q, axis=0)).mean() < 0.5 * rough                        # neighbouring voxels correlate


# --- the board-look filter in scripts/v42_simlook_cells.py --------------------------------------------------------------


def _script():
    spec = importlib.util.spec_from_file_location("v42_simlook_cells", REPO / "scripts" / "v42_simlook_cells.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_calib(tmp_path, rows):
    (tmp_path / "export").mkdir()
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    ev = [{"type": "evidence", "tick": 5 * k, "author": {"agent": "squad-0"},
           "payload": {"squad": 0, "pos_vox": [1, 2, 3], "look_p_ppm": lp, "p_tumour_ppm": 100}} for k, (lp, _) in enumerate(rows)]
    (tmp_path / "export" / "r1.jsonl").write_text(
        "\n".join(json.dumps(x) for x in [{"arm": "SW-B:calibration", "split": "dev"}] + ev))
    dec = [{"squad": 0, "round": k, "tick": 5 * k, "look_p_ppm": dl, "error": False} for k, (_, dl) in enumerate(rows)]
    (tmp_path / "runs" / "r1" / "decisions.jsonl").write_text("\n".join(json.dumps(x) for x in dec))


def test_board_looks_keep_only_matching_decision_rows(tmp_path):
    _fake_calib(tmp_path, [(1000, 1000), (2000, 2500), (3000, None), (4000, 4000)])
    drops: Counter = Counter()
    kept = _script().board_looks(tmp_path, "r1", drops)
    assert [k[2] for k in kept] == [1000, 4000]
    assert drops == {"value_mismatch": 1, "decision_look_null": 1}


def test_script_refuses_non_dev_or_wrong_arm(tmp_path):
    _fake_calib(tmp_path, [(1000, 1000)])
    p = tmp_path / "export" / "r1.jsonl"
    p.write_text(p.read_text().replace('"split": "dev"', '"split": "eval"'))
    with pytest.raises(SystemExit):
        _script().board_looks(tmp_path, "r1", Counter())


def test_committed_table_is_counts_only_and_dev_sourced():
    d = REPO / "docs" / "research" / "v4.2-20261014" / "simlook"
    t = sl.load_table(d / "cells_v4_2.json")
    assert t["source"] == "boards" and t["meta"]["split"] == "dev" and t["meta"]["kept"] == t["n_looks"] == 7680
    assert t["meta"]["dropped"] == {}
    blob = (d / "cells_v4_2.json").read_text() + (d / "reliability.json").read_text()
    assert "pos_vox" not in blob and "BRATS" not in blob and "case_id" not in blob
    rel = json.loads((d / "reliability.json").read_text())
    assert set(rel["auroc_by_zone"]) == {"edge", "mid", "far"} and set(rel["per_band"]) == set(sl.BANDS)
