import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from antelligence.experiments import v41_split as sp
from antelligence.experiments.v41_gate import GateError, assert_eval_allowed

ROOT = Path(__file__).resolve().parents[3]
COMMITTED = ROOT / "config" / "v4_1" / "split.json"


def test_split_deterministic_and_sizes():
    ids = [f"BRATS_{i:03d}" for i in range(484)]
    a, b = sp.compute_split(ids), sp.compute_split(list(reversed(ids)))
    assert a == b
    assert (len(a["dev_a"]), len(a["dev_b"]), len(a["eval"]), len(a["reserve"])) == (30, 10, 80, 364)
    assert len(set(sp.split_ids(a))) == 484
    order = sorted(ids, key=lambda c: hashlib.sha256(("v4.1|" + c).encode()).hexdigest())
    assert a["dev_a"] + a["dev_b"] == order[:40] and a["eval"] == order[40:120]


def test_ids_from_dataset_strips_suffix():
    ds = {"training": [{"image": "./imagesTr/BRATS_001.nii.gz", "label": "./labelsTr/BRATS_001.nii.gz"}]}
    assert sp.ids_from_dataset(ds) == ["BRATS_001"]


def test_split_reproduces_committed_file():
    text = COMMITTED.read_text()
    doc = json.loads(text)
    again = sp.compute_split(reversed(sp.split_ids(doc)))
    assert again == doc
    assert sp.dump(again) == text


# --- gate ---------------------------------------------------------------------


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture()
def world(tmp_path):
    doc = sp.compute_split([f"C{i:03d}" for i in range(150)])
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    prereg = repo / "PREREGISTRATION.md"
    prereg.write_text("locked")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"tx_hash": "0xab", "status": 1, "readback_ok": True, "chain_id": 300}))
    g1 = tmp_path / "G1_APPROVED"
    g1.write_text("approved by operator 2026-10-13")
    return dict(doc=doc, repo=repo, prereg=prereg, receipt=receipt, g1=g1)


def _call(w, case, **over):
    args = dict(prereg_path=w["prereg"], anchor_receipt=w["receipt"], g1_file=w["g1"])
    args.update(over)
    return assert_eval_allowed(case, w["doc"], **args)


def test_gate_allows_dev(world, tmp_path):
    for case in (world["doc"]["dev_a"][0], world["doc"]["dev_b"][0]):
        assert assert_eval_allowed(case, world["doc"], tmp_path / "no", tmp_path / "no", tmp_path / "no") == "dev"


def test_gate_refuses_eval_before_lock(world):
    case = world["doc"]["eval"][0]
    with pytest.raises(GateError, match="not committed"):
        _call(world, case)  # prereg exists but untracked
    _git(world["repo"], "add", "PREREGISTRATION.md")
    _git(world["repo"], "commit", "-qm", "prereg")
    assert _call(world, case) == "eval"  # all three satisfied
    world["prereg"].write_text("tampered")
    with pytest.raises(GateError, match="uncommitted"):
        _call(world, case)
    world["prereg"].write_text("locked")
    for bad, msg in (({"tx_hash": "", "status": 1, "readback_ok": True, "chain_id": 300}, "tx_hash"),
                     ({"tx_hash": "0x1", "status": 0, "readback_ok": True, "chain_id": 300}, "status"),
                     ({"tx_hash": "0x1", "status": 1, "readback_ok": False, "chain_id": 300}, "readback"),
                     ({"tx_hash": "0x1", "status": 1, "readback_ok": True, "chain_id": 1}, "chain_id")):
        world["receipt"].write_text(json.dumps(bad))
        with pytest.raises(GateError, match=msg):
            _call(world, case)
    world["receipt"].write_text(json.dumps({"tx_hash": "0x1", "status": 1, "readback_ok": True, "chain_id": 300}))
    world["g1"].unlink()
    with pytest.raises(GateError, match="G1"):
        _call(world, case)
    with pytest.raises(GateError, match="reserve"):
        _call(world, world["doc"]["reserve"][0])
    with pytest.raises(GateError, match="not in the split"):
        _call(world, "UNKNOWN")
