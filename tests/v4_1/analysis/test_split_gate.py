import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from antelligence.experiments import v41_split as sp
from antelligence.experiments.v41_gate import GateError, assert_eval_allowed

ROOT = Path(__file__).resolve().parents[3]
COMMITTED = ROOT / "config" / "v4_1" / "split.json"
TX = "0x" + "ab" * 32


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


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture()
def world(tmp_path):
    doc = sp.compute_split([f"C{i:03d}" for i in range(150)])
    repo = tmp_path / "repo"
    (repo / "config" / "v4_1").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    split = repo / "config" / "v4_1" / "split.json"
    split.write_text(sp.dump(doc))
    prereg = repo / "PREREGISTRATION.md"
    prereg.write_text(f"locked\n| split file sha256 | `{_sha(split)}` |\n")
    receipt = tmp_path / "receipt.json"
    g1 = tmp_path / "G1_APPROVED"
    g1.write_text("approved by operator 2026-10-13")
    w = dict(doc=doc, repo=repo, prereg=prereg, receipt=receipt, g1=g1, split=split)
    write_receipt(w)
    return w


def write_receipt(w, **over):
    rec = dict(tx_hash=TX, status=1, readback_ok=True, chain_id=300, prereg_sha256=_sha(w["prereg"]))
    rec.update(over)
    w["receipt"].write_text(json.dumps(rec))


def _call(w, case, **over):
    args = dict(repo_root=w["repo"], prereg_path=w["prereg"], anchor_receipt=w["receipt"], g1_file=w["g1"])
    args.update(over)
    return assert_eval_allowed(case, **args)


def _commit(w):
    _git(w["repo"], "add", "-A")
    _git(w["repo"], "commit", "-qm", "lock")


def test_gate_allows_dev(world, tmp_path):
    for case in (world["doc"]["dev_a"][0], world["doc"]["dev_b"][0]):
        assert _call(world, case, prereg_path=tmp_path / "no", anchor_receipt=tmp_path / "no", g1_file=tmp_path / "no") == "dev"


def test_gate_refuses_eval_before_lock(world):
    case = world["doc"]["eval"][0]
    with pytest.raises(GateError, match="not committed"):
        _call(world, case)
    _commit(world)
    assert _call(world, case) == "eval"
    world["prereg"].write_text(world["prereg"].read_text() + "tampered")
    with pytest.raises(GateError, match="uncommitted"):
        _call(world, case)
    _git(world["repo"], "add", "-A")  # staged but not committed is still a modification
    with pytest.raises(GateError, match="uncommitted"):
        _call(world, case)
    _git(world["repo"], "reset", "-q", "--hard")
    write_receipt(world)
    world["g1"].unlink()
    with pytest.raises(GateError, match="G1"):
        _call(world, case)
    world["g1"].write_text("ok")
    with pytest.raises(GateError, match="reserve"):
        _call(world, world["doc"]["reserve"][0])
    with pytest.raises(GateError, match="not in the split"):
        _call(world, "UNKNOWN")


@pytest.mark.parametrize("over,msg", [
    ({"tx_hash": ""}, "tx_hash"), ({"tx_hash": "0x" + "AB" * 32}, "tx_hash"), ({"tx_hash": "0xab"}, "tx_hash"), ({"tx_hash": "0x" + "ab" * 32 + "\n"}, "tx_hash"),
    ({"status": 0}, "status"), ({"status": True}, "status"), ({"status": "1"}, "status"),
    ({"readback_ok": False}, "readback"), ({"chain_id": 1}, "chain_id"), ({"chain_id": "300"}, "chain_id"),
    ({"prereg_sha256": "0" * 64}, "prereg_sha256"), ({"prereg_sha256": None}, "prereg_sha256"),
])
def test_gate_receipt_strict(world, over, msg):
    _commit(world)
    write_receipt(world, **over)
    with pytest.raises(GateError, match=msg):
        _call(world, world["doc"]["eval"][0])


def test_gate_receipt_requires_prereg_sha_key(world):
    _commit(world)
    rec = json.loads(world["receipt"].read_text())
    del rec["prereg_sha256"]
    world["receipt"].write_text(json.dumps(rec))
    with pytest.raises(GateError, match="prereg_sha256"):
        _call(world, world["doc"]["eval"][0])


def test_gate_split_checks(world, tmp_path):
    case = world["doc"]["eval"][0]
    _commit(world)
    assert _call(world, case) == "eval"
    # the split is loaded from the repo; changing it breaks the hash recorded in the prereg
    doc = json.loads(world["split"].read_text())
    doc["reserve"].append(doc["reserve"].pop(0))
    world["split"].write_text(sp.dump(doc))
    with pytest.raises(GateError, match="split file sha256 differs"):
        _call(world, case)
    # an id in two lists is refused for every case, dev included
    doc = json.loads(COMMITTED.read_text())
    doc["reserve"][0] = doc["eval"][0]
    world["split"].write_text(sp.dump(doc))
    with pytest.raises(GateError, match="appears in both"):
        _call(world, world["doc"]["dev_a"][0])


def test_gate_prereg_outside_repo(world, tmp_path):
    outside = tmp_path / "PREREG.md"
    outside.write_text(world["prereg"].read_text())
    write_receipt(world, prereg_sha256=_sha(outside))
    with pytest.raises(GateError, match="outside the study repo"):
        _call(world, world["doc"]["eval"][0], prereg_path=outside)
