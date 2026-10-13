"""Rekor's public instance is SHARDED: the entry's top-level ``logIndex`` is global across shards, while checkpoints
and inclusion proofs are per shard. Regression for the first LIVE run, whose honest anchors all failed with
"the archived checkpoint (size 3123301898) predates its own entry (index 3245206158)".

* ``rekor_sharded_live_20261013.json`` is a sanitized copy of real, public rekor.sigstore.dev data (read-only GETs).
* The synthetic tests run the whole verifier against a sharded mock log.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from antelligence.anchoring import mocks
from antelligence.anchoring.rekor import (
    RekorClient,
    RekorError,
    check_record,
    entry_tree_id,
    parse_checkpoint,
    parse_entry,
)
from antelligence.anchoring.tamper import N_ENTRIES, Lab, _anchor_all, _export, _plan
from antelligence.anchoring.verify import OutsideVerifier
from antelligence.kernel.board import anchor_sizes

FIXTURE = Path(__file__).parent / "fixtures" / "rekor_sharded_live_20261013.json"
RUN_ID = "06b78f539544af309ed7e398286d8bcec6faf3a341fe315901f0d1334f54311f"


class Replay:
    """Serves the recorded public answers; any request the fixture lacks is an error, never a network call."""

    def __init__(self, data):
        self.d = data

    def log_public_key(self):
        return self.d["log_public_key_pem"].encode()

    def get_entry(self, uuid):
        return {uuid: self.d["entries"][uuid]}

    def search_by_pubkey(self, pem):
        return list(self.d["entries"])

    def log_info(self):
        return self.d["log_info"]

    def consistency_proof(self, first, last, tree_id):
        return self.d["consistency_proofs"][f"{first}-{last}"]


@pytest.fixture
def live():
    d = json.loads(FIXTURE.read_text())
    from antelligence.anchoring.rekor import REKOR_LOG_KEY_SHA256
    client = RekorClient(Replay(d), pinned_log_key_sha256=REKOR_LOG_KEY_SHA256)
    records = [parse_entry(u, e) for u, e in d["entries"].items()]
    return d, client, records


def _verifier(tmp_path, d, client):
    (tmp_path / f"{RUN_ID}.anchors.json").write_text(json.dumps(d["anchors"]))
    by_size = {r["size"]: r for r in d["anchors"]}
    run = SimpleNamespace(run_id=RUN_ID, entries=[None] * 439,
                          root=lambda s: bytes.fromhex(by_size[s]["root"]),
                          digest=lambda s: bytes.fromhex(by_size[s]["digest"]))
    v = OutsideVerifier(tmp_path, {}, rekor=client, conditions={"R"})
    return v, run


def test_live_fixture_really_is_sharded(live):
    d, _, records = live
    for r in records:
        assert r.log_index > r.tree_size                      # the global index dwarfs the shard
        assert r.shard_index < r.tree_size
        assert r.tree_id == parse_checkpoint(r.inclusion_proof["checkpoint"]).tree_id == "1193050959916656506"


def test_live_entries_pass_record_checks(live):
    d, client, records = live
    for r in records:
        assert check_record(r, d["anchor_public_key_pem"].encode(), client.log_key()) == []


def test_honest_live_run_split_view_verifies(live, tmp_path):
    d, client, records = live
    v, run = _verifier(tmp_path, d, client)
    problems = []
    unverified = v._split_view(run, records, client.log_key(), problems)
    assert problems == [] and unverified == []


def test_live_forged_checkpoint_fails(live, tmp_path):
    d, client, records = live
    d = copy.deepcopy(d)
    cp = d["anchors"][0]["rekor_checkpoint"]
    d["anchors"][0]["rekor_checkpoint"] = cp.replace("UGAAo9X3", "UGAAo9X4")
    v, run = _verifier(tmp_path, d, client)
    problems = []
    v._split_view(run, records, client.log_key(), problems)
    assert any("archived checkpoint" in p for p in problems)


def test_live_checkpoint_from_before_the_entry_in_its_shard_fails(live, tmp_path):
    d, client, records = live
    d = copy.deepcopy(d)
    d["anchors"][-1]["rekor_checkpoint"] = d["anchors"][0]["rekor_checkpoint"]     # covers only entry #1
    v, run = _verifier(tmp_path, d, client)
    problems = []
    v._split_view(run, records, client.log_key(), problems)
    assert any("predates its own entry" in p for p in problems)


def test_live_checkpoint_of_another_shard_fails(live, tmp_path):
    d, client, records = live
    d = copy.deepcopy(d)
    cp = d["anchors"][0]["rekor_checkpoint"]
    d["anchors"][0]["rekor_checkpoint"] = cp.replace("1193050959916656506", "1")
    v, run = _verifier(tmp_path, d, client)
    problems = []
    v._split_view(run, records, client.log_key(), problems)
    assert problems                                          # bad signature or wrong tree: never a pass


def test_live_split_view_fails(live, tmp_path):
    d, client, records = live
    d = copy.deepcopy(d)
    # the log serves proofs that do not link what it signed at anchor time to what it shows now
    d["consistency_proofs"] = {k: ["00" * 32] for k in d["consistency_proofs"]}
    client = RekorClient(Replay(d), pinned_log_key_sha256=client.pinned_log_key_sha256)
    v, run = _verifier(tmp_path, d, client)
    problems = []
    v._split_view(run, records, client.log_key(), problems)
    assert any("split view" in p for p in problems)


def test_live_missing_proof_fails(live):
    d, client, records = live
    r = copy.copy(records[0])
    r.inclusion_proof = None
    assert any("no inclusion proof" in p for p in check_record(r, d["anchor_public_key_pem"].encode(), client.log_key()))


def test_uuid_tree_id():
    assert entry_tree_id("108e9186e8c5677a" + "0" * 64) == "1193050959916656506"
    assert entry_tree_id("ab" * 32) == ""                    # unsharded UUID


# -- end to end against a sharded mock log ------------------------------------------------------------------------

@pytest.fixture
def sharded(tmp_path):
    lab = Lab()
    lab.mock_rekor = mocks.MockRekor(clock=lab.mock_rekor.clock, tree_id=0x108E9186E8C5677A, global_offset=3_245_000_000)
    lab.rekor = RekorClient(lab.mock_rekor, pinned_log_key_sha256=lab.mock_rekor.log_key_sha256)
    board, keys, akey = mocks.synthetic_run(N_ENTRIES, seed=3)
    out = tmp_path / "export"
    _export(board, akey, out)
    _anchor_all(lab, board, akey, out, anchor_sizes(N_ENTRIES))
    return lab, board, out, _plan(board, akey)


def test_sharded_honest_run_verifies_for_condition_r(sharded):
    lab, board, out, plan = sharded
    report = OutsideVerifier(out, plan, rekor=lab.rekor, conditions={"R"}).verify()
    chk = report["runs"][board.run_id]["checks"]["rekor"]
    assert chk["ok"], chk
    assert not any("predates" in p for p in chk["problems"])


def test_anchorer_archives_shard_fields(sharded):
    lab, board, out, plan = sharded
    recs = json.loads((out / f"{board.run_id}.anchors.json").read_text())
    for r in recs:
        assert r["rekor_tree_id"] == "1193050959916656506"
        assert r["rekor_log_index"] >= 3_245_000_000 and r["rekor_shard_index"] < r["rekor_tree_size"]
        assert parse_checkpoint(r["rekor_checkpoint"]).tree_id == r["rekor_tree_id"]


def test_sharded_wrong_archived_shard_index_fails(sharded):
    lab, board, out, plan = sharded
    path = out / f"{board.run_id}.anchors.json"
    recs = json.loads(path.read_text())
    recs[0]["rekor_shard_index"] += 1
    path.write_text(json.dumps(recs))
    report = OutsideVerifier(out, plan, rekor=lab.rekor, conditions={"R"}).verify()
    assert report["verdict"] == "failed"
