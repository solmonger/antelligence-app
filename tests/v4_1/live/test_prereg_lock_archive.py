"""scripts/v41_prereg_lock.py on the local node with mock OTS/Rekor, and scripts/v41_archive_l1.py with a fake RPC."""
import hashlib
import json
import subprocess

import pytest

from antelligence.anchoring import keys
from antelligence.anchoring.anchorer import single_shot_id
from antelligence.anchoring.live import build_live_anchoring
from antelligence.experiments import v41_gate
from antelligence.kernel.canonical import jcs_bytes
from scripts import v41_archive_l1, v41_prereg_lock as lock


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def study(tmp_path, live_client, mocks_lab):
    """A tiny git repo holding a committed prereg that states the study key, plus services against the local node."""
    ledger = tmp_path / "ledger"
    fp = keys.create_study_key(ledger)
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    (repo / "PREREGISTRATION.md").write_text(f"# prereg\n\nStudy key fingerprint (sha256 of SPKI DER): `{fp}`\n")
    (repo / "runplan.json").write_text(json.dumps({"runs": [{"b": 1, "a": 2}]}))
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "prereg")
    live = build_live_anchoring(client=live_client, ots=mocks_lab.ots_client(), rekor=mocks_lab.rekor, start_pumper=False,
                                archive_path=tmp_path / "unused.json")
    return {"repo": repo, "ledger": ledger, "live": live, "client": live_client, "out": tmp_path / "chain", "lab": mocks_lab,
            "svc": lambda: {"client": live_client, "ots": live.ots, "rekor": live.rekor}}


def run_lock(study, which, extra=()):
    return lock.main([which, "--prereg", str(study["repo"] / "PREREGISTRATION.md"), "--runplan", str(study["repo"] / "runplan.json"),
                      "--out-dir", str(study["out"]), "--repo-root", str(study["repo"]), "--ledger-dir", str(study["ledger"]), *extra],
                     make_services=study["svc"])


def test_prereg_lock_anchors_everywhere_and_writes_the_gate_receipt(study):
    assert run_lock(study, "prereg") == 0
    receipt = json.loads((study["out"] / "prereg-receipt.json").read_text())
    prereg = study["repo"] / "PREREGISTRATION.md"
    assert receipt["status"] == 1 and receipt["readback_ok"] is True and receipt["chain_id"] == 31337   # 300 on the real chain
    assert receipt["prereg_sha256"] == hashlib.sha256(prereg.read_bytes()).hexdigest()
    assert receipt["tx_hash"].startswith("0x") and len(receipt["tx_hash"]) == 66
    # the gate accepts this receipt shape (chain id swapped to the real one)
    (study["out"] / "gate.json").write_text(json.dumps({**receipt, "chain_id": 300}))
    v41_gate._check_receipt(study["out"] / "gate.json", prereg)
    # chain holds the digest, OTS and Rekor hold it too
    from chain.v4_1 import _calldata

    sid = single_shot_id("prereg")
    out = bytes(study["client"].w3.eth.call({"to": study["client"]._addr("AnchorRegistry"),
                                             "data": _calldata("AnchorRegistry", "onceDigestOf", [sid])}))
    assert out == hashlib.sha256(prereg.read_bytes()).digest()
    assert (study["out"] / "prereg.ots").exists()
    fpr = keys.load_study_key(study["ledger"]).pem
    assert {r.digest for r in study["lab"].rekor.entries_for_key(fpr)} == {out.hex()}
    rows = json.loads((study["out"] / "anchors.json").read_text())
    assert rows[0]["name"] == "prereg" and rows[0]["tx_hash"] == receipt["tx_hash"]


def test_lock_is_resumable_and_idempotent(study):
    assert run_lock(study, "prereg") == 0
    first = json.loads((study["out"] / "prereg-receipt.json").read_text())
    n_ots = (study["out"] / "prereg.ots").read_bytes()
    jobs = len(study["client"].queue.all())
    assert run_lock(study, "prereg") == 0                         # second run: no new chain tx, same OTS file
    assert len(study["client"].queue.all()) == jobs
    assert (study["out"] / "prereg.ots").read_bytes() == n_ots
    assert json.loads((study["out"] / "prereg-receipt.json").read_text())["tx_hash"] == first["tx_hash"]


def test_runplan_needs_prereg_first_then_uses_the_canonical_digest(study):
    assert run_lock(study, "runplan") == 2
    assert not (study["out"] / "runplan-receipt.json").exists()
    assert run_lock(study, "prereg") == 0 and run_lock(study, "runplan") == 0
    receipt = json.loads((study["out"] / "runplan-receipt.json").read_text())
    assert receipt["digest"] == hashlib.sha256(jcs_bytes({"runs": [{"a": 2, "b": 1}]})).hexdigest()
    assert receipt["single_shot_id"] == "0x" + single_shot_id("runplan").hex()


def test_refuses_uncommitted_or_modified_document(study):
    (study["repo"] / "PREREGISTRATION.md").write_text("changed\n")
    assert run_lock(study, "prereg") == 2
    assert study["client"].queue.all() == []                       # nothing was queued, nothing written
    assert not (study["out"] / "prereg-receipt.json").exists()


def test_refuses_when_prereg_names_a_different_study_key(study):
    other = study["ledger"].parent / "alt" / "ledger"
    keys.create_study_key(other)
    study["ledger"] = other
    assert run_lock(study, "prereg") == 2
    assert study["client"].queue.all() == []


def test_refuses_to_overwrite_a_different_digest_already_on_chain(study):
    study["client"].enqueue_anchor_once(single_shot_id("prereg"), b"\x11" * 32)
    study["client"].drain(timeout_s=60)
    assert run_lock(study, "prereg") == 2
    assert not (study["out"] / "prereg-receipt.json").exists()


# --- archive_l1 ----------------------------------------------------------------------------------------------------


def fake_rpc(batches):
    calls = []

    def rpc(method, params):
        calls.append(method)
        if method == "eth_chainId":
            return "0x12c"
        if method == "eth_getTransactionReceipt":
            return {"blockNumber": "0x10", "l1BatchNumber": "0x7"}
        if method == "zks_getL1BatchDetails":
            return batches.get(params[0], {})
        raise AssertionError(method)

    rpc.calls = calls
    return rpc


def test_archive_l1_fills_commit_prove_execute_and_is_rerunnable(tmp_path):
    path = tmp_path / "anchors.json"
    path.write_text(json.dumps([{"run_id": "r", "size": 50, "tx_hash": "0x" + "ab" * 32, "l1_batch": None},
                                {"kind": "once", "name": "prereg", "tx_hash": None}]))
    rpc = fake_rpc({7: {"commitTxHash": "0xc", "proveTxHash": None, "executeTxHash": None, "status": "sealed"}})
    assert v41_archive_l1.main(["--anchors", str(path)], rpc=rpc) == 0
    row = json.loads(path.read_text())[0]
    assert row["l1_batch"] == 7 and row["l2_block"] == 16 and row["l1_commit_tx"] == "0xc" and row["l1_prove_tx"] is None
    rpc2 = fake_rpc({7: {"commitTxHash": "0xc", "proveTxHash": "0xp", "executeTxHash": "0xe", "status": "verified"}})
    assert v41_archive_l1.main(["--anchors", str(path)], rpc=rpc2) == 0
    row = json.loads(path.read_text())[0]
    assert (row["l1_prove_tx"], row["l1_execute_tx"]) == ("0xp", "0xe")
    assert "eth_getTransactionReceipt" not in rpc2.calls            # batch already known: only the batch details are re-read


def test_archive_l1_is_read_only_and_refuses_the_wrong_chain(tmp_path):
    path = tmp_path / "anchors.json"
    path.write_text("[]")
    assert v41_archive_l1.main(["--anchors", str(path)], rpc=lambda m, p: "0x144") == 2
    call = v41_archive_l1.http_rpc("http://127.0.0.1:1")
    for method in ("eth_sendRawTransaction", "eth_sendTransaction", "zks_sendRawTransactionWithDetailedOutput"):
        with pytest.raises(ValueError, match="read-only"):
            call(method, [])
