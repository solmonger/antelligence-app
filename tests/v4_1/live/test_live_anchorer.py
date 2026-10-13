"""The live anchorer on the real contracts (local hardhat node) with mocked OTS and Rekor."""
import json

import pytest

from antelligence.anchoring import mocks
from antelligence.anchoring.live import AnchorArchive, build_live_anchoring
from antelligence.kernel.board import Board, anchor_sizes
from chain.config import ChainConfigError


def drive(live, tmp_path, n, label="a"):
    full, keys, akey = mocks.synthetic_run(n, seed=hash(label) % 1000)
    board = Board(full.header, mode="B", keys=keys.seeds)
    anchorer = live.anchorer_factory(board, akey, tmp_path / f"run-{label}")
    for e in full.entries():
        board.append(e.type, dict(e.author), dict(e.payload), e.tick)
        anchorer.maybe_anchor(board)
    anchorer.finalize(board)
    anchorer.refresh_chain()
    return board, anchorer, akey


def make_live(live_client, lab, tmp_path, **kw):
    return build_live_anchoring(client=live_client, ots=lab.ots_client(), rekor=lab.rekor, start_pumper=False,
                                archive_path=tmp_path / "anchors.json", drain_timeout_s=60, **kw)


def test_board_anchors_land_on_chain_ots_rekor_and_are_archived(live_client, mocks_lab, tmp_path):
    live = make_live(live_client, mocks_lab, tmp_path)
    board, anchorer, akey = drive(live, tmp_path, 120)
    assert anchorer.anchored_sizes == anchor_sizes(120) == [50, 100, 120]
    assert not anchorer.errors
    rid = bytes.fromhex(board.run_id)
    from chain.v4_1 import _calldata

    for w in board.anchors():
        out = bytes(live_client.w3.eth.call({"to": live_client._addr("AnchorRegistry"),
                                             "data": _calldata("AnchorRegistry", "rootAt", [rid, w["size"]])}))
        assert out.hex() == w["root"]                       # chain readback equals the recomputed root
    rows = AnchorArchive(tmp_path / "anchors.json").rows()
    assert [r["size"] for r in rows] == [50, 100, 120]
    for r in rows:
        assert r["tx_hash"].startswith("0x") and len(r["tx_hash"]) == 66
        assert isinstance(r["l2_block"], int) and r["l1_batch"] is None      # hardhat has no L1 batch
        assert r["rekor_uuid"] and r["ots_path"] and r["chain_id"] == 31337
    assert {r.digest for r in mocks_lab.rekor.entries_for_key(akey.pem)} == {w["digest"] for w in board.anchors()}
    assert live_client.queue.counts()["confirmed"] == 3


def test_two_runs_share_one_queue_and_nonce_manager(live_client, mocks_lab, tmp_path):
    live = make_live(live_client, mocks_lab, tmp_path)
    b1, a1, _ = drive(live, tmp_path, 60, "x")
    b2, a2, _ = drive(live, tmp_path, 60, "y")
    assert b1.run_id != b2.run_id
    assert live_client.queue.counts()["confirmed"] == 4
    nonces = [j["nonce"] for j in live_client.queue.all()]
    assert nonces == sorted(set(nonces)) == list(range(nonces[0], nonces[0] + 4))
    assert len(AnchorArchive(tmp_path / "anchors.json").rows()) == 4


def test_mode_d_posts_board_entries_to_boardlog(live_client, mocks_lab, tmp_path):
    from chain.v4_1 import _calldata

    live = make_live(live_client, mocks_lab, tmp_path)
    full, keys, _ = mocks.synthetic_run(6)
    post = live.post_mode_factory()
    result = post.post_round(full.run_id, list(full.entries()))
    assert not result.timed_out and result.visible == {0, 1, 2, 3, 4, 5}
    nxt = bytes(live_client.w3.eth.call({"to": live_client._addr("BoardLog"),
                                         "data": _calldata("BoardLog", "nextSeq", [bytes.fromhex(full.run_id)])}))
    assert int.from_bytes(nxt, "big") == 6


def test_reputation_client_is_the_shared_client(live_client, mocks_lab, tmp_path):
    live = make_live(live_client, mocks_lab, tmp_path)
    assert live.reputation_client is live_client
    jid = live.reputation_client.enqueue_report("11" * 32, "22" * 32, "33" * 32, True)
    live_client.drain(timeout_s=60)
    assert live_client.queue.get(jid)["status"] == "confirmed"


def test_chain_failure_is_recorded_not_raised(live_client, mocks_lab, tmp_path):
    live = make_live(live_client, mocks_lab, tmp_path)
    full, keys, akey = mocks.synthetic_run(50)
    board = Board(full.header, mode="B", keys=keys.seeds)
    a = live.anchorer_factory(board, akey, tmp_path / "r")
    for e in full.entries():
        board.append(e.type, dict(e.author), dict(e.payload), e.tick)
    a.finalize(board)
    live_client.expected_deployer = "0x" + "de" * 20           # now the signer is not the deployer
    assert a.refresh_chain() == 0
    assert any(e["mode"] == "C" and "refusing" in e["error"] for e in a.errors)


def test_preflight_refuses_wrong_code_hash(deployment, tmp_path):
    from chain.v4_1 import ChainClientV41, LocalSigner, TxQueue

    bad = dict(deployment["hashes"], BoardLog="0x" + "00" * 32)
    c = ChainClientV41(deployment["w3"], LocalSigner(deployment["key"]), TxQueue(tmp_path / "q.sqlite"), deployment["addrs"],
                       expected_deployer=deployment["deployer"], expected_chain_id=31337, code_hashes=bad)
    c.enqueue_anchor("aa" * 32, 1, "bb" * 32)
    with pytest.raises(ChainConfigError, match="code hash"):
        c.pump()
    assert c.queue.counts()["pending"] == 1                  # nothing was sent


def test_production_default_refuses_a_local_chain(deployment, tmp_path):
    """Without the test hook the client wants chain 300; a 31337 node is refused before any write."""
    from chain.v4_1 import ChainClientV41, LocalSigner, TxQueue

    c = ChainClientV41(deployment["w3"], LocalSigner(deployment["key"]), TxQueue(tmp_path / "q.sqlite"), deployment["addrs"],
                       expected_deployer=deployment["deployer"])
    with pytest.raises(ChainConfigError, match="refusing"):
        c.preflight()


def test_expected_chain_id_hook_is_local_only(deployment, tmp_path):
    from chain.v4_1 import ChainClientV41, LocalSigner, TxQueue

    with pytest.raises(ChainConfigError):
        ChainClientV41(deployment["w3"], LocalSigner(deployment["key"]), TxQueue(tmp_path / "q.sqlite"), deployment["addrs"],
                       expected_chain_id=324)


def test_open_live_client_loads_signer_from_config_and_pins_from_record(deployment, tmp_path, monkeypatch):
    from chain.v4_1 import open_live_client

    rec = {"chain_id": 300, "contracts": [], "v4_1": {"deployer": deployment["deployer"], "contracts": [
        {"name": n, "address": a, "code_hash": deployment["hashes"][n]} for n, a in deployment["addrs"].items()]}}
    d = tmp_path / "dep"
    d.mkdir()
    (d / "zksync-era-sepolia.json").write_text(json.dumps(rec))
    monkeypatch.setenv("ANTELLIGENCE_DEPLOYMENTS_DIR", str(d))
    monkeypatch.setenv("PRIVATE_KEY", deployment["key"])
    c = open_live_client(tmp_path / "q.sqlite", w3=deployment["w3"], expected_chain_id=31337)
    assert c.signer.address.lower() == deployment["deployer"].lower()
    assert c.code_hashes == deployment["hashes"] and c.contracts == deployment["addrs"]
    # a different signer is refused at open time (before any write)
    from eth_account import Account

    monkeypatch.setenv("PRIVATE_KEY", Account.create().key.hex())
    with pytest.raises(ChainConfigError, match="not the recorded deployer"):
        open_live_client(tmp_path / "q2.sqlite", w3=deployment["w3"], expected_chain_id=31337)
