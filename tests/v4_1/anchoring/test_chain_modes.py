"""Mode C never blocks; mode D waits at most the timeout and reports which entries are visible."""

import time

import pytest
from eth_account import Account

from antelligence.anchoring import mocks
from antelligence.anchoring.chain_modes import BoardPostMode, ChainAnchorMode, compact_entry, payload_ints
from chain.v4_1 import ChainClientV41, LocalSigner, TxQueue
from tests.v4_1.chain.conftest import ADDRS, Clock
from tests.v4_1.chain.fakes import FakeChain


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name in ("ANTELLIGENCE_CHAIN", "ANTELLIGENCE_RPC_URL", "CHAIN_RPC", "ZKSYNC_SEPOLIA_RPC_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTELLIGENCE_DEPLOYMENTS_DIR", str(tmp_path / "no-deployments"))


@pytest.fixture
def rig(tmp_path):
    signer = LocalSigner(Account.create().key.hex())          # throw-away key made in the test
    clock, fake = Clock(), FakeChain(300, ADDRS.values())
    fake.contract_deployer = signer.address
    queue = TxQueue(tmp_path / "q.sqlite", clock=clock)
    client = ChainClientV41(fake, signer, queue, ADDRS, expected_deployer=signer.address, clock=clock)
    return client, fake, clock


def test_mode_c_nonblocking(rig):
    client, fake, _ = rig
    mode = ChainAnchorMode(client)
    board, _, _ = mocks.synthetic_run(120)
    t0 = time.perf_counter()
    jobs = [mode.submit(board.run_id, s, board.root(s)) for s in (50, 100, 120)]
    assert time.perf_counter() - t0 < 0.5
    # submitting touched neither the node nor the mempool: it is three INSERTs
    assert fake.sent_log == [] and fake.mempool == {}
    assert client.queue.counts()["pending"] == 3 and len(set(jobs)) == 3
    # prevRoot chaining is tracked: first is zero, then each previous root
    prevs = [bytes.fromhex(j["calldata"][2:])[4 + 96:4 + 128] for j in map(mode.job, jobs)]
    assert prevs == [b"\x00" * 32, board.root(50), board.root(100)]
    # the pump (a different caller) moves them; confirmation needs the effect readable via eth_call
    mode.pump()
    fake.mine()
    mode.pump()
    assert all(mode.confirmed(j) for j in jobs[:1]) and mode.tx_hash(jobs[0]).startswith("0x")
    # idempotent: resubmitting the same anchor returns the same job
    assert mode.submit(board.run_id, 50, board.root(50)) == jobs[0]


def test_mode_d_visibility_timeout(rig):
    client, fake, clock = rig
    board, _, _ = mocks.synthetic_run(12)
    entries = board.entries()
    # never mined: the wait ends at the timeout, everything stays invisible, the timeout is logged
    mode = BoardPostMode(client, timeout_s=60, poll_s=5, clock=clock, sleep=clock.advance)
    t0 = clock()
    res = mode.post_round(board.run_id, entries[:4])
    assert res.timed_out and res.visible == set() and res.invisible == {0, 1, 2, 3}
    assert 60 <= clock() - t0 < 66                    # waited the timeout, not longer
    assert mode.log[-1] == {"event": "timeout", "invisible_seqs": [0, 1, 2, 3], "waited_s": 60}
    # the batch is still queued: once the node mines it the next round sees it, with a latency record
    def mining_sleep(s):
        clock.advance(s)
        fake.mine()

    mode2 = BoardPostMode(client, timeout_s=60, poll_s=5, clock=clock, sleep=mining_sleep)
    mode2._posted_upto = dict(mode._posted_upto)
    mode2._batches = list(mode._batches)
    res2 = mode2.post_round(board.run_id, entries[4:8])
    assert not res2.timed_out and res2.visible == set(range(8)) and res2.invisible == set()
    assert set(res2.latency_ms) == set(range(8)) and all(v >= 0 for v in res2.latency_ms.values())
    assert [e["event"] for e in mode2.log] == ["included", "included"]


def test_mode_d_rejects_non_contiguous_batches(rig):
    client, _, clock = rig
    board, _, _ = mocks.synthetic_run(6)
    mode = BoardPostMode(client, timeout_s=1, poll_s=1, clock=clock, sleep=clock.advance)
    from antelligence.anchoring.chain_modes import ChainModeError

    with pytest.raises(ChainModeError):
        mode.post_round(board.run_id, board.entries()[2:4])


def test_compact_encoding_posts_integers_only():
    board, _, _ = mocks.synthetic_run(8)
    for e in board.entries():
        ints = payload_ints(e)
        assert all(isinstance(i, int) for i in ints)
        compact_entry(e)                                # encodes without strings or id lists
    dec = board.entries()[2]
    assert dec.type == "decision" and payload_ints(dec)[:3] == [dec.payload["squad"], dec.payload["round"], dec.payload["release"]]
