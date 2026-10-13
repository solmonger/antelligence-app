"""The queue is SQLite: it survives restarts, is idempotent, and enqueueing never touches the network."""

import pytest

from chain.v4_1 import TxQueue, encode_entry_compact

from .conftest import ADDRS, b32


class ExplodingW3:
    """Any attribute access fails: proves enqueue never blocks on, or even touches, the chain."""

    def __getattr__(self, name):
        raise AssertionError(f"enqueue touched the network ({name})")


def test_enqueue_never_touches_the_network(signer, tmp_path, clock):
    from chain.v4_1 import ChainClientV41

    c = ChainClientV41(ExplodingW3(), signer, TxQueue(tmp_path / "q.sqlite", clock=clock), ADDRS,
                       expected_deployer=signer.address, clock=clock)
    c.enqueue_anchor(b32(1), 50, b32(2), b32(0))
    c.enqueue_anchor_once(b32(3), b32(4))
    c.enqueue_board_batch(b32(1), 0, b32(0), [encode_entry_compact(0, "task", b32(9), b32(0), [1, 2])])
    c.enqueue_report(b32(1), b32(5), b32(6), True)
    assert c.queue.counts()["pending"] == 4


def test_queue_survives_restart_and_nonces_resume(make_client, fake, clock, tmp_path):
    db = tmp_path / "persist.sqlite"
    c1 = make_client(db=db, replace_after_s=60)
    fake.drop_next = 3
    ids = [c1.enqueue_anchor(b32(i), 50, b32(100 + i), b32(0)) for i in (1, 2, 3)]
    c1.pump()  # nonces 0,1,2 sent, all lost
    c1.queue.close()

    # "Process restart": new queue object on the same file, new client, no in-memory state.
    c2 = make_client(db=db, replace_after_s=60)
    assert [j["id"] for j in c2.queue.by_status("sent")] == ids
    new_id = c2.enqueue_anchor(b32(4), 50, b32(104), b32(0))
    c2.pump()
    assert c2.queue.get(new_id)["nonce"] == 3  # continues after the persisted max, not the chain's 0

    clock.advance(61)
    c2.pump()  # dropped jobs replaced at their ORIGINAL nonces
    assert sorted(j["nonce"] for j in c2.queue.all()) == [0, 1, 2, 3]
    fake.mine()
    c2.pump()
    assert c2.queue.counts() == {"pending": 0, "sent": 0, "confirmed": 4, "failed": 0, "failed_review": 0}


def test_dedupe_makes_enqueue_idempotent(make_client):
    c = make_client()
    a = c.enqueue_anchor(b32(1), 50, b32(2), b32(0))
    b = c.enqueue_anchor(b32(1), 50, b32(2), b32(0))
    assert a == b and len(c.queue.all()) == 1


def test_report_jobs_are_deduped_by_run_author_claim(make_client, fake):
    c = make_client()
    a = c.enqueue_report(b32(1), b32(5), b32(6), True)
    assert c.enqueue_report(b32(1), b32(5), b32(6), False) == a  # same claim: one job
    assert c.enqueue_report(b32(1), b32(5), b32(7), True) != a   # other claim: new job
    assert c.queue.get(a)["dedupe_key"] == f"report:{b32(1).hex()}:{b32(5).hex()}:{b32(6).hex()}"
    with pytest.raises(TypeError):
        c.enqueue_report(b32(1), b32(5), True)  # claim_id is mandatory


def test_report_confirms_only_when_reported_is_true(make_client, fake):
    c = make_client()
    jid = c.enqueue_report(b32(1), b32(5), b32(6), True)
    c.pump()
    fake.mine()
    c.pump()
    assert c.queue.get(jid)["status"] == "confirmed"
    assert (b32(1), b32(5), b32(6)) in fake.claims
    assert fake.rep[b32(5)] == 500_000


def test_queue_file_is_created_under_the_requested_path(tmp_path, clock):
    q = TxQueue(tmp_path / "nested" / "dir" / "q.sqlite", clock=clock)
    assert (tmp_path / "nested" / "dir" / "q.sqlite").exists()
    q.close()


def test_default_queue_path_is_under_gitignored_ledger():
    from chain.v4_1 import default_queue_path

    assert default_queue_path().parent.name == "ledger"
