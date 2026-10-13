"""Pipelined nonces: contiguous, unique, never more than 8 unconfirmed."""

from .conftest import b32


def test_pipeline_caps_at_eight_in_flight_with_contiguous_nonces(make_client, fake):
    c = make_client()
    for i in range(1, 13):
        c.enqueue_anchor(b32(i), 50, b32(1000 + i), b32(0))
    s = c.pump()
    assert s.sent == 8 and s.deferred == 1
    nonces = sorted(t["nonce"] for t in fake.mempool.values())
    assert nonces == list(range(8))
    assert c.queue.counts()["pending"] == 4

    fake.mine()
    c.pump()  # confirms 8, sends the remaining 4 at nonces 8..11
    assert sorted(t["nonce"] for t in fake.mempool.values()) == [8, 9, 10, 11]
    fake.mine()
    c.pump()
    counts = c.queue.counts()
    assert counts == {"pending": 0, "sent": 0, "confirmed": 12, "failed": 0}
    assert fake.max_mempool <= 8
    used = sorted(j["nonce"] for j in c.queue.all())
    assert used == list(range(12))  # no gaps, no reuse


def test_dependent_anchors_wait_for_their_predecessor(make_client, fake):
    """estimate_gas cannot see the mempool, so size 100 (prevRoot = root50) defers until 50 is mined."""
    c = make_client()
    run = b32(7)
    c.enqueue_anchor(run, 50, b32(50), b32(0))
    c.enqueue_anchor(run, 100, b32(100), b32(50))
    c.enqueue_anchor(run, 150, b32(150), b32(100))
    for _ in range(6):
        c.pump()
        fake.mine()
    c.pump()
    assert c.queue.counts() == {"pending": 0, "sent": 0, "confirmed": 3, "failed": 0}
    assert fake.root_at[(run, 150)] == b32(150)


def test_confirmed_requires_receipt_and_readable_state(make_client, fake):
    c = make_client()
    jid = c.enqueue_anchor_once(b32(1), b32(2))
    c.pump()
    assert c.queue.get(jid)["status"] == "sent"  # broadcast is not confirmation
    fake.mine()
    # Receipt exists, but hide the state: must not count as confirmed.
    saved = dict(fake.once)
    fake.once.clear()
    c.pump()
    assert c.queue.get(jid)["status"] == "sent"
    fake.once.update(saved)
    c.pump()
    assert c.queue.get(jid)["status"] == "confirmed"


def test_job_that_would_revert_on_an_idle_pipeline_fails_without_burning_a_nonce(make_client, fake):
    c = make_client()
    fake.once[b32(1)] = b32(9)  # already taken on chain
    jid = c.enqueue_anchor_once(b32(1), b32(2))
    c.pump()
    assert c.queue.get(jid)["status"] == "failed"  # estimate on an idle pipeline = real revert, nonce not burned
    assert c.queue.max_nonce() is None
