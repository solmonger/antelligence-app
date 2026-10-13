"""A dropped transaction is replaced at the SAME nonce with a higher gas price."""

from .conftest import b32


def test_dropped_tx_is_replaced_at_same_nonce_with_higher_gas(make_client, fake, clock):
    c = make_client(replace_after_s=60)
    fake.drop_next = 1
    jid = c.enqueue_anchor(b32(1), 50, b32(50), b32(0))
    c.pump()
    job = c.queue.get(jid)
    assert job["status"] == "sent" and job["nonce"] == 0 and not fake.mempool  # lost

    c.pump()  # too early: nothing happens
    assert c.queue.get(jid)["attempts"] == 1

    clock.advance(61)
    s = c.pump()
    assert s.replaced == 1
    job = c.queue.get(jid)
    assert job["nonce"] == 0 and job["attempts"] == 2 and len(job["tx_hashes"]) == 2
    (tx,) = fake.mempool.values()
    assert tx["nonce"] == 0 and tx["gas_price"] > fake.sent_log[0]["gas_price"]
    assert [t["nonce"] for t in fake.sent_log] == [0, 0]  # never a fresh nonce

    fake.mine()
    c.pump()
    assert c.queue.get(jid)["status"] == "confirmed"


def test_replacement_gas_bump_clears_the_node_underpriced_rule(make_client, fake, clock):
    c = make_client(replace_after_s=1, gas_bump_pct=15)
    fake.drop_next = 0
    jid = c.enqueue_anchor(b32(2), 50, b32(50), b32(0))
    c.pump()
    # Keep the original in the pool but pretend it is stuck: advance and replace.
    clock.advance(5)
    c.pump()
    assert len(fake.sent_log) == 2
    assert fake.sent_log[1]["gas_price"] * 100 >= fake.sent_log[0]["gas_price"] * 110
    assert len(fake.mempool) == 1  # replaced, not duplicated
    fake.mine()
    c.pump()
    assert c.queue.get(jid)["status"] == "confirmed"


def test_original_mined_before_replacement_is_recognised(make_client, fake, clock):
    c = make_client(replace_after_s=1)
    jid = c.enqueue_anchor(b32(3), 50, b32(50), b32(0))
    c.pump()
    fake.mine()  # the first tx lands; the client has not polled yet
    clock.advance(5)
    s = c.pump()
    assert s.replaced == 0 and c.queue.get(jid)["status"] == "confirmed"


def test_nonce_consumed_by_foreign_tx_requeues_under_fresh_nonce(make_client, fake, clock, signer):
    c = make_client(replace_after_s=1)
    fake.drop_next = 1
    jid = c.enqueue_anchor(b32(4), 50, b32(50), b32(0))
    c.pump()
    fake.mined_nonce[signer.address.lower()] = 1  # someone else used nonce 0
    clock.advance(5)
    c.pump()
    job = c.queue.get(jid)
    assert job["status"] == "sent" and job["nonce"] == 1
    fake.mine()
    c.pump()
    assert c.queue.get(jid)["status"] == "confirmed"


def test_gives_up_after_max_attempts(make_client, fake, clock):
    c = make_client(replace_after_s=1)
    fake.drop_next = 100
    jid = c.enqueue_anchor(b32(5), 50, b32(50), b32(0))
    for _ in range(c.MAX_ATTEMPTS + 2):
        c.pump()
        clock.advance(2)
    assert c.queue.get(jid)["status"] == "failed"
    assert {t["nonce"] for t in fake.sent_log} == {0}
