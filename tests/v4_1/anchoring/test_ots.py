"""OpenTimestamps client against an in-process calendar. No network."""

import pytest

from antelligence.anchoring.mocks import MockOtsTransport
from antelligence.anchoring.ots import OtsClient, OtsError, load_ots, verify_ots

DIGEST = bytes(range(32))
CALS = ("https://cal-a.mock", "https://cal-b.mock")


def client(transport, **kw):
    return OtsClient(transport, calendars=CALS, nonce_fn=lambda: b"n" * 16, **kw)


def test_ots_stamp_offline_mock(tmp_path):
    t = MockOtsTransport()
    res = client(t).stamp(DIGEST, tmp_path / "a.ots")
    assert res.calendars_ok == list(CALS) and not res.calendars_failed
    assert [u for u, _ in t.submitted] == list(CALS)
    # the calendars see the salted digest, never the anchor digest itself
    assert all(d != DIGEST and len(d) == 32 for _, d in t.submitted)
    ots = load_ots(tmp_path / "a.ots")            # a standard detached .ots file
    assert ots.file_digest == DIGEST
    v = verify_ots(tmp_path / "a.ots", DIGEST)
    # a calendar promise is NOT evidence: fail closed
    assert not v.ok and v.status == "unverified" and v.digest_matches
    assert not verify_ots(tmp_path / "a.ots", bytes(32)).ok


def test_ots_stamp_tolerates_one_dead_calendar_and_enforces_minimum(tmp_path):
    t = MockOtsTransport(fail=(CALS[0],))
    res = client(t).stamp(DIGEST, tmp_path / "a.ots")
    assert res.calendars_ok == [CALS[1]] and CALS[0] in res.calendars_failed
    with pytest.raises(OtsError):
        client(MockOtsTransport(fail=CALS)).stamp(DIGEST, tmp_path / "b.ots")
    with pytest.raises(OtsError):
        client(t, min_ok=2).stamp(DIGEST, tmp_path / "c.ots")


def test_ots_upgrade_and_verify_against_block_header(tmp_path):
    t = MockOtsTransport(clock=lambda: 1_700_000_123)
    c = client(t)
    c.stamp(DIGEST, tmp_path / "a.ots")
    first = c.upgrade(tmp_path / "a.ots")
    assert first.upgraded == 0 and first.pending == 2          # nothing mined yet
    t.mine_block()
    second = c.upgrade(tmp_path / "a.ots")
    assert second.upgraded == 2 and second.complete
    v = verify_ots(tmp_path / "a.ots", DIGEST, t.header)
    assert v.ok and v.status == "bitcoin" and v.time == 1_700_000_123
    no_headers = verify_ots(tmp_path / "a.ots", DIGEST)
    assert not no_headers.ok and no_headers.status == "unverified"       # attestation present, never checked
    # a header whose merkle root is not the proof's commitment must fail
    bad = lambda h: type(t.header(h))(hashMerkleRoot=b"\x01" * 32, nTime=1)   # noqa: E731
    assert not verify_ots(tmp_path / "a.ots", DIGEST, bad).ok


def test_ots_rejects_garbage_file(tmp_path):
    (tmp_path / "x.ots").write_bytes(b"not an ots file")
    assert not verify_ots(tmp_path / "x.ots", DIGEST).ok


def test_forged_pending_ots_fails_closed(tmp_path):
    """A forged proof with only a pending attestation (or an unknown one) must never verify."""
    t = MockOtsTransport()
    client(t).stamp(DIGEST, tmp_path / "a.ots")
    for provider in (None, t.header):
        v = verify_ots(tmp_path / "a.ots", DIGEST, provider)
        assert not v.ok and v.status == "unverified"
    # a Bitcoin attestation for a height the header source cannot serve is not a pass either
    t.mine_block()
    client(t).upgrade(tmp_path / "a.ots")
    missing = lambda h: (_ for _ in ()).throw(KeyError(h))   # noqa: E731
    v = verify_ots(tmp_path / "a.ots", DIGEST, missing)
    assert not v.ok and v.status == "unverified"
