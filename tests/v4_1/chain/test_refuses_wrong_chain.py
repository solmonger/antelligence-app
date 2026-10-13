"""No signing unless: right chain (300), signer == recorded deployer, contracts have code."""

import pytest

from chain.config import ChainConfigError

from .conftest import ADDRS, b32
from .fakes import FakeChain


@pytest.mark.parametrize("wrong", [1, 324, 84532, 260, 31337])
def test_refuses_wrong_chain_id(make_client, fake, wrong):
    fake.chain_id = wrong
    c = make_client()
    c.enqueue_anchor(b32(1), 50, b32(2), b32(0))
    with pytest.raises(ChainConfigError, match="refusing"):
        c.pump()
    assert fake.sent_log == []
    assert c.queue.counts()["pending"] == 1


def test_correct_chain_id_300_is_accepted(make_client, fake):
    c = make_client()
    assert c.preflight() == 300


def test_selected_network_must_match_rpc(make_client, fake, monkeypatch):
    monkeypatch.setenv("ANTELLIGENCE_CHAIN", "zksync-local-inmemory")  # selects 260
    c = make_client()
    with pytest.raises(ChainConfigError, match="260"):
        c.preflight()
    fake.chain_id = 260
    assert c.preflight() == 260


def test_refuses_signer_that_is_not_the_deployer(make_client, fake):
    c = make_client(expected_deployer="0x" + "de" * 20)
    c.enqueue_anchor(b32(1), 50, b32(2), b32(0))
    with pytest.raises(ChainConfigError, match="not the recorded deployer"):
        c.pump()
    assert fake.sent_log == []


def test_refuses_when_no_deployer_is_recorded(make_client):
    c = make_client(expected_deployer="")
    with pytest.raises(ChainConfigError, match="no deployer"):
        c.preflight()


def test_refuses_address_without_code(make_client, signer, clock, tmp_path):
    from chain.v4_1 import ChainClientV41, TxQueue

    bare = FakeChain(300, [ADDRS["AnchorRegistry"], ADDRS["BoardLog"]])  # ReputationRegistry has no code
    c = ChainClientV41(bare, signer, TxQueue(tmp_path / "q.sqlite", clock=clock), ADDRS,
                       expected_deployer=signer.address, clock=clock)
    with pytest.raises(ChainConfigError, match="no contract code"):
        c.preflight()


def test_refuses_when_v4_1_is_not_deployed(fake, signer, clock, tmp_path):
    from chain.v4_1 import ChainClientV41, TxQueue

    c = ChainClientV41(fake, signer, TxQueue(tmp_path / "q.sqlite", clock=clock), {},
                       expected_deployer=signer.address, clock=clock)
    with pytest.raises(ChainConfigError):
        c.preflight()


def test_deployer_and_addresses_come_from_the_v4_1_deployments_key(fake, signer, clock, tmp_path, monkeypatch):
    import json

    from chain.v4_1 import ChainClientV41, TxQueue

    d = tmp_path / "dep"
    d.mkdir()
    (d / "zksync-era-sepolia.json").write_text(json.dumps({
        "chain_id": 300, "deployer": "0x" + "00" * 20, "contracts": [],
        "v4_1": {"deployer": signer.address, "contracts": [{"name": n, "address": a} for n, a in ADDRS.items()]},
    }))
    monkeypatch.setenv("ANTELLIGENCE_DEPLOYMENTS_DIR", str(d))
    c = ChainClientV41(fake, signer, TxQueue(tmp_path / "q.sqlite", clock=clock), clock=clock)
    assert c.expected_deployer == signer.address and c.contracts == ADDRS
    assert c.preflight() == 300


def test_refuses_board_log_deployed_by_someone_else(make_client, fake):
    """BoardLog.postBatch is deployer-only; a BoardLog from another deployer would only revert, so never sign."""
    fake.contract_deployer = "0x" + "de" * 20
    c = make_client()
    c.enqueue_board_batch(b32(1), 0, b32(0), [b"\x01"])
    with pytest.raises(ChainConfigError, match="deployed by"):
        c.pump()
    assert fake.sent_log == []


def test_refuses_when_only_board_log_has_a_different_deployer(make_client, fake):
    """AnchorRegistry matches the recorded deployer; ONLY BoardLog differs. The BoardLog check must stand alone."""
    fake.deployer_by_contract[ADDRS["BoardLog"].lower()] = "0x" + "de" * 20
    c = make_client()
    c.enqueue_board_batch(b32(1), 0, b32(0), [b"\x01"])
    with pytest.raises(ChainConfigError, match="BoardLog .*deployed by"):
        c.pump()
    assert fake.sent_log == []
    fake.deployer_by_contract.clear()
    assert c.preflight() == 300
