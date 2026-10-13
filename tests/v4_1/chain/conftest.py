import pytest
from eth_account import Account

from chain.v4_1 import ChainClientV41, LocalSigner, TxQueue

from .fakes import FakeChain

ADDRS = {
    "AnchorRegistry": "0x" + "a1" * 20,
    "BoardLog": "0x" + "b2" * 20,
    "ReputationRegistry": "0x" + "c3" * 20,
}


class Clock:
    def __init__(self, t=1_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


@pytest.fixture(autouse=True)
def clean_chain_env(monkeypatch, tmp_path):
    for name in ("ANTELLIGENCE_CHAIN", "ANTELLIGENCE_RPC_URL", "CHAIN_RPC", "ZKSYNC_SEPOLIA_RPC_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTELLIGENCE_DEPLOYMENTS_DIR", str(tmp_path / "no-deployments"))


@pytest.fixture
def test_key():
    return Account.create().key.hex()  # throw-away key generated in the test; never read from disk/env


@pytest.fixture
def signer(test_key):
    return LocalSigner(test_key)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def fake():
    return FakeChain(300, ADDRS.values())


@pytest.fixture
def make_client(fake, signer, clock, tmp_path):
    def _make(db=None, **kw):
        q = TxQueue(db or tmp_path / "q.sqlite", clock=clock)
        kw.setdefault("expected_deployer", signer.address)
        return ChainClientV41(fake, signer, q, ADDRS, clock=clock, **kw)
    return _make


def b32(n: int) -> bytes:
    return n.to_bytes(32, "big")
