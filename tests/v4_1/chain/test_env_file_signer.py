"""The live signer and RPC can come from the W1 env file in-process, never via os.environ or output."""

import os

from chain import v4_1

FAKE_KEY = "0x" + "ab" * 32


def test_env_file_value_reads_in_process_without_touching_environ(tmp_path, monkeypatch, capsys):
    env = tmp_path / ".env"
    env.write_text(f"PRIVATE_KEY={FAKE_KEY}\nZKSYNC_SEPOLIA_RPC_URL=https://rpc.example/v2/secret\n")
    monkeypatch.setenv("ANTELLIGENCE_ENV_FILE", str(env))
    monkeypatch.delenv("PRIVATE_KEY", raising=False)
    assert v4_1.env_file_value("PRIVATE_KEY") == FAKE_KEY
    assert "PRIVATE_KEY" not in os.environ
    signer = v4_1.signer_from_config()
    assert signer.address  # derived; the key itself stays inside the signer
    out = capsys.readouterr()
    assert FAKE_KEY[2:] not in out.out + out.err


def test_missing_env_file_gives_no_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTELLIGENCE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("PRIVATE_KEY", raising=False)
    monkeypatch.delenv("ANTELLIGENCE_DEPLOYER_PRIVATE_KEY", raising=False)
    assert v4_1.env_file_value("PRIVATE_KEY") == ""
