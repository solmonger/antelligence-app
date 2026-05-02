"""Tests for verifier admin helpers."""

from unittest.mock import patch
import pytest

from backend.chain.verifier_admin import set_verifier_address, submit_proof_verification


class TestVerifierAdmin:
    @patch("backend.chain.verifier_admin.subprocess.run")
    def test_set_verifier_address(self, mock_run):
        mock_run.return_value.stdout = '{"transactionHash":"0xabc"}'
        result = set_verifier_address("0x1111111111111111111111111111111111111111")
        assert result["ok"] is True
        assert result["verifier"] == "0x1111111111111111111111111111111111111111"

    @patch("backend.chain.verifier_admin.subprocess.run")
    def test_submit_proof_verification(self, mock_run):
        mock_run.return_value.stdout = '{"transactionHash":"0xdef"}'
        result = submit_proof_verification("0x1234", "0xabcd")
        assert result["ok"] is True
        assert result["proof_lifecycle"]["stage"] == "verified_onchain"
        assert result["verification_status"]["onchain_ok"] is True

    @patch("backend.chain.config.get_verifier_address")
    @patch("backend.chain.verifier_admin.get_base_sepolia_rpc_url")
    @patch("backend.chain.verifier_admin.get_private_key")
    @patch("backend.chain.verifier_admin.get_tumor_intel_address")
    @patch("backend.chain.verifier_admin.subprocess.run")
    def test_set_verifier_address_invalid_config(self, mock_run, mock_intel, mock_pk, mock_rpc, mock_ver_addr):
        # Simulate missing verifier address in config
        mock_ver_addr.return_value = ""
        mock_intel.return_value = "0x1234567890123456789012345678901234567890"
        mock_pk.return_value = "0xabc"
        mock_rpc.return_value = "http://localhost:8545"
        
        # The implementation should ideally check for this, but for now, we test that it fails or we handle it.
        # If the code doesn't check, this test will fail if we assert an error.
        # Let's see if the current code handles it.
        with pytest.raises(Exception):
            set_verifier_address("0x1111111111111111111111111111111111111111")
