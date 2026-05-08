from pathlib import Path


QUICKSTART_PATH = Path(__file__).resolve().parents[1] / "docs" / "QUICKSTART_TUMOR_SIM.md"


def test_quickstart_matches_current_cli_and_api_surface():
    quickstart = QUICKSTART_PATH.read_text(encoding="utf-8").lower()

    required_phrases = [
        "uv sync --extra test",
        "uv run antelligence simulate",
        "uv run antelligence-api",
        "/simulate",
        "/runs/{run_id}",
        "/health",
        "8001",
        "proof_staged",
        "base sepolia",
    ]
    for phrase in required_phrases:
        assert phrase in quickstart, f"Quickstart is missing current public-facing phrase: {phrase}"

    stale_phrases = [
        "python3 test_tumor_simulation.py",
        "python3 -m uvicorn main:app --reload --port 8000",
        "/simulation/tumor/test",
        "/simulation/tumor/run",
        "io_secret_key",
    ]
    for phrase in stale_phrases:
        assert phrase not in quickstart, f"Quickstart still contains stale instructions: {phrase}"
