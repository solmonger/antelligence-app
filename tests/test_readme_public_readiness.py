from pathlib import Path


README_PATH = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_matches_current_public_product_surface():
    readme = README_PATH.read_text(encoding="utf-8").lower()

    required_phrases = [
        "tumor simulation",
        "base sepolia",
        "proof",
        "antelligence simulate",
        "antelligence-api",
        "/simulate",
        "/runs/{run_id}",
        "/health",
        "api_runs.sqlite3",
        "antelligence_run_db",
    ]
    for phrase in required_phrases:
        assert phrase in readme, f"README is missing public-facing phrase: {phrase}"

    stale_phrases = [
        "ant foraging",
        "food collection",
        "launch io hackathon",
    ]
    for phrase in stale_phrases:
        assert phrase not in readme, f"README still contains stale positioning: {phrase}"


def test_readme_explains_artifact_boundary_between_run_json_and_proof_bundle():
    readme = README_PATH.read_text(encoding="utf-8").lower()

    required_phrases = [
        "artifact_hash",
        "config_hash",
        "proof_bundle",
        "verification_status",
    ]
    for phrase in required_phrases:
        assert phrase in readme, f"README is missing artifact-boundary phrase: {phrase}"
