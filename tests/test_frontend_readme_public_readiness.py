from pathlib import Path


FRONTEND_README_PATH = Path(__file__).resolve().parents[1] / "frontend" / "README.md"


def test_frontend_readme_matches_current_public_surface():
    readme = FRONTEND_README_PATH.read_text(encoding="utf-8").lower()

    required_phrases = [
        "8081",
        "vite_api_base_url",
        "vite_frontend_mode",
        "preview mode",
        "http://127.0.0.1:8001",
        "/comparison",
        "/tumor",
        "/tumor-hunt",
    ]
    for phrase in required_phrases:
        assert phrase in readme, f"Frontend README is missing current phrase: {phrase}"

    stale_phrases = [
        "lovable project",
        "lovable.dev/projects",
        "share -> publish",
        "connect a custom domain",
        "http://localhost:5173",
    ]
    for phrase in stale_phrases:
        assert phrase not in readme, f"Frontend README still contains stale phrase: {phrase}"
