from pathlib import Path


GITIGNORE_PATH = Path(__file__).resolve().parents[1] / ".gitignore"


def test_gitignore_covers_local_agent_and_generated_release_noise():
    gitignore = GITIGNORE_PATH.read_text(encoding="utf-8")

    required_entries = [
        ".env",
        ".hermes/",
        ".serena/",
        "cache/",
        "out/",
        "data/*.sqlite3",
        "data/brats/*.zip",
    ]

    for entry in required_entries:
        assert entry in gitignore, f".gitignore is missing public-readiness guard: {entry}"
