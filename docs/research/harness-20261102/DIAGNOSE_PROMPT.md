Explain the harness smoke run from its saved files and print one block for me to paste back. It makes no model calls,
loads no model and costs nothing. It should take about a minute.

Hard rules:
- Don't push, commit, open a PR or change any git remote.
- Change nothing in antelligence-fairtest, antelligence-dmaddon, antelligence-fresh or antelligence-night.
- Spend nothing. Don't open, print, copy or edit any .env file or key.
- This Mac runs my trading. Never stop a process you didn't start. Leave port 8090 alone. No sudo.

Steps:
1. Go to the antelligence-harness worktree (next to antelligence-fairtest). Find the remote for
   solmonger/antelligence-app (`git remote -v`; call it R), run `git fetch R claude/project-thread-m66msj`, then
   `git checkout --detach R/claude/project-thread-m66msj`. Leave docs/research/harness-20261102/run as it is.
2. Run `uv run --extra test python -m pytest tests/harness -q`. All must pass.
3. Run `uv run python scripts/harness.py diagnose --run docs/research/harness-20261102/run`.
4. Finish with the HARNESS DIAGNOSE block it prints, as one block, unchanged, plus one line on anything that went wrong.
