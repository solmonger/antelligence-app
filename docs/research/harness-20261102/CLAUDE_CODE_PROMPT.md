Run the Antelligence harness smoke test on this Mac and print one result block for me to paste back. It's free:
local models only, nothing paid. It should take about 20 to 40 minutes.

What it does (one command, `scripts/harness.py mac`):
1. Reads the night run's saved answers (480 questions) and learns, for free, which recipe of models and tools wins
   for which kind of question; learns on sets 1 and 2, scores on set 3.
2. Makes 12 brand-new code-made puzzles and runs every tool on them with the local models, including the new code
   checker.
3. Runs the harness live on 4 more new puzzles.

Plan: docs/plans/2026-11-02-harness-v0.md.

Hard rules:
- Don't push, commit, open a PR or change any git remote.
- Change nothing in antelligence-fairtest, antelligence-dmaddon, antelligence-fresh or antelligence-night. Only read
  from them, and run the mlx_lm.server in antelligence-dmaddon's venv.
- Spend nothing. Don't open, print, copy or edit any .env file or key.
- This Mac runs my trading. Never stop a process you didn't start. Leave port 8090 alone. No sudo.

Steps:
1. `F=$(find ~ -maxdepth 4 -type d -name antelligence-fairtest 2>/dev/null)`. There must be exactly one. In `$F`,
   find the remote for solmonger/antelligence-app (`git remote -v`; call it R), run
   `git fetch R claude/project-thread-m66msj`, then
   `git worktree add ../antelligence-harness --detach R/claude/project-thread-m66msj`. Work in antelligence-harness.
2. Run `uv run --extra test python -m pytest tests/harness -q`. All must pass.
3. Run `uv run python scripts/harness.py selftest --out /tmp/harness-selftest`. It must end with `SELFTEST OK`.
4. Checks, and stop and tell me if one fails:
   - nothing listens on ports 11534 or 8097 (`lsof -nP -iTCP:<port> -sTCP:LISTEN`);
   - `memory_pressure -Q` says normal-ish (free 20% or more);
   - the model drive answers: `ls ~/.cache/huggingface/hub | head -3` returns within a few seconds;
   - these exist: `../antelligence-dmaddon/.venv-optiq/bin/mlx_lm.server`, and
     `../antelligence-night/docs/research/night-20261031/set1/run/rows.jsonl` (also set2 and set3).
5. Start it:
   `uv run python scripts/harness.py mac --night ../antelligence-night/docs/research/night-20261031 --mlx-server ../antelligence-dmaddon/.venv-optiq/bin/mlx_lm.server --drive ~/.cache/huggingface/hub --out docs/research/harness-20261102/run 2>&1 | tee harness.log`
   If the drive or a model server stalls, the script waits and retries by itself. If it stops with an error, read
   the end of harness.log, fix only plain setup problems (a path, a busy port), and rerun the same command once; it
   reuses every finished call.
6. Finish with the HARNESS RESULT block from the end of harness.log, as one block, unchanged, plus one line on
   anything that went wrong. Leave the worktree in place.
