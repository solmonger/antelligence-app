Check the harness fixes on this Mac: rerun the same 12 smoke puzzles with the fixes, then explain the run. Print two
blocks for me to paste back. It's free: local models only, nothing paid. It should take about 20 to 40 minutes.

Hard rules:
- Don't push, commit, open a PR or change any git remote.
- Change nothing in antelligence-fairtest, antelligence-dmaddon, antelligence-fresh or antelligence-night. Only read
  from them, and run the mlx_lm.server in antelligence-dmaddon's venv.
- Spend nothing. Don't open, print, copy or edit any .env file or key.
- This Mac runs my trading. Never stop a process you didn't start. Leave port 8090 alone. No sudo.

Steps:
1. Go to the antelligence-harness worktree (next to antelligence-fairtest). Find the remote for
   solmonger/antelligence-app (`git remote -v`; call it R), run `git fetch R claude/project-thread-m66msj`, then
   `git checkout --detach R/claude/project-thread-m66msj`. Leave docs/research/harness-20261102/run in place.
2. Run `uv run --extra test python -m pytest tests/harness -q`. All must pass.
3. Checks, and stop and tell me if one fails: nothing listens on ports 11534 or 8097
   (`lsof -nP -iTCP:<port> -sTCP:LISTEN`); `memory_pressure -Q` shows free 20% or more;
   `ls ~/.cache/huggingface/hub | head -3` returns within a few seconds.
4. Keep the first result: `cp docs/research/harness-20261102/run/RESULT.txt docs/research/harness-20261102/run/RESULT-1.txt`
5. Run (same puzzles as last time, so the numbers compare):
   `uv run python scripts/harness.py mac --seed 258388501 --night ../antelligence-night/docs/research/night-20261031 --mlx-server ../antelligence-dmaddon/.venv-optiq/bin/mlx_lm.server --drive ~/.cache/huggingface/hub --out docs/research/harness-20261102/run 2>&1 | tee harness2.log`
   If the drive or a model server stalls, the script waits and retries by itself. If it stops with an error, read
   the end of harness2.log, fix only plain setup problems (a path, a busy port), and rerun the same command once; it
   reuses every finished call.
6. Run `uv run python scripts/harness.py diagnose --run docs/research/harness-20261102/run`.
7. Finish with the HARNESS RESULT block (end of harness2.log) and the HARNESS DIAGNOSE block, each unchanged, plus one
   line on anything that went wrong. Leave the worktree in place.
