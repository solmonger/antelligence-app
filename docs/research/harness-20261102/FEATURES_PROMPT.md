Score the five borrowed harness parts on this Mac and print one block for me to paste back. It's free: local models
only, nothing paid. It takes several hours, so it's fine to leave it running overnight.

What it does (one command, `scripts/harness_features.py mac`), each stage saved as soon as it's done:
1. free: scores the stop rule, the router and the cache-aware count on the night run's saved answers. No model calls.
2. probe: four quick calls to see whether this mlx_lm.server sends answer odds and reuses a shared prompt start.
3. confsweep: the 8 ants answer all 480 saved questions again, now with answer odds on.
4. gate: the free scoring again with the new answers, now with the confidence gate too.
5. promptopt: a small search for shorter ant instructions, then a check of the pick on set 3.

Plan: docs/plans/2026-11-02-borrowed-parts.md.

Hard rules:
- Don't push, commit, open a PR or change any git remote.
- Change nothing in antelligence-fairtest, antelligence-dmaddon, antelligence-fresh or antelligence-night. Only read
  from them, and run the mlx_lm.server in antelligence-dmaddon's venv.
- Spend nothing. Don't open, print, copy or edit any .env file or key.
- This Mac runs my trading. Never stop a process you didn't start. Leave port 8090 alone. No sudo.

Steps:
1. `F=$(find ~ -maxdepth 4 -type d -name antelligence-fairtest 2>/dev/null)`. There must be exactly one. In `$F`,
   find the remote for solmonger/antelligence-app (`git remote -v`; call it R), run
   `git fetch R claude/project-thread-avftf7`, then
   `git worktree add ../antelligence-features --detach R/claude/project-thread-avftf7` (if that folder already exists,
   go into it and run `git checkout --detach R/claude/project-thread-avftf7` instead). Work in antelligence-features.
   It's its own folder so it never disturbs antelligence-harness.
2. Run `uv run --extra test python -m pytest tests/harness -q`. All must pass.
3. Run `uv run python scripts/harness_features.py mac --dry-run --out /tmp/features-dry`. Its block must say
   "answer odds readable: yes" and have no "failed:" line.
4. Checks, and stop and tell me if one fails:
   - no other harness run is going: `pgrep -fl mlx_standin.py` finds nothing (if it finds one, wait until it's gone,
     checking every 10 minutes, then go on);
   - nothing listens on ports 11544 or 8099 (`lsof -nP -iTCP:<port> -sTCP:LISTEN`);
   - `memory_pressure -Q` says free 20% or more;
   - the model drive answers: `ls ~/.cache/huggingface/hub | head -3` returns within a few seconds;
   - these exist (from antelligence-features): `../antelligence-dmaddon/.venv-optiq/bin/mlx_lm.server`, and
     `../antelligence-night/docs/research/night-20261031/set1/run/rows.jsonl` (also set2 and set3).
5. Start it. If `command -v uvx` finds uvx, add `--odds-server "uvx --from mlx-lm==0.32.2 mlx_lm.server"` to the
   command below. It's used only if the first server sends no answer odds, and then only for this run: uvx keeps
   that copy of Apple's mlx-lm in its own cache and changes no other venv. Without uvx, leave it out.
   `uv run python scripts/harness_features.py mac --night ../antelligence-night/docs/research/night-20261031 --mlx-server ../antelligence-dmaddon/.venv-optiq/bin/mlx_lm.server --drive ~/.cache/huggingface/hub --port 11544 --mlx-port 8099 --out docs/research/harness-20261102/features 2>&1 | tee features.log`
   If the drive or a model server stalls, the script waits and retries by itself. If it stops with an error, read
   the end of features.log, fix only plain setup problems (a path, a busy port), and rerun the same command once; it
   reuses every finished call.
6. Finish with the BORROWED PARTS RESULT block (docs/research/harness-20261102/features/RESULT.txt), as one block,
   unchanged, plus one line on anything that went wrong. Leave the worktree in place.
