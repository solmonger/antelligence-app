Run the Antelligence fresh-question test on this Mac, then print one block for me to paste back. Free except Jev,
which the script caps at 2.5 cents.

What it is: code makes 160 new exam questions on this Mac, so no model can have seen them. The same local reader and
8 ants as last night answer them, then Jev and d1-3B. A second check moves the answer letters on last night's 240
questions. Plan: docs/plans/2026-10-31-fresh-questions.md. Read it first.

Hard rules:
- Don't push, commit, open a PR or change any git remote. Pushing is turned off on purpose.
- Change nothing in antelligence-fairtest, antelligence-srchecks or antelligence-dmaddon. You may only read from them
  and run the Python in antelligence-dmaddon/.venv-d1.
- Spend nothing but Jev, which the script caps. Never pass `--max-usd`. Don't open, print, copy or edit any .env file
  or key. The script reads the key itself. Don't run `freeze`. If the script says the plan changed, stop and tell me.
- This Mac runs my trading. Never stop a process you didn't start yourself.
- One local model at a time. Every 3 minutes while a local model runs, check `memory_pressure` and
  `sysctl vm.swapusage`. If pressure is critical or swap grows by more than 2 GB, stop the run and the servers you
  started, wait for normal, and rerun the same command (everything resumes from its cache). If it happens twice,
  stop and tell me the numbers.

Steps:
1. `F=$(find ~ -maxdepth 4 -type d -name antelligence-fairtest 2>/dev/null)`. There must be exactly one. In `$F`, find
   the remote for solmonger/antelligence-app (`git remote -v`; call it R), run `git fetch R claude/project-thread-fly2s6`,
   then `git worktree add ../antelligence-fresh --detach R/claude/project-thread-fly2s6`. Do everything below in
   antelligence-fresh. Run `uv sync --extra test` and `uv run python -m pytest tests/smarter_swarm/test_fresh.py
   tests/smarter_swarm/test_shared_read.py tests/smarter_swarm/test_dm_addon.py -q`. All must pass.
2. Check that nothing listens on ports 11434, 8090 or 8520 (`lsof -nP -iTCP:<port> -sTCP:LISTEN`). If something
   does, stop and tell me what. Note memory pressure, swap and free disk.
3. `uv run python scripts/fresh_test.py make`
4. Find the `mlx_lm.server` that last night's shared-read run used: `command -v mlx_lm.server`, or
   `find ~ -maxdepth 7 -path '*bin/mlx_lm.server' 2>/dev/null`. If there is none, run `uv tool install mlx-lm==0.32.2`.
   Start the stand-in, noting its PID:
   `nohup uv run python scripts/mlx_standin.py --models docs/research/fresh-20261031/models.json --mlx-server <that path> --log mlx.log > standin.log 2>&1 &`
5. `nohup uv run python scripts/fresh_test.py swarm > swarm.log 2>&1 &`. Watch memory as above and the log tail. If
   it stops, read why and rerun it once. When it finishes, stop the stand-in by its PID. That stops its mlx server
   too. Check port 8090 is free.
6. Jev:
   - `uv run python scripts/fresh_test.py dm --dm jev --live`
   - `uv run python scripts/fresh_test.py leak --dm jev --live --dm1 ../antelligence-dmaddon/docs/research/dm-addon-20261030/run/dm_jev_shared-read.jsonl`

   If Jev's key is missing or a call fails, note it and go on.
7. d1-3B: start `nohup ../antelligence-dmaddon/.venv-d1/bin/python scripts/v41_d1_server.py --host 127.0.0.1 --port 8520 > d1.log 2>&1 &`,
   note its PID, and wait for "d1 shim on". Then run `uv run python scripts/fresh_test.py dm --dm d1-3b --live`, and
   stop the shim by its PID.
8. `uv run python scripts/fresh_test.py score`. Finish with:
   - its whole output, as one block, unchanged;
   - one line each on how long the swarm took, peak memory pressure, anything skipped and why.

   Leave the worktree in place.
