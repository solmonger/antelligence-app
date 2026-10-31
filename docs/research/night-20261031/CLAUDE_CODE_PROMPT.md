Start the Antelligence night run on this Mac, let it run on its own for about 10 hours, then finish its report and
print one block for me to paste back. It's free except Jev, which is capped at 5 cents for the whole night.

What it is: one script (`scripts/night_run.py all`) does everything in order and saves after each stage.

1. A free re-score that sends each kind of question to the setup that did best on it. The routes are picked on
   today's fresh run and frozen, then tested on a brand-new set of questions.
2. "Field notebook" ants: each ant reads what every other model found working alone, once per model, before it
   answers. This tests whether they stop copying the reader's mistakes.
3. One more new set if time is left, to tighten the error ranges.
4. A plain report.

Plan: docs/plans/2026-10-31-night-run.md. Read it first.

Hard rules:
- Don't push, commit, open a PR or change any git remote. Pushing is turned off on purpose.
- Change nothing in antelligence-fairtest, antelligence-srchecks, antelligence-dmaddon or antelligence-fresh. You may
  only read and copy from them, and run the Python and mlx_lm.server in antelligence-dmaddon's venvs.
- Spend nothing but Jev, which the script caps. Don't open, print, copy or edit any .env file or key. The script
  reads the key itself. Don't run `freeze`. If a script says the plan changed, stop and tell me.
- This Mac runs my trading. Never stop a process you didn't start yourself. Leave the chat shim on port 8090 alone:
  the script uses port 8091 for MLX, so the shim doesn't need to stop tonight.
- No sudo, and don't restart system services. I'm away, so nobody can say yes. If the fseventsd file watcher grows
  large again, the script's memory watch pauses and resumes. Just note its size in the report.

Steps:
1. `F=$(find ~ -maxdepth 4 -type d -name antelligence-fairtest 2>/dev/null)`. There must be exactly one. In `$F`, find
   the remote for solmonger/antelligence-app (`git remote -v`; call it R), run `git fetch R claude/project-thread-fly2s6`,
   then `git worktree add ../antelligence-night --detach R/claude/project-thread-fly2s6`. Do everything below in
   antelligence-night (N). Run `uv sync --extra test` and `uv run python -m pytest tests/smarter_swarm/test_night.py
   tests/smarter_swarm/test_fresh.py tests/smarter_swarm/test_shared_read.py tests/smarter_swarm/test_dm_addon.py -q`.
   All must pass.
2. Check that nothing listens on ports 11434, 8091 or 8520 (`lsof -nP -iTCP:<port> -sTCP:LISTEN`). If something does,
   stop and tell me what. Note memory pressure (`memory_pressure -Q`), swap (`sysctl vm.swapusage`), the fseventsd
   size (`ps -axo rss=,comm= | grep fseventsd`) and free disk (at least 20 GB).
3. Copy today's fresh run in as set 1 (copy, never move):
   `mkdir -p docs/research/night-20261031/set1 && cp -R ../antelligence-fresh/docs/research/fresh-20261031/run docs/research/night-20261031/set1/run`.
   It must hold questions.jsonl, answers.jsonl, rows.jsonl, cache.jsonl, dm_jev.jsonl and dm_d1-3b.jsonl. If any is
   missing, stop and tell me.
4. Paths:
   - The mlx_lm.server today's run used: `../antelligence-dmaddon/.venv-optiq/bin/mlx_lm.server`.
   - The d1-3B Python: `../antelligence-dmaddon/.venv-d1/bin/python`.
   - The AGIEval run folder: `../antelligence-dmaddon/docs/research/dm-addon-20261030/run`. Use it only if it holds
     swarm_shared-read.jsonl; otherwise leave `--agieval` out.

   Check each path exists.
5. Start it, and note its PID:
   `nohup uv run python scripts/night_run.py all --hours 10 --mlx-server <mlx path> --d1-python <d1 path> --agieval <dir> > night.log 2>&1 &`
6. While it runs, every 15 minutes:
   - Read `tail -5 night.log` and the last line of `docs/research/night-20261031/logs/memory.jsonl`.
   - The script guards memory itself. Don't stop it for memory.
   - If its PID is gone and night.log doesn't end with the "NIGHT RUN RESULT" block, read why at the end of night.log
     and in docs/research/night-20261031/logs/, then rerun the same command once. It skips finished stages and resumes
     from its caches.
   - If it dies a second time, don't rerun it. Run `uv run python scripts/night_run.py pool` and then `report`, and
     say why in the report.
   - If this session can't keep watching for the whole night, that's fine: the script carries on by itself.
7. When it's finished, open docs/research/night-20261031/REPORT.md. Fill in only its last section ("What helped, what
   didn't, what next") from the numbers above it, in plain words, with no jargon, in 10 lines at most. Change nothing
   else in it.
8. Finish with:
   - the NIGHT RUN RESULT block from the end of night.log, as one block, unchanged;
   - your "What helped, what didn't, what next" section;
   - one line each on the total time, memory stops and the largest fseventsd size, and anything skipped and why.

   Leave the worktree in place.
