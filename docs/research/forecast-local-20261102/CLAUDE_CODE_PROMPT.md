Run the Antelligence free forecast test on this Mac, then print one block for me to paste back. It costs nothing:
every call goes to a local model on this Mac, and nothing calls Nous or any paid service.

What it is: the fair test's 101 real forecasts (questions that resolved July to October 2026), answered by the same
local reader and four ants as the night run. Two tests, both frozen before any call: do 5 "lenses" beat the reader
asking itself 5 times, and does a mixed crowd (the reader plus the four small models) beat the reader alone? Plan:
docs/plans/2026-11-02-forecast-local.md. Read it first.

The models stay where they are, on the WD_BLACK drive (my choice; don't copy or move them). Because that drive froze
once, the script checks it answers within 10 seconds before it starts and again before each model load, and stops
cleanly if it doesn't. A stopped run resumes from its cache.

Hard rules:
- Don't push, commit, open a PR or change any git remote. Pushing is turned off on purpose.
- Change nothing in antelligence-fairtest, antelligence-srchecks, antelligence-dmaddon, antelligence-fresh or
  antelligence-night. You may only read from them and run the Python and mlx_lm.server in antelligence-dmaddon's
  .venv-optiq.
- Spend nothing. Don't open, print, copy or edit any .env file or key. Don't run `freeze`. If the script says the
  plan changed, stop and tell me.
- This Mac runs my trading. Never stop a process you didn't start yourself. Leave ports 8090, 8091, 8094 and 11434
  alone; this run uses port 8611 only.
- No sudo, no system-service restarts, and don't change HF_HOME or any shell profile. The cache folder is passed to
  the script for this run only. Don't copy, move, download or delete any model, and don't change anything on WD_BLACK.
- One local model at a time; the script does this itself. Every 15 minutes while it runs, check `memory_pressure -Q`
  and `sysctl vm.swapusage`. If pressure is critical or swap grows by more than 2 GB, stop only the run you started
  (its PID), wait for normal, and rerun the same command; it resumes from its cache. If that happens twice, stop and
  tell me the numbers. If the fseventsd watcher grows large, don't touch it; just note its size.

Steps:
1. `F=$(find ~ -maxdepth 4 -type d -name antelligence-fairtest 2>/dev/null)`. There must be exactly one.
   - In `$F`, find the remote for solmonger/antelligence-app (`git remote -v`; call it R).
   - Run `git fetch R claude/project-thread-4gyfqw`.
   - Then `git worktree add ../antelligence-forecast --detach R/claude/project-thread-4gyfqw`.
   - Do everything below in antelligence-forecast.
   - Run `uv sync --extra test` and `uv run python -m pytest tests/smarter_swarm/test_forecast_local.py
     tests/smarter_swarm/test_fair_test.py -q`. All must pass.
2. Check the starting point:
   - Nothing may listen on port 8611 (`lsof -nP -iTCP:8611 -sTCP:LISTEN`). If something does, stop and tell me what.
   - No other local model run may be going: `pgrep -fl "mlx_lm.server|mlx_standin|harness.py|night_run"`. If one is,
     wait for it to finish (check every 10 minutes) rather than load a second model next to it. If it's still going
     after 2 hours, stop and tell me.
   - Note memory pressure, swap, the fseventsd size (`ps -axo rss=,comm= | grep fseventsd`) and free disk
     (`df -h ~`). The run's own files need under 1 GB.
   - Set `P=../antelligence-dmaddon/.venv-optiq/bin` and check that `$P/mlx_lm.server` and `$P/python` exist.
3. Find the model cache, without changing it:
   - `H=$($P/python -c "from huggingface_hub import constants; print(constants.HF_HUB_CACHE)")`. Note `realpath "$H"`
     (it should lead to /Volumes/WD_BLACK).
   - Check it reads: `timeout 60 ls "$H"`. If that hangs or fails, the drive is asleep or stuck: stop and tell me. Don't
     unmount, remount or restart anything.
4. `uv run python scripts/forecast_local.py check --mlx-server $P/mlx_lm.server --hf-cache "$H"`.
   It must print CHECK PASSED. If it says a model is missing, stop and tell me which (don't download it). If memory
   is the problem, don't close my apps; tell me.
5. Smoke test, two forecasts:
   - `uv run python scripts/forecast_local.py run --live --mlx-server $P/mlx_lm.server --hf-cache "$H" --limit 2`.
   - Then `uv run python scripts/forecast_local.py score` and look at the "usable" line. If it stopped, or any model is
     under 80% usable, stop and send me its output and the last 30 lines of the newest log in
     docs/research/forecast-local-20261102/run/logs/.
6. Full run, noting its PID:
   `nohup uv run python scripts/forecast_local.py run --live --mlx-server $P/mlx_lm.server --hf-cache "$H" > forecast.log 2>&1 &`.
   It reuses the smoke test's answers.
   - Watch memory as above and `tail -3 forecast.log`.
   - If it prints STOPPED, read why. If the drive didn't answer, wait until `timeout 60 ls "$H"` works. Rerun the same
     command once (it resumes).
   - If it stops a second time, don't rerun. Run `score` and say why in your report.
   - When it ends, check that port 8611 is free again.
7. `uv run python scripts/forecast_local.py score`. Finish with:
   - its whole output (the FORECAST RESULT block), as one block, unchanged;
   - one line each on:
     - how long the run took (from forecast.log);
     - where the model cache resolved to;
     - peak memory pressure and largest swap;
     - the largest fseventsd size;
     - anything skipped and why.

   Leave the worktree in place. Don't delete anything on WD_BLACK.
