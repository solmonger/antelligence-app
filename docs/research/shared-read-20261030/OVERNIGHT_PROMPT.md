I'm away for a few hours. Run three things on this Mac in order, then push the results to a new branch on my fork.
Free except Jev, which is capped at 5 cents in total. Keep memory safe throughout. Where a step says "stop and tell
me", I'm not here, so skip that part, write down why, and carry on with the rest.

Hard rules:
- Never touch upstream Antelligence, never merge, never open a PR. Push only the one new branch in part 3, to the
  remote that points at solmonger/antelligence-app.
- Don't change, stash, check out or delete anything inside antelligence-fairtest. It is only read from.
- Spend nothing but Jev's 5 cents, which the dm-addon script enforces. Never raise `--max-usd`. Don't open, print,
  copy or edit any .env file or API key.
- This Mac runs my trading. Never stop, kill or restart any process you didn't start, with one exception: the
  `mlx_lm.server` and the Ollama-shaped stand-in on port 11434 left over from today's shared-read run. Before
  stopping either, check its command line (`ps -o pid,command -p <pid>`) and that it is that server or that
  stand-in. Stop only those two, and only if they are running. Nothing else.
- Memory: one local model at a time. Every 2 to 3 minutes while a model runs, check `memory_pressure` and
  `sysctl vm.swapusage`. If pressure reaches "critical" or swap grows by more than 2 GB, stop the run and the
  model server you started, wait for normal, and resume (runs resume from their caches). If it gets tight twice,
  skip that model and note the numbers.

Part 1: two free checks on the shared-read answers (seconds, no model)
1. `F=$(find ~ -maxdepth 4 -type d -name antelligence-fairtest 2>/dev/null)`. If there isn't exactly one, stop
   everything and leave me a note in ~/overnight-20261030.md. Set `SR=$F/docs/research/shared-read-20261030`.
   `wc -l $SR/run/rows.jsonl` must say 240, and `$SR/run/cache.jsonl` must exist.
2. In `$F`, find the fork's remote (`git remote -v`; call it R), `git fetch R claude/project-thread-zif8rm
   claude/project-thread-fly2s6`, then `git worktree add ../antelligence-srchecks --detach
   R/claude/project-thread-zif8rm`. In antelligence-srchecks: `uv sync --extra test` and
   `uv run python -m pytest tests/smarter_swarm/test_shared_read.py -q`. All must pass.
3. `mkdir -p docs/research/shared-read-20261030/run && cp $SR/run/rows.jsonl $SR/run/meta.json
   docs/research/shared-read-20261030/run/` and set `M=$SR/models.json`.
4. `uv run python scripts/shared_read.py score --models $M`. Its numbers must match the earlier score: reader
   63.3%, alone 47.5%, notes 57.5%, both 55.4%. If they don't, note it and skip steps 5 and 6.
5. `uv run python scripts/shared_read.py explore --models $M`
6. `uv run python scripts/shared_read.py run --models $M --parser 2 --offline --seed-cache $SR/run/cache.jsonl`,
   then `uv run python scripts/shared_read.py score --models $M --parser 2`. If the run reports more than 100 new
   calls, the saved replies didn't match: note it and skip the score.
   Save the full output of steps 4 to 6 in `~/overnight-20261030-part1.txt`.

Part 2: the decision-model add-on
7. Read docs/research/dm-addon-20261030/CLAUDE_CODE_PROMPT.md from R/claude/project-thread-fly2s6
   (`git show R/claude/project-thread-fly2s6:docs/research/dm-addon-20261030/CLAUDE_CODE_PROMPT.md`) and follow its
   steps 2 to 11, with these changes:
   - Its step 1 is done: the shared-read run finished, and SR above is the folder to use (`$SR/run`).
   - The hard rules above win where they differ, and "stop and tell me" means skip and note.
   - Ollama isn't installed: skip `ollama ps`. Instead list what listens on ports 11434 and 8080
     (`lsof -nP -iTCP:11434 -sTCP:LISTEN`, the same for 8080). If today's `mlx_lm.server` or the 11434 stand-in
     is there, stop them as the rules above allow, and check the memory comes back.
   - If Jev's key is missing or its smoke test fails, skip Jev and carry on with the local models.
   Save every score output, smoke line and line count it asks for in `~/overnight-20261030-part2.txt`.

Part 3: push the results
8. In antelligence-srchecks: `git switch -c claude/overnight-20261030`. Make the folder
   docs/research/overnight-20261030/ and put in it:
   - part1.txt and part2.txt (the two files above);
   - from the srchecks run folders: run/score.json, run/explore.json, run-p2/score.json and run-p2/meta.json;
   - from antelligence-dmaddon's run folder: every score or summary file the dm-addon script wrote (json, md or
     txt), but no cache, no model file and no log;
   - a short SUMMARY.md in plain words: what ran, the main numbers, how long each part took, peak memory pressure,
     what Jev cost, what was skipped and why.
   Add no file over 5 MB, no *cache*.jsonl, no model weights, no .env, no virtual environment.
9. Stage only that folder: `git add docs/research/overnight-20261030`. Before committing, run `git diff --cached --stat` and `git grep --cached -nIi -E "api[_-]?key|sk-[a-z0-9]{8}|TYPESAFE_API_KEY="`.
   If anything looks like a key, remove that file and note it. Then commit with the message
   "Overnight results: shared-read checks and decision-model add-on" and run
   `git push -u R claude/overnight-20261030`. If the push fails, retry twice. If it still fails, leave the commit
   in place and say so.
10. Leave both worktrees in place. Finish by printing the branch name, the commit hash, and the SUMMARY.md text.
