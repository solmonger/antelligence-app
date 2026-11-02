Run the Antelligence "decision models on top of the ants" test on this Mac, after the shared-read run has finished.

What it is: decision models (models that only pick from a list and give a probability for each option) are asked
about questions the swarm has already answered. It makes no new LLM call. Jev is hosted and costs about 1 cent in
total, capped at 5 cents. Liquid d1-3B and d1-2B run locally and are free. The plan is in
docs/plans/2026-10-30-dm-addon.md and the script is scripts/dm_addon.py, both on branch
claude/project-thread-fly2s6. Read both before you start.

Hard rules:
- Work in two folders only: the existing antelligence-fairtest folder, which you only read, and a new git worktree
  next to it called antelligence-dmaddon, which you create in step 2. Find antelligence-fairtest with
  `find ~ -maxdepth 4 -type d -name antelligence-fairtest 2>/dev/null`. If there isn't exactly one, stop and ask me.
- Change nothing inside antelligence-fairtest: no checkout, no stash, no edits, no deleted files. Its run folders
  hold the saved answers this test reads.
- Spend nothing beyond Jev's 5-cent cap, which the script enforces. Don't open, print, copy or edit any .env file or
  API key. The script reads TYPESAFE_API_KEY from ~/.hermes/.env by itself. If it says the key is missing, stop and
  tell me. Push nothing, open no PR, and make no commit unless I ask.
- This Mac also runs my trading and other work. Never stop, kill or restart any process you didn't start yourself,
  Ollama included, and don't change Ollama's global settings. If something else is using the memory you need, stop
  and tell me what it is.
- Never delete a run folder: not docs/research/dm-addon-20261030/run/, nor any run folder in antelligence-fairtest.
  The caches are what let a stopped run resume without paying twice.
- Run one local model at a time, and stop each model server you started before starting the next.
- Don't change the code, the questions, the prompts, the arms or the pass lines, and don't run `freeze`. The plan is
  frozen. If `plan` or `run` says the plan changed, stop and tell me.

Steps:
1. Wait for the shared-read run. Run `pgrep -fl shared_read.py`. If it is still running, check every 10 minutes
   and don't start anything until it has finished and its `score` has been shown to me. Then, in
   antelligence-fairtest, find the full run's rows: `find docs/research/shared-read-20261030 -name rows.jsonl`.
   Use the folder that holds the 240-question run (its meta.json and `wc -l rows.jsonl` should say 240). Call it SR.
   Also check whether the fair test's live run exists: docs/research/fair-test-20261029/rows_live.jsonl and
   cache_live.jsonl. Note what you found, with each file's line count.
2. Get the code. In antelligence-fairtest, find the remote that points at solmonger/antelligence-app
   (`git remote -v`) and run `git fetch <remote> claude/project-thread-fly2s6`. Then
   `git worktree add ../antelligence-dmaddon --detach <remote>/claude/project-thread-fly2s6`. This adds a new folder
   and changes nothing in antelligence-fairtest. In antelligence-dmaddon, run `uv sync --extra test` and
   `uv run python -m pytest tests/smarter_swarm/test_dm_addon.py -q`. All tests must pass before going on. Everything
   from here runs in antelligence-dmaddon.
3. Check room. Show total memory (`sysctl hw.memsize`), memory pressure (`memory_pressure`), swap
   (`sysctl vm.swapusage`), what Ollama has loaded (`ollama ps`) and free disk (`df -h ~`). You need about 12 GB of
   free disk for the two local models plus 10 GB spare. If disk is short, stop and tell me. Don't delete anything.
4. Turn the saved runs into the test's input:
   `uv run python scripts/dm_addon.py export --source shared-read --sr-run <full path to SR>`.
   If the fair test's live run exists, also run
   `uv run python scripts/dm_addon.py export --source fair-test --fair-rows <full path>/rows_live.jsonl --fair-cache <full path>/cache_live.jsonl`.
   If that export stops because the cache and rows don't match, or reasons are missing, stop that source and tell me
   the message. Don't work around it. If there is no live fair-test run yet, say so and do everything below with
   the shared-read source only.
5. Jev (hosted, about 1 cent):
   - `uv run python scripts/dm_addon.py plan --dm jev --source shared-read` (and `--source fair-test` if exported).
     The two together must come to under $0.05. If not, stop and tell me.
   - `uv run python scripts/dm_addon.py smoke --dm jev --live`. It makes 4 calls, well under a cent. It must end with
     "OK". If not, show me what it printed and stop.
   - `uv run python scripts/dm_addon.py run --dm jev --source shared-read --live`, then the same with
     `--source fair-test` if exported. The cap is $0.05 across both, and the script stops there by itself. Never
     raise `--max-usd`.
6. Liquid d1-3B (local, free, about 7 GB in memory):
   - Make it its own environment in antelligence-dmaddon: `uv venv .venv-d1 --python 3.12`, then
     `uv pip install --python .venv-d1 "transformers>=5.14" torch pillow`.
   - Check memory pressure is normal and swap is steady. Then start the server in the background, logging to
     d1-3b.log: `nohup .venv-d1/bin/python scripts/v41_d1_server.py --host 127.0.0.1 --port 8520 > d1-3b.log 2>&1 &`.
     Note its PID. The first start downloads the model from Hugging Face at a pinned revision. Wait until the log
     says "d1 shim on 127.0.0.1:8520". If the download asks for a Hugging Face sign-in, a token or for accepting
     terms, stop that model and tell me. Don't sign in for me.
   - `uv run python scripts/dm_addon.py smoke --dm d1-3b --live`. It must end with "OK".
   - `uv run python scripts/dm_addon.py run --dm d1-3b --source shared-read --live`, then fair-test if exported.
   - Stop the server with `kill <its PID>`, the one you started, and check that the memory came back.
7. Liquid d1-2B at 4 bits (local, free, about 2 GB):
   - `uv venv .venv-optiq --python 3.12`, then `uv pip install --python .venv-optiq "mlx-optiq>=0.7.9"`.
   - Check port 8080 is free (`lsof -nP -iTCP:8080 -sTCP:LISTEN`). If something else uses it, look in
     `.venv-optiq/bin/optiq serve --help` for a port option, use a free port, and pass `--url http://127.0.0.1:<port>`
     to every command below.
   - Start it in the background, logging to d1-2b.log:
     `nohup .venv-optiq/bin/optiq serve --model mlx-community/d1-2B-OptiQ-4bit > d1-2b.log 2>&1 &`. Note its PID.
   - Smoke, then run both sources, as in step 6 with `--dm d1-2b`. If the smoke fails because the server's reply
     looks different from what the script expects, show me the first reply and the error, skip d1-2B, and go on.
   - Stop the server you started.
8. Perplexity Decider Flash (optional). Read its model card on Hugging Face (perplexity-ai/pplx-decider-flash-v1.2).
   Only if it names a runtime that runs on Apple Silicon and serves `POST /v1/systemone` in the model's own protocol,
   start it on port 8081 and do as in step 6 with `--dm pplx-flash`. Stock llama.cpp doesn't speak this model's
   protocol. Don't build or patch a runtime. If there is no such runtime, skip it and say so.
9. Memory watch, for every local model: while it runs, check every 2 to 3 minutes the run's output,
   `memory_pressure` and `sysctl vm.swapusage`. If pressure reaches "critical", or swap grows by more than 2 GB, stop
   the run and the model server you started (the run resumes from its cache), wait for pressure to go back to
   normal, and resume. If it gets tight twice, stop and tell me the numbers.
10. If a run stops for any other reason, read what it prints. Rerun the same command once; it resumes from its
    cache. If it stops again, tell me rather than changing anything.
11. Score: `uv run python scripts/dm_addon.py score --source shared-read`, and `--source fair-test` if exported.
    Then give me, in plain words:
    - the score output as is;
    - how long each model took;
    - what Jev cost in total;
    - peak memory pressure;
    - any model you skipped, and why;
    - anything that went wrong.

    End with a short block I can paste back to Claude in Duet. It should hold:
    - each score output in full;
    - the smoke lines of each model;
    - the line counts from step 1.
