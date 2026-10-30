Run the Antelligence "shared-read" test on this Mac, end to end, without running out of memory.

What it is: one bigger local model (the reader) writes notes on 240 exam questions, and 8 small local models (ants)
answer alone, from the notes, or both. It is free and fully local through Ollama. The plan is in
docs/plans/2026-10-30-shared-read.md and the script is scripts/shared_read.py. Read both before you start.

Hard rules:
- Work only inside the antelligence-fairtest folder and with Ollama. Find the folder with
  `find ~ -maxdepth 4 -type d -name antelligence-fairtest 2>/dev/null`. If there isn't exactly one, stop and ask me.
- Spend nothing. Make no paid API calls and don't touch any API key or .env file. Push nothing, open no PR, and make
  no commit unless I ask.
- This Mac also runs my trading and other work. Never stop, kill or restart any process you didn't start yourself,
  Ollama included, and don't change Ollama's global settings. If something else is using the memory you need, stop
  and tell me what it is.
- Never delete the run folder docs/research/shared-read-20261030/run/. Its cache is what lets a stopped run resume.

Steps:
1. Get the code. In antelligence-fairtest, run `git status`. If there are uncommitted changes, or a fair test is
   running (`pgrep -fl fair_test.py`), stop and tell me. Otherwise find the remote that points at
   solmonger/antelligence-app (`git remote -v`), fetch branch claude/project-thread-zif8rm from it, and check it out.
   It holds everything from the fair-test branch plus this test. Then run `uv sync --extra test` and
   `uv run python -m pytest tests/smarter_swarm/test_shared_read.py -q`. All tests must pass before going on.
2. Check room. Show total memory (`sysctl hw.memsize`), memory pressure (`memory_pressure`), what Ollama has loaded
   (`ollama ps`), and free disk (`df -h ~`). You need about 30 GB of free disk for the models plus 10 GB spare. If
   disk is short, stop and tell me. Don't delete anything to make room.
3. Pull the models listed in docs/research/shared-read-20261030/models.json (`ollama pull <tag>` for the reader and
   each ant), one at a time. If a tag doesn't exist, look on ollama.com for the newest model of the same family at a
   similar size, put it in models.json, and note why.
4. Run `uv run python scripts/shared_read.py check`. Every model must show "ok". For any model it marks as thinking,
   check that models.json tells it to stop thinking.
5. Run `uv run python scripts/shared_read.py bench`. It answers 2 questions with every model and prints the speed, the
   tokens per call and a time estimate. If it warns that a model is still thinking (LONG), fix how models.json turns
   thinking off for that model. Run `curl -s localhost:11434/api/show -d '{"model":"<tag>"}'` to see what the model
   supports, then try in order: `"extra_body": {"reasoning_effort": "none"}`, then `"extra_body": {"think": false}`,
   then `"suffix": "/no_think"`, then the lowest effort it supports. Rerun bench each time until no LONG warning is
   left. Also read the reader's notes that bench prints: they should explain the question without picking an option.
6. If the reader (qwen3.8:27b) can't load with room to spare, or bench puts the full run over 12 hours, switch the
   reader in models.json to gemma4:12b and redo steps 3 to 5. Note why.
7. Only if you changed models.json, run `uv run python scripts/shared_read.py freeze`. No real run has been scored
   yet, so this is still the preregistered test. Show me the diff of models.json.
8. Quick pass: start `uv run python scripts/shared_read.py run --limit 60` in the background with nohup, logging to
   docs/research/shared-read-20261030/run.log. While it runs, check every 2 to 3 minutes: the log tail,
   `memory_pressure`, swap use (`sysctl vm.swapusage`) and `ollama ps`. If pressure reaches "critical", or swap grows
   by more than 2 GB during the run, stop the run process you started (it resumes from its cache), run
   `ollama stop <model>` for the test's own models only, wait for pressure to go back to normal, and resume. If it
   gets tight twice, stop and tell me the numbers. The script also refuses to load a model without room for it plus
   6 GB. If it stops with "MemoryTight", treat that the same way.
9. When the quick pass finishes, run `uv run python scripts/shared_read.py score` and show me the output. Don't
   treat it as the verdict yet.
10. Full run: start `uv run python scripts/shared_read.py run` the same way, with the same memory watch. It reuses the
    60 questions already done. If it stops for any reason other than memory, read the reason it prints, fix only
    what's needed (no change to questions, prompts, arms or pass lines), and resume.
11. Run `uv run python scripts/shared_read.py score` once more. Then give me, in plain words: the score output as is,
    how long the run took, peak memory pressure, any model you swapped and why, and anything that went wrong. End
    with a short block I can paste back to Claude in Duet, holding the full score output and
    docs/research/shared-read-20261030/run/meta.json.
