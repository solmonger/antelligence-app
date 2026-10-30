Run two free checks on the finished shared-read run on this Mac. They only re-read saved answers: no model is
loaded or asked, and nothing costs money.

Rules: don't change, stash, check out or delete anything inside antelligence-fairtest, because its run folder holds
the answers. Work in a new git worktree next to it. Stop no process, push nothing, commit nothing.

1. Find the folder: `F=$(find ~ -maxdepth 4 -type d -name antelligence-fairtest 2>/dev/null)`. If there isn't
   exactly one, stop and ask me. Set `SR=$F/docs/research/shared-read-20261030`. `wc -l $SR/run/rows.jsonl` must
   say 240, and `$SR/run/cache.jsonl` must exist. If not, stop and tell me.
2. In `$F`, find the remote for solmonger/antelligence-app (`git remote -v`), run
   `git fetch <remote> claude/project-thread-zif8rm`, then
   `git worktree add ../antelligence-srchecks --detach <remote>/claude/project-thread-zif8rm`. Work in
   antelligence-srchecks from here on. Run `uv sync --extra test` and
   `uv run python -m pytest tests/smarter_swarm/test_shared_read.py -q`. All must pass.
3. Copy the answers in: `mkdir -p docs/research/shared-read-20261030/run && cp $SR/run/rows.jsonl $SR/run/meta.json
   docs/research/shared-read-20261030/run/`. Use the Mac's own model file for every command below,
   `M=$SR/models.json`, because the saved replies are found by the exact request that was sent.
4. Same score as before: `uv run python scripts/shared_read.py score --models $M`. Its numbers must match the
   earlier score (reader 63.3%, alone 47.5%, notes 57.5%, both 55.4%). If they don't, stop and show me.
5. Check 1, other swarm shapes: `uv run python scripts/shared_read.py explore --models $M`.
6. Check 2, re-read the answers with parser 2, which also reads Gemma's LaTeX answers:
   `uv run python scripts/shared_read.py run --models $M --parser 2 --offline --seed-cache $SR/run/cache.jsonl`,
   then `uv run python scripts/shared_read.py score --models $M --parser 2`. `--offline` means no model is asked.
   If the run reports more than 100 new calls, the saved replies didn't match. Stop and show me its output.
7. Print one block for me to paste back to Claude in Duet, holding the full output of steps 4, 5 and 6. Add any error
   you hit. Leave the antelligence-srchecks folder in place.
