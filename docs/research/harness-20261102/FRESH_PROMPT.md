Check the harness on brand-new puzzles: the policy frozen from the last rescore (learned before these puzzles
existed) against the reader alone, on 40 new puzzles of every kind plus 20 extra syllogisms (10 from the old maker, 10
written in other words). Print one block for me to paste back. It's free except Jev, which the script caps at 5 cents
(it should use well under 1 cent). It should take about 30 to 50 minutes.

Hard rules:
- Don't push, commit, open a PR or change any git remote.
- Change nothing in antelligence-fairtest, antelligence-dmaddon, antelligence-fresh or antelligence-night. Only read
  from them, and run the mlx_lm.server in antelligence-dmaddon's venv.
- Spend nothing but Jev, which the script caps. Don't open, print, copy or edit any .env file or key. The script
  reads the key itself.
- This Mac runs my trading. Never stop a process you didn't start. Leave port 8090 alone. No sudo.

Steps:
1. Go to the antelligence-harness worktree (next to antelligence-fairtest). Find the remote for
   solmonger/antelligence-app (`git remote -v`; call it R), run `git fetch R claude/project-thread-m66msj`, then
   `git checkout --detach R/claude/project-thread-m66msj`. Leave docs/research/harness-20261102/run and
   docs/research/harness-20261102/replay in place.
2. Run `uv run --extra test python -m pytest tests/harness -q`. All must pass.
3. Checks, and stop and tell me if one fails: `docs/research/harness-20261102/replay/saved/policy-tuned.json` exists
   (it's from the last rescore); nothing listens on ports 11534 or 8097 (`lsof -nP -iTCP:<port> -sTCP:LISTEN`);
   `memory_pressure -Q` shows free 20% or more; `ls ~/.cache/huggingface/hub | head -3` returns within a few seconds.
4. Run:
   `uv run python scripts/harness.py fresh --policy docs/research/harness-20261102/replay/saved/policy-tuned.json --per-kind 10 --jev-cap 0.05 --mlx-server ../antelligence-dmaddon/.venv-optiq/bin/mlx_lm.server --drive ~/.cache/huggingface/hub --out docs/research/harness-20261102/fresh 2>&1 | tee fresh.log`
   If the drive or a model server stalls, the script waits and retries by itself. If it stops with an error, read
   the end of fresh.log, fix only plain setup problems (a path, a busy port), and rerun the same command once; it
   reuses the same puzzles and every finished call.
5. Finish with the FRESH CHECK block (end of fresh.log), unchanged, plus one line on anything that went wrong.
   Leave the worktree in place.
