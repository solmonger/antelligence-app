Recheck the borrowed-parts run on this Mac and print one block for me to paste back. It's free and calls no model:
it only rereads the answers last night's run saved, so no model server is started.

What it does: the scoring again, now with a column saying on how many test questions each switch (stop, gate, cache)
was actually used, plus a new readout of which ant model, if any, is as right as the reader when it is very sure.

Hard rules:
- Don't push, commit, open a PR or change any git remote.
- Change nothing in antelligence-fairtest, antelligence-dmaddon, antelligence-fresh or antelligence-night. Only read
  from them.
- Don't delete or change anything under docs/research/harness-20261102/features (last night's results).
- Spend nothing. Don't open, print, copy or edit any .env file or key.
- This Mac runs my trading. Never stop a process you didn't start. Leave port 8090 alone. No sudo.

Steps:
1. Go into the antelligence-features worktree (next to antelligence-fairtest). Find the remote for
   solmonger/antelligence-app (`git remote -v`; call it R), run `git fetch R claude/project-thread-avftf7`, then
   `git checkout --detach R/claude/project-thread-avftf7`.
2. Run `uv run --extra test python -m pytest tests/harness -q`. All must pass.
3. Print which recipes last night's run picked on the test questions (instant):
   `python3 -c "import json;[print(r['config'],'|',r['used']) for s in ('free','gate') for r in json.load(open(f'docs/research/harness-20261102/features/{s}/features.json'))['rows']]"`
4. Run:
   `uv run python scripts/harness_features.py mac --night ../antelligence-night/docs/research/night-20261031 --stages free,gate,calib --confsweep docs/research/harness-20261102/features/confsweep --out docs/research/harness-20261102/features-recheck 2>&1 | tee recheck.log`
5. Finish with the output of step 3 and the BORROWED PARTS RESULT block
   (docs/research/harness-20261102/features-recheck/RESULT.txt), each unchanged, plus one line on anything that went
   wrong.
