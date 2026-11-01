# Dreams

A log of the nightly dream run: five ideas a night, one built and tested for free against the current best.
Verdicts: beats best, no better, broken.

## 2026-10-30

Ideas:
1. Waggle-dance marks: ants with more of their own evidence leave stronger marks, capped (honeybee dance length).
2. Surprisingly popular: each ant also guesses what others will say; trust the answer more popular than predicted.
3. Lateral inhibition: a mark weakens nearby marks that say the same thing, so one crowd can't stack up.
4. Annealed listening: ants listen little early and more later, like cooling metal.
5. Reputation staking: ants stake credibility on marks and earn it back when later readers agree.

Picked: 1, the simplest to test and closest to think alone.

Built: rumour world policy `graded` (mark strength = net private readings, at least 1, cap 3; liars shout at the cap)
and a fair check that tunes every arm's listening weight on seeds 0 to 9 and scores it on fresh seeds.
Branch claude/dream-2026-10-30, commit 2e437b8. Write-up: docs/research/swarm-scale-20261029/GRADED_MARKS.md.

Note: the night's own run could not push (the fork was not in its session's repos), so its code was lost. This entry
is from the rebuild the same morning, re-tested from scratch on new fresh seeds (100 to 119 at 256 ants, 100 to 109
at 1,024). The rebuild picked the same weights and its gains are within about half a point of the night's.

Numbers (share of honest ants right, fixed fade, graded vs think alone tuned to its best loudness):
- No seed: +0.018 (256 ants), +0.022 (1,024). Burst: same. Liars: +0.010 (256), +0.015 (1,024). Every 95% CI above 0.
- Against think alone as it stands in PR 21 (weight 1): about +0.06.
- Cost: liars at cap 3 add +0.014 damage. Cap 2 keeps +0.015 of the gain at +0.006 damage.

Verdict: beats best (about 2 points over a fairly tuned think alone).

Next: port to colouring and foraging; decide cap 2 or 3; then the fair test with models.

Follow-up, same day: ported to colouring and foraging (cap 2) on fresh seeds. No gain in either; slightly worse under
liars in colouring. Waggle-dance marks are a rumour-world idea, not a default for every world. Branch
claude/dream-2026-10-30, docs/research/swarm-scale-20261029/GRADED_MARKS.md.

## 2026-10-31

Ideas:
1. Field notebook: each ant remembers the latest mark it read from each other ant, anywhere, and counts each source once
   (scrub-jay cache memory; a reporter's list of sources; counting by source, not by how many times a claim is heard).
2. Cross-inhibition stop signals: an ant sure of its own evidence weakens rival marks at its site (honeybee head-butts).
3. Olympic trimming: drop the loudest marks before adding up, so liars shouting at the cap count for less.
4. Two-speed memory: a fast trace and a slow trace, trust the slow one unless the fast one strongly disagrees.
5. Sparse listening: read only a few random marks per site, like bagging, to stop pile-ons.

Picked: 1. Until now an ant forgot everything it heard as soon as it walked to the next site.

Built: rumour world option `notebook` (off by default, bit-identical when off). What an ant says is unchanged: only its
own evidence, so nothing heard is passed on. Fair check `scripts/notebook_check.py`: every arm's listening weight and
notebook fade tuned on seeds 0 to 9, tested on fresh seeds 300 to 319 (256 ants) and 300 to 309 (1,024).
Branch claude/dream-2026-10-31, draft PR 24 (stacked on PR 22). Write-up: docs/research/swarm-scale-20261029/NOTEBOOK.md.
Tests: 209 passed (after the rebuild).

Numbers (share of honest ants right, waggle marks + notebook vs waggle marks alone, the current best, both tuned):
- 256 ants: no seed 0.994 vs 0.909 (+0.085 [+0.072, +0.098]); burst +0.082; liars 0.985 vs 0.885 (+0.100). 20 of 20 seeds higher.
- 1,024 ants: no seed 0.997 vs 0.912 (+0.085); burst +0.084; liars 0.990 vs 0.886 (+0.105). 10 of 10 seeds higher.
- Cost: wrong marks linger in notebooks. Damage during a burst rises by about 4.5 points (0.014 to 0.059) and liar
  damage by about 1.2 points, even though more ants end up right.
- The best notebook fade (40 to 80 ticks) is about as long as the run: close to "remember every source once".

Verdict: beats best (about 8 to 10 points).

Push: refused again (the fork is not in this run's repos). Rebuilt the same day from this run's record by a daytime
session; the fair check re-ran from scratch and matched to the last digit. That session also added a probe with a
burst from many authors: the gain holds (+0.079 to +0.082) and burst damage rises to about 0.07.

Next: let notebooks fade wrong sources faster (cut the burst cost); port to colouring and foraging; then the model fair test.


## 2026-11-01

Ideas:
1. One source, one voice: the notebook remembers any source as at most one ant's strongest word (one person, one vote;
   clipping outliers in robust statistics).
2. Seen twice to stick: a source seen writing once fades fast, seen writing twice it fades slowly (short-term memory
   becoming long-term on a second exposure).
3. Quorum of places: trust a claim more when agreeing sources were met at several different sites (independent
   replication in science).
4. Date the evidence: age a remembered mark from when it was written, not when it was read (carbon dating).
5. Warm-up before trusting: lean on the notebook only once an ant has some evidence of its own (immune self-tolerance
   training).

Picked: 1, the plainest fix for last night's cost (wrong marks lingering in notebooks). Tested 2 in the same check.

Built: rumour world options `notebook_one_voice` and `notebook_consolidate` (off by default, bit-identical when off).
Fair check `scripts/one_voice_check.py`: every arm, the current best included, tuned on seeds 0 to 9, tested on fresh
seeds 400 to 419 (256 ants) and 400 to 409 (1,024). Branch claude/dream-2026-11-01, draft PR 26 (on top of PR 24).
Write-up: docs/research/swarm-scale-20261029/ONE_VOICE.md. Tests: 266 passed in tests/smarter_swarm (73 new); 282 in smarter_swarm and swarm_scale after the rebuild.

Numbers (one voice vs the current best, waggle marks + notebook, both tuned; picks the same, weight 3, half-life 80):
- Burst harm: 0.056 to 0.044 at both sizes (-0.012 [-0.013, -0.011]), lower on 30 of 30 seeds.
- Share right: same with no seed and with liars (exactly), a hair higher under a burst (+0.0004), never lower on any seed.
- Seen twice: cuts burst harm to 0.047 but loses a little right under liars and at 1,024 ants; tuning pushed it toward
  off. Both together: harm 0.038, same small accuracy cost. Not worth it yet.

Verdict: beats best (same answers, a fifth less harm from a burst of misinformation). Small, clean win.

Push: refused again (the fork is not in this run's repos). Rebuilt the same day from this run's record by a daytime
session; the fair check re-ran from scratch and matched. That session also checked one voice on PR 24's seeds with a
burst from one author and from many: burst harm down 0.012 and 0.015 to 0.016, every seed lower, never fewer right.

Next: what harm is left comes from the first ticks; try the warm-up idea (5). Then port the notebook to colouring and
foraging, then the model fair test.

