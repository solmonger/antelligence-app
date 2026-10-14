## 6. Analysis (hand-written; every figure is in the tables above or in `results/*.json`)

**Targets.** Engine share under 30 s per episode: hit. SW-B costs about 1.7 s of CPU per episode on the fakes (including
one-off harness setup; the marginal cost of an extra episode in a shared process is about 0.5 s at 16 episodes). Pool
scaling at least 0.7 x linear up to 8: hit by both modes (table 4). Nothing is explained away in G0 because nothing missed.

**What actually limits v4.1 throughput.** Not the engine. In step 7 the slowest-call-per-squad, slowest-squad-per-round
path through the model servers is 79% of wall time (network_ms; vLLM reports no server time, so this is the GPU wait),
and the Qwen-9B calls alone are 350 of the 379 critical-path seconds. The other 21% of step-7 wall (about 100 s per
episode) is the **unlogged remainder**: d1 local looks, queen calls, engine, event loop. The engine part of it is 1.7 s, so
the remainder is almost entirely looks and queen calls that v4.1 did not time. Recommendation for the v4.2 harness: log
`look_ms` and `queen_ms` so this stops being an inference.
Step 9 (cf-all) is slower for a structural reason: each decision runs d1, then the router's pick, then the 4B and 9B
counterfactual calls only after d1 returns. On the recorded timings that path is 1381 s per episode against 894 s if all
of a decision's calls started together (attribution.json, step9 recorded view, `whatif_all_calls_started_together_wait_s`):
a 35% cut for cf-all runs. I did **not** implement it: it issues counterfactual calls even when d1 later fails for good,
which changes attempt logs, tallies and replay behaviour in the failure path of the v4.1 decision loop. It needs a lead decision,
and only matters if Phase 1 uses `--log-counterfactuals all`. The step-9 and Bonsai runs were also slowed by what ran beside
them (see section 1: Bonsai calibration on the same pod, the Mac Bonsai server, the 20 GB lead process); step-9 network_ms
means were 19-25 s per call against 7.5-14 s in step 7.

**Fixes.** Kept: cached Ed25519 key objects in `kernel/board.py:sign_id` (the key was re-parsed on every signature):
identical signatures, paired saving about 0.15 s per episode, test `test_sign_id_cache_is_bit_identical_to_parsing_each_time`.
Tried and removed: a bounding-box `DrugField` diffusion (bit-identical, but slower: the crop is only 191k voxels, so a full
gaussian_filter is about 1 ms and the support fills the volume within a few ticks). Not done, because the profile does not
justify them: A* caching (176 path calls, 0.05 s per episode in total), fewer full-volume operations (coverage scoring and metrics, the full-volume work left once the diffusion itself is excluded, total 0.03 s), batched hashing (a whole board append, hash plus sign plus store, is 0.37 s for 439 entries, so batching could save a fraction of that), and a process pool in
the runner (below). Metrics were identical before and after on the profiled case, three more cases, a board-off run, and 5 S plus 5 S'
episodes.

**Process pool or one process.** At the recorded latencies (scale 1.0) 16 episodes in one process take 1312 s against 1181 s for
one episode: 43.9 episodes/h, efficiency 0.90, with event-loop lag p95 2.2 ms (the probe floor) and 34 s of total CPU. The
0.10 loss is therefore **not** the engine: episodes started together finish when the slowest of the 16 finishes, and
the seeded latency draws differ per case. A pool of 8 processes takes 1181 s for 8 episodes, efficiency 1.0 (the same
straggler effect is smaller at 8, and two different case sets are compared, so read both as 0.9-1.0 rather than a ranking). The
stress runs (scale 0.1 and 0.01) agree: in-process efficiency 0.90 at 16, pool 0.99 at 8, lag p95 at most 20 ms. The pool
costs memory: eight processes at about 170 MB each is 1.4 GB, against 313 MB for 16 episodes in one process, on a Mac whose
swap was already 10.8 of 12 GB. Since the engine is about 0.2% of a recorded-latency episode, **keep one process**; a pool
would only matter if real HTTP clients make the loop lag, which this study cannot measure.

**Limits of this study.** Fakes skip HTTP, JSON parsing and TLS, so the real client adds CPU this does not measure. The
queen is a non-sleeping fake. Latencies come from step 7 (live, uncongested); step 9 was slower. Timings move about 20% with
machine load (before and after were taken at different loads; only the paired A/B is a fair comparison). Scale-0.1 and
scale-0.01 runs shorten the sleeps 10x and 100x, which makes the engine look 10-100x more prominent than it is; the scale-1.0
rows use the recorded latencies unscaled.

**Ticks.** No episode ever ran past T=120. The "about tick 200" in the 14 Oct 09:50 log line matches the **board entry
count**: the 26 step-9 episodes the restart killed left no run log, and their highest OTS anchor checkpoint (anchors fall at board
sizes 50, 100, ...) has median 200, while a finished step-7 SW-B episode has 421-463 board entries (median 438). The 10:10 line already speaks of
"board steps". Test: `test_episode_ends_at_tick_120_and_the_board_is_longer`.

**Attempts, all reported.** A smoke run before any file was kept (console only): CPU 1.936 s on BRATS_016. Then
`phases-before` (3 repeats), `scripted-before` (no fingerprints, kept) and `scripted-before-b` (with fingerprints; the one
used for identical-metric checks), `phases-after` (attempt A: bbox diffusion plus key cache, reverted), `ab-sign-cache`,
`phases-after-final`, `concurrency-s0.1`, `concurrency-s0.01`, `concurrency-s1.0`. The before-fix scripted runs were taken by stashing the
fixes, running, and restoring them; the phases-before run preceded any code change.
