# Decision note (C0): keep one process; engine is not the throughput limit

Date 2026-10-14. Status: proposed by builder v42-c0, for lead sign-off at G0.

1. v4.1 throughput is model-server wait (79% of step-7 wall, `attribution.json`), not engine time (about 1.7 s CPU per
   SW-B episode, `profile.json`). Engine work for Phase 1 is not worth more effort than the one cached-key fix kept here.
2. Run episodes in one asyncio process (efficiency about 0.90 at 16, all of it straggler effect). A process pool scales to
   0.99-1.0 at 8 but costs about 4x the memory per episode on a Mac with swap at 10.8 of 12 GB; add it only if real clients
   show event-loop lag.
3. Open for the lead: starting a decision's counterfactual calls together with d1 would cut cf-all decision wait by 35%
   (1381 to 894 s per episode on step-9 timings) but changes failure-path semantics of the v4.1 decision loop; not done.
4. Log `look_ms` and `queen_ms` in the v4.2 harness; v4.1 did not, which left 21-54% of wall time unattributable.
5. "About tick 200" was a board-entry count; no episode exceeded T=120.
