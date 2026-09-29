# Start here: Antelligence evidence-memory review

**Ready for critical review, not approval to integrate or a claim of improved AI accuracy.**

This self-contained sandbox lets a reviewer inspect and run the recent C0–F3 research components without the author's machine, private logs, model accounts or development branch. Nothing in this directory is wired into the application runtime. The PR deliberately does not publish the full local development history.

## What Antelligence is trying to do

Small agents should cooperate using limited, attributed information, retain useful evidence, and verify whether their actions actually succeeded. Tumor simulation is one application; source-backed research is another. The question is whether this machinery earns its complexity compared with a simpler solver.

## What was built

| Piece | Purpose | Source in this snapshot |
|---|---|---|
| C0: common rules | Separate public evidence from trusted outcomes; unknown is not success | `backend/research_hive_contracts.py` |
| F1: evidence memory | Reuse evidence with source lineage and explicit invalidation | `backend/research_hive_memory.py` |
| F2: task planning/checking | Bounded composition and independent current-state verification | `backend/research_hive_tasks.py`, `backend/research_hive_verifier.py` |
| F3: communication | Validate bounded requests/replies and gate real planner calls | `backend/research_hive_communication.py` |
| Compatibility pilot | Translate existing coldroom tasks, run the new pieces, replay in the old engine | `pilot/fit_pilot.py`, `backend/research_coldroom.py` |

The memory, task and communication components were previously reviewed locally. F3's final source review found no code defects, but the restricted reviewer could not independently hash its snapshots; the controller supplied that binding and preserved the limitation. That history does **not** constitute this PR's independent approval. The broader historical controller suite is not all included here; use the fresh snapshot results below, not earlier test totals.

## Reproduce without model calls

Python **3.11** was exercised in a fresh environment with only pytest and its dependencies. Runtime code uses the standard library. From the repository root:

```sh
cd review/hive-fit-20260911
python3.11 -m venv .venv
.venv/bin/python -m pip install pytest==9.0.3
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests pilot/test_fit.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python pilot/run.py --output output/run-01
```

The runner verifies source hashes, creates a **new** output directory, runs all assigned cases, then replays the saved action lists in the unchanged old evaluator. Use `output/run-02` for another run; existing matrix directories are refused. Do not use Python's `-O` option: the research audit uses assertions. This is a development harness, not a hardened untrusted-file service.

Inspect the generated `summary.json`, `rows.jsonl`, each case's `trace.json`, and `unreported-drift.json`. Raw traces and databases stay local. The checked-in `evidence/summary.json` is an observed result, not a substitute for rerunning.

Fresh packaging checks (including manifest-coverage and moved-output audit regressions): **167 tests passed**, **480 assigned matrix rows replayed**, plus **24 unreported-drift diagnostic cases**. These are one small deterministic task family, not independent real-world trials. See `evidence/verification.json` and `SOURCE-MANIFEST.json` for source identity and packaging changes.

## Observed behavior

24 task variants per row; all attempts remain in their denominators.

| Condition | Existing memory-only | New memory + communication | Simple solver with current facts |
|---|---|---|---|
| Useful memory | 24 completed | 24 completed | 24 completed |
| Empty memory, fresh reply available | 24 stopped | 24 completed | 24 completed |
| Reported source change, fresh reply available | 24 failed with rejected attempts | 24 completed | 24 completed |
| Reported source change, no reply | 24 failed with rejected attempts | 24 stopped before actions | 24 completed |

New memory **without** communication completes only the useful-memory condition and stops in the other conditions. The new full-information composer also completes all conditions. Successful recovery costs a request and a reply; useful-memory cases use no transport messages.

**Critical control:** when the source changes without an invalidation notice, the new memory still returns stale information in 24/24 diagnostic cases. All resulting plans fail the current-state verifier/replay. Do not describe source IDs, revisions or timestamps as automatic truth detection.

## What this evidence supports—and does not

- Supports simulator compatibility, recovery when correct information is supplied, and stopping when known-invalid evidence cannot be refreshed.
- Does **not** beat the simpler fully informed solver. It adds communication overhead.
- New arms receive a trusted source-change notice that the legacy memory path does not. This intentionally tests an added mechanism, **not** equal-information reasoning ability.
- Caches and replies are seeded by trusted harness code. No learned memory or actual multi-model reasoning was measured.
- No accuracy lift, clinical efficacy, live-finance suitability, token savings or latency advantage has been demonstrated.
- The task-condition mix is artificial. Do not aggregate it into a headline improvement percentage, infer significance or call it held-out evaluation.
- This is not integration into the live workbench, tumor engine, API or UI. It is not completion of F4/F5/F6 or release approval.

## Review agenda: please challenge this

1. **Value:** does information recovery justify this architecture, or would ordinary retrieval plus a single solver do the same job more cheaply?
2. **Validity:** are task translation and legacy replay faithful? Can any method see privileged facts through a label, ID or setup artifact?
3. **Freshness and trust:** who would supply invalidation notices in a real task, and which evidence must be checked again immediately before action?
4. **Fair next experiment:** what one source-backed task type can distinguish selective memory/consultation from a capable solo under measured input/output tokens, time and failures?
5. **Scope:** which parts should be integrated, simplified or parked? Do not assume Queen recovery or a viewer must be built before measuring value.

Return findings with file/line references, concrete counterexamples, and one recommended next experiment with a stop criterion. Distinguish a bug from an explicitly excluded capability. Do not merge, deploy, restart feature workers, change model routing, or buy inference as part of reviewing this packet.

## Current work state

- C0–F3: implemented and locally frozen; reviewable snapshot here.
- F5 behavior replay: pending; required feature-root provider previously hit its usage limit. No current quota recovery is asserted.
- F4 Queen recovery: pending.
- F6 fair model comparison: pending, separately budget/provider/evaluator-gated.
- Full application integration and combined independent review: pending.
- This PR: a bounded public review artifact, not the complete current application distribution.

Original local source identity: `81e6289149f3f4a743ba046c494acad4f37258cd`. Original pilot: `a4001f12c19d8fc77a29f8017347cbdf564e3f96`. These identifiers document provenance; their original commits need not be available on GitHub. The exact source files required to run this packet are included and hash-listed.
