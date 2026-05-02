# Antelligence Progress Notes

Purpose
- Keep a concise development record for future publication, patent drafting, and technical retrospectives.
- Focus on architectural milestones, verifier/proof evolution, deployment checkpoints, and validation evidence.

## 2026-05-02 — Release review manifest prepared

The dirty tree has been organized for release review instead of receiving another audit-only green-test note. The detailed dated manifest is `docs/release-notes-2026-05-02.md`.

Summary:
- Intentional review groups are split into proof lifecycle/trust tiers, chain/runtime integration, simulation persistence/replay, frontend preview UI, and dependency/lockfile changes.
- Validation evidence recorded from the latest autonomous scorecard: `uv run --extra test pytest -q` passed with 198 tests, and `npm --prefix frontend run build` passed.
- Likely exclusions are called out explicitly: agent state, local cache/data, generated artifacts, root debug/patch probes, and test-side debug probes.
- Recommended PR split is five parts: proof core, chain/runtime integration, simulation/replay, frontend preview UI, and cleanup-only artifact removal.

Release-review caution:
- Review `frontend/.env.preview` before inclusion to confirm it contains only non-secret preview defaults.
- Exclude `frontend/dist/`, `out/`, `cache/`, local SQLite data, and the local BraTS archive unless a reviewer intentionally asks for generated/local artifacts.
- Promote only durable tests; discard temporary debug/probe scripts before PR handoff.

Pass 7 addendum:
- Current dirty inventory was refreshed at 85 paths: 25 modified tracked paths and 60 untracked paths.
- `docs/plans/` should be treated as planning context, not automatically bundled into implementation PRs.
- `uv.lock` should be reviewed with Python dependency changes rather than slipped into an unrelated cleanup.

## 2026-04-11 — Backend / blockchain / proof-path foundation

... (rest of existing content)
