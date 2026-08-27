# Antelligence — VISION.md

> What success looks like. Read by every loop, every agent, every run.

## What We're Building

Antelligence is a **DeSci swarm-intelligence platform** — a system where small autonomous agents (nanobots) coordinate through local signals to solve complex biomedical problems. 

First target: **glioblastoma tumor simulation**. Long arc: any disease where swarm coordination beats centralized control.

## The Stack

```
Simulation Layer     → Tumor + nanobot swarm physics
Communication Layer  → Pheromone protocol, agent-to-agent signaling
Memory Layer         → Shared on-chain/off-chain state, proof commitments
Verification Layer   → SP1/Groth16 cryptographic proofs of simulation integrity
Coordination Layer   → Queen/worker hierarchy, escalation, strategic guidance
```

## Core Principles

### 1. Local Signals, Global Intelligence
Agents communicate through **pheromones** — local, decaying signals that propagate through the grid. No agent sees the whole picture. Intelligence emerges from local rules.

### 2. Proof-Backed Provenance
Every simulation run must be **replayable, verifiable, and attributable**. The system stages proof artifacts (mock → staged → cryptographically accepted) with honest labeling at each tier. Nothing is claimed "verified" until it actually is.

### 3. Agent-to-Agent, Not Human-to-Agent
The system is designed for **autonomous agents to coordinate with each other**, not for humans to micromanage each step. The operator sets goals and constraints. The swarm figures out execution within those bounds.

### 4. Privacy-Aware Architecture
What goes on-chain is minimal and non-sensitive. Simulation data, patient-derived parameters, and internal agent state stay off-chain or hash-committed only. ZK proofs verify correctness without revealing inputs.

### 5. Local-First, Scale Later
The system runs local simulations first. Blockchain integration (Base Sepolia) is for provenance, not for compute. Cheap local models handle the autonomous loops. Paid frontier APIs are reserved for breakthrough analysis.

## What We're NOT Building (Right Now)

- ❌ A clinical deployment platform
- ❌ Real patient data ingestion
- ❌ Live nanobot hardware control
- ❌ A polished consumer product
- ❌ Glass-box hardware simulation

## Success Looks Like

A swarm simulation where:
- Nanobots coordinate through pheromone trails without central control
- Tumor kill rates are measurable and reproducible
- Every run produces a verifiable proof artifact
- The system improves itself through autonomous loop cycles
- A third party can verify a simulation's integrity without trusting the runner

## The Autonomous Loop Contract

This project runs on autonomous loops. The loops:
1. **Discover** — read repo state, backlog, and current VISION
2. **Plan** — pick the next smallest verifiable delta
3. **Execute** — write code/tests following test-driven development
4. **Verify** — run tests, sandbox, audit gates
5. **Iterate** — ship or fix, then loop again

The loop never commits, pushes, deploys, or spends money without operator approval.
The loop leaves evidence: files changed, commands run, tests passed/failed.

## Reference Architecture

- `backend/` — simulation engine, API, proof helpers, chain integration
- `blockchain/` — Solidity contracts (TumorIntel), Hardhat config
- `tests/` — pytest suite (currently 228+ tests)
- `frontend/` — React/Vite UI for simulation visualization
- `docs/` — proofs, specs, release notes

## Key Decisions (from Jarvis's Vault)

- Core logic over glass-box hardware polish
- Local-first cost discipline (Gemma4/Qwen3-Coder for loops)
- Proof staging: `mock → staged → verified_onchain` with honest labeling
- Public values on-chain: 5 fields only (config_hash, kill_rate_bps, nanobot_count, tumor_radius, steps)
- Richer provenance stays in proof-bundle metadata
- 5-layer verification: prompt mandate → runner gate → sandbox → audit → push gate
