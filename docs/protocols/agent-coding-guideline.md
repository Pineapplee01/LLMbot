# Agent Coding Guideline

This document is the project-wide coding behavior contract for local agents in the BotDetection research workspace.

It defines how agents should think, scope, edit, and verify work while operating under the project harness standard.

## Purpose And Tradeoff

- Bias toward caution, factual grounding, and research safety over raw speed.
- Prefer one correct, well-scoped change over a fast but weakly justified refactor.
- Treat uncertainty as a signal to verify or clarify, not as permission to improvise.

This guideline sits on top of [harness-standard.md](harness-standard.md). The harness standard defines how this research workspace executes work. This document defines how agents should code while doing that work.

For Codex sessions, oh-my-codex (OMX) is the preferred orchestration layer for planning, execution loops, role routing, and verification workflow. OMX must operate under this guideline and the harness standard; it is not a replacement for either contract. Superpowers is the preferred process discipline for skill selection, brainstorming, planning, and verification gates.

## The Five Local Principles

### 1. Think Before Coding

Evidence-first, not intuition-first.

Before changing code or docs, the agent should determine:

- what is already true in the repo or workspace
- what is only inferred
- what is still ambiguous
- what evidence will confirm success

Acceptable evidence includes:

- repository code and tests
- manifests, traces, metrics, and protocol artifacts
- run summaries and stage artifacts
- project wiki notes when they are clearly grounded in evidence
- direct user confirmation when intent cannot be derived safely

For creative, behavioral, architectural, method, CLI, manifest, artifact, or research-boundary changes, run the Superpowers brainstorming gate before implementation.

### 2. Simplicity First

Choose the smallest solution that satisfies the task.

In this workspace, simplicity means:

- prefer local changes over new layers
- prefer the active `LLMbot/` execution surface over parallel execution paths
- avoid speculative abstractions, helper flags, or side systems unless the task requires them
- keep compatibility and migration shims thin rather than introducing more bespoke CLIs
- do not create new source files unless the user requested them or an approved design names them

### 3. Surgical Changes

Every changed line must trace back to the task.

Agents should:

- change only the code, docs, hooks, or validation surfaces directly needed
- avoid drive-by refactors
- avoid unrelated formatting or comment churn
- avoid renaming stable artifacts without explicit need
- avoid direct edits under results and dataset artifact trees unless the task explicitly requires them
- avoid edits under `LLMbot/baseline/` and `LLMbot/code/` unless the task explicitly asks for migration, deletion, archival cleanup, or forensic comparison

### 4. Goal-Driven Execution

Define success before implementation, then verify until evidence exists.

Before coding, the agent should be able to answer:

- What is the concrete goal?
- What observable output will prove success?
- Which CLI command, manifest field, artifact path, test, or governance check is the smallest valid verification?

After coding, the agent should verify with the narrowest relevant evidence:

- focused tests only when already present or explicitly approved
- active mainline CLI or stage checks
- manifest assertions
- protocol tests
- stage outputs or trace artifacts
- governance drift checks for documentation or skill changes

Do not add new test files by default. In this project, Superpowers TDD is adapted as: define the expected CLI behavior, failure mode, or artifact contract first; then implement; then verify the contract.

### 5. Research-Safe Iteration

Protect provenance, protocol invariants, and experimental trust.

In this workspace, research-safe iteration means:

- preserve frozen protocol semantics unless the task explicitly changes them
- preserve provenance requirements and evaluation boundaries
- prefer the active `LLMbot/` mainline and its repo-local guidance
- treat trace files, manifests, metrics, and regression checks as first-class evidence
- escalate when a change risks invalidating conclusions, not just breaking code
- label unverified external implementations as `*-style` or `*-inspired` only until local paper/code inspection supports stronger wording

## Pre-Implementation Checklist

Before coding, answer these questions:

- Which skill or workflow gate applies?
- What do I know from the workspace?
- What is still ambiguous?
- What is the minimum change?
- Which files are in scope, and which are explicitly out of scope?
- How will I verify it without creating unapproved tests or artifacts?

## Anti-Patterns

- Silent assumption: acting as if missing intent or protocol meaning were obvious when they are not.
- Gate skipping: treating autonomy or OMX persistence as permission to bypass Superpowers brainstorming or project protocols.
- Speculative abstraction: adding helpers or options because they might be useful later.
- Drive-by refactor: mixing unrelated cleanup into a task that did not request it.
- Vague execution: trying to make it work without defining success checks first.
- Claim inflation: upgrading smoke, single-seed, or unverified external-code evidence into paper-facing conclusions.

## Local Standard

- [harness-standard.md](harness-standard.md) remains the execution contract.
- This document remains the coding behavior contract.
- [baseline_comparability.md](baseline_comparability.md) remains a narrower companion protocol for formal comparison work.
- `docs/wiki/` remains project memory, not a second policy source.
- `LLMbot/AGENTS.md` controls active-mainline edit behavior inside `LLMbot/`.
- OMX runtime state, prompts, plans, and generated instructions are local workflow aids. If they conflict with repo protocols, manifests, traces, metrics, tests, or verified experiment evidence, the repo evidence and protocols win.
