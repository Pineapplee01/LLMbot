# Agent Coding Guideline

This document is the project-wide coding behavior contract for local agents in the BotDetection research workspace.

It defines how agents should think, scope, and verify work while operating under the project harness standard.

## Purpose and Tradeoff
- Bias toward caution, factual grounding, and research safety over raw speed.
- Prefer one correct, well-scoped change over a fast but weakly justified refactor.
- Treat uncertainty as a signal to verify or clarify, not as permission to improvise.

This guideline sits on top of [harness-standard.md](harness-standard.md). The harness standard defines how this research workspace executes work. This document defines how agents should think and code while doing that work.

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
- manifests, traces, and protocol artifacts
- run summaries and stage artifacts
- project wiki notes when they are clearly grounded in evidence
- direct user confirmation when intent cannot be derived safely

### 2. Simplicity First
Choose the smallest solution that satisfies the task.

In this workspace, simplicity means:
- prefer local changes over new layers
- prefer the existing `LLMbot/baseline/` execution surface over new parallel execution paths
- avoid speculative abstractions, helper flags, or side systems unless the task requires them
- keep compatibility and migration shims thin rather than introducing more bespoke CLIs

### 3. Surgical Changes
Every changed line must trace back to the task.

Agents should:
- change only the code, docs, hooks, or tests directly needed
- avoid drive-by refactors
- avoid unrelated formatting or comment churn
- avoid renaming stable artifacts without explicit need
- avoid direct edits under results and dataset artifact trees unless the task explicitly requires them

### 4. Goal-Driven Execution
Define success before implementation, then verify until evidence exists.

Before coding, the agent should be able to answer:
- What is the concrete goal?
- What observable output will prove success?
- Which tests, traces, or checks are the smallest valid verification?

After coding, the agent should verify with the narrowest relevant evidence:
- focused test targets
- baseline CLI or stage checks when the current execution surface provides them
- manifest assertions
- protocol tests
- stage outputs or trace artifacts

### 5. Research-Safe Iteration
Protect provenance, protocol invariants, and experimental trust.

In this workspace, research-safe iteration means:
- preserve frozen protocol semantics unless the task explicitly changes them
- preserve provenance requirements and evaluation boundaries
- prefer the current default execution surface in `LLMbot/baseline/` and its repo-local guidance
- treat trace files, manifests, and regression checks as first-class evidence
- escalate when a change risks invalidating conclusions, not just breaking code

## Pre-Implementation Checklist
Before coding, answer these questions:
- What do I know from the workspace?
- What is still ambiguous?
- What is the minimum change?
- How will I verify it?

## Anti-Patterns
- Silent assumption: acting as if missing intent or protocol meaning were obvious when they are not.
- Speculative abstraction: adding helpers or options because they might be useful later.
- Drive-by refactor: mixing unrelated cleanup into a task that did not request it.
- Vague execution: trying to make it work without defining success checks first.

## Local Standard
- [harness-standard.md](harness-standard.md) remains the execution contract.
- This document remains the coding behavior contract.
- [baseline_comparability.md](baseline_comparability.md) remains a narrower companion protocol for formal comparison work.
- `docs/wiki/` remains project memory, not a second policy source.
