# Harness Standard

This document is the project-wide execution contract for the BotDetection research workspace.

The current default execution surface is `LLMbot/baseline/`.

## Authority Chain
- This document defines the top-level execution contract.
- [agent-coding-guideline.md](agent-coding-guideline.md) defines the top-level coding behavior contract.
- [baseline_comparability.md](baseline_comparability.md) is a narrower comparison protocol that sits under this execution contract.
- `docs/wiki/` is the durable project memory layer, but it is subordinate to protocol docs, manifests, traces, and verified experiment evidence.
- Repo-local docs such as [../../LLMbot/baseline/AGENTS.md](../../LLMbot/baseline/AGENTS.md) and [../../LLMbot/baseline/README.md](../../LLMbot/baseline/README.md) are implementation-level elaborations and must stay aligned with this contract.

## Non-Negotiable Rules
- Prefer the current project execution surface in `LLMbot/baseline/` for new agent-driven work.
- Preserve frozen protocol semantics and stable artifact contracts unless a task explicitly changes them.
- Centralize new runtime glue inside the current default execution surface instead of growing more ad hoc script-level entrypoints.
- Use focused tests, manifests, and stage traces as the first verification surface for behavior changes.
- Keep fair-comparison work inside the bounds defined by [baseline_comparability.md](baseline_comparability.md).

## Current Project Defaults
- `LLMbot/baseline/` is the default project execution surface and implementation reference.
- `docs/wiki/` is the only project-level memory layer.
- `LMBot/`, `botbr/`, and `HyperScan/` remain comparison, baseline, or legacy/reference code lines; do not treat them as the default execution surface.

## Project Memory Boundary
- Wiki notes may summarize decisions, experiment history, and research context.
- Wiki notes must not redefine protocol semantics, execution rules, or artifact meaning.
- When wiki notes and verified traces disagree, traces, manifests, tests, and protocol docs win.
