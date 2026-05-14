# Harness Standard

This document is the project-wide execution contract for the BotDetection research workspace.

The current default execution surface is `LLMbot/`.

## Authority Chain

- This document defines the top-level execution contract.
- [agent-coding-guideline.md](agent-coding-guideline.md) defines the top-level coding behavior contract.
- [baseline_comparability.md](baseline_comparability.md) is a narrower comparison protocol that sits under this execution contract.
- `docs/wiki/` is the durable project memory layer, but it is subordinate to protocol docs, manifests, traces, metrics, tests, and verified experiment evidence.
- Repo-local docs such as [../../LLMbot/AGENTS.md](../../LLMbot/AGENTS.md) and [../../LLMbot/README.md](../../LLMbot/README.md) are implementation-level elaborations and must stay aligned with this contract.
- For Codex sessions, oh-my-codex (OMX) is preferred orchestration tooling under this contract. It does not redefine execution authority, artifact meaning, or research evidence standards.
- Superpowers is preferred design and implementation-gate tooling under this contract. It does not override direct user instructions, repo protocols, or verified evidence.

## Non-Negotiable Rules

- Prefer `LLMbot/` for all new agent-driven implementation work.
- Treat `LLMbot/baseline/` and `LLMbot/code/` as deprecated legacy surfaces scheduled for deletion. Edit them only for explicit migration, deletion, archival cleanup, or forensic comparison.
- Preserve frozen protocol semantics and stable artifact contracts unless a task explicitly changes them.
- Centralize new runtime glue inside the active mainline instead of growing ad hoc script-level entrypoints.
- Use focused CLI checks, manifests, stage traces, and artifact inspection as the first verification surface for behavior changes.
- Do not add new test files unless the user explicitly approves; define expected CLI behavior or artifact contracts before implementation.
- Keep fair-comparison work inside the bounds defined by [baseline_comparability.md](baseline_comparability.md).

## Current Project Defaults

- `LLMbot/` is the active implementation surface and current mainline.
- `docs/wiki/` is the only project-level memory layer.
- `LMBot/`, `botbr/`, `HyperScan/`, and `SEBot/` remain comparison, baseline, legacy, or reference code lines; do not treat them as the default execution surface.
- `.omx/` is local runtime state. It is not durable project policy.
- `docs/superpowers/` may store reviewed Superpowers specs and plans when a task needs a formal design gate.

## Workflow Boundaries

- Superpowers controls whether design or implementation may begin. Its `brainstorming` hard gate applies to design, behavior, architecture, method, CLI, manifest, artifact, and research-boundary changes.
- OMX controls how approved work is orchestrated. Ralph/team loops may persist execution, but they cannot bypass Superpowers gates or repo research protocols.
- Repo-local role skills under `.agents/skills/` control narrow ownership for implementation, validation, experiment running, analysis writing, and review.

## Project Memory Boundary

- Wiki notes may summarize decisions, experiment history, and research context.
- Wiki notes must not redefine protocol semantics, execution rules, artifact meaning, or claim boundaries.
- When wiki notes and verified traces disagree, traces, manifests, metrics, tests, and protocol docs win.
- When OMX runtime state, prompts, or generated plans conflict with verified traces, manifests, tests, metrics, or protocol docs, the repo evidence and protocols win.
