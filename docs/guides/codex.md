# Codex / OMX Adapter

This guide is a thin adapter for Codex and oh-my-codex (OMX) work in the BotDetection research workspace.

OMX is the preferred Codex orchestration layer for planning, execution loops, role routing, and verification workflow. It stays under the repo protocols; it does not replace the project execution contract, coding behavior contract, reproducibility contract, or verified evidence.

## Read In This Order
1. [../../AGENTS.md](../../AGENTS.md)
2. [../protocols/harness-standard.md](../protocols/harness-standard.md)
3. [../protocols/agent-coding-guideline.md](../protocols/agent-coding-guideline.md)
4. [../../.agents/PLANS.md](../../.agents/PLANS.md)
5. [../../.agents/skills/README.md](../../.agents/skills/README.md)
6. [../wiki/README.md](../wiki/README.md)
7. [../../LLMbot/AGENTS.md](../../LLMbot/AGENTS.md)
8. [../../LLMbot/README.md](../../LLMbot/README.md)

## Codex-Specific Emphasis
- Prefer the current default execution surface in `LLMbot/`.
- Make evidence-first, minimal, protocol-safe changes.
- Verify with the narrowest useful tests, traces, or harness checks.
- Keep project memory in `docs/wiki/`, but do not treat it as the execution contract.
- Use `$ralplan` before complex or cross-boundary changes.
- Use `$ralph` only after a decision-complete plan exists and the task should continue through verification.
- Use repo-local role skills for narrow ownership: `$implementer`, `$test_guardian`, `$experiment_runner`, `$analysis_writer`, and `$reviewer`.

## Project Reading Tools

Use `codegraph` as a first-class project-reading aid when local structure,
symbols, or call relationships matter. Prefer it for indexed navigation and
impact discovery, then open the relevant files to confirm the result.

Useful commands:

- `codegraph status .` to check whether the workspace index exists and is fresh.
- `codegraph init .` when the workspace has no local `.codegraph/` index.
- `codegraph sync .` after edits or when the index may be stale.
- `codegraph query <symbol-or-text>` to find indexed symbols and references.
- `codegraph callers <symbol>` and `codegraph callees <symbol>` to inspect call relationships.
- `codegraph impact <symbol>` or `codegraph affected <files...>` to scope likely downstream validation.

`codegraph` complements `rg`, OMX explore, and source-file inspection. It does
not replace protocol docs, manifests, traces, tests, or verified experiment
evidence.

## Conflict Rule

If OMX prompts, runtime state, plans, or generated instructions conflict with repo protocols, manifests, traces, tests, or verified experiment evidence, the repo protocols and evidence win.

Durable project decisions should be promoted into tracked repo docs such as `docs/`, `docs/wiki/`, `docs/research/`, or `.agents/`. Treat `.omx/` as local runtime state.

## Current Default Implementation
For repo-local commands and implementation detail, use:
- [../../LLMbot/AGENTS.md](../../LLMbot/AGENTS.md)
- [../../LLMbot/README.md](../../LLMbot/README.md)
