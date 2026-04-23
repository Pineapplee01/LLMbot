# Codex Adapter

This guide is a thin adapter for Codex-style agent work in the BotDetection research workspace.

## Read In This Order
1. [../protocols/harness-standard.md](../protocols/harness-standard.md)
2. [../protocols/agent-coding-guideline.md](../protocols/agent-coding-guideline.md)
3. [../wiki/README.md](../wiki/README.md)
4. [../../LLMbot/baseline/AGENTS.md](../../LLMbot/baseline/AGENTS.md)
5. [../../LLMbot/baseline/README.md](../../LLMbot/baseline/README.md)

## Codex-Specific Emphasis
- Prefer the current default execution surface in `LLMbot/baseline/`.
- Make evidence-first, minimal, protocol-safe changes.
- Verify with the narrowest useful tests, traces, or harness checks.
- Keep project memory in `docs/wiki/`, but do not treat it as the execution contract.

## Current Default Implementation
For repo-local commands and implementation detail, use:
- [../../LLMbot/baseline/AGENTS.md](../../LLMbot/baseline/AGENTS.md)
- [../../LLMbot/baseline/README.md](../../LLMbot/baseline/README.md)
