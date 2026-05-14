# Conventions

This file is a routing index for the active project standards. It is no longer the primary execution or coding contract.

## Primary Standards
- [docs/protocols/harness-standard.md](protocols/harness-standard.md) — project-wide execution contract
- [docs/protocols/agent-coding-guideline.md](protocols/agent-coding-guideline.md) — project-wide coding behavior contract
- [docs/protocols/baseline_comparability.md](protocols/baseline_comparability.md) — narrower companion protocol for formal comparison work
- [docs/guides/codex.md](guides/codex.md) — Codex/OMX orchestration adapter under the repo protocols

## Current Defaults
- The current default execution surface lives in `LLMbot/`.
- Repo-local implementation guidance lives in [../LLMbot/AGENTS.md](../LLMbot/AGENTS.md) and [../LLMbot/README.md](../LLMbot/README.md).
- `LLMbot/baseline/` and `LLMbot/code/` are deprecated legacy/reference surfaces scheduled for deletion.
- `LMBot/` remains a legacy/reference implementation line.
- `docs/wiki/` is the only project-level memory layer.
- Wiki notes are durable context, but they do not override protocol docs, manifests, traces, or verified experiment evidence.
- The canonical high-level research phase names live in [research/project_phase_taxonomy.md](research/project_phase_taxonomy.md). They are planning labels, not CLI `--stage` names.
- OMX runtime state and generated plans are local workflow aids. Durable decisions belong in tracked docs, and repo protocols plus verified evidence win on conflict.

## Where to Go Next
- Need the project entrypoint: [../AGENTS.md](../AGENTS.md)
- Need the current implementation details: [../LLMbot/AGENTS.md](../LLMbot/AGENTS.md)
- Need the current repo-local guide: [../LLMbot/README.md](../LLMbot/README.md)
- Need baseline comparison rules: [protocols/baseline_comparability.md](protocols/baseline_comparability.md)
- Need Codex/OMX workflow guidance: [guides/codex.md](guides/codex.md)
- Need research phase naming: [research/project_phase_taxonomy.md](research/project_phase_taxonomy.md)
- Need persistent project context: [wiki/README.md](wiki/README.md)
