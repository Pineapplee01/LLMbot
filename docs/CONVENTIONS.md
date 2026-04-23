# Conventions

This file is a routing index for the active project standards. It is no longer the primary execution or coding contract.

## Primary Standards
- [docs/protocols/harness-standard.md](protocols/harness-standard.md) — project-wide execution contract
- [docs/protocols/agent-coding-guideline.md](protocols/agent-coding-guideline.md) — project-wide coding behavior contract
- [docs/protocols/baseline_comparability.md](protocols/baseline_comparability.md) — narrower companion protocol for formal comparison work

## Current Defaults
- The current default execution surface lives in `LLMbot/baseline/`.
- Repo-local implementation guidance lives in [../LLMbot/baseline/AGENTS.md](../LLMbot/baseline/AGENTS.md) and [../LLMbot/baseline/README.md](../LLMbot/baseline/README.md).
- `LMBot/` remains a legacy/reference implementation line.
- `docs/wiki/` is the only project-level memory layer.
- Wiki notes are durable context, but they do not override protocol docs, manifests, traces, or verified experiment evidence.
- The canonical high-level research phase names live in [research/project_phase_taxonomy.md](research/project_phase_taxonomy.md). They are planning labels, not CLI `--stage` names.

## Where to Go Next
- Need the project entrypoint: [../AGENTS.md](../AGENTS.md)
- Need the current implementation details: [../LLMbot/baseline/AGENTS.md](../LLMbot/baseline/AGENTS.md)
- Need the current repo-local guide: [../LLMbot/baseline/README.md](../LLMbot/baseline/README.md)
- Need baseline comparison rules: [protocols/baseline_comparability.md](protocols/baseline_comparability.md)
- Need research phase naming: [research/project_phase_taxonomy.md](research/project_phase_taxonomy.md)
- Need persistent project context: [wiki/README.md](wiki/README.md)
