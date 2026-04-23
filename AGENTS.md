# AGENTS.md - BotDetection Research Workspace

Read this first. It is the project entrypoint for agents and future sessions.

## Authority Chain
1. [docs/protocols/harness-standard.md](docs/protocols/harness-standard.md)
2. [docs/protocols/agent-coding-guideline.md](docs/protocols/agent-coding-guideline.md)
3. [docs/protocols/baseline_comparability.md](docs/protocols/baseline_comparability.md) for formal comparison work
4. [docs/wiki/README.md](docs/wiki/README.md) for durable project memory
5. Repo-local docs such as [LLMbot/baseline/AGENTS.md](LLMbot/baseline/AGENTS.md) and [LLMbot/baseline/README.md](LLMbot/baseline/README.md) for current implementation detail

## Current Default Implementation
The current default execution surface lives in `LLMbot/baseline/`.

Start with:
- [LLMbot/baseline/AGENTS.md](LLMbot/baseline/AGENTS.md)
- [LLMbot/baseline/README.md](LLMbot/baseline/README.md)

## Project Memory and Comparison Context
- `docs/wiki/` is the only project-level memory layer.
- Wiki content is durable context and decision memory, but it does not override protocol docs or verified experiment evidence.
- `LMBot/`, `botbr/`, and `HyperScan/` remain important comparison, baseline, or legacy/reference code lines. Use [docs/protocols/baseline_comparability.md](docs/protocols/baseline_comparability.md) when making formal comparisons.
- Use [docs/research/project_phase_taxonomy.md](docs/research/project_phase_taxonomy.md) for the high-level A-G research phase vocabulary. These phase labels do not rename CLI `--stage` values.

## Quick Entry Points
- Project docs hub: [docs/README.md](docs/README.md)
- Current project memory: [docs/wiki/query_pack.md](docs/wiki/query_pack.md)
- Current implementation repo: [LLMbot/baseline/AGENTS.md](LLMbot/baseline/AGENTS.md)
- Current repo-local guide: [LLMbot/baseline/README.md](LLMbot/baseline/README.md)
- Formal comparison protocol: [docs/protocols/baseline_comparability.md](docs/protocols/baseline_comparability.md)
- Research phase vocabulary: [docs/research/project_phase_taxonomy.md](docs/research/project_phase_taxonomy.md)

## Working Agreements
- Ground decisions in repo evidence, manifests, traces, tests, or explicit user direction.
- Prefer `LLMbot/baseline/` for new implementation unless the task names another surface.
- Keep implementation, experiment execution, analysis writing, and review as separate jobs.
- Make the smallest task-scoped change; avoid drive-by cleanup and speculative abstractions.
- State the validation used before closing work.

## Edit Zones
- Default research-wide zones: `LLMbot/`, `LMBot/`, `rewrite_pipeline/`, `docs/`, `scripts/`, `.agents/`, `tools/`, plus root governance docs.
- Comparison-only zones: `botbr/`, `HyperScan/`, and `SEBot/`; edit only for explicit baseline-comparison work.
- Evidence zones are read/verify by default, not manual-edit targets: `datasets/`, `results/`, `LLMbot/saved_artifacts/`, `LLMbot/checkpoints/`, and `LMBot/results_*`.
- Restricted zones: `docs/published/`, `.claude/worktrees/`, caches, temp dirs, generated logs, and `__pycache__/`.
- If a task needs an evidence-zone write, it must be produced by a documented command and accompanied by a manifest or run note.

## Filename Conventions
- Use descriptive lowercase snake_case for new scripts, docs, configs, and manifests.
- Keep established fixed names: `AGENTS.md`, `README.md`, `SKILL.md`, `PLANS.md`, and `manifest.json`.
- Prefer dated memory docs as `topic_YYYY-MM-DD.md`.
- Preserve established experiment path tokens such as `seed_<n>`, `results_<backbone>`, and `stage_<name>`.
- Do not create vague names such as `temp`, `new`, `final`, `v2`, `misc`, or `helper` unless required by an external interface.

## Dependency Policy
- Do not add dependencies for governance checks; root validation scripts must use Python standard library, Bash, and Git only.
- Any dependency change must explain why existing dependencies or standard-library code are insufficient.
- Dependency changes must include the affected command, environment, lock/config file, and rollback risk.

## Planning Rule
- Use [.agents/PLANS.md](.agents/PLANS.md) before code changes when a task is cross-subsystem, changes interfaces or artifacts, affects evaluation logic, adds dependencies, touches comparison lines, or needs multiple roles.

## Role Skills
- Use `$implementer` for scoped implementation.
- Use `$test_guardian` for tests, gates, and regression evidence.
- Use `$experiment_runner` for experiment execution and manifest capture.
- Use `$analysis_writer` only after metrics and manifest evidence exist.
- Use `$reviewer` for final engineering or research-claim review.
- Prefer explicit invocation when role choice is ambiguous.

## Done Means
- Engineering done: scope is legal, filenames are descriptive, behavior is verified by the narrowest relevant check, and changed interfaces/docs are aligned.
- Research done: engineering done plus manifest metadata is complete, metrics/artifact paths exist, comparisons follow protocol, and conclusions do not exceed evidence.
