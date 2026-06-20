# AGENTS.md - BotDetection Research Workspace

Read this first. It is the top-level operating contract for agents and future sessions in `G:\Research\BotDetection`.

## Authority Chain

1. [docs/protocols/harness-standard.md](docs/protocols/harness-standard.md)
2. [docs/protocols/agent-coding-guideline.md](docs/protocols/agent-coding-guideline.md)
3. [docs/protocols/baseline_comparability.md](docs/protocols/baseline_comparability.md) for formal comparison work
4. [docs/wiki/README.md](docs/wiki/README.md) for durable project memory
5. Repo-local surface docs such as [LLMbot/AGENTS.md](LLMbot/AGENTS.md), [LLMbot/README.md](LLMbot/README.md), [NLPCC/AGENTS.md](NLPCC/AGENTS.md), and [NLPCC/README.md](NLPCC/README.md)

Direct user instructions outrank this file. Deeper `AGENTS.md` files govern their subtrees when they are more specific.

## Current Active Surfaces

`LLMbot/` is the active code mainline for future bot-detection pipeline work.
`NLPCC/` is the active NLPCC paper/task line for lightweight manuscript
source, paper-task documentation, and artifact inventories.

For code work, start with:

- [LLMbot/AGENTS.md](LLMbot/AGENTS.md)
- [LLMbot/README.md](LLMbot/README.md)

For NLPCC paper work, start with:

- [NLPCC/AGENTS.md](NLPCC/AGENTS.md)
- [NLPCC/README.md](NLPCC/README.md)
- [NLPCC/docs/artifact_inventory.md](NLPCC/docs/artifact_inventory.md)

`LLMbot/baseline/` and `LLMbot/code/` are deprecated legacy surfaces scheduled for deletion. Do not add implementation, tests, or documentation there unless the user explicitly asks for migration, deletion, archival cleanup, or forensic comparison.

## Superpowers, OMX, And Project Protocols

Use this fixed division of responsibility:

- **Superpowers controls design and implementation permission.** Use `superpowers:using-superpowers` to select applicable skills. Use `superpowers:brainstorming` before design, behavior, architecture, research-boundary, CLI, manifest, artifact, or method changes. Its hard gate takes priority over ordinary autonomous-execution pressure.
- **OMX controls execution orchestration after approval.** Use OMX for planning, persistent loops, role routing, and long-running verification only after the design and implementation plan are clear. OMX is workflow tooling, not a separate source of research truth.
- **Project protocols control research meaning.** Repo protocols, manifests, traces, metrics, and verified evidence define edit zones, artifact meaning, comparison fairness, and claim safety.

OMX runtime state under `.omx/` is local execution state. Durable project decisions belong in tracked docs under `docs/`, `docs/wiki/`, `docs/research/`, `.agents/`, or repo-local governance files.

## Project Memory And Comparison Context

- `docs/wiki/` is the only project-level memory layer.
- Wiki content is durable context and decision memory, but it does not override protocol docs, manifests, traces, tests, or verified experiment evidence.
- `LMBot/`, `botbr/`, `HyperScan/`, and `SEBot/` remain comparison, baseline, legacy, or reference code lines. Use [docs/protocols/baseline_comparability.md](docs/protocols/baseline_comparability.md) for formal comparisons.
- Use [docs/research/project_phase_taxonomy.md](docs/research/project_phase_taxonomy.md) for the high-level A-G research phase vocabulary. Phase labels do not rename CLI `--stage` values.

## Quick Entry Points

- Active implementation: [LLMbot/AGENTS.md](LLMbot/AGENTS.md)
- Active mainline guide: [LLMbot/README.md](LLMbot/README.md)
- Active NLPCC paper line: [NLPCC/AGENTS.md](NLPCC/AGENTS.md)
- Active NLPCC guide: [NLPCC/README.md](NLPCC/README.md)
- NLPCC artifact inventory: [NLPCC/docs/artifact_inventory.md](NLPCC/docs/artifact_inventory.md)
- Project docs hub: [docs/README.md](docs/README.md)
- Current project memory: [docs/wiki/query_pack.md](docs/wiki/query_pack.md)
- Formal comparison protocol: [docs/protocols/baseline_comparability.md](docs/protocols/baseline_comparability.md)
- Research phase vocabulary: [docs/research/project_phase_taxonomy.md](docs/research/project_phase_taxonomy.md)
- Repo-local role skills: [.agents/skills/README.md](.agents/skills/README.md)

## Working Agreements

- Ground decisions in repo evidence, manifests, traces, metrics, tests, or explicit user direction.
- Prefer `LLMbot/` for new implementation and `NLPCC/` for NLPCC paper/task work. Treat deprecated directories as read-only unless the task explicitly names a migration or archival action.
- Record large or generated NLPCC artifacts in [NLPCC/docs/artifact_inventory.md](NLPCC/docs/artifact_inventory.md) instead of tracking or manually editing them.
- Keep design, implementation, validation, experiment execution, analysis writing, and review as separate jobs.
- Make the smallest task-scoped change; avoid drive-by cleanup and speculative abstractions.
- Do not create new source files unless the user requested them or an approved design names the path and explains why existing files are insufficient.
- Do not create new test files unless the user explicitly approves. Prefer CLI-argument checks, smoke commands, manifest inspection, and governance drift checks.
- State the validation used before closing work.

## Edit Zones

- Default research-wide zones: `LLMbot/`, `NLPCC/`, `LMBot/`, `rewrite_pipeline/`, `docs/`, `scripts/`, `.agents/`, `tools/`, plus root governance docs.
- Deprecated default-read-only zones: `LLMbot/baseline/` and `LLMbot/code/`; edit only for explicit migration, deletion, archival cleanup, or forensic comparison.
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

- Do not add dependencies for governance checks; root validation scripts must use Python standard library, Bash, PowerShell, and Git only.
- Any dependency change must explain why existing dependencies or standard-library code are insufficient.
- Dependency changes must include the affected command, environment, lock/config file, and rollback risk.

## Planning Rule

Use [.agents/PLANS.md](.agents/PLANS.md) before code changes when a task is cross-subsystem, changes interfaces or artifacts, affects evaluation logic, adds dependencies, touches comparison lines, changes research-claim boundaries, or needs multiple roles.

Superpowers `brainstorming` is required before creative or behavioral work. A plan does not bypass that hard gate.

## Role Skills

Use repo-local role skills after design and scope are clear:

- `$implementer` for approved scoped code, config, or operational-doc changes.
- `$test_guardian` for CLI-arg validation, manifest checks, regression evidence, and governance gates.
- `$experiment_runner` for experiment commands, provenance, manifests, and artifact organization.
- `$analysis_writer` only after metrics and manifest evidence exist.
- `$reviewer` for read-only engineering, research-boundary, and claim-safety review.

Prefer explicit invocation when role choice is ambiguous. Do not let one role collapse implementation, experiment execution, analysis, and final review into a single claim path.

## Done Means

- Engineering done: scope is legal, filenames are descriptive, behavior is verified by the narrowest relevant check, changed interfaces/docs are aligned, and no unapproved source/test files were added.
- Research done: engineering done plus manifest metadata is complete, metrics/artifact paths exist, comparisons follow protocol, and conclusions do not exceed evidence.
