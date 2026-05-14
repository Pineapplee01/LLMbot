# AGENTS.md - LLMbot Active Mainline

This file governs `LLMbot/` and every child path under it.

## Active Mainline

- `LLMbot/` root is the only active mainline for future code and research pipeline work.
- Run the main CLI from this directory with `python main.py ...`.
- Treat `baseline/` and `code/` as deprecated legacy surfaces scheduled for deletion.
- Do not add or modify implementation, tests, or documentation under `baseline/` or `code/` unless the user explicitly asks for migration, deletion, archival cleanup, or forensic comparison.

## Edit Boundaries

- Default editable source files are root-level mainline modules, including `main.py`, `parser_args.py`, `trainer.py`, `estimators.py`, `operators.py`, `model_building.py`, `subgroups.py`, `GNNs.py`, `RGT.py`, `LM.py`, and `dataloader.py`.
- Shared utility helpers belong in the `utils/` package. Do not recreate a root-level `utils.py`.
- Package directories such as `models/`, `trainers/`, `evaluation/`, `data/`, and `utils/` are editable only when the task names that subsystem or the existing mainline code already imports it.
- Evidence and generated artifact directories are read/verify by default: `checkpoints/`, `saved_artifacts/`, caches, `__pycache__/`, and temporary run outputs.
- Do not create new source files by default. New files require an explicit user request or a pre-approved design that names the file path and explains why existing files are insufficient.

## File Creation Rules

- Prefer editing existing files over creating new files.
- Allowed new files without extra approval: governance or docs directly requested by the current task, such as `AGENTS.md`, `README.md`, or a Superpowers design record.
- Do not create new test files. This project validates routine changes through CLI argument combinations, smoke commands, artifact checks, manifest inspection, or governance drift checks unless the user explicitly approves test-code additions.
- Do not create vague names such as `temp`, `new`, `final`, `misc`, `helper`, or versioned scratch names.

## Superpowers, OMX, And Local Skills

Use this fixed routing:

- `superpowers:using-superpowers` selects applicable skills first.
- `superpowers:brainstorming` is mandatory before creative, behavioral, architectural, method, CLI, manifest, artifact, or research-boundary changes. Its hard gate is stronger than ordinary autonomous-execution pressure.
- Repo-local skills under `.agents/skills/` own narrow work after the design and edit zones are clear: `$implementer`, `$test_guardian`, `$reviewer`, `$experiment_runner`, and `$analysis_writer`.
- OMX `$ralph` and `$team` may orchestrate approved work, but they must not bypass Superpowers gates, local role boundaries, or project research protocols.
- Superpowers TDD is adapted to this project policy: define the expected CLI command, failure mode, artifact contract, or manifest field before implementation, but do not create new test files unless the user approves.

## Validation Protocol

- Use CLI args to validate behavior instead of adding tests by default.
- For parser/config changes, run a lightweight argument surface check such as `python main.py --help` when the active mainline imports are available.
- For stage behavior, prefer bounded smoke commands with explicit `--stage`, `--seeds`, `--disable_wandb`, and small limits when available.
- For research artifact changes, verify manifest fields and output paths produced by the CLI command.
- For governance or skill changes, run drift checks for stale `LLMbot/baseline/` default-mainline wording, role-boundary language, and skill frontmatter.
- If a CLI run is unsafe, too expensive, or missing data/GPU, state the exact command that would validate it and explain why it was not run.

## Code Quality Rules

- Keep changes small, reviewable, and local to the requested behavior.
- Preserve existing public CLI flags and artifact contracts unless the user explicitly approves a migration.
- Reuse existing helpers and patterns before adding abstractions.
- Use clear, domain-specific names. Avoid generic names like `manager`, `processor`, `handler`, `util2`, or `new_*`.
- Do not hide research-boundary issues behind implementation details. If the code is only an ablation, smoke, engineering proxy, or `*-style` implementation, name it that way in code, manifests, docs, and final reports.
- Do not hand-edit evidence artifacts; produce them through documented commands.

## Research Boundary Gate

- Official reproduction claims require local inspection of the referenced paper and code implementation.
- If official code or paper details are unavailable, stop and ask the user whether to:
  - provide a local clone/zip or allow retrieval,
  - proceed as `*-style` / `*-inspired` ablation only,
  - or defer the implementation.
- Do not silently implement a `*-style` fallback and rely only on `official_code_verified=false`.
- Any unverified reference implementation must be reflected consistently in README, manifests, reports, acceptance gates, and final response.

## Final Response Requirements

- List changed files.
- State whether any new source or test files were created.
- Report the CLI args, manifest checks, or governance commands used for validation.
- State remaining risks, especially research-claim or external-code-verification gaps.
