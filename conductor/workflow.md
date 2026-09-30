# LLMbot Governance Workflow

Date: 2026-07-04

## Default Workflow

1. Read `AGENTS.md`, `LLMbot/AGENTS.md`, and this `conductor/` context before
   changing active-mainline structure, docs, CLI, manifest, or artifact
   wording.
2. Use `UBIQUITOUS_LANGUAGE.md` for canonical terms.
3. Use CodeGraph first for cross-file relationship analysis when changing or
   documenting code boundaries.
4. Keep active source under the flat `LLMbot/code/` directory. Do not introduce
   `stages/`, `methods/`, or similar package splits until a later approved
   cleanup slice.
5. Prefer small governance/documentation changes before large implementation
   moves.
6. Verify the edit stayed inside the approved scope before reporting completion.

## First-Stage Edit Zone

Allowed in the first governance stage:

- `conductor/product.md`
- `conductor/tech-stack.md`
- `conductor/workflow.md`
- `conductor/tracks.md`
- `UBIQUITOUS_LANGUAGE.md`
- `docs/ARCHITECTURE.md`
- `docs/code/parser.md`
- `docs/code/research.md`
- `code.md`
- minimal `AGENTS.md` entrypoint alignment when needed

Out of scope in the first governance stage:

- splitting `trainer_legacy_impl.py`, `precompute.py`, `router.py`, or
  `estimators.py`
- introducing `stages/`, `methods/`, or another package taxonomy under
  `LLMbot/code/`
- adding temporary scripts, scratch files, test files, or experiment files
- modifying `LLMbot/experiments/` or `LLMbot/server_logs/`
- changing CLI behavior, manifest schema, model behavior, or evaluation logic

## Documentation Sync

When code behavior changes in later stages:

- parser, public flags, task names, or examples update `docs/code/parser.md`
  and `LLMbot/README.md`
- runtime flow, module boundaries, stage ownership, or extraction status update
  `docs/ARCHITECTURE.md` and `docs/code/research.md`
- maintainability risk, unresolved debt, or migration status update `code.md`

Documentation sync is part of completion, not later cleanup.

## Naming Rules

- New public documentation and commands use canonical names only.
- Legacy names may appear only in compatibility sections, migration notes, or
  alias maps.
- Write new module and document names in lowercase `snake_case` unless the repo
  has a fixed conventional name such as `AGENTS.md`, `README.md`, `SKILL.md`,
  or `UBIQUITOUS_LANGUAGE.md`.
- Do not create vague names such as `temp`, `new`, `final`, `misc`, or
  `helper` unless required by an external interface.

## Verification

For governance-only changes, use these checks:

- `git diff --name-only` to confirm only approved governance files changed.
- `git diff --check` to catch whitespace errors.
- a file-existence check for all referenced active-mainline module names.
- a path-scope check confirming no generated artifacts under
  `LLMbot/experiments/**` or `LLMbot/server_logs/**` changed.
- for source-location work, confirm root `LLMbot/main.py`,
  `LLMbot/precompute.py`, and `LLMbot/preprocess.py` remain compatibility
  entrypoints and active implementation files are under `LLMbot/code/`.

Do not run training or experiment commands for governance-only changes.
