# Planning Rules

Use a written execution plan before code changes when any condition below is true:

- The task touches two or more code subsystems.
- The task changes a CLI, config schema, manifest, artifact path, or public output format.
- The task changes evaluation logic, metrics, split handling, or research-claim boundaries.
- The task adds or removes dependencies.
- The task changes `NLPCC/` manuscript structure, artifact inventory, evidence mapping, or paper-task boundaries.
- The task changes the boundary between `NLPCC/` paper claims and `LLMbot/` experiment/code evidence.
- The task touches `LLMbot/baseline/`, `LLMbot/code/`, `botbr/`, `HyperScan/`, or `SEBot/`.
- The task requires implementation, experiment execution, analysis, and review handoffs.
- The task could invalidate previous experiment evidence or paper-facing conclusions.
- The task is being handed from `$ralplan` into `$ralph`, `$team`, or another long-running OMX execution workflow.

## Minimum Plan

A valid plan must state:

- Goal: the concrete outcome.
- Edit zones: files or directories that may change.
- Out of scope: files, artifacts, or claims that must not change.
- Risks: protocol, reproducibility, or compatibility risks.
- Validation: commands, tests, manifests, artifact checks, or review checks required.
- Handoff: which role skill owns the next step.

## Plan Discipline

- Keep plans short and decision-complete.
- Do not use a plan to justify broader edits than requested.
- Update the plan if the task scope changes.
- Do not interpret experiment results until the required metadata and metrics paths exist.
- Treat OMX workflows as orchestration aids. Repo-local role skills still own their narrow research responsibilities and must not collapse implementation, experiment execution, analysis, and review into one job.
- Treat Superpowers brainstorming as the design gate for creative or behavioral work. A plan does not bypass that gate.
- Do not add new test files unless the user explicitly approves them.

## Current Active Mainline Governance

- `LLMbot/` is the active implementation mainline.
- `NLPCC/` is the active NLPCC paper/task line.
- `LLMbot/baseline/` and `LLMbot/code/` are deprecated legacy surfaces scheduled for deletion.
- New plans must not use `LLMbot/baseline/core` as an edit zone unless the task explicitly asks for migration, deletion, archival cleanup, or forensic comparison.
- Routine validation should prefer CLI-argument checks, smoke commands, manifest inspection, artifact existence checks, and governance drift checks over new test-code creation.
- Historical plans that referenced `LLMbot/baseline/core` are not active unless a user explicitly reactivates them with a migration or forensic-comparison scope.
