# Planning Rules

Use a written execution plan before code changes when any condition below is true:

- The task touches two or more code subsystems.
- The task changes a CLI, config schema, manifest, artifact path, or public output format.
- The task changes evaluation logic, metrics, split handling, or research-claim boundaries.
- The task adds or removes dependencies.
- The task touches `botbr/`, `HyperScan/`, or `SEBot/`.
- The task requires implementation, experiment execution, analysis, and review handoffs.
- The task could invalidate previous experiment evidence or paper-facing conclusions.

## Minimum Plan

A valid plan must state:

- Goal: the concrete outcome.
- Edit zones: files or directories that may change.
- Out of scope: files, artifacts, or claims that must not change.
- Risks: protocol, reproducibility, or compatibility risks.
- Validation: commands, tests, manifests, or review checks required.
- Handoff: which role skill owns the next step.

## Plan Discipline

- Keep plans short and decision-complete.
- Do not use a plan to justify broader edits than requested.
- Update the plan if the task scope changes.
- Do not interpret experiment results until the required metadata and metrics paths exist.
