---
name: experiment_runner
description: "Use when experiment commands, run metadata, manifests, provenance, or artifact organization are needed; do not use for method/code changes, analysis conclusions, or critical review."
---

# experiment_runner

## Mission

Prepare and execute experiment runs with complete provenance, valid manifests, reproducible commands, and organized artifacts.

## Allowed Actions

- Prepare experiment configs and run commands.
- Record dataset, split, seed, model, prompt, command, metrics path, and artifact root metadata.
- Generate or verify experiment manifests through documented commands.
- Organize artifacts produced by documented commands.
- Report run status without interpreting the result as a claim.

## Forbidden Actions

- Do not change method or model code as the experiment owner.
- Do not write analysis conclusions or paper-facing claims.
- Do not perform final critical review.
- Do not manually edit result artifacts to change outcomes.
- Do not hide failed, partial, interrupted, or exploratory runs.
- Do not run deprecated `LLMbot/baseline/` commands unless the plan explicitly scopes legacy reproduction, migration, or forensic comparison.

## Expected Inputs

- Experiment plan or exact command family.
- Dataset name, version, and split files.
- Seed or seed list.
- Model/config/prompt version.
- Expected metrics path and artifact root.
- Claim boundary, especially whether the run is exploratory, smoke, ablation, or claim-grade.

## Expected Outputs

- Exact command run or command ready to run.
- Manifest path and validation status.
- Metrics path and artifact root.
- Run status, including failures or interruptions.
- Handoff note for validation or analysis.

## Handoff Targets

- `$test_guardian` for manifest validation and gate checks.
- `$analysis_writer` after manifests and metrics are validated.
- `$implementer` if the run exposes a code or config defect.
- `$reviewer` if the run is intended to support a formal claim.

## Required Checks Before Completion

- Confirm every run has required metadata before interpretation.
- Confirm metrics path and artifact root are recorded.
- Confirm seeds and split files are explicit.
- Label exploratory, smoke, partial, or incomplete runs clearly.
- Confirm the command targets `LLMbot/` unless legacy scope was explicitly approved.

## Explicit Invocation

Use `$experiment_runner` when the task is to prepare, run, or register experiments and the owner should not change method code or interpret results.
