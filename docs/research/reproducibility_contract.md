# Reproducibility Contract

This contract defines when work in this repository is complete enough to support engineering handoff, experiment execution, analysis writing, or research claims.

The core rule is strict: implementation completion does not imply experiment completion, analysis completion, or claim completion.

## Completion States

| State | Owner | Required Evidence | Not Sufficient For |
| --- | --- | --- | --- |
| Implementation completion | `$implementer` | Code, config, or doc change is scoped, reviewable, and passes the narrowest relevant engineering check. | Reporting experiment results or making research claims. |
| Experiment completion | `$experiment_runner` | A run command was executed or prepared, artifacts are organized, metrics path is recorded, and a manifest contains the required metadata. | Interpreting results without validation and review. |
| Analysis completion | `$analysis_writer` | Tables, summaries, or error analysis cite validated manifests, metrics, and artifact paths. | Claiming support beyond recorded evidence. |
| Claim completion | `$reviewer` | The claim has passed manifest, artifact, comparison, ablation, rerun/stability, and unsupported-claim checks. | Broader claims, different datasets, or unstated settings. |

## Implementation Completion

Implementation is complete only when:

- The changed files stay inside the allowed task scope.
- The change is minimal and tied to the requested method, code, config, script, or doc task.
- The narrowest relevant engineering validation has passed or the reason it cannot run is recorded.
- Any changed CLI, config, manifest, artifact, or public output behavior is documented.
- No experiment result or research claim is inferred from implementation alone.

Implementation completion must hand off to `$test_guardian` for validation when behavior, gates, or regressions are involved.

## Experiment Completion

An experiment is complete only when:

- The experiment manifest exists and contains every required field in `docs/research/experiment_manifest_schema.md`.
- `metrics_output_path` points to the machine-readable metrics used later for analysis.
- `artifact_output_path` points to the artifacts, checkpoints, logs, traces, or diagnostics needed to audit the run.
- `seed_list`, `command`, `config_path`, runtime notes, and split information are recorded.
- Failed, partial, interrupted, or exploratory runs are labeled in `notes`.

Experiment completion must hand off to `$test_guardian` for manifest and artifact validation before interpretation.

## Analysis Completion

Analysis is complete only when:

- Every table, summary, or error-analysis statement cites recorded metrics or artifacts.
- Primary metrics are separated from diagnostics, slices, and qualitative observations.
- Missing baselines, missing ablations, weak seed coverage, or incomplete reruns are stated as caveats.
- Unsupported conclusions are marked as unsupported, not softened into claims.
- The analysis does not omit relevant failed or partial runs.

Analysis completion must hand off to `$reviewer` before the text is used for paper-facing claims or durable project memory.

## Claim Completion

A claim is complete only when:

- The underlying experiment manifest is complete.
- Recorded metrics and artifacts directly support the claim wording.
- Baseline comparison status is known and appropriate for the claim.
- Ablation status is known when the claim depends on a component, mechanism, or design choice.
- Rerun or stability status is known when the claim implies robustness, repeatability, or publication-grade evidence.
- The result review checklist returns `supported`.

Claims that receive `partial`, `not_supported`, or `pending` must not be written as established conclusions.

## Required Handoff Order

Use these handoffs unless the task explicitly has narrower scope:

```text
$implementer -> $test_guardian -> $experiment_runner -> $test_guardian -> $analysis_writer -> $reviewer
```

For docs-only or planning-only work, skip roles that do not own the task, but keep their boundaries intact.

## Operational Rules

- Do not interpret metrics before experiment metadata exists.
- Do not write conclusions before artifacts and metrics are recorded.
- Do not treat exploratory or single-seed runs as publication-grade evidence.
- Do not compare methods unless baseline status, split identity, and metric definitions are recorded.
- Do not claim mechanisms unless ablation evidence is present or the claim is explicitly labeled as hypothesis.
- Do not claim stability unless rerun or multi-seed evidence is recorded.
