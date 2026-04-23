# Result Review Checklist

Use this checklist before writing conclusions, comparison tables, paper text, durable wiki claims, or claim-facing summaries.

## Manifest Gate

- [ ] A manifest exists for every run being interpreted.
- [ ] The manifest uses the required fields from `docs/research/experiment_manifest_schema.md`.
- [ ] `experiment_id`, `commit_hash`, `task_name`, `command`, and `seed_list` are recorded.
- [ ] Dataset identity is recorded through `dataset_name`, `dataset_version`, and either `split_id` or `split_description`.
- [ ] `preprocessing_version`, `model_name`, `model_version_or_date`, and `config_path` are recorded.
- [ ] `prompt_or_template_version` is recorded, or explicitly `none`.
- [ ] `hardware_or_runtime_notes`, `metrics_output_path`, `artifact_output_path`, `baseline_reference`, and `notes` are recorded.

## Artifact Support Gate

- [ ] `metrics_output_path` exists and contains the metrics being discussed.
- [ ] `artifact_output_path` exists and contains enough outputs, traces, logs, checkpoints, or diagnostics to audit the run.
- [ ] Conclusions are directly supported by recorded artifacts, not by memory or screenshots alone.
- [ ] Primary metrics are separated from diagnostics, slices, and qualitative observations.
- [ ] Failed, partial, interrupted, or exploratory runs are represented in the review.

## Baseline Comparison Gate

- [ ] `baseline_reference` is not empty.
- [ ] Baseline status is one of: reproduced, paper-reported, attempted-failed, not-applicable.
- [ ] Compared methods use compatible dataset versions, splits, metrics, and selection rules.
- [ ] Formal baseline comparisons follow `docs/protocols/baseline_comparability.md`.
- [ ] If the baseline is not comparable, the conclusion is limited to non-comparison language.

## Ablation Status Gate

- [ ] The review states whether ablations are complete, partial, missing, or not applicable.
- [ ] Any mechanism or component claim identifies the relevant ablation evidence.
- [ ] Ablations keep dataset, split, seed policy, and metrics aligned with the parent run unless justified.
- [ ] Missing ablations downgrade mechanism claims to hypotheses or remove them.

## Rerun and Stability Gate

- [ ] The review states whether reruns or multi-seed stability checks are complete, partial, missing, or not applicable.
- [ ] Stability, robustness, or publication-grade claims have rerun or multi-seed evidence.
- [ ] Single-seed results are labeled as single-seed or exploratory.
- [ ] Failed or divergent reruns are reported rather than omitted.

## Unsupported-Claim Rejection Rules

Reject or downgrade a claim when any rule applies:

- No manifest exists for the result.
- Required manifest fields are missing.
- `metrics_output_path` or `artifact_output_path` is missing or unauditable.
- The conclusion is not directly supported by recorded artifacts.
- A comparison claim lacks a comparable `baseline_reference`.
- A mechanism or component claim lacks ablation evidence.
- A stability or robustness claim lacks rerun or multi-seed evidence.
- The claim generalizes beyond the recorded dataset, split, seed policy, model version, prompt/template version, or preprocessing version.
- The text hides failed, partial, interrupted, or exploratory runs that affect interpretation.

## Verdict

Use exactly one verdict:

- `supported`: all relevant gates pass and the claim wording matches the artifacts.
- `partial`: evidence is relevant but incomplete, narrow, single-seed, or caveated.
- `not_supported`: evidence is missing, contradictory, incomparable, or too weak for the claim.
- `pending`: evidence exists but has not been reviewed.

## Review Output

A completed review should report:

- Verdict.
- Manifest path or experiment id.
- Metrics path and artifact path.
- Baseline comparison status.
- Ablation status.
- Rerun or stability status.
- Unsupported or downgraded claims.
- Required next handoff role, if any.
