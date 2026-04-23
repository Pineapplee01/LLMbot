# Experiment Manifest Schema

Experiment manifests are JSON files that record the minimum provenance required before a result can be interpreted.

Use this flat schema for new experiment records. Existing legacy manifests may keep their historical shape, but new claim-bearing runs must include these fields.

## Required Fields

| Field | Type | Required | Operational Meaning |
| --- | --- | --- | --- |
| `experiment_id` | string | yes | Stable run identifier used in docs, tables, and artifact folders. |
| `commit_hash` | string | yes | Git commit used for the run, or `uncommitted:<reason>` when unavoidable. |
| `task_name` | string | yes | Human-readable task or experiment block name. |
| `dataset_name` | string | yes | Dataset name, for example `TwiBot-20`. |
| `dataset_version` | string | yes | Dataset release, checksum, dated local version, or documented source. |
| `split_id` | string | conditional | Required unless `split_description` is provided. |
| `split_description` | string | conditional | Required unless `split_id` is provided. |
| `preprocessing_version` | string | yes | Preprocessing pipeline, feature version, or `none`. |
| `model_name` | string | yes | Method, backbone, model, or pipeline name. |
| `model_version_or_date` | string | yes | Model version, checkpoint date, config date, or dated method label. |
| `prompt_or_template_version` | string | yes | Prompt/template version when applicable; otherwise `none`. |
| `config_path` | string | yes | Path to the config file, args file, or frozen command config. |
| `seed_list` | array of integers | yes | All seeds intended for this manifest. |
| `command` | string | yes | Exact command or script invocation. |
| `hardware_or_runtime_notes` | string | yes | Device, runtime, environment, or relevant execution notes. |
| `metrics_output_path` | string | yes | Path to machine-readable metrics. |
| `artifact_output_path` | string | yes | Path to artifacts, traces, checkpoints, diagnostics, or logs. |
| `baseline_reference` | string | yes | Baseline run, paper-reported baseline, comparison protocol, or `none`. |
| `notes` | string | yes | Caveats, failures, exploratory status, or other audit notes. |

## Split Rule

At least one of these fields must be non-empty:

- `split_id`
- `split_description`

Use `split_id` for canonical or named splits. Use `split_description` when the split is ad hoc, inherited, or described by file paths instead of a stable id.

## Prompt Rule

`prompt_or_template_version` is always present.

- Use a concrete version, path, date, or hash when prompts, templates, instructions, or text formatting affect inputs.
- Use `none` only when no prompt or template is part of the experiment.

## Path Rules

- `config_path`, `metrics_output_path`, and `artifact_output_path` should be repository-relative when possible.
- Paths must point to auditable files or directories by experiment completion.
- Do not use a metrics path for artifacts or an artifact path for metrics.

## Baseline Reference Rule

`baseline_reference` must say how comparison evidence should be interpreted:

- Use an experiment id for reproduced baselines.
- Use a citation or report label for paper-reported baselines.
- Use `none` only when the run is not a comparison and no baseline is being invoked.

## Minimal Example

```json
{
  "experiment_id": "twibot20_legacy_distill_rgcn_seed_1",
  "commit_hash": "git:<commit>",
  "task_name": "legacy_distill_rgcn_smoke",
  "dataset_name": "TwiBot-20",
  "dataset_version": "canonical-local",
  "split_id": "canonical",
  "split_description": "",
  "preprocessing_version": "baseline-loader-current",
  "model_name": "LLMbot baseline RGCN",
  "model_version_or_date": "2026-04-23",
  "prompt_or_template_version": "none",
  "config_path": "LLMbot/baseline/core/parser_args.py",
  "seed_list": [1],
  "command": "python main.py --stage legacy_distill --dataset TwiBot-20 --use_GNN --GNN_model rgcn --seeds 1",
  "hardware_or_runtime_notes": "record Python, PyTorch, CUDA, GPU/CPU, and remote/local runtime here",
  "metrics_output_path": "results/example/metrics.json",
  "artifact_output_path": "results/example",
  "baseline_reference": "none",
  "notes": "Single-seed smoke manifest; not claim-complete."
}
```

## Completion Use

- `$experiment_runner` owns manifest creation and artifact organization.
- `$test_guardian` verifies required fields and path existence.
- `$analysis_writer` may use the manifest only after metrics and artifacts are present.
- `$reviewer` decides whether the manifest is sufficient for the stated claim.
