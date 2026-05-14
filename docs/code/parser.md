# LLMbot CLI and Mainline Development Contract

This note documents the active `LLMbot/` root mainline after the top-level migration.
It is an engineering contract, not experiment evidence and not a research-claim record.

## Active Code Boundary

- Active implementation lives in `LLMbot/` root modules.
- `LLMbot/baseline/` and `LLMbot/code/` are deprecated legacy/reference surfaces. New mainline code must not import from them.
- New source files and new test files are not created by default. Prefer editing existing root modules.
- Shared artifact/checkpoint helpers belong in the `LLMbot/utils/` package public API. There is no root-level `LLMbot/utils.py` source of truth.
- Model construction, model adapters, input feature resolution, and selector construction belong in `LLMbot/model_building.py`.
- Stage orchestration, frozen artifact preparation, and preparation-stage gate logic belong in `LLMbot/trainer.py`.

## Current Stage Entrypoints

`--stage` currently accepts:

- Preparation stages: `frozen_g0`, `frozen_gats`, `semantic_finetune`
- Compatibility stage: `legacy_distill`
- Formal staged surfaces: `vertical_minimal`, `estimator_matrix`, `semantic_matrix`, `semantic_source_matrix`, `repair_matrix`, `selector_matrix`, `positioning_matrix`, `backbone_stress`, `appendix`
- EQC v8 surface: `eqc_v8_matrix`

Preparation and formal stages require canonical splits (`--reset_split -1`).
`main.py` formats stage results, but stage execution returns structured dictionaries.

## Stage Result Contract

Top-level stage results should expose:

- `stage`: execution stage name
- `seed`: integer seed
- `status`: execution status, usually `completed`
- `artifact_dir`: primary artifact directory when available

Nested stage-specific results may include additional fields such as `metrics`, `best_mode`, `cell_id`, or `eqc_v8_block`.

Frozen G0 artifacts return:

- `manifest`
- `outputs`
- `selection_metrics`
- `checkpoint_path`
- `artifact_dir`

The compatibility key `dir` is retained temporarily for existing callers.

## Parser Normalization

`parser_args.normalize_args(args)` owns compatibility mapping after argparse parsing.

Recommended parameters:

- `--emb_path`: cached semantic embedding tensor for Phase A / frozen G0 graph detection
- `--risk_budgets`: comma-separated risk budgets for router and risk-ranking surfaces
- `--risk_variant`: residual-risk selector feature family

Deprecated compatibility aliases:

- `--g0_feature_path`: alias for `--emb_path`
- `--router_budgets`: alias for `--risk_budgets`
- `--router_variant`: alias for `--risk_variant`

Normalization rules:

- If `--emb_path` is provided, it wins and is copied to `g0_feature_path`.
- If only `--g0_feature_path` is provided, it is copied to `emb_path`.
- If `--risk_variant` is omitted, `--router_variant` supplies it.
- If `--risk_budgets` is provided, it wins and is copied to `router_budgets`.
- If only `--router_budgets` is provided, it is copied to `risk_budgets`.

`--is_processed` uses an explicit compatibility boolean parser. Accepted values include
`true/false`, `1/0`, `yes/no`, and `on/off`.

## Research Boundary Notes

- `frozen_g0` is the SimTeG-style frozen graph detector preparation artifact, not a new research method by itself.
- `frozen_gats` is a post-hoc calibration/gate artifact used as an anchor. It is implemented in the mainline trainer and records `source_impl: trainer.GraphGATSCalibrator`.
- Official reproduction claims still require paper/code verification. A `*-style` implementation must be named and reported as such in manifests and analysis.
- The CLI contract does not make claims about social bot detection performance.

## Freeze and Removal Plan

Keep the deprecated aliases until at least one full experiment cycle has run with the normalized parameters.
After that cycle:

1. Update experiment scripts to use only recommended parameters.
2. Add a warning period for deprecated aliases.
3. Remove deprecated aliases only after manifests and docs no longer depend on them.

Do not delete compatibility keys like frozen G0 `dir` until all current stage callers use `artifact_dir`.
