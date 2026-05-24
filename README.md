# LLMbot Active Mainline

`LLMbot/` is the only active mainline for current bot-detection pipeline work.

## Entry Point

Run commands from this directory:

```bash
cd LLMbot
python main.py --experiment_task distillation_pipeline --dataset TwiBot-20 --seeds 1 --disable_wandb
```

Preferred public flags:

- `--experiment_task`
- `--graph_backbone`
- `--text_encoder`
- `--semantic_encoder`
- `--embedding_path`

Hidden compatibility aliases still parse for one migration window:

- `--stage`
- `--GNN_model`
- `--LM_model`
- `--semantic_backbone`
- `--emb_path`
- `--g0_feature_path`

Use canonical flags in new commands, docs, manifests, and analysis notes.

## Public Task Surface

The parser currently exposes these canonical public tasks:

- `distillation_pipeline`
- `semantic_encoder_finetune`
- `graph_detector_prepare`
- `graph_calibration_prepare`
- `local_conformal_diagnostic`
- `local_conflict_prune_diag`
- `joint_router_refinement`
- `minimal_pipeline`
- `estimator_ablation`
- `semantic_operator_ablation`
- `semantic_source_ablation`
- `repair_operator_ablation`
- `selector_ablation`
- `positioning_ablation`
- `backbone_stress_test`
- `appendix_ablation`

Internal-only implemented branches are not part of the public CLI contract:

- `glance_oracle_refinement_internal`
- `glance_full_graph_refinement_internal`
- `glance_counterfactual_router_internal`
- `glance_budgeted_refinement_internal`
- `glance_refiner_analysis_internal`
- `phase_a_single_cell_internal`

`joint_router_refinement` is the only public GLANCE-family task. It is still a
GLANCE-style implementation under the current cached semantic-embedding path,
not a full official-pipeline reproduction. The strict router path now mirrors
the paper more closely by fitting a lightweight auxiliary MLP `Q` on node
features and using its predicted probabilities to build the soft local
homophily feature for routing.

## Command Examples

```bash
# Distillation without graph execution
python main.py \
  --experiment_task distillation_pipeline \
  --dataset TwiBot-20 \
  --seeds 1 \
  --disable_wandb

# Distillation with graph execution
python main.py \
  --experiment_task distillation_pipeline \
  --dataset TwiBot-20 \
  --use_GNN \
  --graph_backbone botrgcn \
  --seeds 1 \
  --disable_wandb

# Semantic encoder finetune
python main.py \
  --experiment_task semantic_encoder_finetune \
  --dataset TwiBot-20 \
  --semantic_encoder roberta_finetuned \
  --seeds 1 \
  --disable_wandb

# Graph detector preparation artifact
python main.py \
  --experiment_task graph_detector_prepare \
  --dataset TwiBot-20 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path datasets/TwiBot-20/finetuned_roberta_embeddings_iter_2_seed1.pt \
  --seeds 1 \
  --disable_wandb

# Graph calibrator preparation artifact
python main.py \
  --experiment_task graph_calibration_prepare \
  --dataset TwiBot-20 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path datasets/TwiBot-20/finetuned_roberta_embeddings_iter_2_seed1.pt \
  --seeds 1 \
  --disable_wandb

# Local conformal diagnostic
python main.py \
  --experiment_task local_conformal_diagnostic \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path datasets/TwiBot-20/finetuned_roberta_embeddings_iter_2_seed1.pt \
  --seeds 1 \
  --disable_wandb

# Local conflict diagnostic
python main.py \
  --experiment_task local_conflict_prune_diag \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path datasets/TwiBot-20/finetuned_roberta_embeddings_iter_2_seed1.pt \
  --conflict_router_budget 0.10 \
  --conflict_topk_per_bucket 1 \
  --seeds 1 \
  --disable_wandb

# Public GLANCE-style router/refiner task
python main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path datasets/TwiBot-20/finetuned_roberta_embeddings_iter_2_seed1.pt \
  --seeds 1 \
  --disable_wandb
```

## Mainline Layout

Root-level modules are the default implementation surface:

- `main.py` - CLI entrypoint and canonical task dispatch
- `parser_args.py` - argument contract and compatibility normalization
- `stage_registry.py` - stage visibility and naming source of truth
- `trainer.py` - thin compatibility facade
- `trainer_legacy_impl.py` - current large implementation body during the migration window
- `stage_runner.py`, `trainer_preparation.py`, `trainer_semantic.py`, `trainer_graph.py`, `trainer_glance.py`, `stage_helpers.py` - new canonical module surfaces for continued extraction
- `estimators.py`, `operators.py`, `model_building.py` - estimator/operator/model construction
- `utils/` - artifact IO, manifests, metrics, calibration, data loading, and reproducibility helpers

## Artifact Naming

New writes use canonical namespaces:

- `seed_<n>/preparation/semantic_encoder`
- `seed_<n>/preparation/graph_detector`
- `seed_<n>/preparation/graph_calibrator`
- `seed_<n>/stages/<canonical_task_name>`

Legacy artifact locations such as `frozen/g0`, `frozen/gates/gats`, and old
stage names remain readable for compatibility, but they are no longer the
active naming surface.

## Current Mainline Notes

- `trainer.py` is already a thin compatibility facade, but most execution logic
  still lives in `trainer_legacy_impl.py`.
- `stage_runner.py`, `trainer_preparation.py`, `trainer_semantic.py`,
  `trainer_graph.py`, `trainer_glance.py`, and `stage_helpers.py` already act
  as canonical import surfaces, but most still forward into
  `trainer_legacy_impl.py` while extraction continues.
- `estimators.py` and `trainer_legacy_impl.py` remain the main refactor
  hotspots.
- `python main.py --help` is now intended to work as a parser-only check even if
  the training runtime is not fully installed.

## Refactor Snapshot 2026-05-24

- public task naming is now canonical and owned by `stage_registry.py`
- parser output includes canonical fields first and legacy shadow fields only
  for compatibility
- `main.py` resolves stage behavior through the registry instead of maintaining
  a separate public-task truth table
- preparation artifacts now write to canonical namespaces under
  `seed_<n>/preparation/`
- runtime-only helper artifacts are still in migration and may retain
  compatibility fallbacks
- strict GLANCE now derives its router-side soft local homophily signal from a
  lightweight auxiliary MLP `Q`, replacing the earlier logistic-regression
  placeholder
- LOGIN-style uncertainty routing remains explicitly bounded to hard-node
  selection only; manifests record `official_code_verified = false`,
  `repo_locally_verified = false`, and
  `verified_scope = node_selection_uncertainty_only`
- `joint_router_refinement` remains the only public GLANCE-family task; richer
  GLANCE branches are implemented but internal-only

## Code Governance Status 2026-05-24

The active mainline is now governance-aligned, but not yet structurally
finished.

Completed governance work:

- `LLMbot/` is the only active mainline
- canonical public tasks and canonical public flags are now the operator-facing
  contract
- hidden legacy aliases are isolated to a compatibility window instead of
  remaining the public interface
- internal GLANCE branches are explicitly separated from the public CLI surface
- the active code-development docs now describe the canonical mainline rather
  than deprecated `baseline/core` terminology

Still incomplete:

- most execution logic still resides in `trainer_legacy_impl.py`
- several extracted modules are currently boundary surfaces rather than true
  logic owners
- active internal code still carries a canonical-to-legacy compatibility layer
- `StageSpec` is only partially consumed as a runtime policy source
- runtime-only helper artifact naming is still in migration

This means the project has passed the public-contract cleanup phase, but has
not yet finished the implementation extraction phase.

## Next Refactor Plan

1. Make `trainer_preparation.py` the real owner of preparation logic.
   Move `load_frozen_g0`, `build_or_load_frozen_g0`, and
   `build_or_load_faithful_gats` out of `trainer_legacy_impl.py`, and move
   shared path/provenance helpers into `stage_helpers.py`.
2. Make `trainer_semantic.py` the real owner of semantic finetune execution.
   Move `run_semantic_finetune_seed` and its manifest/report helpers out of
   `trainer_legacy_impl.py`.
3. Finish the active parser-namespace migration inside code.
   Keep legacy flags parse-compatible, but make active mainline code read
   canonical fields such as `experiment_task`, `graph_backbone`,
   `text_encoder`, `semantic_encoder`, and `embedding_path`.
4. Consume `StageSpec` more uniformly at runtime.
   Replace remaining hand-written dispatch or gate special cases in `main.py`
   with registry fields such as `runner_kind`, `claim_grade_allowed`, and
   `forces_use_gnn`.
5. Extract graph and GLANCE execution ownership into
   `trainer_graph.py`, `trainer_glance.py`, and `stage_runner.py`.
   The goal is for those modules to own their execution branches directly,
   rather than re-exporting `trainer_legacy_impl.py`.
6. Canonicalize runtime-only helper artifacts after extraction.
   New helper outputs should stop writing migration-era naming where practical,
   while read compatibility remains in place for one transition window.

## Documentation Sync Rule

Every future code change in `LLMbot/` must update the matching code-development
docs in the same task.

Minimum mapping:

- parser or CLI changes: update `docs/code/parser.md` and this file
- dispatch, module-boundary, or extraction changes: update
  `docs/ARCHITECTURE.md` and `docs/code/research.md`
- maintainability or unresolved-risk changes: update `code.md`

Do not treat documentation sync as optional cleanup.

## Deprecated Surfaces

`baseline/` and `code/` are deprecated legacy directories scheduled for
deletion. Do not add implementation, tests, or documentation there unless the
task explicitly asks for migration, deletion, archival cleanup, or forensic
comparison.
