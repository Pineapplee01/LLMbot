# LLMbot Active Mainline

`LLMbot/` is the only active mainline for current bot-detection pipeline work.

## Current Entry Point

Run commands from this directory:

```bash
cd LLMbot
python main.py --experiment_task legacy_distill --dataset TwiBot-20 --seeds 1 --disable_wandb
```

Canonical public flags:

- `--experiment_task`
- `--graph_backbone`
- `--text_encoder`
- `--semantic_encoder`
- `--embedding_path`

Legacy-compatible aliases still parse:

- `--stage`
- `--GNN_model`
- `--LM_model`
- `--semantic_backbone`
- `--emb_path`
- `--g0_feature_path`

Use the canonical flags in new commands, docs, manifests, and notes.

## Public CLI Surface

The parser currently exposes these public task names:

- `legacy_distill`
- `semantic_finetune`
- `frozen_g0`
- `frozen_gats`
- `local_conformal_prune_diag`
- `glance_joint_router_refine`
- `vertical_minimal`
- `estimator_matrix`
- `semantic_matrix`
- `semantic_source_matrix`
- `repair_matrix`
- `selector_matrix`
- `positioning_matrix`
- `backbone_stress`
- `appendix`

`glance_joint_router_refine` is the only public GLANCE stage in this workspace. It is the mainline GLANCE implementation here, aligned to the paper text at the router/refiner training-contract level while still using the current cached semantic embedding path rather than the paper's prompt-serialized LLM stack.

## Supported Command Examples

```bash
# Legacy distillation without GNN
python main.py \
  --experiment_task legacy_distill \
  --dataset TwiBot-20 \
  --seeds 1 \
  --disable_wandb

# Legacy distillation with GNN
python main.py \
  --experiment_task legacy_distill \
  --dataset TwiBot-20 \
  --use_GNN \
  --graph_backbone botrgcn \
  --seeds 1 \
  --disable_wandb

# Semantic encoder finetune
python main.py \
  --experiment_task semantic_finetune \
  --dataset TwiBot-20 \
  --semantic_encoder roberta_finetuned \
  --seeds 1 \
  --disable_wandb

# Frozen graph detector artifact
python main.py \
  --experiment_task frozen_g0 \
  --dataset TwiBot-20 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path datasets/TwiBot-20/finetuned_roberta_embeddings_iter_2_seed1.pt \
  --seeds 1 \
  --disable_wandb

# Local conformal prune diagnostic
python main.py \
  --experiment_task local_conformal_prune_diag \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path datasets/TwiBot-20/finetuned_roberta_embeddings_iter_2_seed1.pt \
  --seeds 1 \
  --disable_wandb

# Strict GLANCE-style joint router/refiner baseline
python main.py \
  --experiment_task glance_joint_router_refine \
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

- `main.py` - CLI entrypoint and task dispatch
- `parser_args.py` - argument source of truth
- `trainer.py` - stage runner and training/evaluation orchestration
- `estimators.py`, `operators.py`, `model_building.py` - estimator/operator/model construction
- `subgroups.py`, `GNNs.py`, `RGT.py`, `LM.py`, `dataloader.py` - support modules
- `utils/` - public utility package for artifact IO, metrics, calibration, data loading, and reproducibility helpers

## Current Risks To Keep In Mind

- Some historical names such as `frozen_g0` remain public stage values even
  though the CLI flag names were canonicalized.
- `trainer.py` and `estimators.py` are still large and should be treated as
  refactor hotspots rather than extension points for more ad hoc logic.

## Deprecated Surfaces

`baseline/` and `code/` are deprecated legacy directories scheduled for
deletion. Do not add implementation, tests, or documentation there unless the
task explicitly asks for migration, deletion, archival cleanup, or forensic
comparison.
