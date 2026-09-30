# Model Guide - LLMbot Active Mainline

Updated: 2026-05-23
Scope: current active pipeline in `LLMbot/`

## Overview

The active model/runtime surface is the root `LLMbot/` mainline.

- `main.py` owns task dispatch and seed iteration
- `parser_args.py` owns the public CLI contract
- `trainer.py` owns frozen-artifact preparation and structured stage execution
- `model_building.py` and `estimators.py` supply the backbone and post-hoc model logic

## Canonical CLI Vocabulary

Preferred flags for new commands:

- `--experiment_task`
- `--graph_backbone`
- `--text_encoder`
- `--semantic_encoder`
- `--embedding_path`

Legacy aliases such as `--stage`, `--GNN_model`, `--LM_model`, and `--emb_path`
still parse, but they are compatibility paths only.

## Public Task Families

### Onboarding and preparation

- `legacy_distill`
- `semantic_finetune`
- `frozen_g0`
- `frozen_gats`

### Public structured tasks

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

## Internal Implemented Branches

The codebase still contains non-public GLANCE-related handlers inside
`trainer.py`:

- `glance_oracle_refine`
- `glance_full_graph_refine`
- `glance_counterfactual_router`
- `glance_budgeted_refine`
- `glance_refiner_analysis`

They are currently implementation branches, not public parser-exposed commands.

## Recommended Commands

Run from `LLMbot/`.

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

# Strict GLANCE-style joint baseline
python main.py \
  --experiment_task glance_joint_router_refine \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path datasets/TwiBot-20/embeddings_iter_-1_seed_1.pt \
  --seeds 1 \
  --disable_wandb
```

## Current Structural Caveat

The code contract is stronger than the documentation contract right now: many
artifact and manifest checks exist in `trainer.py`, but the public docs must be
kept aligned with `parser_args.py` or users will be routed toward commands that
the parser rejects.
