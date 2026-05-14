# LLMbot Mainline

`LLMbot/` is the active mainline for the bot detection research pipeline.

## Current Entry Point

Run commands from this directory:

```bash
cd LLMbot
python main.py --stage legacy_distill --dataset TwiBot-20 --seeds 1 --disable_wandb
```

Use `--seeds`, not `--seed`. Add `--use_GNN` for graph-backed runs:

```bash
python main.py \
  --stage legacy_distill \
  --dataset TwiBot-20 \
  --use_GNN \
  --GNN_model botrgcn \
  --seeds 1 \
  --disable_wandb
```

## Mainline Layout

Root-level modules are the default implementation surface:

- `main.py` - CLI entrypoint and stage dispatch.
- `parser_args.py` - argument source of truth.
- `trainer.py` - stage runner and training/evaluation orchestration.
- `estimators.py`, `operators.py`, `model_building.py` - estimator/operator/model construction.
- `subgroups.py`, `GNNs.py`, `RGT.py`, `LM.py`, `dataloader.py` - pipeline support modules.
- `utils/` - public utility package for data loading, artifact IO, checkpoint loading, metrics, losses, calibration helpers, and reproducibility setup.
- `models/`, `trainers/`, `evaluation/`, `data/` - package surfaces used only when the task names or imports them.

## Deprecated Surfaces

`baseline/` and `code/` are deprecated legacy directories and are scheduled for deletion. Do not add implementation, tests, or documentation there unless the task explicitly asks for migration, deletion, archival cleanup, or forensic comparison.

## Validation Style

This project prefers CLI-argument validation over adding new test code.

Use bounded smoke commands with explicit args:

```bash
python main.py --help
python main.py --stage legacy_distill --dataset TwiBot-20 --seeds 1 --disable_wandb
python main.py --stage estimator_matrix --dataset TwiBot-20 --seeds 1 --estimator_mode login_uncertainty_router --disable_wandb
```

If a command needs unavailable data, GPU, remote artifacts, or excessive runtime, record the exact command and why it was not run.

## Agent Rules

Read `AGENTS.md` before making changes. The important defaults are:

- `LLMbot/` root is the only active mainline.
- `baseline/` and `code/` are deprecated and default read-only.
- Do not create new source files unless explicitly approved.
- Do not create new test files unless explicitly approved.
- Use project-local skills for implementation, validation, review, experiment running, and analysis writing.
- Use Superpowers for process discipline, especially brainstorming and verification.
- Official paper or open-source reproduction claims require local verification of the referenced implementation.

## Research Boundary

Any unverified external reference implementation must be labeled as `*-style` or `*-inspired` ablation only. It must not be presented as official reproduction, full loop implementation, SOTA evidence, or paper-ready claim until the referenced code has been inspected locally and the validation artifacts support that claim.
