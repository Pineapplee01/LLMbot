# Remote GPU Server Guide

This guide records the validated access path and safe default workflow for the shared GPU server used by the current `LLMbot/baseline/` mainline.

## Validated Server State

- SSH endpoint: `ssh root@172.31.106.108 -p 10011`
- Optional local alias: `Host GPU` in `C:\Users\p\.ssh\config`
- OS: CentOS Linux 7
- Hostname: `bd11691db314`
- GPUs: `2 x NVIDIA GeForce RTX 3090 (24 GB)`
- Driver / CUDA reported by `nvidia-smi`: `535.129.03 / 12.2`
- Conda root: `/root/mambaforge`
- Validated Conda environments: `base`, `Qwen`, `lmbot`
- Remote workspace root: `/root/workspace/LMbot`

## Safety Notes

- The remote repo is currently dirty. Do not run `git pull`, `git reset`, or overwrite existing experiment artifacts unless the task explicitly requires it.
- Preserve existing result directories such as `TwiBot-20_seed_*`, `TwiBot-20_RGT_seed_*`, logs, and archived tarballs.
- Prefer the current default execution surface in `/root/workspace/LMbot/LLMbot/baseline/core`.
- Treat root-level legacy scripts as historical reproduction helpers, not the default path for new work.

## Connect And Activate The Correct Environment

Do not rely on the default login shell Python. Activate `lmbot` explicitly before running any baseline commands.

```bash
ssh root@172.31.106.108 -p 10011
source /root/mambaforge/etc/profile.d/conda.sh
conda activate lmbot
```

Validated package versions in `lmbot`:

- `torch 2.0.1+cu117`
- `torch-geometric 2.1.0`
- `transformers 4.29.0`
- `scikit-learn 1.6.1`
- `wandb 0.14.0`

## Default Working Directory

For current mainline work, start from:

```bash
cd /root/workspace/LMbot/LLMbot/baseline/core
```

The dataset is already available at:

```text
/root/workspace/LMbot/datasets/TwiBot-20
```

On the remote machine, `LLMbot/baseline/core/datasets` is a symlink to `/root/workspace/LMbot/datasets`, so the baseline loader can resolve the dataset directly from the current mainline path.

## Smoke Test

Run this before a new session or after any environment change:

```bash
cd /root/workspace/LMbot/LLMbot/baseline/core
nvidia-smi
python -c "import torch, torch_geometric, transformers, sklearn; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

If you want the same check from the local Windows machine, use:

```powershell
powershell -ExecutionPolicy Bypass -File tools/remote_gpu_smoke_test.ps1
```

The helper script reports `wandb` when it is installed, but it does not treat a missing `wandb` package as a baseline failure because the current mainline can fall back to a `NullRun`.

## Minimal Baseline Commands

From `/root/workspace/LMbot/LLMbot/baseline/core` with `lmbot` activated:

```bash
# Minimal LM -> MLP run
python main.py \
  --stage legacy_distill \
  --dataset TwiBot-20 \
  --seeds 1

# Minimal GNN-backed run
python main.py \
  --stage legacy_distill \
  --dataset TwiBot-20 \
  --use_GNN \
  --GNN_model rgcn \
  --seeds 1
```

## Formal Comparison Rule

For formal comparison stages or any run that must preserve canonical splits, pass `--reset_split -1` explicitly.

```bash
python main.py \
  --stage <formal_stage> \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --seeds 1
```

Keep this aligned with:

- `docs/protocols/harness-standard.md`
- `docs/protocols/agent-coding-guideline.md`
- `docs/protocols/baseline_comparability.md`
- `LLMbot/baseline/AGENTS.md`
- `LLMbot/baseline/README.md`

## Legacy Scripts Versus Current Mainline

Remote root-level scripts still exist, including:

- `/root/workspace/LMbot/run_rgt.sh`
- `/root/workspace/LMbot/run_rgt_5seed.sh`
- `/root/workspace/LMbot/run_with_monitoring.sh`

Use them only for historical reproduction lines that intentionally target the root-level legacy pipeline.

For current work:

- Prefer `/root/workspace/LMbot/LLMbot/baseline/core`
- Use `deploy_frmi_v2.sh` only when you are intentionally working on the FRMI v2 helper flow for TwiBot-22

`run_with_monitoring.sh` is not a safe default launcher for the current setup because it writes logs under `/workspace/LMbot/...`, which does not match the validated repo root `/root/workspace/LMbot/...`.
