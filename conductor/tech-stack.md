# LLMbot Tech Stack

Date: 2026-07-04

## Runtime

- Language: Python.
- Primary operator root: `LLMbot/`.
- Primary active source root: `LLMbot/code/`.
- Main CLI: `python main.py ...` from `LLMbot/`.
- Prompt/evidence precompute CLI: `python precompute.py ...` from `LLMbot/`.
- Operating environment: Windows / PowerShell is the default local operator
  environment for this workspace.
- Known Conda environment in runner scripts:
  `D:\Anaconda\envs\llmbot\python.exe`.

## ML And Graph Stack

The active mainline uses PyTorch-centered training and graph workloads:

- PyTorch tensors, checkpoints, and `.pt` artifacts.
- HuggingFace / Transformers model loading for RoBERTa, Qwen, and related
  encoder or causal-LM paths.
- PyTorch Geometric and DHG surfaces for graph and hypergraph branches where
  selected by parser flags.
- Weights & Biases is optional runtime logging and must be disableable through
  `--disable_wandb`.

## Local Model Cache Contract

Model downloads and HuggingFace/Transformers caches for BotDetection
experiments must stay under:

- `G:\Research\BotDetection\models`
- `HF_HOME=G:\Research\BotDetection\models\huggingface`
- `HF_HUB_CACHE=G:\Research\BotDetection\models\huggingface\hub`
- `TRANSFORMERS_CACHE=G:\Research\BotDetection\models\huggingface\hub`

Runner scripts such as
`LLMbot/run_sampled_twibot22_base_5seed_20260620.py` already set these cache
variables before invoking `main.py`.

## Code Surfaces

- Compatibility entrypoints: `LLMbot/main.py`, `LLMbot/precompute.py`,
  `LLMbot/preprocess.py`.
- Main entry implementation and CLI contract: `LLMbot/code/main.py`,
  `LLMbot/code/parser_args.py`.
- Stage metadata and orchestration: `LLMbot/code/stage_registry.py`,
  `LLMbot/code/stage_runner.py`.
- Stage owner surfaces: `LLMbot/code/trainer_preparation.py`,
  `LLMbot/code/trainer_semantic.py`, `LLMbot/code/trainer_graph.py`,
  `LLMbot/code/trainer_glance.py`, `LLMbot/code/trainer_distillation.py`.
- Compatibility facade and migration body: `LLMbot/code/trainer.py`,
  `LLMbot/code/trainer_legacy_impl.py`.
- Algorithm surfaces: `LLMbot/code/model_building.py`,
  `LLMbot/code/estimators.py`, `LLMbot/code/router.py`,
  `LLMbot/code/GNNs.py`, `LLMbot/code/operators.py`,
  `LLMbot/code/hypergnn.py`.
- Precompute bounded context: `LLMbot/code/precompute.py` and prompt
  construction in `LLMbot/code/prompt.py`.

## Dependency Policy

- Do not add dependencies for governance checks.
- Root validation and governance scripts must remain implementable with Python
  standard library, PowerShell, Bash, and Git.
- Any future dependency change must document affected commands, environment or
  lock/config files, and rollback risk.

## Artifact Surface

Generated evidence and runtime outputs include:

- `LLMbot/experiments/`
- `LLMbot/server_logs/`
- seed-level `preparation/` and `stages/` directories
- `manifest.json`, `metrics.json`, `outputs.pt`, `checkpoint.pt`,
  `per_node_test.jsonl`, queue manifests, and log files

These are read/verify surfaces by default, not manual-edit targets.
