# LMBot Social Bot Detection Workspace

> For AI agents, start with [AGENTS.md](AGENTS.md).

## Default Pipeline

The default onboarding and reproduction surface for the LMbot line is now `LLMbot/baseline/`.

- Use `LLMbot/baseline/core/main.py` as the primary CLI entrypoint.
- Start standard runs with `--stage legacy_distill`.
- Use `--seeds` for seed control. There is no `--seed` flag in the baseline CLI.
- Treat `LLMbot/code/` as historical and experimental work, not the default pipeline.

## Project Structure

```text
BotDetection/
|-- AGENTS.md               # Workspace entrypoint for agents
|-- docs/                   # Architecture, guides, protocols, and wiki memory
|-- LLMbot/                 # LMbot code line
|   |-- baseline/           # Active mainline and default onboarding surface
|   |-- code/               # Historical and experimental D3F work
|   |-- README.md           # Repo-local overview for the LLMbot line
|-- LMBot/                  # Harness and adjacent implementation line
|-- datasets/               # Canonical workspace datasets
|-- botbr/                  # BotBR comparison line
|-- HyperScan/              # HyperScan comparison line
|-- SEBot/                  # SEBot comparison line
```

## Quick Start

1. Read [docs/wiki/query_pack.md](docs/wiki/query_pack.md) and [docs/wiki/project/mainline_switch_context_2026-04-23.md](docs/wiki/project/mainline_switch_context_2026-04-23.md) for the current research state and routing.
2. Make sure `TwiBot-20` is available where the baseline loader can find it when run from `LLMbot/baseline/core`:
   - `LLMbot/baseline/core/datasets/TwiBot-20`
   - `LLMbot/baseline/datasets/TwiBot-20`
   - `LLMbot/datasets/TwiBot-20`
   - `datasets/TwiBot-20`
3. Run one of the baseline entrypoints below.

```bash
cd LLMbot/baseline/core

# LM -> MLP legacy distillation
python main.py \
  --stage legacy_distill \
  --dataset TwiBot-20 \
  --seeds 1

# LM + GNN + MLP legacy distillation
python main.py \
  --stage legacy_distill \
  --dataset TwiBot-20 \
  --use_GNN \
  --GNN_model rgcn \
  --seeds 1
```

## Current Positioning

- `LLMbot/baseline/` is the default mainline for onboarding, reproduction, and comparison-facing work.
- `LLMbot/code/` remains available for historical and exploratory RACE-Bot-D3F work.
- `docs/wiki/` remains the durable project memory layer.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Codebase map and execution surfaces
- [docs/guides/REMOTE_GPU_SERVER.md](docs/guides/REMOTE_GPU_SERVER.md) - Validated remote GPU server workflow
- [docs/guides/model.md](docs/guides/model.md) - Model and stage overview
- [docs/wiki/](docs/wiki/) - Project memory and experiment context
- [docs/protocols/](docs/protocols/) - Execution and comparison protocols
- [LLMbot/README.md](LLMbot/README.md) - LLMbot repo-local overview
- [LLMbot/baseline/README.md](LLMbot/baseline/README.md) - Baseline mainline quick reference
