# BotDetection Research Workspace

> For AI agents, start with [AGENTS.md](AGENTS.md).

## Active Surfaces

- `LLMbot/` - active code mainline for bot-detection pipeline work.
- `NLPCC/` - active NLPCC paper and task line.
- `docs/` - workspace architecture, protocols, guides, and wiki memory.
- `datasets/` and `results/` - evidence zones; do not hand-edit generated evidence.
- `LMBot/`, `botbr/`, `HyperScan/`, and `SEBot/` - comparison/reference lines.

## Quick Start

For code work, start with [AGENTS.md](AGENTS.md), then
[LLMbot/AGENTS.md](LLMbot/AGENTS.md) and [LLMbot/README.md](LLMbot/README.md).

For NLPCC paper work, start with [NLPCC/AGENTS.md](NLPCC/AGENTS.md),
[NLPCC/README.md](NLPCC/README.md), and
[NLPCC/docs/artifact_inventory.md](NLPCC/docs/artifact_inventory.md).

## Project Structure

```text
BotDetection/
|-- AGENTS.md               # Workspace entrypoint for agents
|-- docs/                   # Architecture, guides, protocols, and wiki memory
|-- LLMbot/                 # Active code mainline and nested repository boundary
|-- NLPCC/                  # Active NLPCC paper/task line
|-- LMBot/                  # Harness and adjacent implementation line
|-- datasets/               # Canonical workspace datasets
|-- botbr/                  # BotBR comparison line
|-- HyperScan/              # HyperScan comparison line
|-- SEBot/                  # SEBot comparison line
```

## Current Positioning

- `LLMbot/` is the current active code mainline.
- `NLPCC/` is the current active NLPCC paper/task line.
- `LLMbot/baseline/` and `LLMbot/code/` are deprecated legacy/reference surfaces.
- `docs/wiki/` remains the durable project memory layer.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Codebase map and execution surfaces
- [docs/CONVENTIONS.md](docs/CONVENTIONS.md) - Routing index for current standards
- [docs/guides/REMOTE_GPU_SERVER.md](docs/guides/REMOTE_GPU_SERVER.md) - Validated remote GPU server workflow
- [docs/guides/model.md](docs/guides/model.md) - Model and stage overview
- [docs/wiki/](docs/wiki/) - Project memory and experiment context
- [docs/protocols/](docs/protocols/) - Execution and comparison protocols
- [LLMbot/README.md](LLMbot/README.md) - LLMbot repo-local overview
- [NLPCC/README.md](NLPCC/README.md) - NLPCC paper/task overview
- [NLPCC/docs/artifact_inventory.md](NLPCC/docs/artifact_inventory.md) - NLPCC local artifact inventory
