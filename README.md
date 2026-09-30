# BotDetection Research Workspace

Start with [AGENTS.md](G:\Research\BotDetection\AGENTS.md).

## Active Surfaces

The active implementation surface is `LLMbot/code/`. The NLPCC submission line
has a trimmed source package under `NLPCC/code/` so submitted code can be
audited separately from later exploratory refactors.

- Primary CLI entrypoint: `LLMbot/main.py`
- Primary repo-local guidance: `LLMbot/AGENTS.md`
- Primary operator guide: `LLMbot/README.md`
- NLPCC submission package: `NLPCC/code/`

`LLMbot/baseline/` is a deprecated historical surface kept only for migration,
archival cleanup, or forensic comparison. `LLMbot/code/` is not deprecated; it
is the current active code surface.

## Quick Start

Run commands from `LLMbot/`:

```bash
cd LLMbot

# canonical onboarding path
python main.py \
  --experiment_task distillation_pipeline \
  --dataset TwiBot-20 \
  --seeds 1 \
  --disable_wandb

# graph-backed onboarding path
python main.py \
  --experiment_task distillation_pipeline \
  --dataset TwiBot-20 \
  --use_GNN \
  --graph_backbone botrgcn \
  --seeds 1 \
  --disable_wandb
```

Canonical public flags are:

- `--experiment_task`
- `--graph_backbone`
- `--text_encoder`
- `--semantic_encoder`
- `--embedding_path`

Legacy aliases such as `--stage`, `--GNN_model`, `--LM_model`, and `--emb_path`
still parse for compatibility, but they are no longer the preferred interface.

## Current Documentation Entry Points

- [docs/README.md](G:\Research\BotDetection\docs\README.md)
- [LLMbot/README.md](G:\Research\BotDetection\LLMbot\README.md)
- [docs/code/parser.md](G:\Research\BotDetection\docs\code\parser.md)
- [docs/ARCHITECTURE.md](G:\Research\BotDetection\docs\ARCHITECTURE.md)
- [docs/code/research.md](G:\Research\BotDetection\docs\code\research.md)
- [code.md](G:\Research\BotDetection\code.md)

## Current Refactor Snapshot

The active mainline has already moved onto a canonical public contract:

- `LLMbot/stage_registry.py` is the stage/task naming source of truth
- `LLMbot/parser_args.py` exposes canonical public flags and keeps hidden legacy aliases for one migration window
- `LLMbot/main.py` resolves tasks through the registry and supports a parser-only `--help` path
- `LLMbot/trainer.py` is now a thin compatibility facade
- `LLMbot/trainer_legacy_impl.py` still holds most execution logic during the extraction transition
- new artifact writes prefer canonical `preparation/` and `stages/<canonical_task>` namespaces

This means the public contract is cleaner than the execution body. Ongoing
maintenance work should keep pushing internal implementation toward the same
canonical vocabulary.

The current split modules such as `stage_runner.py`, `trainer_preparation.py`,
`trainer_semantic.py`, `trainer_graph.py`, `trainer_glance.py`, and
`stage_helpers.py` already define the intended canonical module surface, but
most still delegate into `trainer_legacy_impl.py` during the migration window.

## Documentation Sync Rule

Future code changes must update the matching code-development docs in the same
task. The minimum active set is:

- `LLMbot/README.md`
- `docs/ARCHITECTURE.md`
- `docs/code/parser.md`
- `docs/code/research.md`
- `code.md`

## Repository Structure

```text
BotDetection/
|-- AGENTS.md
|-- code.md
|-- docs/
|-- LLMbot/                # active mainline
|   |-- code/              # active flat Python source + claim governance markers
|-- NLPCC/
|   |-- code/              # trimmed NLPCC submission package
|-- LMBot/                 # adjacent legacy/reference line
|-- datasets/
|-- results/
|-- botbr/
|-- HyperScan/
|-- SEBot/
```

## Research Boundary

The active mainline currently contains:

- a public parser-exposed CLI surface
- additional internal stage handlers behind the public contract
- historical artifacts and notes from older routing eras

Do not assume every implemented branch in code is currently part of the public
CLI contract. Use `LLMbot/README.md` and `docs/code/parser.md` as the current
contract reference.

The NLPCC package is deliberately narrower than `LLMbot/code/`. It keeps only:
`main.py`, `parser_args.py`, `graph_detector.py`, `router.py`, `models.py`,
`data_io.py`, and `utils.py`.
