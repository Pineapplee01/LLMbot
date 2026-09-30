# Project Context - Active Mainline

This note provides a compact implementation context for the current project
state.

## Active Mainline

All current implementation work should start from `LLMbot/`.

Primary files:
- `LLMbot/main.py`
- `LLMbot/parser_args.py`
- `LLMbot/trainer.py`
- `LLMbot/model_building.py`
- `LLMbot/estimators.py`

## Deprecated Surfaces

The following directories are no longer active implementation surfaces:

- `LLMbot/baseline/`
- `LLMbot/code/`

They remain available only for migration, archival cleanup, deletion, or
forensic comparison.

## Public CLI Contract

Current canonical flags:
- `--experiment_task`
- `--graph_backbone`
- `--text_encoder`
- `--semantic_encoder`
- `--embedding_path`

Legacy aliases still parse internally, but they are compatibility shims rather
than the preferred interface.

## Current Structural Reality

- public parser surface exists
- internal stage surface is larger than public parser surface
- `trainer.py` is the main orchestration hotspot
- naming cleanup is partial: CLI naming is partly canonicalized, internal
  namespace is still mostly legacy

## How To Read The Codebase

Use this order:

1. `LLMbot/parser_args.py` for public interface truth
2. `LLMbot/main.py` for task routing and claim-grade gates
3. `LLMbot/trainer.py` for stage execution and artifact flow
4. `LLMbot/model_building.py` and `LLMbot/estimators.py` for model/risk modules
5. `code.md` for current risk summary

## Current Refactor Priorities

1. reconcile public and internal stage registries
2. reduce misleading legacy naming in public surfaces
3. split `trainer.py` by responsibility
4. tighten provenance for cached calibration and external graph inputs
