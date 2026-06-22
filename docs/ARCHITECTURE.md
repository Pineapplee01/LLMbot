# LLMbot Architecture Map

This document is the code-ownership and dependency map for the active
`LLMbot/` mainline. Keep operator commands in `README.md`, experiment launch
rules in `experiment_runbook.md`, and run records in `experiments.md`.

## Execution Flow

```text
main.py
  -> parser_args.py / stage_registry.py
  -> trainer_semantic.py for semantic preparation stages
  -> trainer_preparation.py for graph detector and graph calibration artifacts
  -> trainer_distillation.py for distillation_pipeline
  -> stage_runner.py for graph, GLANCE, router, diagnostic, and ablation stages
       -> trainer_graph.py GraphStageMixin
       -> trainer_glance.py GlanceStageMixin
       -> trainer_legacy_impl.py compatibility fallback
```

`runtime_env.py` is shared by queue scripts, report helpers, and runtime
training helpers. It owns local process environment cleanup, model-cache
variables, subprocess logging, CSV/JSON helpers, and queue manifest lifecycle
helpers.

## Module Responsibilities

| Module | Responsibility | Main dependencies | Current refactor note |
| --- | --- | --- | --- |
| `main.py` | CLI entry point, seed loop, claim/data validation, high-level stage dispatch | `parser_args`, `stage_registry`, `artifact_contracts`, `trainer_*` | Keep thin; move deeper runtime policy into stage modules. |
| `parser_args.py` | Public CLI flags, legacy alias compatibility, normalized args | `stage_registry` | Parser-only changes must keep `python main.py --help` green. |
| `stage_registry.py` | Canonical task names, visibility, legacy names, stage policy metadata | none local | Source of truth for public/internal stage naming. |
| `artifact_contracts.py` | Canonical artifact paths, split provenance, frozen G0 path contract | `stage_registry`, `model_building`, `utils` | Owner for artifact/stage contract helpers; legacy modules should import from here. |
| `stage_runner.py` | Runtime stage orchestration and mixin dispatch | `trainer_graph`, `trainer_glance`, `trainer_legacy_impl`, `artifact_contracts` | Transitional bridge from legacy monolith to extracted stage owners. |
| `trainer.py` | Compatibility facade for older imports | `stage_runner`, `trainer_preparation`, `trainer_semantic`, `trainer_distillation` | Keep thin and avoid new logic here. |
| `trainer_legacy_impl.py` | Legacy compatibility sink and fallback implementation body | `trainer_preparation`, `trainer_semantic`, `model_building`, `estimators`, `runtime_env` | Do not add new stage logic; delete duplicated helpers as owners move out. Keep its legacy stage-read map local until alias compatibility is reconciled with `stage_registry.py`. |
| `trainer_preparation.py` | Graph detector/GATS artifact creation, graph refinement, manifest matching, and training entrypoints | `GNNs`, `hypergnn`, `model_building`, `artifact_contracts`, `trainer_preparation_artifacts` | Artifact/gate loading now lives in `trainer_preparation_artifacts.py`; next split is graph-refinement ownership. |
| `trainer_preparation_artifacts.py` | Preparation artifact and faithful-gate path resolution, manifest loading, and read compatibility | `artifact_contracts`, `stage_registry`, `utils` | Keep pure I/O and contract checks here; no training or graph-refinement logic. |
| `trainer_semantic.py` | Semantic finetune, embedding classifier, semantic correction gate | `model_building`, `runtime_env`, `utils` | Candidate split: semantic models, feature builders, stage entrypoints. |
| `trainer_graph.py` | Graph diagnostic and graph-stage mixin behavior for `StageRunner` | `artifact_contracts`, `estimators`, `trainer_preparation` | Remove remaining legacy fallback imports before larger extraction. |
| `trainer_glance.py` | GLANCE/router/refiner stages, prompt-expert routing, selector diagnostics | `estimators`, `model_building`, `router`, `trainer_semantic` | Large hotspot; split models/utilities before moving stage methods. |
| `trainer_distillation.py` | Distillation pipeline trainers and legacy graph seed execution | `dataloader`, `model_building`, `runtime_env` | Candidate split: trainer classes vs stage entrypoint vs metrics helpers. |
| `model_building.py` | LM/GNN/estimator/operator construction and feature bundle resolution | `GNNs`, `LM`, `estimators`, `operators`, `subgroups` | Construction layer; avoid importing trainer modules here. |
| `GNNs.py` | Graph backbones, HyperScan-style branches, routed contrast losses | `RGT`, `SimpleHGN`, `hypergnn` | Model layer; do not add artifact or experiment orchestration here. |
| `estimators.py` | Risk estimators, router metrics, budget curves, calibration utilities | none local | Large hotspot; split estimator families only with metric regression checks. |
| `operators.py` | Semantic and repair operators used by model construction | none local | Keep operator factory contracts stable. |
| `hypergnn.py` | KNN/hypergraph construction and dynamic hypergraph helpers | none local | Shared by model and precompute paths. |
| `precompute.py` | Prompt/explanation/cache generation and embedding precompute CLI | `hypergnn`, `prompt` | Large standalone CLI; avoid importing trainer stack. |
| `prompt.py` | Prompt templates and prompt construction helpers | none local | Text-generation support layer. |
| `router.py` | Router model utilities and reliability feature bundle | none local | Used by GLANCE and legacy compatibility paths. |
| `subgroups.py` | Structural feature and subgroup manifest helpers | none local | Shared analysis/operation support. |
| `dataloader.py` | Legacy LM/GNN/MLP dataloader builders | none local | Used by distillation and legacy paths. |
| `runtime_env.py` | Process/env/path/manifest helper layer | none local | Shared utility owner for queue and runtime helpers. |

## Supporting Root Modules

| Module | Responsibility | Main dependencies | Current refactor note |
| --- | --- | --- | --- |
| `LM.py` | Text encoder wrapper for local HuggingFace encoder loading | none local | Model layer; keep trainer logic out. |
| `RGT.py` | RGT semantic-attention layer definitions | none local | Imported by `GNNs.py`; keep as backbone component. |
| `SimpleHGN.py` | SimpleHGN convolution implementation | none local | Imported by `GNNs.py`; file uses BOM-compatible encoding. |
| `preprocess.py` | Dataset preprocessing and artifact materialization CLI | none local | Generated artifacts must be produced by documented commands only. |
| `conflict_refiner.py` | DIGNN-style structural conflict/refinement model utilities | none local | Used by `trainer_dignn_conflict.py`; model utility layer. |
| `trainer_dignn_conflict.py` | Local DIGNN conflict-refinement diagnostic stage | `conflict_refiner`, `model_building` | Keep as diagnostic stage owner. |
| `llm_evidence_refiner.py` | Standalone LLM-evidence refiner utilities | none local | Isolated support layer; verify imports before moving. |

## Experiment Helper Surface

Root-level dated scripts are operational launch or reporting surfaces, not
model logic:

| Script | Role | Shared dependency |
| --- | --- | --- |
| `launch_sampled_twibot22_base5_20260620.py` | Windows launcher for sampled TwiBot-22 queue | `runtime_env` |
| `run_sampled_twibot22_base_5seed_20260620.py` | Sampled TwiBot-22 queue | `runtime_env` |
| `run_twibot20_formal5_ablation_completion_20260620.py` | Formal TwiBot-20 ablation queue | `runtime_env` |
| `run_twibot20_full_selective_residual_5seed_20260620.py` | Full selective residual queue | `runtime_env` |
| `extract_raw_roberta_embeddings_20260620.py` | Support artifact generator | `runtime_env` |
| `summarize_formal_runs_20260620.py` | Report snapshot helper | `runtime_env` |
| `check_experiment_helpers_20260620.py` | Non-generating dry validation | `runtime_env` |

## Complete Python File Index

This index covers the tracked Python implementation surface under this
mainline, excluding generated caches and evidence directories. It is the first
place to check before moving functions across files.

| File | Role | Local dependency direction |
| --- | --- | --- |
| `artifact_contracts.py` | Canonical stage/artifact path and provenance contract helpers | Imports `stage_registry`, `model_building`, and `utils`; imported by trainer and runner layers. |
| `check_experiment_helpers_20260620.py` | Dry validation for queue/helper command contracts | Imports `runtime_env`; must stay non-generating. |
| `conflict_refiner.py` | DIGNN-style structural conflict/refinement model utilities | Model/support layer; no trainer imports. |
| `dataloader.py` | Legacy dataloader builders for LM/GNN/MLP flows | Consumed by distillation and legacy paths. |
| `estimators.py` | Risk estimators, calibration, router metrics, utility curves | Consumed by GLANCE, graph, and legacy trainers; should not import trainers. |
| `extract_raw_roberta_embeddings_20260620.py` | Operational support-artifact extraction helper | Imports `runtime_env`; evidence writes must come from documented runs. |
| `GNNs.py` | Graph backbone implementations and high-order branch logic | Imports `RGT`, `SimpleHGN`, and `hypergnn`; model layer only. |
| `hypergnn.py` | Hypergraph/KNN construction and dynamic incidence helpers | Shared by `GNNs.py` and `precompute.py`; no trainer imports. |
| `launch_sampled_twibot22_base5_20260620.py` | Windows launcher wrapper for sampled TwiBot-22 queue | Imports `runtime_env`; operational surface only. |
| `llm_evidence_refiner.py` | Standalone LLM-evidence refinement helpers | Support layer; keep isolated from stage dispatch. |
| `LM.py` | Local HuggingFace text encoder wrapper | Model layer; imported through construction paths. |
| `main.py` | CLI entry point, validation, seed loop, stage dispatch | Imports parser/registry/contracts and trainer entrypoints. |
| `model_building.py` | Factory layer for LM/GNN/estimator/operator construction | Imports model and utility layers; must not import trainer modules. |
| `operators.py` | Semantic/repair operator classes and factories | Construction dependency; should remain trainer-free. |
| `parser_args.py` | Public CLI schema and compatibility normalization | Imports `stage_registry`; parser checks must stay cheap. |
| `precompute.py` | Standalone prompt, explanation, cache, and embedding precompute CLI | Imports `hypergnn` and `prompt`; should not import training orchestration. |
| `preprocess.py` | Dataset preprocessing/artifact materialization CLI | Data-prep surface; generated outputs require documented commands. |
| `prompt.py` | Prompt templates and prompt construction helpers | Text-support layer; no runtime orchestration. |
| `RGT.py` | RGT semantic-attention backbone components | Imported by `GNNs.py`; model layer only. |
| `router.py` | Router utilities and reliability feature bundle helpers | Consumed by GLANCE/legacy router paths. |
| `run_sampled_twibot22_base_5seed_20260620.py` | Sampled TwiBot-22 queue definition | Imports `runtime_env`; operational surface only. |
| `run_twibot20_formal5_ablation_completion_20260620.py` | Formal TwiBot-20 ablation-completion queue | Imports `runtime_env`; operational surface only. |
| `run_twibot20_full_selective_residual_5seed_20260620.py` | Full selective-residual queue | Imports `runtime_env`; operational surface only. |
| `runtime_env.py` | Process environment, offline model cache, subprocess, JSON/CSV, queue manifest helpers | Shared utility layer; do not reimplement these helpers in trainers. |
| `SimpleHGN.py` | SimpleHGN convolution component | Imported by `GNNs.py`; model layer only. |
| `stage_registry.py` | Canonical stage names, aliases, visibility, and stage policy metadata | Registry source of truth; should stay dependency-light. |
| `stage_runner.py` | Stage orchestration bridge and mixin composition | Imports graph/GLANCE mixins, contracts, and legacy fallback. |
| `subgroups.py` | Structural feature and subgroup manifest helpers | Support layer for construction/analysis paths. |
| `summarize_formal_runs_20260620.py` | Report snapshot helper for formal run manifests | Imports `runtime_env`; report refreshes are not refactor validation. |
| `trainer.py` | Thin compatibility facade preserving historical imports | Re-exports extracted owners and `StageRunner`; keep logic out. |
| `trainer_dignn_conflict.py` | DIGNN conflict-refinement diagnostic stage owner | Imports `conflict_refiner` and `model_building`; diagnostic stage only. |
| `trainer_distillation.py` | Distillation trainers, stage entrypoint, and legacy graph seed execution | Imports `dataloader`, `model_building`, and `runtime_env`. |
| `trainer_glance.py` | GLANCE/router/refiner stage logic and prompt-expert selector families | Imports estimator, router, semantic, and construction layers; major split target. |
| `trainer_graph.py` | Graph-stage mixin and graph diagnostics for `StageRunner` | Imports contracts, estimators, and preparation helpers. |
| `trainer_legacy_impl.py` | Compatibility fallback for not-yet-extracted legacy stage bodies | May import extracted owners as shims; should lose ownership over time. |
| `trainer_preparation.py` | Graph detector, GATS calibrator training, graph-refinement, manifest matching | Imports graph/model/contract utilities plus `trainer_preparation_artifacts`; next primary split target is graph-refinement. |
| `trainer_preparation_artifacts.py` | Frozen G0 and GATS artifact/gate loaders with canonical/legacy read compatibility | Imports contract, registry, and utility layers; do not add training code. |
| `trainer_semantic.py` | Semantic finetune, embedding classifier, semantic gate, feature preparation | Imports model/runtime/utils layers; split after preparation ownership. |
| `utils/__init__.py` | Utility package marker and re-export surface | Keep tiny and side-effect-free. |
| `utils/calibration.py` | Calibration and conformal helper utilities | Utility layer used by estimators/trainers. |
| `utils/losses.py` | Shared loss helpers | Utility layer; no orchestration imports. |
| `utils/metrics.py` | Shared metric helpers | Utility layer; no orchestration imports. |
| `utils/misc.py` | IO, reproducibility, hashing, artifact, and convenience helpers | Shared utility layer imported by contracts/trainers/model helpers. |

## Dependency Map Summary

The intended import direction is:

```text
CLI/queues
  -> registry/contracts/runtime helpers
  -> stage orchestration
  -> extracted trainer owners
  -> model construction
  -> models/estimators/operators/utils
```

Refactors should move code down to the owner that already supplies the data or
contract, then leave a compatibility import in the old location only when older
callers still need that symbol. Do not move model code upward into trainers,
do not import trainer code from `model_building.py`, `GNNs.py`, `estimators.py`,
or `utils/`, and do not let dated queue scripts become reusable libraries.

## Import Direction Rules

- CLI and orchestration layers may import trainer, registry, and contract
  modules.
- Model construction may import model/backbone/operator/estimator modules, but
  must not import trainer modules.
- `artifact_contracts.py` owns path/provenance helpers that are shared across
  trainer modules.
- `runtime_env.py` owns local process and queue-helper concerns; training logic
  should not reimplement environment cleanup, CSV/JSON writing, or queue
  manifest lifecycle helpers.
- `trainer_legacy_impl.py` is compatibility-only. New or cleaned-up owners
  should live in the extracted modules and be imported by the legacy file only
  as shims.
- `trainer_legacy_impl.py` still keeps a local legacy stage-read mapping because
  it is intentionally narrower than `stage_registry.py` for at least
  `semantic_embedding_classifier -> lm_embedding_classifier`. Do not replace
  that read-candidate map with the global registry helper without a dedicated
  compatibility migration and smoke check.

## Refactor Queue

1. Keep deleting duplicate helpers from `trainer_legacy_impl.py` after the
   extracted owner is proven by imports and smoke checks.
2. Continue splitting `trainer_preparation.py`: artifact/gate loading now lives
   in `trainer_preparation_artifacts.py`; next extract graph-refinement
   request/build/apply helpers.
3. Split `trainer_semantic.py` into stage entrypoints, feature builders, and
   semantic gate model classes.
4. Split `trainer_glance.py` model classes and routing utilities before moving
   stage methods.
5. Split `trainer_distillation.py` into trainer classes, metrics helpers, and
   `run_legacy_graph_seed`.

## Validation Gates

For architecture or ownership changes, run at least:

```powershell
python check_experiment_helpers_20260620.py
python main.py --help
python -m py_compile trainer_legacy_impl.py stage_runner.py trainer.py trainer_preparation.py trainer_preparation_artifacts.py trainer_semantic.py trainer_graph.py trainer_glance.py trainer_distillation.py
git diff --check
```

Do not use queue scripts, training runs, report refreshes, or evidence-zone
writes as routine refactor validation.
