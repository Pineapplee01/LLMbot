# Architecture - LLMbot Root Mainline

> High-level codebase map for the current LLMbot onboarding surface.

## System Overview

The default pipeline inside `LLMbot/` is now the root mainline.

- Primary entrypoint: `LLMbot/main.py`
- Primary onboarding stage: `--stage legacy_distill`
- Seed control: `--seeds`
- Optional graph path: `--use_GNN`
- Deprecated legacy/reference surfaces: `LLMbot/baseline/` and `LLMbot/code/`

## Runtime Flow

```mermaid
graph LR
    A["Twibot-20 dataset"] --> B["utils package load_raw_data"]
    B --> C["main.py"]
    C --> D["legacy_distill without --use_GNN"]
    C --> E["legacy_distill with --use_GNN"]
    D --> F["LM_Trainer -> MLP_Trainer"]
    E --> G["LM_Trainer -> GNN trainer -> MLP_Trainer"]
    C --> H["StageRunner matrix, appendix, and EQC stages"]
```

## Active Mainline Files

| File | Role |
|------|------|
| `LLMbot/main.py` | Parses args, loops over `--seeds`, dispatches legacy or structured stages |
| `LLMbot/parser_args.py` | Source of truth for documented CLI flags and stage names |
| `LLMbot/trainer.py` | Legacy distillation loop plus `StageRunner` for matrix-style stages |
| `LLMbot/model_building.py` | Builds LM, GNN, estimator, semantic, repair, and selector components |
| `LLMbot/LM.py` | Language-model branch used by baseline training |
| `LLMbot/GNNs.py` | Graph backbones including `botrgcn` and `rgcn` |
| `LLMbot/utils/` | Dataset resolution, seed setup, artifact paths, checkpoint loading, and distilled-knowledge loading |

## Mainline Stage Surface

### `legacy_distill`

Default onboarding and reproduction stage.

- Without `--use_GNN`: LM to MLP legacy distillation
- With `--use_GNN`: LM, GNN, and MLP legacy distillation

### Structured follow-on stages

Available through `parser_args.py` and dispatched by `StageRunner`:

- `vertical_minimal`
- `estimator_matrix`
- `semantic_source_matrix`
- `semantic_matrix`
- `repair_matrix`
- `selector_matrix`
- `positioning_matrix`
- `backbone_stress`
- `appendix`
- `eqc_v8_matrix`

These stages remain part of the root mainline surface, but they are not the first-stop onboarding path.

## Research Phase Alignment

Use [research/project_phase_taxonomy.md](research/project_phase_taxonomy.md) for high-level research phase naming. These phase labels do not rename CLI `--stage` values.

| CLI or Work Surface | Research Phase Alignment |
| --- | --- |
| `semantic_source_matrix` | 阶段 A：semantic encoder -> GNN |
| `vertical_minimal`, `estimator_matrix` | 阶段 B：GNN 输出 -> post-hoc estimator |
| `semantic_matrix` | 阶段 C：risk/regime -> local semantic enhancement |
| `repair_matrix` | 阶段 D：risk/regime -> local propagation repair |
| `selector_matrix` | 阶段 E：action selection |
| same-head refinement work | 阶段 F：same-head refinement |
| `positioning_matrix`, `backbone_stress` | 阶段 G：positioning / stress test |

## Historical and Experimental Surface

`LLMbot/baseline/` and `LLMbot/code/` contain legacy/reference implementation lines. Keep them available for migration, deletion planning, archival cleanup, or forensic comparison, but do not use them as the default onboarding pipeline.

## Dataset Expectations

When the active CLI runs from `LLMbot/`, dataset discovery starts from the active loader context and configured dataset paths.

Document commands accordingly so the working directory and dataset lookup behavior stay aligned.
