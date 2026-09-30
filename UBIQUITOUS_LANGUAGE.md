# Ubiquitous Language

Date: 2026-07-04

## Pipeline Structure

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Active Mainline** | The supported operator surface under `LLMbot/` for future bot-detection pipeline work. | `baseline/core`, old mainline |
| **Active Source Directory** | The flat implementation directory `LLMbot/code/` that owns active Python business code. | legacy code dir, source package, methods dir |
| **Compatibility Entrypoint** | A root `LLMbot/*.py` shim that preserves operator commands while delegating to `LLMbot/code/*.py`. | real implementation, duplicate source |
| **Main Entry** | The operator entrypoint `LLMbot/main.py`, invoked as `python main.py ...` from `LLMbot/`, backed by `LLMbot/code/main.py`. | driver, launcher, train script |
| **CLI Contract** | The public argument surface and normalization rules owned by `LLMbot/code/parser_args.py`. | parser glue, flags blob |
| **Stage Registry** | The compatibility name for the experiment-task metadata source of truth in `LLMbot/code/stage_registry.py`. | task table, stage list |
| **Experiment Task** | A canonical runnable pipeline unit selected by `--experiment_task` and resolved through `StageSpec.canonical_name`. | stage, task alias |
| **Task Spec** | The preferred internal variable name for the runtime metadata record; the Python class is still named `StageSpec` during compatibility cleanup. | stage_spec in new code |
| **StageSpec** | The existing immutable metadata class containing canonical name, visibility, runner kind, split and graph requirements, and artifact namespace. | new public term |
| **Orchestration Layer** | The code that resolves canonical args, loops over seeds, chooses stage execution, and writes stage-level provenance. | training logic, algorithm code |
| **Stage Runner** | The orchestration object in `LLMbot/code/stage_runner.py` that dispatches graph, GLANCE, estimator, and diagnostic stages. | trainer, generic runner |
| **Stage Owner Surface** | A module such as `trainer_preparation.py`, `trainer_semantic.py`, `trainer_graph.py`, or `trainer_glance.py` that owns a coherent stage family. | helper file, extracted chunk |
| **Compatibility Facade** | `LLMbot/code/trainer.py`, which preserves legacy imports while redirecting active owners to narrower modules. | trainer implementation |
| **Migration Body** | `LLMbot/code/trainer_legacy_impl.py`, the large transitional execution body that still holds estimator/matrix, remaining graph-helper, and compatibility logic. | main trainer, legacy owner |

## Algorithm And Evidence

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Algorithm Layer** | Pure model, estimator, router, graph, and operator logic in modules such as `code/model_building.py`, `code/estimators.py`, `code/router.py`, `code/GNNs.py`, `code/operators.py`, and `code/hypergnn.py`. | orchestration, runner script |
| **Reliability Router** | The reliability-first router implementation owned by `LLMbot/code/router.py`, trained to rank likely base-detector failures. | GLANCE router, selector, classifier |
| **Estimator Surface** | The uncertainty, conformal, KNN-risk, and router-risk families concentrated in `LLMbot/code/estimators.py`. | metrics script, router module |
| **Precompute Context** | The independent bounded context in `LLMbot/code/precompute.py`, reached through root `LLMbot/precompute.py`, that materializes prompt, evidence, embedding, and LLM-guide caches. | stage, preparation stage |
| **Prompt Cache** | A precomputed tensor/sidecar payload consumed later by semantic, graph, router, or refiner stages. | result, model output |
| **Artifact Surface** | Generated files under `experiments/`, `server_logs/`, seed `preparation/`, seed `stages/`, and queue manifests. | source code, docs |
| **Manifest** | A structured provenance record such as `manifest.json` or a queue manifest that describes command, config, artifact paths, and claim-boundary metadata. | note, log |
| **Runner Script** | A curated `run_*.py`, `.ps1`, `.cmd`, or shell script that launches one or more CLI runs and records queue state. | main entry, experiment code |

## Naming

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Canonical Name** | The current public spelling used for new commands, docs, manifests, and governance records. | preferred alias |
| **Legacy Alias** | An old spelling kept readable for compatibility only. | public name |
| **Public Stage** | A stage whose `StageSpec.visibility` is `public` and may be documented as operator-facing. | implemented branch |
| **Internal Stage** | A stage whose `StageSpec.visibility` is `internal` and must not be advertised as public or claim-grade. | hidden feature |
| **Claim Boundary** | The evidence limit beyond which docs and reports must not make method, reproduction, or performance claims. | caveat, disclaimer |
| **Runtime-Only Artifact** | A helper output under runtime or sidecar paths used by code execution but not intended as a public artifact contract. | public result |

## NLPCC Submission Package

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **NLPCC Submission Package** | The seven-module source package under `NLPCC/code/` that exposes only the paper-facing graph-detector line. | LLMbot snapshot, claim folder, runner surface |
| **Graph Detector Task** | The NLPCC task selected by `--experiment_task graph_detector_prepare`. | stage, frozen_g0, GLANCE task |
| **Task Description** | A side-effect-free dictionary describing the selected NLPCC graph-detector run contract. | training result, ready status, dry run |
| **NLPCC Router** | The small reliability-ranking utility surface in `NLPCC/code/router.py`. | risk_router, GLANCE router, metrics module |
| **Submission Module Interface** | The explicit `__all__` list that defines each NLPCC module's public surface. | helper export, wildcard surface |

## Relationships

- A **Compatibility Entrypoint** preserves root-level operator commands and
  delegates to the **Active Source Directory**.
- A **Main Entry** consumes the **CLI Contract** and resolves one canonical
  **StageSpec** per requested task.
- A **StageSpec** belongs to exactly one **Stage Registry** and defines the
  stage's public/internal boundary plus runtime gates such as `runner_kind`,
  `claim_grade_allowed`, `forces_use_gnn`, and `graph_data_mode`.
- The **Orchestration Layer** may call **Stage Owner Surfaces**, the
  **Compatibility Facade**, or the **Stage Runner**, but it should not absorb
  **Algorithm Layer** logic.
- The **Precompute Context** produces **Prompt Caches** that later stages may
  consume through explicit path flags.
- A **Runner Script** invokes a CLI entrypoint and writes queue/log metadata to
  the **Artifact Surface**.
- A **Manifest** records provenance for an **Artifact Surface** item and is not
  a substitute for source code.
- The **NLPCC Submission Package** contains exactly one public **Graph Detector
  Task** and exposes only side-effect-free **Task Description** and utility
  interfaces until training code is intentionally ported.

## Refactor Naming Standard 2026-07-04

Use canonical names in new code, public docs, manifests, and governance notes:

| Concept | Canonical name | Legacy aliases kept only for compatibility notes |
| --- | --- | --- |
| Public task selector | `experiment_task` | `stage` |
| Graph backbone | `graph_backbone` | `GNN_model` |
| Text encoder | `text_encoder` | `LM_model` |
| Semantic encoder | `semantic_encoder` | `semantic_backbone` |
| Semantic embedding path | `embedding_path` | `emb_path`, `g0_feature_path` |
| Graph-detector preparation task | `graph_detector_prepare` | `frozen_g0` |
| Final detector representation | `fused_x` | `node_repr` |

New runner methods still use `_run_<StageSpec.canonical_name>` while the
compatibility class name remains `StageSpec`. New orchestration locals should
prefer `task_spec`, `requested_task`, and `execution_task`; `stage_spec`,
`requested_stage`, and `execution_stage` remain compatibility aliases only.
`parser_args.py` is the only active-mainline module allowed to create legacy
shadow fields; new code should read canonical fields.

Domain-specific internal variable names:

- node sets: `target_node_ids`, `routed_node_ids`, `center_node_ids`
- masks and indices: `train_idx`, `valid_idx`, `test_idx`, `train_mask`,
  `valid_mask`, `test_mask`
- representations: `fused_x`, `x_low`, `x_new`, `x_high`,
  `semantic_embeddings`
- router signals: `risk_score`, `router_score`, `oracle_advantage`,
  `utility_reward`
- artifacts: `stage_dir`, `artifact_namespace`, `manifest`,
  `feature_manifest`

Medium cleanup may remove hidden deprecated-only compatibility such as
`eqc_v8_matrix`, but it must not remove public canonical task names or common
renamed aliases in the same slice.

Owner modules are the implementation source of truth. For distillation,
`trainer_distillation.py` owns `LM_Trainer`, `GNN_Trainer`, `MLP_Trainer`,
`_safe_pseudo_label_training_index`, and `run_legacy_graph_seed`;
`trainer_legacy_impl.py` may expose those names only as compatibility aliases.

Runtime gate status: `LLMbot/code/main.py` now consumes `StageSpec.runner_kind`,
`StageSpec.claim_grade_allowed`, `StageSpec.forces_use_gnn`,
`StageSpec.requires_canonical_split`, and `StageSpec.graph_data_mode` as the
first runtime policy source. `local_conflict_prune_diag` structural-view and
router helpers now belong to `LLMbot/code/trainer_graph.py`, not the
**Migration Body**. The DIGNN-style conflict refiner core now lives directly in
`LLMbot/code/trainer_dignn_conflict.py`; `LLMbot/code/conflict_refiner.py` was
removed as a single-owner helper file. `LM_Model` now lives in
`LLMbot/code/model_building.py`, while `RGTLayer` and `SimpleHGNConv` live in
`LLMbot/code/GNNs.py`.

## LLMbot Naming Governance 2026-07-05

This section governs new and migrated code in the active `LLMbot/code/`
research mainline. Existing historical code may retain legacy names while the
migration body shrinks, but new public docs, new claim docs, and new source
surfaces should use the canonical vocabulary below.

### Common canonical vocabulary

| Concept | Canonical name | Legacy-only aliases |
| --- | --- | --- |
| Public runnable selector | `experiment_task` | `stage` |
| Runtime metadata local | `task_spec` | `stage_spec`, `stage_name` |
| Graph backbone | `graph_backbone` | `GNN_model` |
| Text encoder | `text_encoder` | `LM_model` |
| Semantic encoder | `semantic_encoder` | `semantic_backbone` |
| Semantic embedding path | `embedding_path` | `emb_path` |
| Final detector representation | `fused_x` | `node_repr` |
| Target-node set | `target_node_ids` | target ids, focal ids |
| Routed-node set | `routed_node_ids` | selected ids, routed ids |
| Center-node set | `center_node_ids` | center ids |
| Risk estimate | `risk_score` | uncertainty score |
| Router ranking signal | `router_score` | selector score |
| Artifact namespace | `artifact_namespace` | stage namespace |
| Provenance record | `manifest` | log note |

`stage_dir` remains the canonical artifact-path variable because the existing
artifact layout uses `stages/<canonical_task>`. `StageSpec`,
`stage_registry.py`, and `stage_runner.py` remain compatibility names for the
current orchestration layer; do not use them as a reason to introduce new
business variables named `stage`.

### Claim vocabulary

| Claim | Canonical vocabulary |
| --- | --- |
| `phase_a_foundations` | `semantic_embeddings`, `graph_detector_outputs`, `fused_x`, `train_idx`, `valid_idx`, `test_idx`, `calibration_logits` |
| `high_order_graph_consumption` | `x_low`, `x_new`, `x_high`, `second_view_edges`, `routed_highpass_bundle`, `center_node_ids` |
| `conformal_risk_routing` | `risk_score`, `risk_scores`, `router_score`, `risk_budget`, `selected_node_ids`, `coverage_curve` |
| `routed_llm_evidence_refinement` | `prompt_expert_bundle`, `evidence_card`, `refiner_features`, `oracle_advantage`, `utility_reward`, `gate_prob` |
| `local_graph_repair_diagnostics` | `conflict_score`, `pruned_edge_index`, `repair_delta`, `structural_view_logits`, `deferral_decision` |
| `semantic_candidate_correction` | `semantic_gate_score`, `candidate_output_path`, `correction_delta`, `semantic_evidence` |
| `ablation_positioning_legacy` | `replay_compatibility`, `legacy_alias`, `historical_command` |

Before adding or moving code, choose the claim first, then use the common
canonical vocabulary plus that claim's vocabulary. `ablation_positioning_legacy`
must not define new public names; it only documents replay-compatible legacy
terms.

## Example Dialogue

> **Dev:** "Should I add this new routed evidence path as another `main.py`
> branch?"

> **Domain expert:** "Only if it is a public **Stage**. If it materializes
> prompt evidence before the pipeline consumes it, keep it in the
> **Precompute Context** and pass the cache through an explicit path flag."

> **Dev:** "Can I call `frozen_g0` the stage name in the docs?"

> **Domain expert:** "No. Use the **Canonical Name**
> `graph_detector_prepare`; mention `frozen_g0` only as a **Legacy Alias** in a
> compatibility note."

> **Dev:** "Where should a queue manifest from `run_*.py` live?"

> **Domain expert:** "It is part of the **Artifact Surface**. Read and verify
> it, but do not hand-edit it during governance work."

> **Dev:** "Should I create `stages/` or `methods/` under `code/` now?"

> **Domain expert:** "No. Keep the **Active Source Directory** flat until dead
> code deletion and function consolidation reduce the migration body."

## Flagged Ambiguities

- `stage` and `experiment_task` both appear in code paths. Use
  `experiment_task` for public CLI and docs; treat `stage` as compatibility
  unless discussing internal normalization.
- `GNN_model` and `graph_backbone` both exist. Use `graph_backbone` for new
  public commands and docs; keep `GNN_model` only as a legacy alias.
- `node_repr` and `fused_x` can refer to final detector hidden state in older
  artifacts. Use `fused_x` for new final-detector-space wording; mention
  `node_repr` only as a compatibility alias or explicit control space.
- `precompute.py` can feed preparation and refinement workflows, but it is not
  a normal `main.py` stage and should not be documented as one.
- `LLMbot/code/` used to be named in older docs as a legacy surface. It is now
  the **Active Source Directory**; only `LLMbot/baseline/` remains deprecated.
- `NLPCC/code/` is not another **Active Source Directory**. It is the
  **NLPCC Submission Package**, so historical LLMbot terms such as `stage`,
  `joint_router_refinement`, `trainer_glance`, `dry_run`, and `claims/` should
  not reappear there.
