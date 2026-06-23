# Research Pipeline and Code Alignment

This document fixes the active research pipeline and records how much of it is implemented in the `LLMbot/` root mainline.

It is a research-code alignment document. It does not claim experimental performance, paper readiness, causal graph repair, or official reproduction of external methods.

## Fixed Data Flow

```text
SimTeG-style LM+GNN
  -> 2-hop conformal estimator
  -> hard node router
  -> local retrieval/rewriter
  -> evidence graph
  -> LLM/refiner
```

The active design is post-hoc and local: freeze or reuse the LM+GNN base detector, judge uncertain ego contexts, refine only selected hard nodes, serialize refined ego evidence, and optionally use an LLM embedding/refiner. It does not retrain a full LM-GNN fusion model in the main path.

## 1. SimTeG-style LM+GNN

**Research purpose.** Use the finetuned LM hidden states as node text embeddings, then train/use a GNN over the original social graph. This anchors the project in the SimTeG-style two-step TAG pipeline rather than introducing a new LM-GNN foundation model.

**Method boundary.** We migrate SimTeG's idea that downstream-label finetuned LM representations can be fed into a GNN. We do not claim to reproduce SimTeG exactly, and we do not treat this stage as the novelty of the current research.

**Current code position.**

- `LLMbot/main.py`: CLI stage dispatch; `frozen_g0` preparation stage.
- `LLMbot/trainer.py`: `train_frozen_g0`, `load_frozen_g0`, `build_or_load_frozen_g0`, `StageRunner`.
- `LLMbot/model_building.py`: `PhaseAInputAdapter`, feature resolution and projection helpers.
- `LLMbot/utils/__init__.py`: artifact, checkpoint, data loading, path, and reproducibility helpers.

**Implementation status.** `implemented`, engineering-smoke verified. The former `frozen_g0.py` has been merged into the root mainline. Full TwiBot training was not run in the cleanup pass, so research performance is `not-yet-verified`.

**Input/output contract.**

- Input: user text, labels/splits for training, original graph tensors when `--use_GNN`, optional cached semantic embeddings through `--emb_path`.
- Output: frozen G0 artifact with `manifest`, `outputs`, `selection_metrics`, `checkpoint_path`, `artifact_dir`, plus temporary compatibility key `dir`.

**Current risk.** This is a preparation artifact. It should be described as SimTeG-style feature preparation and graph detector setup, not as a new contribution.

**Next acceptance check.**

```powershell
cd LLMbot
D:\Anaconda\python.exe main.py --stage frozen_g0 --dataset TwiBot-20 --seeds 1 --disable_wandb
```

Run only when data/GPU/runtime are available; otherwise retain compile/import/CLI checks as engineering smoke.

## 2. 2-hop Conformal Estimator

**Research purpose.** Turn GNN posterior uncertainty into a graph-aware error judge for the center node's local ego context. The estimator selects hard nodes and provides quality signals for later local refinement.

**Method boundary.** We migrate conformal prediction-set interfaces from CF-GNN and local nonconformity aggregation ideas from DAPS/NAPS and SNAPS. In the current project, the estimator is a post-hoc quality proxy and router signal; it does not claim a new conformal theorem or guaranteed coverage after graph rewriting.

**Current code position.**

- `LLMbot/estimators.py`: `GraphConformalSetEstimator`, `GNN2HopConformalEstimator`, `build_estimator`.
- `LLMbot/model_building.py`: metadata for `graph_conformal_set_estimator`, `gnn_2hop_conformal`, and `eqc_v8_stage1`.
- `LLMbot/parser_args.py`: `--estimator_mode graph_conformal_set_estimator|gnn_2hop_conformal|eqc_v8_stage1`.

**Implementation status.** `implemented` for the estimator classes and CLI mode registration; `not-yet-verified` for full experimental behavior. `GNN2HopConformalEstimator` records literature basis and a SNAPS code reference in metadata, but external official code has not been vendored.

**Input/output contract.**

- Input: GNN logits/posterior, labels and calibration split, original `edge_index`; optional graph information for 2-hop aggregation.
- Output: `prediction_sets`, `set_size`, `pred_label_score`, `coverage_margin`, `abstain_risk`, and estimator metadata.

**Current risk.**

- `abstain_risk` is a compatibility scalar derived from set size and margin. It must not be turned into an independent learned router without a new design.
- The method should be called `valid_cal-quantile-calibrated` or `post-hoc conformal-style` where assumptions are not satisfied.
- It does not use test labels for threshold fitting.

**Next acceptance check.**

```powershell
cd LLMbot
D:\Anaconda\python.exe main.py --stage estimator_matrix --estimator_mode gnn_2hop_conformal --dataset TwiBot-20 --seeds 1 --disable_wandb
```

Minimum artifact check: prediction sets and coverage margins exist in the estimator risk manifest for validation/test nodes.

## 3. Hard Node Router

**Research purpose.** Select only uncertain or locally poor-quality nodes for expensive local refinement and optional LLM enhancement.

**Method boundary.** We migrate GLANCE/LOGIN's selective processing idea, but the current main router is GNN-side and conformal-quality-driven. It is not a second bot classifier and not an LM-GNN fusion predictor.

**Current code position.**

- `LLMbot/estimators.py`: `abstain_risk`, prediction set, and margin outputs.
- `LLMbot/trainer.py`: estimator matrix, risk bundles, EQC v8 hard-node budget logic.
- `LLMbot/parser_args.py`: `--risk_budgets`, `--router_budgets` compatibility alias, `--eqc_v8_hard_node_budget`.

**Implementation status.** `partial`. The signal path exists, but the full hard-node selection artifact for the current research pipeline still needs a dedicated smoke run and manifest inspection.

**Input/output contract.**

- Input: conformal estimator output plus budget.
- Output: hard node ids, ranking/provenance, budget metadata.

**Current risk.**

- It is tempting to add a learned router head. Do not do that in the main path without a new design, because it would become another predictor.
- Existing residual-risk selector code may be useful for comparison, but the current research should keep the router deterministic or conformal-score-driven unless explicitly changed.

**Next acceptance check.** Add/verify an artifact field that records selected hard nodes, source estimator mode, and budget fraction without using test labels for fitting.

## 4. Local Retrieval / Rewriter

**Research purpose.** For hard nodes, construct a local candidate ego context that combines structure and text, then perform a conservative local graph rewrite.

**Method boundary.** We migrate SKETCH/Taming's semantic/structural context separation and GAugLLM/CTGL's text-graph coupled candidate logic. We do not implement full SKETCH decoupled aggregation training, GAugLLM contrastive augmentation, or CTGL coupled training in the main path. The first safe policy is existing-edge soft reweight plus virtual/evidence candidates.

**Current code position.**

- `LLMbot/operators.py`: `EgoRefinementRepairOperator`, `IterativeStructuralTextConfirmRetriever`, `EvidenceGraphRewriter`.
- `LLMbot/model_building.py`: metadata for `eqc_v8_stage4`.
- `LLMbot/parser_args.py`: `--repair_mode ego_refinement|eqc_v8_stage4` and EQC v8 retrieval/rewrite args.
- `LLMbot/trainer.py`: `_run_eqc_v8_matrix` imports and wires `IterativeStructuralTextConfirmRetriever` and `EvidenceGraphRewriter`.

**Implementation status.** `partial` and partly `ablation-only`. The v8/EQC classes exist and can be migrated into the current main research, but they should not be described as fully validated local graph rewriting. The older `ego_refinement` mode is a compatibility/scaffold surface.

**Input/output contract.**

- Input: hard node ids, original ego edges, GNN posterior/embedding, RoBERTa/SimTeG embeddings, local graph structure.
- Output: retrieval artifact, confirmed candidates, calibrated text similarity, rewritten existing ego edges, rollback/accept metadata.

**Current risk.**

- Edge roles and bucket rules may be over-manual if promoted as the core method. They should be recorded as artifact/provenance unless a learned or literature-faithful modifier is implemented.
- Binary reliability is occupied by BotBR/BECE and should not be the main claim.
- Soft rewrite should only modify existing propagation edges; new candidates should remain virtual/evidence context unless a later GSL/GCL branch is explicitly approved.

**Next acceptance check.** Run a bounded `eqc_v8_matrix --eqc_v8_block sanity` or repair-mode smoke and inspect that virtual candidates are not written back to the main graph.

## 5. Evidence Graph

**Research purpose.** Convert the refined local ego context into an evidence artifact that downstream code can consume either as graph data or as text for an LLM embedding model.

**Method boundary.** We migrate GraphText-style graph-to-text serialization only as evidence formatting. We also apply the TMLR graph-prompt analysis boundary: the project does not claim that LLMs perform faithful graph reasoning from prompts. The LLM reads a structured evidence paragraph/artifact.

**Current code position.**

- `LLMbot/operators.py`: `EvidenceGraphRewriter.build_evidence_graph`, serialization helpers, schema modes.
- `LLMbot/parser_args.py`: `--eqc_v8_schema_mode`, `--eqc_v8_token_budget`.
- `LLMbot/trainer.py`: `_run_eqc_v8_matrix` stage report and evidence graph plumbing.

**Implementation status.** `partial`. Evidence graph construction exists in v8/EQC scaffolding, including canonical and scramble modes, but it needs mainline artifact validation before being described as the final research artifact.

**Input/output contract.**

- Input: retrieval artifact, target payload, rewritten edge payload, quality scores, schema mode.
- Output: deterministic evidence graph artifact, serialized evidence text/prompt, schema metadata, token budget metadata.

**Current risk.**

- Fixed schema, ordering, Q66/Q33 buckets, or role labels are implementation hypotheses, not settled theory.
- Evidence graph should record enough provenance to audit why a node was selected and how its ego context changed.

**Next acceptance check.** Verify that canonical, shuffle-role, shuffle-order, collapse, and minimal schema variants produce distinct artifacts and that the canonical artifact respects the token budget.

## 6. LLM / Refiner

**Research purpose.** Use the evidence graph as LLM-consumable context and fuse the resulting evidence embedding with GNN/text representations for final prediction on hard nodes.

**Method boundary.** We migrate GLANCE and LOGIN's LLM-as-enhancer idea. The main path is embedding plus refiner, not LLM direct prediction. The project does not claim that the LLM is the final social-bot judge.

**Current code position.**

- `LLMbot/llm_evidence_refiner.py`: `EvidenceEncoderSpec`, `LLMEvidenceRefiner`, block-E encoder specs and acceptance reporting.
- `LLMbot/trainer.py`: `_run_eqc_v8_matrix` Stage-6 import and stage report.
- `LLMbot/parser_args.py`: `--eqc_v8_llm_backbone`, `--eqc_v8_schema_mode`.

**Implementation status.** `partial` and `not-yet-verified`. The scaffold supports encoder registration and linear-probe refiner logic. It does not include a verified local Qwen/Mistral embedding execution path in the current cleanup evidence.

**Input/output contract.**

- Input: post-rewrite GNN hidden state, RoBERTa/SimTeG embedding, evidence prompts, train/cal labels for fitting the refiner.
- Output: `z_evidence` embedding, linear-probe posterior, refiner metadata and acceptance report.

**Current risk.**

- No-LLM and RoBERTa serialization controls must remain in experiments.
- Any LLM call ratio, performance gain, or robustness claim requires manifest-backed experiment evidence.
- Direct LLM label prediction is a diagnostic baseline only and does not belong in the recommended main method.

**Next acceptance check.** Run `eqc_v8_matrix` with `--eqc_v8_llm_backbone none` first, then add cached/frozen encoder execution only after the artifact-only path is verified.

## Current Code Readiness Summary

**Implementation-boundary note.** Detailed active-module ownership is tracked
inside `LLMbot/docs/ARCHITECTURE.md`. Current refactor work is separating
compatibility surfaces from reusable owners: for example, shared distillation
gain/cost budget-curve, paired-bootstrap delta, and empty cost-report helpers
now live in `LLMbot/trainer_distillation_metrics.py`, while
`LLMbot/trainer_distillation.py` keeps trainer classes and graph-seed execution.
Split-safe pseudo-label training-index helpers now live in
`LLMbot/trainer_indexing.py`. These are engineering boundary cleanups only;
they do not add or validate a new research claim.

| Stage | Code status | Research status | Main blockers |
| --- | --- | --- | --- |
| SimTeG-style LM+GNN | implemented | preparation artifact | full training not rerun |
| 2-hop conformal estimator | implemented | main estimator candidate | needs estimator artifact smoke |
| hard node router | partial | main routing candidate | needs explicit hard-node artifact audit |
| local retrieval/rewriter | partial / ablation-only | candidate implementation | needs boundary cleanup and artifact validation |
| evidence graph | partial | likely central artifact | schema choices not finalized by evidence |
| LLM/refiner | partial | optional enhancer path | no verified local LLM embedding run |

## Next Development Goals

1. Validate `gnn_2hop_conformal` as the first research milestone: prediction sets, margin, and hard-node ranking must be correct and split-safe.
2. Produce a hard-node artifact contract before extending the rewriter.
3. Promote only one local retrieval/rewriter implementation into the main path; keep v8/EQC alternatives as ablation-only until validated.
4. Verify evidence graph serialization before any LLM embedding experiments.
5. Keep LLM/refiner optional until no-LLM evidence artifact and RoBERTa serialization controls are clean.

