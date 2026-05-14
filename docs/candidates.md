# Candidate and Weakly Related References

This document records papers, directions, and implementation ideas discussed during planning that are not currently strong mainline references. Strong mainline references are recorded in `docs/reference.md` and are not duplicated here as entries.

## Candidate Route: Scalar Calibration-Only Estimators

- **Why considered.** Scalar calibration is easier to implement than conformal prediction sets and can provide confidence scores for routing.
- **Why not mainline.** The current research needs an operational ego-quality signal, not only a calibrated probability. Scalar methods cannot directly express prediction-set size or coverage margin.
- **Upgrade condition.** If `gnn_2hop_conformal` fails to produce useful hard-node slices, scalar calibration can become a baseline or fallback router.
- **Difference from mainline references.** It lacks the prediction-set interface used by the current conformal estimator.

## Candidate Route: Full GCL/GSL After Rewriting

- **Why considered.** Modified graphs can be used for graph contrastive learning or graph structure learning after local rewriting.
- **Why not mainline.** It risks shifting the paper from post-hoc ego refinement into a broad training-time graph learning method.
- **Upgrade condition.** Only after local evidence artifacts prove useful should a separate branch test whether refined graphs improve GCL/GSL.
- **Difference from strong references.** Mainline keeps refined edges as local, budgeted, and reversible artifacts; GCL/GSL would train on modified views and needs a separate claim.

## Candidate Route: Learned Edge Role Head

- **Why considered.** A learned modifier could reduce hand-coded edge rules and better imitate structural-text coupling.
- **Why not mainline.** It introduces another trainable component and may become the real predictor unless carefully controlled.
- **Upgrade condition.** Promote only if deterministic/soft utility variants fail and counterfactual supervision can be split-safe and computationally feasible.
- **Difference from strong references.** Mainline currently treats role labels as artifact/provenance rather than the final method.

## Candidate Route: Four-Way Edge Roles

- **Why considered.** `supportive / suspicious / neutral / uncertain` roles preserve evidence that binary reliability deletion would remove.
- **Why not mainline.** The bins are easy to overfit as hand-crafted semantics without strong empirical support.
- **Upgrade condition.** Upgrade if ablations show role-aware evidence improves hard-node slices beyond continuous utility and binary reliability baselines.
- **Difference from strong references.** Strong social-bot reliability work motivates edge confidence, but not necessarily four fixed evidence roles.

## Candidate Route: LLM-as-Final-Predictor

- **Why considered.** Direct LLM labels from evidence prompts are easy to inspect and may be strong for selected hard nodes.
- **Why not mainline.** It would shift the contribution from graph-quality-guided refinement to LLM prediction and weaken graph-method claims.
- **Upgrade condition.** Keep only as diagnostic baseline. Promote only if the project intentionally changes to an LLM-predictor paper.
- **Difference from strong references.** Mainline follows LLM-as-enhancer / embedding-refiner framing.

## Candidate Route: Full-Graph Rewrite

- **Why considered.** Full graph structure learning could globally repair noisy social graphs.
- **Why not mainline.** It is expensive, less auditable, and too close to general graph structure learning rather than local hard-node refinement.
- **Upgrade condition.** Consider only after local ego refinement has clear benefits and full-graph compute/rollback risks are addressed.
- **Difference from strong references.** Mainline borrows soft/probabilistic edge thinking but keeps the operation local and post-hoc.

## Candidate Route: Binary Reliability Headline

- **Why considered.** Binary reliable/unreliable edges are intuitive and align with social-bot unreliable-edge literature.
- **Why not mainline.** This direction is already crowded in social bot detection, and bot-human heterophily may be evidence rather than noise.
- **Upgrade condition.** Use only as a baseline or ablation, not as the main novelty claim.
- **Difference from strong references.** Mainline should not erase suspicious edges; it should reduce propagation while preserving evidence.

## Candidate Social-Bot Baselines and Context Papers

These papers may help appendix positioning, but they are not direct mainline mechanisms for the current pipeline:

- **BotRGCN.** Useful base social-bot graph detector context; not a post-hoc local refinement method.
- **RGT / heterogeneous graph transformer bot methods.** Useful backbone context; not a conformal/rewrite/evidence pipeline.
- **BotSCL / social-bot contrastive learning.** Useful contrastive baseline context; not the current local post-hoc path.
- **BotMoE / multimodal expert fusion.** Useful multimodal fusion baseline context; not evidence-ego graph refinement.
- **RF-GNN / ensemble graph methods.** Useful robustness baseline context; not the current estimator-rewriter mechanism.
- **Dynamic bot graph methods such as BotDGT.** Useful future extension if temporal graphs enter scope; not current static ego refinement.

## Candidate TAG / LLM-Graph Methods Outside Mainline

- **LLM-as-GNN / graph vocabulary learning.** Interesting for foundation-style TAG models, but it changes the base architecture instead of refining a frozen detector.
- **LLM-generated text-attributed graphs.** Useful if the project needs synthetic node descriptions, but not the current social-bot ego repair path.
- **GraphRAG / knowledge-graph prompting systems.** Useful conceptual context for evidence serialization, but usually not node classification on social bot graphs.

## Candidate Engineering Work

- **Split `trainer.py`, `estimators.py`, and `operators.py`.** Needed for maintainability, but should be a separate interface-preserving refactor.
- **Delete or archive `LLMbot/baseline/` and `LLMbot/code/`.** Needed for repository hygiene, but separate from method implementation.
- **Remote GPU guide refresh.** Current remote notes may still point to deprecated paths; update only when the new root-mainline remote workflow is tested.

