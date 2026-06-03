# LLMbot Active Mainline

`LLMbot/` is the only active mainline for current bot-detection pipeline work.

## Entry Point

Run commands from this directory:

```bash
cd LLMbot
python main.py --experiment_task distillation_pipeline --dataset TwiBot-20 --seeds 1 --disable_wandb
```

Preferred public flags:

- `--experiment_task`
- `--graph_backbone`
- `--text_encoder`
- `--semantic_encoder`
- `--embedding_path`
- `--graph_data_variant`
- `--support_embedding_path`
- `--joint_refiner_embedding_path`

GLANCE prompt-cache precompute entry:

```bash
python precompute.py \
  --dataset TwiBot-20 \
  --prompt_mode glance_concat_ego_hop1_hop2 \
  --output_path datasets/TwiBot-20/glance_qwen3_prompt_cache.pt
```

Support-extended raw preprocessing:

```bash
python preprocess.py \
  --dataset_root ../datasets/TwiBot-20 \
  --mode artifacts
```

This helper is intentionally separate from `main.py`. It reuses the current
11826 labeled `norm_user_text.json` prefix, appends support-node texts from
`support.json`, rebuilds the full TwiBot-20 graph from the official
`edge.csv` with normalized `friend` / `follow` directions, and writes:

- `edge_new.json`
- `norm_user_text_new.json`
- `edge_index_new.pt`
- `edge_type_new.pt`
- `support_idx.pt`
- `preprocess_manifest_new.json`

It does not rewrite `labels.pt`, `train_idx.pt`, `valid_idx.pt`, or
`test_idx.pt`; support nodes are added to the node/semantic space while the
supervision split stays unchanged.

First-round full-graph support:

- `graph_data_variant=full_graph_support` is currently supported only for:
  - `graph_detector_prepare`
  - `estimator_ablation` (conformal-style router ablations only)
  - `joint_router_refinement`
- full-graph mode keeps supervision on the original labeled split only
- the graph comes from:
  - `edge_index_new.pt`
  - `edge_type_new.pt`
- the semantic tensor is built at runtime by concatenating:
  - labeled RoBERTa embeddings from `--embedding_path`
  - support RoBERTa embeddings from `--support_embedding_path`

Example full-graph Phase A smoke:

```bash
python main.py \
  --experiment_task graph_detector_prepare \
  --dataset TwiBot-20 \
  --graph_data_variant full_graph_support \
  --graph_backbone rgcn \
  --support_embedding_path datasets/TwiBot-20/support_roberta_embeddings_new.pt \
  --graph_detector_epochs 1 \
  --seeds 1 \
  --disable_wandb
```

Optional support-only RoBERTa encoding:

```bash
python preprocess.py \
  --dataset_root ../datasets/TwiBot-20 \
  --mode encode_support \
  --roberta_model_path /path/to/finetuned-roberta
```

Prompt-cache helper modes:

- legacy GLANCE-style:
  - `glance_ego`
  - `glance_hop1`
  - `glance_hop2`
  - `glance_concat_ego_hop1_hop2`
- relation-aware social-context:
  - `relation_aware_ego`
  - `relation_aware_1hop`
- prompt-expert bundles:
  - `expert_ego`
  - `expert_graph_following`
  - `expert_graph_follower`
  - `expert_tweet`
  - `expert_conflict`
  - `expert_concat_v1`

Prompt-family versioning:

- `--prompt_family_version v1` preserves the older mixed prompt-expert cache
  contract
- `--prompt_family_version v2` upgrades the expert path to a literature-aligned
  explanation-first family
- `expert_concat_v1 + --prompt_family_version v2` is the new four-expert
  mainline and writes `semantic_view_mode=prompt_expert_bundle_v2`
- v2 keeps `following` and `follower` split as separate experts, keeps raw
  metadata mostly out of prompt bodies, and uses encoder-aware output naming
  such as `glance_prompt_expert_concat_v2_roberta_finetuned_embed.pt`
- when `--model_path` is omitted in v2 expert modes, explanation embeddings now
  default to the SimTeG finetuned RoBERTa family
  (`--model_path roberta_finetuned`, resolved offline to
  `yzxjb/roberta-finetuned-20`) instead of `roberta-base`
- if the `roberta_finetuned` model source is not available as a local path or
  cached HF snapshot, precompute fails fast; it never silently substitutes
  `roberta-base`
- if a frozen SimTeG LM checkpoint is available at
  `TwiBot-20_seed_<seed>/checkpoints/LM_pretrain/best.pkl`, v2 loads that
  encoder state after resolving the real finetuned-RoBERTa source; use
  `--finetuned_roberta_checkpoint_path` to pin a specific checkpoint
- that default follows the frozen SimTeG text-consumption contract:
  `padding=True`, `truncation=True`, effective max length `512`, final
  hidden-state plain mean pooling, no added special tokens, and no l2
  normalization unless `--normalize true` is explicitly passed

Raw tweet source compatibility:

- `--tweet_source_mode {norm_user_text,raw_post_edges}`
- `--node_source_path`
- `--edge_source_path`
- `--tweet_sample_size`
- `--tweet_clean_level {light,norm_compatible}`
- `--tweet_keep_hashtag_surface`
- `--tweet_keep_emoji_surface`
- `raw_post_edges` is a precompute-only compatibility path:
  - it upgrades `expert_tweet`
  - and tweet-dependent `conflict` generation
  - while leaving `ego / graph_following / graph_follower` source handling
    unchanged
- raw mode uses:
  - `node_new.json` as the raw node object store
  - `edge_new.json` when present as the grouped `post`-edge cache
  - otherwise the authoritative `edge.csv` `post` relations
  - fallback to `norm_user_text(.json/.new.json)` when a user has no
    recoverable raw tweets

Relation-aware center-induced prompt-expert selection:

- `precompute.py` now supports
  `--neighbor_sampling_policy center_induced_relation_aware`
- it can build prompt-expert evidence from either:
  - `--context_graph_variant labeled`
  - `--context_graph_variant full_graph_support`
- center scope is explicit:
  - `--center_node_scope labeled`
  - `--center_node_scope all_graph_nodes`
- routed-node-only precompute is also supported through
  `--routed_nodes_path <file>`:
  - accepts `.jsonl`, `.json`, `.pt`, or plain-text node-id lists
  - interprets ids in graph-global node-index space
  - only those nodes go through prompt-expert building, explanation generation,
    and encoder inference
  - saved tensors are still scattered back to full-graph row shape with zeros on
    unselected nodes so `--joint_refiner_embedding_path` stays alignment-safe
- for `expert_*` prompt modes, each direction is split into:
  - support neighbors
  - contrast neighbors
- ranking is centered on cosine similarity between the center node and
  candidate neighbor embeddings, with deterministic tie-breakers from
  reciprocity, common-neighbor count, degree, and text length
- the output payload keeps the same expert-component contract
  (`ego/graph_following/graph_follower/tweet/conflict`) but upgrades the
  semantic marker to `prompt_expert_bundle_center_induced_v1` and adds
  selection statistics as scalar side channels

Example relation-aware 1-hop cache:

```bash
python precompute.py \
  --dataset TwiBot-20 \
  --prompt_mode relation_aware_1hop \
  --following_quota 3 \
  --follower_quota 3 \
  --output_path datasets/TwiBot-20/glance_qwen3_prompt_cache_relation_aware_1hop.pt
```

Example prompt-expert bundle cache:

```bash
python precompute.py \
  --dataset TwiBot-20 \
  --prompt_mode expert_concat_v1 \
  --prompt_family_version v1 \
  --explain_model_path /path/to/qwen-instruct \
  --output_path datasets/TwiBot-20/glance_prompt_expert_concat_v1_qwen3_embed.pt
```

Example literature-aligned prompt-expert bundle v2:

```bash
python precompute.py \
  --dataset TwiBot-20 \
  --prompt_mode expert_concat_v1 \
  --prompt_family_version v2 \
  --explain_model_path /path/to/qwen-instruct \
  --routed_nodes_path /path/to/routed_nodes.jsonl \
  --explain_batch_size 0 \
  --explain_max_new_tokens 128 \
  --explain_log_every 50 \
  --explain_quality_gate true \
  --project_name llmbot-precompute \
  --experiment_name prompt_expert_v2 \
  --output_path datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_embed.pt
```

Use `--model_path roberta_finetuned` or `--model_path /path/to/local/finetuned-roberta`
when you want to make the encoder choice explicit. Passing `--model_path
roberta-base` remains a valid ablation, but it is no longer the v2 default.
`--explain_quality_gate true` is the default for explanation-first expert runs:
it refuses to reuse empty, punctuation-only, or prompt-echo explanation sidecars
and regenerates those rows instead. For routed-node reruns after a bad cache,
prefer a new output stem or a clean `--explain_component_cache_dir` so the
quality gate cannot be bypassed by stale artifacts.

Example raw tweet smoke:

```bash
python precompute.py \
  --dataset TwiBot-20 \
  --prompt_mode expert_tweet \
  --tweet_source_mode raw_post_edges \
  --tweet_sample_size 6 \
  --tweet_clean_level light \
  --limit 8 \
  --output_path datasets/TwiBot-20/glance_prompt_expert_tweet_raw_post_edges_qwen3_embed.pt
```

Example center-induced prompt-expert bundle on the full graph:

```bash
python precompute.py \
  --dataset TwiBot-20 \
  --graph_data_variant full_graph_support \
  --context_graph_variant full_graph_support \
  --center_node_scope labeled \
  --prompt_mode expert_concat_v1 \
  --neighbor_sampling_policy center_induced_relation_aware \
  --selection_embedding_path datasets/TwiBot-20/embeddings_iter_-1_seed_1.pt \
  --support_selection_embedding_path datasets/TwiBot-20/support_roberta_embeddings_new.pt \
  --following_quota 3 \
  --follower_quota 3 \
  --output_path datasets/TwiBot-20/glance_prompt_expert_center_induced_qwen3_embed.pt
```

For prompt-expert explanation generation, `precompute.py` resolves
`--explain_model_path` in offline-first mode:

- if the argument points to a local snapshot path, it loads that model
- if the argument points to a local model parent directory with exactly one
  snapshot/model child, it resolves that child automatically
- if the argument is an HF repo id that already exists in the local HF cache, it
  resolves the cached snapshot
- if no local explain-model snapshot is available, it falls back to the built-in
  deterministic profile-card explanation instead of blocking on a HuggingFace
  download attempt
- pass `--explain_required` for real LLM-as-explainer experiments; this fails
  fast instead of writing a deterministic-fallback cache when the local explain
  model is missing or generation fails
- in `--prompt_family_version v2`, this explanation-first path applies to all
  four mainline experts: `tweet`, `graph_following`, `graph_follower`, and
  `conflict`
- Qwen model handling in `precompute.py` now follows the official model-card
  split explicitly:
  - `Qwen2.5-7B-Instruct` is treated as a chat-generation model via
    `apply_chat_template(..., add_generation_prompt=True)` and continuation is
    sliced from `output_ids[len(input_ids):]`
  - `Qwen3-Embedding-8B` is treated as an embedding encoder with
    left-padding-compatible `last_token_pool` plus optional l2 normalization
- long v2 runs are resumable at the component level:
  - each generated expert explanation is appended to
    `<output_stem>_<component>_explanations.jsonl`
  - reruns skip rows with matching `node_id` and prompt hash
  - `--explain_component_cache_dir` can move these sidecars away from the final
    cache directory
- the default `--explain_max_new_tokens` is `128`; raise it only when the
  explanation prompt genuinely needs longer output
- the default `--explain_batch_size 0` enables a conservative auto policy
  (`2` on CUDA, `1` on CPU); pass `4` manually only after confirming GPU memory
- `--explain_log_every` controls per-component progress logging and wandb
  updates during long generation loops

For the current local/server layout, `LLMbot/models/` is a Python package, not a
pretrained model snapshot. Use the actual instruct model snapshot path instead,
for example `../models/Qwen2.5-7B-Instruct` locally or
`/root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct` on the GPU server.

Prompt precompute wandb monitoring is optional. Passing `--project_name` starts
a wandb run and logs prompt-bundle construction, per-batch explanation progress,
per-component explanation completion, per-component encoder progress, and the
final artifact summary.
`--experiment_name` and `--wandb_run_name` control the run label, while
`--disable_wandb` forces no-op behavior for offline smoke checks.

Hidden compatibility aliases still parse for one migration window:

- `--stage`
- `--GNN_model`
- `--LM_model`
- `--semantic_backbone`
- `--emb_path`
- `--g0_feature_path`

Use canonical flags in new commands, docs, manifests, and analysis notes.

## Implementation Ownership Snapshot 2026-05-26

This round changed module ownership without changing the public CLI contract.

Current owner split:

- `artifact_contracts.py`: artifact/path/provenance contracts used by the active mainline
- `runtime_env.py`: device and CUDA runtime helpers
- `stage_runner.py`: shared `StageRunner` skeleton, dependency loading, provenance wiring, and top-level dispatch
- `trainer_graph.py`: shared graph bundle / provenance / edge-override rerun owner, plus public graph diagnostic executors
- `trainer_glance.py`: GLANCE owner for shared semantic wiring, public `joint_router_refinement`, and migrated internal GLANCE execution lanes
- `trainer_preparation.py`: real owner for `graph_detector_prepare` and `graph_calibration_prepare`
- `trainer_semantic.py`: real owner for `semantic_encoder_finetune`
- `trainer_distillation.py`: real owner for legacy distillation trainers and `run_legacy_graph_seed`
- `trainer.py`: thin compatibility facade only

Still transitional:

- `trainer_legacy_impl.py` still contains remaining GLANCE helper tails
  (especially prompt-expert / relation-aware helper clusters) and duplicate
  legacy blocks

Removed in this round:

- `stage_helpers.py` is no longer part of the active mainline

## Public Task Surface

The parser currently exposes these canonical public tasks:

- `distillation_pipeline`
- `semantic_encoder_finetune`
- `semantic_embedding_classifier`
- `graph_detector_prepare`
- `graph_calibration_prepare`
- `local_conformal_diagnostic`
- `local_conflict_prune_diag`
- `local_dignn_conflict_refine_diag` (paper-faithful DIGNN-style dual-view proxy: topology-view MLP + attribute-view MLP + attention fusion + MI objective)
- `joint_router_refinement`
- `minimal_pipeline`
- `estimator_ablation`
- `semantic_operator_ablation`
- `semantic_source_ablation`
- `repair_operator_ablation`
- `selector_ablation`
- `positioning_ablation`
- `backbone_stress_test`
- `appendix_ablation`

Internal-only implemented branches are not part of the public CLI contract:

- `glance_oracle_refinement_internal`
- `glance_full_graph_refinement_internal`
- `glance_counterfactual_router_internal`
- `glance_budgeted_refinement_internal`
- `glance_refiner_analysis_internal`
- `phase_a_single_cell_internal`

`joint_router_refinement` is the only public GLANCE-family task. It is still a
GLANCE-style implementation under the current cached semantic-embedding path,
not a full official-pipeline reproduction. The current public router path is
now best understood as a TwiBot20-oriented reliability adaptation under
GLANCE-style joint training:

- a lightweight auxiliary MLP `Q` still provides the router-side soft
  homophily proxy
- base-detector confidence features are temperature-scaled before router
  feature construction
- direction-aware social features are added for TwiBot20, including `in/out`
  degree, relation counts, directional homophily/disagreement, reciprocity,
  and sparse-node indicators
- the learned router scorer is now trained to rank `base_wrong` reliability
  rather than to fit the paper's utility / advantage target directly
- `oracle_advantage` remains recorded as a diagnostic quantity for routed-set
  analysis, not as the primary router supervision signal

The public strict stage now also enforces
same-run provenance: it must read the current run's own
`preparation/graph_detector` artifact, and its semantic tensor must match that
artifact's `feature_manifest.path` instead of mixing external backbone or
embedding roots across seeds. For prompt-cache ablations that should not alter
the base detector, `--joint_refiner_embedding_path` can override only the
refiner semantic branch while keeping the backbone provenance pinned to the
same-root `graph_detector_prepare` artifact.

Its public evaluation contract is now intentionally TwiBot20-adapted:

- training keeps the paper-text batch top-k routing schedule
- final evaluation ranks the whole split by `router_score`
- `--risk_budgets` defines the candidate global budgets
- the final budget is selected on validation only
- test is evaluated once under that locked validation-selected budget

This means the stage is paper-text aligned at training time, but no longer
paper-text aligned at final evaluation time, and its router objective is now
task-adapted rather than a pure paper-text GLANCE advantage router.

`joint_router_refinement` also exposes `--joint_train_node_cap` for controlled
comparisons between:

- `3000`: paper-style capped train subset
- `0`: TwiBot20-adapted full train split for router/refiner training

For routed-refiner diagnostics under the same strict joint stage, it also
supports:

- `--joint_refiner_explicit_gate`: add an explicit keep/change gate that mixes
  the frozen GNN path with the routed refiner path
- `--joint_refiner_target_mode {predict,keep_change}`: compare direct label
  prediction against explicit keep/change learning on routed nodes
- `--joint_refiner_gate_target {base_wrong,utility_positive}`: choose whether
  the explicit gate learns base-wrong routing or raw-refiner positive utility
- `--joint_refiner_weight_mode {off,base_wrong,utility_positive,base_wrong_plus_utility}`:
  routed-node loss reweighting toward base-wrong and/or oracle-utility-positive samples

The stage manifest records both the configured cap and the effective train-node
count so capped and full-train runs remain directly distinguishable.

The joint stage now also records per-epoch router diagnostics and routed-node
refiner fix/break deltas so the training trace can show whether the router or
the refiner saturates first.

For router-focused inspection, each `joint_router_refinement` seed directory now
also writes:

- `router_performance_summary.json`
- `router_performance_summary.csv`
- `router_epoch_curve.csv`
- `valid_budget_curve.csv`
- `test_budget_curve.csv`

When you run multiple seeds in one command, the experiment root also writes:

- `router_seed_summary.json`
- `router_seed_summary.csv`
- `router_budget_curves_all_seeds.csv`
- `router_epoch_curves_all_seeds.csv`

These files are the quickest way to inspect router AUROC/AUPRC, routed wrong
precision/coverage, selected budget, and cross-seed stability without manually
opening every seed-level `metrics.json`.

## Command Examples

```bash
# Distillation without graph execution
python main.py \
  --experiment_task distillation_pipeline \
  --dataset TwiBot-20 \
  --seeds 1 \
  --disable_wandb

# Distillation with graph execution
python main.py \
  --experiment_task distillation_pipeline \
  --dataset TwiBot-20 \
  --use_GNN \
  --graph_backbone botrgcn \
  --seeds 1 \
  --disable_wandb

# Semantic encoder finetune
python main.py \
  --experiment_task semantic_encoder_finetune \
  --dataset TwiBot-20 \
  --semantic_encoder roberta_finetuned \
  --seeds 1 \
  --disable_wandb

# Cached finetuned-RoBERTa embedding -> direct classifier
# Trains a lightweight MLP on the existing embedding tensor only.
python main.py \
  --experiment_task semantic_embedding_classifier \
  --dataset TwiBot-20 \
  --embedding_path datasets/TwiBot-20/embeddings_iter_-1_seed_1.pt \
  --seeds 1 \
  --disable_wandb

# Graph detector preparation artifact
# Default RoBERTa semantic regime: seed-aware pre-iter
# seed 1 -> embeddings_iter_-1_seed_1.pt
python main.py \
  --experiment_task graph_detector_prepare \
  --dataset TwiBot-20 \
  --use_GNN \
  --graph_backbone rgcn \
  --seeds 1 \
  --disable_wandb

# Iter-2 compatibility regime: explicit cross-seed compat branch
python main.py \
  --experiment_task graph_detector_prepare \
  --dataset TwiBot-20 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path datasets/TwiBot-20/finetuned_roberta_embeddings_iter_2_seed1.pt \
  --seeds 1 \
  --disable_wandb

# Graph calibrator preparation artifact
python main.py \
  --experiment_task graph_calibration_prepare \
  --dataset TwiBot-20 \
  --use_GNN \
  --graph_backbone rgcn \
  --seeds 1 \
  --disable_wandb

# Local conformal diagnostic
python main.py \
  --experiment_task local_conformal_diagnostic \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --seeds 1 \
  --disable_wandb

# Local conflict diagnostic
python main.py \
  --experiment_task local_conflict_prune_diag \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --conflict_router_budget 0.10 \
  --conflict_topk_per_bucket 1 \
  --seeds 1 \
  --disable_wandb

# DIGNN-style local conflict refiner
python main.py \
  --experiment_task local_dignn_conflict_refine_diag \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --local_dignn_conflict_router_budget 0.10 \
  --local_dignn_conflict_topk_per_bucket 1 \
  --seeds 1 \
  --disable_wandb

# Public GLANCE-style router/refiner task
# First run graph_detector_prepare for the same experiment root and seed.
# joint_router_refinement now inherits its semantic tensor from that
# preparation artifact and rejects cross-root backbone reuse.
# Final public evaluation selects a global routing budget from --risk_budgets
# on validation and locks that budget on test.
# Default RoBERTa regime again uses seed-aware embeddings_iter_-1_seed_{seed}.pt
python main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --risk_budgets 0.05,0.10,0.15,0.20,0.25 \
  --seeds 1 \
  --disable_wandb

# Iter-2 compatibility branch under the same backbone / refiner family
python main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path datasets/TwiBot-20/finetuned_roberta_embeddings_iter_2_seed1.pt \
  --risk_budgets 0.05,0.10,0.15,0.20,0.25 \
  --seeds 1 \
  --disable_wandb

# Full router experiment summary across seeds 1/2/3
# After the run, inspect router_seed_summary.csv/json at the experiment root.
python main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --risk_budgets 0.05,0.10,0.15,0.20,0.25 \
  --seeds 1,2,3 \
  --disable_wandb

# Full-graph conformal router ablation on top of a frozen SimTeG backbone
# Reuses an existing full-graph graph_detector_prepare root through --external_frozen_g0_root.
# Current full-graph estimator_ablation support is intentionally narrow:
#   - posthoc_calibrated_ranker
#   - calibrated_local_risk_router
#   - graph_conformal_set_estimator
#   - gnn_2hop_conformal
python main.py \
  --experiment_task estimator_ablation \
  --dataset TwiBot-20 \
  --graph_data_variant full_graph_support \
  --use_GNN \
  --graph_backbone rgcn \
  --estimator_mode posthoc_calibrated_ranker \
  --external_frozen_g0_root tmp_iterm1_forward_fullgraph/seed_1 \
  --experiment_name fullgraph_conformal_posthoc_seed1 \
  --seeds 1 \
  --disable_wandb

# Calibrated posterior + localized 1-hop relation-aware scalar risk ablation
# Same public estimator mode, but the current contract is the stronger v2 path:
# score-family selection uses valid_tune, conformal calibration uses valid_cal,
# and local risk can use node_repr similarity weights when available.
python main.py \
  --experiment_task estimator_ablation \
  --dataset TwiBot-20 \
  --graph_data_variant full_graph_support \
  --use_GNN \
  --graph_backbone rgcn \
  --estimator_mode calibrated_local_risk_router \
  --external_frozen_g0_root tmp_iterm1_forward_fullgraph/seed_1 \
  --experiment_name fullgraph_conformal_localrisk_seed1 \
  --seeds 1 \
  --disable_wandb

# Refiner-only prompt-cache override on top of an unchanged qwen3 backbone
python main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path datasets/TwiBot-20/qwen3_emb_last.pt \
  --joint_refiner_embedding_path ../remote_prompt_study_20260525/ego/glance_qwen3_prompt_cache_glance_ego.pt \
  --risk_budgets 0.05,0.10,0.15,0.20,0.25 \
  --seeds 1 \
  --disable_wandb

# TwiBot20-adapted full-train strict GLANCE
python main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --joint_train_node_cap 0 \
  --risk_budgets 0.05,0.10,0.15,0.20,0.25 \
  --seeds 1 \
  --disable_wandb
```

Prompt-cache experiment note:

- `precompute.py` writes the selected feature tensor under `payload["embeddings"]`.
- To use a prompt-cache run with strict GLANCE, rerun `graph_detector_prepare`
  with that exact `--embedding_path` inside the target experiment root first.
- `joint_router_refinement` will then reuse the same-root
  `preparation/graph_detector` artifact and reject detached prompt caches.
- For the default RoBERTa backbone family, omitting `--embedding_path` now
  resolves `datasets/TwiBot-20/embeddings_iter_-1_seed_{seed}.pt` per seed.
- `datasets/TwiBot-20/finetuned_roberta_embeddings_iter_2_seed1.pt` remains a
  supported explicit compatibility branch for pre-iter vs iter-2 comparisons.
- If the goal is to test prompt caches without changing the base detector,
  keep `graph_detector_prepare` on the original semantic tensor and pass the
  prompt cache through `--joint_refiner_embedding_path`; prompt payloads with
  `ego/hop1/hop2` are consumed as direct refiner views, while
  prompt-expert payloads with `ego/graph_following/graph_follower/tweet/conflict`
  are consumed through `semantic_view_mode=prompt_expert_bundle_v1`,
  `prompt_expert_bundle_center_induced_v1`, or `prompt_expert_bundle_v2` and
  kept strictly on the refiner branch instead of being fed back into the
  backbone.
- LLM-as-explainer execution is a `precompute.py` cache-construction concern,
  not a router feature. No router code changes are required unless a future
  experiment intentionally feeds explainer-derived signals into the router.
- `full_graph_support` now also accepts `--joint_refiner_embedding_path` for
  prompt-expert refiner-only overrides, as long as the payload is either:
  - graph-wide
  - or a labeled-prefix prompt-expert bundle aligned to the routed labeled
    nodes
- If a graph-wide prompt-expert payload is paired with a labeled-only frozen
  backbone through `--external_frozen_g0_root`, `joint_router_refinement` slices
  the prompt-expert components and `target_node_mask` to the labeled prefix
  before training the refiner.
- When a high-performing backbone artifact already exists, the same ablation
  can be run read-only through `--external_frozen_g0_root` together with
  `--joint_refiner_embedding_path`, so the prompt cache changes only the
  refiner branch and reuses the frozen backbone provenance unchanged.

Prompt-expert bundle v1 note:

- `expert_ego` uses a two-stage flow: generate an explanation from the
  structured profile card, then embed that explanation with Qwen. When no local
  explain-model snapshot is available, it falls back to the deterministic
  profile-card explanation path rather than attempting a blocking HF download.
- `expert_graph_following`, `expert_graph_follower`, `expert_tweet`, and
  `expert_conflict` stay embedding-only and write component tensors plus scalar
  side channels into one payload.
- `expert_concat_v1` shares the same offline-first explain path for its `ego`
  component and still concatenates `ego/graph_following/graph_follower/tweet/conflict`
  on output.
- `center_induced_relation_aware` upgrades the graph-side evidence for
  `expert_graph_following`, `expert_graph_follower`, `expert_conflict`, and
  `expert_concat_v1`:
  - each direction is partitioned into support and contrast subsets
  - sidecar JSONL rows record selected neighbor ids, support/contrast
    assignment, similarity summary, and fallback usage
- `joint_router_refinement` projects each expert view before routed-node
  refinement and uses a lightweight graph-view gate over
  relation-aware structural side channels. The base set remains
  `count_following/count_follower/has_following/has_follower`, and
  center-induced bundles append candidate-count, selected-count, mean-similarity,
  and reciprocal-ratio statistics.
  Prompt-expert refiner diagnostics now also record whether a node was actually
  eligible for prompt-based refinement, so routed-set ablations can separate
  "not selected" from "not eligible".
- When `--joint_refiner_explicit_gate --joint_refiner_target_mode keep_change`
  is used with `prompt_expert_bundle_v1`, the prompt-expert refiner also emits
  an abstain gate. `--joint_refiner_gate_target utility_positive` trains that
  gate to prefer the expert path only when raw refiner utility is positive;
  artifacts record gate rates and write per-node `gate_prob` / `gate_decision`.

Prompt-expert bundle v2 note:

- `prompt_expert_bundle_v2` is the literature-aligned explanation-first mainline
  for prompt experts in this repo.
- It keeps four experts on the semantic branch:
  - `tweet`
  - `graph_following`
  - `graph_follower`
  - `conflict`
- All four components follow `explanation -> embedding`:
  - a local instruct model writes a compact evidence-centered summary
  - the current finetuned RoBERTa encoder embeds that summary
- v2 prompt semantics are summary-first instead of label-first:
  - keep target-node cues relatively detailed
  - keep directed neighborhood evidence compressed
  - summarize consistencies, tensions, and missing evidence without asking the
    LLM to decide `human` or `bot` inside the cache text
- v2 explanation generation now reuses one loaded explain model for the whole
  precompute run instead of reloading it once per component.
- `graph_following` and `graph_follower` remain separate because the two
  directions represent different social evidence roles: who the account chooses
  to follow versus who chooses to follow it.
- v2 keeps prompt-side metadata compressed into natural-language-friendly cues
  (`account_age_bucket`, `follow_ratio_bucket`, `posting_density_bucket`,
  `verified/protected`, `bio_present`) while exact counts and rates stay in the
  refiner side channel. It also writes `metadata_structured`, a non-prompt
  structured metadata expert built from profile logs, ratios, missingness
  indicators, and bucket one-hots.
- The strict refiner still uses the same routed projector architecture, but it
  now accepts `prompt_expert_bundle_v2`, keeps the graph-fusion gate on
  directional count/presence features only, and appends the full scalar
  side-channel after the projected expert embeddings.
- Prompt-expert runs can choose the expert fusion head with
  `--joint_prompt_expert_fusion {projector_concat,mpe_gated,gaugllm_selector,raw_concat_single_graph_following,raw_concat_single_graph_follower,raw_concat_single_tweet,raw_concat_single_conflict,raw_concat_following_triplet,raw_concat_follower_triplet,raw_concat_metadata_anchor,raw_concat_metadata_only}`. The
  default `projector_concat` preserves the existing projected-concat refiner;
  `mpe_gated` keeps the existing refiner-only node-conditioned softmax over
  `graph_following`, `graph_follower`, `tweet`, `conflict`, and
  `metadata_structured`. `gaugllm_selector` is the strict runtime-reuse
  selector transplant: it reuses the routed prompt cache recorded by
  `--joint_refiner_embedding_path`, reads the adjacent
  `<cache_stem>_manifest.json`, loads the four explanation sidecars
  (`graph_following`, `graph_follower`, `tweet`, `conflict`) with duplicate
  `node_id` rows resolved by last-row-wins, encodes one selector-context text
  per routed node and expert with the same finetuned SimTeG RoBERTa line, and
  applies a dual-branch context-aware softmax over those four experts only.
  `metadata_structured` is not selectable in `gaugllm_selector`; it stays as a
  structured side-channel and as node context for selector-text construction.
  Both selector modes write per-node weights plus the selected expert into the
  stage outputs. `raw_concat_single_*` modes remove
  all expert projectors and `graph_fused`, then feed
  `[z_gnn || one_raw_expert || structural_side_channel]` directly into the
  routed-node MLP for expert-specialty diagnostics. `raw_concat_following_triplet`
  removes all expert projectors and `graph_fused`, then feeds
  `[z_gnn || graph_following || tweet || conflict || structural_side_channel]`
  directly into the routed-node MLP. `raw_concat_follower_triplet` does the
  same with `graph_follower` instead of `graph_following`, giving a clean
  direction-comparison baseline without the duplicated fused graph slot.
  `raw_concat_metadata_anchor` adds `metadata_structured` to the current
  follower-triplet anchor, while `raw_concat_metadata_only` isolates the
  structured metadata expert.
- When `--routed_nodes_path` is used, the prompt cache records the explicit
  target source in the manifest, but still writes a full-graph tensor layout so
  the strict refiner can consume the cache without a separate scatter step.
  These routed-targeted caches now also persist `target_node_ids` and a
  full-graph `target_node_mask`, and `joint_router_refinement` uses that mask to
  keep prompt-expert routing/application on the original routed set rather than
  inferring scope from zero-filled rows.

Frozen-router reuse note:

- `joint_router_refinement` now supports
  `--joint_routing_protocol frozen_router_reuse`
- when enabled, the stage reads a prior `joint_router_refinement` artifact
  through `--joint_router_reuse_root`
- the reused artifact supplies:
  - frozen router weights
  - router scaler
  - selected validation budget
  - selected beta
- this path is the claim-grade comparison mode for refiner-only evidence
  upgrades because it keeps the routed set fixed while changing only the
  routed-node evidence/refiner branch
- the standalone routed-node diagnostic script
  `scripts/routed_explain_qwen3_embedding_mlp.py` now supports two diagnostic
  modes:
  - `--diagnostic_mode routed_classifier`:
    - on-the-fly Qwen3 embedding generation through `--embedding_model_path`
    - or an existing prompt-expert cache through
      `--cached_embedding_path <cache.pt> --cached_embedding_key <tensor_key>`
  - `--diagnostic_mode frozen_gnn_feature_swap`:
    - explanation documents built from the routed-node expert sidecars
    - encoded with a RoBERTa-family text encoder through `--roberta_model_path`
    - then scattered back onto the routed node rows and replayed through the
      original frozen GNN weights with no retraining
- this keeps clean routed-node explain studies available in both forms:
  - routed-node local classifier evidence
  - frozen-backbone feature-replacement diagnostics bound to the same
    high-base SimTeG root

## Mainline Layout

Root-level modules are the default implementation surface:

- `main.py` - CLI entrypoint and canonical task dispatch
- `parser_args.py` - argument contract and compatibility normalization
- `stage_registry.py` - stage visibility and naming source of truth
- `trainer.py` - thin compatibility facade
- `trainer_legacy_impl.py` - current large implementation body during the migration window
- `stage_runner.py`, `trainer_preparation.py`, `trainer_semantic.py`, `trainer_graph.py`, `trainer_glance.py`, `stage_helpers.py` - new canonical module surfaces for continued extraction
- `estimators.py`, `operators.py`, `model_building.py` - estimator/operator/model construction
- `utils/` - artifact IO, manifests, metrics, calibration, data loading, and reproducibility helpers

## Artifact Naming

New writes use canonical namespaces:

- `seed_<n>/preparation/semantic_encoder`
- `seed_<n>/preparation/semantic_embedding_classifier`
- `seed_<n>/preparation/graph_detector`
- `seed_<n>/preparation/graph_calibrator`
- `seed_<n>/stages/<canonical_task_name>`

Legacy artifact locations such as `frozen/g0`, `frozen/gates/gats`, and old
stage names remain readable for compatibility, but they are no longer the
active naming surface.

## Current Mainline Notes

- `trainer.py` is already a thin compatibility facade, but most execution logic
  still lives in `trainer_legacy_impl.py`.
- `stage_runner.py`, `trainer_preparation.py`, `trainer_semantic.py`,
  `trainer_graph.py`, `trainer_glance.py`, and `stage_helpers.py` already act
  as canonical import surfaces, but most still forward into
  `trainer_legacy_impl.py` while extraction continues.
- `estimators.py` and `trainer_legacy_impl.py` remain the main refactor
  hotspots.
- `python main.py --help` is now intended to work as a parser-only check even if
  the training runtime is not fully installed.

## Refactor Snapshot 2026-05-24

- public task naming is now canonical and owned by `stage_registry.py`
- parser output includes canonical fields first and legacy shadow fields only
  for compatibility
- `main.py` resolves stage behavior through the registry instead of maintaining
  a separate public-task truth table
- preparation artifacts now write to canonical namespaces under
  `seed_<n>/preparation/`
- runtime-only helper artifacts are still in migration and may retain
  compatibility fallbacks
- strict GLANCE now derives its router-side soft local homophily signal from a
  lightweight auxiliary MLP `Q`, augments router features with GNN logit-based
  confidence signals, and trains its learned router scorer with continuous
  utility regression, pairwise ranking, and an auxiliary reliability objective
  rather than binary-only supervision; training uses a batch-size-32 top-k
  budget that decays from 32 to 8, while evaluation uses validation-selected
  global-budget routing
- strict GLANCE can now run either in the paper-style capped mode
  (`--joint_train_node_cap 3000`) or in a TwiBot20-adapted full-train mode
  (`--joint_train_node_cap 0`); manifests record the distinction explicitly
- public strict GLANCE now forbids cross-root `frozen_g0` reuse and requires
  its semantic tensor to match the current run's `graph_detector_prepare`
  feature provenance
- in the counterfactual GLANCE lane, `router_score` is the learned routing
  proxy used for deterministic top-k selection, while `oracle_advantage` is the
  post-hoc counterfactual reward trace (`loss_gnn - loss_refiner - cost`)
- LOGIN-style uncertainty routing remains explicitly bounded to hard-node
  selection only; manifests record `official_code_verified = false`,
  `repo_locally_verified = false`, and
  `verified_scope = node_selection_uncertainty_only`
- `joint_router_refinement` remains the only public GLANCE-family task; richer
  GLANCE branches are implemented but internal-only

## Code Governance Status 2026-05-24

The active mainline is now governance-aligned, but not yet structurally
finished.

Completed governance work:

- `LLMbot/` is the only active mainline
- canonical public tasks and canonical public flags are now the operator-facing
  contract
- hidden legacy aliases are isolated to a compatibility window instead of
  remaining the public interface
- internal GLANCE branches are explicitly separated from the public CLI surface
- the active code-development docs now describe the canonical mainline rather
  than deprecated `baseline/core` terminology

Still incomplete:

- most execution logic still resides in `trainer_legacy_impl.py`
- several extracted modules are currently boundary surfaces rather than true
  logic owners
- active internal code still carries a canonical-to-legacy compatibility layer
- `StageSpec` is only partially consumed as a runtime policy source
- runtime-only helper artifact naming is still in migration

This means the project has passed the public-contract cleanup phase, but has
not yet finished the implementation extraction phase.

## Next Refactor Plan

1. Make `trainer_preparation.py` the real owner of preparation logic.
   Move `load_frozen_g0`, `build_or_load_frozen_g0`, and
   `build_or_load_faithful_gats` out of `trainer_legacy_impl.py`, and move
   shared path/provenance helpers into `stage_helpers.py`.
2. Make `trainer_semantic.py` the real owner of semantic finetune execution.
   Move `run_semantic_finetune_seed` and its manifest/report helpers out of
   `trainer_legacy_impl.py`.
3. Finish the active parser-namespace migration inside code.
   Keep legacy flags parse-compatible, but make active mainline code read
   canonical fields such as `experiment_task`, `graph_backbone`,
   `text_encoder`, `semantic_encoder`, and `embedding_path`.
4. Consume `StageSpec` more uniformly at runtime.
   Replace remaining hand-written dispatch or gate special cases in `main.py`
   with registry fields such as `runner_kind`, `claim_grade_allowed`, and
   `forces_use_gnn`.
5. Extract graph and GLANCE execution ownership into
   `trainer_graph.py`, `trainer_glance.py`, and `stage_runner.py`.
   The goal is for those modules to own their execution branches directly,
   rather than re-exporting `trainer_legacy_impl.py`.
6. Canonicalize runtime-only helper artifacts after extraction.
   New helper outputs should stop writing migration-era naming where practical,
   while read compatibility remains in place for one transition window.

## Documentation Sync Rule

Every future code change in `LLMbot/` must update the matching code-development
docs in the same task.

Minimum mapping:

- parser or CLI changes: update `docs/code/parser.md` and this file
- dispatch, module-boundary, or extraction changes: update
  `docs/ARCHITECTURE.md` and `docs/code/research.md`
- maintainability or unresolved-risk changes: update `code.md`

Do not treat documentation sync as optional cleanup.

## Deprecated Surfaces

`baseline/` and `code/` are deprecated legacy directories scheduled for
deletion. Do not add implementation, tests, or documentation there unless the
task explicitly asks for migration, deletion, archival cleanup, or forensic
comparison.
