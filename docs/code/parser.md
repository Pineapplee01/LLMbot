# LLMbot Parser Reference

This document is the current parser contract and audit note for the active
mainline parser at `LLMbot/code/parser_args.py`.

It is an engineering reference. It is not experiment evidence and not a paper
claim record.

## Scope And Source

- active source: `LLMbot/code/parser_args.py`
- active caller: `LLMbot/code/main.py`
- operator entrypoint: `LLMbot/main.py` compatibility shim
- active implementation surface: `LLMbot/code/`
- deprecated reference surfaces: `LLMbot/baseline/`

Only `LLMbot/code/parser_args.py` is treated as the active parser contract.

## Canonical Naming Standard 2026-07-04

New public commands, docs, manifests, and governance records must use canonical
parser names. Legacy names remain parse-compatible only so old commands and
artifacts can be read during the migration window.

Canonical examples:

- use `--experiment_task`, not `--stage`
- use `--graph_backbone`, not `--GNN_model`
- use `--text_encoder`, not `--LM_model`
- use `--semantic_encoder`, not `--semantic_backbone`
- use `--embedding_path`, not `--emb_path` or `--g0_feature_path`
- use `--graph_detector_epochs`, not `--g0_epochs`
- use `fused_x` for final detector hidden state; mention `node_repr` only as a
  compatibility alias or an explicit control-space term
- use `graph_detector_prepare` as the public stage name; mention `frozen_g0`
  only as a legacy alias

`LLMbot/precompute.py` is not a `main.py --experiment_task` stage. It is a
compatibility entrypoint into `LLMbot/code/precompute.py`. Document that
implementation as the precompute bounded context for prompt, evidence,
embedding, and LLM-guide caches that later stages consume through explicit
path flags.

## Auxiliary Prompt-Cache Helper

`LLMbot/precompute.py` / `LLMbot/code/precompute.py` is a separate helper CLI,
not part of `main.py` parser dispatch, but it is part of the active GLANCE
operator workflow because it can materialize semantic prompt caches consumed later through
`graph_detector_prepare -> joint_router_refinement`.

Current public `main.py` selective-routing additions:

- `--joint_router_family {reliability_mlp,selectivenet}`
- `--joint_selectivenet_coverages`
- `--joint_selectivenet_alpha`
- `--joint_selectivenet_lambda`
- `--routed_nodes_path`
- `--finetuned_roberta_checkpoint_path`

Current public graph-representation comparison additions:

- `--mhlgc_contrast_space {fused_x,node_repr,x_new,low_high_concat,low_high,semantic}`
- `--conformal_knn_repr_source {fused_x,node_repr,x_low,x_new,x_high}`
- `--routed_contrast_family {none,low_high,three_view_control,supcon_class,hybrid,bot_edge_mask_human}`
- `--routed_contrast_gate {none,heuristic_reliable}`
- `--routed_contrast_weight`
- `--routed_contrast_temperature`
- `--routed_contrast_frozen_path`
- `--routed_highpass_mode {off,low_only,high_only,adaptive}`
- `--routed_highpass_target {logits,x_high}`
- `--routed_highpass_candidate_scope {relation_1hop,relation_1hop_plus_xnew_knn}`
- `--routed_highpass_max_neighbors`
- `--routed_highpass_loss_weight`
- `--routed_highpass_preserve_weight`
- `--routed_highpass_risk_gate_weight`
- `--routed_highpass_edge_role_weight`
- `--routed_highpass_risk_path`

Contract note:

- `fused_x` is now the preferred explicit name for the final detector hidden
  state exported by `graph_detector_prepare`.
- `node_repr` remains a compatibility alias for the same final hidden so older
  artifacts and older CLI invocations remain readable.
- Current `rgcn` and `botrgcn` frozen SimTeG artifacts also export
  forward-native `x_low` and `x_new=cat(x_low,x_in)` tensors. This lets router
  runs consume the same-forward relation-hidden construction space instead of
  reconstructing it later from legacy `node_repr`.
  HyperScan-aligned `x_new` runs now require that exported tensor; artifacts
  without `outputs["x_new"]` must be regenerated or used only in explicit
  final-hidden control runs.
- For clean same-protocol `SimTeG fused_x` vs `HyperScan fused_x` comparison,
  use `fused_x` explicitly in both router and MH-LGC flags instead of relying
  on the older `node_repr` alias.
- For HyperScan view-pair diagnostics, `low_high_concat` explicitly optimizes
  `cat(x_low,x_high)` before detector fusion; this requires a HyperScan-style
  backbone that exports both tensors and should not be confused with the
  construction-only `x_new` space.
- `--mhlgc_contrast_space` now defaults to `fused_x` in the active mainline.
  `node_repr` remains accepted as a compatibility alias for older manifests
  and scripts, but new routed-node MH-LGC runs should treat `fused_x` as the
  primary detector-space contract.
- `--routed_contrast_*` is a separate ablation namespace and must not be
  confused with legacy `--mhlgc_*`.
- `--routed_contrast_family` is only valid for `graph_detector_prepare`,
  requires full-batch training, and always restricts contrast to routed train
  nodes from `--routed_nodes_path`.
- `bot_edge_mask_human` is the asymmetric routed contrast variant: routed bot
  nodes are anchors, the same routed bot under edge masking is the positive,
  and routed human nodes are negatives in `fused_x` space.
- `--routed_contrast_gate heuristic_reliable` currently depends on routed
  multiview rows serialized by `precompute.py --prompt_mode mhlgc_llm_guide`
  and passed through `--mhlgc_semantic_embedding_path`; this gate is a
  training-side reliability filter, not an LLM-guided loss.
- `three_view_control` uses `--routed_contrast_frozen_path` only as a stop-grad
  frozen teacher aligned to `x_low`; it does not directly constrain `x_high`.
- `--routed_highpass_*` is a separate routed-only correction namespace. It
  applies after the detector fusion head in `fused_x` space, leaves non-routed
  rows on the base logits by default, and uses `--routed_nodes_path` for the
  forward application mask. Train labels supervise only routed train nodes.
- `--routed_highpass_risk_path` is optional. When supplied, its
  `risk_score`/`router_score` vector supervises correction utility and the
  high-pass gate; it is not interpreted as bot probability.
- `--routed_highpass_candidate_scope relation_1hop` is the clean default.
  `relation_1hop_plus_xnew_knn` is an ablation that augments relation-supported
  candidates with forward-native `x_new` KNN candidates, then ranks/truncates
  the merged pool by `x_new` similarity.
- `--graph_refine_mode {hyperscan_knn_hypergraph_proxy_augment, relation_overlap_knn_proxy_augment, relation_overlap_knn_repr_prefit_augment}`
  is now explicitly a diagnostic-control family. These modes can still be used
  for simple KNN augment comparisons, but they are not the active mainline for
  graph consumption.
- HyperScan-style mainline graph consumption now refers to
  `routed_dynamic_hyperscan_branch` or `hyperscan_neighborloader_batch_local_branch`
  together with `--graph_second_view_fusion multiattn` or
  `multiattn_adaptive`. For `graph_detector_prepare`, those second-view modes
  default to `multiattn` unless the caller explicitly overrides the fusion flag.
- Mainline representation roles are now fixed in manifests as:
  - `z_sem`: semantic input space
  - `z_construct = x_new`: high-order construction space
  - `z_pred = fused_x`: final detector space
  New runs should not silently reuse one task-shaped semantic tensor for all
  three roles unless they are marked as explicit controls.
- `--graph_second_view_candidate_policy llm_retain` is an offline
  LLM-guided KNN edge-retention variant for dynamic second-view branches. It
  requires `--graph_second_view_llm_edge_retain_path`, produced by
  `precompute.py --prompt_mode llm_knn_edge_retain_v1`. The cache stores
  target-centered non-routed KNN candidates and Qwen KEEP/DROP decisions; the
  graph detector consumes those decisions only as an `x_high` support-member
  filter before HGNN propagation. It is not a bot/human prediction cache and
  is intentionally not supported for NeighborLoader batch-local KNN.
- `--graph_second_view_candidate_policy router_support_transfer` is a narrow
  neighborhood-transfer diagnostic for dynamic second-view branches. It requires
  `--graph_second_view_router_support_path`, pointing to a conformal-router
  artifact that exports `support_group_payload.center_candidate_node_ids`. The
  graph detector then reuses those router-selected support rows directly as the
  per-center second-view candidate neighborhoods. This does not change routed
  center selection, detector fusion, or router score semantics; it tests only
  whether the adaptive router neighborhood can be directly consumed as the
  classifier neighborhood.

Current public `main.py` semantic correction-gate additions:

- `--semantic_gate_base_outputs_path`
- `--semantic_gate_candidate_output_paths`
- `--semantic_gate_candidate_names`
- `--semantic_gate_epochs`
- `--semantic_gate_hidden_dim`
- `--semantic_gate_learning_rate`
- `--semantic_gate_weight_decay`
- `--semantic_gate_break_weight`
- `--semantic_gate_threshold_policy`
- `--semantic_gate_selection_policy {threshold,defer_softmax}`
- `--semantic_gate_safety_policy {none,break_first}`
- `--semantic_gate_break_budget`
- `--semantic_gate_feature_family {probability,node_attribute,local_competence}`
- `--semantic_gate_local_k`

Current public staged selective-routing surfaces:

- `joint_router_refinement`
- `router_only_ablation`
- `semantic_correction_gate`

Contract note:

- `router_only_ablation` is a frozen-backbone routing diagnostic stage:
  - it requires an existing or same-root `graph_detector_prepare` artifact
  - it trains only the router on frozen SimTeG full-graph outputs
  - it reports router diagnostics and budget curves without refiner updates
- `--joint_router_family reliability_mlp` preserves the current
  reliability-first router MLP
- `--joint_router_family selectivenet` enables a SelectiveNet-style selective
  prediction router over the same reliability feature bundle
- `semantic_correction_gate` is a routed-node preparation stage that reads a
  frozen base `outputs.pt` plus one or more candidate semantic `outputs.pt`
  files, trains a small action-wise gate on routed train nodes, locks an
  accept/defer threshold on routed validation nodes, and evaluates only the
  resulting keep/change policy on routed test nodes. It is not a prompt
  generator and does not update the candidate LLM or MLP.
- `--semantic_gate_feature_family probability` keeps the original probability
  meta-feature gate. `node_attribute` appends target-level metadata/tweet cues
  parsed from `norm_user_text` and labeled-graph attributes loaded from the
  dataset, while preserving the same routed split and validation-locked
  threshold protocol. `local_competence` additionally computes per-candidate
  local action-competence features from top-k nearest routed train nodes only:
  local candidate correctness, base-wrong rate, fix rate, break rate, net
  estimate, and neighbor-similarity support. `--semantic_gate_local_k`
  controls this train-neighborhood size.
- `--semantic_gate_selection_policy threshold` preserves the legacy independent
  candidate BCE gates. `defer_softmax` changes the target to an action-level
  `{keep_base, accept_candidate_i}` learning-to-defer softmax. With
  `--semantic_gate_safety_policy break_first`, the stage also trains a
  per-candidate break-risk head and locks a break threshold on routed
  validation nodes. `--semantic_gate_break_budget` can constrain the validation
  correct-node break rate during threshold selection.

Current public helper flags:

- `--dataset`
- `--graph_data_variant`
- `--context_graph_variant`
- `--center_node_scope`
- `--routed_nodes_path`
- `--routed_nodes_split`
- `--text_path`
- `--output_path`
- `--project_name`
- `--experiment_name`
- `--wandb_run_name`
- `--disable_wandb`
- `--model_path`
- `--embedding_model_class`
- `--embedding_pooling_mode`
- `--finetuned_roberta_checkpoint_path`
- `--device`
- `--batch_size`
- `--seed`
- `--prompt_mode`
- `--prompt_family_version`
- `--residual_base_outputs_path`
- `--residual_prompt_variant`
- `--residual_include_neighbor_base_distribution`
- `--explain_prompt_style`
- `--neighbor_sampling_policy`
- `--selection_embedding_path`
- `--support_selection_embedding_path`
- `--tweet_source_mode`
- `--node_source_path`
- `--edge_source_path`
- `--tweet_sample_size`
- `--tweet_clean_level`
- `--tweet_keep_hashtag_surface`
- `--tweet_keep_emoji_surface`
- `--neighbor_cap`
- `--following_quota`
- `--follower_quota`
- `--max_length_ego`
- `--max_length_hop`
- `--normalize`
- `--save_dtype`
- `--limit`
- `--overwrite`
- `--explain_model_path`
- `--explain_trust_remote_code`
- `--explain_required`
- `--explain_batch_size`
- `--explain_max_input_length`
- `--explain_max_new_tokens`
- `--explain_log_every`
- `--explain_quality_gate`
- `--explain_component_cache_dir`
- `--ultratag_base_embedding_path`
- `--ultratag_neighbor_quota`
- `--ultratag_tau1`
- `--ultratag_tau2`
- `--ultratag_pagerank_ratio`
- `--ultratag_edge_reconfig_max_pairs`
- `--ultratag_edge_reconfig`
- `--ultratag_virtual_edge_policy`
- `--ultratag_virtual_edge_relation`
- `--prompt_mode llm_knn_edge_retain_v1`

Current prompt modes:

- `glance_ego`
- `glance_hop1`
- `glance_hop2`
- `glance_concat_ego_hop1_hop2`
- `relation_aware_ego`
- `relation_aware_1hop`
- `expert_ego`
- `expert_graph_following`
- `expert_graph_follower`
- `expert_tweet`
- `expert_conflict`
- `expert_concat_v1`
- `ultratag_s_subgraph_v1`
- `residual_audit_v1`
- `dgp_predictor_v1`
- `dgp_predictor_v2`
- `mhlgc_llm_guide`

Contract note:

- The helper always writes the downstream-consumed tensor under
  `payload["embeddings"]`.
- `--embedding_model_class causal_lm` plus
  `--embedding_pooling_mode causal_last_hidden_last_token` is the explicit
  LLaMA/Qwen/Mistral-style semantic-view path: `precompute.py` loads the local
  model with `AutoModelForCausalLM`, requests hidden states, and pools
  `outputs.hidden_states[-1]` at the last valid token. The default helper path
  remains `AutoModel` with encoder-aligned pooling.
- Prompt-expert bundle modes additionally write per-expert component tensors
  (`ego`, `graph_following`, `graph_follower`, `tweet`, `conflict`, and
  `metadata_structured` when available) and the scalar side channels needed by
  strict refiner routing analysis.
- `mhlgc_llm_guide` prompt caches now also persist routed-node explicit
  multi-view rows under the `.pt` payload when available:
  `following_view`, `follower_view`, `mutual_view`, and `semantic_knn_view`
  are stored per target node in addition to the encoded semantic tensor and
  `target_node_mask`. This is the active contract consumed by routed-only
  detector-space multiattention refinement.
  - in the direct non-explanation prompt path, `expert_ego` is now a
    BotSay-aligned `tweet + metadata` classifier prompt surface over the target
    account only
  - in the direct non-explanation prompt path, `expert_graph_following` and
    `expert_graph_follower` now use a narrower GLANCE-style
    `EGO + HOP1 + Category?` social-bot prompt shell instead of the earlier
    heavier social-hint encoder body
- `residual_audit_v1` is a routed-node correction-utility prompt-cache mode.
  It builds one `residual_audit` prompt per target node and encodes that prompt
  with `--model_path`. It does not generate explanations and does not alter the
  router/refiner protocol. `--residual_prompt_variant base_as_hypothesis`
  exposes the frozen SimTeG prediction from `--residual_base_outputs_path` as a
  fallible hypothesis, `no_base` hides base-model state, and
  `base_as_assertion` is an anchoring negative-control. True labels are never
  inserted into the prompt text. Each prompt row stores Qwen2.5-Instruct-style
  `system` and `user` messages plus the combined embedding text; the requested
  output is strict JSON over evidence label, correction action, confidence,
  short evidence arrays, and uncertainty/risk notes.
- `dgp_predictor_v1` is a routed-node DGP-style LLM-as-predictor prompt-cache
  mode. It builds one `dgp_predictor` prompt per target node and encodes that
  prompt with `--model_path`. `--dgp_prompt_variant
  target_fine_neighbor_coarse` keeps detailed target profile/tweet evidence and
  compresses following/follower context into ranked coarse cards and counts;
  `target_only` removes neighbor cards as an ablation. The requested output is
  now a minimal DGP-style `ASSISTANT_ANSWER: Yes|No` token rather than a JSON
  label wrapper. True labels and oracle fix/break outcomes are never inserted
  into prompt text. The prompt sidecar is the input surface for DGP-style Qwen
  PEFT predictor tuning, while the `.pt` embedding cache is the input surface
  for CALM-style query-embedding-MLP comparison.
- `dgp_predictor_v2` keeps the same downstream cache contract but changes the
  prompt construction to a more DGP-aligned norm-text summarization flow:
  `target norm_user_text` is the fine-grained tweet+metadata evidence; selected
  directional neighbors are summarized from their own `norm_user_text`; those
  summaries are compressed into relation-level context summaries; the final
  `dgp_predictor` prompt asks for exactly one answer token, `Yes` for bot or
  `No` for human. The mainline variant is
  `--dgp_prompt_variant norm_text_following_summary
  --dgp_neighbor_summary_k 5`; follower and following+follower variants are
  ablations. The mode uses `--explain_model_path` for the two summarization
  stages and writes resumable `dgp_neighbor_summary` /
  `dgp_context_summary` sidecars before encoding the final predictor prompt.
  The `norm_user_text` rows are not passed through as raw delimiter strings:
  they are parsed and rendered into `PROFILE`, `TWEET_BEHAVIOR`, and
  `TWEET_SAMPLES` sections, with special tokens, XML-like fragments, `@USER`,
  and `HTTPURL` cleaned before generation or final prompt encoding.
  Empty selected-neighbor contexts use a deterministic sparse-context summary,
  so the LLM is only called for relation-context summaries when selected
  neighbor evidence exists. DGP-v2 summary prompts require English output and
  tell the model to describe non-English/noisy source text as evidence quality
  rather than copying it. If a single generated row still fails the quality
  gate after bounded retries, it is recorded as `generation_mode =
  quality_fallback` with `quality_fallback_reason`; `--explain_required` still
  fails on model-load failure rather than silently building a fully
  deterministic cache. When this prompt sidecar is consumed by
  `semantic_encoder_finetune --semantic_encoder qwen3_peft
  --semantic_supervision_mode answer_token`, the active mainline now performs
  answer-token SFT over the final `ASSISTANT_ANSWER:` slot instead of only
  training a hidden-state classifier head.
- `--tweet_source_mode raw_post_edges` is a precompute-only compatibility path
  for `expert_tweet` and tweet-dependent `conflict` generation:
  - raw user/tweet nodes come from `node_new.json`
  - `post` linkage comes from `edge_new.json` when available, otherwise
    `edge.csv`
  - uncovered users fall back to the serialized `norm_user_text` path
- `--prompt_family_version v1` preserves the earlier mixed prompt-expert cache
  meaning and historical `*_qwen3_embed.pt` naming.
- `--prompt_family_version v2` upgrades the prompt-expert family to
  `semantic_view_mode=prompt_expert_bundle_v2`:
  - `expert_concat_v1` becomes a four-expert explanation-first bundle over
    `graph_following`, `graph_follower`, `tweet`, and `conflict`, plus a
    non-prompt `metadata_structured` component built from profile metadata
    logs, ratios, missingness indicators, and bucket one-hots
  - prompt text for legacy semantic prompts and v2/v3 explanation prompts is
    now centralized in `LLMbot/prompt.py`; `precompute.py` prepares evidence,
    fallbacks, sidecars, and embeddings, then calls those builders
  - explanation generation reuses a single loaded local explain-model runtime
    across all v2 expert components in the same precompute run
  - v2 summary outputs now follow a minimal analytic-template contract rather than
    fixed prose formatting:
    `Bot-like Evidence`,
    `Human-like Evidence`,
    `Uncertainty`,
    `View Leaning`,
    `Rationale`, and `Judgement`
  - `--explain_prompt_style {default,botsay}` further controls only the
    explanation-generation framing for v2/v3:
    - `default` keeps the local explanation-first wording
    - `botsay` uses BotSay-style task framing and
      `Label: bot or human` followed by `Explanation: ...`
    - this remains a prompt-style adaptation only; it does not import BotSay's
      labeled in-context examples or neighbor labels into the generated prompt
  - component filenames become encoder-aware, such as
    `glance_prompt_expert_concat_v2_roberta_finetuned_embed.pt`
  - if `--model_path` is omitted, v2 expert modes default to the SimTeG
    finetuned RoBERTa encoder alias
    `roberta_finetuned`, resolved offline to `yzxjb/roberta-finetuned-20`, not
    `roberta-base`
  - if the finetuned source is not present as a local model path or cached HF
    snapshot, precompute raises an error instead of substituting `roberta-base`
  - when available, v2 then loads the frozen SimTeG LM checkpoint from
    `TwiBot-20_seed_<seed>/checkpoints/LM_pretrain/best.pkl`; pass
    `--finetuned_roberta_checkpoint_path` to pin a specific checkpoint
  - the finetuned-RoBERTa default uses the same long-text contract as the
    frozen SimTeG semantic path: tokenizer padding, truncation at effective
    max length 512, no added special tokens, final hidden-state plain mean
  - `v2/v3` are the routed-only evidence path:
    - routed nodes should already be fixed through `--routed_nodes_path`
    - do not pass `--selection_embedding_path` or
      `--support_selection_embedding_path`
    - do not use `--neighbor_sampling_policy center_induced_relation_aware`
    - those KNN-coupled options belong only to
      `prompt_expert_bundle_center_induced_v1` and `mhlgc_llm_guide`
- `--prompt_family_version v3` keeps the v2 explanation-first tensor and
  sidecar contract but changes the generated text schema to source-grounded
  structured evidence cards:
  - `semantic_view_mode=prompt_expert_bundle_v3`
  - `manifest["evidence_schema"] = "structured_evidence_card"`
  - `expert_concat_v1` writes default stems such as
    `glance_prompt_expert_concat_v3_roberta_finetuned_embed.pt`
  - single-expert v3 caches include `_v3_` in their default names to avoid
    overwriting v2 summary-style caches
  - prompts forbid final human/bot labels, probabilities, confidence scores,
    recommendations, and base-model correction instructions
  - omitted `--model_path` follows the same SimTeG finetuned RoBERTa default as
    v2
    pooling, and no l2 normalization unless `--normalize true` is explicitly
    passed
  - manifests record prompt-family version, embedding encoder tag, prompt roles,
    component order, requested/effective component length budgets, encoder
    contract, and the explanation-generation provenance
- `--explain_model_path` is offline-first and must resolve to a local
  HuggingFace tokenizer + generation-model source for real LLM explanation generation:
  - when omitted, generation-based prompt modes default to
    `/root/workspace/LMbot/hf_models/Qwen3.5-9B`
  - Qwen3.5 conditional-generation snapshots are supported through the
    generation-loader fallback when the standard causal-LM auto class cannot
    load the checkpoint architecture
  - exact snapshot/model directories are accepted
  - parent directories with exactly one snapshot/model child are accepted
  - cached HF repo ids are accepted only when already present in the local HF
    cache
  - `LLMbot/models/` is not a valid explain-model path; it is the project model
    package
- `--explain_required` disables deterministic fallback and should be used for
  real LLM-as-explainer runs where fallback explanations would invalidate the
  experiment.
- `--explain_quality_gate` defaults to true and validates both resumable
  explanation sidecars and fresh LLM outputs. Empty, punctuation-only,
  special-token, repeated-placeholder, and prompt-echo explanations are
  rejected before reuse so a rerun cannot silently consume a corrupted sidecar
  as a successful explanation cache. Fresh LLM outputs that fail the gate use a
  bounded plain-English retry before `--explain_required` aborts the run.
- Wandb monitoring for `precompute.py` is optional:
  - `--project_name` enables wandb logging
  - `--experiment_name` and `--wandb_run_name` control the run label
  - `--disable_wandb` forces no-op behavior
  - long prompt-expert runs log prompt-bundle, per-batch explanation progress,
    explanation-component completion, encoding-component, and final artifact
    summary metrics
- v2 explanation generation is component-resumable:
  - each component appends rows to
    `<output_stem>_<component>_explanations.jsonl`
  - reruns skip rows with matching `node_id` and prompt hash
  - quality-gate failures are treated as pending rows and regenerated instead
    of being counted as resumed rows
  - `--explain_component_cache_dir` redirects these sidecars
  - final concat payloads also write component-only `.pt` caches for the
    selected experts
  - `--explain_batch_size 0` is the default auto policy: CUDA uses 2 and CPU
    uses 1; larger values remain explicit operator choices
- `--routed_nodes_path` optionally narrows precompute to an explicit routed-node
  subset:
  - accepts `.jsonl`, `.json`, `.pt`, or newline-delimited node-id files
  - ids are interpreted in graph-global node index space
  - this path is restricted to `expert_*` prompt modes and
    `ultratag_s_subgraph_v1`
  - prompt construction, explain generation, and encoder inference run only on
    the selected routed nodes
  - prompt-expert output tensors are still scattered back to graph row shape
    with zeros on unselected rows so downstream `joint_refiner_embedding_path`
    consumption stays contract-compatible
  - UltraTAG-S output tensors keep the row layout selected by
    `--graph_data_variant`; `--context_graph_variant` can independently expand
    the neighbor-text context to `full_graph_support`
  - payloads persist `target_node_ids` plus `target_node_mask`, so downstream
    code can stay bounded to the original routed-node set instead of inferring
    scope from zero-filled rows
- `--routed_nodes_split {all,train,valid,val,test}` selects a split from
  mapping-style routed-node files. Routed-node `ultratag_s_subgraph_v1`
  adaptation intentionally requires `--routed_nodes_split test`. Omit
  `--routed_nodes_path` and use `--center_node_scope all_graph_nodes` for the
  full-graph UltraTAG text-augmentation diagnostic. Files that store a flat
  `node_ids` array plus `split_counts` are interpreted in `train -> valid ->
  test` order.
- `--prompt_mode ultratag_s_subgraph_v1` is the bounded UltraTAG-S adaptation
  path:
  - requires `--ultratag_base_embedding_path`
  - runs on explicit routed test nodes when `--routed_nodes_path` is supplied,
    or on the selected `--center_node_scope` otherwise
  - for UltraTAG-S, `--graph_data_variant` defines the output embedding/edge
    universe consumed by graph replay, while `--context_graph_variant` defines
    the text/edge universe used only for target-plus-neighbor context
  - builds propagated target-plus-neighbor text from the target profile plus
    ranked following/follower cards
  - asks the local instruct model for summary, keywords, and a soft
    bot/human label
  - encodes the augmented text with the selected embedding encoder, normally
    the finetuned SimTeG RoBERTa line, and replaces the target rows in the base
    embedding tensor supplied by `--ultratag_base_embedding_path`
  - creates same-soft-label cosine virtual edges with `--ultratag_tau1` only
    when `--ultratag_virtual_edge_policy same_soft_label_cosine`; use
    `--ultratag_virtual_edge_policy none` to disable soft-label-derived graph
    rewiring
  - PageRank-selects a small subset of the target-induced context graph and
    optionally applies an LLM edge reconfiguration threshold `--ultratag_tau2`;
    use `--ultratag_edge_reconfig false` for text-only diagnostics
  - writes `<stem>_edge_index.pt`, `<stem>_edge_type.pt`, sidecars for prompts,
    augmentations, virtual edges, and edge-reconfiguration decisions, plus a
    manifest that states this is not the full UltraTAG-S dual-GNN
    structure-learning reproduction.
- Under `--neighbor_sampling_policy center_induced_relation_aware`, expert
  bundles additionally record:
  - candidate counts
  - selected counts
  - support/contrast similarity summaries
  - reciprocal ratios
  - sidecar JSONL provenance for selected neighbor ids and fallback usage
- Strict GLANCE provenance still requires the same-root
  `graph_detector_prepare` stage to be rerun with that exact `--embedding_path`
  before `joint_router_refinement`.

## Canonical CLI Surface

Preferred public flags:

- `--experiment_task`
- `--graph_backbone`
- `--text_encoder`
- `--semantic_encoder`
- `--embedding_path`
- `--graph_construct_embedding_path`
- `--graph_data_variant`
- `--support_embedding_path`
- `--joint_refiner_embedding_path`
- `--joint_routing_protocol`
- `--joint_router_reuse_root`
- `--routed_nodes_path`
- `--finetuned_roberta_checkpoint_path`
- `--semantic_text_source_path`
- `--semantic_text_field`

Semantic routed-node contract:

- `semantic_encoder_finetune` and `semantic_embedding_classifier` accept
  `--routed_nodes_path`
- the file must be a routed-node JSON artifact with:
  - `node_ids`
  - `split_counts.train`
  - `split_counts.valid`
  - `split_counts.test`
- semantic stages interpret these indices in canonical labeled order
  `train + dev + test`
- when present, routed train/valid/test replace canonical `train_idx`,
  `valid_idx`, and `test_idx` for those semantic stages
- `--semantic_text_source_path` optionally replaces `data["user_text"]` rows by
  `node_id` from a JSONL sidecar. Rows may use the configured
  `--semantic_text_field` or fall back to `prompt`, `user`, `text`, or `full`.
  This keeps full-graph row alignment while allowing DGP-style prompt sidecars
  to feed `semantic_encoder_finetune`; it does not change the split protocol.
- `--semantic_supervision_mode {classifier,answer_token}` applies to
  `semantic_encoder_finetune`:
  - `classifier` preserves the legacy hidden-state classifier-head CE path
  - `answer_token` upgrades the Qwen PEFT route to causal-LM supervision on
    the prompt's final `ASSISTANT_ANSWER:` slot and scores routed-node
    evaluation by `No` (human) versus `Yes` (bot) conditional completion
    likelihood
  - `answer_token` now enforces that the consumed prompt text ends exactly at
    the final `ASSISTANT_ANSWER:` slot; prompt-sidecar rows with trailing answer
    text or raw non-prompt text are rejected instead of being silently used
  - the answer-token path tokenizes that prompt body with
    `add_special_tokens=False` so the manual `Yes`/`No` suffix is appended onto
    the same explicit answer slot during both training and evaluation
  - `answer_token` currently requires `--semantic_encoder qwen3_peft`
- `--finetuned_roberta_checkpoint_path` optionally pins the SimTeG LM
  checkpoint used to initialize `--semantic_encoder roberta_finetuned`; when
  omitted, the stage searches for
  `TwiBot-20_seed_<seed>/checkpoints/LM_pretrain/best.pkl`

Hidden compatibility aliases still parse:

- `--stage`
- `--GNN_model`
- `--LM_model`
- `--semantic_backbone`
- `--emb_path`
- `--g0_feature_path`
- `--router_budgets`

The parser produces canonical fields:

- `args.experiment_task`
- `args.graph_backbone`
- `args.text_encoder`
- `args.semantic_encoder`
- `args.embedding_path`
- `args.graph_data_variant`
- `args.support_embedding_path`

Legacy shadow fields are backfilled for compatibility:

- `args.stage`
- `args.GNN_model`
- `args.LM_model`
- `args.semantic_backbone`
- `args.emb_path`
- `args.g0_feature_path`

## Current Public Task Surface

Current parser-exposed `--experiment_task` choices:

- `distillation_pipeline`
- `semantic_encoder_finetune`
- `semantic_embedding_classifier`
- `graph_detector_prepare`
- `graph_calibration_prepare`
- `local_conformal_diagnostic`
- `local_conflict_prune_diag`
- `local_dignn_conflict_refine_diag` (paper-faithful DIGNN-style dual-view proxy stage)
- `joint_router_refinement`
- `prompt_expert_quality_audit`
- `minimal_pipeline`
- `estimator_ablation`
- `semantic_operator_ablation`
- `semantic_source_ablation`
- `repair_operator_ablation`
- `selector_ablation`
- `positioning_ablation`
- `backbone_stress_test`
- `appendix_ablation`

Internal-only branches are implemented but not exposed as public task choices:

- `glance_oracle_refinement_internal`
- `glance_full_graph_refinement_internal`
- `glance_counterfactual_router_internal`
- `glance_budgeted_refinement_internal`
- `glance_refiner_analysis_internal`
- `phase_a_single_cell_internal`

Removed deprecated historical hidden-stage compatibility:

- `--stage eqc_v8_matrix` is rejected
- `--experiment_task eqc_v8_matrix` is rejected

## Refactor Naming Standard 2026-07-04

New commands and public docs use canonical parser names only:
`--experiment_task`, `--graph_backbone`, `--text_encoder`,
`--semantic_encoder`, and `--embedding_path`. Hidden renamed aliases such as
`--stage`, `--GNN_model`, `--LM_model`, `--semantic_backbone`, `--emb_path`,
and `--g0_feature_path` remain for historical command replay, but new code
should read canonical fields after `normalize_args()`.

Canonical task names must come from `StageSpec.canonical_name` in
`LLMbot/code/stage_registry.py`. `StageSpec` remains the compatibility class
name, but new orchestration code should use local names such as `task_spec`,
`requested_task`, and `execution_task` rather than introducing new
`stage_spec`/`execution_stage` usage. Hidden deprecated-only task values that are
not `StageSpec` aliases may be removed during medium cleanup; `eqc_v8_matrix`
is no longer accepted on either the public `--experiment_task` path or the
hidden `--stage` path.

The distillation owner cleanup does not change parser behavior. The public
entry remains `--experiment_task distillation_pipeline`, and legacy command
replay through parser aliases still normalizes before dispatch. The
implementation owner is `LLMbot/code/trainer_distillation.py`; any
`trainer_legacy_impl.py` distillation names are compatibility aliases, not a
second parser contract.

Preferred variable names for new parser-adjacent code:

- node sets: `target_node_ids`, `routed_node_ids`, `center_node_ids`
- masks and indices: `train_idx`, `valid_idx`, `test_idx`, `train_mask`,
  `valid_mask`, `test_mask`
- representations: `fused_x`, `x_low`, `x_new`, `x_high`,
  `semantic_embeddings`
- router signals: `risk_score`, `router_score`, `oracle_advantage`,
  `utility_reward`
- artifacts: `stage_dir`, `artifact_namespace`, `manifest`,
  `feature_manifest`

## Parser Function Reference

### `_parse_bool(value)`

Compatibility boolean parser used by hidden legacy flags such as
`--is_processed`.

### `normalize_args(args, raw_args=None)`

Post-parse compatibility layer.

Current normalization behavior:

- resolves requested task to canonical `args.experiment_task`
- backfills legacy shadow fields
- normalizes `embedding_path <-> emb_path <-> g0_feature_path`
- normalizes `risk_budgets <-> router_budgets`
- records explicit hidden-legacy usage in `args.deprecated_cli_flags`

Joint strict-GLANCE note:

- `--joint_refiner_embedding_path` is a public refiner-only override for
  `joint_router_refinement`.
- It does not change the same-root `graph_detector_prepare` backbone artifact.
- under `--joint_routing_protocol frozen_router_reuse`, this is the preferred
  fixed-router routed-only evidence/classifier comparison path
- It accepts either a plain `[num_nodes, d]` tensor or a prompt-cache payload
  with direct `ego/hop1/hop2` semantic views.
- It also accepts prompt-expert bundle payloads that expose
  `semantic_view_mode=prompt_expert_bundle_v1` or
  `semantic_view_mode=prompt_expert_bundle_center_induced_v1`.
- It now also accepts `semantic_view_mode=prompt_expert_bundle_v2` and
  `semantic_view_mode=prompt_expert_bundle_v3`, which keep the same
  projector-style prompt-expert refiner but treat the cache as an
  explanation-first four-expert bundle.
- In prompt-expert modes, the strict refiner projects
  `ego/graph_following/graph_follower/tweet/conflict/metadata_structured`
  separately, uses a small graph-fusion gate over directional count/presence
  features, and keeps the bundle on the refiner branch only.
- `--joint_utility_advantage_experiment {off,utility_gate_only,botmoe_selector_only,utility_gate_botmoe_selector}`
  is the public preset surface for GLANCE utility-advantage comparisons. The
  default `off` preserves existing behavior; the utility-gate paths rely on the
  explicit keep/change gate, while BotMoE selector paths keep expert selection
  separate from that keep/change decision.
- `--joint_refiner_gate_policy {soft_mix,hard_keep_change}` and
  `--joint_refiner_gate_threshold` control the public application policy for
  the explicit gate. `soft_mix` remains the default interpolation policy, and
  the threshold default is `0.5`.
- `--joint_prompt_expert_fusion {projector_concat,mpe_gated,gaugllm_selector,gaugllm_mope,botmoe_selector,utility_correction_moe,metades_selector,conflict_aware_correction_moe,raw_concat_ego_following,raw_concat_ego_follower,raw_concat_ego_following_follower,raw_concat_single_graph_following,raw_concat_single_graph_follower,raw_concat_single_tweet,raw_concat_single_conflict,raw_concat_follower_tweet,raw_concat_following_triplet,raw_concat_follower_triplet,ultratag_propagated_follower_triplet,raw_concat_metadata_anchor,raw_concat_metadata_only}`
  selects the prompt-expert fusion head. `projector_concat` is the
  compatibility default. `mpe_gated` keeps the existing node-conditioned
  softmax over `graph_following/graph_follower/tweet/conflict/metadata_structured`.
  `gaugllm_selector` is the strict runtime-reuse migration: it requires the
  routed prompt cache path from `--joint_refiner_embedding_path`, reads the
  adjacent `<cache_stem>_manifest.json`, loads the four explanation sidecars
  recorded under `component_explanation_sidecar_paths`, deduplicates by
  `node_id` with last-row-wins, encodes per-node selector-context texts with the
  finetuned SimTeG RoBERTa line from the cache manifest, and stores both the
  four selector weights and the selected expert in `outputs.pt` and
  `per_node_test.jsonl`. `gaugllm_mope` reuses that same cache and sidecar
  contract, but the fusion head follows the official GAugLLM
  `SimilarityAttentionMLP` default: projected expert content logits are added to
  per-expert content/context dot-product similarity and softmaxed with
  temperature `0.2`. `--joint_prompt_expert_mope_attention` is exposed for this
  mode and currently accepts only `similarity`. `--joint_prompt_expert_mope_temperature`
  makes that softmax temperature configurable for diagnostics while preserving
  `0.2` by default. `--joint_prompt_expert_mope_logit_norm {none,branch_zscore,combined_zscore}`
  adds explicit selector-collapse diagnostics/calibration: `none` is the
  official formula, `branch_zscore` normalizes content and similarity logits
  separately per node before summing, and `combined_zscore` normalizes the
  summed logits before temperature. `--joint_prompt_expert_mope_entropy_weight`
  and `--joint_prompt_expert_mope_load_balance_weight` add optional routed-node
  selector calibration losses for diagnostics: entropy regularization penalizes
  low per-node selector entropy, while load-balance regularization penalizes
  deviation between batch-mean selector weights and the current batch's
  expert-availability prior. MoPE runs also write `content_logits`,
  `similarity_logits`, `combined_logits`, `selector_logits`, and routed-test
  selector-distribution summaries to
  metrics and tensors to `outputs.pt`. `botmoe_selector` is the implemented
  sparse BotMoE-style selector for the GLANCE utility-advantage experiment line:
  it uses train-time noisy top-k gating and selects sparsely among only
  `graph_following`, `graph_follower`, `tweet`, and `conflict`. Its companion
  flags are `--joint_prompt_expert_botmoe_top_k`,
  `--joint_prompt_expert_botmoe_noisy_gating` /
  `--no-joint_prompt_expert_botmoe_noisy_gating`, and
  `--joint_prompt_expert_botmoe_aux_weight`, where the auxiliary term scales
  `cv_squared(importance) + cv_squared(load)` on routed training batches.
  `metadata_structured` remains a structured side feature and is not selectable
  in `gaugllm_selector`, `gaugllm_mope`, or `botmoe_selector`; `abstain` and
  `base` are also not selectable experts. The keep/base versus change/correct
  decision belongs to the explicit refiner utility gate, not the selector.
  The raw-concat family includes direct no-projector routed-node checks over
  `ego + graph_following`, `ego + graph_follower`, and
  `ego + graph_following + graph_follower` before the broader triplet and
  metadata-anchored variants.
  `utility_correction_moe` is the fixed-router correction MoE validation mode:
  it selects among `graph_follower`, `tweet`, `conflict`, and
  `metadata_structured`, trains a label head for each expert, and trains
  per-expert utility logits. `--joint_correction_moe_utility_target
  {loss_advantage,decision_gain,hybrid,net_gain}` controls that supervision:
  `loss_advantage` preserves `base_loss - expert_loss - beta > 0`,
  `decision_gain` uses `base_wrong && expert_pred_correct`, `hybrid` accepts
  either signal, and `net_gain` keeps fix actions positive while making
  base-correct/expert-wrong break actions explicit negative utility through
  `--joint_correction_moe_break_weight`. It uses the maximum utility probability
  as the keep/change abstain score when calibration is off, so it can run with
  `--joint_refiner_gate_policy hard_keep_change` without an extra explicit gate
  head. Its loss weights are
  `--joint_correction_moe_expert_weight` and
  `--joint_correction_moe_utility_weight`; `--joint_correction_moe_ranking_weight`
  and `--joint_correction_moe_ranking_margin` add within-node pairwise action
  ranking over fix/no-gain/break utility rewards. `--joint_correction_moe_gate_calibration
  {global_threshold,per_action_threshold}` fits threshold(s) on validation and
  applies the locked policy to test. Utility-correction runs write raw per-expert
  logits, losses, advantages, predictions, target/reward matrices, calibration
  metadata, and calibrated decisions into the stage artifacts without requiring
  post-hoc file surgery.
  `metades_selector` is a META-DES-style routed correction selector that shares
  those correction-head and utility-target flags, but selects among
  `graph_follower`, `tweet`, `conflict`, and a runtime-derived
  `follower_triplet` action. Its utility heads receive base/expert competence
  meta-features including confidence, margin, entropy, bot probability,
  disagreement, and probability gap. `metadata_structured` remains a side
  feature and is not selectable in this mode. The runtime `follower_triplet`
  action is derived from existing expert projections, so no prompt cache
  regeneration is needed; per-node artifacts include `selector_weights` and
  `selector_weight_follower_triplet`.
  `conflict_aware_correction_moe` is a BotMoE-inspired correction adaptation for
  the same routed-cache setup. It selects only the first-order prompt experts
  `graph_following`, `graph_follower`, and `tweet`, while `conflict` is used as
  a projected cross-view safety/context feature inside every action head. It also
  reuses the runtime explanation-context sidecars already required by
  `gaugllm_selector`/`gaugllm_mope`, but it applies the existing correction MoE
  utility targets, ranking loss, and validation-locked calibration instead of a
  plain classification selector.
  `raw_concat_single_*` modes are
  no-projector routed-node ablations that feed
  `[z_gnn || one_raw_expert || structural]` into the refiner MLP for
  expert-specialty diagnostics. `raw_concat_follower_tweet`,
  `raw_concat_following_triplet`, and `raw_concat_follower_triplet` are
  no-projector routed-node ablations that feed
  `[z_gnn || graph_follower || tweet || structural]`,
  `[z_gnn || graph_following || tweet || conflict || structural]`, or
  `[z_gnn || graph_follower || tweet || conflict || structural]` directly into
  the refiner MLP. `raw_concat_metadata_anchor` adds the structured metadata
  expert to the follower-triplet anchor. `ultratag_propagated_follower_triplet`
  is a legacy mean-propagation routed-node ablation, not an UltraTAG-S
  reproduction: it reuses the existing prompt-expert cache, performs one-hop
  mean propagation over the cached expert embeddings at runtime, and feeds
  `[z_gnn || ultratag_graph_follower || ultratag_tweet || ultratag_conflict ||
  structural]` to the same no-projector MLP. It remains only as a legacy
  comparison; the paper-aligned test-routed UltraTAG-S subgraph path is the
  separate `precompute.py --prompt_mode ultratag_s_subgraph_v1` helper.
  `raw_concat_metadata_only` isolates the structured metadata expert.
- `full_graph_support` now also accepts prompt-expert refiner-only overrides
  when the payload is either graph-wide, a labeled-prefix bundle aligned to
  the routed supervision prefix, or a routed-node-targeted cache that has
  already been scattered back into full-graph row layout by `precompute.py`.
  When a graph-wide prompt-expert payload is consumed with a labeled-only
  frozen backbone through `--external_frozen_g0_root`, the joint stage slices
  prompt-expert components and `target_node_mask` to the labeled prefix.
  Routed-node-targeted prompt caches keep their explicit target membership, so
  `joint_router_refinement` now applies prompt-expert routing only on
  `split_idx ∩ target_node_mask`.
- When paired with `--external_frozen_g0_root`, the public joint stage may
  reuse a read-only frozen backbone root for the same refiner-only ablation.
- `--joint_routing_protocol {joint_train,frozen_router_reuse}` now controls
  whether the public joint stage trains router+refiner together or reuses a
  prior frozen router artifact while training only the refiner.
- `--joint_router_reuse_root` points to the prior `joint_router_refinement`
  artifact used when `--joint_routing_protocol frozen_router_reuse`.
- `--prompt_expert_quality_cache_path` is the explicit cache input for
  `prompt_expert_quality_audit`. If omitted, that diagnostic stage falls back to
  `--joint_refiner_embedding_path`.
- `--prompt_expert_quality_reference_stage` points to a reference
  `joint_router_refinement` stage directory containing `manifest.json` and
  `outputs.pt`. If omitted, the diagnostic stage falls back to
  `--joint_router_reuse_root`.
- `--prompt_expert_quality_components` selects the comma-separated expert
  component list audited by `prompt_expert_quality_audit`; the default is
  `graph_following,graph_follower,tweet,conflict`.
- `--prompt_expert_quality_probe_C` and
  `--prompt_expert_quality_threshold_grid` control the lightweight balanced
  logistic-regression probes used for base-wrong and per-action utility
  separability. These probes are diagnostics only, not deployable refiner
  training.
- `graph_data_variant=full_graph_support` is a public graph-contract switch
  for the first full-graph rollout.
- In that mode, the public mainline currently supports only:
  - `graph_detector_prepare`
  - `estimator_ablation` for conformal-style router modes
  - `joint_router_refinement`
  - `local_conflict_prune_diag` for diagnostic-only support-augmented failure analysis
- the currently supported full-graph `estimator_mode` values are:
  - `posthoc_calibrated_ranker`
  - `calibrated_local_risk_router`
  - `conformal_knn_risk_router`
  - `graph_conformal_set_estimator`
  - `gnn_2hop_conformal`
- when `local_conflict_prune_diag` runs with `graph_data_variant=full_graph_support`,
  it keeps supervision and scoring on the labeled prefix but writes
  `error_migration`, `support_exposure`, `support_consistency`,
  `structural_shock`, `error_regime_shift`, and four failure-mechanism labels
  into the stage metrics/per-node artifacts
- `calibrated_local_risk_router` keeps the same public mode name but now maps
  to the v2 contract:
  - scalar post-hoc conformal base risk from the frozen GNN posterior
  - localized 1-hop relation-aware risk aggregation
  - optional `node_repr` similarity weighting when graph-wide hidden states are
    available
  - `tune_idx` for score-family selection and `cal_idx` for conformal
    calibration
- `conformal_knn_risk_router` reuses the same conformal split/threshold and
  stage-2 reporting contract as `calibrated_local_risk_router`, but replaces
  relation-neighbor aggregation with target-node KNN support-group risk
  features:
  - each target node is the KNN center; the selected KNN neighbors are evidence
    for that target node's risk, not the router outputs themselves
  - `--conformal_knn_k` controls the KNN neighborhood size
  - `--conformal_knn_candidate_scope` chooses `hyperscan_full`,
    `labeled_full`, `labeled_relation_1hop`, or `undirected_relation_1hop`
  - when the frozen G0 artifact was trained with a HyperScan-style second-view
    KNN branch, omitted `--conformal_knn_k`,
    `--conformal_knn_candidate_scope`, `--conformal_knn_repr_source`, and
    `--conformal_knn_local_calibration_scope`
    default to that artifact's Hyper KNN configuration; explicit
    `--conformal_knn_*` flags still override inheritance
  - the effective inherited-or-explicit router KNN config is written under
    `risk_manifest.calibration_metadata.conformal_knn_config.config_source`
  - `hyperscan_full` is the strict HyperScan-aligned support-group mode:
    paired with `--conformal_knn_repr_source x_new`, it estimates each labeled
    center's risk from the graph-wide feature-KNN group produced by
    HyperScan-style `x_low + x_in` similarity
  - in `hyperscan_full`, `k` is the HyperScan hyperedge size including the
    center node; the router consumes at most `k - 1` other members as support
    evidence for that center
  - the full-pool implementation uses exact batched `torch` top-k over
    normalized feature vectors, rather than high-dimensional cKDTree, so the
    method remains exact while completing on the full graph
  - KNN-updated centers are restricted to the labeled graph; support nodes keep
    base conformal risk and are not routed by the selected-node manifest
  - `--conformal_knn_repr_source` chooses the KNN representation space:
    `node_repr` uses the final detector hidden alias, `x_low` uses the
    exported relation-hidden view when present, and `x_new` consumes the
    exported `cat(x_low,x_in)` construction space. HyperScan-aligned `x_new`
    runs do not reconstruct this tensor from `node_repr`; regenerate older
    artifacts that do not export `outputs["x_new"]`
  - `--conformal_knn_target_top_n` controls how many top-ranked target-node
    diagnostics are stored; legacy `--conformal_knn_anchor_top_n` is still
    accepted as a hidden compatibility alias
  - `--conformal_knn_shrinkage_tau` controls shrinkage of KNN local risk back
    toward the base conformal risk
  - `--conformal_knn_ncp_lambda` controls the NCP-style localization
    temperature for `exp(-distance/lambda_L)` support weights, matching the
    weighting form used by the official `1995subhankar1995/NCP` implementation
  - `--conformal_knn_neighbor_mode` enables router-only, non-model KNN support
    filtering with `standard`, `mutual`, `threshold`, `adaptive`, or
    `mutual_adaptive`; these modes only filter support evidence and do not
    train a router head or alter classifier logits
  - `--conformal_knn_similarity_threshold` is the cosine-similarity floor used
    by threshold/adaptive support filtering; values `<= -1` disable this filter
  - `--conformal_knn_min_support` makes sparse or weak filtered neighborhoods
    fall back to target-only/global local-calibration risk instead of forcing
    noisy KNN evidence
  - `--conformal_knn_adaptive_max_k` lets adaptive/threshold modes fetch a
    larger candidate pool before filtering while preserving the same
    non-modelized score-family contract
  - `--conformal_knn_hubness_correction {none,degree}` optionally downweights
    high-degree feature-space hub neighbors in NCP support weights
  - `--conformal_knn_learning_mode fixed` preserves the historical
    validation-selected fixed score families; `ncp_local` computes
    target-specific conformal features from KNN calibration neighbors weighted
    by `exp(-distance/lambda_L)`, then selects among nonparametric NCP-local
    score families on the tune split by AUPRC-error first without fitting a
    logistic head; `learned_logistic` keeps the same target-node KNN/conformal
    feature bundle but replaces the fixed family scorer with a balanced
    logistic residual-risk ranker fit on train labels. This is router-only: it
    does not alter classifier logits, does not consume LLM outputs, and should
    be compared against `fixed` / `ncp_local` under matched frozen-G0, split,
    and budget settings
  - `--conformal_knn_local_calibration_scope independent_knn` preserves the
    previous NCP-local behavior by querying calibration neighbors separately in
    the same representation space; `same_hyperedge` reuses the target-centered
    HyperScan KNN hyperedge itself and scores directly on the selected-K
    members contained inside that local group
  - `--conformal_knn_score_family_override auto` preserves validation-tune
    score-family selection, while `base_only` gives the strict target-only
    control and `ncp_local_conformal`, `ncp_local_margin`, or
    `ncp_knn_weighted_mean` force direct NCP-local evidence baselines for
    target->KNN support-group ablations. When
    `--conformal_knn_local_calibration_scope same_hyperedge` is active, `auto`
    now selects from same-hyperedge tail-risk families, including
    calibration-only inner-tail variants, rather than forcing a
    local-conformal-only family.
    The current stabilization path also includes an ESS-shrunk
    calibration-tail variant that interpolates back toward the broader
    same-hyperedge global tail when the local calibration support is sparse.
    Treat `same_hyperedge_calibration_shrunk_tail` as the fixed
    same-hyperedge anchor family for future ablations; `auto` remains an
    exploratory selector, not the anchor definition.
  - manifests write `selected_nodes` for the default budget,
    `selected_nodes_by_budget` for the full budget sweep, and
    `top_ranked_targets` for diagnostics. `top_ranked_anchors` remains a
    deprecated alias for old artifact readers, but the routed objects are
    target nodes.
  - this manifest is the required target-node baseline for future router
    comparisons; new router modes should report matched frozen G0, split,
    budget, labeled-only scope, AUROC-error, AUPRC-error, AURC, ErrRecall@K,
    Precision@K, Lift@K, and selected-node overlap against it
  - future router schemes should be evaluated as target-node risk identifiers:
    they may consume target->KNN support-group evidence, but claims are about
    ranking target nodes for LLM/refiner intervention
- `--support_embedding_path` points to the support-node RoBERTa embedding used
  for runtime concatenation with the labeled `--embedding_path` tensor.

### `parser_args(argv=None)`

Builds the parser, parses either `argv` or `sys.argv[1:]`, then normalizes the
namespace.

## Parser Parameter Audit 2026-05-26

### What is aligned

- canonical task names are now the public parser choices
- canonical flag names exist for backbone, text encoder, semantic encoder, and
  embedding path
- hidden aliases preserve historical commands
- deprecated `eqc_v8_matrix` is blocked on both public and hidden entry surfaces
- parser now emits canonical task fields while preserving compatibility shadow
  fields

### What is still intentionally transitional

1. internal execution still contains compatibility reads of legacy shadow fields
2. some hidden flags remain for historical command replay
3. `StageRunner`, graph-aware branches, and GLANCE branches are still being
   extracted from `trainer_legacy_impl.py`

## Parser Status Snapshot 2026-05-26

- `LLMbot/main.py --help` is intended to work through a parser-only fast path
- public `--experiment_task` choices now come from the canonical stage registry
- hidden `--stage` remains a compatibility entry surface for historical
  renamed task aliases, but deprecated-only `eqc_v8_matrix` is no longer
  accepted
- the parser is still ahead of the remaining execution-body cleanup: public
  naming is already cleaner than some internal compatibility reads
- this round did not change the parser surface; it changed execution ownership:
  `stage_helpers.py` was removed and preparation/semantic execution moved to
  dedicated owner modules

### Terms used in this audit

- `compat alias`: accepted for historical commands, hidden from the active help
  surface
- `deprecated-unwired`: still parsed for compatibility but not part of the
  intended active mainline workflow
- `inactive matrix placeholder`: retained parser slot for historical scaffolds,
  hidden from the active interface
- `rename-required`: old naming still present in runtime or artifact internals,
  even if public CLI is already canonical

## Argument Families

### Core routing

- `--experiment_task`

### Experiment metadata

- `--project_name`
- `--experiment_name`
- `--artifact_root`
- `--disable_wandb`
- `--reuse_existing_artifacts`
- `--force_retrain_backbone`
- `--claim_grade`

### Dataset and batching

- `--dataset`
- `--lm_batch_size`
- `--gnn_batch_size`
- `--raw_data_filepath`
- `--reset_split`

Hidden compatibility / deprecated-unwired:

- `--batch_size_LM`
- `--batch_size_GNN`
- `--batch_size_MLP`
- `--is_processed`

### Graph backbone and graph inputs

- `--graph_backbone`
- `--use_GNN`
- `--n_layers`
- `--hidden_dim`
- `--n_relations`
- `--activation`
- `--gnn_optimizer`
- `--gnn_dropout`
- `--att_heads`
- `--graph_second_view_scope`
- `--graph_second_view_candidate_scope`
- `--graph_second_view_training_geometry`
- `--graph_second_view_hypergraph_backend`
- `--graph_second_view_fusion`
- `--hyperscan_detector_style`
- `--SimpleHGN_att_res`
- `--RGT_semantic_heads`
- `--graph_refine_mode`
- `--graph_refine_budget`
- `--graph_refine_knn_k`
- `--mhlgc_enable`
- `--mhlgc_semantic_embedding_path`
- `--mhlgc_loss_weight`
- `--mhlgc_beta`
- `--mhlgc_gamma`
- `--mhlgc_temperature`
- `--mhlgc_feature_mask_probability`
- `--mhlgc_edge_mask_probability`
- `--mhlgc_hyperedge_mask_probability`
- `--mhlgc_anchors_per_batch`
- `--mhlgc_negative_count`
- `--mhlgc_positive_label`
- `--local_conf_disable_degree_guard`
- `--local_conf_similarity_gate_threshold`
- `--conflict_router_budget`
- `--conflict_topk_per_bucket`
- `--conflict_disable_degree_guard`
- `--local_dignn_conflict_router_budget`
- `--local_dignn_conflict_topk_per_bucket`
- `--local_dignn_conflict_disable_degree_guard`
- `--external_graph_edge_index_path`
- `--external_graph_edge_type_path`
- `--external_frozen_g0_root`
- `--risk_budgets`

Contract note:

- `--graph_refine_mode hyperscan_knn_hypergraph_proxy_augment` is the original
  HyperScan-inspired full-graph proxy:
  - it builds feature-KNN groups from the current Phase-A semantic tensor
  - it writes those groups back as bidirectional center-neighbor proxy edges
    under a new relation id
  - it is a similarity-grouping proxy only, not a full HyperScan HGNN branch
- `--graph_refine_mode relation_overlap_knn_proxy_augment` is the current
  evidence-driven local follow-up:
  - center nodes come from `--routed_nodes_path` when provided, otherwise from
    the labeled prefix
  - `--routed_nodes_split {all,train,valid,val,test}` selects which routed-node
    subset supplies those centers
  - KNN candidate search is restricted to the center's undirected relation
    1-hop neighborhood rather than the whole graph
  - `--graph_refine_candidate_scope labeled_relation_1hop` further restricts
    that local neighborhood to labeled-prefix nodes only
  - cosine similarity is used only to rank relation-supported candidates inside
    that local ego neighborhood
  - the emitted relation is therefore an overlap-style proxy for
    "semantic similarity supported by existing relation structure", not an
    unconditional global semantic star expansion
- `--graph_refine_mode relation_overlap_knn_repr_prefit_augment` is the
  closer HyperScan-style local follow-up:
  - it keeps the same routed-center and relation-ego candidate restriction as
    `relation_overlap_knn_proxy_augment`
  - but before KNN selection it fits a relation-view GNN on the original graph
    and uses `node_repr + g0_input` as the similarity feature space
  - this is the active-mainline proxy for HyperScan's
    `x_low + x_in -> feature-kNN hypergraph` idea, while still materializing
    the result as proxy edges instead of adding an HGNN branch
- `--graph_second_view_scope {auto,none,labeled_prefix,routed_nodes,neighborloader_batch}`
  is the canonical scope switch for the HyperScan-style training-time second view:
  - `auto` preserves the legacy `--graph_refine_mode`
  - `none` maps to `--graph_refine_mode none`
  - `labeled_prefix` maps to a labeled-prefix dynamic branch
  - `routed_nodes` maps to `routed_dynamic_hyperscan_branch` and requires
    `--routed_nodes_path`
  - `neighborloader_batch` maps to `hyperscan_neighborloader_batch_local_branch`
  - model-side second-view metadata must name `x_low_plus_x_in_dynamic_forward`;
    final `node_repr` is rejected for this dynamic branch and remains only an
    explicit legacy/control representation outside the HyperScan-aligned flow
- `--graph_second_view_candidate_scope {auto,undirected_relation_1hop,labeled_relation_1hop,labeled_full,hyperscan_full}`
  is the canonical KNN candidate switch for second-view grouping. `auto`
  preserves `--graph_refine_candidate_scope`.
  - `undirected_relation_1hop` and `labeled_relation_1hop` keep the local
    relation-supported candidate contract.
  - `labeled_full` and `hyperscan_full` are diagnostic routed-support pools
    for testing whether hard nodes need stable non-routed support outside their
    relation ego neighborhoods.
- `--graph_second_view_candidate_policy {default,exclude_routed,post_topk_exclude_routed,stable_quota,mixed_quota,llm_retain,router_support_transfer}`
  is the dynamic second-view member-selection ablation. `default` keeps ordinary
  `x_new` top-k; `exclude_routed` removes routed/high-risk candidates before
  filling top-k from the remaining pool; `post_topk_exclude_routed` first
  retrieves ordinary top-k and then drops routed/high-risk members without
  refilling; `stable_quota` reserves low-risk non-routed support; `mixed_quota` reserves
  hard/stable/counterfactual buckets before filling remaining members by
  current `x_new` similarity; `llm_retain` consumes an offline LLM retain/drop
  cache; `router_support_transfer` replaces the dynamic candidate row with the
  conformal router's exported support row. Quota policies require a label-free
  full-graph risk vector via `--graph_second_view_candidate_risk_path`;
  `mixed_quota` also requires `--graph_second_view_candidate_pred_path` so
  counterfactual support is defined by base-prediction disagreement, not by
  labels.
- `--graph_second_view_candidate_routed_nodes_path` optionally supplies the
  routed/high-risk mask used only for second-view candidate-member filtering.
  It decouples member filtering from `--routed_nodes_path`, which still selects
  routed centers under `--graph_second_view_scope routed_nodes`.
- `--graph_second_view_router_support_path` optionally supplies the exported
  conformal-router support payload used only by
  `--graph_second_view_candidate_policy router_support_transfer`.
  In `--graph_second_view_scope neighborloader_batch`, the same mask is mapped
  onto each sampled subgraph through `batch.node_id`; only `default`,
  `exclude_routed`, and `post_topk_exclude_routed` are supported there. This
  keeps the NeighborLoader seed nodes and batch-local KNN geometry unchanged
  while changing only the selected hyperedge members.
- `--graph_second_view_training_geometry {auto,full_batch,neighbor_subgraph}`
  is the canonical graph-detector geometry switch. `neighborloader_batch` scope
  requires the `neighbor_subgraph` geometry and uses `--graph_neighbor_num_neighbors`
  as the per-layer fanout.
- `--graph_neighborloader_contract {seed_only,hyperscan_sampled_subgraph}`
  controls how NeighborLoader sampled-subgraph batches are supervised and scored:
  - `seed_only` is the default active-mainline contract and keeps supervision plus
    validation/test aggregation on the first `batch.batch_size` seed rows only
  - `hyperscan_sampled_subgraph` is the explicit HyperScan-faithful contract for
    `--graph_second_view_scope neighborloader_batch`: train/valid/test all consume
    the full sampled-subgraph rows, validation/test allow repeated node counting
    across batches, and checkpoint selection switches to validation accuracy
  - this contract is valid only with
    `graph_second_view_scope=neighborloader_batch`,
    `graph_training_loader_mode=neighbor_subgraph`, and
    `graph_data_variant=labeled`
- `--graph_second_view_consumer_scope {all_nodes,routed_only,risk_gated_all_nodes}` controls which
  sampled-subgraph rows consume the second-view detector branch after `x_high`
  has already been built:
  - `all_nodes` preserves the current HNN detector behavior
  - `routed_only` keeps `x_low -> x_new -> KNN -> x_high` shared but mixes the
    final detector outputs row-wise so only routed/high-risk nodes consume the
    HNN detector path
  - `risk_gated_all_nodes` keeps all-node `x_low -> x_new -> KNN -> x_high`
    construction/consumption but replaces the binary second-view switch with a
    continuous residual gate read from `--graph_second_view_risk_path`
- `--graph_second_view_risk_path` is the required frozen full-graph risk payload
  for `risk_gated_all_nodes`; it must resolve to a node-aligned scalar vector
  derived from the `x_new` conformal KNN router/artifact
- `--graph_second_view_risk_gate_mode {linear_sigmoid}` is the current v1
  consumer mapping for that mode and learns `alpha=sigmoid(a*risk+b)`
- `--graph_second_view_nonconsumer_fallback {low_only}` defines the v1 fallback
  for non-consumer rows under routed-selective consumption
- v1 routed-selective consumption is intentionally narrow and parser-validated:
  it currently requires `graph_backbone=rgcn_hyperscan_dhg_nodeinput`,
  `graph_second_view_scope=neighborloader_batch`,
  `graph_neighborloader_contract=hyperscan_sampled_subgraph`,
  `graph_second_view_fusion in {multiattn,residual}`, and
  `--routed_nodes_path`
- under `graph_second_view_fusion=residual`, routed-selective consumption keeps
  the same shared `x_low -> x_new -> KNN -> x_high` construction but mixes the
  final detector outputs row-wise and sends non-consumer rows through the
  explicit `low_only` detector fallback
- v1 risk-gated all-node consumption is also intentionally narrow and
  parser-validated: it requires an active HyperScan-style second view,
  `graph_second_view_fusion=residual`, and a valid
  `--graph_second_view_risk_path`; it does not change KNN member policy
- `--graph_second_view_hypergraph_backend {pyg,dhg}` and
  `--graph_second_view_fusion {residual,multiattn,multiattn_adaptive,construct_acm}` split second-view
  realization into two independent parser axes:
  - `pyg` uses `torch_geometric.nn.HypergraphConv`
  - `dhg` uses DHG `HGNNConv` over the same dynamic incidence specification
- `--graph_second_view_use_bn` is the optional DHG-only alignment knob for
  second-view HGNN batch normalization; PyG backends ignore it
  - `residual` uses `hidden = x_low + f([x_low || x_high]) * incident_mask`
  - `multiattn` uses the HyperScan-faithful bidirectional MultiAttn concat
    detector over two channels only: `x_low` and `x_high`
- `multiattn_adaptive` keeps the same HyperScan-style bidirectional MultiAttn
  tokens but applies a node-wise FAGCN-style low/high adaptive mix before the
  final classifier
- `construct_acm` uses a construct-space identity/low/high gate over
  `x_in_construct`, `x_low_construct`, and `x_high_construct`; this is a
  dual-space three-channel extension rather than the faithful HyperScan
  baseline
- `--hyperscan_detector_style {residual,original_cross_attention,multiattn,original_cross_attention_adaptive,multiattn_adaptive}` is
  retained as a legacy fusion alias. Prefer `--graph_second_view_fusion` for new
  experiments.
- `--graph_node_input_family hyperscan_meta_tweet_proxy` rebuilds a
  HyperScan-style labeled-graph node input from three channels:
  - `tweet_tensor`: the current 768-d semantic embedding
  - `num_prop`: 5 numeric metadata proxies parsed from `norm_user_text`
  - `cat_prop`: 3 boolean/category proxies parsed from `norm_user_text`
- `--graph_node_input_family hyperscan_meta_tweet_proxy` uses that
  paper-inspired tweet/num/cat preprocessing before the current
  HyperScan-style KNN branch. Legacy `rgcn_hyperscan_nodeinput` aliases remain
  accepted for old commands.
  - with `graph_data_variant=full_graph_support`, this node-input family can
    also consume a faithful preprocessed full-graph tensor directly from
    `--embedding_path` when that tensor is already ordered `tweet|num|cat`;
    this bypasses labeled-only `norm_user_text.json` reconstruction and keeps
    the official HyperScan preprocessing contract outside `LLMbot`
- `--graph_construct_embedding_path` is the clean construct/high-order tensor
  path used by `rgcn_h2fag_dualspace_hyperscan*` backbones. It defines the
  construct representation used by the rewritten graph branch:
  `x_in_construct -> x_low_construct -> x_new_construct -> x_high_construct`.
  KNN / hypergraph construction is built from `x_new_construct` directly, and
  the construct tensor must not point to iter_-1 embeddings or exported final
  `node_repr`.
  For `rgcn_h2fag_dualspace_hyperscan_nodeinput`, `--graph_node_input_family
  hyperscan_meta_tweet_proxy` is applied to the construct-side nodeinput path
  only and defines `x_in_construct`.
- `--graph_backbone` now also includes:
  - `rgcn_h2fag_dualspace`
  - `rgcn_h2fag_dualspace_hyperscan`
  - `rgcn_h2fag_dualspace_nodeinput`
  - `rgcn_h2fag_dualspace_hyperscan_nodeinput`
  These rewritten backbones keep the graph branch fully inside construct space
  and stamp `backbone_contract_version=construct_complete_v2` into artifacts.
  Older artifacts under the same backbone names are pre-rewrite and must not be
  reused as comparable results.
  In v1 they are intentionally bounded to the clean detector-consumption path
  and reject `--mhlgc_enable`, `--routed_contrast_family`, and
  `--routed_highpass_mode`.
- `--mhlgc_enable` adds an MH-LGC-style auxiliary graph contrastive loss during
  `graph_detector_prepare`:
  - it requires `--mhlgc_semantic_embedding_path`, a graph-node-aligned tensor
    produced by a routed-node prompt cache such as
    `precompute.py --prompt_mode mhlgc_llm_guide --routed_nodes_path ...`
  - `LLMbot/prompt.py` owns the prompt builder; the prompt follows the paper's
    visible structure `Instruction / [Role Description] / [Task Definition] /
    [Input Graph]`, serializes the original directed relation view and a
    HyperScan-style KNN hypergraph view, and explicitly forbids final bot/human
    labels
- `LLMbot/GNNs.py` owns the GCL loss; borderline anchors are positive-label
  nodes with the lowest current positive-class score, the positive view is
  the same node under feature/edge masking, and hard-negative weights combine
  GNN similarity with LLM semantic similarity through `--mhlgc_gamma`
- `--mhlgc_contrast_space {fused_x,node_repr,x_new,low_high_concat,low_high,semantic}` selects the optimized
  representation space:
  - `fused_x` is the preferred explicit final detector hidden state
  - `node_repr` is the legacy alias for that same final detector hidden
  - `x_new` uses the HyperScan-style construction space and therefore requires
    a backbone that exports `outputs["x_new"]`
  - `low_high_concat` uses `cat(outputs["x_low"], outputs["x_high"])`, the
    pre-detector relation/hypergraph view pair, and therefore requires a
    HyperScan-style backbone that exports both tensors
  - `semantic` is a diagnostic input-semantic-space contrast over the current
    LM embedding tensor, not over the LLM guide cache
- `--mhlgc_anchor_source {semantic_nonzero_positive,routed_target_mask,positive_label}`
  separates the legacy implicit nonzero-guide anchor mask from explicit routed
  target-node anchoring
- `--mhlgc_pair_mode {augmentation,repair_aware}` keeps the existing masked-view
  positive pair by default; `repair_aware` turns the LLM guide into a routed
  correction signal by building a guide-conditioned repaired second view for
  HyperScan-style backbones, then contrasting the non-guided graph
  representation against that repaired view in the selected contrast space
- `--mhlgc_semantic_projector {auto,none}` controls whether repair-aware mode
  may learn a train-time linear projector when guide and graph dimensions do
  not match
- `--mhlgc_hyperedge_mask_probability` optionally extends the augmented view
  to the active second-view hypergraph by masking node-hyperedge incidence
  entries after dynamic KNN construction; `0` keeps the older
  feature/edge-only augmentation contract
  - `--mhlgc_negative_count 0` preserves the legacy all-negative weighting;
    positive values retain only that many top-hardness negatives per anchor.
    Use `--mhlgc_negative_count 3` with `--mhlgc_anchors_per_batch 1` for the
  closer paper-style anchor-plus-three-negatives setting.
- explicit zero-valued MH-LGC float args are preserved. In particular,
  `--mhlgc_gamma 0.0` now truly means structure-only hardness weighting instead
  of falling back to the default `0.5`.
- `LLMbot/trainer_preparation.py` records the MH-LGC configuration in the
  graph-detector manifest and rejects cache reuse when the requested MH-LGC
  setting differs from the existing artifact
  - this is an MH-LGC-style social-bot adaptation, not a claim that the
    original fraud-detection codebase or temporal/frequency views were fully
    reproduced

Routed high-pass correction:

- `--routed_highpass_mode {off,low_only,high_only,adaptive}` enables a
- full-batch-only routed correction module. `--routed_highpass_target logits`
  preserves the legacy post-detector delta-logit correction, while
  `--routed_highpass_target x_high` applies the self/low/high correction to the
  HGNN high-order branch before HyperScan-style cross-attention fusion.
- `--routed_highpass_candidate_scope {relation_1hop,relation_1hop_plus_xnew_knn}`
  controls the candidate policy. The clean default uses relation-supported
  1-hop candidates. The `plus_xnew_knn` ablation adds forward-native `x_new`
  KNN candidates before top-k truncation; `x_new` is treated as a construction
  and candidate-ranking space, not as the final detector representation.
- `--routed_highpass_preserve_weight` keeps non-routed train nodes close to
  base logits, and `--routed_highpass_edge_role_weight` adds labeled-labeled
  same-label/ different-label edge-role supervision.
- Training is staged: the base graph detector is selected with the ordinary CE
  objective first; the routed high-pass module is then trained with the base
  detector frozen.
- This branch is mutually exclusive with `--mhlgc_enable` and
  `--routed_contrast_family` so a run tests one correction mechanism at a time.

Hidden compatibility / deprecated-unwired:

- `--GNN_model`

### Text and semantic encoder

- `--text_encoder`
- `--semantic_encoder`
- `--qwen_model_path`
- `--qwen_trust_remote_code`
- `--max_length`
- `--lm_optimizer`
- `--dropout`
- `--LM_classifier_n_layers`
- `--LM_classifier_hidden_dim`
- `--lm_dropout`
- `--lm_attention_dropout`
- `--label_smoothing_factor`
- `--warmup`

Hidden compatibility / deprecated-unwired:

- `--LM_model`
- `--semantic_backbone`

### Training and optimization

- `--seeds`
- `--device`
- `--joint_train_node_cap`
- `--joint_routing_protocol`
- `--joint_router_reuse_root`
- `--joint_refiner_explicit_gate`
- `--joint_refiner_target_mode`
- `--joint_refiner_gate_target`
- `--joint_refiner_weight_mode`
- `--joint_refiner_base_wrong_weight`
- `--joint_refiner_utility_weight`
- `--joint_refiner_gate_weight`
- `--joint_utility_advantage_experiment`
- `--joint_refiner_gate_policy`
- `--joint_refiner_gate_threshold`
- `--joint_correction_moe_expert_weight`
- `--joint_correction_moe_utility_weight`
- `--joint_correction_moe_utility_target`
- `--LM_pretrain_epochs`
- `--semantic_max_steps`
- `--semantic_train_limit`
- `--LM_eval_patience`
- `--LM_accumulation`
- `--max_iters`
- `--GNN_epochs_per_iter`
- `--LM_epochs_per_iter`
- `--temperature`
- `--pl_ratio_LM`
- `--pl_ratio_GNN`
- `--alpha`
- `--beta`
- `--lm_learning_rate`
- `--lm_weight_decay`
- `--gnn_learning_rate`
- `--gnn_weight_decay`

Hidden compatibility / deprecated-unwired:

- `--optimizer_LM`
- `--optimizer_GNN`
- `--LM_dropout`
- `--LM_att_dropout`
- `--GNN_dropout`
- `--lr_LM`
- `--weight_decay_LM`
- `--lr_GNN`
- `--weight_decay_GNN`
- `--gamma`
- MLP legacy knobs:
  - `--MLP_n_layers`
  - `--MLP_hidden_dim`
  - `--optimizer_MLP`
  - `--MLP_dropout`
  - `--MLP_KD_epochs`
  - `--MLP_epochs_per_iter`
  - `--pl_ratio_MLP`
  - `--lr_MLP`
  - `--weight_decay_MLP`

### Estimator / router / repair controls

- `--estimator_mode`
- `--sparse_degree_quantile`
- `--propagation_quantile`
- `--bootstrap_samples`
- `--risk_budgets`
- `--conformal_knn_k`
- `--conformal_knn_candidate_scope`
- `--conformal_knn_target_top_n`
- `--conformal_knn_neighbor_mode`
- `--conformal_knn_similarity_threshold`
- `--conformal_knn_min_support`
- `--conformal_knn_adaptive_max_k`
- `--conformal_knn_hubness_correction`
- `--conformal_knn_score_family_override`
- `--conformal_knn_shrinkage_tau`
- `--conformal_knn_ncp_lambda`
- `--conformal_knn_repr_source`
- `--conformal_knn_learning_mode`
- `--conformal_knn_local_calibration_scope`
- `--router_oof_mode`

Hidden compatibility / inactive placeholders:

- `--semantic_mode`
- `--repair_mode`
- `--selector_mode`
- `--appendix_mode`
- `--router_budgets`
- `--selector_budget`
- `--gain_clip_value`

### Phase A and semantic feature controls

- `--embedding_path`
- `--phase_a_project_dim`
- `--phase_a_projector`
- `--peft`
- `--peft_rank`
- `--peft_alpha`
- `--graph_detector_epochs`
- `--gate_calibrator_max_iter`

Hidden compatibility / deprecated-unwired:

- `--emb_path`
- `--g0_feature_path`
- `--g0_epochs`
- `--gats_max_iter`
- `--phase_a_semantic_sources`
- `--phase_a_feature_paths`
- `--phase_a_gnn_family`
- `--phase_a_fixed_gnn`
- `--phase_a_fixed_semantic_source`
- `--phase_a_latency_repeats`
- `--phase_a_include_structural_smoke`

## Canonical Naming Map

| Legacy flag | Canonical flag |
| --- | --- |
| `--stage` | `--experiment_task` |
| `--GNN_model` | `--graph_backbone` |
| `--LM_model` | `--text_encoder` |
| `--semantic_backbone` | `--semantic_encoder` |
| `--emb_path` | `--embedding_path` |
| `--g0_feature_path` | `--embedding_path` |
| `--batch_size_LM` | `--lm_batch_size` |
| `--batch_size_GNN` | `--gnn_batch_size` |
| `--optimizer_LM` | `--lm_optimizer` |
| `--optimizer_GNN` | `--gnn_optimizer` |
| `--LM_dropout` | `--lm_dropout` |
| `--LM_att_dropout` | `--lm_attention_dropout` |
| `--GNN_dropout` | `--gnn_dropout` |
| `--lr_LM` | `--lm_learning_rate` |
| `--lr_GNN` | `--gnn_learning_rate` |
| `--weight_decay_LM` | `--lm_weight_decay` |
| `--weight_decay_GNN` | `--gnn_weight_decay` |
| `--g0_epochs` | `--graph_detector_epochs` |
| `--gats_max_iter` | `--gate_calibrator_max_iter` |

## Naming And Maintainability Risks

### Canonical CLI is ahead of internal extraction

The public parser surface is now canonical. The remaining work is inside the
implementation body, not the CLI.

### Runtime-only artifacts are still in migration

Some runtime helper outputs, compatibility manifests, and internal notes still
carry historical labels. The mainline rule is now:

- new public command docs use canonical names
- new stage/preparation writes use canonical namespaces
- legacy names remain readable only for compatibility

## Strict Joint Provenance Note

- For `roberta` / `roberta_finetuned`, omitting `--embedding_path` now means:
  - default to `datasets/TwiBot-20/embeddings_iter_-1_seed_{seed}.pt`
  - if that file is missing, fall back to `datasets/TwiBot-20/embeddings_roberta.pt`
- `datasets/TwiBot-20/finetuned_roberta_embeddings_iter_2_seed1.pt` is now a
  historical explicit compatibility branch for legacy iter-2 comparisons; the
  default semantic embedding path remains `embeddings_iter_-1_seed_{seed}.pt`.
- `--external_frozen_g0_root` still exists for diagnostic graph-aware stages
- public `joint_router_refinement` no longer accepts cross-root backbone reuse
- public `joint_router_refinement` binds its semantic tensor to the current
  run's `preparation/graph_detector` artifact
- an explicit `--embedding_path` remains parse-valid, but for
  `joint_router_refinement` it must equal that preparation artifact's
  `feature_manifest.path`
- `--risk_budgets` now also governs the public strict GLANCE evaluation sweep:
  the stage evaluates whole-split router rankings at those candidate budgets,
  selects the best budget on validation only, and locks that budget on test
- public `joint_router_refinement` also exposes `--joint_train_node_cap` for
  controlled strict-stage comparisons:
  - `3000` keeps the paper-style capped train subset
  - `0` switches the strict stage to the full TwiBot20 train split

### `trainer_legacy_impl.py` is still a concentration point

The parser contract is now much cleaner than the remaining execution body.
Recent cleanup moved real preparation and semantic execution out of the
monolith, moved GLANCE runner/helper tails into `trainer_glance.py`, and moved
`local_conflict_prune_diag` structural-conflict helpers into
`trainer_graph.py`. The remaining extraction pressure is now estimator/matrix,
local conformal helper tails, and compatibility paths.

## Legacy Parser Inventory

Deprecated parser/reference surfaces:

- `LLMbot/baseline/core/*`
- `LLMbot/baseline/analysis/*`
- `LLMbot/baseline/baselines/*`

These remain reference inventory only. `LLMbot/code/*` is no longer a legacy
surface; it is the active flat source directory.

## Future Refactor Notes

1. continue moving estimator/matrix, local conformal helper tails, and
   remaining GLANCE compatibility logic out of `trainer_legacy_impl.py`
2. remove compatibility reads of legacy shadow fields from active mainline code
3. finish canonicalizing runtime-only artifact surfaces
4. remove hidden aliases only after a clean round of canonical-command
   validation

## Documentation Maintenance Rule

Any future change to `LLMbot/code/parser_args.py`, public CLI examples, hidden
alias policy, or task exposure must update this file in the same task. If the
public command surface changes, update `LLMbot/README.md` as well.
- `--graph_backbone rgcn_hyperscan` is the canonical HyperScan-style relation
  backbone. Legacy aliases such as `rgcn_hyperscan_routed`,
  `rgcn_hyperscan_dhg`, `rgcn_hyperscan_nodeinput`, and
  `rgcn_hyperscan_dhg_nodeinput` remain accepted for old commands, but new
  experiments should switch realization with `--graph_second_view_*` flags.
- `--graph_refine_mode routed_dynamic_hyperscan_branch`:
  - requires `--graph_backbone rgcn_hyperscan` or a legacy `rgcn_hyperscan_*`
    alias
  - keeps the original graph tensors unchanged
  - reads centers from `--routed_nodes_path` and `--routed_nodes_split`
  - by default restricts candidates to relation 1-hop ego neighbors
  - `--graph_second_view_candidate_scope labeled_relation_1hop` further
    filters those relation neighbors to the labeled prefix only
  - `--graph_second_view_candidate_scope labeled_full` or `hyperscan_full`
    expands the candidate pool for diagnostic support-policy ablations only
  - builds local KNN hyperedges from `x_low + x_in` during forward rather than
    materializing proxy edges before training
  - this HyperScan-style KNN construction space is `x_new=cat(x_low,x_in)`;
    `node_repr` is the final detector hidden space and must not be used for
    this flow except as a separate non-HyperScan control ablation
  - writes dynamic-branch telemetry into `outputs.pt["aux_features"]` and
    `graph_refine_stats.json`
- `--graph_refine_mode hyperscan_neighborloader_batch_local_branch`:
  - requires `--graph_data_variant labeled`
  - requires `--graph_backbone rgcn_hyperscan` or a legacy `rgcn_hyperscan_*`
    alias
  - uses `NeighborLoader` for train/valid/infer graph-detector passes
  - rebuilds the KNN hypergraph from `x_low + x_in` on each sampled subgraph
    rather than from a predeclared global center list
- `--graph_second_view_hypergraph_backend {pyg,dhg}` selects the second-view
  HGNN realization independently from scope and candidate range.
- `--graph_second_view_use_bn` optionally enables DHG HGNN batch normalization
  for closer official-HyperScan alignment.
- `--graph_second_view_fusion {residual,multiattn,multiattn_adaptive,construct_acm}` selects the second-view
  fusion head independently from the HGNN backend.
- When `conformal_knn_risk_router` is paired with a HyperScan second-view graph
  mode, parser normalization requires `--conformal_knn_repr_source x_new`.
  Explicit `node_repr` is rejected for that paired flow because it is the
  post-classifier hidden representation and can echo base-detector predictions.
- `--graph_training_loader_mode neighbor_subgraph`:
  - applies to graph-detector optimization regardless of refine mode
  - uses `NeighborLoader` for train/valid/infer graph passes
  - scatters seed-node outputs back to graph order through an explicit
    `node_id` attribute carried in the sampled subgraph batches
  - when paired with `--graph_neighborloader_contract hyperscan_sampled_subgraph`,
    training supervision uses the full sampled-subgraph rows and the contract
    writes an additional `neighborloader_contract_metrics.json` sidecar while
    keeping `outputs.pt` as a full-graph deduplicated export
