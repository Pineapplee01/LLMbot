# LLMbot Active Mainline Risk And Status Log

Date: 2026-05-26

Scope:

- active mainline: `LLMbot/`
- active source directory: `LLMbot/code/`
- compatibility entrypoints: `LLMbot/main.py`, `LLMbot/precompute.py`,
  `LLMbot/preprocess.py`
- review focus:
  - `LLMbot/code/main.py`
  - `LLMbot/code/parser_args.py`
  - `LLMbot/code/stage_registry.py`
  - `LLMbot/code/trainer.py`
  - `LLMbot/code/trainer_legacy_impl.py`
  - `LLMbot/code/model_building.py`
  - `LLMbot/code/estimators.py`

This file is the active maintainability and implementation-risk log for the
current mainline. It is not an experiment result record.

## Governance Baseline 2026-07-04

First-stage governance establishes a short context layer and canonical terms
before any large Python movement:

- `conductor/product.md` defines the active-mainline goal and claim boundary.
- `conductor/tech-stack.md` records the runtime, ML stack, and local model
  cache contract under `G:\Research\BotDetection\models`.
- `conductor/workflow.md` fixes edit zones, naming rules, and governance-only
  verification.
- `conductor/tracks.md` records the first governance track and the next
  candidate slices.
- `UBIQUITOUS_LANGUAGE.md` fixes canonical terms for main entry, CLI contract,
  orchestration layer, algorithm layer, precompute context, artifact surface,
  runner script, public/internal stage, and manifest.

The first source-location migration places active Python business files under
`LLMbot/code/` while preserving root compatibility entrypoints. Generated
experiment artifacts and server logs are not part of this governance baseline.

## NLPCC Submission Package 2026-07-05

`NLPCC/code/` now contains a trimmed seven-module source package for the NLPCC
submission line. It is not a full `LLMbot/code/` snapshot and does not include
claim folders, runner folders, GLANCE modules, compatibility orchestration, or
artifact outputs.

Current NLPCC modules:

- `main.py`
- `parser_args.py`
- `graph_detector.py`
- `router.py`
- `models.py`
- `data_io.py`
- `utils.py`

Risk: future fixes in `LLMbot/code/` can still matter to NLPCC, but they should
not be copied wholesale. Recommended rule: decide the paper-facing interface
first, then port only the smallest required implementation into one of the
seven NLPCC modules.

Interface and naming standard:

- `parser_args.py` is the only CLI module and uses `experiment_task`; do not
  add `cli.py`, `stage`, or hidden legacy aliases.
- `graph_detector.py` describes the graph-detector task through
  `describe_graph_detector_task`; do not use fake readiness/status fields to
  imply training was executed.
- `router.py` is the reliability-ranking module; do not rename it to
  `risk_router.py` or reintroduce GLANCE/router-refiner vocabulary.
- `models.py` owns lightweight config dataclasses only; no framework-specific
  model builders are ported until the NLPCC training interface is explicitly
  designed.
- `data_io.py` owns JSON/path I/O; `utils.py` stays narrow and currently owns
  only seed parsing.
- Every module's `__all__` is the module interface and must remain covered by
  `tests/test_nlpcc_code_migration_contract.py`.

## LLMbot Naming Governance 2026-07-05

`LLMbot/code/` is still a mixed research mainline, not a cleaned single-claim
package. Current large modules retain legacy names for compatibility, but new
or migrated code must use canonical names from `UBIQUITOUS_LANGUAGE.md`.

Common canonical names: `experiment_task`, `task_spec`, `graph_backbone`,
`text_encoder`, `semantic_encoder`, `embedding_path`, `fused_x`,
`target_node_ids`, `routed_node_ids`, `center_node_ids`, `risk_score`,
`router_score`, `stage_dir`, `artifact_namespace`, and `manifest`.

Legacy-only aliases: `stage`, `stage_name`, `GNN_model`, `LM_model`,
`semantic_backbone`, `emb_path`, `node_repr`, and `frozen_g0`. These may appear
in compatibility paths, historical runners, manifests, and replay notes, but
they should not be advertised as new claim vocabulary.

Each claim README under `LLMbot/code/claims/` now carries a canonical claim
vocabulary. Future governance work should update the claim vocabulary before
moving or renaming code. The static guard is
`tests/test_llmbot_naming_governance.py`.

## Claim Package Skeleton 2026-07-06

Claim code now targets `LLMbot/code/<claim_name>/` packages. The existing
`LLMbot/code/claims/` tree stays as a governance-note surface for now and is
not the destination for business code.

Created packages: `phase_a_foundations/`, `high_order_graph_consumption/`,
`conformal_risk_routing/`, `routed_llm_evidence_refinement/`,
`local_graph_repair_diagnostics/`, `semantic_candidate_correction/`, and
`ablation_positioning_legacy/`.

`LLMbot/code/claim_map.json` records file-level and symbol-level claim
classification. It distinguishes `owned`, `shared`, `compat_shim`,
`do_not_move_yet`, and `legacy_replay` status.

First moved implementations:

- `router.py` moved to `conformal_risk_routing/router.py`; root `router.py`
  is now a compatibility shim.
- `llm_evidence_refiner.py` moved to
  `routed_llm_evidence_refinement/llm_evidence_refiner.py`; root
  `llm_evidence_refiner.py` is now a compatibility shim.
- `trainer_dignn_conflict.py` moved to
  `local_graph_repair_diagnostics/trainer_dignn_conflict.py`; root
  `trainer_dignn_conflict.py` is now a compatibility shim.

Large mixed files remain top-level and are marked `do_not_move_yet` in the
claim map: `trainer_glance.py`, `precompute.py`, `estimators.py`,
`trainer_preparation.py`, `trainer_legacy_impl.py`, `model_building.py`,
`GNNs.py`, and `prompt.py`.

## Refactor Naming Standard 2026-07-04

This cleanup slice keeps the flat `LLMbot/code/` source surface and avoids new
`stages/` or `methods/` directories. New public commands, docs, manifests, and
owner-module code use canonical names: `experiment_task`, `graph_backbone`,
`text_encoder`, `semantic_encoder`, `embedding_path`,
`graph_detector_prepare`, and `fused_x`.

Legacy aliases are compatibility vocabulary only: `stage`, `GNN_model`,
`LM_model`, `semantic_backbone`, `emb_path`, `g0_feature_path`, `frozen_g0`,
and `node_repr`. `parser_args.py` remains the only active-mainline module that
creates legacy shadow fields. Medium cleanup may remove hidden deprecated-only
surfaces such as `eqc_v8_matrix`, but must not remove public canonical task
names or common renamed aliases in the same slice.

Current cleanup status:

- `eqc_v8_matrix` is rejected on both `--experiment_task` and hidden
  `--stage`.
- `trainer_graph.py` now owns the full-graph structural-conflict RGCN helper
  instead of importing it from `trainer_legacy_impl.py`.
- `trainer_glance.py` owns the GLANCE runner/helper tail methods that were
  duplicated in `trainer_legacy_impl.py`.
- the dead optional `semantic_ib_edl_head.py` dynamic loader was removed from
  active `code/` modules because no local tracked file exists.
- `main.py` and `stage_runner.py` now use `task_spec`, `requested_task`, and
  `execution_task` as the preferred local terminology while preserving
  `StageSpec`, `args.stage`, and `args.stage_spec` for compatibility.
- single-owner files were consolidated: `conflict_refiner.py` into
  `trainer_dignn_conflict.py`, `LM.py` into `model_building.py`, and
  `RGT.py`/`SimpleHGN.py` into `GNNs.py`.
- duplicate `trainer_legacy_impl.py` contract/path helper definitions already
  rebound to `artifact_contracts.py` were removed; the legacy body remains for
  matrix/fallback behavior still in the inheritance chain.
- `trainer_distillation.py` is now the only implementation owner for
  `LM_Trainer`, `GNN_Trainer`, `MLP_Trainer`,
  `_safe_pseudo_label_training_index`, and `run_legacy_graph_seed`;
  `trainer_legacy_impl.py` keeps those names only as compatibility aliases.

## Current Mainline Snapshot

The active mainline has already been refactored onto a cleaner public contract:

- `LLMbot/` is the only active operator mainline
- `LLMbot/code/` is the active flat Python source surface
- `LLMbot/code/stage_registry.py` is the public task naming and visibility source of truth
- `LLMbot/code/parser_args.py` exposes canonical public flags and canonical task names
- `LLMbot/code/main.py` resolves tasks through the stage registry and supports a parser-only `--help` path
- `LLMbot/code/trainer.py` is a thin compatibility facade
- `LLMbot/code/trainer_legacy_impl.py` still contains most execution logic during the extraction transition
- `LLMbot/code/artifact_contracts.py` now owns contract/path/provenance helpers
- `LLMbot/code/runtime_env.py` now owns device/CUDA runtime helpers
- `LLMbot/code/stage_runner.py` now owns the shared `StageRunner` skeleton
- `LLMbot/code/trainer_graph.py` now owns shared graph runtime logic and the public
  local graph diagnostic executors
- `LLMbot/code/trainer_glance.py` now owns shared GLANCE semantic input logic and
  the public `joint_router_refinement` executor plus migrated internal GLANCE
  execution lanes
- `LLMbot/code/trainer_preparation.py` now owns preparation execution
- `LLMbot/code/trainer_preparation.py` and `LLMbot/code/GNNs.py` now expose
  `--routed_highpass_mode` plus `--routed_highpass_target`, a routed-only
  correction branch that separates self, low-pass neighbor aggregation, and
  high-pass residual channels either after detector fusion (`target=logits`) or
  before cross-attention on the HGNN high-order branch (`target=x_high`). The
  branch uses staged training: select the base detector first, then freeze it
  and train only the correction module. This is an implementation surface for
  H2GCN/FAGCN/BotSCL-inspired heterophily diagnosis, not an experiment result
  or performance claim.
- `LLMbot/GNNs.py`, `LLMbot/parser_args.py`, and
  `LLMbot/trainer_preparation.py` now also expose
  `--graph_second_view_fusion multiattn_adaptive`, a detector-only
  HyperScan-style fusion option that preserves the existing relation branch,
  dynamic `x_new -> x_high` high-order branch, and bidirectional
  cross-attention tokens, then applies a node-wise FAGCN-style adaptive
  low/high mix before the final classifier. Current maintainability risk:
  reports must keep this scoped as a fusion-head adaptation rather than a full
  FAGCN reproduction or a new graph-construction method.
- `LLMbot/GNNs.py`, `LLMbot/parser_args.py`, and
  `LLMbot/trainer_preparation.py` now also expose
  `--graph_second_view_fusion construct_acm`, a construct-complete detector
  consumer that adaptively mixes `x_in_construct`, `x_low_construct`, and
  `x_high_construct`. Current maintainability risk: this is an ACM/FAGCN-style
  detector consumer, not an official ACM-GNN reproduction and not the
  HyperScan-faithful two-channel baseline.
- `LLMbot/GNNs.py`, `LLMbot/parser_args.py`, and
  `LLMbot/trainer_preparation.py` now also expose routed-selective second-view
  detector consumption through
  `--graph_second_view_consumer_scope {all_nodes,routed_only,risk_gated_all_nodes}` plus
  `--graph_second_view_nonconsumer_fallback low_only`. This keeps
  `x_low -> x_new -> x_high` construction shared while letting only routed
  rows consume the HNN detector path. Current maintainability risk: future
  reports must keep this framed as detector-consumption unification rather
  than as a router upgrade or a new hyperedge-member selection method.
  The parser now also allows `routed_only + residual` under the same narrow
  `dhg_nodeinput + neighborloader_batch + hyperscan_sampled_subgraph`
  contract; this matches existing model behavior and removes the need for
  post-parse experiment-driver overrides.
- the same implementation surface now also exposes conformal-risk-gated
  all-node residual consumption through `--graph_second_view_risk_path` and
  `--graph_second_view_risk_gate_mode linear_sigmoid`. Current maintainability
  risk: this is a detector-only scalar gate over a frozen external `x_new`
  risk vector, not a learned router and not a `multiattn` result.
- `LLMbot/GNNs.py`, `LLMbot/model_building.py`, `LLMbot/parser_args.py`, and
  `LLMbot/trainer_preparation.py` now also expose dual-space
  `rgcn_h2fag_dualspace*` backbones under the rewritten
  `construct_complete_v2` contract. These now keep the graph branch fully
  self-consistent inside construct space:
  `construct_x -> x_in_construct -> x_low_construct -> x_new_construct ->
  x_high_construct`.
  The repo's faithful HyperScan comparison line remains the two-channel
  `rgcn_hyperscan_dhg_nodeinput + multiattn` family; `construct_complete_v2`
  should be treated as a separate dual-space extension line.
  The active dual-space HyperScan branch now also honors
  `--graph_second_view_hypergraph_backend {pyg,dhg}` inside its construct-side
  high-order encoder, and `--graph_second_view_use_bn` can be used to align
  DHG-backed runs with the released HyperScan TwiBot20 HGNN batch-normalization
  setting.
  The active implementation no longer uses the old clean-projector
  `x_new_clean` path, and it hard-rejects MH-LGC / routed-contrast /
  routed-highpass overlays on dual-space hyperscan backbones in v1.
  The `*_nodeinput` variant applies tweet/num/cat preprocessing only on the
  construct-side `--graph_construct_embedding_path`.
  Current maintainability risk: future code must not assume
  pre-rewrite `rgcn_h2fag_dualspace*` artifacts are comparable unless
  `backbone_contract_version=construct_complete_v2` is manifest-verified.
- `LLMbot/trainer_semantic.py` now owns semantic finetune execution
- `LLMbot/trainer_distillation.py` now owns the legacy distillation trainers
- `LLMbot/router.py` now owns the reliability-first strict router module,
  including temperature scaling, direction-aware social feature construction,
  and the router MLP itself
- `LLMbot/precompute.py` now owns the legacy prompt-cache helpers and routed
  prompt/explain/embedding orchestration, while the shared HyperScan-style KNN
  grouping and hypergraph helper logic now lives in `LLMbot/hypergnn.py`
- `LLMbot/estimators.py` now exposes `conformal_knn_candidate_scope=hyperscan_full`
  for router-side HyperScan-aligned full feature-pool KNN risk estimation.
  This is a post-hoc target-node risk router support-group mode, not evidence
  that the training-time HyperScan second-view detector itself improved.
- `LLMbot/estimators.py` now also allows `conformal_knn_learning_mode=learned_logistic`
  as a bounded learned-router ablation inside the same target-node
  Conformal-KNN surface. It learns a residual-risk scorer over the existing
  base-risk + structured KNN/local feature bundle, but it does not alter the
  frozen graph-detector logits. Current maintainability risk: future reports
  must keep "router-quality improvement" separate from "classifier-quality
  improvement" and must compare `learned_logistic` against `fixed` / `ncp_local`
  under matched frozen-G0, split, and budget settings.
- `LLMbot/precompute.py` now also owns routed-node-targeted prompt-expert
  subset execution while preserving full-graph tensor layout for downstream
  refiner compatibility
- `LLMbot/precompute.py` now also owns a raw tweet source compatibility layer
  for `expert_tweet` and tweet-dependent `conflict`, rebuilding user-centered
  tweet evidence from `node_new.json` and `edge_new.json` / `edge.csv(post)`
  while falling back to serialized `norm_user_text` tweet segments when needed
- `LLMbot/precompute.py` now also owns `prompt_expert_bundle_v3`, a structured
  evidence-card variant of the v2 explanation-first prompt-expert contract for
  routed-node correction studies
- explanation-first `prompt_expert_bundle_v2/v3` is now explicitly bounded
  away from `selection_embedding_path` / semantic-KNN prompt selection:
  the canonical path is `router -> routed nodes -> LLM evidence -> routed-only
  node classifier`, while center-induced KNN prompt construction remains a
  separate diagnostic under `prompt_expert_bundle_center_induced_v1` or
  `mhlgc_llm_guide`
- `LLMbot/trainer_glance.py` now owns a legacy mean-propagation routed-node
  ablation, `ultratag_propagated_follower_triplet`, which reuses cached
  prompt-expert embeddings and one-hop mean-propagates them at runtime before
  the no-projector follower-triplet refiner. It is intentionally not described
  as an UltraTAG-S reproduction.
- `LLMbot/precompute.py` now owns the bounded UltraTAG-S adaptation path,
  `ultratag_s_subgraph_v1`: target-plus-neighbor text propagation, LLM
  summary/keywords/soft-label augmentation, optional same-soft-label virtual
  edges, optional PageRank-selected LLM edge reconfiguration,
  finetuned-RoBERTa row replacement, and `edge_index`/`edge_type` output for
  `graph_detector_prepare`.
- `LLMbot/precompute.py`, `LLMbot/trainer_preparation.py`,
  `LLMbot/hypergnn.py`, and `LLMbot/GNNs.py` now expose a narrower
  `llm_knn_edge_retain_v1 -> llm_retain` path: Qwen3.5-style generation is run
  offline over target-centered non-routed KNN pairs, producing KEEP/DROP
  support-edge decisions that filter dynamic HGNN `x_high` members before
  cross-attention. This is an UltraTAG-S-inspired edge-reconfiguration
  ablation combined with FAGCN-style low/high-pass motivation, not an
  LLM-as-final-predictor path or a full UltraTAG-S reproduction.
  The generation loader keeps the standard causal-LM path first and falls back
  to Qwen3.5 conditional-generation classes when a local checkpoint declares
  the `qwen3_5` architecture; server runs may still need an isolated
  Transformers-main `PYTHONPATH` until the base Qwen environment exposes those
  classes directly.
- `scripts/router_knn_evidence_reconfiguration.py` now owns a routed-node
  GraphEdit-style baseline for the conformal-router KNN support space. It
  exports GraphEdit pairwise True/False prompts for two stages: existing
  routed KNN edges as drop/keep candidates and second-ring KNN candidates as
  add candidates. Qwen3.5-9B generation is supported through an isolated
  Transformers-main path and must use `enable_thinking=False`; the resulting
  JSONL sidecar is consumed by the external routed-node refiner.
  `--llm_prompt_variant graphedit_same_behavior` preserves the direct
  same-behavior GraphEdit prompt baseline, while
  `--llm_prompt_variant graphedit_edge_usefulness` keeps the same two-stage
  edge-editing protocol but asks whether a candidate user is useful behavioral
  evidence for refining the target user's bot-detection decision. This is a
  GraphEdit-style baseline over fixed router candidates, not a full GraphEdit
  reproduction, a free-form LLM graph search, or an LLM-as-final-predictor
  method.
- `LLMbot/precompute.py` now owns `residual_audit_v1`, a routed-node prompt
  cache for correction-utility studies. It consumes frozen SimTeG
  prediction/probability outputs as fallible prompt context, supports
  no-base/anchoring variants, and produces one encoded `residual_audit`
  component without changing router/refiner training or leaking dataset labels.
  The prompt rows retain Qwen2.5-Instruct-style `system` / `user` messages and
  constrain the requested response to strict JSON over evidence label,
  correction action, confidence, evidence cues, and uncertainty/risk notes.
- `LLMbot/precompute.py` now owns `dgp_predictor_v1`, a routed-node DGP-style
  predictor prompt cache. It keeps target account evidence fine-grained,
  compresses following/follower context into ranked coarse cards, and writes
  both a prompt sidecar for Qwen PEFT predictor tuning and an embedding cache
  for CALM-style query-embedding-MLP comparison.
- `LLMbot/precompute.py` now also owns `dgp_predictor_v2`, a norm-text
  DGP-style routed-node surface. It treats target `norm_user_text` as
  fine-grained tweet+metadata evidence, summarizes top-K selected directional
  neighbor `norm_user_text` rows, compresses those summaries into relation
  context summaries, and encodes a final Yes/No predictor prompt. The stable
  mainline is K=5 following-neighbor summary; follower-side variants are
  explicit ablations. The prompt sidecar can now be consumed by
  `semantic_encoder_finetune --semantic_encoder qwen3_peft
  --semantic_supervision_mode answer_token`, which upgrades the old
  hidden-state classifier route to answer-token SFT while preserving the same
  `embeddings.pt` / `outputs.pt` artifact contract.
  The DGP prompt surface has been tightened to a stricter paper-style form:
  final prediction prompts use a short `Instruct / Query / ASSISTANT_ANSWER`
  shell with `Yes|No` first-token supervision, and v2 neighbor/context
  summarization now uses task-agnostic `Summarize ... within 10 tokens`
  prompts instead of the older requirement-heavy social-bot-specific schema.
- `LLMbot/precompute.py`, `LLMbot/prompt.py`, `LLMbot/GNNs.py`, and
  `LLMbot/trainer_preparation.py` now expose a bounded MH-LGC-style
  adaptation path:
  - `precompute.py --prompt_mode mhlgc_llm_guide --routed_nodes_path ...`
    constructs graph-node-aligned LLM-guide prompts for routed hard nodes from
    an original directed relation view plus a HyperScan-style KNN hypergraph
    view derived from `--selection_embedding_path`
  - `precompute.py` can now encode those flattened views with an explicit
    causal-LM hidden-state backend:
    `--embedding_model_class causal_lm --embedding_pooling_mode
    causal_last_hidden_last_token` loads `AutoModelForCausalLM`, requests
    hidden states, and pools `outputs.hidden_states[-1]` at the last valid
    token; this is the paper-style semantic-view path, while generated explain
    text remains diagnostic only
  - `prompt.py` owns the prompt wording and keeps the LLM-as-guide boundary by
    using the paper-style `Instruction / [Role Description] /
    [Task Definition] / [Input Graph]` shell and forbidding final bot/human
    labels or probabilities
  - `GNNs.py` owns the hard-negative contrastive loss, selecting borderline
    positive-label anchors from the lowest current positive-class scores and
    weighting negatives by GNN plus LLM semantic similarity
  - `--mhlgc_anchors_per_batch 1 --mhlgc_negative_count 3` is the closer
    paper-style adapted setting: one borderline positive anchor and three
    hard negatives per batch. `--mhlgc_negative_count 0` intentionally retains
    the prior all-negative exploratory behavior for direct ablation continuity.
- `trainer_preparation.py --mhlgc_enable` consumes the semantic embedding
  tensor during `graph_detector_prepare`, records the MH-LGC config in the
  manifest, and rejects cache reuse when the MH-LGC setting changes
- the active MH-LGC adaptation now also exposes:
  - `--mhlgc_contrast_space {fused_x,node_repr,x_new,low_high_concat,low_high,semantic}` to separate
    explicit final detector space, the legacy final-hidden alias, HyperScan
    construction space, HyperScan pre-detector `cat(x_low,x_high)` view-pair
    space, and LM input semantic space
  - `--mhlgc_anchor_source routed_target_mask` so routed hard nodes can be
    used as explicit anchors instead of relying on implicit nonzero guide rows
  - the same routed-target payload now also carries explicit multiview rows
    (`following_view`, `follower_view`, `mutual_view`, `semantic_knn_view`),
    which are consumed by a routed-only detector-space refiner in `GNNs.py`
  - `--mhlgc_pair_mode repair_aware`, which treats the guide as a routed-node
    repair signal rather than only a hard-negative weight
- `graph_detector_prepare` now exports explicit `fused_x` tensors alongside the
  legacy `node_repr` field so same-protocol `SimTeG fused_x` vs
  `HyperScan fused_x` comparisons no longer depend on an implicit alias
- current implementation guardrail: routed multiview refinement is now scoped
  strictly to routed nodes with non-empty auxiliary context. Non-routed rows
  and routed rows with only a self token are skipped instead of receiving a
  silent detector-hidden update.
- current implementation guardrail: semantic-guide caches no longer need to
  match detector `hidden_dim`. `trainer_preparation.py` records the actual
  guide width at runtime and `GNNs.py` inserts a narrow projector before the
  bidirectional MultiAttn/Transformer refiner when dimensions differ.
- current implementation guardrail: for `graph_second_view_fusion=multiattn`,
  routed multiview refinement must also project auxiliary relation/hypergraph
  view tokens into detector space before stacking. Without that, `fused_x`
  (256) and view tokens (`x_low/x_high`, 128) crash at runtime.
- current risk: `repair_aware` is a bounded training adaptation, not yet a
  full explain-guided hyperedge-repair implementation; the classifier path is
  unchanged and the guide now repairs the HyperScan second-view construction
  path for auxiliary contrast, but it still does not directly learn or lock a
  discrete hyperedge-membership editor
- fixed a zero-value CLI bug in `trainer_preparation.py`: explicit
  `--mhlgc_gamma 0.0` (and other zero-valued MH-LGC float args) used to fall
  back to the default through Python truthiness, which invalidated
  structure-only gamma ablations until corrected.
- current risk: this is a social-bot task adaptation of Ou et al.'s
  LLM-guided contrastive idea, not a full reproduction of the fraud paper's
  domain-specific views or official code path
- `LLMbot/model_building.py` now also allows
  `--graph_node_input_family hyperscan_meta_tweet_proxy` to consume a faithful
  preprocessed full-graph tensor through `--embedding_path` when
  `graph_data_variant=full_graph_support` and the tensor is already ordered
  `tweet|num|cat`. This keeps official HyperScan preprocessing outside
  `LLMbot` and avoids silently rebuilding full-graph metadata from the
  labeled-only `norm_user_text` path.
- `LLMbot/trainer_semantic.py` now accepts `--semantic_text_source_path` and
  `--semantic_text_field` for semantic finetune/classifier stages. These flags
  replace `data["user_text"]` rows by `node_id` while preserving full-graph row
  alignment; they are a text-source override, not a split, graph, or router
  protocol change.
- `LLMbot/trainer_semantic.py` answer-token supervision now enforces the DGP
  prompt-consumption contract: the consumed sidecar text must end at the final
  `ASSISTANT_ANSWER:` slot, and prompt tokenization disables implicit special
  tokens before appending the supervised `Yes`/`No` suffix. This removes a
  routed DGP/Qwen PEFT implementation mismatch without changing the method
  boundary.
- `LLMbot/trainer_semantic.py` now also owns `semantic_correction_gate`, a
  routed-node base-aware keep/change stage for existing semantic candidates.
  It is intentionally narrower than another refiner head: no prompt cache is
  rebuilt, candidate LLM/MLP outputs are fixed, a tiny action-wise gate learns
  accept/defer utility from base/candidate probability meta-features, and a
  validation-selected threshold is locked before routed-test replay. Current
  risk: this reduces break relative to direct candidate replacement but can
  still overfit validation on the fixed high-base routed split, so it should be
  reported as exploratory until multi-seed or stronger feature/threshold
  calibration validates it.
- `semantic_correction_gate` now has a `node_attribute` feature family. The
  default `probability` path preserves the previous behavior; the attribute
  path appends train-zscored metadata/tweet/graph cue features to each
  candidate action so the gate can learn candidate competence conditional on
  target-node type rather than only on candidate confidence.
- `semantic_correction_gate` now has a `local_competence` feature family for
  the next gate diagnostic. It keeps probability plus node-attribute action
  descriptors, then appends per-candidate top-k routed-train neighborhood
  competence estimates: local candidate correctness, base-wrong density, fix
  rate, break rate, net utility, and similarity support. This is intentionally
  a gate-feature change only; prompts, candidate semantic outputs, routed split,
  and frozen SimTeG remain fixed.
- `semantic_correction_gate` now has a `defer_softmax` selection policy and an
  optional `break_first` safety policy. The legacy `threshold` path is still the
  default. The new policy trains one `{keep_base, accept_candidate_i}` action
  softmax plus, when enabled, a per-candidate break-risk head whose threshold is
  selected on routed validation nodes. This targets the current failure mode
  directly: independent candidate gates can over-accept break-prone candidates,
  while the new path makes keep-base an explicit action and can reject unsafe
  candidate actions before test replay.
- prompt text for legacy semantic prompts plus v2/v3 explanation prompts is
  now centralized in `LLMbot/prompt.py`, reducing prompt-template sprawl inside
  `precompute.py` while keeping evidence assembly, fallback handling, sidecars,
  and cache writing in the existing helper
- canonical manifests and compatibility traces now explicitly carry
  `deprecated_cli_flags`
- `trainer_glance.py` now owns GLANCE semantic input loading, the public joint
  executor, and migrated internal GLANCE execution lanes
- `joint_router_refinement` now has two public routing protocols:
  - `joint_train`
  - `frozen_router_reuse`
- `frozen_router_reuse` improves comparability for refiner-only evidence
  ablations, but it also increases orchestration complexity inside
  `trainer_glance.py`
- new artifact writes prefer canonical preparation and stage namespaces
- the active mainline now has a first-round explicit full-graph contract behind
  `--graph_data_variant full_graph_support` for:
  - `graph_detector_prepare`
  - conformal-style `estimator_ablation`
  - `joint_router_refinement`
  - diagnostic-only `local_conflict_prune_diag`
- this rollout keeps supervision on the original labeled split while switching
  graph propagation and router feature construction to the `229580`-node graph
  plus runtime concatenation of labeled and support RoBERTa embeddings
- `trainer_preparation.py` now expands labeled targets to full-graph length for
  `neighbor_subgraph + full_graph_support` and scores validation by routed node
  id instead of relying on the shorter labeled-only `y` tensor. This closes the
  majority-class checkpoint-selection bug that previously made
  NeighborLoader-based runs look like MH-LGC failures.
- `graph_detector_prepare` now also has a HyperScan-style graph augmentation
  proxy through `--graph_refine_mode hyperscan_knn_hypergraph_proxy_augment`:
  it constructs feature-KNN local groups from the current Phase-A input tensor
  and materializes them as a new relation via bidirectional center-neighbor
  proxy edges before GNN training
- this path is intentionally labeled a proxy augmentation rather than a full
  HyperScan reproduction: it borrows the similarity-hypergraph grouping idea
  without adding an HGNN branch to the active mainline
- current engineering/research risk is now clearer:
  - routed-node evidence did not support the crude full-graph semantic
    star-expansion proxy as the next enhancement surface
  - the likely failure mode is not "similarity groups are useless", but
    "global KNN proxy edges are too coarse and insufficiently relation-aware"
- `graph_detector_prepare` therefore now also supports
  `--graph_refine_mode relation_overlap_knn_proxy_augment`:
  - it keeps the same proxy-edge materialization contract for engineering
    simplicity
  - but it restricts candidate semantic grouping to undirected relation 1-hop
    ego neighborhoods and can center only on routed nodes through
    `--routed_nodes_path` plus `--routed_nodes_split`
  - this is the current bounded follow-up for post-router graph augmentation;
    it should still be reported as a local relation-overlap proxy rather than a
    faithful HyperScan hypergraph implementation
- `graph_detector_prepare` now also supports
  `--graph_refine_mode relation_overlap_knn_repr_prefit_augment`:
  - this is the next mechanism correction after the negative overlap-only test
  - it uses a relation-view prefit `node_repr` plus the original Phase-A input
    as the local similarity space, instead of using the Phase-A input alone
  - the goal is to move the KNN grouping step closer to HyperScan's actual
    `x_low + x_in` construction without yet introducing a separate HGNN branch
- `graph_detector_prepare` now also supports
  `--graph_second_view_scope routed_nodes` with
  `--graph_backbone rgcn_hyperscan`:
  - this is the current direct response to the main failure diagnosis from the
    routed-node KNN experiments
  - it stops treating similarity groups as static proxy edges and instead runs
    a training-time HypergraphConv branch over routed-node local KNN groups
  - those groups are rebuilt from the current relation-view `x_low` plus the
    original Phase-A input `x_in`, matching the key HyperScan geometry more
    closely than the previous two graph-rewrite ablations
  - the second-view realization is now split into parser axes:
    `--graph_second_view_hypergraph_backend {pyg,dhg}` and
    `--graph_second_view_fusion {residual,multiattn,multiattn_adaptive,construct_acm}`
  - use `--graph_second_view_scope labeled_prefix` for the all-labeled-center
    dynamic branch; `routed_nodes` now denotes explicit routed-node centers
    from `--routed_nodes_path`
  - `--graph_second_view_candidate_policy` supports diagnostic hard/stable
    support ablations for routed dynamic hyperedges. The trainer tags
    candidates without ground-truth labels from routed membership, risk vectors,
    and base predictions; `hypergnn.py` enforces the quota inside forward over
    the current `x_new` geometry. Do not preselect quota members statically
    outside the model unless the experiment is explicitly labeled as a
    non-HyperScan control.
  - `neighborloader_batch` now also supports the non-quota member-filtering
    policies `exclude_routed` and `post_topk_exclude_routed` by mapping the
    routed-node mask onto each sampled subgraph through `batch.node_id`.
    This changes only KNN hyperedge members; it does not turn the
    NeighborLoader seeds into routed-only training nodes. Quota policies remain
    outside the batch-local contract.
  - `LLMbot/parser_args.py` and `LLMbot/trainer_preparation.py` now also expose
    `--graph_neighborloader_contract {seed_only,hyperscan_sampled_subgraph}` so
    sampled-subgraph supervision/evaluation semantics are explicit instead of
    implicit. The faithful contract matches released HyperScan-style
    train/valid/test repeated-row scoring and validation-accuracy checkpoint
    selection, while preserving deduplicated full-graph `outputs.pt` for
    downstream router/LLM consumers.
  - remaining risk: this is still an active-mainline migration surface, not a
    full released-code reproduction claim; experimental gains or failures
    should still be discussed as HyperScan-style evidence unless the full
    external-code protocol is satisfied
- `local_conflict_prune_diag` now emits full-graph-specific failure artifacts:
  `error_migration`, `support_exposure`, `support_consistency`,
  `structural_shock`, `error_regime_shift`, and four failure mechanisms
  (`sparse_isolated`, `dense_directional_recoverable`,
  `conflict_dominant`, `support_induced_fragile`)
- engineering risk: the full-graph local-conflict stage is materially heavier
  than the labeled-graph path because it evaluates single-edge counterfactual
  deletions on the support-augmented graph; reuse an existing full-graph
  `graph_detector_prepare` artifact and treat low-budget smoke validation as
  the default engineering check before any long full-budget diagnostic run
- the conformal full-graph ablation path is intentionally bounded to
  labeled-prefix evaluation and labeled-graph subgroup analysis, rather than
  pretending oracle labels exist for the support-node suffix
- a new `calibrated_local_risk_router` ablation now sits between
  `posthoc_calibrated_ranker` and heavier graph-aware conformal routes:
  it keeps posterior calibration scalar-only and applies local graph structure
  only to the risk object
- the current implementation is now the stronger v2 variant, not the original
  coarse fixed-family draft:
  - localized 1-hop relation/direction-aware aggregation
  - optional `node_repr` similarity weighting
  - tune/cal split discipline for score-family selection vs conformal threshold
  - upgraded manifest contract `relation_aware_scalar_risk_aggregation_v2`
- `conformal_knn_risk_router` is now the required target-node KNN-risk
  baseline for future router work:
  - the routed object is always the target node; KNN neighbors are the target's
    local support group for risk evidence, not the selected objects
  - by default, the router now inherits `k`, candidate scope, and `x_new`
    representation from a consumed HyperScan-style frozen G0 second-view
    artifact; explicit `--conformal_knn_*` flags remain the override path
  - HyperScan-style frozen G0 artifacts now export true `x_low` / `x_new`
    tensors in `outputs.pt`, so router-side `x_new` can stop approximating the
    second-view space from `node_repr` when the artifact was prepared on the
    updated mainline
  - HyperScan-aligned router and LLM-KNN evidence runs now require exported
    `outputs["x_new"]`; missing `x_new` is a regeneration issue, not permission
    to reconstruct the construction space from final-hidden `node_repr`
  - manifests record the effective inherited-or-explicit config so later router
    experiments can verify that the risk estimator and Hyper KNN branch used
    matched support semantics
  - for `conformal_knn_candidate_scope=hyperscan_full`, `k` is the HyperScan
    hyperedge size including the target center, so the router consumes at most
    `k - 1` non-center support members for target risk estimation
  - the full-pool KNN backend is exact batched `torch` top-k over normalized
    feature vectors, replacing high-dimensional cKDTree to keep the same KNN
    ordering while making the full 229k-node graph run practical
  - the router now exposes an NCP-style local calibration path:
    `--conformal_knn_ncp_lambda` controls official NCP-like
    `exp(-distance/lambda_L)` support weighting, and
    `--conformal_knn_learning_mode ncp_local` computes target-specific
    conformal features from KNN calibration neighbors and selects among
    nonparametric NCP-local score families without a logistic head
  - non-model support-quality sweeps are available through
    `--conformal_knn_neighbor_mode`,
    `--conformal_knn_similarity_threshold`, `--conformal_knn_min_support`,
    `--conformal_knn_adaptive_max_k`, and
    `--conformal_knn_hubness_correction`; these flags filter or reweight KNN
    evidence only and must not be described as graph augmentation, LLM
    intervention, or downstream Conformal Risk Control
  - `--conformal_knn_learning_mode fixed` remains the compatibility/default
    path for historical fixed score-family ablations
  - compare later learned / LLM-consuming / refiner-selection routers against
    this artifact under the same frozen G0, split, budget, and labeled-only
    KNN scope
  - report AUROC-error, AUPRC-error, AURC, ErrRecall@K, Precision@K, Lift@K,
    selected-node overlap, and downstream refiner gain per selected node when
    a refiner is involved
  - do not claim a new router improves hard-node selection unless it beats or
    clearly complements this target-node Conformal-KNN baseline under matched
    conditions
  - require future router proposals to state how target-only evidence,
    target->KNN support-group evidence, and optional LLM/refiner evidence are
    fused into a target-node risk score
  - use `--conformal_knn_score_family_override base_only` as the strict
    target-only control and the default `auto` setting as the target+KNN
    support-group router
  - current HyperScan-aligned router interpretation is now explicit:
    the outer neighborhood prior is the frozen G0 HyperScan-style
    `KNN(x_new, k)` support group, while the router itself is a post-hoc
    risk estimator over that support group rather than a second HGNN/fusion
    classifier head
  - current evidence says pure `same_hyperedge_selected_tail` is not a viable
    router score in this pipeline: it treats semantic-group outlierness as
    error risk and collapses target ranking quality
  - restricting the inner risk reference to
    `selected ∩ calibration_idx` repairs much of that collapse but remains
    sample-starved; the local calibration effective sample size stays well
    below the raw support size on average
  - the current stabilized follow-up is an ESS-shrunk
    `same_hyperedge_calibration_shrunk_tail` variant: it keeps the HyperScan
    support prior fixed, computes a calibration-only inner-tail risk inside
    that group, and shrinks back toward the broader same-hyperedge global-tail
    risk when the local calibration evidence is sparse
  - treat `same_hyperedge_calibration_shrunk_tail` as the fixed same-hyperedge
    router anchor for future method comparisons; `auto` may still be logged
    for exploratory family search, but anchor-grade reporting should compare
    against this explicit family
  - current research risk remains: this router line is now structurally aligned
    with the HyperScan neighborhood prior, but the best-performing score is
    still closer to a stabilized local-global conformal hybrid than to a pure
    same-group local conformal estimator

Current reviewer summary:

- `code-reviewer` recommendation: `COMMENT`
- `architect` status: `WATCH`
- `verifier` status: `PARTIAL`
- overall conclusion: `COMMENT`

## Governance Status 2026-05-26

Current governance conclusion:

- public contract cleanup: substantially complete
- documentation alignment: substantially complete
- execution-body extraction: not complete
- OMX `ralph`-style maintainability end-state: not reached yet

What has been achieved:

- `LLMbot/` is the only active mainline
- canonical tasks, canonical flags, and canonical artifact namespaces define
  the public operator-facing surface
- the active code-development docs are synchronized around the same mainline
  interpretation
- internal-only GLANCE work is explicitly separated from the public CLI and
  current claim boundary

What remains open:

- the implementation still depends heavily on `trainer_legacy_impl.py`
- the remaining concentration is now mostly estimator/matrix paths, local
  conformal helper tails, compatibility paths, and duplicate legacy blocks
- legacy compatibility fields and runtime-only helper naming are still present
- downstream modules still need canonical-field cleanup even though `main.py`
  now consumes the core `StageSpec` runtime gates
- the iter-2 comparison lane is intentionally cross-seed compatible rather than
  per-seed semantic-faithful because only `finetuned_roberta_embeddings_iter_2_seed1.pt`
  exists locally
- the first-round full-graph path is intentionally narrow and does not yet
  cover semantic finetune, graph calibration, or refiner-only semantic
  override branches

## Risks Already Reduced

### RESOLVED-01 Graph calibrator reuse now validates graph-detector provenance

Current behavior:

- graph calibrator reuse validates:
  - `input_g0_outputs_sha256`
  - graph backbone
  - feature path
  - split provenance

Why it matters:

- this closes the earlier high-risk path where a changed graph detector could
  silently reuse an incompatible calibration artifact

### RESOLVED-02 Strict GLANCE no longer falls back to seed-1 semantic artifacts

Current behavior:

- strict GLANCE paths now require:
  - same-run preparation provenance
  - and for public `joint_router_refinement`, semantic input that matches the
    current run's `graph_detector_prepare` `feature_manifest.path`

Why it matters:

- this removes hidden cross-seed and cross-root provenance shortcuts that would
  change the meaning of multi-seed experiments while still looking
  paper-aligned at the training loop level

### RESOLVED-05 Strict GLANCE router homophily proxy and scorer training are now aligned to the current strict contract

Current behavior:

- strict GLANCE now trains a lightweight auxiliary MLP `Q` on node features
- router soft-local-homophily is derived from `Q` probabilities rather than a
  logistic-regression proxy
- the learned `router_score` is now owned by `LLMbot/router.py` and comes from
  a reliability-first MLP trained on `base_wrong` with pairwise ranking,
  while `oracle_advantage` remains a post-hoc counterfactual reward trace only
- strict router features now temperature-scale the base GNN confidence signals
  and add direction-aware social reliability features for TwiBot20, including
  `in/out` degree, relation counts, reciprocity, directional
  homophily/disagreement, and sparse-node indicators
- the public strict stage now records whether router/refiner training used the
  paper-style `3000`-node cap or a full-train TwiBot20-adapted regime via
  `joint_train_node_cap`, `effective_train_node_count`, and `train_cap_mode`

Why it matters:

- this is less paper-faithful than the previous advantage-router branch, but it
  better matches the current TwiBot20 reliability-routing objective and
  reduces one of the main task-mismatch risks identified in recent diagnosis

### RESOLVED-06 Public strict GLANCE evaluation no longer hard-codes batch top-k inference

Current behavior:

- strict GLANCE training still uses the paper-text batch top-k schedule
- public final evaluation now scores the whole validation/test split with the
  learned router
- candidate global budgets come from `--risk_budgets`
- the final budget is selected on validation only and then locked on test

Why it matters:

- this removes a TwiBot20-misaligned inference artifact where fixed per-batch
  `K=8` could consume about 25% query budget while still under-covering wrong
  nodes globally

### RESOLVED-07 Prompt-expert refiner can abstain before changing predictions

Current behavior:

- `prompt_expert_bundle_v1` now supports the explicit keep/change gate used by
  the public joint refiner
- `--joint_refiner_gate_target utility_positive` trains that gate toward
  positive raw-refiner utility instead of only base-wrong status
- stage artifacts record gate rates and per-node `gate_prob` / `gate_decision`

Why it matters:

- current prompt experts can repair some base-wrong nodes but also break
  correct nodes; the abstain gate tests whether preservation can improve before
  introducing heavier multi-expert fusion

### RESOLVED-07B Correction-selector calibration is now stage-owned

Current behavior:

- `joint_router_refinement` now promotes the META-DES / correction-MoE
  threshold sweep from post-hoc analysis into the stage contract through
  `--joint_correction_moe_gate_calibration`
- `per_action_threshold` fits one utility threshold per selected action on
  validation and locks the calibrated keep/change policy before test evaluation
- `--joint_correction_moe_utility_target net_gain` keeps true fixes positive
  while treating base-correct/expert-wrong actions as negative utility
- `--joint_correction_moe_ranking_weight` adds within-node pairwise action
  ranking over fix / no-gain / break rewards
- stage artifacts now persist calibration metadata, calibrated gate decisions,
  selected action scores, break targets, and utility rewards

Why it matters:

- this closes the earlier evidence gap where per-action calibration produced a
  weak positive result only as a side artifact; future runs can now compare
  calibrated policies from first-class `metrics.json`, `manifest.json`,
  `outputs.pt`, `checkpoint.pt`, and `per_node_test.jsonl`

### RESOLVED-07 Public strict GLANCE now supports routed-refiner target and weighting diagnostics

Current behavior:

- `joint_router_refinement` can now optionally:
  - add an explicit keep/change gate inside the routed refiner
  - train the routed refiner either for direct label prediction or explicit
    keep/change targets
  - reweight routed-node losses toward `base_wrong`, `oracle_advantage > 0`, or both
  - switch prompt-expert fusion from fixed `projector_concat` to the refiner-only
    `mpe_gated` softmax over `graph_following`, `graph_follower`, `tweet`, and
    `conflict`
  - run a narrower raw-concat GLANCE refiner through `raw_concat_follower_tweet`
    to isolate `[z_gnn || graph_follower || tweet || structural_side_channel]`

Why it matters:

- the recent evidence suggests router ranking is improving faster than refiner
  conversion quality, so the strict joint path needs a way to test refiner-side
  fixes without changing the router contract again

### RESOLVED-08 Prompt-expert ego explanation loading no longer blocks on remote HF downloads

Current behavior:

- `LLMbot/precompute.py` now resolves `--model_path` and `--explain_model_path`
  in offline-first mode
- cached HF repo ids are converted into concrete local snapshot paths before
  loading
- `expert_ego` and the `ego` branch inside `expert_concat_v1` now fall back to
  deterministic profile-card explanations when no local explain-model snapshot
  is available

### RESOLVED-09 Prompt-expert v2 precompute no longer needs full labeled-node explain passes

Current behavior:

- `LLMbot/precompute.py` now accepts `--routed_nodes_path` so prompt-expert
  prompt building, explanation generation, and encoder inference can run only
  on an explicit routed-node subset
- the same helper scatters those selected-node tensors back into a full-graph
  cache with zero-filled unselected rows, preserving the existing
  `joint_refiner_embedding_path` contract
- v2 explanation generation now reuses one loaded local explain model across
  all expert components in a run instead of reloading per component
- `--explain_required` now fails fast when a real LLM-as-explainer run cannot
  load or generate from a local instruct-model snapshot
- local model resolution no longer treats `LLMbot/models/` as a valid HF
  snapshot; parent directories are accepted only when they contain exactly one
  valid local model/snapshot child
- `precompute.py --project_name ...` now starts an optional wandb run and logs
  long-running prompt-bundle, per-batch explanation, encoding, and final
  artifact progress
- v2 explanation generation now writes component-level resumable sidecars keyed
  by `node_id + prompt_hash`, and concat runs emit component-only `.pt` caches
- v2 prompt-expert explanation encoding now defaults to the SimTeG finetuned
  RoBERTa family rather than `roberta-base`, and records the requested/effective
  component token budgets plus the SimTeG-compatible 512-token truncation and
  no-special-token final-hidden-state mean pooling contract in the manifest
- when the seed-specific frozen SimTeG LM checkpoint is present, the v2
  explanation encoder loads `TwiBot-20_seed_<seed>/checkpoints/LM_pretrain/best.pkl`
  before encoding; `--finetuned_roberta_checkpoint_path` exists to pin that
  source explicitly
- missing finetuned-RoBERTa weights are now a hard precompute error, not a
  silent fallback to `roberta-base`
- `precompute.py --prompt_mode ultratag_s_subgraph_v1` supports routed-test and
  full-graph target scopes. Routed-node runs still require
  `--routed_nodes_split test`; full-graph runs omit `--routed_nodes_path` and
  use `--center_node_scope all_graph_nodes`. For the current routed-node
  diagnostic, keep `--graph_data_variant labeled` for the output SimTeG/GNN row
  layout and set `--context_graph_variant full_graph_support` only for
  neighbor evidence. Use `--ultratag_virtual_edge_policy none
  --ultratag_edge_reconfig false` for the text-only diagnostic that avoids
  soft-label-derived graph rewiring. The path still does not implement the full
  UltraTAG-S dual-GNN structure-learning stage.
  for cheaper single-expert reuse
- the prompt-expert explainer batch default is now automatic
  (`--explain_batch_size 0`: CUDA=2, CPU=1) to avoid accidentally launching
  long server jobs with batch-1 generation
- `--prompt_family_version v3` now keeps the same routed-node, component
  sidecar, SimTeG finetuned-RoBERTa, and full-graph scatter contracts as v2, but
  records `semantic_view_mode=prompt_expert_bundle_v3` and
  `evidence_schema=structured_evidence_card`
- v3 prompts request source-grounded evidence-card fields and explicitly forbid
  final labels, probabilities, confidence scores, recommendations, and
  base-model correction instructions
- v3 default cache stems include `_v3_` / `concat_v3` so structured-card caches
  do not overwrite summary-first v2 caches
- `trainer_glance.py` now recognizes `prompt_expert_bundle_v3` and consumes it
  through the same prompt-expert refiner path and directional count/presence
  graph-gate input as v2

Why it matters:

- this reduces the wall-clock and VRAM churn of explanation-first prompt-expert
  studies while avoiding a second alignment shim inside `trainer_glance.py`
- this prevents deterministic fallback explanations from being mistaken for a
  completed Qwen Instruct explainer run
- it keeps the remote GPU path explicit: embedding weights and explain-model
  weights are separate local assets, and real explainer experiments should
  point to the uploaded instruct snapshot
- it separates a source-grounded correction-evidence cache from the v2 summary
  cache without claiming representation repair or LLM-as-predictor behavior

### RESOLVED-03 LOGIN claim boundary is now explicitly downgraded

Current behavior:

- LOGIN-related metadata records:
  - `official_code_verified = false`
  - `paper_aligned = true`
  - `repo_locally_verified = false`
  - `verified_scope = node_selection_uncertainty_only`

Why it matters:

- this prevents the current implementation from being mistaken for a locally
  verified official-code reproduction

### RESOLVED-04 Public CLI surface is now canonicalized

Current behavior:

- public task names are canonical
- public flags are canonical
- hidden legacy aliases remain parse-compatible for one migration window

Why it matters:

- the public operator surface is now substantially cleaner and easier to keep
  stable

## Remaining Implementation Risks

### IR-01 [HIGH] `trainer_legacy_impl.py` still concentrates estimator/matrix and compatibility responsibilities

Evidence:

- `LLMbot/trainer.py` is now only a facade
- first-layer owners are now real for contracts/runtime, preparation,
  semantic finetune, and legacy distillation
- `trainer_graph.py` now owns the shared graph runtime layer, public graph
  diagnostics, and the full `local_conflict_prune_diag` structural-conflict
  path
- `stage_runner.py` now owns the shared runtime skeleton
- `trainer_glance.py` now owns the shared GLANCE input layer and the public
  joint executor plus migrated internal GLANCE execution lanes
- GLANCE runner/helper tails now reside in `trainer_glance.py`
- estimator/matrix, local conformal helper tails, and compatibility paths still
  reside in `LLMbot/code/trainer_legacy_impl.py`
- the file still contains historical duplicate preparation and semantic blocks
  during the current rebinding window

Risk:

- local edits can still cross preparation, graph, estimator, and GLANCE
  responsibilities too easily
- first-layer owner migration reduced some coupling, but the remaining
  estimator/matrix and compatibility body is still structurally dominant

Recommended next work:

1. split estimator/matrix compatibility cleanup from algorithm cleanup so
   `estimators.py`, `model_building.py`, and `router.py` can be reviewed
   independently
2. move remaining local conformal helper tails out of
   `trainer_legacy_impl.py` after a focused reference scan
3. continue removing legacy shadow-field reads from active owner modules while
   keeping parser compatibility intact

### IR-02 [MEDIUM-HIGH] active code still backfills and sometimes reads legacy shadow fields

Evidence:

- canonical parser fields now exist:
  - `args.experiment_task`
  - `args.graph_backbone`
  - `args.text_encoder`
  - `args.semantic_encoder`
  - `args.embedding_path`
- compatibility shadow fields are still backfilled:
  - `args.stage`
  - `args.GNN_model`
  - `args.LM_model`
  - `args.semantic_backbone`
  - `args.emb_path`
  - `args.g0_feature_path`

Risk:

- if active mainline code keeps reading legacy shadow fields directly, the
  compatibility window can turn into permanent naming debt

Recommended next work:

1. migrate active internal reads toward canonical parser fields
2. keep hidden aliases parse-only during the remaining migration window

### IR-03 [MEDIUM] runtime-only artifact naming is still in transition

Evidence:

- canonical preparation writes now exist
- some runtime helper outputs still retain compatibility fallback behavior, such as:
  - `runtime/router_oof/...`
  - `runtime/lm_only_head/...`
  - legacy fallback locations in historical paths

Risk:

- runtime-only artifacts can still accumulate naming drift and provenance noise
  if their namespaces are not finished cleanly

Recommended next work:

1. define fixed runtime-only artifact namespaces
2. keep them separate from public stage artifacts and preparation artifacts

### IR-04 [MEDIUM] internal GLANCE branches are now hidden, but their boundary text still needs continued cleanup

Evidence:

- internal-only stages are now explicit in `LLMbot/code/stage_registry.py`
- public docs already avoid advertising them as public tasks
- some internal helper text and runtime notes still reflect older terminology

Risk:

- later agents may still misunderstand internal-only branches as operator-facing
  or claim-grade features unless the boundary stays explicit everywhere

Recommended next work:

1. continue canonicalizing internal manifest metadata
2. keep public docs limited to public canonical tasks

### IR-05 [LOW] `StageSpec` runtime gates now drive `main.py`, but downstream owners still need cleanup

Evidence:

- `LLMbot/code/stage_registry.py` now defines:
  - `runner_kind`
  - `claim_grade_allowed`
  - `requires_canonical_split`
  - `forces_use_gnn`
  - `graph_data_mode`
- `LLMbot/code/main.py` directly consumes:
  - `runner_kind`
  - `claim_grade_allowed`
  - `requires_canonical_split`
  - `forces_use_gnn`
  - `graph_data_mode`

Risk:

- the main runtime gate is now registry-backed, but downstream owner modules
  still contain legacy shadow-field reads and compatibility-specific naming

Recommended next work:

1. keep `StageSpec` as the main runtime policy source
2. avoid adding new parallel stage-policy tables outside `stage_registry.py`
3. migrate downstream owner modules toward canonical parser fields

## Remaining Maintainability Risks

### MR-01 [HIGH] naming cleanup is underway, but two vocabularies still coexist internally

Evidence:

- public naming is canonical
- internal compatibility and historical helper naming still contain terms such as:
  - `frozen_g0`
  - `semantic_finetune`
  - `glance_joint_router_refine`
  - `vertical_minimal`

Risk:

- future agents may reintroduce legacy terminology into code, manifests, or docs

Recommended next work:

1. forbid new public-facing legacy names
2. continue second-pass cleanup inside helpers, notes, and internal metadata

### MR-02 [MEDIUM-HIGH] canonical manifest injection has improved, but side-artifact coverage still needs auditing

Evidence:

- shared manifest-provenance helpers now exist
- some stage outputs have moved onto canonical manifest writers
- stage-local JSON and helper outputs still need ongoing audit

Risk:

- if only the main manifest is canonical while side outputs keep historical
  wording, documentation drift returns quickly

Recommended next work:

1. continue auditing stage-local JSON, notes, and dependency manifests
2. keep public naming canonical in every newly written artifact

### MR-03 [MEDIUM] parser-only help path is a quality win that must remain protected

Evidence:

- `LLMbot/main.py --help` now has an early parser-only path

Risk:

- if later edits tie help back to heavy runtime imports, fast CLI verification
  will regress

Recommended next work:

1. keep parser/help validation independent from training-runtime availability
2. treat this as a regression-sensitive developer contract

### MR-04 [MEDIUM] documentation is much closer to the code, but it must now stay in lockstep

Evidence:

- active docs have been updated toward the canonical mainline
- the implementation is still evolving, especially around extraction from
  `trainer_legacy_impl.py`

Risk:

- if documentation updates stop, the next drift will come from structure and
  naming changes that are no longer reflected in the developer-facing docs

Recommended next work:

1. update the relevant code-development docs every time the code changes
2. treat doc sync as part of completion, not later cleanup

## Required Documentation Sync For Future Code Changes

When the active mainline changes, update the matching docs in the same task:

- parser, public flags, task exposure, examples:
  - `LLMbot/README.md`
  - `docs/code/parser.md`
- runtime flow, module boundaries, extraction status:
  - `docs/ARCHITECTURE.md`
  - `docs/code/research.md`
- maintainability hotspots, unresolved debt, migration risk:
  - `code.md`

This rule is part of the active governance contract, not a best-effort note.

## Current Alignment Conclusion

The mainline is no longer in a severe public-contract mismatch state.

What is true now:

- the public contract is canonicalized
- the implementation is real and usable
- the code no longer treats deprecated `baseline/core` as the active source of truth

What is still true:

- the execution body remains in transition
- internal naming and runtime artifact cleanup are not finished
- the next phase is structural extraction and continued maintainability cleanup,
  not another public-contract redesign

## Governance Phase 2 Candidate Slices 2026-07-04

Before introducing any deeper directories, keep all active implementation under
`LLMbot/code/`. Deletion and function consolidation should happen inside that
flat source surface first.

1. Flat-source cleanup and dead-code deletion:
   identify duplicate or unreachable code inside `LLMbot/code/` and remove it
   in small slices with CLI/import checks. Do not create `stages/`, `methods/`,
   or other taxonomy directories in this slice.
2. Local conformal helper owner transfer:
   move remaining local conformal router/helper tails from
   `code/trainer_legacy_impl.py` into `code/trainer_graph.py` without changing
   stage outputs.
3. GLANCE/helper tail owner transfer:
   move remaining prompt-expert and relation-aware helper tails from
   `code/trainer_legacy_impl.py` into `code/trainer_glance.py`, with behavior
   locked by CLI/artifact checks before code movement.
4. Runner script governance:
   define naming, queue manifest, and log-path rules for `run_*.py` and a
   future `runners/` target, then migrate at most one low-risk runner first.

## Next Refactor Plan 2026-07-04

This supersedes the 2026-05-24 path assumptions. Active implementation now
lives under `LLMbot/code/`, and the next code-governance phase should stay
inside that flat source directory.

Ordered implementation plan:

1. Flat-source cleanup first.
   - target: `LLMbot/code/`
   - remove duplicate, unreachable, or obviously retired code in small slices
   - do not introduce `stages/`, `methods/`, or other package taxonomy
2. Preparation and semantic owner cleanup second.
   - target files:
     - `LLMbot/code/trainer_preparation.py`
     - `LLMbot/code/trainer_semantic.py`
     - `LLMbot/code/trainer_legacy_impl.py`
   - exit condition:
     - preparation and semantic flows no longer rely on the monolithic module
       as their primary logic owner
3. Canonical parser-field migration third.
   - target files:
     - `LLMbot/code/main.py`
     - extracted trainer modules under `LLMbot/code/`
   - rule:
     - active reads use canonical fields
     - legacy fields remain backfilled for compatibility only
4. Runtime policy audit fourth.
   - target files:
     - `LLMbot/code/main.py`
     - `LLMbot/code/stage_registry.py`
     - `LLMbot/code/stage_runner.py`
   - goal:
     - keep `runner_kind`, `claim_grade_allowed`, `requires_canonical_split`,
       `forces_use_gnn`, and `graph_data_mode` consumed through `StageSpec`
       and prevent new parallel policy tables
5. Graph and GLANCE execution extraction fifth.
   - target files:
     - `LLMbot/code/trainer_graph.py`
     - `LLMbot/code/trainer_glance.py`
     - `LLMbot/code/stage_runner.py`
   - goal:
     - finish transferring local conformal helper tails and remaining GLANCE
       execution branches out of `LLMbot/code/trainer_legacy_impl.py`
6. Runner and runtime-only artifact cleanup last.
   - target:
     - root `LLMbot/run_*.py`, launch scripts, queue manifests, and log rules
   - goal:
     - define runner naming and artifact namespaces before migrating at most
       one low-risk runner script

This plan is intentionally structural. It should reduce maintenance risk
without changing the current research claim boundary or expanding the public
task surface.
