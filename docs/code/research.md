# Research Pipeline And Code Alignment

This document records how the current research pipeline maps onto the active
mainline under `LLMbot/`. Active Python source lives in `LLMbot/code/`; root
`LLMbot/main.py`, `LLMbot/precompute.py`, and `LLMbot/preprocess.py` remain
operator-facing compatibility entrypoints.

It is a code-alignment note, not an experiment result or paper-claim document.

## Fixed Active-Mainline Interpretation

The active mainline is a staged post-hoc pipeline centered on:

1. semantic feature source resolution
2. graph detector posterior generation
3. post-hoc estimator/router diagnostics
4. optional local graph-aware refinement branches

The public CLI also still exposes `distillation_pipeline` as a compatibility
pipeline task for end-to-end teacher/student training flow inside the active
mainline.

The codebase still contains more implemented branches than the public CLI
surface exposes. Public contract and internal implementation inventory must be
kept separate.

## Governance Boundary Snapshot 2026-07-04

The active research pipeline is split into four governance surfaces:

- orchestration surface: `code/parser_args.py`, `code/main.py`,
  `code/stage_runner.py`, and `code/stage_registry.py`
- stage owner surface: `code/trainer_preparation.py`,
  `code/trainer_semantic.py`, `code/trainer_graph.py`,
  `code/trainer_glance.py`, and `code/trainer_distillation.py`
- algorithm surface: `code/model_building.py`, `code/estimators.py`,
  `code/router.py`, `code/GNNs.py`, `code/operators.py`, and
  `code/hypergnn.py`
- artifact surface: `experiments/`, `server_logs/`, stage manifests, metrics,
  checkpoints, outputs, queue manifests, and logs

`code/precompute.py` is an independent bounded context, reached by the root
`precompute.py` compatibility entrypoint. It prepares prompt/evidence and
embedding caches, then hands those artifacts to the main pipeline through
explicit path flags such as `--joint_refiner_embedding_path`,
`--mhlgc_semantic_embedding_path`, or
`--graph_second_view_llm_edge_retain_path`. It should not be described as a
normal public stage unless it is later moved into `stage_registry.py` with an
explicit `StageSpec`.

First-stage governance and the flat `code/` source-location migration do not
change research meaning, public task exposure, artifact schema, or evaluation
logic. They fix terminology and migration order so later implementation work
can be reviewed in small slices.

## Refactor Naming Standard 2026-07-04

Research-code changes now follow a naming-first rule: new public commands,
docs, manifests, and owner-module code use canonical names
`experiment_task`, `graph_backbone`, `text_encoder`, `semantic_encoder`,
`embedding_path`, `graph_detector_prepare`, and `fused_x`. Legacy spellings
`stage`, `GNN_model`, `LM_model`, `semantic_backbone`, `emb_path`,
`g0_feature_path`, `frozen_g0`, and `node_repr` are compatibility aliases or
historical artifact vocabulary only.

`StageSpec.canonical_name` is the experiment-task dispatch source of truth.
`StageSpec` remains the class name during the compatibility window, while new
orchestration code should use `task_spec`, `requested_task`, and
`execution_task` as local names. Medium cleanup may remove hidden
deprecated-only task values such as `eqc_v8_matrix`, but it does not remove
public canonical task names or common renamed CLI aliases in the same slice.
Resolved experiment-config snapshots serialize both the compatibility
`stage_spec` field and the canonical-local `task_spec` field as primitive
dictionaries; runtime dataclass instances must not leak into JSON artifacts.
This slice also merges single-owner implementation files into their owners:
DIGNN conflict refiner logic into `trainer_dignn_conflict.py`, `LM_Model` into
`model_building.py`, and `RGTLayer`/`SimpleHGNConv` into `GNNs.py`. The
distillation owner surface is also consolidated: `trainer_distillation.py`
owns `LM_Trainer`, `GNN_Trainer`, `MLP_Trainer`,
`_safe_pseudo_label_training_index`, and `run_legacy_graph_seed`, while
`trainer_legacy_impl.py` keeps only compatibility aliases for those names.

## Stage Alignment Snapshot

### Phase A: semantic encoder -> GNN detector

Current code anchors:

- `LLMbot/code/main.py`
- `LLMbot/code/artifact_contracts.py`
- `LLMbot/code/runtime_env.py`
- `LLMbot/code/trainer_preparation.py`
- `LLMbot/code/trainer_semantic.py`
- `LLMbot/code/trainer_distillation.py`
- `LLMbot/code/trainer_legacy_impl.py`
- `LLMbot/code/model_building.py`

Public tasks:

- `distillation_pipeline`
- `semantic_encoder_finetune`
- `semantic_embedding_classifier`
- `graph_detector_prepare`
- `graph_calibration_prepare`
- `semantic_source_ablation`

Internal task:

- `phase_a_single_cell_internal`

Interpretation:

- Phase A is the active preparation layer
- preparation artifacts now use canonical namespaces under `preparation/`
- contract/path helpers and runtime/device helpers are now split out of the
  execution body into `artifact_contracts.py` and `runtime_env.py`
- `trainer_preparation.py` now owns graph-detector and graph-calibrator
  preparation execution
- `trainer_semantic.py` now owns semantic finetune execution and the direct
  cached-embedding classifier ablation
- `trainer_semantic.py` now also supports routed-node-specialized semantic
  experiments through `--routed_nodes_path`:
  - routed json `split_counts` replace canonical `train/valid/test` indices
    for `semantic_encoder_finetune` and `semantic_embedding_classifier`
  - this is the active-mainline path for comparing node-specialized semantic
    adaptation on the frozen high-base routed set without changing the graph
    detector or router protocol
- `trainer_semantic.py` also accepts `--semantic_text_source_path` for those
  two semantic stages. The sidecar is read by `node_id` and replaces
  `data["user_text"]` rows while preserving full-graph row alignment. This is
  the bridge for DGP-style prompt tuning experiments: the DGP prompt sidecar can
  feed `semantic_encoder_finetune --semantic_encoder qwen3_peft`, while the
  corresponding embedding cache feeds `semantic_embedding_classifier` for the
  CALM-style query-embedding-MLP comparison.
- `semantic_encoder_finetune` also accepts
  `--semantic_supervision_mode {classifier,answer_token}`:
  - `classifier` keeps the compatibility hidden-state classifier-head CE path
  - `answer_token` is the active DGP-faithful upgrade for
    `--semantic_encoder qwen3_peft`: the model is supervised on the final
    prompt answer slot, with `Yes -> bot` and `No -> human`
  - the semantic stage now enforces an explicit DGP prompt-consumption
    contract: the sidecar text must end at the final `ASSISTANT_ANSWER:` slot,
    and prompt tokenization disables extra special-token insertion before the
    supervised `Yes`/`No` suffix is appended
  - evaluation remains deterministic and non-generative at runtime by scoring
    the conditional completion likelihood of `Yes` versus `No` and writing the
    result back into the same `outputs.pt` binary-classification contract
- `semantic_encoder_finetune` with `--semantic_encoder roberta_finetuned`
  now initializes from the existing SimTeG LM checkpoint
  `TwiBot-20_seed_<seed>/checkpoints/LM_pretrain/best.pkl` when available, or
  from `--finetuned_roberta_checkpoint_path` when pinned explicitly
- `trainer_distillation.py` now owns the legacy distillation trainers
- `semantic_embedding_classifier` is the narrow Phase-A direct-classification
  ablation:
  - input is a cached semantic embedding tensor, typically the frozen SimTeG
    finetuned-RoBERTa embedding resolved from `--embedding_path` or the
    seed-aware default
  - supervision remains the canonical `train/valid/test` split on the labeled
    prefix unless `--routed_nodes_path` is provided, in which case it switches
    to routed `train/valid/test`
  - the classifier is a lightweight MLP over embeddings only; it does not run
    graph propagation, router logic, or refiner logic
  - this makes it the cleanest active-mainline answer to
    "can the frozen semantic embedding alone classify TwiBot20 nodes?"
- `semantic_correction_gate` is the active base-aware routed-node correction
  gate:
  - input is a frozen SimTeG `outputs.pt` plus one or more existing semantic
    candidate `outputs.pt` files, for example a DGP-v2 Qwen2.5 PEFT predictor
    and a DGP/CALM embedding-MLP classifier
  - it trains only a small action-wise gate over probability/meta-features from
    base and candidate predictions
  - the utility target is accept/defer, not direct bot/human classification:
    accepting a candidate is positive only when the base is wrong and the
    candidate is correct; accepting a candidate that breaks a base-correct node
    is an explicit weighted negative through `--semantic_gate_break_weight`
  - `--semantic_gate_feature_family node_attribute` implements the current
    node-attribute gate variant: it keeps the probability competence features
    and adds target-account metadata/tweet cues plus local graph attributes.
    This directly tests whether candidate utility is conditional on node type,
    such as dense social-neighborhood accounts, URL-heavy accounts, retweet-heavy
    accounts, or sparse evidence accounts.
  - `--semantic_gate_feature_family local_competence` is the evidence-backed
    next diagnostic after generic node attributes failed to generalize. It
    follows dynamic classifier-selection / META-DES reasoning: candidate
    competence should be estimated from a local region of similar routed train
    nodes rather than only from global posterior features. For each
    node-candidate action it appends local candidate correctness, base-wrong
    density, fix rate, break rate, net utility, and neighbor-similarity support
    while using no routed validation/test labels for competence estimation.
  - `--semantic_gate_selection_policy defer_softmax` implements the narrower
    learning-to-defer variant: the model chooses among `keep_base` and the
    fixed semantic candidates instead of training independent candidate BCE
    gates and taking the max. `--semantic_gate_safety_policy break_first`
    adds a break-risk head, so candidate acceptance can be filtered by a
    validation-locked break threshold before routed-test replay.
  - the threshold is selected on routed validation nodes and then locked for
    routed test evaluation
  - this stage is the current implementation of the "do not replace high-base
    SimTeG directly; learn when to defer from base to semantic candidates"
    research boundary. It is aligned with selective prediction, learning to
    defer, and dynamic classifier-selection literature, but current single-seed
    results remain exploratory until they generalize beyond the fixed
    high-base routed split.
- `phase_a_single_cell_internal` is an internal execution route for
  module-controls comparisons and is not a public task
- `LLMbot/precompute.py` can now prepare prompt-cache semantic tensors in two
  families before Phase A plus versioned refiner-only expert bundles:
  - legacy GLANCE prompt modes (`glance_ego`, `glance_hop1`, `glance_hop2`,
    `glance_concat_ego_hop1_hop2`)
  - relation-aware social-context prompt modes (`relation_aware_ego`,
    `relation_aware_1hop`)
  - prompt-expert bundle v1 modes (`expert_ego`, `expert_graph_following`,
    `expert_graph_follower`, `expert_tweet`, `expert_conflict`,
    `expert_concat_v1`)
  - prompt-expert bundle v2, activated with
    `expert_* + --prompt_family_version v2`
  - prompt-expert bundle v3, activated with
    `expert_* + --prompt_family_version v3`
  - residual-audit prompt caches (`residual_audit_v1`) for fixed routed-node
    correction studies
- prompt-expert bundle v2 is the current literature-aligned explanation-first
  mainline:
  - `tweet`, `graph_following`, `graph_follower`, and `conflict` all use
    `explanation -> encoder embedding`
  - `conflict` is the only second-layer expert and consumes the other expert
    summaries rather than raw long-form evidence
  - `metadata_structured` is a non-prompt structured expert built from profile
    metadata logs, ratios, missingness indicators, and bucket one-hots; it is
    recorded with a feature-name schema and does not invoke the LLM
  - prompt text is now centralized in `LLMbot/prompt.py`, while
    `precompute.py` remains responsible for evidence assembly, fallback text,
    sidecars, encoder execution, and cache writing
  - prompt semantics now use a minimal analytic-template explanation contract:
    `Bot-like Evidence`,
    `Human-like Evidence`,
    `Uncertainty`,
    `View Leaning`,
    `Rationale`, and `Judgement`
    where the explain model first writes the view-grounded explanation
    template and then gives a final bot/human judgment from that explanation
  - v2/v3 can optionally switch their explanation-generation prompt framing
    with `--explain_prompt_style botsay`, which keeps the same evidence inputs
    but rewrites the task framing into BotSay-style
    `Label: bot or human` plus `Explanation: ...`
  - this BotSay-style path is intentionally narrow: it adapts prompt wording
    only, and does not import BotSay's labeled in-context exemplars or
    follower/following labels into the explanation prompt
  - sparse or missing evidence is treated as a limitation, not direct proof of
    automation
  - `graph_following` and `graph_follower` stay split because they represent
    different social roles (`who this account follows` vs `who follows it`)
  - the canonical explanation-first consumption path is now fixed as
    `router -> routed nodes -> LLM evidence -> routed-only node classifier`
  - in this path, routed nodes are chosen before any LLM call
  - `prompt_expert_bundle_v2/v3` therefore do not use
    `--selection_embedding_path`, support-selection embeddings, or
    `center_induced_relation_aware`
  - semantic-KNN prompt construction is reserved for
    `prompt_expert_bundle_center_induced_v1` and `mhlgc_llm_guide`, which are
    separate KNN-coupled diagnostics rather than the evidence-classifier
    mainline
  - raw metadata is compressed into prompt-friendly cues, while exact counts and
    ratios stay in the cache side-channel for the refiner
  - by default, v2 expert explanations are encoded with the same SimTeG
    finetuned RoBERTa family used by the frozen RoBERTa backbone line
    (`roberta_finetuned`, resolved offline to `yzxjb/roberta-finetuned-20`),
    and loads `TwiBot-20_seed_<seed>/checkpoints/LM_pretrain/best.pkl` when
    that frozen SimTeG LM checkpoint is available; the encoder then uses
    512-token truncation, no added special tokens, and final hidden-state plain
    mean pooling; using `roberta-base` is now an explicit ablation through
    `--model_path`, and missing finetuned-RoBERTa weights fail fast rather than
    falling back
- the older direct non-explanation expert path remains available for routed
  ablations and now has a clearer split:
  - `expert_ego` is a BotSay-aligned target-account classifier prompt over
    `tweet + metadata`
  - `expert_graph_following` / `expert_graph_follower` use a narrower
    GLANCE-style classifier shell with `EGO + HOP1 + Category? </END>`
  - this keeps direct-prompt routed-node experiments distinct from the
    explanation-first v2/v3 family
- prompt-expert bundle v3 is the structured evidence-card adaptation of the
  same explanation-first contract:
  - it keeps the v2 four-expert tensor layout and non-prompt
    `metadata_structured` side channel
  - generated text is constrained to source-grounded fields for bot-like cues,
    human-like cues, relation ambiguity, possible benign explanations, evidence
    coverage, evidence consistency, and source support
  - prompts forbid final labels, probabilities, confidence scores,
    recommendations, false-positive/false-negative claims, and base-model
    correction instructions
  - the cache records `semantic_view_mode=prompt_expert_bundle_v3`,
    `evidence_schema=structured_evidence_card`, and evidence-card field names
    in both payload/manifest metadata
  - this should be described as evidence augmentation for utility-gated
    correction, not as faithful GNN explanation, representation repair, or an
    LLM-as-predictor method without additional validation
- `residual_audit_v1` is the active prompt-construction surface for testing
  whether an embedding model can consume SimTeG residual/correction evidence:
  - it targets explicit routed nodes through `--routed_nodes_path`
  - it reads frozen SimTeG `outputs.pt` through
    `--residual_base_outputs_path`, but only uses prediction/probability state
    as prompt context
  - `base_as_hypothesis` treats the base prediction as a fallible hypothesis,
    `no_base` hides base-model state for ablation, and `base_as_assertion`
    deliberately measures anchoring risk
  - prompt rows store Qwen2.5-Instruct-compatible `system` and `user` messages
    plus a combined embedding prompt; the mainline asks for independent
    evidence assessment before comparing against the base state and constrains
    output to one strict JSON object
  - prompt text includes compact profile, tweet, and local graph context plus
    optional neighbor base-prediction distributions; it never includes dataset
    labels or oracle fix/break information
  - the produced cache is a prompt-embedding artifact with
    `semantic_view_mode=residual_audit_v1`; it does not change the router,
    refiner, split protocol, or graph backbone by itself
- `dgp_predictor_v1` is the active DGP-style prompt-construction surface for
  routed-node LLM-as-predictor validation:
  - it targets explicit routed nodes through `--routed_nodes_path`
  - it follows the DGP transfer boundary of detailed target evidence plus
    compressed neighbor context, adapted from fraud graph prompts to social-bot
    detection
  - `target_fine_neighbor_coarse` includes detailed profile/tweet evidence,
    representative tweets, following/follower counts, and ranked coarse
    neighbor cards; `target_only` removes neighbor cards as an ablation
  - prompts now use a minimal DGP-style classifier shell and request exactly
    one `Yes` or `No` answer token; they never include labels, oracle
    fix/break information, or frozen SimTeG correctness
  - the `*_prompts.jsonl` sidecar is consumed by routed-node Qwen PEFT
    classifier tuning; the `.pt` prompt embedding cache is consumed by the
    CALM-style query-embedding-MLP baseline
  - when consumed with `--semantic_supervision_mode classifier`, this remains
    the compatibility DGP-style hidden-state classifier route
  - faithful answer-token SFT is now available through the same prompt sidecar
    by switching `semantic_encoder_finetune` to
    `--semantic_supervision_mode answer_token`
- `dgp_predictor_v2` is the norm-text summary variant for the same research
  boundary:
  - target evidence is the node's LLM-friendly `norm_user_text`, treated as
    combined tweet+metadata evidence following BotSay-style textualization
    intuition
  - `norm_user_text` is parsed and rendered into `PROFILE`,
    `TWEET_BEHAVIOR`, and `TWEET_SAMPLES` sections before Qwen consumes it,
    avoiding the raw serialized delimiter string that previously triggered
    special-token and placeholder repetition
  - the mainline uses K=5 `following` neighbors; `follower` and
    `following+follower` are explicit ablation variants
  - selected neighbor `norm_user_text` rows are summarized first, then
    compressed into relation-level context summaries, and only those summaries
    are appended to the final Yes/No predictor prompt
  - empty selected-neighbor contexts are represented by an explicit
    deterministic sparse-context summary rather than an LLM-generated empty
    evidence paragraph
  - neighbor and relation summaries now use a minimal task-agnostic
    `Summarize ... within 10 tokens` prompt, matching the DGP paper's
    anti-task-aware ablation result more closely than the older requirement-
    heavy social-bot-specific wording; multilingual or noisy source text is
    still summarized as an evidence-quality cue to avoid source-span copying
    into downstream classifier prompts
  - single-row quality failures after retry are converted to explicit
    `quality_fallback` limited-evidence summaries, preserving full-run coverage
    without admitting bad generated text into the cache
  - this better matches DGP's fine target / coarse neighborhood principle
  - paired with `semantic_encoder_finetune --semantic_encoder qwen3_peft
    --semantic_supervision_mode answer_token`, this is now the active
    answer-token DGP route in the mainline
  - `--semantic_supervision_mode classifier` remains available as the old
    hidden-state classifier ablation on the same prompt text
- `botsay_knn_summary_predictor_v1` is the routed-node BotSay-style KNN
  evidence predictor route:
  - routed center nodes must already be fixed by `--routed_nodes_path`; this
    mode does not call an LLM before routing and does not let the LLM select KNN
    neighbors
  - account neighbors are selected by top-K cosine neighbors from
    `--selection_embedding_path` under
    `--neighbor_sampling_policy center_induced_relation_aware`
  - each retrieved account is first compressed by a label-free summary prompt:
    describe observable profile/content/activity/social-behavior signals, but
    do not decide bot versus human and do not mention similarity scores
  - the final prompt keeps the target account as fine-grained
    `norm_user_text`, appends retrieved-account summaries in highest-to-lowest
    semantic-similarity order, intentionally omits numeric KNN similarity
    values, and exposes only non-leaking observed social-connection metadata as
    natural language
  - internal relation tags remain in prompt sidecars for reproducibility, but
    are not the main text shown to the LLM
  - graph-global node ids remain sidecar metadata for auditing and are not
    exposed in the visible final prediction prompt
  - the final prompt ends at `ASSISTANT_ANSWER:`
  - this is a BotSay-style structural adaptation only; it does not import
    BotSay's neighbor true-label fields, labeled demonstrations, or any oracle
    fix/break information
  - downstream consumption should use the same routed-node answer-token
    contract as DGP v2:
    `semantic_encoder_finetune --semantic_encoder qwen3_peft
    --semantic_supervision_mode answer_token`
- for prompt-expert bundles, `precompute.py` now also supports a
  center-induced relation-aware neighbor evidence policy:
  - `--neighbor_sampling_policy center_induced_relation_aware`
  - `--context_graph_variant {labeled,full_graph_support}`
  - `--center_node_scope {labeled,all_graph_nodes}`
  - the shared HyperScan-style KNN grouping and partition logic now lives in
    `LLMbot/hypergnn.py`; `precompute.py` consumes that module but remains
    responsible only for prompt assembly, explanation generation, and cache
    writing
  - labeled and support selection embeddings can be supplied separately so
    center nodes remain labeled while neighbors may come from the full graph
  - each direction (`following`, `follower`) is partitioned into support and
    contrast subsets using cosine similarity as the primary signal and
    reciprocity/common-neighbor/degree/text-length tie-breakers
- `precompute.py` now also supports a raw tweet source compatibility path for
  expert prompts:
  - `tweet_source_mode=norm_user_text` preserves the current serialized-text
  route
  - `tweet_source_mode=raw_post_edges` rebuilds user-centered tweet evidence
  from `node_new.json` plus grouped `post` edges from `edge_new.json` when
  available, otherwise `edge.csv(post)`
- prompt-expert precompute can now also target a fixed routed-node subset with
  `--routed_nodes_path`:
  - this is restricted to `expert_*` prompt modes
  - only those graph-global node ids go through prompt building, explanation,
    and encoder inference
  - the saved payload is still expanded back into full-graph row layout with
    zero fill on unselected nodes so refiner-side cache injection does not need
    a second alignment layer
  - the payload now also carries explicit `target_node_ids` and
    `target_node_mask` membership metadata, and the strict joint refiner uses
    that mask to keep routing/application on the original routed-node set
  - when the same full-graph payload is injected into a labeled-only frozen
    SimTeG backbone through `--external_frozen_g0_root`, the joint stage slices
    prompt-expert components and membership masks to the labeled prefix before
    refiner training
  - this is the intended efficient path for fixed-router / fixed-routed-set
    prompt-expert comparisons
- explanation-first v2/v3 precompute now keeps the explain model loaded once per
  run instead of reloading it once per expert component
  - v2/v3 explanation generation now covers the selected explanation-first
    expert components in the same run (`graph_following`, `graph_follower`,
    `tweet`, and `conflict` for `expert_concat_v1`)
  - real LLM-as-explainer studies should pass `--explain_required` so missing
    or invalid local instruct-model weights fail before any deterministic
    fallback cache is written
  - `LLMbot/models/` is the project Python model package, not an explain-model
    snapshot; local/server runs must point `--explain_model_path` at the actual
    HuggingFace causal-LM snapshot or a parent directory with exactly one such
    snapshot child
  - uncovered users fall back to the serialized `norm_user_text(.json/.new.json)`
    tweet segment and record that fallback in the manifest / sidecar
  - optional wandb monitoring is available through `--project_name`; it records
    prompt-bundle construction, per-batch explanation progress, per-component
    explanation completion, per-component encoding, and final artifact
    dimensions without changing the downstream cache contract
  - component-level explanation sidecars are append-only and resumable by
    `node_id + prompt_hash`; reruns skip completed rows
  - concat prompt-expert payloads additionally emit component-only `.pt` caches,
    so a failed or ablated expert can be rerun without treating the concat cache
    as the only reusable artifact
  - `--explain_batch_size 0` now selects a conservative automatic batch size
    (`2` on CUDA, `1` on CPU) so server runs avoid the old batch-1 default unless
    the operator pins it explicitly
- these prompt caches are helper artifacts only; strict GLANCE still requires
  a same-root `graph_detector_prepare` run to register the cache provenance
  before `joint_router_refinement`
- `prompt_expert_quality_audit` is the diagnostic answer to whether the current
  explanations and `explain -> LM embedding` tensors provide separable evidence
  for routed-node correction:
  - it reuses an existing prompt-expert cache and adjacent explanation sidecars
  - it reuses a reference `joint_router_refinement` stage for routed masks,
    labels, base predictions, and optional correction MoE utility tensors
  - it checks sidecar coverage/text quality, same-node cross-expert similarity
    and CKA, expert identity separability, base-wrong separability, and
    per-action utility separability when available
  - it does not regenerate explanations, retrain the router, retrain the
    refiner, or claim a deployable performance gain
  - a passing text-quality gate only means the sidecars are usable for
    downstream diagnosis; it does not imply that routed wrong/correct nodes are
    separable in the current embedding space
- for refiner-only prompt ablations, `joint_router_refinement` can now keep the
  backbone fixed to the same-root `graph_detector_prepare` artifact while
  loading a separate semantic override through
  `--joint_refiner_embedding_path`; prompt payloads with `ego/hop1/hop2` are
  consumed as direct refiner views instead of being recycled through the base
  detector, and prompt-expert payloads are consumed under
  `semantic_view_mode=prompt_expert_bundle_v1` or
  `prompt_expert_bundle_center_induced_v1`, while v2/v3 prompt-expert payloads
  are consumed under `semantic_view_mode=prompt_expert_bundle_v2` or
  `prompt_expert_bundle_v3`
- the same ablation path may also reuse an existing high-base frozen backbone
  through `--external_frozen_g0_root`, but only when the prompt cache is still
  attached as a refiner-only override rather than as a backbone feature
- within that prompt-expert path, the strict refiner now projects
  `ego/graph_following/graph_follower/tweet/conflict/metadata_structured`
  separately and uses a
  lightweight graph-view gate driven by following/follower count-presence
  features instead of collapsing everything back into the backbone semantic
  tensor; the projected expert stack is followed by the full scalar
  side-channel, while center-induced bundles extend that side-channel with
  candidate-count, selected-count, support/contrast similarity, and
  reciprocal-ratio evidence
- prompt-expert fusion is now a controlled refiner-only variable:
  `--joint_prompt_expert_fusion projector_concat` preserves the fixed projected
  concat path, while `--joint_prompt_expert_fusion mpe_gated` keeps the earlier
  node-conditioned softmax over `graph_following`, `graph_follower`, `tweet`,
  `conflict`, and `metadata_structured`. `--joint_prompt_expert_fusion
  gaugllm_selector` is the stricter GAugLLM-style selector transplant: it does
  not rebuild caches, reuses the routed prompt cache plus adjacent explanation
  sidecars, builds one selector-context text per routed node and expert at
  runtime, encodes those texts with the finetuned SimTeG RoBERTa line recorded
  in the cache manifest, and then combines expert-attention and context-aware
  attention before the routed-node classifier. `--joint_prompt_expert_fusion
  gaugllm_mope` uses the same runtime-reuse cache/sidecar contract but aligns
  the fusion head with GAugLLM's official default `SimilarityAttentionMLP`:
  four projected content experts produce MLP logits, four selector-context
  projections provide per-expert dot-product similarity, the two logits are
  added, divided by default temperature `0.2`, and softmaxed before weighted
  content pooling. The default `--joint_prompt_expert_mope_logit_norm none`
  preserves this official-style formula; `branch_zscore` and `combined_zscore`
  are diagnostic calibration settings for selector-collapse studies, not
  official-reproduction claims. MoPE runs persist content, similarity,
  combined, and selector logit summaries so routed-node expert collapse can be
  attributed to a specific branch before adding stronger regularization.
  The current mainline also exposes two lightweight selector-calibration terms
  for this diagnostic lane only: `--joint_prompt_expert_mope_entropy_weight`
  encourages higher per-node selector entropy, and
  `--joint_prompt_expert_mope_load_balance_weight` encourages batch-mean expert
  usage to track the batch availability prior rather than collapsing to a
  single globally dominant expert.
  `--joint_prompt_expert_fusion botmoe_selector` is the implemented
  GLANCE utility-advantage plus BotMoE-selector experiment line. It keeps the
  BotMoE-style sparse selector scoped to `graph_following`, `graph_follower`,
  `tweet`, and `conflict` only. `metadata_structured` stays as a side feature
  and selector-context support source rather than a selectable expert. The
  selector also does not contain `abstain` or `base`: the explicit utility gate
  owns keep/base versus change/correct, and the selector only chooses which
  prompt expert to use if a change is allowed. Its public control surface is
  `--joint_prompt_expert_botmoe_top_k`,
  `--joint_prompt_expert_botmoe_noisy_gating` /
  `--no-joint_prompt_expert_botmoe_noisy_gating`, and
  `--joint_prompt_expert_botmoe_aux_weight`, where the auxiliary term scales
  `cv_squared(importance) + cv_squared(load)` on routed-node training batches.
  This remains a refiner-only fusion change: it is not graph augmentation and
  does not alter the router feature family.
- `--joint_prompt_expert_fusion utility_correction_moe` is the fixed-router
  correction MoE validation lane. It starts from the best routed-node anchor
  family rather than the full four-prompt selector: `graph_follower`, `tweet`,
  `conflict`, and `metadata_structured` each get an expert-specific correction
  head, and a per-node/per-expert utility head learns whether that expert has
  positive utility over the frozen base. The default target,
  `--joint_correction_moe_utility_target loss_advantage`, keeps
  `base_loss - expert_loss - beta > 0`; `decision_gain` changes the supervision
  to the true discrete correction event `base_wrong && expert_pred_correct`;
  `hybrid` accepts either signal; `net_gain` keeps fix actions positive while
  treating base-correct/expert-wrong break actions as negative utility through
  `--joint_correction_moe_break_weight`. The maximum utility probability is used
  as the abstain/keep-change score under `--joint_refiner_gate_policy
  hard_keep_change` when stage calibration is disabled. The same lane can add
  within-node pairwise action ranking with `--joint_correction_moe_ranking_weight`
  and validation-locked keep/change thresholding with
  `--joint_correction_moe_gate_calibration {global_threshold,per_action_threshold}`.
  This lane is intentionally fixed-router and routed-set-only;
  it is not evidence for joint router-expert optimization, which would require
  broader explanation coverage as routed nodes change. The stage outputs persist
  raw per-expert logits, loss, advantage, prediction, target/reward matrices,
  calibration metadata, and calibrated decisions so target/beta/threshold
  diagnostics can be audited from first-class stage artifacts.
  When paired with v3 caches, the lane tests whether structured evidence-card
  embeddings help utility-gated correction; it still does not implement
  representation-level repair.
- `--joint_prompt_expert_fusion metades_selector` is the META-DES-style
  competence-selector migration for the same fixed-router correction setting.
  It keeps the routed set, backbone, router features, prompt cache, and
  correction utility targets unchanged, but changes the selectable actions to
  `graph_follower`, `tweet`, `conflict`, and a runtime-derived
  `follower_triplet`. The `follower_triplet` action is built from existing
  `graph_follower + tweet + conflict` projections inside the refiner, so it does
  not require a new cache. The utility head receives base/expert competence
  meta-features such as confidence, margin, entropy, bot probability,
  disagreement, and probability gap. This is a META-DES-style routed correction
  selector rather than an official full META-DES reproduction, and
  `metadata_structured` stays as a side feature rather than a selectable action.
  The stage writes generic `selector_weights`, `selector_weight_follower_triplet`,
  selected action names, and the shared correction-target diagnostics to
  `per_node_test.jsonl` and `outputs.pt`.
- `--joint_prompt_expert_fusion conflict_aware_correction_moe` is the
  BotMoE-inspired correction adaptation used after the sparse BotMoE selector
  showed high fix but higher break from selecting `conflict` as a global
  dominant expert. It keeps the routed set and cache fixed, selects only
  first-order prompt experts (`graph_following`, `graph_follower`, `tweet`),
  and uses `conflict` plus runtime explanation-context embeddings as
  cross-view safety/context inputs to the expert-specific correction and utility
  heads. It shares the same correction utility targets, break-aware ranking, and
  validation-locked thresholding as `utility_correction_moe`, so the research
  question is whether `conflict` is better used as selector evidence than as a
  hard top-1 action.
- raw-concat routed-node ablations now also exist under the same flag:
  - `raw_concat_ego_following` uses
    `[z_gnn || ego || graph_following || structural_side_channel]`
  - `raw_concat_ego_follower` uses
    `[z_gnn || ego || graph_follower || structural_side_channel]`
  - `raw_concat_ego_following_follower` uses
    `[z_gnn || ego || graph_following || graph_follower ||
    structural_side_channel]`
  - `raw_concat_single_graph_following`, `raw_concat_single_graph_follower`,
    `raw_concat_single_tweet`, and `raw_concat_single_conflict` use
    `[z_gnn || one_raw_expert || structural_side_channel]` to isolate each
    expert's fix/break pattern without projector compression or `graph_fused`
  - `raw_concat_follower_tweet` uses
    `[z_gnn || graph_follower || tweet || structural_side_channel]`
  - it is the narrowest GLANCE-style prompt-expert refiner surface for the
    current routed cache and tests whether the positive `graph_follower`
    evidence still needs `conflict` as a calibration/stability channel
  - `raw_concat_following_triplet` uses
    `[z_gnn || graph_following || tweet || conflict || structural_side_channel]`
  - `raw_concat_follower_triplet` uses
    `[z_gnn || graph_follower || tweet || conflict || structural_side_channel]`
  - both remove the extra `graph_fused` slot so following-vs-follower evidence
    can be compared without projector compression or duplicate graph summaries
  - `ultratag_propagated_follower_triplet` is a legacy mean-propagation
    routed-node ablation. It reuses the current routed prompt-expert cache,
    mean-propagates the cached expert embeddings over one graph hop at runtime,
    and then uses `[z_gnn || ultratag_graph_follower || ultratag_tweet ||
    ultratag_conflict || structural_side_channel]` in the same no-projector
    refiner. It is not an UltraTAG-S reproduction and should not be cited as
    the paper-aligned migration.
  - `precompute.py --prompt_mode ultratag_s_subgraph_v1` is the bounded
    UltraTAG-S adaptation path. It applies target-plus-neighbor text
    propagation, LLM summary/keywords/soft-label text augmentation, optional
    same-soft-label cosine virtual edges, PageRank selection, optional LLM edge
    reconfiguration, and finetuned-RoBERTa row replacement for the selected
    target scope. Routed-node runs remain restricted to the test split; full
    graph target runs use `--center_node_scope all_graph_nodes`. The current
    routed diagnostic keeps the target/output graph on labeled nodes and uses
    `--context_graph_variant full_graph_support` only to draw neighbor evidence
    from the full support graph. Text-only runs should use
    `--ultratag_virtual_edge_policy none` and `--ultratag_edge_reconfig false`
    to avoid using noisy soft labels as graph supervision. This still does not
    claim full UltraTAG-S dual-GNN structure-learning reproduction.
  - `precompute.py --prompt_mode llm_knn_edge_retain_v1` is a narrower
    UltraTAG-S-inspired edge-retention cache for the current HyperScan-style
    second-view branch. It retrieves non-routed KNN candidates in
    `--selection_embedding_path` space and asks the local Qwen explain model
    for KEEP/DROP decisions about whether each candidate is reliable
    high-order propagation support. Downstream
    `--graph_second_view_candidate_policy llm_retain` consumes only the cache
    as an `x_high` member filter; the LLM output is not a bot/human label and
    is not used by the conformal router.
  - `scripts/router_knn_evidence_reconfiguration.py` is the separate
    post-hoc routed-node GraphEdit-style baseline. It does not change the
    UltraTAG-S text-augmentation path. It keeps the conformal-router
    HyperScan-style KNN candidate space fixed, builds GraphEdit-like pair
    prompts over two stages (`drop` for existing routed KNN support and `add`
    for second-ring candidates), generates a Qwen JSONL edge-decision sidecar,
    and then lets the existing routed-node refiner consume `keep/drop/add`
    decisions. Qwen3.5-9B requires the Transformers-main isolated path plus
    `enable_thinking=False`; otherwise it emits thinking traces before the
    final True/False answer. `--llm_prompt_variant graphedit_same_behavior`
    preserves the direct GraphEdit-style same-behavior baseline, while
    `--llm_prompt_variant graphedit_edge_usefulness` keeps the same two-stage
    candidate-editing surface but asks only whether the neighbor is useful
    behavioral evidence for the target node. The baseline should be reported
    as GraphEdit-style candidate refinement over fixed router candidates, not
    as a full GraphEdit reproduction, LLM-selected free-form neighbors, or an
    LLM final bot/human predictor.
  - `raw_concat_metadata_anchor` adds `metadata_structured` to the
    follower-triplet anchor
  - `raw_concat_metadata_only` isolates
    `[z_gnn || metadata_structured || structural_side_channel]`
- explanation generation remains a semantic-cache construction step. The public
  router feature family is unchanged unless a future experiment explicitly adds
  explainer-derived router inputs.
- `expert_ego` and the `ego` branch inside `expert_concat_v1` now resolve their
  explain model in offline-first mode: local snapshot path first, then cached HF
  snapshot, and finally a deterministic profile-card explanation fallback when
  no local explain-model weights are available
- `LLMbot/preprocess.py` is now the standalone support-extension raw-data
  helper for TwiBot-20:
  - it reuses the current labeled `norm_user_text.json` prefix
  - it appends support texts from `datasets/TwiBot-20/support.json` into
    `norm_user_text_new.json`
  - it rebuilds `edge_new.json` from the official TwiBot-20 `edge.csv`
    relation file
  - it normalizes user-user edge direction as:
    `friend(source,target) -> source -> target`,
    `follow(source,target) -> target -> source`
  - it writes `edge_index_new.pt` and `edge_type_new.pt` from the normalized
    full user-user graph rather than from the older labeled-only subgraph
  - it writes `support_idx.pt` as the appended support-node index range
  - it does not change `labels.pt` or the existing
    `train_idx.pt/valid_idx.pt/test_idx.pt` supervision contract
- first-round full-graph execution now exists for the active mainline:
  - `graph_data_variant=full_graph_support` is currently supported only for
    `graph_detector_prepare`, conformal-style `estimator_ablation`, and
    `joint_router_refinement`, plus the diagnostic-only
    `local_conflict_prune_diag`
  - `graph_detector_prepare` now also supports
    `--graph_refine_mode hyperscan_knn_hypergraph_proxy_augment`, which builds
    HyperScan-style feature-KNN groups from the current Phase-A input tensor
    and writes a proxy augmented graph by expanding each local group into
    bidirectional center-neighbor edges under a new relation id
  - `graph_detector_prepare` also now supports
    `--graph_refine_mode relation_overlap_knn_proxy_augment`, which is the
    current evidence-driven correction to that first proxy:
    - centers come from `--routed_nodes_path` when provided, otherwise the
      labeled prefix
    - `--routed_nodes_split` chooses which routed subset supplies those centers
    - semantic neighbors are searched only inside each center's undirected
      relation 1-hop ego neighborhood
    - the resulting added relation is an overlap-style proxy for
      relation-supported semantic grouping, not a graph-wide semantic rewrite
  - `graph_detector_prepare` now also supports
    `--graph_refine_mode relation_overlap_knn_repr_prefit_augment`:
    - it keeps the same routed-node local overlap construction
    - but it first fits the relation-view branch on the original graph and
      then builds local KNN overlap groups from `node_repr + g0_input`
    - this is the closer active-mainline proxy to HyperScan's
      `x_low + x_in` similarity-hypergraph path
  - `graph_detector_prepare` now also supports
    `--graph_second_view_scope routed_nodes` together with
    `--graph_backbone rgcn_hyperscan`:
    - this is the first active-mainline path that changes the failure mode we
      identified in experiments, because it does not rewrite the graph into
      static proxy edges before training
    - the relation branch still runs on the original graph and produces
      `x_low`
    - routed-node local KNN groups are rebuilt from `x_low + x_in` inside each
      forward and consumed by a separate HypergraphConv branch
    - the model-side second-view branch now rejects `feature_source` metadata
      that names final `node_repr`; `node_repr` remains only for the explicit
      prefit/static proxy and final-hidden control ablations
    - frozen `outputs.pt` now preserves those true intermediate tensors as
      `x_low` and `x_new`, so downstream router ablations can consume the exact
      HyperScan KNN space instead of reconstructing it from `node_repr`
    - `--graph_second_view_candidate_scope labeled_relation_1hop` keeps that
      second view inside the labeled prefix even when the relation backbone
      still runs on `full_graph_support`
    - `--graph_second_view_candidate_policy` adds routed-support ablations for
      the diagnosed hard-hard high-order neighborhood problem: `default` keeps
      ordinary `x_new` top-k, `exclude_routed` removes routed candidates,
      `stable_quota` reserves low-risk non-routed support, and `mixed_quota`
      reserves hard/stable/counterfactual support buckets before filling by
      current `x_new` similarity. These policies are diagnostic and label-free:
      stable support is based on a supplied full-graph risk vector, and
      `router_support_transfer` is the neighborhood-transfer diagnostic that
      reuses the conformal router's exported support row directly as the
      second-view candidate neighborhood for the same center. It does not alter
      router selection or detector fusion; it only tests whether the adaptive
      router neighborhood can serve as the classifier neighborhood.
      counterfactual support is based on base-prediction disagreement rather
      than ground-truth labels.
    - `--graph_second_view_hypergraph_backend {pyg,dhg}` and
      `--graph_second_view_fusion {residual,multiattn,multiattn_adaptive,construct_acm}` now split HGNN backend
      and fusion realization into separate parser axes
    - `--graph_second_view_use_bn` is the extra DHG-only alignment knob for
      matching the released HyperScan TwiBot20 HGNN batch-normalization setting
    - `--graph_second_view_scope labeled_prefix` is the corresponding
      all-labeled-center dynamic branch when no routed-node file should be
      consumed
  - `graph_detector_prepare` now also supports
    `--graph_second_view_scope neighborloader_batch`:
    - this is the closest active-mainline proxy to the released HyperScan
      TwiBot20 code path rather than to our earlier routed-node ablations
    - it requires `graph_data_variant=labeled`
    - it switches Phase-A graph training to `NeighborLoader` sampled subgraph
      batches
    - it rebuilds the similarity hypergraph from `x_low + x_in` inside each
      sampled subgraph batch, rather than from a global center list over the
      full graph
    - `--graph_neighborloader_contract hyperscan_sampled_subgraph` now makes the
      released-code supervision/evaluation contract explicit: train/valid/test
      all consume the full sampled subgraph rows, repeated nodes across batches
      are counted repeatedly, and best-checkpoint selection moves to validation
      accuracy
    - `--graph_neighborloader_contract seed_only` preserves the older
      active-mainline seed-row contract for same-code ablations
    - this narrows the migration gap on the training regime side without yet
      claiming a full paper-faithful reproduction
    - the same faithful line now also supports a routed-selective detector
      consumption contract:
      `--graph_second_view_consumer_scope routed_only` keeps the sampled-subgraph
      `x_new -> KNN -> x_high` construction shared, but only routed/high-risk
      rows consume the HNN detector path; non-routed rows use an explicit
      `low_only` fallback
    - routed-selective consumption is a detector-side contract, not a router or
      member-policy change; in v1 it is parser-validated for
      `graph_backbone=rgcn_hyperscan_dhg_nodeinput`,
      `graph_second_view_scope=neighborloader_batch`,
      `graph_neighborloader_contract=hyperscan_sampled_subgraph`, and
      `graph_second_view_fusion in {multiattn,residual}`
    - the residual detector line now also supports a continuous all-node
      consumption contract:
      `--graph_second_view_consumer_scope risk_gated_all_nodes` keeps the same
      sampled-subgraph `x_new -> KNN -> x_high` construction, but scales the
      residual high-order contribution with a frozen external `x_new` conformal
      risk through `alpha=sigmoid(a*risk+b)`
    - `--graph_second_view_risk_path` supplies that frozen full-graph risk
      payload; v1 keeps the mapping scalar, node-wise, and detector-only
    - this should be described as a detector-consumption unification, not as a
      new router, not as a new hyperedge-member policy, and not as evidence
      that “RGCN is already proven to be the high-order special case”
  - `graph_node_input_family=hyperscan_meta_tweet_proxy` and
    `graph_backbone=rgcn_hyperscan` let us test a second migration axis beyond
    graph geometry:
    - the tweet channel remains the current semantic embedding
    - numeric and categorical metadata proxies are rebuilt from the labeled
      `norm_user_text` serialization
    - the resulting backbone is a paper-inspired node-input preprocessing proxy,
      not a byte-for-byte reproduction of the original hidden sizes or legacy
      precomputed `.pt` artifacts
  - `graph_training_loader_mode=neighbor_subgraph` can now be applied without a
    dynamic branch, which gives us a direct `base labeled + NeighborLoader`
    control instead of only a dynamic-vs-base mismatch
  - `graph_training_max_steps` now makes sampled-subgraph and full-batch graph
    runs comparable on optimizer-step budget rather than only on epoch count
    - `graph_second_view_hypergraph_backend` and `graph_second_view_fusion` are
      now the detector-realization controls for HyperScan-style branches:
      - `pyg` and `dhg` consume the same dynamic incidence specification
    - `graph_second_view_use_bn` lets DHG-backed runs match the released
      HyperScan HGNN batch-normalization setting without changing the rest of
      the training contract
      - `residual`, `multiattn`, and `multiattn_adaptive` preserve the relation
        branch, KNN scope, data split, loader, and feature tensor while changing
        only the fusion head
      - within this family, `multiattn` is the HyperScan-faithful two-channel
        detector over `x_low` and `x_high`
    - `hyperscan_detector_style=original_cross_attention` remains a legacy
      alias for `graph_second_view_fusion=multiattn`
    - `multiattn_adaptive` is a bounded FAGCN-style migration: it keeps
      HyperScan's relation branch, dynamic hypergraph branch, and bidirectional
      cross-attention tokens unchanged, then learns a node-wise low/high
      mixture before the final classifier. It should be described as detector
      fusion adaptation, not as a full spectral FAGCN reproduction.
    - `construct_acm` belongs to the separate `construct_complete_v2`
      dual-space line and should be described as a three-channel
      identity/low/high extension rather than as the faithful HyperScan
      baseline
  - `mhlgc_llm_guide` plus `graph_detector_prepare --mhlgc_enable` is the
    bounded legacy MH-LGC-style migration branch:
    - prompt construction lives in `LLMbot/prompt.py` and is called by
      `LLMbot/precompute.py --prompt_mode mhlgc_llm_guide --routed_nodes_path ...`
    - the prompt follows the visible structure in the paper figure:
      `Instruction`, `[Role Description]`, `[Task Definition]`, and
      `[Input Graph]`
    - each routed-node prompt serializes the original directed relation view
      and a HyperScan-style KNN hypergraph view built from
      `--selection_embedding_path`
    - for a closer paper-style semantic view, `precompute.py` can load a local
      LLaMA/Qwen/Mistral model with `--embedding_model_class causal_lm` and
      extract `outputs.hidden_states[-1]` at the last valid token through
      `--embedding_pooling_mode causal_last_hidden_last_token`; free-form
      explain text remains only a diagnostic surface
    - graph contrastive learning lives in `LLMbot/GNNs.py`; the training loop
      in `LLMbot/trainer_preparation.py` adds it to supervised CE only when
      `--mhlgc_enable` and a graph-node-aligned semantic embedding tensor are
      supplied
    - this legacy branch makes the contrast space explicit:
      `semantic` is the LM input embedding space for mechanism diagnosis,
      `x_new` is the HyperScan-style construction space, `x_high` is the
      HGNN high-order branch before detector fusion, `fused_x` is the preferred
      explicit final detector hidden space, `low_high_concat` is
      `cat(x_low,x_high)` before detector fusion, and `node_repr` remains the
      legacy alias for that same final hidden
    - dual-space H2GCN/FAGCN backbones extend this role split further:
      `z_construct = hyperscan_clean_representation`,
      `z_pred = fused_x`. In the rewritten construct-complete contract,
      Hyperscan clean features define the whole graph branch
      `x_in_construct -> x_low_construct -> x_new_construct -> x_high_construct`
      and KNN / hypergraph structure is built directly from `x_new_construct`
      - the `*_nodeinput` variant keeps the same graph-branch contract, but
        obtains `x_in_construct` from tweet/num/cat preprocessing
      - iter_-1 remains only a semantic control line via separate non-dualspace
        baselines
    - the current v1 implementation keeps this family isolated from
      `--mhlgc_enable`, `--routed_contrast_family`, and
      `--routed_highpass_mode` so the comparison stays on detector consumption
      rather than routed-node repair
    - `--mhlgc_anchor_source routed_target_mask` is the cleaner routed-node
      adaptation: anchors come from routed prompt-cache targets rather than
      only from the historical nonzero semantic-cache heuristic
    - under that routed-target contract, `precompute.py` also persists
      explicit `following_view`, `follower_view`, `mutual_view`, and
      `semantic_knn_view` rows per routed node. `trainer_preparation.py`
      keeps supervised CE on the full graph, but passes those rows into a
      routed-only detector-space refiner in `GNNs.py`
    - the routed multiview refiner is bounded: it does not rewrite the full
      graph or globally replace the detector hidden state. It refines only
      routed nodes with non-empty auxiliary relation/semantic context and
      leaves non-routed rows untouched
    - `--mhlgc_pair_mode repair_aware` is the bounded repair-aware extension
      inside this legacy branch:
      for HyperScan-style backbones the LLM guide now perturbs the routed
      second-view construction path before dynamic hypergraph construction,
      then contrasts the original and repaired views in the chosen contrast
      space while the classifier path itself stays unchanged
    - `--routed_contrast_family` is the separate routed-node contrast ablation
      branch for `graph_detector_prepare`:
      - it keeps the HyperScan-style backbone unchanged and adds only an
        auxiliary loss
      - it is full-batch only and applies only to routed **train** nodes from
        `--routed_nodes_path`
      - `low_high` applies same-node `x_low <-> x_high` InfoNCE on eligible
        routed nodes
      - `three_view_control` keeps the same low/high same-node objective and
        adds a stop-grad frozen semantic teacher aligned only to `x_low`
      - `supcon_class` is a BotSCL-style same-class positive /
        different-class negative control over `fused_x`
      - `hybrid` combines the low/high same-node path with the class-aware
        supervised contrast path
      - `bot_edge_mask_human` is the asymmetric routed contrast variant:
        routed bot nodes act as anchors, edge-masked same-node `fused_x` acts
        as the positive, and routed human `fused_x` acts as the negative pool
      - `--routed_contrast_gate heuristic_reliable` is a training-only support
        reliability filter computed from routed multiview support rows already
      serialized by `precompute.py --prompt_mode mhlgc_llm_guide`; it is not
      an extra predictor or router
    - this branch is an ablation surface for full-test Accuracy/Macro-F1 and
      routed-node mechanism diagnosis, not a promoted default mainline
    - `--routed_highpass_mode` is the routed-only high-pass correction lane:
      it keeps the base graph detector and HyperScan-style fusion intact,
      then applies correction only to nodes selected by `--routed_nodes_path`
    - `--routed_highpass_target logits` preserves the legacy detector-space
      delta-logit correction; `--routed_highpass_target x_high` moves the
      correction before cross-attention and constrains the HGNN high-order view
    - implementation follows a staged training contract: first train/select the
      clean base detector with supervised CE, then freeze it and train only the
      routed high-pass correction module
    - the high-pass lane separates `h_self`, relation-neighbor low-pass
      aggregation, and residual high-pass aggregation in the selected target
      space. This is the H2GCN/FAGCN/BotSCL-inspired heterophily handling path,
      not an LLM-GCL variant
    - `relation_1hop` is the clean candidate scope; `relation_1hop_plus_xnew_knn`
      is an ablation that adds forward-native `x_new` KNN candidates before
      top-k truncation. `x_new` remains a construction/candidate-ranking space
      rather than the final detector representation
    - `--routed_highpass_risk_path` can supply conformal risk as utility/gate
      supervision; risk is not treated as a human/bot posterior
    - high-pass correction is full-batch only and mutually exclusive with
      `--mhlgc_enable` and `--routed_contrast_family` to keep method evidence
      isolated
    - because routed semantic-guide caches may come from embedding models whose
      width differs from detector `hidden_dim`, the active mainline now treats
      semantic-guide consumption as a projection problem: the cache width is
      recorded at training time and `GNNs.py` learns a small projector so the
      semantic token enters the MultiAttn/Transformer block in detector space;
      when the HyperScan detector uses the `multiattn` head, the auxiliary
      relation/hypergraph view tokens are projected into that same detector
      space before routed multiview refinement
    - `--mhlgc_hyperedge_mask_probability` now lets the augmented branch mask
      node-hyperedge incidence in the active second-view hypergraph as well as
      node features and relation edges; this remains a paper-inspired
      hyperedge-membership approximation rather than a full temporal/frequency
      hyper-view reproduction
    - `--mhlgc_anchors_per_batch 1 --mhlgc_negative_count 3` is the closer
      paper-style batch contract: one borderline positive-label anchor per
      batch, then three hardest negative-label nodes selected by the combined
      GNN/LLM semantic hardness score. The default
      `--mhlgc_negative_count 0` preserves the older all-negative exploratory
      behavior for ablation continuity.
    - the transferable claim here is only LLM-as-guide hard-negative weighting
      over original/augmented graph views; the LLM is not a predictor, labels
      are not inserted into prompts, and inference remains GNN-only
    - this branch should be reported as a legacy exploratory/diagnostic lane,
      separate from the current routed-only evidence/classifier mainline
    - this should not be described as a full Ou et al. reproduction because
      the social-bot version replaces the fraud paper's domain-specific
      transaction/hypergraph views with original social relations plus the
      current HyperScan-style KNN hypergraph view
  - the active research boundary is therefore:
    - global KNN star expansion is treated as a coarse HyperScan-inspired proxy
    - `relation_overlap_knn_proxy_augment` is the bounded local follow-up for
      routed-node enhancement
    - `relation_overlap_knn_repr_prefit_augment` is the bounded
      relation-view-aware follow-up that moves the KNN feature space closer to
      HyperScan's original two-view construction
    - `routed_dynamic_hyperscan_branch` is the current strongest active
      HyperScan-style proxy because it does add a separate training-time
      hypergraph branch over explicit routed-node hyperedges
    - none of the three paths should be described as a full HyperScan
      reproduction unless the detector/fusion-head, node-input, and hypergraph
      backend variants being used are stated explicitly
  - full-graph mode keeps the original labeled supervision split but switches
    graph propagation to `edge_index_new.pt / edge_type_new.pt`
  - the Phase-A semantic tensor is built at runtime by concatenating the
    labeled RoBERTa embedding from `--embedding_path` with
    `--support_embedding_path`
  - the resulting frozen GNN outputs and router feature tensors are graph-wide
    (`229580` rows), while losses and evaluation still use the labeled
    `train/valid/test` indices only
  - for `local_conflict_prune_diag`, full-graph mode now compares the labeled
    prefix against the support-augmented graph and writes:
    - error migration
    - support exposure
    - support consistency
    - structural shock
    - regime-shift summaries
    - four failure mechanisms:
      `sparse_isolated`, `dense_directional_recoverable`,
      `conflict_dominant`, `support_induced_fragile`
  - for `estimator_ablation`, the current full-graph support is intentionally
    narrow and routed through labeled-prefix evaluation:
    - `posthoc_calibrated_ranker`
    - `calibrated_local_risk_router`
    - `conformal_knn_risk_router`
    - `graph_conformal_set_estimator`
    - `gnn_2hop_conformal`

### Phase B: post-hoc estimator / hard-node routing

Current code anchors:

- `LLMbot/code/estimators.py`
- `LLMbot/code/model_building.py`
- `LLMbot/code/stage_runner.py`
- `LLMbot/code/trainer_legacy_impl.py`

Public tasks:

- `estimator_ablation`
- `local_conformal_diagnostic`
- `joint_router_refinement`

Important implementation notes:

- `login_uncertainty_router` exists as an estimator mode inside
  `estimator_ablation`
- current LOGIN-related implementation is explicitly bounded to
  `node_selection_uncertainty_only`
- full-graph conformal ablations now reuse the graph-wide frozen SimTeG
  backbone outputs but evaluate residual-error routing only on the labeled
  prefix; subgroup and stage2 reporting fall back to labeled-prefix structural
  analysis rather than trying to attach oracle labels to the support suffix
- `calibrated_local_risk_router` is the new conformal-family ablation that
  keeps posterior calibration scalar-only and lets local graph structure adjust
  the risk object rather than smoothing the posterior itself
- the current implementation is the stronger v2 form of that ablation:
  - local aggregation is strictly 1-hop
  - relation and direction are kept as separate channels
  - `node_repr` can localize neighborhood risk through similarity-weighted
    aggregation
  - score-family selection happens on `valid_tune`, while the conformal
    threshold remains fit on `valid_cal`
  - manifests record `local_risk_contract =
    relation_aware_scalar_risk_aggregation_v2`
- `conformal_knn_risk_router` is the target-node KNN-risk counterpart to
  `calibrated_local_risk_router`:
  - it keeps the same conformal split discipline, budget metrics, and
    stage-2 screening-only boundary
  - each target node is the KNN center; the KNN support group contributes local
    risk, prediction-disagreement, high-risk-mass, safe-support, and similarity
    evidence back to that target node's router score
  - it uses KNN similarity support groups over frozen node representations
    instead of relation-channel 1-hop aggregation, but KNN-updated centers are
    restricted to labeled graph nodes; support nodes keep base conformal risk
  - when the consumed frozen G0 artifact already contains a HyperScan-style
    second-view KNN branch, the router now defaults to that artifact's KNN
    setup for `k`, candidate scope, `x_new` representation, and HyperScan-style
    local calibration scope unless an
    explicit `--conformal_knn_*` flag overrides it
  - for `neighborloader_batch`, this inheritance is necessarily post-hoc:
    the router inherits the branch semantics and deterministic labeled-graph
    support space, but it does not replay stochastic training-time subgraph
    batches
  - `--conformal_knn_candidate_scope hyperscan_full` is the strict
    HyperScan-aligned experiment for identifying concrete high-threat users:
    it estimates each labeled center's risk from the graph-wide KNN support
    group, matching HyperScan's full feature-pool hyperedge construction when
    paired with `--conformal_knn_repr_source x_new`
  - in `hyperscan_full`, `k` follows the HyperScan hyperedge convention and
    includes the center node; the router consumes at most `k - 1` other members
    as support evidence for that center
  - the active full-pool backend is exact batched `torch` top-k over normalized
    feature vectors. This preserves the same nearest-neighbor ordering as
    Euclidean KNN on normalized vectors, but avoids high-dimensional cKDTree
    single-core stalls on the 229k-node full graph.
  - `labeled_full` and `labeled_relation_1hop` remain bounded ablations
  - `--conformal_knn_repr_source x_new` is the HyperScan-style construction-space
    ablation: it builds the KNN space from `cat(x_low, x_in)`, where the
    router now first consumes exported frozen G0 `x_new` when present; current
    `rgcn` / `botrgcn` frozen SimTeG artifacts export this forward-native
    relation-hidden construction space. Older artifacts that do not export
    `outputs["x_new"]` are no longer treated as HyperScan-aligned inputs; they
    should be regenerated or kept as explicit final-hidden controls
  - `node_repr` is no longer a valid HyperScan-aligned KNN construction space.
    It remains only a final-hidden control for non-HyperScan ablations; pairing
    it with a HyperScan second-view graph mode is rejected because routed
    base-wrong nodes showed prediction-echo behavior in that space.
  - `--conformal_knn_repr_source fused_x` is the matched final-hidden
    comparison surface: it consumes the explicit exported detector fused hidden
    when available and otherwise falls back to legacy `node_repr`
  - the router now also computes official NCP-style weighted support features:
    it applies `exp(-distance/lambda_L)` over the selected KNN support group,
    following `1995subhankar1995/NCP`'s `HypertuningBothLamdas` /
    `TestProcedure` weighting pattern, and records the effective support size
    in the manifest
  - `--conformal_knn_ncp_lambda` controls that localization temperature;
    `--conformal_knn_learning_mode fixed` preserves the historical fixed score
    families, while `ncp_local` computes target-specific conformal features
    from KNN calibration neighbors and selects among nonparametric NCP-local
    score families by tune AUPRC-error first, with no tune-split logistic
    residual-error model. `learned_logistic` is the bounded learned-router
    ablation on top of the same feature bundle: it fits a balanced logistic
    residual-risk scorer over target-node base risk plus structured KNN/local
    risk features, keeps the frozen G0 logits unchanged, and should be framed
    as a router-quality comparison rather than a new classifier claim
  - `--conformal_knn_local_calibration_scope same_hyperedge` is the closer
    HyperScan-aligned router realization: it treats the target-centered
    HyperScan KNN hyperedge as the direct selected-K risk neighborhood and
    scores target risk directly from that same local group, instead of issuing
    an additional calibration-only KNN query
  - `--conformal_knn_neighbor_mode`,
    `--conformal_knn_similarity_threshold`, `--conformal_knn_min_support`,
    `--conformal_knn_adaptive_max_k`, and
    `--conformal_knn_hubness_correction` are router-only support-quality
    ablation axes. They test mutual/adaptive/threshold-filtered or hubness
    downweighted KNN evidence without changing the frozen graph detector,
    consuming LLM outputs, or claiming downstream
    Conformal Risk Control.
  - `--conformal_knn_score_family_override base_only` is the strict target-only
    control; `ncp_local_conformal`, `ncp_local_margin`, and
    `ncp_knn_weighted_mean` force direct NCP-local evidence baselines; `auto`
    selects among target+KNN support-group score families on the validation
    tune split. Under `same_hyperedge`, that auto pool is now the
    same-hyperedge tail-risk family set, including calibration-only inner-tail
    variants, rather than the pure local-conformal family set.
  - the current local-sparsity follow-up adds an ESS-shrunk calibration-tail
    variant: the router keeps the HyperScan support hyperedge fixed, computes
    a calibration-only inner-tail risk inside that group, and shrinks back
    toward the broader same-hyperedge global-tail risk when the local
    calibration effective sample size is too small
  - for current research reporting, treat
    `same_hyperedge_calibration_shrunk_tail` as the fixed same-hyperedge
    router anchor. `auto` remains useful for within-family exploration, but it
    should not replace the anchor in matched comparisons.
  - manifests expose default-budget `selected_nodes`,
    `selected_nodes_by_budget`, and `top_ranked_targets`, making the selected
    KNN-risk target users directly reusable as `--routed_nodes_path`.
    `top_ranked_anchors` remains a deprecated compatibility alias only.
  - this router is the required target-node baseline for future router work:
    learned routers, LLM-consuming routers, and refiner-selection routers must
    report matched frozen G0, split, budget, and labeled-only scope comparisons
    against it before claiming router-quality improvement
  - minimum comparison metrics are AUROC-error, AUPRC-error, AURC,
    ErrRecall@K, Precision@K, Lift@K, selected-node overlap, and downstream
    routed-refiner gain per selected node when a refiner is involved
  - future router proposals should state how they score a target node using
    target-only evidence, target->KNN support-group evidence, and any optional
    LLM/refiner evidence; the comparison claim is about ranking target nodes,
    not about selecting KNN neighbors as the routed objects
  - it does not call an LLM, rewrite the graph, or claim classifier
    improvement by itself
  - graph-consumption positioning:
    - simple KNN augment under `graph_refine_mode` is a control family for
      diagnosing whether task-shaped semantic KNN helps plain RGCN
  - the active mainline graph path is HyperScan-style second-view
    construction with explicit `x_low`, `x_new`, `x_high`, and `fused_x`
    roles
  - routed-selective HNN runs now additionally expose
    `highorder_consumer_mask`, `fused_x_hnn`, `fused_x_lowonly`,
    `logits_hnn`, and `logits_lowonly` in `outputs.pt`; `fused_x` remains the
    actual final detector hidden after routed/non-routed mixing
  - risk-gated residual runs now additionally expose
    `highorder_consumer_gate` and `highorder_risk_score` in `outputs.pt`; these
    are the realized per-node residual strength and the aligned frozen risk
    input, respectively
  - routed high-pass is positioned as routed-only graph-consumption
    correction, not as generic KNN augmentation
- manifests now record:
  - `official_code_verified = false`
  - `paper_aligned = true`
  - `repo_locally_verified = false`
  - `verified_scope = node_selection_uncertainty_only`

### Phase C-D: local enhancement / graph-aware repair

Current code anchors:

- `LLMbot/code/operators.py`
- `LLMbot/code/trainer_graph.py`
- `LLMbot/code/trainer_legacy_impl.py`

Public tasks:

 - `local_conformal_diagnostic`
 - `local_conflict_prune_diag`
 - `local_dignn_conflict_refine_diag` (paper-faithful DIGNN-style dual-view proxy: topology-view MLP + attribute-view MLP + attention fusion + MI objective; keeps removed-edge provenance for later evidence use)
- `semantic_operator_ablation`
- `repair_operator_ablation`
- parts of `joint_router_refinement`

Current boundary:

- local graph editing and propagation diagnostics are implemented
- the DIGNN-style local conflict refiner is now available as a separate
  public diagnostic stage
- `trainer_graph.py` now owns shared graph bundle resolution, graph provenance,
  edge-override reruns, and the public local graph diagnostic executors
- richer GLANCE repair branches remain internal-only
- external graph override now carries stronger provenance metadata, but the
  richer graph-aware / GLANCE branches are still concentrated in
  `trainer_legacy_impl.py`

### Phase E-F: selector / refinement

Current code anchors:

- `LLMbot/code/trainer_glance.py`
- `LLMbot/code/stage_runner.py`
- `LLMbot/code/trainer_legacy_impl.py`

Public tasks:

- `joint_router_refinement`
- `prompt_expert_quality_audit`
- `selector_ablation`
- `positioning_ablation`

Internal tasks:

- `glance_oracle_refinement_internal`
- `glance_full_graph_refinement_internal`
- `glance_counterfactual_router_internal`
- `glance_budgeted_refinement_internal`
- `glance_refiner_analysis_internal`

Current boundary:

- there is real internal implementation here
- `trainer_glance.py` now owns semantic artifact loading and semantic payload
  normalization for GLANCE-family stages, and now also owns the public
  `joint_router_refinement` executor, the public
  `prompt_expert_quality_audit` diagnostic executor, plus migrated internal
  GLANCE execution lanes
- only `joint_router_refinement` and `prompt_expert_quality_audit` are public
- internal GLANCE branches must not be described as public CLI capability or as
  completed official reproduction

### Phase G: positioning / stress

Public tasks:

- `minimal_pipeline`
- `backbone_stress_test`
- `appendix_ablation`

Interpretation:

- these remain diagnostic and positioning layers
- they do not imply that every upstream implementation branch is already fully
  modularized or claim-grade

## Current Mismatch Summary

### What is already aligned

- the active RoBERTa default is now seed-aware:
  `embeddings_iter_-1_seed_{seed}.pt` is the default semantic regime for
  seed-scoped backbone/refiner runs unless `--embedding_path` is explicitly set
- the old `finetuned_roberta_embeddings_iter_2_seed1.pt` path remains a
  historical compatibility branch inside the same backbone/refiner family, but
  the active semantic default for current runs is still
  `embeddings_iter_-1_seed_{seed}.pt`
- public task naming now uses canonical task names
- parser emits canonical fields and keeps hidden compatibility aliases
- stage visibility is explicitly represented in `stage_registry.py`
- public docs can now refer to `LLMbot/` without relying on deprecated
  `baseline/core` naming
- strict GLANCE no longer silently falls back to seed-1 semantic artifacts
- the current public `joint_router_refinement` router is now explicitly a
  task-adapted reliability router rather than a paper-text GLANCE advantage
  router:
  - confidence features are temperature-scaled before router feature building
  - router social features are direction-aware for TwiBot20 (`in/out` degree,
    relation counts, directional homophily/disagreement, sparse-node flags,
    reciprocity)
  - router supervision is now `base_wrong` reliability with pairwise ranking,
    while `oracle_advantage` remains a diagnostic trace rather than the main
    supervision target
- the joint training loop still keeps GLANCE-style batch top-k routing and
  mixed-path refiner optimization
- for full-seed router inspection, the public stage now writes per-seed router
  summaries plus experiment-root multi-seed aggregate CSV/JSON tables so
  router quality can be compared directly without hand-merging stage manifests
- strict GLANCE can now be run in two training-data regimes under the same
  joint protocol:
  - `--joint_train_node_cap 3000` for the paper-style capped subset
  - `--joint_train_node_cap 0` for a TwiBot20-adapted full-train comparison
- strict GLANCE joint refiner now also exposes task-specific routed-node
  diagnostics through:
  - `--joint_refiner_explicit_gate` for an explicit keep/change gate
  - `--joint_refiner_target_mode {predict,keep_change}` to compare direct label
    prediction against keep/change learning on routed nodes
  - `--joint_refiner_gate_target {base_wrong,utility_positive}` to aim the gate
    either at base detector errors or positive raw-refiner utility
  - `--joint_refiner_weight_mode` and companion weights to re-emphasize
    base-wrong / oracle-utility-positive routed samples without changing the
    router contract
  - `--joint_utility_advantage_experiment {off,utility_gate_only,botmoe_selector_only,utility_gate_botmoe_selector}`
    to expose the approved utility-gate, BotMoE-selector, and combined
    comparison presets without changing the default behavior
  - `--joint_refiner_gate_policy {soft_mix,hard_keep_change}` and
    `--joint_refiner_gate_threshold` to reserve the public gate-application
    policy for soft interpolation versus thresholded keep/change decisions
  - `--joint_correction_moe_expert_weight` and
    `--joint_correction_moe_utility_weight` to control the auxiliary expert-head
    CE and per-expert utility BCE terms for `utility_correction_moe`
  - `--joint_correction_moe_utility_target {loss_advantage,decision_gain,hybrid}`
    to switch that utility BCE between loss-margin improvement, discrete
    correction gain, or their union
- strict GLANCE joint training now records per-epoch router diagnostics and
  routed-refiner fix/break deltas so bottlenecks can be attributed separately
  from the final mixed-path score
- CSV summary writers now append target-specific diagnostic columns discovered
  in later rows before writing, so utility-target sweeps can add valid/test-only
  correction metrics without breaking artifact export.
- public `joint_router_refinement` now also has a frozen-router reuse protocol:
  - `--joint_routing_protocol frozen_router_reuse`
  - `--joint_router_reuse_root`
  - this path reuses a prior router checkpoint, scaler, selected budget, and
    selected beta so routed-node evidence upgrades can be compared on the same
    routed set instead of through a re-trained router
  - this is the preferred graph-stage comparison surface when the intended
    method contract is fixed-router routed-only evidence, rather than
    LLM-assisted routing or KNN construction
- strict GLANCE counterfactual routing keeps the learned `router_score` proxy
  separate from `oracle_advantage`, which is only a post-hoc reward trace
- public strict GLANCE now requires same-run dependency provenance:
  `joint_router_refinement` must read the current run's own
  `preparation/graph_detector` artifact and its semantic tensor must match that
  artifact's recorded `feature_manifest.path`
- the one public exception is the new refiner-only prompt-ablation path:
  `--joint_refiner_embedding_path` may replace the refiner semantic branch
  while backbone-side provenance remains pinned to the same-root
  `graph_detector_prepare` artifact
- read-only external backbone reuse is also allowed for that same ablation
  path, as long as `--external_frozen_g0_root` is paired with
  `--joint_refiner_embedding_path`
- the routed-node diagnostic script
  `scripts/routed_explain_qwen3_embedding_mlp.py` now covers two distinct
  study surfaces under one file:
  - `routed_classifier`:
    - direct Qwen3-Embedding-8B encoding from explanation sidecars
    - or precomputed prompt-expert tensors loaded from
      `--cached_embedding_path`, with `--cached_embedding_key` selecting either
      the full concat embedding or a single expert component
  - `frozen_gnn_feature_swap`:
    - build one explanation document per routed node from the expert sidecars
    - encode that document with a RoBERTa-family encoder using the current
      SimTeG-style text-consumption contract
    - replace the routed-node feature rows and replay the original frozen GNN
      weights without retraining
- this keeps routed-node explain studies aligned with the clean
  explanation-first prompt-expert cache path while also supporting a stricter
  frozen-backbone counterfactual: whether explanation-derived node features
  alone can help the already-strong SimTeG GNN
- GATS reuse now validates input graph-detector provenance before reuse

### What is still transitional

1. most execution logic still lives in `trainer_legacy_impl.py`
2. some runtime-only helper artifacts remain in a compatibility transition
3. `joint_router_refinement` is no longer a pure paper-text router alignment;
   it should be described as a reliability-first social-bot adaptation under
   GLANCE-style joint routing/refinement
4. active code still backfills legacy shadow fields for one compatibility window
5. not every internal manifest and runtime note has been fully renamed yet
6. GLANCE runner/helper tails have moved into `trainer_glance.py`, and
   `local_conflict_prune_diag` has moved into `trainer_graph.py`; estimator
   matrix paths and some graph-helper compatibility tails still remain in
   `trainer_legacy_impl.py`
7. `trainer_legacy_impl.py` still retains historical duplicate preparation and
   semantic blocks during the current rebinding window
8. the active pre-iter semantic comparison branch is already seed-aware:
   `LLMbot/experiments/embedding_compare_20260527/rgt_pre_iter_default`
   uses `embeddings_iter_-1_seed_{seed}.pt` for seeds `1..5`

## Current Claim Boundary

Do not state the following based on current code alone:

- that all implemented GLANCE branches are public and supported
- that LOGIN official code has been fully verified
- that all router/refiner paths are already claim-grade ready
- that the current mainline has finished a stable multi-seed production-quality
  modular extraction

## Current Readiness Summary

| Area | Implementation status | Alignment status | Main blocker |
| --- | --- | --- | --- |
| parser and dispatch | implemented | strong | remaining compatibility window |
| preparation artifacts | implemented | strong | artifact/runtime naming cleanup |
| estimator/router layer | implemented | moderate | execution-body concentration |
| GLANCE refinement family | implemented with public/internal split | moderate | modular extraction not finished |
| documentation | updated toward canonical surface | moderate | risk log and runtime notes still need ongoing maintenance |

## Refactor Status 2026-05-26

The current mainline is no longer in a public-contract mismatch state.

What is true now:

- public task names and public CLI flags are canonical
- stage visibility is explicit in `LLMbot/code/stage_registry.py`
- `LLMbot/code/main.py` dispatch is registry-based
- `LLMbot/code/trainer.py` is a thin facade
- `code/artifact_contracts.py` and `code/runtime_env.py` replaced `stage_helpers.py`
- `code/stage_runner.py` now owns the shared `StageRunner` skeleton
- `code/trainer_graph.py` now owns shared graph runtime logic
- `code/trainer_glance.py` now owns shared GLANCE semantic input logic and the
  public `joint_router_refinement` executor plus migrated internal GLANCE
  execution lanes
- `code/trainer_preparation.py`, `code/trainer_semantic.py`, and
  `code/trainer_distillation.py` are now real first-layer owners
- canonical artifact namespaces exist for preparation and public stage writes

What is not finished:

- `trainer_legacy_impl.py` still concentrates estimator/matrix execution,
  local conformal helper tails, and compatibility rebinding
- GLANCE runner/helper tails have moved into `trainer_glance.py`, and
  `local_conflict_prune_diag` has moved into `trainer_graph.py`
- some runtime-only helper artifacts still retain migration-era compatibility
  naming
- the active code still supports one compatibility window for legacy fields and
  legacy task names
- `main.py` now consumes `runner_kind`, `claim_grade_allowed`,
  `requires_canonical_split`, `forces_use_gnn`, and `graph_data_mode` from
  `stage_registry.py`; further cleanup should focus on downstream legacy
  shadow-field reads rather than adding another task truth table

## Code Governance Status 2026-05-26

The codebase is now aligned at the public-contract layer:

- active mainline identity is fixed to `LLMbot/`
- public task names and public flags are canonical
- internal-only GLANCE branches are explicitly separated from public tasks
- research-boundary notes and code-development docs now describe the same
  active mainline

The codebase is not yet aligned at the implementation-ownership layer:

- `trainer_legacy_impl.py` is still the dominant execution body
- several extracted modules still act as re-export or forwarding surfaces
- some runtime-only helper artifacts and internal notes still carry migration
  vocabulary

This means the remaining work is structural refactoring and contract cleanup,
not another change in research pipeline meaning.

## Source-Layout Status 2026-07-04

The active implementation has been moved into a flat `LLMbot/code/` directory.
This is intentionally not a large-project package taxonomy: do not add
`stages/`, `methods/`, or similar subdirectories until deletion and function
consolidation reduce the current migration body.

Current interpretation:

- `LLMbot/` is the operator working directory, runner-script surface, and
  artifact parent.
- `LLMbot/code/` is the active Python source surface.
- `LLMbot/main.py`, `LLMbot/precompute.py`, and `LLMbot/preprocess.py` are
  compatibility entrypoints into `code/`.
- `LLMbot/run_*.py`, launch scripts, `experiments/`, and `server_logs/` remain
  outside `code/`.
- `LLMbot/baseline/` remains deprecated.

## Near-Term Engineering Goals

1. keep the flat `LLMbot/code/` active source directory stable while removing
   dead or duplicate code in small reviewed slices
2. extract the remaining prompt-expert / relation-aware GLANCE helper tails
   into `trainer_glance.py`
3. remove direct active-mainline reads of legacy shadow fields while keeping
   parse compatibility for one window
4. continue migrating active internal reads away from legacy shadow fields
   after parser normalization
5. finish canonicalizing runtime-only artifact surfaces after module ownership
   is transferred

## Documentation Maintenance Rule

If stage behavior, public/internal stage boundaries, or pipeline-to-code
alignment changes, update this file in the same task together with
`docs/ARCHITECTURE.md` and `code.md`.
