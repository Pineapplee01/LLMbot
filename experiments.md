# LLMbot Server Experiments

This file is the registry for GPU-server experiments under:

`/root/workspace/LMbot/LLMbot`

Use it before launching, resuming, or reusing an experiment. The goal is to keep
server runs reproducible without scattering one-off timestamped directories.

## Operating Rules

- Do not create new arbitrary timestamp-suffixed artifact roots.
- Use stable semantic experiment IDs and reuse/update the registry entry.
- Record every server run here before or immediately after launch.
- Keep commands runnable from `/root/workspace/LMbot/LLMbot`.
- Keep artifact roots, manifest paths, metrics paths, and status explicit.
- Do not move old artifact directories unless the user explicitly asks for
  archival cleanup; old paths may be referenced by manifests.
- For frozen SimTeG comparisons, reuse the high-base seed-1 baseline unless the
  experiment explicitly declares another baseline:
  - Acc: `0.863905325443787`
  - Macro-F1: `0.862442357206568`
  - Frozen root:
    `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_roberta_20260527/seed_1`
  - Base outputs:
    `/root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_classifier_qwen3_20260601/base_outputs_for_routed_classifier.pt`

## Naming Convention

Use stable IDs:

`<topic>__<method>__<baseline>__seed<seed>`

Examples:

- `mpe__clean_v2_roberta_cache_classifier__highbase__seed1`
- `mpe__roberta_base_feature_swap_gnn__highbase__seed1`
- `mpe__joint_refiner_gate__highbase__seed1`

Recommended artifact root:

`/root/workspace/LMbot/LLMbot/experiments/<experiment_id>`

If an old timestamped root already exists, keep it in place and record it below.
Do not create a second timestamped clone for reruns.

## Reusable Inputs

| name | path | notes |
|---|---|---|
| high-base frozen SimTeG root | `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_roberta_20260527/seed_1` | canonical high-base seed-1 frozen GNN root |
| high-base routed nodes | `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json` | train 752, valid 592, test 296, total 1640 |
| routed Glance/BotSay prompt patch | local code patch on `2026-06-07` | direct `expert_ego` now uses a BotSay-aligned `tweet + metadata` classifier shell; direct `expert_graph_following` / `expert_graph_follower` now use Glance-style `EGO + HOP1 + Category?`; routed refiner adds `raw_concat_ego_following` and `raw_concat_ego_follower` |
| clean v2 routed explanations | `/root/workspace/LMbot/datasets/TwiBot-20/prompt_expert_v2_clean_routed_preiter_cache_20260531` | clean sidecars for graph_following, graph_follower, tweet, conflict |
| clean v2 finetuned-RoBERTa cache | `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt` | routed explanation expert cache |
| pretrained roberta-base snapshot | `/root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b` | explicit pretrained encoder path |

## Experiment Registry

### dgp__v2_strict_answer_token__highbase_routed_ctxfull__seed1

- Status: pending launch.
- Scope: strict-DGP prompt shell, routed-node answer-token finetune.
- Boundary:
  - uses the modified strict-DGP `precompute.py` prompt shell:
    - predictor prompt uses minimal `Instruct / Query / ASSISTANT_ANSWER`
    - summary prompts use task-agnostic
      `Summarize ... within 10 tokens`
  - supervision scope is the fixed routed split:
    `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
  - neighbor/context support is drawn from `full_graph_support` during
    precompute through `--context_graph_variant full_graph_support`
  - semantic finetune remains on the labeled graph runtime contract and
    consumes the generated prompt sidecar via
    `--semantic_text_source_path`
  - backbone path is Qwen2.5 answer-token SFT:
    `semantic_encoder_finetune --semantic_encoder qwen3_peft --semantic_supervision_mode answer_token`
- Planned prompt cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v2_strict_norm_text_following_summary_qwen25_routed_ctxfull_seed1.pt`
- Planned artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/dgp__v2_strict_answer_token__highbase_routed_ctxfull__seed1`
- Planned log:
  `/root/workspace/LMbot/LLMbot/server_logs/dgp__v2_strict_answer_token__highbase_routed_ctxfull__seed1.log`

### dgp__v2_strict_answer_token__highbase_labeled_ctxfull__seed1

- Status: pending launch.
- Scope: strict-DGP prompt shell, labeled-node answer-token finetune with
  full-graph support context.
- Boundary:
  - uses the modified strict-DGP `precompute.py` prompt shell:
    - predictor prompt uses minimal `Instruct / Query / ASSISTANT_ANSWER`
    - summary prompts use task-agnostic
      `Summarize ... within 10 tokens`
  - supervision scope is the default labeled train/valid/test split
  - prompt construction targets labeled nodes only through
    `--center_node_scope labeled`
  - neighbor/context support is drawn from `full_graph_support` during
    precompute through `--context_graph_variant full_graph_support`
  - semantic finetune remains on the labeled graph runtime contract and
    consumes the generated prompt sidecar via
    `--semantic_text_source_path`
  - backbone path is Qwen2.5 answer-token SFT:
    `semantic_encoder_finetune --semantic_encoder qwen3_peft --semantic_supervision_mode answer_token`
- Planned prompt cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v2_strict_norm_text_following_summary_qwen25_labeled_ctxfull_seed1.pt`
- Planned artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/dgp__v2_strict_answer_token__highbase_labeled_ctxfull__seed1`
- Planned log:
  `/root/workspace/LMbot/LLMbot/server_logs/dgp__v2_strict_answer_token__highbase_labeled_ctxfull__seed1.log`

### dgp__v2_norm_text_following_summary_qwen25__highbase_routed__seed1

- Status: pending server smoke / full routed run.
- Scope: DGP v2 routed-node LLM-as-predictor migration.
- Boundary:
  - uses `precompute.py --prompt_mode dgp_predictor_v2`
  - main variant is `--dgp_prompt_variant norm_text_following_summary`
  - uses `--dgp_neighbor_summary_k 5`
  - target node evidence is `norm_user_text`, treated as fine-grained
    tweet+metadata evidence
  - raw `norm_user_text` is parsed into LLM-friendly `PROFILE`,
    `TWEET_BEHAVIOR`, and `TWEET_SAMPLES` sections before Qwen generation or
    final prompt encoding
  - routed nodes with no selected following neighbor get a deterministic
    sparse-context summary; Qwen relation-context generation is reserved for
    nodes with selected neighbor evidence
  - neighbor and relation summaries are constrained to English; multilingual
    or noisy source text is summarized as an evidence-quality cue rather than
    copied into the sidecar
  - row-level quality failures after bounded retry are written as explicit
    `quality_fallback` limited-evidence summaries with the failure reason
    recorded; this avoids bad text while preserving routed-node coverage
  - selected following neighbor `norm_user_text` rows are summarized first,
    then compressed into a following-context summary
  - final prompt asks for exactly one answer token: `Yes` means bot and `No`
    means human
  - no dataset labels, frozen SimTeG correctness, or oracle fix/break outcomes
    are inserted into the prompt text
  - `semantic_encoder_finetune --semantic_encoder qwen3_peft
    --semantic_supervision_mode answer_token` is now the active faithful
    answer-token route for this prompt sidecar
  - `--semantic_supervision_mode classifier` remains the compatibility ablation
    that trains a hidden-state classifier head on the same prompt text
- Stable routed split:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
- Planned prompt cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v2_norm_text_following_summary_qwen25_embed.pt`
- Planned clean component cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v2_norm_text_following_summary_qwen25_component_cache_clean_english`
- Planned artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/dgp__v2_norm_text_following_summary_qwen25__highbase_routed__seed1`
- Planned stages:
  1. routed prompt precompute smoke with `--limit 2`
  2. full routed prompt precompute over routed union
  3. `semantic_encoder_finetune` over the prompt sidecar with Qwen2.5 PEFT
     and `--semantic_supervision_mode answer_token`
  4. `semantic_embedding_classifier` over the Qwen2.5 query embedding cache
- Primary comparisons:
  - frozen SimTeG full-test Acc/F1: `0.8639 / 0.8624`
  - previous DGP v1 Qwen2.5 PEFT predictor: fix/break/net `42 / 76 / -34`
  - previous CALM v1 query-embedding MLP: fix/break/net `57 / 114 / -57`

### dgp__v2_qwen25_answer_token__highbase_routed__seed1

- Status: pending server smoke / full run.
- Scope: latest DGP v2 prompt sidecar plus answer-token finetune.
- Boundary:
  - reuses the existing routed DGP v2 prompt sidecar:
    `/root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v2_norm_text_following_summary_qwen25_embed_prompts.jsonl`
  - does not rebuild prompts or summary sidecars
  - trains `semantic_encoder_finetune --semantic_encoder qwen3_peft
    --semantic_supervision_mode answer_token`
  - label mapping is fixed to `Yes -> bot`, `No -> human`
  - runtime evaluation remains deterministic by scoring the conditional
    completion likelihood of `Yes` versus `No`, then writing back the standard
    `outputs.pt {logits, prob, pred, labels}` contract
  - this is the active faithful DGP token-finetune route in the current
    mainline; the older hidden-state classifier route remains a compatibility
    ablation
- Stable routed split:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
- Planned artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/dgp__v2_qwen25_answer_token__highbase_routed__seed1`
- Planned validation steps:
  1. smoke finetune with routed train cap and small step count
  2. full routed answer-token finetune
  3. replay against frozen high-base SimTeG for routed-node fix/break/net
- Primary comparison:
  - `dgp__v2_qwen25_peft_predictor_norm_text_following_summary__highbase_routed__seed1`
    as the old hidden-state classifier-head baseline on the same prompt sidecar

### dgp__qwen25_peft_predictor__highbase_routed__seed1

- Status: completed on server.
- Scope: DGP-style routed-node LLM-as-predictor migration.
- Boundary:
  - uses `precompute.py --prompt_mode dgp_predictor_v1`
  - prompt keeps detailed target profile/tweet evidence and coarse ranked
    following/follower neighbor context
  - trains `semantic_encoder_finetune --semantic_encoder qwen3_peft` over the
    generated prompt sidecar through `--semantic_text_source_path`
  - actual model path is Qwen2.5-7B-Instruct; `qwen3_peft` is the existing
    mainline Qwen CausalLM PEFT classifier plumbing name
  - this is DGP-style Qwen PEFT classifier tuning, not a full generative
    answer-token SFT reproduction
- Qwen3-Embedding note:
  - attempted `Qwen3-Embedding-8B` loading failed under server
    `transformers==4.49.0` because `model_type=qwen3` is unsupported
  - completed run therefore uses Qwen2.5 hidden-state query embeddings and
    Qwen2.5 PEFT predictor; do not report it as Qwen3-Embedding evidence
- Stable routed split:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
- Stable DGP prompt cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v1_target_fine_neighbor_coarse_qwen25_embed.pt`
- Stable prompt sidecar:
  `/root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v1_target_fine_neighbor_coarse_qwen25_embed_prompts.jsonl`
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/dgp__qwen25_peft_predictor__highbase_routed__seed1`
- Key artifacts:
  - manifest: `/root/workspace/LMbot/LLMbot/experiments/dgp__qwen25_peft_predictor__highbase_routed__seed1/seed_1/preparation/semantic_encoder/manifest.json`
  - metrics: `/root/workspace/LMbot/LLMbot/experiments/dgp__qwen25_peft_predictor__highbase_routed__seed1/seed_1/preparation/semantic_encoder/metrics.json`
  - outputs: `/root/workspace/LMbot/LLMbot/experiments/dgp__qwen25_peft_predictor__highbase_routed__seed1/seed_1/preparation/semantic_encoder/outputs.pt`
  - embeddings: `/root/workspace/LMbot/LLMbot/experiments/dgp__qwen25_peft_predictor__highbase_routed__seed1/seed_1/preparation/semantic_encoder/embeddings.pt`
  - adapter: `/root/workspace/LMbot/LLMbot/experiments/dgp__qwen25_peft_predictor__highbase_routed__seed1/seed_1/preparation/semantic_encoder/adapter`
- Direct routed-node metrics:
  - train Acc/F1: `0.5957 / 0.5656`
  - valid Acc/F1: `0.5152 / 0.4910`
  - test Acc/F1: `0.5236 / 0.4979`
- Frozen SimTeG replacement metrics on full test:
  - base full-test Acc/F1: `0.8639 / 0.8624`
  - after replacing routed-test nodes: Acc/F1 `0.8352 / 0.8328`
  - fix/break/net on routed test: `42 / 76 / -34`
- Validation:
  - manifest status `completed`
  - routed split counts `train=752, valid=592, test=296, total=1640`
  - `semantic_text_source` override count `1640`
  - `outputs.pt` logits/prob and `embeddings.pt` are finite, no NaN

### calm__dgp_query_qwen25_embedding_mlp__highbase_routed__seed1

- Status: completed on server.
- Scope: CALM-style query-embedding-MLP comparison using the same DGP-style
  routed prompt text.
- Boundary:
  - uses the `.pt` embedding cache produced by
    `precompute.py --prompt_mode dgp_predictor_v1`
  - runs `semantic_embedding_classifier --embedding_path ...`
  - uses the same routed train/valid/test split as the DGP PEFT run
  - does not update the original GNN, router, or joint refiner
  - actual embedding cache is Qwen2.5-Instruct last-token hidden-state
    embedding, not Qwen3-Embedding-8B
- Qwen dtype note:
  - first Qwen2.5 embedding run forced fp16 and produced NaN rows; that run was
    discarded
  - valid run uses checkpoint dtype (`torch_dtype="auto"`) and passed finite
    checks
- Stable routed split:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
- Stable embedding cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v1_target_fine_neighbor_coarse_qwen25_embed.pt`
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/calm__dgp_query_qwen25_embedding_mlp__highbase_routed__seed1`
- Key artifacts:
  - manifest: `/root/workspace/LMbot/LLMbot/experiments/calm__dgp_query_qwen25_embedding_mlp__highbase_routed__seed1/seed_1/preparation/semantic_embedding_classifier/manifest.json`
  - metrics: `/root/workspace/LMbot/LLMbot/experiments/calm__dgp_query_qwen25_embedding_mlp__highbase_routed__seed1/seed_1/preparation/semantic_embedding_classifier/metrics.json`
  - outputs: `/root/workspace/LMbot/LLMbot/experiments/calm__dgp_query_qwen25_embedding_mlp__highbase_routed__seed1/seed_1/preparation/semantic_embedding_classifier/outputs.pt`
  - embedding ref: `/root/workspace/LMbot/LLMbot/experiments/calm__dgp_query_qwen25_embedding_mlp__highbase_routed__seed1/seed_1/preparation/semantic_embedding_classifier/embeddings_ref.pt`
- Direct routed-node metrics:
  - train Acc/F1: `0.5186 / 0.4680`
  - valid Acc/F1: `0.4561 / 0.3860`
  - test Acc/F1: `0.4459 / 0.3838`
- Frozen SimTeG replacement metrics on full test:
  - base full-test Acc/F1: `0.8639 / 0.8624`
  - after replacing routed-test nodes: Acc/F1 `0.8157 / 0.8157`
  - fix/break/net on routed test: `57 / 114 / -57`
- Validation:
  - manifest status `completed`
  - routed split counts `train=752, valid=592, test=296, total=1640`
  - embedding cache row count `11826`, target node count `1640`, dim `3584`
  - `outputs.pt` logits/prob are finite, no NaN

### peft__qwen25_routed_descandmeta__highbase__seed1

- Status: completed on server
- Scope: routed-node-specialized Qwen LoRA ablation
- Claim boundary:
  this is a BotSay-compatible routed-node PEFT adaptation, not an official
  BotSay reproduction. It reuses the existing `botsay_official` SFT shell,
  but trains and evaluates only on the routed-node `train / valid / test`
  subsets from the fixed high-base routed set.
- Code root:
  `/root/workspace/LMbot/botsay_official`
- Routed split source:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
- Planned commands:
  1. `CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python approach-finetune.py -d Twibot-20 -a descandmeta -n 4 --tweet 3 --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json --routed_split train --output_name Twibot-20-descandmeta-routed-train-norm-user-text`
  2. `CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python sft.py -i Twibot-20-descandmeta-routed-train-norm-user-text -m qwen25 -e 5`
  3. `CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python approach-finetune_eval.py -d Twibot-20 -a descandmeta --base_model qwen25 --tuned_model_name Twibot-20-descandmeta-routed-train-norm-user-text -n 4 --tweet 3 --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json --routed_split valid --batch_size 4 --max_new_tokens 24 --temperature 0.0 --output_path /root/workspace/LMbot/botsay_official/probs/descandmeta_routed_valid_qwen25_peft_n4_b4.json`
  4. `CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python approach-finetune_eval.py -d Twibot-20 -a descandmeta --base_model qwen25 --tuned_model_name Twibot-20-descandmeta-routed-train-norm-user-text -n 4 --tweet 3 --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json --routed_split test --batch_size 4 --max_new_tokens 24 --temperature 0.0 --output_path /root/workspace/LMbot/botsay_official/probs/descandmeta_routed_test_qwen25_peft_n4_b4.json`
- Expected artifacts:
  - `/root/workspace/LMbot/botsay_official/corpus/Twibot-20-descandmeta-routed-train-norm-user-text.jsonl`
  - `/root/workspace/LMbot/botsay_official/corpus/Twibot-20-descandmeta-routed-train-norm-user-text/`
  - `/root/workspace/LMbot/botsay_official/probs/descandmeta_routed_valid_qwen25_peft_n4_b4.json`
  - `/root/workspace/LMbot/botsay_official/probs/descandmeta_routed_test_qwen25_peft_n4_b4.json`
- Actual completion artifacts:
  - adapter:
    `/root/workspace/LMbot/botsay_official/corpus/Twibot-20-descandmeta-routed-train-norm-user-text/adapter_model.safetensors`
  - valid eval:
    `/root/workspace/LMbot/botsay_official/probs/descandmeta_routed_valid_qwen25_peft_n4_b2.json`
  - test eval:
    `/root/workspace/LMbot/botsay_official/probs/descandmeta_routed_test_qwen25_peft_n4_b2.json`
  - log:
    `/root/workspace/LMbot/LLMbot/server_logs/peft__qwen25_routed_descandmeta__highbase__seed1.log`
- Notes:
  - the completed eval outputs are the `_b2` files above, not the originally
    planned `_b4` paths
  - routed eval JSON rows contain `id / gold / pred / response / prompt_chars`
    and align to the routed split in file order
- Direct routed-node metrics:
  - valid:
    - Acc: `0.5389`
    - Macro-F1: `0.5325`
    - frozen SimTeG routed-valid base: `0.6622 / 0.6530`
    - replay against frozen base: `fix=96`, `break=169`, `net=-73`
  - test:
    - Acc: `0.5642`
    - Macro-F1: `0.5605`
    - frozen SimTeG routed-test base: `0.6385 / 0.6292`
    - replay against frozen base: `fix=57`, `break=79`, `net=-22`
- Full-test override replay:
  - frozen SimTeG base full-test Acc/F1: `0.8639 / 0.8624`
  - after replacing routed-test nodes with Qwen PEFT predictions:
    `0.8453 / 0.8441`
- Interpretation boundary:
  - this routed-specialized Qwen PEFT line has real wrong-node recovery ability
    (`57` fixes on routed test), but calibration is still poor because
    `79` correct routed nodes are broken, so it does not improve the high-base
    frozen SimTeG end-to-end result

### semantic__roberta_base_routed__highbase__seed1

- Status: completed on server
- Scope: routed-node-specialized LLMbot semantic finetune and cached-embedding classifier
- Claim boundary:
  this is the clean routed-node RoBERTa comparison line. It reuses
  `LLMbot` semantic training code, starts from pretrained `roberta-base`,
  and replaces canonical `train / valid / test` with routed-node subsets from
  the fixed high-base routed set. It does not reuse the existing
  `finetuned-roberta` weights.
- Code root:
  `/root/workspace/LMbot/LLMbot`
- Routed split source:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
- Planned commands:
  1. `CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python main.py --experiment_task semantic_encoder_finetune --dataset TwiBot-20 --semantic_encoder roberta --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json --seeds 1 --device 0 --disable_wandb --experiment_name experiments/semantic__roberta_base_routed__highbase__seed1`
  2. `CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python main.py --experiment_task semantic_embedding_classifier --dataset TwiBot-20 --embedding_path /root/workspace/LMbot/LLMbot/experiments/semantic__roberta_base_routed__highbase__seed1/seed_1/preparation/semantic_encoder/embeddings.pt --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json --seeds 1 --device 0 --disable_wandb --experiment_name experiments/semantic__roberta_base_routed_classifier__highbase__seed1`
- Expected artifacts:
  - `/root/workspace/LMbot/LLMbot/experiments/semantic__roberta_base_routed__highbase__seed1/seed_1/preparation/semantic_encoder/manifest.json`
  - `/root/workspace/LMbot/LLMbot/experiments/semantic__roberta_base_routed__highbase__seed1/seed_1/preparation/semantic_encoder/metrics.json`
  - `/root/workspace/LMbot/LLMbot/experiments/semantic__roberta_base_routed_classifier__highbase__seed1/seed_1/preparation/semantic_embedding_classifier/manifest.json`
  - `/root/workspace/LMbot/LLMbot/experiments/semantic__roberta_base_routed_classifier__highbase__seed1/seed_1/preparation/semantic_embedding_classifier/metrics.json`
- Actual artifact roots:
  - `/root/workspace/LMbot/LLMbot/experiments/semantic__roberta_base_routed__highbase__seed1/seed_1/preparation/semantic_encoder`
  - `/root/workspace/LMbot/LLMbot/experiments/semantic__roberta_base_routed_classifier__highbase__seed1/seed_1/preparation/semantic_embedding_classifier`
- Server logs:
  - `/root/workspace/LMbot/LLMbot/server_logs/semantic__roberta_base_routed__highbase__seed1.log`
- Result summary:
  - routed semantic finetune test:
    - Acc: `0.5270`
    - Macro-F1: `0.5221`
  - routed cached-embedding classifier test:
    - Acc: `0.5338`
    - Macro-F1: `0.4265`
  - frozen SimTeG routed-test base on the same 296 nodes:
    - Acc: `0.6385`
    - Macro-F1: `0.6292`
  - routed replay against frozen SimTeG base:
    - semantic finetune direct head: `fix=30`, `break=63`, `net=-33`
    - cached-embedding classifier: `fix=47`, `break=78`, `net=-31`
  - merged full-test performance after overriding routed test nodes:
    - base frozen SimTeG: `Acc=0.8639`, `Macro-F1=0.8624`
    - semantic finetune direct head override: `Acc=0.8360`, `Macro-F1=0.8347`
    - cached-embedding classifier override: `Acc=0.8377`, `Macro-F1=0.8328`

### phase_a__roberta_base_routed_semantic_to_gnn__highbase__seed1

- Status: planned / launching
- Scope: Phase-A semantic-to-GNN replay
- Claim boundary:
  this is not a full legacy `distillation_pipeline` rerun. It is the closest
  active-mainline replay for the question "if we feed the routed-node-finetuned
  RoBERTa embedding back into the SimTeG-style GNN detector, does graph-side
  supervised training recover the utility?" The experiment swaps only the
  semantic tensor source in `graph_detector_prepare`, keeps canonical
  `train/valid/test` supervision on the graph detector, and measures the final
  graph classifier result.
- Code root:
  `/root/workspace/LMbot/LLMbot`
- Semantic source:
  `/root/workspace/LMbot/LLMbot/experiments/semantic__roberta_base_routed__highbase__seed1/seed_1/preparation/semantic_encoder/embeddings.pt`
- Planned command:
  `CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python main.py --experiment_task graph_detector_prepare --dataset TwiBot-20 --use_GNN --graph_backbone rgcn --embedding_path /root/workspace/LMbot/LLMbot/experiments/semantic__roberta_base_routed__highbase__seed1/seed_1/preparation/semantic_encoder/embeddings.pt --seeds 1 --device 0 --disable_wandb --experiment_name experiments/phase_a__roberta_base_routed_semantic_to_gnn__highbase__seed1`
- Expected artifacts:
  - `/root/workspace/LMbot/LLMbot/experiments/phase_a__roberta_base_routed_semantic_to_gnn__highbase__seed1/seed_1/preparation/graph_detector/manifest.json`
  - `/root/workspace/LMbot/LLMbot/experiments/phase_a__roberta_base_routed_semantic_to_gnn__highbase__seed1/seed_1/preparation/graph_detector/outputs.pt`
  - `/root/workspace/LMbot/LLMbot/experiments/phase_a__roberta_base_routed_semantic_to_gnn__highbase__seed1/seed_1/preparation/graph_detector/selection_metrics.json`

### botsay__qwen25_practical_descandmeta_icl__twibot20__test

- Status: pending server relaunch after runtime upgrade
- Scope: external-reference ablation
- Claim boundary:
  this is a practical BotSay-style Qwen ablation, not an official BotSay
  base-model reproduction. It keeps the official `descandmeta` task and prompt
  family, but uses the local `Qwen2.5-7B-Instruct` with full-GPU-preferred
  loading, batched decoding, bounded generation, and JSON result capture.
- Code root:
  `/root/workspace/LMbot/botsay_official`
- Dataset:
  `TwiBot-20`
- Approach:
  `descandmeta`
- Command:
  `CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python approach-descandmeta.py -m qwen25 -d Twibot-20 -n 4 --split test --batch_size 4 --max_new_tokens 24 --temperature 0.0 --output_path /root/workspace/LMbot/botsay_official/probs/descandmeta_qwen25_test_n4_b4.json`
- Expected output:
  `/root/workspace/LMbot/botsay_official/probs/descandmeta_qwen25_test_n4_b4.json`
- Server log:
  `/root/workspace/LMbot/LLMbot/server_logs/botsay_qwen25_descandmeta_icl_practical.log`
- Runtime note:
  this run replaces the earlier single-sample 16-shot path with batched decode,
  explicit result persistence, and resume support.

### botsay__qwen25_norm_user_text_icl__twibot20__smoke32

- Status: completed
- Scope: external-reference ablation
- Claim boundary:
  this is a BotSay-compatible ICL ablation aligned to the project's
  `norm_user_text.json` input format. It keeps the BotSay few-shot shell
  (`instruction + exemplars + target + Label:`), but replaces the raw
  `description + metadata` construction with a naturalized evidence card parsed
  from `norm_user_text.json`.
- Code root:
  `/root/workspace/LMbot/botsay_official`
- Dataset:
  `TwiBot-20`
- Command:
  `CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python approach-descandmeta.py -m qwen25 -d Twibot-20 -n 4 --split test --limit 32 --batch_size 4 --max_new_tokens 24 --temperature 0.0 --tweet_limit 3 --output_path /root/workspace/LMbot/botsay_official/probs/descandmeta_norm_user_text_smoke32_v3_qwen25_test_n4_b4.json`
- Output:
  `/root/workspace/LMbot/botsay_official/probs/descandmeta_norm_user_text_smoke32_v3_qwen25_test_n4_b4.json`
- Result:
  - Acc: `0.8125`
  - Macro-F1: `0.7692307692307693`
- Runtime note:
  the evidence card is built from `METADATA / DESCRIPTION / TWEET` spans in
  `norm_user_text.json`, while exemplar retrieval uses a lighter surrogate text
  built from bio plus tweet snippets.

### botsay__qwen25_norm_user_text_icl__twibot20__test

- Status: completed
- Scope: external-reference ablation
- Claim boundary:
  same setting as `botsay__qwen25_norm_user_text_icl__twibot20__smoke32`, but
  evaluated on the full TwiBot-20 test split.
- Code root:
  `/root/workspace/LMbot/botsay_official`
- Dataset:
  `TwiBot-20`
- Command:
  `CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python approach-descandmeta.py -m qwen25 -d Twibot-20 -n 4 --split test --batch_size 4 --max_new_tokens 24 --temperature 0.0 --tweet_limit 3 --output_path /root/workspace/LMbot/botsay_official/probs/descandmeta_norm_user_text_qwen25_test_n4_b4.json`
- Output:
  `/root/workspace/LMbot/botsay_official/probs/descandmeta_norm_user_text_qwen25_test_n4_b4.json`
- Result:
  - Acc: `0.6686390532544378`
  - Macro-F1: `0.6913385826771653`
  - Precision: `0.6968253968253968`
  - Recall: `0.6859375`
- Summary:
  Aligning the BotSay-compatible ICL shell to the project's `norm_user_text.json`
  evidence card yields a large improvement over the earlier practical
  `descandmeta` ICL line (`Acc 0.5883`, `Macro-F1 0.5331`). The stronger result
  comes from both input alignment and more LLM-friendly serialization of
  profile/bio/tweet evidence.

### botsay__qwen25_norm_user_text_peft__twibot20__test

- Status: completed
- Scope: external-reference ablation
- Claim boundary:
  this is a BotSay-compatible Qwen PEFT ablation aligned to
  `norm_user_text.json`. It reuses the same naturalized evidence-card input
  format as `botsay__qwen25_norm_user_text_icl__twibot20__test`, so the final
  comparison is ICL vs PEFT under the same evidence construction rather than
  under two different prompt/input families.
- Code root:
  `/root/workspace/LMbot/botsay_official`
- Dataset:
  `TwiBot-20`
- Approach:
  `descandmeta_norm_user_text`
- Chain:
  1. `approach-finetune.py -d Twibot-20 -a descandmeta -n 4 --tweet 3 --output_name Twibot-20-descandmeta-norm-user-text-instruction-tuning`
  2. `sft.py -i Twibot-20-descandmeta-norm-user-text-instruction-tuning -m qwen25 -e 5`
  3. `approach-finetune_eval.py -d Twibot-20 -a descandmeta --base_model qwen25 --tuned_model_name Twibot-20-descandmeta-norm-user-text-instruction-tuning -n 4 --tweet 3 --split test --batch_size 4 --max_new_tokens 24 --temperature 0.0 --output_path /root/workspace/LMbot/botsay_official/probs/descandmeta_norm_user_text_qwen25_peft_test_n4_b4.json`
  4. `approach-finetune_eval.py -d Twibot-20 -a descandmeta --base_model qwen25 --tuned_model_name Twibot-20-descandmeta-norm-user-text-instruction-tuning -n 4 --tweet 3 --split test --batch_size 2 --max_new_tokens 24 --temperature 0.0 --output_path /root/workspace/LMbot/botsay_official/probs/descandmeta_norm_user_text_qwen25_peft_test_n4_b4.json --resume --allow_offload`
- Output:
  `/root/workspace/LMbot/botsay_official/probs/descandmeta_norm_user_text_qwen25_peft_test_n4_b4.json`
- Local copy:
  [descandmeta_norm_user_text_qwen25_peft_test_n4_b4.json](G:/Research/BotDetection/botsay_official/probs/descandmeta_norm_user_text_qwen25_peft_test_n4_b4.json)
- Adapter root:
  `/root/workspace/LMbot/botsay_official/corpus/Twibot-20-descandmeta-norm-user-text-instruction-tuning/`
- Server logs:
  - `/root/workspace/LMbot/LLMbot/server_logs/botsay_qwen25_descandmeta_norm_user_text_sft.log`
  - `/root/workspace/LMbot/LLMbot/server_logs/botsay_qwen25_descandmeta_norm_user_text_peft_eval_resume.log`
- Result:
  - Acc: `0.775993237531699`
  - Macro-F1: `0.799090219863533`
  - Precision: `0.7761413843888071`
  - Recall: `0.8234375`
  - Evaluated: `1183`
- Runtime note:
  the first eval attempt with `batch_size=4` hit CUDA OOM after writing a
  partial `32`-row output; the completed result above comes from the resumed
  eval run with `batch_size=2` and `--allow_offload`.
- Runtime note:
  the finetune data generator and finetune eval now both reuse the same
  `norm_user_text` evidence-card helpers as the ICL path, so PEFT is trained
  and evaluated on the aligned input surface instead of the original
  `description + metadata` serialization.

### botsay__official_pipeline__qwen25_substitute__twibot20__descandmeta_icl

- Status: superseded by `botsay__qwen25_practical_descandmeta_icl__twibot20__test`
- Scope: external-reference ablation
- Claim boundary:
  this is not an official BotSay base-model reproduction, because the official
  paper/repo defaults use Mistral/LLaMA/OpenAI models. This run keeps the
  official BotSay ICL pipeline and prompt construction, but substitutes the
  base model with the server-local `Qwen2.5-7B-Instruct`.
- Code root:
  `/root/workspace/LMbot/botsay_official`
- Dataset:
  `TwiBot-20`
- Approach:
  `descandmeta`
- Command:
  `CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python approach-descandmeta.py -m qwen25 -d Twibot-20 -n 16`
- Server log:
  `/root/workspace/LMbot/LLMbot/server_logs/botsay_qwen25_descandmeta_icl.log`
- Runtime note:
  the run uses the official BotSay ICL path with a local Qwen alias added in
  `botsay_official/lm_utils.py`.
- Expected output:
  stdout-only metric report in the server log unless later wrapped by a result
  capture script.

### botsay__official_pipeline__qwen25_substitute__twibot20__descandmeta_peft

- Status: superseded by `botsay__qwen25_practical_descandmeta_peft__twibot20__test`
- Scope: external-reference ablation
- Claim boundary:
  this is not an official BotSay base-model reproduction, because the official
  paper/repo defaults use Mistral/LLaMA/OpenAI models. This run keeps the
  official BotSay SFT/PEFT data-generation and evaluation pipeline, but
  substitutes the base model with the server-local `Qwen2.5-7B-Instruct`.
- Code root:
  `/root/workspace/LMbot/botsay_official`
- Dataset:
  `TwiBot-20`
- Approach:
  `descandmeta`
- Chain:
  1. `approach-finetune.py -d Twibot-20 -a descandmeta -n 16`
  2. `sft.py -i Twibot-20-descandmeta-instruction-tuning -m qwen25 -e 5`
  3. `approach-finetune_eval.py -d Twibot-20 -a descandmeta --base_model qwen25 --tuned_model_name Twibot-20-descandmeta-instruction-tuning -n 16`
- Server log:
  `/root/workspace/LMbot/LLMbot/server_logs/botsay_qwen25_descandmeta_peft.log`
- Runtime note:
  to avoid memory contention on the server, this chain is queued to start after
  the running `descandmeta` ICL job exits.

### botsay__qwen25_structure_nolabel__twibot20__routed_test

- Status: pending server smoke / full routed-test run
- Scope: external-reference ablation
- Claim boundary:
  this is a fair BotSay-style structure prompt ablation, not the official
  BotSay structure setting. It preserves the direction-split graph prompt
  layout (`followers` block, `followings` block, `Target user`, `Label:`),
  but removes neighbor `Label:` lines to avoid test-time label leakage.
- Code root:
  `/root/workspace/LMbot/botsay_official`
- Dataset:
  `Twibot-20`
- Model:
  `Qwen2.5-7B-Instruct`
- Prompt variant:
  `approach-structure.py -t random --neighbor_label_mode none`
- Routed split:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
- Planned smoke command:
  `CUDA_VISIBLE_DEVICES=1 /root/mambaforge/envs/Qwen/bin/python approach-structure.py -m qwen25 -d Twibot-20 -t random --neighbor_label_mode none --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json --routed_split test --limit 32 --batch_size 4 --max_new_tokens 32 --temperature 0.0 --output_path /root/workspace/LMbot/botsay_official/probs/structure_routed_test_qwen25_random_nolabel_smoke32.json --allow_offload`
- Planned full command:
  `CUDA_VISIBLE_DEVICES=1 /root/mambaforge/envs/Qwen/bin/python -u approach-structure.py -m qwen25 -d Twibot-20 -t random --neighbor_label_mode none --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json --routed_split test --batch_size 4 --max_new_tokens 32 --temperature 0.0 --output_path /root/workspace/LMbot/botsay_official/probs/structure_routed_test_qwen25_random_nolabel.json --allow_offload`
- Planned output:
  `/root/workspace/LMbot/botsay_official/probs/structure_routed_test_qwen25_random_nolabel.json`
- Planned log:
  `/root/workspace/LMbot/LLMbot/server_logs/botsay_structure_routed_test_qwen25_random_nolabel.log`
- Comparison note:
  compare against the label-informed reference run
  `/root/workspace/LMbot/botsay_official/probs/structure_routed_test_qwen25_random.json`
  and keep the leakage caveat explicit in any result table.

### mpe__botsay_precompute_smoke_preview32__seed1

- Status: pending local sync / server launch
- Scope: precompute smoke only
- Claim boundary:
  prompt-generation smoke for BotSay-style explanation-first experts on a
  32-node routed test preview; not a downstream classifier or refiner result
- Dataset: TwiBot-20
- Seed: `1`
- Routed preview nodes:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_test_nodes_seed1_smoke_preview.json`
- Explain model:
  `/root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct`
- Embedding model:
  `/root/workspace/LMbot/hf_models/roberta_finetuned_simteg_twibot20_seed1`
- Expected checkpoint:
  `/root/workspace/LMbot/TwiBot-20_seed_1/checkpoints/LM_pretrain/best.pkl`
- Output cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_botsay_roberta_finetuned_routed_test_preview32_seed1.pt`
- Explanation sidecar dir:
  `/root/workspace/LMbot/datasets/TwiBot-20/prompt_expert_botsay_smoke_preview32_seed1`
- Manifest:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_botsay_roberta_finetuned_routed_test_preview32_seed1_manifest.json`
- Prompt sidecar:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_botsay_roberta_finetuned_routed_test_preview32_seed1_prompts.jsonl`
- Server log:
  `/root/workspace/LMbot/LLMbot/server_logs/mpe__botsay_precompute_smoke_preview32__seed1.log`
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u precompute.py \
  --dataset TwiBot-20 \
  --prompt_mode expert_concat_v1 \
  --prompt_family_version v2 \
  --explain_prompt_style botsay \
  --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_test_nodes_seed1_smoke_preview.json \
  --explain_model_path /root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct \
  --model_path /root/workspace/LMbot/hf_models/roberta_finetuned_simteg_twibot20_seed1 \
  --explain_required \
  --explain_batch_size 2 \
  --explain_log_every 8 \
  --device cuda \
  --output_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_botsay_roberta_finetuned_routed_test_preview32_seed1.pt \
  --explain_component_cache_dir /root/workspace/LMbot/datasets/TwiBot-20/prompt_expert_botsay_smoke_preview32_seed1 \
  --overwrite \
  --disable_wandb
```

### mpe__botsay_testonly_feature_swap__highbase__seed1

- Status: completed
- Scope: BotSay-style explanation-first prompt experts on the full routed
  test-only slice (`296` nodes), followed by frozen-GNN feature-replacement
  replay
- Claim boundary:
  diagnostic test-only evaluation only; this run does not train a routed-node
  classifier or refiner and does not support a claim about learned correction
  policies
- Baseline: canonical high-base frozen SimTeG seed 1
  - Acc: `0.863905325443787`
  - Macro-F1: `0.862442357206568`
- Routed test-only node file:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529_test_only.json`
- BotSay prompt-expert cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_botsay_roberta_finetuned_routed_highbase_preiter_seed1_test_only.pt`
- BotSay prompt-expert manifest:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_botsay_roberta_finetuned_routed_highbase_preiter_seed1_test_only_manifest.json`
- Explanation sidecar dir:
  `/root/workspace/LMbot/datasets/TwiBot-20/prompt_expert_botsay_testonly_seed1`
- Downstream artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__botsay_testonly_feature_swap__highbase__seed1/seed_1`
- Downstream artifacts:
  - manifest:
    `/root/workspace/LMbot/LLMbot/experiments/mpe__botsay_testonly_feature_swap__highbase__seed1/seed_1/manifest.json`
  - metrics:
    `/root/workspace/LMbot/LLMbot/experiments/mpe__botsay_testonly_feature_swap__highbase__seed1/seed_1/metrics.json`
  - outputs:
    `/root/workspace/LMbot/LLMbot/experiments/mpe__botsay_testonly_feature_swap__highbase__seed1/seed_1/outputs.pt`
  - per-node:
    `/root/workspace/LMbot/LLMbot/experiments/mpe__botsay_testonly_feature_swap__highbase__seed1/seed_1/per_node_test.jsonl`
- Implementation note:
  the first full test-only BotSay precompute attempt failed on `conflict`
  prompt-echo quality gating. The active fix was:
  1. widen BotSay postprocess stop markers for prompt-restart text
  2. preserve the BotSay `prompt_role` for `conflict` generation rows so the
     BotSay-specific postprocess actually runs on that expert
  After the fix, the resumed run completed all `296` `conflict` explanations.
- Result:
  - Base test Acc / Macro-F1:
    `0.863905325443787 / 0.862442357206568`
  - Swapped test Acc / Macro-F1:
    `0.8131868131868132 / 0.8131846773749833`
  - Delta Acc / Macro-F1:
    `-0.05071851225697377 / -0.04925767983158469`
  - Net gain: `-60`
  - Improved / degraded: `55 / 115`
  - Routed fix / break: `55 / 114`
  - Wrong-node fix rate / correct-node break rate:
    `0.3416149068322981 / 0.11252446183953033`
  - Routed test base Acc / Macro-F1:
    `0.6385135135135135 / 0.6291638858641564`
  - Routed test swapped Acc / Macro-F1:
    `0.4391891891891892 / 0.397340921356032`
  - Non-routed test preservation Acc:
    `0.9379932356257046` after swap vs base `0.939120631341601`
- Summary:
  On the full routed test-only slice, replacing routed-node features with
  BotSay-style explanation embeddings encoded by the finetuned SimTeG RoBERTa
  significantly harms the strong frozen SimTeG baseline. The representation has
  some wrong-node recovery ability (`55` fixes) but breaks far more correct
  routed nodes (`114`), so this diagnostic does not support direct all-change
  consumption of the BotSay explanation features inside the frozen backbone.

### mpe__botsay_testonly_feature_swap_roberta_base__highbase__seed1

- Status: planned for server launch
- Scope: BotSay-style explanation-first prompt experts on the same routed
  test-only slice (`296` nodes), but encoded with raw pretrained
  `roberta-base` instead of the finetuned SimTeG RoBERTa line
- Claim boundary:
  diagnostic test-only evaluation only; this run reuses the exact same BotSay
  explanation sidecars as `mpe__botsay_testonly_feature_swap__highbase__seed1`
  and changes only the explanation encoder in the frozen-GNN feature-swap
  replay
- Baseline: canonical high-base frozen SimTeG seed 1
  - Acc: `0.863905325443787`
  - Macro-F1: `0.862442357206568`
- Routed test-only node file:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529_test_only.json`
- Reused BotSay explanation sidecar dir:
  `/root/workspace/LMbot/datasets/TwiBot-20/prompt_expert_botsay_testonly_seed1`
- Reused explanation stem:
  `glance_prompt_expert_concat_v2_botsay_roberta_finetuned_routed_highbase_preiter_seed1_test_only`
- Encoder under test:
  `/root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b`
- Downstream artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__botsay_testonly_feature_swap_roberta_base__highbase__seed1/seed_1`
- Planned server log:
  `/root/workspace/LMbot/LLMbot/server_logs/mpe__botsay_testonly_feature_swap_roberta_base__highbase__seed1.log`
- Planned command:

```bash
cd /root/workspace/LMbot/LLMbot
/root/mambaforge/envs/lmbot/bin/python scripts/routed_explain_qwen3_embedding_mlp.py \
  --diagnostic_mode frozen_gnn_feature_swap \
  --output_dir /root/workspace/LMbot/LLMbot/experiments/mpe__botsay_testonly_feature_swap_roberta_base__highbase__seed1/seed_1 \
  --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529_test_only.json \
  --base_outputs_path /root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_classifier_qwen3_20260601/base_outputs_for_routed_classifier.pt \
  --dataset_root /root/workspace/LMbot/datasets/TwiBot-20 \
  --labels_path /root/workspace/LMbot/datasets/TwiBot-20/labels.pt \
  --train_idx_path /root/workspace/LMbot/datasets/TwiBot-20/train_idx.pt \
  --valid_idx_path /root/workspace/LMbot/datasets/TwiBot-20/valid_idx.pt \
  --test_idx_path /root/workspace/LMbot/datasets/TwiBot-20/test_idx.pt \
  --explanation_dir /root/workspace/LMbot/datasets/TwiBot-20/prompt_expert_botsay_testonly_seed1 \
  --explanation_stem glance_prompt_expert_concat_v2_botsay_roberta_finetuned_routed_highbase_preiter_seed1_test_only \
  --roberta_model_path /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --device cuda:0 \
  --batch_size 64 \
  --roberta_max_length 512 \
  --component_char_budget 2500 \
  --seed 1
```

- Success criteria:
  produce a directly comparable raw `roberta-base` feature-swap result against
  the existing finetuned-RoBERTa BotSay run, with `manifest.json`,
  `metrics.json`, `outputs.pt`, and `per_node_test.jsonl` under the stable
  artifact root.

### mpe__prompt_expert_quality_audit__clean_v2_highbase__seed1

- Status: completed
- Scope: read-only prompt-expert quality and separability audit
- Claim boundary:
  diagnostic stage only; no prompt-cache rebuild, no explanation regeneration,
  no router training, and no refiner training
- Baseline: high-base frozen SimTeG seed 1
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__prompt_expert_quality_audit__clean_v2_highbase__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__prompt_expert_quality_audit__clean_v2_highbase__seed1/seed_1/stages/prompt_expert_quality_audit`
- Reused routed cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
- Reused adjacent manifest:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531_manifest.json`
- Reused reference stage:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__conflict_aware_correction_moe__highbase__seed1/seed_1/stages/joint_router_refinement`
- Manifest:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__prompt_expert_quality_audit__clean_v2_highbase__seed1/seed_1/stages/prompt_expert_quality_audit/manifest.json`
- Metrics:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__prompt_expert_quality_audit__clean_v2_highbase__seed1/seed_1/stages/prompt_expert_quality_audit/metrics.json`
- Quality gate:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__prompt_expert_quality_audit__clean_v2_highbase__seed1/seed_1/stages/prompt_expert_quality_audit/quality_gate.json`
- Probe artifacts:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__prompt_expert_quality_audit__clean_v2_highbase__seed1/seed_1/stages/prompt_expert_quality_audit/base_wrong_probe.json`,
  `/root/workspace/LMbot/LLMbot/experiments/mpe__prompt_expert_quality_audit__clean_v2_highbase__seed1/seed_1/stages/prompt_expert_quality_audit/utility_probe.json`
- Audit inputs:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__prompt_expert_quality_audit__clean_v2_highbase__seed1/seed_1/stages/prompt_expert_quality_audit/audit_inputs.pt`
- Routed split summary:
  - train: `600` routed, `217` base-wrong
  - valid: `473` routed, `173` base-wrong
  - test: `237` routed, `92` base-wrong
- Quality results:
  - all four sidecars cover `1640 / 1640` routed-union nodes
  - `empty=0`, `prompt_echo=0`, `many_exclamation_marks=0` for all four experts
  - `graph_following` had `30` duplicate rows resolved by last-row-wins
- Separability results:
  - expert identity probe test accuracy: `0.9968354430379747`
  - best base-wrong test AUC: `0.5107946026986506` (`conflict`)
  - best base-wrong valid-locked test net: `0`
  - best utility test AUC: `0.5720439691027926` (`graph_following`)
  - best utility valid-locked test net: `0`
- Interpretation:
  The clean explanations pass text-quality checks and the expert embeddings are
  distinguishable by component identity, but they do not provide a stable
  routed-node wrong/correct or action-utility boundary under the current
  `explain -> finetuned RoBERTa embedding` representation. This supports
  treating prompt-expert explanations as usable diagnostic material, not yet as
  sufficient correction evidence for a learned keep/change selector.
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task prompt_expert_quality_audit \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --prompt_expert_quality_cache_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt \
  --prompt_expert_quality_reference_stage /root/workspace/LMbot/LLMbot/experiments/mpe__conflict_aware_correction_moe__highbase__seed1/seed_1/stages/joint_router_refinement \
  --prompt_expert_quality_components graph_following,graph_follower,tweet,conflict \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/mpe__prompt_expert_quality_audit__clean_v2_highbase__seed1 \
  --seeds 1 \
  --device 0 \
  --disable_wandb
```

### mpe__gaugllm_mope_similarity__highbase__seed1

- Status: completed
- Scope: strict GAugLLM MoPE selector-head migration over the existing routed
  prompt-expert cache; no prompt-cache rebuild and no explanation regeneration
- Claim boundary:
  selector-head transplant only; backbone, router protocol, routed-node
  definition, and prompt cache are unchanged
- Baseline: high-base frozen SimTeG seed 1
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_similarity__highbase__seed1`
- Reused routed cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
- Reused adjacent manifest:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531_manifest.json`
- Reused explanation sidecars:
  `graph_following`, `graph_follower`, `tweet`, `conflict`
- Frozen router reuse root:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement`
- Prompt fusion:
  `gaugllm_mope`
- MoPE attention:
  `similarity` (official GAugLLM `SimilarityAttentionMLP` default,
  temperature `0.2`)
- Refiner input:
  `[z_gnn || gaugllm_mope_similarity(graph_following, graph_follower, tweet, conflict) || metadata_structured_proj || structural_side_channel]`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_similarity__highbase__seed1/seed_1/stages/joint_router_refinement`
- Manifest:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_similarity__highbase__seed1/seed_1/stages/joint_router_refinement/manifest.json`
- Metrics:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_similarity__highbase__seed1/seed_1/stages/joint_router_refinement/metrics.json`
- Analysis:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_similarity__highbase__seed1/seed_1/stages/joint_router_refinement/analysis_summary.json`
- Outputs:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_similarity__highbase__seed1/seed_1/stages/joint_router_refinement/outputs.pt`
- Checkpoint:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_similarity__highbase__seed1/seed_1/stages/joint_router_refinement/checkpoint.pt`
- Per-node test:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_similarity__highbase__seed1/seed_1/stages/joint_router_refinement/per_node_test.jsonl`
- Metrics:
  - Base Acc / Macro-F1: `0.863905325443787 / 0.862442357206568`
  - Final Acc / Macro-F1: `0.8672865595942519 / 0.8659068117644426`
  - Delta Acc / Macro-F1: `0.0033812341504649845 / 0.0034644545578745856`
  - Fixed / broken / net gain: `5 / 1 / 4`
  - Routed wrong precision: `0.3755274261603376`
  - Conditional fix / break on routed nodes: `0.056179775280898875 / 0.006756756756756757`
  - Wrong-node fix rate / correct-node break rate: `0.031055900621118012 / 0.0009784735812133072`
- Selector behavior:
  - all `237` routed test nodes still select `tweet`
  - mean routed weights are approximately:
    `graph_following=0.0000006`, `graph_follower=0.0000861`,
    `tweet=0.9999133`, `conflict≈0`
- Summary:
  Strict GAugLLM similarity MoPE improves over the earlier custom
  `gaugllm_selector` by reducing break (`9 -> 1`) and moving net gain from `0`
  to `+4`, but it still underperforms the best no-projector
  `raw_concat_follower_triplet` anchor (`+6`) and keeps the same tweet-only
  selected-expert collapse.
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --external_frozen_g0_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_roberta_20260527/seed_1 \
  --joint_refiner_embedding_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt \
  --joint_routing_protocol frozen_router_reuse \
  --joint_router_reuse_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement \
  --joint_prompt_expert_fusion gaugllm_mope \
  --joint_prompt_expert_mope_attention similarity \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_similarity__highbase__seed1 \
  --seeds 1 \
  --disable_wandb
```

### mpe__clean_v2_roberta_cache_classifier__highbase__seed1

- Status: completed
- Scope: diagnostic routed-node classifier
- Baseline: high-base frozen SimTeG seed 1
- Artifact root:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_classifier_roberta_cache_20260601/seed_1`
- Manifest:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_classifier_roberta_cache_20260601/seed_1/manifest.json`
- Metrics:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_classifier_roberta_cache_20260601/seed_1/metrics.json`
- Outputs:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_classifier_roberta_cache_20260601/seed_1/outputs.pt`
- Summary:
  clean v2 explanation evidence had correction ability, but direct
  routed-node classifier all-change degraded the high-base line; confidence
  abstain selected almost no test nodes.

### mpe__joint_refiner_reuse_clean_v2__fullgraph_reuse__seed1

- Status: completed
- Scope: diagnostic full-graph refiner reuse
- Baseline: full-graph reuse base, not the high-base `0.8639 / 0.8624` line
- Artifact root:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_fullgraph_reuse_mpe_gate_20260601/seed_1/stages/joint_router_refinement`
- Manifest:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_fullgraph_reuse_mpe_gate_20260601/seed_1/stages/joint_router_refinement/manifest.json`
- Metrics:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_fullgraph_reuse_mpe_gate_20260601/seed_1/stages/joint_router_refinement/metrics.json`
- Outputs:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_fullgraph_reuse_mpe_gate_20260601/seed_1/stages/joint_router_refinement/outputs.pt`
- Summary:
  mild positive net gain under a different full-graph base; not directly
  comparable to the fixed high-base SimTeG baseline.

### mpe__roberta_base_feature_swap_gnn__highbase__seed1

- Status: completed
- Scope: diagnostic frozen-GNN replay with routed-node feature replacement
- Baseline: high-base frozen SimTeG seed 1
- Artifact root:
  `/root/workspace/LMbot/LLMbot/server_routed_explain_roberta_base_feature_swap_highbase_20260601/seed_1`
- Manifest:
  `/root/workspace/LMbot/LLMbot/server_routed_explain_roberta_base_feature_swap_highbase_20260601/seed_1/manifest.json`
- Metrics:
  `/root/workspace/LMbot/LLMbot/server_routed_explain_roberta_base_feature_swap_highbase_20260601/seed_1/metrics.json`
- Outputs:
  `/root/workspace/LMbot/LLMbot/server_routed_explain_roberta_base_feature_swap_highbase_20260601/seed_1/outputs.pt`
- Per-node test:
  `/root/workspace/LMbot/LLMbot/server_routed_explain_roberta_base_feature_swap_highbase_20260601/seed_1/per_node_test.jsonl`
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
/root/mambaforge/envs/lmbot/bin/python scripts/routed_explain_qwen3_embedding_mlp.py \
  --diagnostic_mode frozen_gnn_feature_swap \
  --output_dir /root/workspace/LMbot/LLMbot/server_routed_explain_roberta_base_feature_swap_highbase_20260601/seed_1 \
  --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json \
  --base_outputs_path /root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_classifier_qwen3_20260601/base_outputs_for_routed_classifier.pt \
  --dataset_root /root/workspace/LMbot/datasets/TwiBot-20 \
  --labels_path /root/workspace/LMbot/datasets/TwiBot-20/labels.pt \
  --train_idx_path /root/workspace/LMbot/datasets/TwiBot-20/train_idx.pt \
  --valid_idx_path /root/workspace/LMbot/datasets/TwiBot-20/valid_idx.pt \
  --test_idx_path /root/workspace/LMbot/datasets/TwiBot-20/test_idx.pt \
  --explanation_dir /root/workspace/LMbot/datasets/TwiBot-20/prompt_expert_v2_clean_routed_preiter_cache_20260531 \
  --explanation_stem glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531 \
  --roberta_model_path /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --device cuda:0 \
  --batch_size 64 \
  --roberta_max_length 512 \
  --component_char_budget 2500 \
  --seed 1
```

- Metrics:
  - Base Acc / Macro-F1: `0.863905325443787 / 0.862442357206568`
  - Swapped Acc / Macro-F1: `0.8174133558748944 / 0.8174132254081492`
  - Delta Acc / Macro-F1: `-0.046491969568892566 / -0.04502913179841883`
  - Net gain: `-55`
  - Improved / degraded: `51 / 106`
  - Wrong-node fix rate: `0.3167701863354037`
  - Correct-node break rate: `0.10371819960861056`
- Summary:
  replacing routed-node features with pretrained RoBERTa embeddings of clean
  explanations harms the high-base frozen SimTeG line under frozen-GNN replay.

### semantic_encoder__raw_roberta_base_gnn__seed1

- Status: completed
- Scope: full TwiBot-20 graph detector comparison
- Question:
  replace the frozen SimTeG semantic encoder output with raw pretrained
  `roberta-base` node embeddings, then train/evaluate the same RGCN graph
  detector on the canonical split.
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_encoder__raw_roberta_base_gnn__seed1`
- Raw embedding:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_encoder__raw_roberta_base_gnn__seed1/raw_roberta_base_embeddings.pt`
- Raw embedding manifest:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_encoder__raw_roberta_base_gnn__seed1/raw_roberta_base_embedding_manifest.json`
- Graph-detector manifest:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_encoder__raw_roberta_base_gnn__seed1/seed_1/preparation/graph_detector/manifest.json`
- Graph-detector selection metrics:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_encoder__raw_roberta_base_gnn__seed1/seed_1/preparation/graph_detector/selection_metrics.json`
- Graph-detector test metrics:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_encoder__raw_roberta_base_gnn__seed1/seed_1/preparation/graph_detector/test_metrics.json`
- Outputs:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_encoder__raw_roberta_base_gnn__seed1/seed_1/preparation/graph_detector/outputs.pt`
- Router/refiner stage:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_encoder__raw_roberta_base_gnn__seed1/seed_1/stages/joint_router_refinement`
- Router/refiner manifest:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_encoder__raw_roberta_base_gnn__seed1/seed_1/stages/joint_router_refinement/manifest.json`
- Router/refiner metrics:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_encoder__raw_roberta_base_gnn__seed1/seed_1/stages/joint_router_refinement/metrics.json`
- Router performance summary:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_encoder__raw_roberta_base_gnn__seed1/seed_1/stages/joint_router_refinement/router_performance_summary.json`
- Embedding precompute command:

```bash
cd /root/workspace/LMbot/LLMbot
/root/mambaforge/envs/lmbot/bin/python /tmp/precompute_raw_roberta_base.py
```

- Graph-detector command:

```bash
cd /root/workspace/LMbot/LLMbot
/root/mambaforge/envs/lmbot/bin/python main.py \
  --experiment_task graph_detector_prepare \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path /root/workspace/LMbot/LLMbot/experiments/semantic_encoder__raw_roberta_base_gnn__seed1/raw_roberta_base_embeddings.pt \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/semantic_encoder__raw_roberta_base_gnn__seed1 \
  --graph_detector_epochs 200 \
  --seeds 1 \
  --disable_wandb \
  --force_retrain_backbone
```

- Router/refiner command:

```bash
cd /root/workspace/LMbot/LLMbot
/root/mambaforge/envs/lmbot/bin/python main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path /root/workspace/LMbot/LLMbot/experiments/semantic_encoder__raw_roberta_base_gnn__seed1/raw_roberta_base_embeddings.pt \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/semantic_encoder__raw_roberta_base_gnn__seed1 \
  --risk_budgets 0.05,0.10,0.15,0.20,0.25 \
  --seeds 1 \
  --disable_wandb \
  --reuse_existing_artifacts
```

- Encoding contract:
  - encoder: raw pretrained `roberta-base`
  - finetuned: `false`
  - text: `datasets/TwiBot-20/norm_user_text.json`
  - max length: `512`
  - add special tokens: `false`
  - pooling: final hidden-state mean pooling
  - output shape: `[11826, 768]`
- Metrics:
  - Valid Acc / Macro-F1: `0.7852008456659619 / 0.7828706715824478`
  - Test Acc / Macro-F1 / Bot-F1:
    `0.7633136094674556 / 0.7625056641868522 / 0.7763578274760383`
  - Compared with high-base frozen SimTeG test:
    - Acc delta: `-0.10059171597633135`
    - Macro-F1 delta: `-0.09993669301971584`
- Router/refiner metrics:
  - Selected budget / beta: `0.15 / 0.2`
  - Router test AUROC / AUPRC for base-wrong: `0.7311026736275905 / 0.408365767407105`
  - Routed test count: `296`
  - Routed wrong precision / coverage:
    `0.40540540540540543 / 0.42857142857142855`
  - Conditional fix rate on selected wrong: `0.075`
  - Overall Acc / Macro-F1 after refiner:
    `0.757396449704142 / 0.755854983234495`
  - Delta Macro-F1 vs raw-RoBERTa base: `-0.006650680952357213`
  - Net gain: `-7`
  - Improved / degraded: `9 / 16`
  - Wrong-node fix rate / correct-node break rate:
    `0.03214285714285714 / 0.017718715393133997`
- Summary:
  raw pretrained `roberta-base` embeddings are much weaker than the high-base
  SimTeG finetuned semantic encoder output for the RGCN graph detector.
  Under this weaker base, the router can identify wrong nodes better than
  chance, but the joint refiner still has negative net gain.

## Pending / Planned

Record future runs here before launch. Prefer stable artifact roots under:

`/root/workspace/LMbot/LLMbot/experiments/<experiment_id>`

### mpe__utility_gate_only__highbase__seed1

- Status: completed
- Scope: GLANCE-style utility-advantage keep/change gate over the existing
  high-base routed prompt-expert cache
- Claim boundary:
  refiner-only utility gate; no prompt-cache rebuild, no explanation
  regeneration, no backbone change, and no router retraining
- Baseline: high-base frozen SimTeG seed 1
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__utility_gate_only__highbase__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__utility_gate_only__highbase__seed1/seed_1/stages/joint_router_refinement`
- Reused routed cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
- Reused adjacent manifest:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531_manifest.json`
- Frozen router reuse root:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement`
- Prompt fusion:
  `raw_concat_follower_triplet`
- Gate protocol:
  `--joint_refiner_explicit_gate --joint_refiner_target_mode keep_change --joint_refiner_gate_target utility_positive --joint_refiner_weight_mode utility_positive --joint_refiner_gate_policy hard_keep_change`
- Refiner input:
  `[z_gnn || graph_follower || tweet || conflict || structural_side_channel]`
- Expected artifacts:
  `manifest.json`, `metrics.json`, `analysis_summary.json`, `outputs.pt`,
  `checkpoint.pt`, `per_node_test.jsonl`
- Metrics:
  - Base Acc / Macro-F1: `0.863905325443787 / 0.862442357206568`
  - Final Acc / Macro-F1: `0.8571428571428571 / 0.8559422117341884`
  - Delta Acc / Macro-F1: `-0.006762468300929858 / -0.006500145472379648`
  - Routed test count / wrong precision: `237 / 0.3881856540084388`
  - Fixed / broken / net gain: `3 / 11 / -8`
  - Wrong-node fix rate / correct-node break rate:
    `0.018633540372670808 / 0.010763209393346379`
  - Conditional fix / break on routed wrong/correct:
    `0.03260869565217391 / 0.07586206896551724`
  - Gate positive rate / precision / recall:
    `0.10548523206751055 / 0.36 / 0.09782608695652174`
  - Mean routed gate probability / utility-positive rate:
    `0.48883492446146937 / 0.3881856540084388`
- Summary:
  the GLANCE-style utility-positive gate did not preserve the
  `raw_concat_follower_triplet` anchor. It reduced how often the refiner changes
  routed nodes, but it also kept too many fixable wrong nodes unchanged, yielding
  negative net gain.

### mpe__botmoe_selector__highbase__seed1

- Status: completed
- Scope: BotMoE-style sparse selector transplant over the existing high-base
  routed prompt-expert cache
- Claim boundary:
  selector transplant only; no prompt-cache rebuild, no explanation
  regeneration, no backbone change, and no router retraining
- Baseline: high-base frozen SimTeG seed 1
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__botmoe_selector__highbase__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__botmoe_selector__highbase__seed1/seed_1/stages/joint_router_refinement`
- Reused routed cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
- Reused adjacent manifest:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531_manifest.json`
- Frozen router reuse root:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement`
- Prompt fusion:
  `botmoe_selector`
- BotMoE selector protocol:
  `--joint_prompt_expert_botmoe_top_k 1 --joint_prompt_expert_botmoe_noisy_gating --joint_prompt_expert_botmoe_aux_weight 0.01`
- Selectable experts:
  `graph_following`, `graph_follower`, `tweet`, `conflict`
- Non-selectable side channels:
  `metadata_structured`, structural side features, and base/abstain
- Expected artifacts:
  `manifest.json`, `metrics.json`, `analysis_summary.json`, `outputs.pt`,
  `checkpoint.pt`, `per_node_test.jsonl`
- Metrics:
  - Base Acc / Macro-F1: `0.863905325443787 / 0.862442357206568`
  - Final Acc / Macro-F1: `0.852916314454776 / 0.8524979935794543`
  - Delta Acc / Macro-F1: `-0.01098901098901095 / -0.009944363627113706`
  - Routed test count / wrong precision: `237 / 0.3881856540084388`
  - Fixed / broken / net gain: `23 / 36 / -13`
  - Wrong-node fix rate / correct-node break rate:
    `0.14285714285714285 / 0.03522504892367906`
  - Conditional fix / break on routed wrong/correct:
    `0.25 / 0.2482758620689655`
  - Routed selected-expert distribution:
    `conflict=219`, `graph_follower=17`, `tweet=1`, `graph_following=0`
- Summary:
  the sparse BotMoE-style selector increases the number of corrected wrong
  nodes, but it also breaks far more correct nodes. The result confirms that
  selection without an effective keep/change policy is not sufficient under the
  high-base frozen SimTeG setting.

### mpe__utility_gate_botmoe_selector__highbase__seed1

- Status: completed
- Scope: combined GLANCE utility-positive keep/change gate plus BotMoE-style
  sparse expert selector on the same high-base routed prompt-expert cache
- Claim boundary:
  refiner-only gate/selector experiment; no prompt-cache rebuild, no explanation
  regeneration, no backbone change, and no router retraining
- Baseline: high-base frozen SimTeG seed 1
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__utility_gate_botmoe_selector__highbase__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__utility_gate_botmoe_selector__highbase__seed1/seed_1/stages/joint_router_refinement`
- Reused routed cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
- Reused adjacent manifest:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531_manifest.json`
- Frozen router reuse root:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement`
- Prompt fusion:
  `botmoe_selector`
- Gate protocol:
  `--joint_refiner_explicit_gate --joint_refiner_target_mode keep_change --joint_refiner_gate_target utility_positive --joint_refiner_weight_mode utility_positive --joint_refiner_gate_policy hard_keep_change`
- BotMoE selector protocol:
  `--joint_prompt_expert_botmoe_top_k 1 --joint_prompt_expert_botmoe_noisy_gating --joint_prompt_expert_botmoe_aux_weight 0.01`
- Expected artifacts:
  `manifest.json`, `metrics.json`, `analysis_summary.json`, `outputs.pt`,
  `checkpoint.pt`, `per_node_test.jsonl`
- Metrics:
  - Base Acc / Macro-F1: `0.863905325443787 / 0.862442357206568`
  - Final Acc / Macro-F1: `0.8571428571428571 / 0.8559866249163517`
  - Delta Acc / Macro-F1: `-0.006762468300929858 / -0.006455732290216343`
  - Routed test count / wrong precision: `237 / 0.3881856540084388`
  - Fixed / broken / net gain: `4 / 12 / -8`
  - Wrong-node fix rate / correct-node break rate:
    `0.024844720496894408 / 0.011741682974559686`
  - Conditional fix / break on routed wrong/correct:
    `0.043478260869565216 / 0.08275862068965517`
  - Gate positive rate / precision / recall:
    `0.2911392405063291 / 0.3333333333333333 / 0.25`
  - Mean routed gate probability / utility-positive rate:
    `0.5001939989091978 / 0.3881856540084388`
  - Routed selected-expert distribution:
    `tweet=177`, `conflict=48`, `graph_follower=10`, `graph_following=2`
- Summary:
  adding the GLANCE-style utility gate to the sparse BotMoE selector reduces the
  break count relative to selector-only, but it also suppresses most fixes. The
  combined path remains below the high-base SimTeG baseline and below the
  `raw_concat_follower_triplet` anchor.

### mpe__raw_concat_following_triplet__highbase__seed1

- Status: completed
- Scope: fixed-router comparable routed-node refiner ablation
- Baseline: high-base frozen SimTeG seed 1
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_following_triplet__highbase__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_following_triplet__highbase__seed1/seed_1/stages/joint_router_refinement`
- Manifest:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_following_triplet__highbase__seed1/seed_1/stages/joint_router_refinement/manifest.json`
- Metrics:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_following_triplet__highbase__seed1/seed_1/stages/joint_router_refinement/metrics.json`
- Analysis:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_following_triplet__highbase__seed1/seed_1/stages/joint_router_refinement/analysis_summary.json`
- Outputs:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_following_triplet__highbase__seed1/seed_1/stages/joint_router_refinement/outputs.pt`
- Checkpoint:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_following_triplet__highbase__seed1/seed_1/stages/joint_router_refinement/checkpoint.pt`
- Per-node test:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_following_triplet__highbase__seed1/seed_1/stages/joint_router_refinement/per_node_test.jsonl`
- Routed cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
- Routed nodes:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
- Frozen router reuse root:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement`
- Prompt fusion:
  `raw_concat_following_triplet`
- Refiner input:
  `[z_gnn || graph_following || tweet || conflict || structural_side_channel]`
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
/root/mambaforge/envs/lmbot/bin/python main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --external_frozen_g0_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_roberta_20260527/seed_1 \
  --joint_refiner_embedding_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt \
  --joint_routing_protocol frozen_router_reuse \
  --joint_router_reuse_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement \
  --joint_prompt_expert_fusion raw_concat_following_triplet \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_following_triplet__highbase__seed1 \
  --seeds 1 \
  --disable_wandb
```

- Metrics:
  - Base Acc / Macro-F1: `0.863905325443787 / 0.862442357206568`
  - Final Acc / Macro-F1: `0.8579881656804734 / 0.8569034041817868`
  - Delta Acc / Macro-F1: `-0.005917159763313529 / -0.005538953024781246`
  - Routed test count / wrong precision: `237 / 0.3755274261603376`
  - Routed wrong coverage: `0.5527950310559007`
  - Fixed / broken / net gain: `8 / 15 / -7`
  - Wrong-node fix rate: `0.049689440993788817`
  - Correct-node break rate: `0.014677103718199608`
  - Conditional fix / break on routed wrong/correct:
    `0.0898876404494382 / 0.10135135135135136`
  - Selected budget / beta: `0.2 / 0.2`
- Summary:
  Direct raw concat without projectors or `graph_fused` is negative when the
  graph expert is `graph_following`; break exceeds fix on the fixed routed set.

### mpe__raw_concat_follower_triplet__highbase__seed1

- Status: completed
- Scope: fixed-router comparable routed-node refiner ablation
- Baseline: high-base frozen SimTeG seed 1
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_follower_triplet__highbase__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_follower_triplet__highbase__seed1/seed_1/stages/joint_router_refinement`
- Manifest:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_follower_triplet__highbase__seed1/seed_1/stages/joint_router_refinement/manifest.json`
- Metrics:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_follower_triplet__highbase__seed1/seed_1/stages/joint_router_refinement/metrics.json`
- Analysis:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_follower_triplet__highbase__seed1/seed_1/stages/joint_router_refinement/analysis_summary.json`
- Outputs:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_follower_triplet__highbase__seed1/seed_1/stages/joint_router_refinement/outputs.pt`
- Checkpoint:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_follower_triplet__highbase__seed1/seed_1/stages/joint_router_refinement/checkpoint.pt`
- Per-node test:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_follower_triplet__highbase__seed1/seed_1/stages/joint_router_refinement/per_node_test.jsonl`
- Routed cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
- Routed nodes:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
- Frozen router reuse root:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement`
- Prompt fusion:
  `raw_concat_follower_triplet`
- Refiner input:
  `[z_gnn || graph_follower || tweet || conflict || structural_side_channel]`
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
/root/mambaforge/envs/lmbot/bin/python main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --external_frozen_g0_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_roberta_20260527/seed_1 \
  --joint_refiner_embedding_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt \
  --joint_routing_protocol frozen_router_reuse \
  --joint_router_reuse_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement \
  --joint_prompt_expert_fusion raw_concat_follower_triplet \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_follower_triplet__highbase__seed1 \
  --seeds 1 \
  --disable_wandb
```

- Metrics:
  - Base Acc / Macro-F1: `0.863905325443787 / 0.862442357206568`
  - Final Acc / Macro-F1: `0.8689771766694844 / 0.867375626451534`
  - Delta Acc / Macro-F1: `0.005071851225697421 / 0.004933269244965954`
  - Routed test count / wrong precision: `237 / 0.3755274261603376`
  - Routed wrong coverage: `0.5527950310559007`
  - Fixed / broken / net gain: `14 / 8 / 6`
  - Wrong-node fix rate: `0.08695652173913043`
  - Correct-node break rate: `0.007827788649706457`
  - Conditional fix / break on routed wrong/correct:
    `0.15730337078651685 / 0.05405405405405406`
  - Selected budget / beta: `0.2 / 0.2`
- Summary:
  Direct raw concat without projectors or `graph_fused` is positive when the
  graph expert is `graph_follower`; compared with `graph_following`, it fixes
  more selected wrong nodes and breaks fewer selected correct nodes under the
  exact same reused router and routed set.

### mpe__raw_concat_single_graph_following__highbase__seed1

- Status: completed
- Scope: fixed-router comparable single-expert routed-node refiner ablation
- Baseline: high-base frozen SimTeG seed 1
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_graph_following__highbase__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_graph_following__highbase__seed1/seed_1/stages/joint_router_refinement`
- Manifest:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_graph_following__highbase__seed1/seed_1/stages/joint_router_refinement/manifest.json`
- Metrics:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_graph_following__highbase__seed1/seed_1/stages/joint_router_refinement/metrics.json`
- Analysis:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_graph_following__highbase__seed1/seed_1/stages/joint_router_refinement/analysis_summary.json`
- Outputs:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_graph_following__highbase__seed1/seed_1/stages/joint_router_refinement/outputs.pt`
- Checkpoint:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_graph_following__highbase__seed1/seed_1/stages/joint_router_refinement/checkpoint.pt`
- Per-node test:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_graph_following__highbase__seed1/seed_1/stages/joint_router_refinement/per_node_test.jsonl`
- Routed cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
- Frozen router reuse root:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement`
- Prompt fusion:
  `raw_concat_single_graph_following`
- Refiner input:
  `[z_gnn || graph_following || structural_side_channel]`
- Metrics:
  - Base Acc / Macro-F1: `0.863905325443787 / 0.862442357206568`
  - Final Acc / Macro-F1: `0.8571428571428571 / 0.8564865561353223`
  - Delta Acc / Macro-F1: `-0.006762468300929858 / -0.005955801071245714`
  - Routed test count / wrong precision: `237 / 0.3755274261603376`
  - Fixed / broken / net gain: `17 / 25 / -8`
  - Wrong-node fix rate / correct-node break rate:
    `0.10559006211180125 / 0.02446183953033268`
  - Conditional fix / break on routed wrong/correct:
    `0.19101123595505617 / 0.16891891891891891`
- Summary:
  `graph_following` is high-recall but high-risk: it fixes the most wrong
  nodes among single experts, but breaks many correct nodes, so net gain is
  negative.

### mpe__raw_concat_single_graph_follower__highbase__seed1

- Status: completed
- Scope: fixed-router comparable single-expert routed-node refiner ablation
- Baseline: high-base frozen SimTeG seed 1
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_graph_follower__highbase__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_graph_follower__highbase__seed1/seed_1/stages/joint_router_refinement`
- Manifest:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_graph_follower__highbase__seed1/seed_1/stages/joint_router_refinement/manifest.json`
- Metrics:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_graph_follower__highbase__seed1/seed_1/stages/joint_router_refinement/metrics.json`
- Analysis:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_graph_follower__highbase__seed1/seed_1/stages/joint_router_refinement/analysis_summary.json`
- Outputs:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_graph_follower__highbase__seed1/seed_1/stages/joint_router_refinement/outputs.pt`
- Checkpoint:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_graph_follower__highbase__seed1/seed_1/stages/joint_router_refinement/checkpoint.pt`
- Per-node test:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_graph_follower__highbase__seed1/seed_1/stages/joint_router_refinement/per_node_test.jsonl`
- Routed cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
- Frozen router reuse root:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement`
- Prompt fusion:
  `raw_concat_single_graph_follower`
- Refiner input:
  `[z_gnn || graph_follower || structural_side_channel]`
- Metrics:
  - Base Acc / Macro-F1: `0.863905325443787 / 0.862442357206568`
  - Final Acc / Macro-F1: `0.849535080304311 / 0.8488610785732331`
  - Delta Acc / Macro-F1: `-0.014370245139475935 / -0.013581278633334914`
  - Routed test count / wrong precision: `237 / 0.3755274261603376`
  - Fixed / broken / net gain: `13 / 30 / -17`
  - Wrong-node fix rate / correct-node break rate:
    `0.08074534161490683 / 0.029354207436399216`
  - Conditional fix / break on routed wrong/correct:
    `0.14606741573033707 / 0.20270270270270271`
- Summary:
  `graph_follower` alone is the weakest single expert because it breaks the
  most correct nodes. Its positive triplet result requires tweet/conflict
  context and is not explained by follower evidence alone.

### mpe__raw_concat_single_tweet__highbase__seed1

- Status: completed
- Scope: fixed-router comparable single-expert routed-node refiner ablation
- Baseline: high-base frozen SimTeG seed 1
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_tweet__highbase__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_tweet__highbase__seed1/seed_1/stages/joint_router_refinement`
- Manifest:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_tweet__highbase__seed1/seed_1/stages/joint_router_refinement/manifest.json`
- Metrics:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_tweet__highbase__seed1/seed_1/stages/joint_router_refinement/metrics.json`
- Analysis:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_tweet__highbase__seed1/seed_1/stages/joint_router_refinement/analysis_summary.json`
- Outputs:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_tweet__highbase__seed1/seed_1/stages/joint_router_refinement/outputs.pt`
- Checkpoint:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_tweet__highbase__seed1/seed_1/stages/joint_router_refinement/checkpoint.pt`
- Per-node test:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_tweet__highbase__seed1/seed_1/stages/joint_router_refinement/per_node_test.jsonl`
- Routed cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
- Frozen router reuse root:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement`
- Prompt fusion:
  `raw_concat_single_tweet`
- Refiner input:
  `[z_gnn || tweet || structural_side_channel]`
- Metrics:
  - Base Acc / Macro-F1: `0.863905325443787 / 0.862442357206568`
  - Final Acc / Macro-F1: `0.8647506339814032 / 0.8633680976635136`
  - Delta Acc / Macro-F1: `0.0008453085376162184 / 0.0009257404569456007`
  - Routed test count / wrong precision: `237 / 0.3755274261603376`
  - Fixed / broken / net gain: `7 / 6 / 1`
  - Wrong-node fix rate / correct-node break rate:
    `0.043478260869565216 / 0.005870841487279843`
  - Conditional fix / break on routed wrong/correct:
    `0.07865168539325842 / 0.04054054054054054`
- Summary:
  `tweet` is conservative and slightly positive. It fixes fewer nodes than graph
  experts but has much lower break, making it useful as stabilizing evidence.

### mpe__raw_concat_single_conflict__highbase__seed1

- Status: completed
- Scope: fixed-router comparable single-expert routed-node refiner ablation
- Baseline: high-base frozen SimTeG seed 1
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_conflict__highbase__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_conflict__highbase__seed1/seed_1/stages/joint_router_refinement`
- Manifest:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_conflict__highbase__seed1/seed_1/stages/joint_router_refinement/manifest.json`
- Metrics:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_conflict__highbase__seed1/seed_1/stages/joint_router_refinement/metrics.json`
- Analysis:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_conflict__highbase__seed1/seed_1/stages/joint_router_refinement/analysis_summary.json`
- Outputs:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_conflict__highbase__seed1/seed_1/stages/joint_router_refinement/outputs.pt`
- Checkpoint:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_conflict__highbase__seed1/seed_1/stages/joint_router_refinement/checkpoint.pt`
- Per-node test:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_single_conflict__highbase__seed1/seed_1/stages/joint_router_refinement/per_node_test.jsonl`
- Routed cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
- Frozen router reuse root:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement`
- Prompt fusion:
  `raw_concat_single_conflict`
- Refiner input:
  `[z_gnn || conflict || structural_side_channel]`
- Metrics:
  - Base Acc / Macro-F1: `0.863905325443787 / 0.862442357206568`
  - Final Acc / Macro-F1: `0.8664412510566357 / 0.8647831773645618`
  - Delta Acc / Macro-F1: `0.002535925612848766 / 0.0023408201579937504`
  - Routed test count / wrong precision: `237 / 0.3755274261603376`
  - Fixed / broken / net gain: `8 / 5 / 3`
  - Wrong-node fix rate / correct-node break rate:
    `0.049689440993788817 / 0.004892367906066536`
  - Conditional fix / break on routed wrong/correct:
    `0.0898876404494382 / 0.033783783783783786`
- Summary:
  `conflict` is the best single expert by net gain. It is not the highest-fix
  expert, but it has the lowest break and therefore best preserves the strong
  SimTeG base.

### single-expert routed-node specialty summary

- Scope: seed-1 diagnostic only, fixed router reused from
  `server_prompt_expert_highbase_preiter_graph_following_20260527`.
- Shared routed set:
  - test routed count: `237`
  - routed wrong precision: `0.3755274261603376`
  - routed wrong coverage: `0.5527950310559007`
- Single-expert union:
  - union fixed wrong nodes: `26`
  - union broken correct nodes: `39`
  - all four experts fix the same wrong node: `1`
  - no correct node is broken by all four experts
- Unique single-expert fixes:
  - `graph_following`: `6`
  - `graph_follower`: `2`
  - `tweet`: `0`
  - `conflict`: `3`
- Unique single-expert breaks:
  - `graph_following`: `5`
  - `graph_follower`: `9`
  - `tweet`: `0`
  - `conflict`: `3`
- Node-mode observation:
  - actual fixes are concentrated in dense-neighborhood routed nodes
    (`1-hop=11+`, both following and follower present)
  - no-directional-neighbor nodes are essentially preserved, with no fixes and
    no breaks in these single-expert runs
  - graph experts are high-variance evidence; tweet/conflict are lower-break
    stabilizers
  - the positive `graph_follower + tweet + conflict` triplet is not a simple
    single-expert effect: it creates `6` fixes not present in the single-expert
    union while adding only `2` breaks outside the single-expert break union

### mpe__expert_selection_diagnostic__highbase__seed1

- Status: completed
- Scope: post-hoc selector diagnostic over existing fixed-router refiner outputs
- Claim boundary:
  diagnostic only; no mainline selector code changed, and no test labels were
  used by the learned selector.
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__expert_selection_diagnostic__highbase__seed1`
- Script:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__expert_selection_diagnostic__highbase__seed1/run_selector_diag.py`
- Diagnostics:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__expert_selection_diagnostic__highbase__seed1/diagnostics.json`
- Summary CSV:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__expert_selection_diagnostic__highbase__seed1/summary.csv`
- Candidate actions:
  `{base, graph_following, graph_follower, tweet, conflict, following_triplet, follower_triplet}`
- Base test:
  - Acc / Macro-F1: `0.863905325443787 / 0.862442357206568`
- Best actual candidate:
  - `follower_triplet`
  - Acc / Macro-F1: `0.8689771766694844 / 0.867375626451534`
  - fix / break / net: `14 / 8 / 6`
- Oracle upper bounds:
  - single experts only:
    - Acc / Macro-F1: `0.885883347421809 / 0.8848880883443648`
    - fix / break / net: `26 / 0 / 26`
  - all candidates:
    - Acc / Macro-F1: `0.8909551986475064 / 0.8898965609043941`
    - fix / break / net: `32 / 0 / 32`
- Learned selector diagnostic:
  - multinomial logistic selector trained on routed train and selected on routed
    validation:
    - selected `C=3.0`
    - test Acc / Macro-F1: `0.8571428571428571 / 0.8563838766519192`
    - fix / break / net: `32 / 40 / -8`
    - interpretation: it can find many fixable wrong nodes, but cannot protect
      correct nodes.
  - confidence-threshold best-expert selector selected on validation:
    - selected threshold: `0.5`
    - test Acc / Macro-F1: `0.8664412510566357 / 0.8654993063191162`
    - fix / break / net: `14 / 11 / 3`
    - interpretation: conservative confidence selection is positive but still
      below the fixed `follower_triplet` candidate.
- Design implication:
  The evidence supports future node-specific expert selection/fusion because
  oracle action choice has substantial headroom, but it does not support naive
  multiclass expert routing. The next method should train explicit per-action
  utility or abstain heads that optimize `change only if expected gain > 0`,
  with validation-calibrated break control.

### mpe__gaugllm_selector__highbase__seed1

- Status: completed
- Scope: strict runtime-reuse migration of only the GAugLLM-style
  mixture-of-prompt-expert / context-aware selector into the current
  routed-node refiner
- Claim boundary:
  selector transplant only; no prompt-cache rebuild, no router retraining, no
  backbone change, and no metadata expert selection
- Baseline: high-base frozen SimTeG seed 1
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_selector__highbase__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_selector__highbase__seed1/seed_1/stages/joint_router_refinement`
- Manifest:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_selector__highbase__seed1/seed_1/stages/joint_router_refinement/manifest.json`
- Metrics:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_selector__highbase__seed1/seed_1/stages/joint_router_refinement/metrics.json`
- Analysis:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_selector__highbase__seed1/seed_1/stages/joint_router_refinement/analysis_summary.json`
- Outputs:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_selector__highbase__seed1/seed_1/stages/joint_router_refinement/outputs.pt`
- Checkpoint:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_selector__highbase__seed1/seed_1/stages/joint_router_refinement/checkpoint.pt`
- Per-node test:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_selector__highbase__seed1/seed_1/stages/joint_router_refinement/per_node_test.jsonl`
- Reused routed cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
- Reused adjacent manifest:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531_manifest.json`
- Reused explanation sidecars:
  `graph_following`, `graph_follower`, `tweet`, `conflict`
- Reuse contract:
  - no prompt-cache rebuild
  - no explanation regeneration
  - runtime loader reads sidecars from `component_explanation_sidecar_paths`
  - duplicate sidecar rows are resolved by `node_id` with last-row-wins
  - `metadata_structured` is not selectable; it stays as selector-context
    support plus refiner side-channel
- Frozen router reuse root:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement`
- Prompt fusion:
  `gaugllm_selector`
- Refiner input:
  `[z_gnn || fused_selector(graph_following, graph_follower, tweet, conflict) || metadata_structured_proj || structural_side_channel]`
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
/root/mambaforge/envs/lmbot/bin/python main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --external_frozen_g0_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_roberta_20260527/seed_1 \
  --joint_refiner_embedding_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt \
  --joint_routing_protocol frozen_router_reuse \
  --joint_router_reuse_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement \
  --joint_prompt_expert_fusion gaugllm_selector \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_selector__highbase__seed1 \
  --seeds 1 \
  --disable_wandb
```

- Metrics:
  - Base Acc / Macro-F1: `0.863905325443787 / 0.862442357206568`
  - Final Acc / Macro-F1: `0.863905325443787 / 0.862845320676646`
  - Delta Acc / Macro-F1: `0.0 / 0.00040296347007795497`
  - Fixed / broken / net gain: `9 / 9 / 0`
  - Wrong-node fix rate: `0.055900621118012424`
  - Correct-node break rate: `0.008806262230919765`
  - Selected budget / beta: `0.2 / 0.2`
- Routed selector behavior:
  - per-node artifact inspection shows the selector collapses to `tweet` on all
    routed test nodes (`237` routed rows, `89` routed base-wrong rows)
  - no routed test node selects `graph_following`, `graph_follower`, or
    `conflict`
- Comparable anchor:
  - `mpe__raw_concat_follower_triplet__highbase__seed1` remains stronger on the
    same high-base routed set:
    - Acc / Macro-F1: `0.8689771766694844 / 0.867375626451534`
    - Fixed / broken / net gain: `14 / 8 / 6`
- Comparison caveat:
  - the available `mpe_gated` artifact on server currently comes from
    `/root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_fullgraph_reuse_mpe_gate_20260601/...`
    and uses a weaker full-graph-reuse base (`0.8495 / 0.8486`), so it is not a
    strict same-base comparator for this entry
- Summary:
  The runtime-reuse migration works technically and preserves the requested
  cache/sidecar contract, but under the strong frozen SimTeG base it does not
  outperform the existing `raw_concat_follower_triplet` anchor. The current
  selector collapses to tweet-only routing, so the immediate next diagnostic is
  to inspect selector-logit balance and why context-aware attention is not
  activating the graph/conflict experts on routed test nodes.

### mpe__gaugllm_mope_diag_matrix__highbase__seed1

- Status: completed
- Scope: post-audit MoPE diagnostic matrix for the strict GAugLLM
  `SimilarityAttentionMLP`-style selector path, with the same routed cache and
  frozen-router reuse line as `mpe__gaugllm_mope_similarity__highbase__seed1`
- Claim boundary:
  selector-head transplant plus diagnostic calibration only; this is not a full
  end-to-end GAugLLM reproduction
- Important run note:
  the first launcher attempt accidentally used the CLI default `--device -1`
  and then exposed a real parameter-plumbing bug in the new MoPE diagnostic
  flags. Final recorded results below come from the corrected GPU run with:
  - `--device 0`
  - `--joint_prompt_expert_mope_temperature 0.2`
  - `--joint_prompt_expert_mope_logit_norm {none,branch_zscore,combined_zscore}`
- Shared command base:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=1 /root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --external_frozen_g0_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_roberta_20260527/seed_1 \
  --joint_refiner_embedding_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt \
  --joint_routing_protocol frozen_router_reuse \
  --joint_router_reuse_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement \
  --joint_prompt_expert_fusion gaugllm_mope \
  --joint_prompt_expert_mope_attention similarity \
  --joint_prompt_expert_mope_temperature 0.2 \
  --joint_prompt_expert_mope_logit_norm <none|branch_zscore|combined_zscore> \
  --device 0 \
  --seeds 1 \
  --disable_wandb \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/<experiment_id>
```

- Artifact roots:
  - `none`:
    `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_diag_none_t02__highbase__seed1`
  - `branch_zscore`:
    `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_diag_branch_zscore_t02__highbase__seed1`
  - `combined_zscore`:
    `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_diag_combined_zscore_t02__highbase__seed1`
- Baseline:
  - Base Acc / Macro-F1: `0.863905325443787 / 0.862442357206568`
- Results:

| run | Acc | Macro-F1 | fix | break | net | routed wrong precision | selected expert on routed test |
|---|---:|---:|---:|---:|---:|---:|---|
| `orig_similarity` | `0.8672865595942519` | `0.8659068117644426` | `5` | `1` | `+4` | `0.3755274261603376` | old artifact showed tweet-only collapse |
| `none` | `0.8647506339814032` | `0.8634603509865364` | `6` | `5` | `+1` | `0.3881856540084388` | `232 graph_follower + 5 conflict` |
| `branch_zscore` | `0.8630600169061707` | `0.8620139968895801` | `9` | `10` | `-1` | `0.3881856540084388` | `237 graph_following` |
| `combined_zscore` | `0.8655959425190194` | `0.8641511478002752` | `4` | `2` | `+2` | `0.3881856540084388` | `237 conflict` |

- Diagnostic conclusion:
  - After fixing the launcher/device path and MoPE parameter plumbing, the
    selector no longer reproduces the old tweet-only collapse.
  - The failure mode is now clearer: each diagnostic variant still collapses
    almost globally to one dominant expert family:
    - `none -> graph_follower`
    - `branch_zscore -> graph_following`
    - `combined_zscore -> conflict`
  - So the current issue is not “all weights are always tweet”; it is “the
    selector learns a run-level global bias rather than a node-specific expert
    policy”.
  - `combined_zscore` slightly improves over the corrected `none` run in break
    control (`2` vs `5`) and net gain (`+2` vs `+1`), but it still trails the
    earlier `orig_similarity` anchor (`+4`) and the stronger
    `raw_concat_follower_triplet` anchor (`+6`).

### mpe__gaugllm_mope_selector_calibration__highbase__seed1

- Status: completed
- Scope: narrow selector-calibration diagnostic for the corrected
  `gaugllm_mope` path under the same routed cache and frozen-router reuse line
  as `mpe__gaugllm_mope_diag_matrix__highbase__seed1`
- Claim boundary:
  calibration-only ablation over the selector head; no cache rebuild, no
  explanation regeneration, no router retraining, and no end-to-end official
  GAugLLM reproduction claim
- Goal:
  test whether temperature alone, then very light selector entropy /
  load-balance regularization, can break the run-level single-expert bias and
  produce even weak node-specific expert selection
- Shared semantic inputs:
  - routed cache:
    `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
  - adjacent manifest:
    `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531_manifest.json`
  - frozen router reuse root:
    `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement`
  - baseline:
    `0.863905325443787 / 0.862442357206568`
- Shared command base:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=1 /root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --external_frozen_g0_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_roberta_20260527/seed_1 \
  --joint_refiner_embedding_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt \
  --joint_routing_protocol frozen_router_reuse \
  --joint_router_reuse_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement \
  --joint_prompt_expert_fusion gaugllm_mope \
  --joint_prompt_expert_mope_attention similarity \
  --joint_prompt_expert_mope_logit_norm none \
  --joint_prompt_expert_mope_temperature <temperature> \
  --joint_prompt_expert_mope_entropy_weight <entropy_weight> \
  --joint_prompt_expert_mope_load_balance_weight <load_balance_weight> \
  --device 0 \
  --seeds 1 \
  --disable_wandb \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/<experiment_id>
```

- Log files:
  - temperature sweep:
    `/root/workspace/LMbot/LLMbot/server_logs/mpe_gaugllm_mope_tempcal_highbase_seed1.log`
  - regularization sweep:
    `/root/workspace/LMbot/LLMbot/server_logs/mpe_gaugllm_mope_regcal_highbase_seed1.log`
- Artifact roots:
  - `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_temp_t10__highbase__seed1`
  - `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_temp_t05__highbase__seed1`
  - `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_diag_none_t02__highbase__seed1`
  - `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_temp_t01__highbase__seed1`
  - `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_t02_entropy001__highbase__seed1`
  - `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_t02_balance001__highbase__seed1`
  - `/root/workspace/LMbot/LLMbot/experiments/mpe__gaugllm_mope_t02_entropy001_balance001__highbase__seed1`
- Artifact validation:
  all seven runs have `manifest.json`, `metrics.json`, `analysis_summary.json`,
  `outputs.pt`, `checkpoint.pt`, and `per_node_test.jsonl`
- Results:

| run | temp | entropy | balance | Acc | Macro-F1 | fix | break | net | selector behavior on routed test | mean max weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `temp_t10` | `1.0` | `0.0` | `0.0` | `0.8639053` | `0.8626744` | `6` | `6` | `0` | `237 tweet` | `0.9993064` |
| `temp_t05` | `0.5` | `0.0` | `0.0` | `0.8622147` | `0.8613392` | `13` | `15` | `-2` | `232 graph_follower + 5 conflict` | `0.9999996` |
| `temp_t02` | `0.2` | `0.0` | `0.0` | `0.8647506` | `0.8634604` | `6` | `5` | `+1` | near-global follower-dominant collapse (see prior diag matrix entry) | `~1.0` |
| `temp_t01` | `0.1` | `0.0` | `0.0` | `0.8622147` | `0.8610567` | `7` | `9` | `-2` | `237 conflict` | `1.0` |
| `t02_entropy001` | `0.2` | `0.01` | `0.0` | `0.8596788` | `0.8589157` | `15` | `20` | `-5` | `237 tweet` | `1.0` |
| `t02_balance001` | `0.2` | `0.0` | `0.01` | `0.8672866` | `0.8658599` | `6` | `2` | `+4` | `237 tweet` | `1.0` |
| `t02_entropy001_balance001` | `0.2` | `0.01` | `0.01` | `0.8672866` | `0.8656643` | `8` | `4` | `+4` | `232 graph_follower + 5 tweet` | `1.0` |

- Diagnostic conclusion:
  - Temperature alone changes which expert dominates, but it does not produce
    meaningful node-specific routing. The dominant family simply moves across
    `tweet`, `graph_follower`, and `conflict`.
  - The highest selector entropy among the pure temperature runs is still the
    `temp=1.0` tweet-only case (`mean_max_weight = 0.9993064`), so "more
    temperature" is not enough to escape global collapse.
  - Light entropy regularization is actively harmful here: `temp=0.2,
    entropy=0.01` increases fix count but drives break even higher (`20`),
    collapsing back to tweet-only routing and dropping Macro-F1 below base.
  - Light load-balance regularization is the only positive control in this
    sweep: `temp=0.2, balance=0.01` reaches `+4` net gain and
    `0.8658599` Macro-F1, but it does so while still selecting `tweet` for all
    routed test nodes.
  - Adding both regularizers keeps positive net gain (`+4`) but still does not
    yield genuine node-specific routing; the routed test set remains almost
    globally dominated by `graph_follower` with only five tweet selections.
  - Therefore this calibration sweep supports a narrow but important boundary:
    lightweight selector calibration can improve fix/break tradeoff, but the
    gain does not come from learned node-specific expert choice. Under the
    current supervision, the selector still behaves like a globally biased soft
    fusion head rather than a per-node expert policy.
- Next-step implication:
  do not interpret these results as evidence that the current MoPE selector has
  learned heterogeneous expert usage. The next method should change the
  supervision/decision target itself, for example with explicit keep/change or
  per-action utility modeling, before investing in stronger node-specific
  selector architectures.

### mpe__utility_correction_moe__highbase__seed1

- Status: completed on server, fixed-router validation, no cache rebuild
- Scope:
  Stage A fixed-router correction MoE. The router and routed set stay frozen;
  only routed-node correction expert heads, per-expert utility selector, and
  abstain/keep-change behavior are trained.
- Claim boundary:
  This tests whether expert-specific utility supervision can reduce break and
  improve net gain on the existing routed union. It is not true joint
  router-expert optimization, because routed nodes do not change during
  training.
- Baseline / anchors:
  - frozen SimTeG high-base: `Acc=0.863905325443787`,
    `Macro-F1=0.862442357206568`
  - current strongest routed-node anchor:
    `mpe__raw_concat_follower_triplet__highbase__seed1`,
    `Acc=0.8689771766694844`, `Macro-F1=0.867375626451534`,
    `fix/break/net=14/8/+6`
- Reused semantic inputs:
  - routed cache:
    `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
  - adjacent manifest:
    `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531_manifest.json`
  - frozen router reuse root:
    `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement`
- Method:
  `--joint_prompt_expert_fusion utility_correction_moe` uses
  `graph_follower`, `tweet`, `conflict`, and `metadata_structured` as correction
  experts. Each expert has a label head; each node-expert pair also has a
  utility head trained against `base_loss - expert_loss - beta > 0`. The maximum
  utility probability is the keep/change abstain score under
  `--joint_refiner_gate_policy hard_keep_change`.
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --external_frozen_g0_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_roberta_20260527/seed_1 \
  --joint_refiner_embedding_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt \
  --joint_routing_protocol frozen_router_reuse \
  --joint_router_reuse_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement \
  --joint_prompt_expert_fusion utility_correction_moe \
  --joint_refiner_gate_policy hard_keep_change \
  --joint_refiner_gate_threshold 0.5 \
  --joint_correction_moe_expert_weight 0.5 \
  --joint_correction_moe_utility_weight 1.0 \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/mpe__utility_correction_moe__highbase__seed1 \
  --seeds 1 \
  --device 0 \
  --disable_wandb
```

- Required artifacts after run:
  `manifest.json`, `metrics.json`, `analysis_summary.json`, `outputs.pt`,
  `checkpoint.pt`, and `per_node_test.jsonl` under
  `/root/workspace/LMbot/LLMbot/experiments/mpe__utility_correction_moe__highbase__seed1/seed_1/stages/joint_router_refinement`

### mpe__metades_selector__highbase__seed1

- Status: planned for fixed-router validation, no cache rebuild
- Scope:
  META-DES-style routed correction selector over the existing clean v2 routed
  prompt-expert cache. The router, routed set, backbone, prompt cache, and
  correction utility target protocol stay fixed.
- Claim boundary:
  This is a META-DES-style competence-selector migration inside the current
  routed-node refiner, not an official full META-DES reproduction. It tests
  whether base/expert competence meta-features improve per-action utility
  selection without changing graph propagation or router features.
- Baseline / anchors:
  - frozen SimTeG high-base: `Acc=0.863905325443787`,
    `Macro-F1=0.862442357206568`
  - current strongest routed-node anchor:
    `mpe__raw_concat_follower_triplet__highbase__seed1`,
    `Acc=0.8689771766694844`, `Macro-F1=0.867375626451534`,
    `fix/break/net=14/8/+6`
- Reused semantic inputs:
  - routed cache:
    `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
  - adjacent manifest:
    `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531_manifest.json`
  - frozen router reuse root:
    `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement`
- Method:
  `--joint_prompt_expert_fusion metades_selector` selects among
  `graph_follower`, `tweet`, `conflict`, and a runtime-derived
  `follower_triplet`. Each action has a label head and a utility head. The
  utility head receives base/expert competence meta-features including
  confidence, margin, entropy, bot probability, disagreement, and probability
  gap. `metadata_structured` remains a side feature and is not selectable.
  `follower_triplet` is built from existing expert projections, so no prompt
  cache is regenerated.
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__metades_selector__highbase__seed1`
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --external_frozen_g0_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_roberta_20260527/seed_1 \
  --joint_refiner_embedding_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt \
  --joint_routing_protocol frozen_router_reuse \
  --joint_router_reuse_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement \
  --joint_prompt_expert_fusion metades_selector \
  --joint_refiner_gate_policy hard_keep_change \
  --joint_refiner_gate_threshold 0.5 \
  --joint_correction_moe_expert_weight 0.5 \
  --joint_correction_moe_utility_weight 1.0 \
  --joint_correction_moe_utility_target decision_gain \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/mpe__metades_selector__highbase__seed1 \
  --seeds 1 \
  --device 0 \
  --disable_wandb
```

- Required artifacts after run:
  `manifest.json`, `metrics.json`, `analysis_summary.json`, `outputs.pt`,
  `checkpoint.pt`, and `per_node_test.jsonl` under
  `/root/workspace/LMbot/LLMbot/experiments/mpe__metades_selector__highbase__seed1/seed_1/stages/joint_router_refinement`
- Completed artifacts:
  - `manifest.json`: present, `fusion_mode=metades_selector`
  - `metrics.json`: present
  - `analysis_summary.json`: present
  - `outputs.pt`: present
  - `checkpoint.pt`: present
  - `per_node_test.jsonl`: present, `1183` test rows
- Result:
  - base frozen SimTeG: `Acc=0.863905325443787`,
    `Macro-F1=0.862442357206568`
  - `metades_selector`: `Acc=0.863905325443787`,
    `Macro-F1=0.862442357206568`
  - fix / break / net: `0 / 0 / 0`
  - routed test nodes: `237`, with `92` base-wrong and `145` base-correct
  - routed wrong precision: `0.3881856540084388`
  - conditional fix rate on selected wrong: `0.0`
  - gate positive rate: `0.0`
  - mean gate probability on routed nodes: `0.1027783093038742`
- Selector diagnostic:
  - selected action distribution on routed test nodes:
    `graph_follower=192`, `tweet=42`, `conflict=3`,
    `follower_triplet=0`
  - maximum observed gate probability on routed test nodes was about `0.155`,
    so the configured `--joint_refiner_gate_threshold 0.5` produced complete
    abstention.
  - Lower-threshold post-hoc inspection shows potential decision-gain positives
    exist among selected actions (`28` at threshold `0.05`, `23` at `0.10`,
    `14` at `0.125`), but this was not an executed-threshold result and should
    be treated only as calibration evidence.
- Interpretation boundary:
  `metades_selector` successfully migrates META-DES-style competence features
  into the routed-node correction selector and avoids breaking correct nodes,
  but this default thresholded run does not improve over the high-base frozen
  SimTeG baseline or the `raw_concat_follower_triplet` anchor. The next
  experiment should tune/validate the keep-change threshold or utility
  calibration on validation before claiming correction benefit.
- Post-hoc validation calibration:
  - Artifacts:
    - `metades_gate_calibration_summary.json`
    - `metades_gate_calibration_sweep.csv`
    - `metades_extended_calibration_summary.json`
    - `metades_global_threshold_sweep.csv`
  - Global threshold selection:
    - selection split: validation
    - rule: maximize validation Macro-F1, then validation net, then lower break,
      then higher threshold
    - selected threshold: `0.5`
    - locked test result: unchanged from base, `fix/break/net=0/0/0`
    - interpretation: a single global gate threshold is not calibrated well
      enough to recover the low-threshold hindsight positives.
  - Per-action threshold selection:
    - `graph_follower`: threshold `0.5`, valid `0/0/0`, test `0/0/0`
    - `tweet`: threshold `0.085`, valid `1/0/+1`, test `2/1/+1`
    - `conflict`: threshold `0.5`, valid `0/0/0`, test `0/0/0`
    - `follower_triplet`: threshold `0.5`, valid `0/0/0`, test `0/0/0`
    - combined locked test result:
      `Acc=0.8647506237030029`, `Macro-F1=0.863223553759618`,
      `fix/break/net=2/1/+1`, `opened_count=10`
  - Bounds:
    - open all selected actions on test: `fix/break/net=28/25/+3`
    - selected-action oracle on test: `fix/break/net=28/0/+28`
  - Interpretation:
    Per-action calibration is weakly positive and validates that utility scores
    contain some actionable signal, but the effect is far below the current
    `raw_concat_follower_triplet` anchor (`+6`) and far below the
    selected-action oracle (`+28`). The next optimization target should be
    per-action utility calibration / ranking, not another global gate.

### mpe__metades_selector_per_action_calibrated__highbase__seed1

- Status: completed on server.
- Scope: promote the prior post-hoc per-action gate calibration into the
  `joint_router_refinement` stage. Router, routed set, backbone, and prompt
  cache stay frozen; no prompt cache is rebuilt.
- Baselines for comparison:
  - frozen SimTeG high-base: `Acc=0.863905325443787`,
    `Macro-F1=0.862442357206568`
  - strongest current routed-node anchor:
    `mpe__raw_concat_follower_triplet__highbase__seed1`,
    `Acc=0.8689771766694844`, `Macro-F1=0.867375626451534`,
    `fix/break/net=14/8/+6`
  - previous post-hoc per-action calibration:
    `Acc=0.8647506237030029`, `Macro-F1=0.863223553759618`,
    `fix/break/net=2/1/+1`
- Reused routed cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
- Reused frozen router stage:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement`
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__metades_selector_per_action_calibrated__highbase__seed1`
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --external_frozen_g0_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_roberta_20260527/seed_1 \
  --joint_refiner_embedding_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt \
  --joint_routing_protocol frozen_router_reuse \
  --joint_router_reuse_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement \
  --joint_prompt_expert_fusion metades_selector \
  --joint_refiner_gate_policy hard_keep_change \
  --joint_refiner_gate_threshold 0.5 \
  --joint_correction_moe_expert_weight 0.5 \
  --joint_correction_moe_utility_weight 1.0 \
  --joint_correction_moe_utility_target net_gain \
  --joint_correction_moe_gate_calibration per_action_threshold \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/mpe__metades_selector_per_action_calibrated__highbase__seed1 \
  --seeds 1 \
  --device 0 \
  --disable_wandb
```

- Expected artifacts:
  `manifest.json`, `metrics.json`, `analysis_summary.json`, `outputs.pt`,
  `checkpoint.pt`, `per_node_test.jsonl`.
- Artifact contract additions:
  `gate_calibration_metadata`, `calibration_gate_decision`,
  `calibration_threshold`, `calibration_selected_action_score`,
  `correction_moe_break_target`, and `correction_moe_utility_reward`.
- Completed artifacts:
  - stage root:
    `/root/workspace/LMbot/LLMbot/experiments/mpe__metades_selector_per_action_calibrated__highbase__seed1/seed_1/stages/joint_router_refinement`
  - `manifest.json`: present
  - `metrics.json`: present
  - `analysis_summary.json`: present
  - `outputs.pt`: present, includes calibration tensors and correction utility rewards
  - `checkpoint.pt`: present, includes `gate_calibration_metadata`
  - `per_node_test.jsonl`: present, `1183` test rows
- Result:
  - Acc: `0.8664412510566357`
  - Macro-F1: `0.8648840504279436`
  - delta Macro-F1 vs high-base: `+0.0024416932213755516`
  - fix / break / net: `4 / 1 / +3`
  - changed test predictions: `5`
  - routed test nodes: `237`
  - routed wrong precision: `0.3881856540084388`
  - conditional fix rate on selected wrong: `0.043478260869565216`
  - conditional break rate on selected correct: `0.006896551724137931`
  - gate positive rate: `0.05063291139240506`
  - mean gate probability on routed nodes: `0.10277830928815568`
  - calibration open count in full graph output tensor: `35`
- Validation-locked calibration:
  - mode: `per_action_threshold`
  - thresholds:
    - `graph_follower`: `0.15`
    - `tweet`: `0.085`
    - `conflict`: `0.147175`
    - `follower_triplet`: `1.0` (`no_valid_examples_for_selected_action`)
  - validation selected policy:
    `fix/break/net=1/0/+1`, `opened_count=6`
- Selector distribution on routed test nodes:
  - `graph_follower=192`
  - `tweet=42`
  - `conflict=3`
  - `follower_triplet=0`
- Interpretation boundary:
  Stage-owned per-action calibration improves over the previous uncalibrated
  `metades_selector` (`0/0/0`) and improves over the earlier post-hoc
  per-action result (`+1`) by producing `+3` net gain. It still does not exceed
  the current strongest raw-concat anchor (`raw_concat_follower_triplet`,
  `+6`), so it is evidence that calibration helps but not yet evidence that the
  learned selector has solved routed-node correction.

### mpe__metades_selector_rank_break_per_action__highbase__seed1

- Status: completed on server.
- Scope: same as `mpe__metades_selector_per_action_calibrated__highbase__seed1`,
  but adds break-aware within-node action ranking. This tests whether the
  selector can learn which action is worth changing for the same node, rather
  than only whether each action is utility-positive independently.
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__metades_selector_rank_break_per_action__highbase__seed1`
- Command delta:

```bash
  --joint_correction_moe_break_weight 2.0 \
  --joint_correction_moe_ranking_weight 0.5 \
  --joint_correction_moe_ranking_margin 0.1 \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/mpe__metades_selector_rank_break_per_action__highbase__seed1
```

- Full command is the same as
  `mpe__metades_selector_per_action_calibrated__highbase__seed1` with the
  command delta above.
- Completed artifacts:
  - stage root:
    `/root/workspace/LMbot/LLMbot/experiments/mpe__metades_selector_rank_break_per_action__highbase__seed1/seed_1/stages/joint_router_refinement`
  - `manifest.json`: present
  - `metrics.json`: present
  - `analysis_summary.json`: present
  - `outputs.pt`: present, includes calibration tensors and correction utility rewards
  - `checkpoint.pt`: present, includes `gate_calibration_metadata`
  - `per_node_test.jsonl`: present, `1183` test rows
- Result:
  - Acc: `0.8655959425190194`
  - Macro-F1: `0.8640537491336089`
  - delta Macro-F1 vs high-base: `+0.0016113919270408505`
  - fix / break / net: `3 / 1 / +2`
  - changed test predictions: `4`
  - routed test nodes: `237`
  - routed wrong precision: `0.3881856540084388`
  - conditional fix rate on selected wrong: `0.03260869565217391`
  - conditional break rate on selected correct: `0.006896551724137931`
  - gate positive rate: `0.05485232067510549`
  - mean gate probability on routed nodes: `0.17921475739167209`
  - calibration open count in full graph output tensor: `77`
- Validation-locked calibration:
  - mode: `per_action_threshold`
  - thresholds:
    - `conflict`: `0.103272`
    - `follower_triplet`: `0.195236`
    - `graph_follower`: `1.0` (`no_valid_examples_for_selected_action`)
    - `tweet`: `1.0` (`no_valid_examples_for_selected_action`)
  - validation selected policy:
    `fix/break/net=4/2/+2`, `opened_count=32`
- Selector distribution on routed test nodes:
  - `follower_triplet=232`
  - `conflict=5`
  - `graph_follower=0`
  - `tweet=0`
- Interpretation boundary:
  Break-aware ranking changes the selected-action distribution strongly toward
  the derived `follower_triplet` action and raises the mean utility score, but
  the locked test result is weaker than the simpler per-action calibrated run
  (`+2` vs `+3`). This suggests the ranking reward is not yet calibrated enough
  to improve the fix/break tradeoff; it should be treated as a diagnostic, not
  the new mainline.

### mpe__conflict_aware_correction_moe__highbase__seed1

- Status: completed on server.
- Scope: fixed-router routed-node correction experiment. This adapts the
  BotMoE selector lesson by removing `conflict` from the selectable action set
  and using it as a cross-view safety/context feature instead. Selectable
  actions are `graph_following`, `graph_follower`, and `tweet`; the action heads
  also receive `conflict`, `metadata_structured`, structural side-channel
  features, and runtime explanation-context embeddings from the existing routed
  sidecars.
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__conflict_aware_correction_moe__highbase__seed1`
- Reused prompt cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
- Reused frozen-router/high-base root:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_roberta_20260527/seed_1`
- Reused frozen routed set:
  `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement`
- No prompt-cache rebuild is part of this run.
- Planned command:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path /root/workspace/LMbot/TwiBot-20_seed_1/intermediate/LM/embeddings_iter_-1.pt \
  --external_frozen_g0_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_roberta_20260527/seed_1 \
  --joint_routing_protocol frozen_router_reuse \
  --joint_router_reuse_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement \
  --joint_refiner_embedding_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt \
  --joint_prompt_expert_fusion conflict_aware_correction_moe \
  --joint_refiner_target_mode keep_change \
  --joint_refiner_gate_policy hard_keep_change \
  --joint_correction_moe_utility_target net_gain \
  --joint_correction_moe_break_weight 2.0 \
  --joint_correction_moe_ranking_weight 0.5 \
  --joint_correction_moe_ranking_margin 0.1 \
  --joint_correction_moe_gate_calibration per_action_threshold \
  --risk_budgets 0.05,0.10,0.15,0.20,0.25 \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/mpe__conflict_aware_correction_moe__highbase__seed1 \
  --seeds 1 \
  --device 0 \
  --disable_wandb
```

- Success criteria:
  compare against high-base frozen SimTeG (`Acc=0.8639`, `Macro-F1=0.8624`),
  `raw_concat_follower_triplet` (`fix/break/net=14/8/+6`), and
  `metades_selector_per_action_calibrated` (`4/1/+3`). Primary diagnostics are
  fix/break/net, calibrated gate open count, selected-action distribution, and
  whether removing `conflict` as a selectable expert reduces the BotMoE-style
  false-positive-to-human break pattern.
- Completed artifacts:
  - stage root:
    `/root/workspace/LMbot/LLMbot/experiments/mpe__conflict_aware_correction_moe__highbase__seed1/seed_1/stages/joint_router_refinement`
  - `manifest.json`: present, `joint_prompt_expert_fusion=conflict_aware_correction_moe`
  - `metrics.json`: present
  - `analysis_summary.json`: present
  - `outputs.pt`: present, includes `correction_moe_*` tensors and selector weights
  - `checkpoint.pt`: present
  - `per_node_test.jsonl`: present, `1183` test rows
- Result:
  - final test `Acc / Macro-F1`: unchanged from high-base frozen SimTeG
    (`0.863905325443787 / 0.862442357206568`)
  - fix / break / net: `0 / 0 / 0`
  - changed test predictions: `0`
  - routed test nodes / routed wrong precision: `237 / 0.3881856540084388`
  - routed wrong coverage: `0.5714285714285714`
  - gate positive rate: `0.0`
  - mean gate probability on routed nodes: `0.26809688126236075`
  - utility-positive rate on routed test nodes: `0.3037974683544304`
- Validation-locked calibration:
  - mode: `per_action_threshold`
  - valid selected policy: `fix/break/net=1/0/+1`, `opened_count=3`
  - thresholds:
    - `graph_following`: `0.2`
    - `graph_follower`: `0.4`
    - `tweet`: `0.119998`
- Selector distribution on routed test nodes:
  - `graph_follower=232`
  - `graph_following=5`
  - `tweet=0`
  - selector mean entropy: `0.9907491207122803`
  - selector normalized mean entropy: `0.9018186926841736`
  - selector mean max weight: `0.5440521240234375`
- Diagnostic boundary:
  - This run does break the old BotMoE-style `conflict` collapse:
    `conflict` is no longer a selectable action, and the selector now operates
    over first-order experts only.
  - But the learned policy remains too conservative under validation-locked
    calibration, so the official stage output is full abstain on test.
  - Post-hoc diagnostics on the same artifacts show the opposite failure mode if
    abstain is removed: the selected action without thresholding would be
    `fix/break/net = 35/44/-9`, and even the best post-hoc global threshold on
    test over the selected action score would only reach about `+2`. Therefore
    the current improvement is structural cleanup of the selector design, not a
    new best-performing correction model.

### semantic_encoder__tape_explain_roberta_finetuned_gnn__seed1

- Status: planned for server launch
- Scope: full TwiBot-20 graph-detector experiment using TAPE-style
  `explanation -> encoder embedding` semantic features
- Claim boundary:
  this is a TAPE-style consumption experiment, not an official TAPE
  reproduction. The pipeline uses the current LLMbot v2 prompt-expert
  explanation generator (`Qwen2.5-Instruct`) and encodes the resulting
  explanations with the current SimTeG-finetuned RoBERTa line before training
  the standard RGCN graph detector on the canonical split.
- Baseline for comparison:
  current high-base frozen SimTeG semantic encoder line
  (`Acc=0.863905325443787`, `Macro-F1=0.862442357206568`)
- Explain model:
  `/root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct`
- Explanation encoder:
  `/root/workspace/LMbot/hf_models/roberta_finetuned_simteg_twibot20_seed1`
- Required LM checkpoint:
  `/root/workspace/LMbot/TwiBot-20_seed_1/checkpoints/LM_pretrain/best.pkl`
- Planned full-graph cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_tape_fullgraph_seed1.pt`
- Planned full-graph cache manifest:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_tape_fullgraph_seed1_manifest.json`
- Planned explanation sidecar dir:
  `/root/workspace/LMbot/datasets/TwiBot-20/prompt_expert_v2_tape_fullgraph_seed1`
- Planned artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_encoder__tape_explain_roberta_finetuned_gnn__seed1`
- Planned server log:
  `/root/workspace/LMbot/LLMbot/server_logs/semantic_encoder__tape_explain_roberta_finetuned_gnn__seed1.log`
- Planned commands:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u precompute.py \
  --dataset TwiBot-20 \
  --prompt_mode expert_concat_v1 \
  --prompt_family_version v2 \
  --explain_model_path /root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct \
  --model_path /root/workspace/LMbot/hf_models/roberta_finetuned_simteg_twibot20_seed1 \
  --finetuned_roberta_checkpoint_path /root/workspace/LMbot/TwiBot-20_seed_1/checkpoints/LM_pretrain/best.pkl \
  --explain_required \
  --explain_batch_size 2 \
  --explain_log_every 64 \
  --max_length_ego 512 \
  --max_length_hop 512 \
  --batch_size 8 \
  --device cuda \
  --output_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_tape_fullgraph_seed1.pt \
  --explain_component_cache_dir /root/workspace/LMbot/datasets/TwiBot-20/prompt_expert_v2_tape_fullgraph_seed1 \
  --overwrite \
  --disable_wandb

CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task graph_detector_prepare \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_tape_fullgraph_seed1.pt \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/semantic_encoder__tape_explain_roberta_finetuned_gnn__seed1 \
  --graph_detector_epochs 200 \
  --seeds 1 \
  --device 0 \
  --disable_wandb \
  --force_retrain_backbone
```

- Success criteria:
  produce a full-graph explanation-first semantic cache and a canonical
  `graph_detector_prepare` run with manifest, selection metrics, test metrics,
  and outputs under the stable artifact root.

### semantic_encoder__tape_explain_roberta_base_gnn__seed1

- Status: planned for server launch
- Scope: full TwiBot-20 graph-detector ablation using TAPE-style
  `explanation -> encoder embedding` semantic features with raw pretrained
  `roberta-base`
- Claim boundary:
  this is a TAPE-style explanation-consumption ablation, not an official TAPE
  reproduction. It isolates whether the downstream explanation encoder must be
  task-aligned/finetuned by replacing the finetuned SimTeG RoBERTa encoder with
  raw pretrained `roberta-base`, while keeping the explanation-generation stage
  and graph-detector training protocol fixed.
- Baseline for comparison:
  - high-base frozen SimTeG:
    `Acc=0.863905325443787`, `Macro-F1=0.862442357206568`
  - paired TAPE-style finetuned-RoBERTa run:
    `semantic_encoder__tape_explain_roberta_finetuned_gnn__seed1`
- Explain model:
  `/root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct`
- Explanation encoder:
  `/root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b`
- Planned full-graph cache:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_base_tape_fullgraph_seed1.pt`
- Planned full-graph cache manifest:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_base_tape_fullgraph_seed1_manifest.json`
- Planned explanation sidecar dir:
  `/root/workspace/LMbot/datasets/TwiBot-20/prompt_expert_v2_tape_fullgraph_roberta_base_seed1`
- Planned artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_encoder__tape_explain_roberta_base_gnn__seed1`
- Planned server log:
  `/root/workspace/LMbot/LLMbot/server_logs/semantic_encoder__tape_explain_roberta_base_gnn__seed1.log`
- Planned commands:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u precompute.py \
  --dataset TwiBot-20 \
  --prompt_mode expert_concat_v1 \
  --prompt_family_version v2 \
  --explain_model_path /root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct \
  --model_path /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --explain_required \
  --explain_batch_size 2 \
  --explain_log_every 64 \
  --max_length_ego 512 \
  --max_length_hop 512 \
  --batch_size 8 \
  --device cuda \
  --output_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_base_tape_fullgraph_seed1.pt \
  --explain_component_cache_dir /root/workspace/LMbot/datasets/TwiBot-20/prompt_expert_v2_tape_fullgraph_roberta_base_seed1 \
  --overwrite \
  --disable_wandb

CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task graph_detector_prepare \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_base_tape_fullgraph_seed1.pt \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/semantic_encoder__tape_explain_roberta_base_gnn__seed1 \
  --graph_detector_epochs 200 \
  --seeds 1 \
  --device 0 \
  --disable_wandb \
  --force_retrain_backbone
```

- Success criteria:
  produce a full-graph explanation-first semantic cache and canonical
  `graph_detector_prepare` outputs that can be compared directly against the
  finetuned-RoBERTa TAPE-style run and the existing raw-text
  `semantic_encoder__raw_roberta_base_gnn__seed1` ablation.

### mpe__ultratag_propagated_follower_triplet__highbase__seed1

- Status: completed on server.
- Scope: legacy mean-propagation routed-node validation over the fixed
  high-base routed set. This is not the paper-aligned UltraTAG-S migration: it
  mean-propagates existing prompt-expert embeddings over one graph hop at
  runtime, then feeds
  `[z_gnn || ultratag_graph_follower || ultratag_tweet || ultratag_conflict ||
  structural_side_channel]` into the same no-projector routed-node MLP family as
  `raw_concat_follower_triplet`.
- Claim boundary:
  this is a negative/legacy mean-propagation ablation. It does not regenerate
  LLM text, run PageRank-style node selection, reconfigure edges, or retrain a
  full-graph LM/GNN over propagated text. Do not cite it as an UltraTAG-S
  reproduction or as the active paper-style subgraph adaptation.
- Baseline:
  high-base frozen SimTeG seed 1,
  `Acc=0.863905325443787`, `Macro-F1=0.862442357206568`.
- Reused semantic inputs:
  - backbone embedding:
    `/root/workspace/LMbot/TwiBot-20_seed_1/intermediate/LM/embeddings_iter_-1.pt`
  - high-base frozen root:
    `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_roberta_20260527/seed_1`
  - frozen-router reuse root:
    `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement`
  - routed prompt-expert cache:
    `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__ultratag_propagated_follower_triplet__highbase__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__ultratag_propagated_follower_triplet__highbase__seed1/seed_1/stages/joint_router_refinement`
- Server log:
  `/root/workspace/LMbot/LLMbot/server_logs/mpe__ultratag_propagated_follower_triplet__highbase__seed1.log`
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path /root/workspace/LMbot/TwiBot-20_seed_1/intermediate/LM/embeddings_iter_-1.pt \
  --external_frozen_g0_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_roberta_20260527/seed_1 \
  --joint_routing_protocol frozen_router_reuse \
  --joint_router_reuse_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_graph_following_20260527/seed_1/stages/joint_router_refinement \
  --joint_refiner_embedding_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt \
  --joint_prompt_expert_fusion ultratag_propagated_follower_triplet \
  --risk_budgets 0.05,0.10,0.15,0.20,0.25 \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/mpe__ultratag_propagated_follower_triplet__highbase__seed1 \
  --seeds 1 \
  --device 0 \
  --disable_wandb
```

- Completed artifacts:
  - `manifest.json`: present
  - `metrics.json`: present,
    `joint_prompt_expert_fusion=ultratag_propagated_follower_triplet`
  - `analysis_summary.json`: present
  - `outputs.pt`: present
  - `checkpoint.pt`: present
  - `per_node_test.jsonl`: present
- Result:
  - final test `Acc / Macro-F1`:
    `0.8647506339814032 / 0.8634146341463415`
  - delta vs high-base SimTeG Macro-F1: `+0.0009722769397734199`
  - fix / break / net: `6 / 5 / +1`
  - wrong-node fix rate: `0.037267080745341616`
  - correct-node break rate: `0.004892367906066536`
  - selected budget / beta: `0.2 / 0.2`
  - routed test nodes: `237`
  - routed wrong precision:
    `0.3881856540084388`
  - conditional fix rate on selected wrong:
    `0.06521739130434782`
- Comparison:
  - high-base frozen SimTeG:
    `Acc=0.863905325443787`, `Macro-F1=0.862442357206568`,
    `fix/break/net=0/0/0`
  - `raw_concat_follower_triplet` anchor:
    `Acc=0.8689771766694844`, `Macro-F1=0.867375626451534`,
    `fix/break/net=14/8/+6`
  - `metades_selector_per_action_calibrated`:
    `fix/break/net=4/1/+3`
  - legacy mean-propagation ablation:
    `Acc=0.8647506339814032`, `Macro-F1=0.8634146341463415`,
    `fix/break/net=6/5/+1`
- Summary:
  the legacy mean-propagation runtime path is technically integrated and
  yields a small positive gain over frozen SimTeG on test routed nodes, but it
  does not beat the current strongest no-projector anchor. Under the current
  routed-only cache, propagation reduces break relative to the raw follower
  triplet but loses too much fix capacity, so this result argues that one-hop
  embedding propagation is not the missing ingredient unless a future run
  builds true full-graph expert/evidence embeddings.

### mpe__ultratag_s_subgraph_test_routed__highbase__seed1

- Status: completed on server.
- Scope: UltraTAG-S test-routed subgraph adaptation only.
- Claim boundary:
  this run migrates the UltraTAG-S data/text/structure augmentation modules to
  the fixed high-base test routed-node slice. It performs target-plus-neighbor
  text propagation, LLM summary/keywords/soft-label augmentation,
  same-soft-label cosine virtual edges, PageRank-selected LLM edge
  reconfiguration, finetuned-RoBERTa row replacement, and augmented-edge
  replay in `graph_detector_prepare`. It is still not a full UltraTAG-S
  dual-GNN graph-structure-learning reproduction and does not generate train or
  validation routed-node augmentations.
- Baseline:
  high-base frozen SimTeG seed 1,
  `Acc=0.863905325443787`, `Macro-F1=0.862442357206568`.
- Reused inputs:
  - base embeddings:
    `/root/workspace/LMbot/TwiBot-20_seed_1/intermediate/LM/embeddings_iter_-1.pt`
  - high-base routed nodes:
    `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
  - Qwen instruct model:
    `/root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct`
  - finetuned RoBERTa model:
    `/root/workspace/LMbot/hf_models/roberta_finetuned_simteg_twibot20_seed1`
  - SimTeG checkpoint:
    `/root/workspace/LMbot/TwiBot-20_seed_1/checkpoints/LM_pretrain/best.pkl`
- Precompute output:
  `/root/workspace/LMbot/datasets/TwiBot-20/ultratag_s_subgraph_test_routed_seed1.pt`
- Precompute manifest:
  `/root/workspace/LMbot/datasets/TwiBot-20/ultratag_s_subgraph_test_routed_seed1_manifest.json`
- Augmented graph:
  - edge index:
    `/root/workspace/LMbot/datasets/TwiBot-20/ultratag_s_subgraph_test_routed_seed1_edge_index.pt`
  - edge type:
    `/root/workspace/LMbot/datasets/TwiBot-20/ultratag_s_subgraph_test_routed_seed1_edge_type.pt`
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__ultratag_s_subgraph_test_routed__highbase__seed1`
- Server log:
  `/root/workspace/LMbot/LLMbot/server_logs/mpe__ultratag_s_subgraph_test_routed__highbase__seed1.log`
- Graph replay log:
  `/root/workspace/LMbot/LLMbot/server_logs/mpe__ultratag_s_subgraph_test_routed__highbase__seed1_graph_prepare.log`
- Precompute command:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u precompute.py \
  --dataset TwiBot-20 \
  --prompt_mode ultratag_s_subgraph_v1 \
  --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json \
  --routed_nodes_split test \
  --ultratag_base_embedding_path /root/workspace/LMbot/TwiBot-20_seed_1/intermediate/LM/embeddings_iter_-1.pt \
  --model_path /root/workspace/LMbot/hf_models/roberta_finetuned_simteg_twibot20_seed1 \
  --finetuned_roberta_checkpoint_path /root/workspace/LMbot/TwiBot-20_seed_1/checkpoints/LM_pretrain/best.pkl \
  --explain_model_path /root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct \
  --explain_required \
  --explain_batch_size 2 \
  --explain_max_new_tokens 96 \
  --max_length_hop 512 \
  --device cuda \
  --output_path /root/workspace/LMbot/datasets/TwiBot-20/ultratag_s_subgraph_test_routed_seed1.pt \
  --overwrite \
  --disable_wandb
```

- Graph replay command:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task graph_detector_prepare \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path /root/workspace/LMbot/datasets/TwiBot-20/ultratag_s_subgraph_test_routed_seed1.pt \
  --external_graph_edge_index_path /root/workspace/LMbot/datasets/TwiBot-20/ultratag_s_subgraph_test_routed_seed1_edge_index.pt \
  --external_graph_edge_type_path /root/workspace/LMbot/datasets/TwiBot-20/ultratag_s_subgraph_test_routed_seed1_edge_type.pt \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/mpe__ultratag_s_subgraph_test_routed__highbase__seed1 \
  --force_retrain_backbone \
  --seeds 1 \
  --device 0 \
  --disable_wandb
```

- Completed artifacts:
  - precompute cache:
    `/root/workspace/LMbot/datasets/TwiBot-20/ultratag_s_subgraph_test_routed_seed1.pt`
  - precompute manifest:
    `/root/workspace/LMbot/datasets/TwiBot-20/ultratag_s_subgraph_test_routed_seed1_manifest.json`
  - precompute sidecars:
    `_prompts.jsonl`, `_augmentations.jsonl`, `_virtual_edges.jsonl`,
    `_edge_reconfig_decisions.jsonl`, `_summary.jsonl`, `_keywords.jsonl`,
    `_soft_label.jsonl`, `_edge_reconfig.jsonl`
  - graph detector stage:
    `/root/workspace/LMbot/LLMbot/experiments/mpe__ultratag_s_subgraph_test_routed__highbase__seed1/seed_1/preparation/graph_detector`
  - graph detector artifacts:
    `manifest.json`, `selection_metrics.json`, `outputs.pt`, `checkpoint.pt`,
    `external_edge_index.pt`, `external_edge_type.pt`,
    `ultratag_s_test_slice_metrics.json`
- Precompute manifest summary:
  - target nodes: `296` test routed nodes
  - generation mode: LLM generation, no fallback
  - generated rows:
    `summary=296`, `keywords=296`, `soft_label=296`, `edge_reconfig=435`
  - PageRank-selected nodes: `30`
  - original / added / augmented edge count:
    `16908 / 46621 / 63529`
  - embedding encoder:
    `roberta_finetuned`
  - checkpoint:
    `/root/workspace/LMbot/TwiBot-20_seed_1/checkpoints/LM_pretrain/best.pkl`
- Graph replay manifest summary:
  - feature path:
    `/root/workspace/LMbot/datasets/TwiBot-20/ultratag_s_subgraph_test_routed_seed1.pt`
  - graph override:
    `external_graph_override`
  - external edge count: `63529`
  - validation-selected epoch / validation Acc / validation Macro-F1:
    `102 / 0.8570824524312897 / 0.8540117058004492`
- Test results vs canonical high-base frozen SimTeG:

| split | n | base Acc | base Macro-F1 | UltraTAG-S Acc | UltraTAG-S Macro-F1 | fix | break | net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all test | 1183 | 0.8639 | 0.8624 | 0.8385 | 0.8384 | 57 | 87 | -30 |
| routed test | 296 | 0.6385 | 0.6292 | 0.5372 | 0.5281 | 57 | 87 | -30 |
| non-routed test | 887 | 0.9391 | 0.9387 | 0.9391 | 0.9387 | 0 | 0 | 0 |

- Routed-slice diagnostics:
  - changed routed predictions: `144 / 296`
  - base wrong / correct on routed test: `107 / 189`
  - wrong-node fix rate: `57 / 107 = 0.5327`
  - correct-node break rate: `87 / 189 = 0.4603`
- Summary:
  the test-routed UltraTAG-S subgraph adaptation is now implemented and
  executed with the paper-aligned augmentation modules that are feasible in the
  fixed test-routed setting. It does not improve the high-base frozen SimTeG
  detector: all gains and harms are confined to the 296 routed test nodes, and
  the high fix count is outweighed by a larger break count. This supports the
  existing diagnosis that all-change routed-node representation replacement is
  unsafe without a keep/change or correction-utility policy.

### mpe__ultratag_full_graph_text_only__highbase__seed1

- Status: stopped / incomplete.
- Server process:
  - stopped precompute PID: `5046`
  - log:
    `/root/workspace/LMbot/LLMbot/server_logs/mpe__ultratag_full_graph_text_only__highbase__seed1_precompute.log`
  - partial sidecar:
    `/root/workspace/LMbot/datasets/TwiBot-20/ultratag_full_graph_text_only_seed1_summary.jsonl`
    with about `106` summary rows before interruption
- Scope: superseded full-graph-target UltraTAG text-augmentation diagnostic for
  the high-base frozen SimTeG seed 1 line.
- Claim boundary:
  this run was stopped because the active experiment scope was corrected to
  `test split routed_nodes` as target nodes while allowing neighbor context from
  the full support graph. The partial sidecar is an interrupted artifact and is
  not used as a claim-grade result.
- Baseline:
  high-base frozen SimTeG seed 1,
  `Acc=0.863905325443787`, `Macro-F1=0.862442357206568`.
- Intended precompute output:
  `/root/workspace/LMbot/datasets/TwiBot-20/ultratag_full_graph_text_only_seed1.pt`
- Intended artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__ultratag_full_graph_text_only__highbase__seed1`
- Precompute command:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u precompute.py \
  --dataset TwiBot-20 \
  --prompt_mode ultratag_s_subgraph_v1 \
  --center_node_scope all_graph_nodes \
  --ultratag_base_embedding_path /root/workspace/LMbot/TwiBot-20_seed_1/intermediate/LM/embeddings_iter_-1.pt \
  --model_path /root/workspace/LMbot/hf_models/roberta_finetuned_simteg_twibot20_seed1 \
  --finetuned_roberta_checkpoint_path /root/workspace/LMbot/TwiBot-20_seed_1/checkpoints/LM_pretrain/best.pkl \
  --explain_model_path /root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct \
  --explain_required \
  --explain_batch_size 2 \
  --explain_max_new_tokens 96 \
  --max_length_hop 512 \
  --ultratag_virtual_edge_policy none \
  --ultratag_edge_reconfig false \
  --device cuda \
  --output_path /root/workspace/LMbot/datasets/TwiBot-20/ultratag_full_graph_text_only_seed1.pt \
  --overwrite \
  --disable_wandb
```

- Graph replay command after precompute completion:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task graph_detector_prepare \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path /root/workspace/LMbot/datasets/TwiBot-20/ultratag_full_graph_text_only_seed1.pt \
  --external_graph_edge_index_path /root/workspace/LMbot/datasets/TwiBot-20/ultratag_full_graph_text_only_seed1_edge_index.pt \
  --external_graph_edge_type_path /root/workspace/LMbot/datasets/TwiBot-20/ultratag_full_graph_text_only_seed1_edge_type.pt \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/mpe__ultratag_full_graph_text_only__highbase__seed1 \
  --force_retrain_backbone \
  --seeds 1 \
  --device 0 \
  --disable_wandb
```

- Expected manifest checks:
  - `target_node_scope=all_graph_nodes`
  - `payload_row_layout=full_graph_augmented_embeddings`
  - `virtual_edge_policy=none`
  - `edge_reconfig_enabled=false`
  - `added_edge_count=0`
- Rationale:
  the prior test-routed UltraTAG-S adaptation added `46621` edges and used
  noisy soft labels as graph supervision, producing `fix=57`, `break=87`,
  `net=-30`. The corrected follow-up keeps the routed-test target slice fixed,
  disables graph rewiring, and uses full-graph data only as neighbor context.

### mpe__ultratag_routed_test_fullgraph_context_text_only__highbase__seed1

- Status: completed
- Scope:
  test split routed-node UltraTAG text augmentation with full-graph neighbor
  evidence. The evaluated and replaced target nodes remain the high-base
  `test` routed nodes only; following/follower cards are selected from the full
  support graph through `--context_graph_variant full_graph_support`.
- Claim boundary:
  this is a bounded UltraTAG-S adaptation, not the full UltraTAG-S dual-GNN
  structure-learning reproduction. It disables soft-label virtual edges and LLM
  edge reconfiguration, so `soft_label` is generated as text evidence only and
  is not used to build graph edges. The downstream graph replay consumes the
  original labeled graph plus target-row text replacement.
- Baseline:
  high-base frozen SimTeG seed 1,
  `Acc=0.863905325443787`, `Macro-F1=0.862442357206568`.
- Stable precompute output:
  `/root/workspace/LMbot/datasets/TwiBot-20/ultratag_routed_test_fullgraph_context_text_only_seed1.pt`
- Stable artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__ultratag_routed_test_fullgraph_context_text_only__highbase__seed1`
- Stable precompute log:
  `/root/workspace/LMbot/LLMbot/server_logs/mpe__ultratag_routed_test_fullgraph_context_text_only__highbase__seed1_precompute.log`
- Stable graph replay log:
  `/root/workspace/LMbot/LLMbot/server_logs/mpe__ultratag_routed_test_fullgraph_context_text_only__highbase__seed1_graph_prepare.log`
- Completed artifacts:
  - precompute manifest:
    `/root/workspace/LMbot/datasets/TwiBot-20/ultratag_routed_test_fullgraph_context_text_only_seed1_manifest.json`
  - precompute cache:
    `/root/workspace/LMbot/datasets/TwiBot-20/ultratag_routed_test_fullgraph_context_text_only_seed1.pt`
  - precompute edge files:
    `/root/workspace/LMbot/datasets/TwiBot-20/ultratag_routed_test_fullgraph_context_text_only_seed1_edge_index.pt`,
    `/root/workspace/LMbot/datasets/TwiBot-20/ultratag_routed_test_fullgraph_context_text_only_seed1_edge_type.pt`
  - graph detector stage:
    `/root/workspace/LMbot/LLMbot/experiments/mpe__ultratag_routed_test_fullgraph_context_text_only__highbase__seed1/seed_1/preparation/graph_detector`
  - graph detector artifacts:
    `manifest.json`, `selection_metrics.json`, `outputs.pt`, `checkpoint.pt`,
    `external_edge_index.pt`, `external_edge_type.pt`,
    `routed_test_comparison_metrics.json`,
    `routed_test_comparison_per_node.jsonl`
- Precompute command:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u precompute.py \
  --dataset TwiBot-20 \
  --prompt_mode ultratag_s_subgraph_v1 \
  --graph_data_variant labeled \
  --context_graph_variant full_graph_support \
  --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json \
  --routed_nodes_split test \
  --ultratag_base_embedding_path /root/workspace/LMbot/TwiBot-20_seed_1/intermediate/LM/embeddings_iter_-1.pt \
  --model_path /root/workspace/LMbot/hf_models/roberta_finetuned_simteg_twibot20_seed1 \
  --finetuned_roberta_checkpoint_path /root/workspace/LMbot/TwiBot-20_seed_1/checkpoints/LM_pretrain/best.pkl \
  --explain_model_path /root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct \
  --explain_required \
  --explain_batch_size 2 \
  --explain_max_new_tokens 96 \
  --max_length_hop 512 \
  --ultratag_virtual_edge_policy none \
  --ultratag_edge_reconfig false \
  --device cuda \
  --output_path /root/workspace/LMbot/datasets/TwiBot-20/ultratag_routed_test_fullgraph_context_text_only_seed1.pt \
  --overwrite \
  --disable_wandb
```

- Graph replay command after precompute completion:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task graph_detector_prepare \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --embedding_path /root/workspace/LMbot/datasets/TwiBot-20/ultratag_routed_test_fullgraph_context_text_only_seed1.pt \
  --external_graph_edge_index_path /root/workspace/LMbot/datasets/TwiBot-20/ultratag_routed_test_fullgraph_context_text_only_seed1_edge_index.pt \
  --external_graph_edge_type_path /root/workspace/LMbot/datasets/TwiBot-20/ultratag_routed_test_fullgraph_context_text_only_seed1_edge_type.pt \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/mpe__ultratag_routed_test_fullgraph_context_text_only__highbase__seed1 \
  --force_retrain_backbone \
  --seeds 1 \
  --device 0 \
  --disable_wandb
```

- Validated precompute manifest checks:
  - `target_node_scope=explicit_routed_nodes`
  - `routed_nodes_split=test`
  - `output_graph_variant=labeled`
  - `context_graph_variant=full_graph_support`
  - `payload_row_layout=labeled_base_embedding_with_target_rows_replaced`
  - `virtual_edge_policy=none`
  - `edge_reconfig_enabled=false`
  - `added_edge_count=0`
  - `context_edge_count=227979`
  - payload `embeddings` shape: `(11826, 768)`
  - emitted labeled edge count / max node id: `16908 / 11825`
- Graph replay selection metrics:
  - validation Acc / Macro-F1:
    `0.8570824524312897 / 0.8533965447676337`
- Test results vs canonical high-base frozen SimTeG:

| split | n | base Acc | base Macro-F1 | method Acc | method Macro-F1 | fix | break | net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all test | 1183 | 0.8639 | 0.8624 | 0.8225 | 0.8223 | 62 | 112 | -50 |
| routed test | 296 | 0.6385 | 0.6292 | 0.4696 | 0.3828 | 62 | 112 | -50 |

- Routed-test diagnostics:
  - routed test base wrong / correct:
    `107 / 189`
  - changed routed predictions:
    `174 / 296`
  - wrong-node fix rate:
    `62 / 107 = 0.5794392523364486`
  - correct-node break rate:
    `112 / 189 = 0.5925925925925926`
  - outcome counts:
    `both_correct=77`, `both_wrong=45`, `fix=62`, `break=112`
  - transition counts `(label, base_pred, method_pred)`:
    `(0,0,0)=68`, `(0,0,1)=3`, `(0,1,0)=57`, `(0,1,1)=5`,
    `(1,0,0)=40`, `(1,0,1)=5`, `(1,1,0)=109`, `(1,1,1)=9`
- Rationale:
  this run isolates whether better full-graph neighbor evidence helps the
  routed-node text replacement path without repeating the prior failure mode
  where unreliable soft labels and dense virtual edges rewrote the routed
  subgraph.
- Summary:
  using full-graph neighbor evidence while keeping the target slice fixed to
  test routed nodes and disabling graph rewiring still does not improve the
  high-base frozen SimTeG detector. The method recovers many base-wrong routed
  nodes (`62` fixes), but it breaks more base-correct routed nodes (`112`),
  producing `net=-50` and lowering full-test Macro-F1 by about `0.0401`. This
  confirms that removing soft-label virtual edges and LLM edge reconfiguration
  avoids the previous subgraph-rewrite failure mode, but text replacement alone
  remains unsafe without a keep/change or correction-utility policy.

### mpe__norm_text_direct_predictor_testonly__highbase__seed1

- Status: completed
- Scope: direct LLM-as-predictor diagnostic on the routed test-only slice
- Claim boundary:
  this run does not train a classifier, refiner, or backbone. It directly
  prompts the local Qwen2.5 instruct model with each routed test node's raw
  `norm_user_text` and asks for a `bot/human` decision plus explanation, then
  replays those decisions onto the frozen high-base test outputs for a
  test-only diagnostic.
- Baseline: canonical high-base frozen SimTeG seed 1
  - Acc: `0.863905325443787`
  - Macro-F1: `0.862442357206568`
- Inputs:
  - routed test-only nodes:
    `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529_test_only.json`
  - raw node text:
    `/root/workspace/LMbot/datasets/TwiBot-20/norm_user_text.json`
  - labels:
    `/root/workspace/LMbot/datasets/TwiBot-20/labels.pt`
  - base outputs:
    `/root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_classifier_qwen3_20260601/base_outputs_for_routed_classifier.pt`
  - Qwen instruct model:
    `/root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct`
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__norm_text_direct_predictor_testonly__highbase__seed1/seed_1`
- Server log:
  `/root/workspace/LMbot/LLMbot/server_logs/mpe__norm_text_direct_predictor_testonly__highbase__seed1.log`
- Expected artifacts:
  - manifest:
    `/root/workspace/LMbot/LLMbot/experiments/mpe__norm_text_direct_predictor_testonly__highbase__seed1/seed_1/manifest.json`
  - metrics:
    `/root/workspace/LMbot/LLMbot/experiments/mpe__norm_text_direct_predictor_testonly__highbase__seed1/seed_1/metrics.json`
  - per-node:
    `/root/workspace/LMbot/LLMbot/experiments/mpe__norm_text_direct_predictor_testonly__highbase__seed1/seed_1/per_node_test.jsonl`
  - outputs:
    `/root/workspace/LMbot/LLMbot/experiments/mpe__norm_text_direct_predictor_testonly__highbase__seed1/seed_1/outputs.pt`
- Result:
  - Base routed-test Acc / Macro-F1:
    `0.6385135135135135 / 0.6291638858641564`
  - Direct-predictor routed-test Acc / Macro-F1:
    `0.5202702702702703 / 0.5094304388422035`
  - fix / break / net:
    `54 / 89 / -35`
  - wrong-node fix rate / correct-node break rate:
    `0.5046728971962616 / 0.4708994708994709`
  - changed prediction count:
    `143`
  - changed wrong precision:
    `0.3776223776223776`
  - parse OK / fallback:
    `289 / 7`
- Summary:
  Directly prompting Qwen2.5 on routed test nodes' raw `norm_user_text` can
  recover many frozen-base wrong nodes (`54` fixes), but it breaks far more
  routed correct nodes (`89`). This makes the direct-predictor path clearly
  negative under the high-base frozen SimTeG setting, even before any
  explanation embedding or backbone replacement is involved. The result
  supports treating LLM text judgments as unstable replacement decisions on the
  routed slice, not as a drop-in substitute for the frozen backbone's semantic
  representation.

### mpe__tweet_only_direct_predictor_testonly__highbase__seed1

- Status: completed
- Scope: DGP-style direct LLM-as-predictor diagnostic on the routed test-only
  slice using tweet evidence only
- Claim boundary:
  this run does not train a classifier, refiner, or backbone. It directly
  prompts the local Qwen2.5 instruct model with only the parsed `TWEET:` slice
  from each routed test node and asks for a `bot/human` decision plus
  explanation, then replays those decisions onto the frozen high-base test
  outputs for a test-only diagnostic.
- Baseline: canonical high-base frozen SimTeG seed 1
  - Acc: `0.863905325443787`
  - Macro-F1: `0.862442357206568`
- Inputs:
  - routed test-only nodes:
    `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529_test_only.json`
  - raw node text:
    `/root/workspace/LMbot/datasets/TwiBot-20/norm_user_text.json`
  - labels:
    `/root/workspace/LMbot/datasets/TwiBot-20/labels.pt`
  - base outputs:
    `/root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_classifier_qwen3_20260601/base_outputs_for_routed_classifier.pt`
  - Qwen instruct model:
    `/root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct`
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__tweet_only_direct_predictor_testonly__highbase__seed1/seed_1`
- Result:
  - Base routed-test Acc / Macro-F1:
    `0.6385135135135135 / 0.6291638858641564`
  - Direct-predictor routed-test Acc / Macro-F1:
    `0.5135135135135135 / 0.5044642857142857`
  - fix / break / net:
    `62 / 99 / -37`
  - wrong-node fix rate / correct-node break rate:
    `0.5794392523364486 / 0.5238095238095238`
  - changed prediction count:
    `161`
  - parse OK / fallback:
    `295 / 1`
- Summary:
  Restricting the LLM to tweet evidence only increases routed wrong-node
  recovery relative to the raw `norm_user_text` direct-predictor baseline, but
  it also increases break even more. This indicates that tweet-only prompting
  makes the LLM more willing to override the frozen base without enough
  counter-balancing profile constraints.

### mpe__tweet_plus_basic_metadata_direct_predictor_testonly__highbase__seed1

- Status: completed
- Scope: DGP-style direct LLM-as-predictor diagnostic on the routed test-only
  slice using parsed tweets plus human-readable basic metadata
- Claim boundary:
  this run does not train a classifier, refiner, or backbone. It directly
  prompts the local Qwen2.5 instruct model with parsed tweet evidence plus a
  compact field-named profile card and asks for a `bot/human` decision plus
  explanation, then replays those decisions onto the frozen high-base test
  outputs for a test-only diagnostic.
- Baseline: canonical high-base frozen SimTeG seed 1
  - Acc: `0.863905325443787`
  - Macro-F1: `0.862442357206568`
- Inputs:
  - routed test-only nodes:
    `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529_test_only.json`
  - raw node text:
    `/root/workspace/LMbot/datasets/TwiBot-20/norm_user_text.json`
  - labels:
    `/root/workspace/LMbot/datasets/TwiBot-20/labels.pt`
  - base outputs:
    `/root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_classifier_qwen3_20260601/base_outputs_for_routed_classifier.pt`
  - Qwen instruct model:
    `/root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct`
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__tweet_plus_basic_metadata_direct_predictor_testonly__highbase__seed1/seed_1`
- Result:
  - Base routed-test Acc / Macro-F1:
    `0.6385135135135135 / 0.6291638858641564`
  - Direct-predictor routed-test Acc / Macro-F1:
    `0.5337837837837838 / 0.5259713131875783`
  - fix / break / net:
    `58 / 89 / -31`
  - wrong-node fix rate / correct-node break rate:
    `0.5420560747663551 / 0.4708994708994709`
  - changed prediction count:
    `147`
  - parse OK / fallback:
    `295 / 1`
- Summary:
  Adding field-named basic metadata to the tweet-only prompt improves over both
  the tweet-only and raw `norm_user_text` direct-predictor baselines, which
  supports the diagnosis that the original raw `norm_user_text` serialization
  is not prompt-friendly for LLM judgment. However, the run remains strongly
  negative on net gain, so better prompt structuring alone is not enough to
  make direct routed-node replacement safe under the high-base frozen SimTeG
  setting.

### mpe__raw_concat_follower_tweet__highbase_jointtrain__seed1

- Status: completed on server
- Scope: full joint-training GLANCE refiner ablation
- Claim boundary:
  this run keeps the public `joint_router_refinement` protocol, trains the
  router and routed-node refiner jointly under the validation-selected global
  budget protocol, reuses the canonical high-base frozen SimTeG backbone
  through `--external_frozen_g0_root`, and changes only the prompt-expert
  refiner input surface to the narrow no-projector
  `[z_gnn || graph_follower || tweet || structural_side_channel]` path.
  It is not a backbone change, prompt-cache rebuild, or frozen-router-reuse
  comparison.
- Baseline:
  canonical high-base frozen SimTeG seed 1
  - Acc: `0.863905325443787`
  - Macro-F1: `0.862442357206568`
- Reused inputs:
  - external frozen g0 root:
    `/root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_roberta_20260527/seed_1`
  - routed prompt-expert cache:
    `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt`
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_follower_tweet__highbase_jointtrain__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_follower_tweet__highbase_jointtrain__seed1/seed_1/stages/joint_router_refinement`
- Server log:
  `/root/workspace/LMbot/LLMbot/server_logs/mpe__raw_concat_follower_tweet__highbase_jointtrain__seed1.log`
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task joint_router_refinement \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --use_GNN \
  --graph_backbone rgcn \
  --external_frozen_g0_root /root/workspace/LMbot/LLMbot/server_prompt_expert_highbase_preiter_roberta_20260527/seed_1 \
  --joint_refiner_embedding_path /root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt \
  --joint_prompt_expert_fusion raw_concat_follower_tweet \
  --risk_budgets 0.05,0.10,0.15,0.20,0.25 \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_follower_tweet__highbase_jointtrain__seed1 \
  --seeds 1 \
  --device 0 \
  --disable_wandb
```

- Completed artifacts:
  - `manifest.json`: present
  - `metrics.json`: present
  - `analysis_summary.json`: present
  - `outputs.pt`: present
  - `checkpoint.pt`: present
  - `per_node_test.jsonl`: present
  - router summaries:
    `router_seed_summary.json`, `router_seed_summary.csv`,
    `router_budget_curves_all_seeds.csv`, `router_epoch_curves_all_seeds.csv`
- Joint-training result:
  - routing protocol: `joint_train`
  - prompt fusion: `raw_concat_follower_tweet`
  - selected budget / beta: `0.25 / 0.3`
  - query rate: `0.2502`
  - overall test `Acc / Macro-F1`:
    `0.8605240912933221 / 0.8593518635274621`
  - delta vs high-base frozen SimTeG:
    `-0.0033812341504648735 / -0.0030904936791059656`
  - fix / break / net:
    `6 / 10 / -4`
  - wrong-node fix rate / correct-node break rate:
    `0.037267080745341616 / 0.009784735812133072`
  - routed wrong precision / routed wrong coverage:
    `0.3614864864864865 / 0.6645962732919255`
  - conditional fix / break on routed wrong/correct:
    `0.056074766355140186 / 0.05291005291005291`
  - router test AUROC / AUPRC:
    `0.763598351788601 / 0.29737513259889875`
  - advantage-router diagnostic test AUROC / AUPRC:
    `0.6022146507666098 / 0.017528703211556054`
- Comparison:
  - high-base frozen SimTeG:
    `Acc=0.863905325443787`, `Macro-F1=0.862442357206568`
  - fixed-router `raw_concat_follower_triplet` anchor:
    `Acc=0.8689771766694844`, `Macro-F1=0.867375626451534`,
    `fix/break/net=14/8/+6`
- Summary:
  Removing `conflict` from the strongest follower-side raw-concat anchor hurts
  the full joint-training result. The narrower
  `[z_gnn || graph_follower || tweet || structural_side_channel]` refiner still
  finds some real fixes, but break exceeds fix (`6 / 10`), final Macro-F1 drops
  below the frozen SimTeG baseline, and it stays well below the fixed-router
  `raw_concat_follower_triplet` anchor. Under the current high-base setting,
  `conflict` is acting more like a stabilizing calibration channel than a
  disposable extra expert.

### dgp__v2_norm_text_following_summary_qwen25__highbase_routed__seed1

- Status: completed on server
- Scope: DGP-inspired routed-node prompt construction and two downstream
  consumers over the fixed high-base routed union
- Claim boundary:
  this is a DGP-style migration, not a full DGP reproduction. It does not
  perform answer-token generative SFT. The prompt cache uses Qwen2.5-Instruct
  to summarize top-K following neighbor evidence, builds a final
  Yes/No-style routed-node predictor prompt, then evaluates two consumers:
  a Qwen2.5 LoRA semantic predictor and a cached embedding MLP.
- Baseline:
  canonical high-base frozen SimTeG seed 1
  - Acc: `0.863905325443787`
  - Macro-F1: `0.862442357206568`
- Inputs:
  - routed union:
    `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
  - Qwen instruct / embedding model:
    `/root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct`
  - base outputs for replay:
    `/root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_classifier_qwen3_20260601/base_outputs_for_routed_classifier.pt`
- Prompt construction:
  - prompt mode: `dgp_predictor_v2`
  - variant: `norm_text_following_summary`
  - target evidence: LLM-friendly rendering of `norm_user_text` into
    `PROFILE`, `TWEET_BEHAVIOR`, and `TWEET_SAMPLES`
  - neighbor evidence: top-K following neighbors with `K=5`
  - neighbor summary language: English only
  - empty following context policy:
    `deterministic_sparse_summary_no_llm`
- Precompute command:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u precompute.py \
  --dataset TwiBot-20 \
  --prompt_mode dgp_predictor_v2 \
  --dgp_prompt_variant norm_text_following_summary \
  --dgp_neighbor_summary_k 5 \
  --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json \
  --routed_nodes_split all \
  --explain_model_path /root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct \
  --model_path /root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct \
  --explain_required \
  --explain_batch_size 2 \
  --explain_max_input_length 1024 \
  --explain_max_new_tokens 96 \
  --explain_log_every 50 \
  --explain_component_cache_dir /root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v2_norm_text_following_summary_qwen25_component_cache_clean_english \
  --batch_size 2 \
  --max_length_hop 1024 \
  --output_path /root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v2_norm_text_following_summary_qwen25_embed.pt \
  --device cuda \
  --disable_wandb \
  --overwrite
```

- Precompute artifacts:
  - cache:
    `/root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v2_norm_text_following_summary_qwen25_embed.pt`
  - manifest:
    `/root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v2_norm_text_following_summary_qwen25_embed_manifest.json`
  - final prompts:
    `/root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v2_norm_text_following_summary_qwen25_embed_prompts.jsonl`
  - generated summaries sidecar:
    `/root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v2_norm_text_following_summary_qwen25_embed_explanations.jsonl`
  - component cache:
    `/root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v2_norm_text_following_summary_qwen25_component_cache_clean_english`
- Precompute audit:
  - `semantic_view_mode`: `dgp_predictor_v2`
  - `target_node_count`: `1640`
  - `num_nodes / embedding_dim / dtype`: `11826 / 3584 / torch.float16`
  - prompt rows / unique node ids: `1640 / 1640`
  - bad raw markers in final prompts: `0`
  - context-summary sidecar rows: `189`, all `llm_generation`
  - neighbor-summary sidecar rows: `338`, with `333 llm_generation` and
    `5 quality_fallback` rows due to `too_short`

#### dgp__v2_qwen25_peft_predictor_norm_text_following_summary__highbase_routed__seed1

- Status: completed on server
- Stage: `semantic_encoder_finetune`
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/dgp__v2_qwen25_peft_predictor_norm_text_following_summary__highbase_routed__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/dgp__v2_qwen25_peft_predictor_norm_text_following_summary__highbase_routed__seed1/seed_1/preparation/semantic_encoder`
- Server log:
  `/root/workspace/LMbot/LLMbot/server_logs/dgp__v2_qwen25_peft_predictor_norm_text_following_summary__highbase_routed__seed1.log`
- Completed artifacts:
  - `manifest.json`
  - `metrics.json`
  - `outputs.pt`
  - `embeddings.pt`
  - `classifier.pt`
  - `adapter/adapter_model.safetensors`
  - `replay_against_frozen_simteg.json`
  - `per_node_test_replay.jsonl`
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task semantic_encoder_finetune \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --seeds 1 \
  --semantic_encoder qwen3_peft \
  --qwen_model_path /root/workspace/LMbot/hf_models/Qwen2.5-7B-Instruct \
  --semantic_text_source_path /root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v2_norm_text_following_summary_qwen25_embed_prompts.jsonl \
  --semantic_text_field prompt \
  --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json \
  --lm_batch_size 1 \
  --max_length 1024 \
  --semantic_max_steps 0 \
  --LM_pretrain_epochs 1 \
  --peft_rank 8 \
  --peft_alpha 16 \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/dgp__v2_qwen25_peft_predictor_norm_text_following_summary__highbase_routed__seed1 \
  --device 0 \
  --disable_wandb
```

- Routed-node training contract:
  - routed train / valid / test: `752 / 592 / 296`
  - max steps: `752`
  - max length: `1024`
  - trainable LoRA params: `2523136`
  - trainable classifier params: `459138`
- Routed-test classifier result:
  - Acc / Macro-F1: `0.5641891891891891 / 0.47675231243576566`
  - Bot-F1: `0.6906474820143885`
- Frozen SimTeG replay on routed test:
  - full-test Acc / Macro-F1 after routed replacement:
    `0.8453085376162299 / 0.840942403530156`
  - delta vs high-base frozen SimTeG:
    `-0.018596787827557026 / -0.021499953676411998`
  - fix / break / net: `47 / 69 / -22`
  - changed predictions: `116`
  - conditional fix rate on routed wrong:
    `0.4392523364485981`
  - correct-node break rate:
    `0.36507936507936506`
  - routed prediction distribution:
    base `{0: 116, 1: 180}`, method `{0: 42, 1: 254}`

#### calm__dgp_v2_norm_text_following_summary_qwen25__highbase_routed__seed1

- Status: completed on server
- Stage: `semantic_embedding_classifier`
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/calm__dgp_v2_norm_text_following_summary_qwen25__highbase_routed__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/calm__dgp_v2_norm_text_following_summary_qwen25__highbase_routed__seed1/seed_1/preparation/semantic_embedding_classifier`
- Server log:
  `/root/workspace/LMbot/LLMbot/server_logs/calm__dgp_v2_norm_text_following_summary_qwen25__highbase_routed__seed1.log`
- Completed artifacts:
  - `manifest.json`
  - `metrics.json`
  - `outputs.pt`
  - `classifier.pt`
  - `embeddings_ref.pt`
  - `replay_against_frozen_simteg.json`
  - `per_node_test_replay.jsonl`
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=1 /root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task semantic_embedding_classifier \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --seeds 1 \
  --embedding_path /root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v2_norm_text_following_summary_qwen25_embed.pt \
  --semantic_text_source_path /root/workspace/LMbot/datasets/TwiBot-20/dgp_predictor_v2_norm_text_following_summary_qwen25_embed_prompts.jsonl \
  --semantic_text_field prompt \
  --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json \
  --LM_classifier_n_layers 2 \
  --LM_classifier_hidden_dim 128 \
  --dropout 0.4 \
  --LM_pretrain_epochs 5 \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/calm__dgp_v2_norm_text_following_summary_qwen25__highbase_routed__seed1 \
  --device 0 \
  --disable_wandb
```

- Routed-node training contract:
  - routed train / valid / test: `752 / 592 / 296`
  - embedding dim / rows: `3584 / 11826`
  - trainable classifier params: `459394`
- Routed-test classifier result:
  - Acc / Macro-F1: `0.44932432432432434 / 0.31002331002331`
  - Bot-F1: `0.0`
- Frozen SimTeG replay on routed test:
  - full-test Acc / Macro-F1 after routed replacement:
    `0.8165680473372781 / 0.8161260632106724`
  - delta vs high-base frozen SimTeG:
    `-0.047337278106508895 / -0.04631629399589565`
  - fix / break / net: `62 / 118 / -56`
  - changed predictions: `180`
  - conditional fix rate on routed wrong:
    `0.5794392523364486`
  - correct-node break rate:
    `0.6243386243386243`
  - routed prediction distribution:
    base `{0: 116, 1: 180}`, method `{0: 296}`

- Summary:
  Both DGP-v2 consumers have real wrong-node recovery, but neither is safe as
  a direct routed-node replacement under the high-base frozen SimTeG setting.
  The Qwen2.5 PEFT predictor is less collapsed than the embedding MLP, but it
  still breaks more correct nodes than it fixes (`47 / 69 / -22`). The cached
  embedding MLP collapses to predicting every routed-test node as class `0`
  and is strongly negative (`62 / 118 / -56`). This supports keeping DGP-style
  evidence as candidate evidence for a calibrated keep/change refiner rather
  than replacing frozen SimTeG predictions directly.

### semantic_gate__dgp_v2_peft_mlp__highbase_routed__seed1

- Status: completed on server
- Stage: `semantic_correction_gate`
- Scope: base-aware keep/change gate over two fixed DGP-v2 semantic
  candidates:
  - `qwen25_peft`
  - `dgp_embed_mlp`
- Claim boundary:
  this is a routed-node learning-to-defer / selective-correction diagnostic.
  It does not regenerate prompts, update the Qwen PEFT predictor, update the
  embedding MLP, modify the graph, or retrain frozen SimTeG. It trains only a
  lightweight gate over base/candidate probability meta-features and locks the
  accept threshold on routed validation nodes.
- Baseline:
  canonical high-base frozen SimTeG seed 1
  - full-test Acc: `0.863905325443787`
  - full-test Macro-F1: `0.862442357206568`
- Inputs:
  - routed split:
    `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
  - base outputs:
    `/root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_classifier_qwen3_20260601/base_outputs_for_routed_classifier.pt`
  - Qwen2.5 PEFT candidate outputs:
    `/root/workspace/LMbot/LLMbot/experiments/dgp__v2_qwen25_peft_predictor_norm_text_following_summary__highbase_routed__seed1/seed_1/preparation/semantic_encoder/outputs.pt`
  - DGP/CALM embedding-MLP candidate outputs:
    `/root/workspace/LMbot/LLMbot/experiments/calm__dgp_v2_norm_text_following_summary_qwen25__highbase_routed__seed1/seed_1/preparation/semantic_embedding_classifier/outputs.pt`
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_gate__dgp_v2_peft_mlp__highbase_routed__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_gate__dgp_v2_peft_mlp__highbase_routed__seed1/seed_1/preparation/semantic_correction_gate`
- Completed artifacts:
  - `manifest.json`
  - `metrics.json`
  - `outputs.pt`
  - `checkpoint.pt`
  - `per_node_test.jsonl`
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
/root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task semantic_correction_gate \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --seeds 1 \
  --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json \
  --semantic_gate_base_outputs_path /root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_classifier_qwen3_20260601/base_outputs_for_routed_classifier.pt \
  --semantic_gate_candidate_output_paths /root/workspace/LMbot/LLMbot/experiments/dgp__v2_qwen25_peft_predictor_norm_text_following_summary__highbase_routed__seed1/seed_1/preparation/semantic_encoder/outputs.pt,/root/workspace/LMbot/LLMbot/experiments/calm__dgp_v2_norm_text_following_summary_qwen25__highbase_routed__seed1/seed_1/preparation/semantic_embedding_classifier/outputs.pt \
  --semantic_gate_candidate_names qwen25_peft,dgp_embed_mlp \
  --semantic_gate_epochs 200 \
  --semantic_gate_hidden_dim 64 \
  --semantic_gate_learning_rate 0.001 \
  --semantic_gate_weight_decay 0.0001 \
  --semantic_gate_break_weight 2.0 \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/semantic_gate__dgp_v2_peft_mlp__highbase_routed__seed1 \
  --device 0 \
  --disable_wandb
```

- Gate contract:
  - contract: `semantic_correction_gate_v1`
  - candidate names: `qwen25_peft`, `dgp_embed_mlp`
  - validation-locked threshold: `0.16848692297935486`
  - action utility: `base wrong + candidate correct` is positive;
    `base correct + candidate wrong` is a weighted negative with
    `break_weight=2.0`
- Validation split result:
  - base Acc / Macro-F1: `0.6621621621621622 / 0.6530382595648913`
  - gated Acc / Macro-F1: `0.6722972972972973 / 0.668663089262016`
  - fix / break / net: `20 / 14 / +6`
- Routed-test result:
  - base Acc / Macro-F1: `0.6385135135135135 / 0.6291638858641564`
  - gated Acc / Macro-F1: `0.6283783783783784 / 0.6254888428801473`
  - fix / break / net: `9 / 12 / -3`
  - selected actions: `base=275`, `qwen25_peft=3`, `dgp_embed_mlp=18`
  - fixes by action: `qwen25_peft=2`, `dgp_embed_mlp=7`
  - breaks by action: `qwen25_peft=1`, `dgp_embed_mlp=11`
- Canonical full-test replay:
  - base Acc / Macro-F1: `0.863905325443787 / 0.862442357206568`
  - gated Acc / Macro-F1: `0.8613693998309383 / 0.8603514893960071`
  - fix / break / net: `9 / 12 / -3`
- Candidate direct-replacement diagnostics on routed test:
  - `qwen25_peft`: fix / break / net `47 / 69 / -22`
  - `dgp_embed_mlp`: fix / break / net `62 / 118 / -56`
- Summary:
  The gate successfully reduces break compared with direct candidate
  replacement (`69` or `118` breaks down to `12`), and validation improves
  by `+6` net. However, the validation-locked policy does not generalize to
  routed test (`net=-3`) and therefore is not a positive high-base result.
  The evidence supports the research direction of base-aware accept/defer
  correction, but not a claim that the current probability-only gate solves the
  routed-node refiner problem. The next narrow step is to add node attributes,
  graph-density/router-score features, and candidate-specific calibration
  before considering heavier MoE/fusion heads.

### semantic_gate_nodeattr__dgp_v2_peft_mlp__highbase_routed__seed1

- Status: completed on server
- Stage: `semantic_correction_gate`
- Scope: node-attribute competence gate over the same two fixed DGP-v2 semantic
  candidates used by `semantic_gate__dgp_v2_peft_mlp__highbase_routed__seed1`.
- Claim boundary:
  this is a diagnostic node-attribute gate, not a positive main result. It
  appends target-account metadata/tweet cues and labeled-graph attributes to
  the probability-only gate, but keeps the same fixed candidate outputs,
  routed split, frozen SimTeG base, and validation-locked threshold protocol.
- Literature motivation:
  - BotRGCN / BotMoE / TwiBot-20 support profile, tweet, and graph attributes
    as real social-bot evidence channels
  - META-DES supports classifier competence features beyond raw posterior
    scores
  - SelectiveNet / Learning-to-Defer support accept/defer gating rather than
    direct candidate replacement
- Stable routed split:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
- Artifact roots:
  - break weight 2:
    `/root/workspace/LMbot/LLMbot/experiments/semantic_gate_nodeattr__dgp_v2_peft_mlp__highbase_routed__seed1`
  - break weight 4:
    `/root/workspace/LMbot/LLMbot/experiments/semantic_gate_nodeattr_bw4__dgp_v2_peft_mlp__highbase_routed__seed1`
  - break weight 8:
    `/root/workspace/LMbot/LLMbot/experiments/semantic_gate_nodeattr_bw8__dgp_v2_peft_mlp__highbase_routed__seed1`
- Feature contract:
  - `--semantic_gate_feature_family node_attribute`
  - final action feature dim: `52`
  - appended node-attribute dim: `36`
  - graph attribute source: `dataset_labeled_graph`
  - graph attributes available: `true`
  - attribute normalization: train-split z-score with clamp 10
- Command template:

```bash
cd /root/workspace/LMbot/LLMbot
/root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task semantic_correction_gate \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --seeds 1 \
  --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json \
  --semantic_gate_base_outputs_path /root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_classifier_qwen3_20260601/base_outputs_for_routed_classifier.pt \
  --semantic_gate_candidate_output_paths /root/workspace/LMbot/LLMbot/experiments/dgp__v2_qwen25_peft_predictor_norm_text_following_summary__highbase_routed__seed1/seed_1/preparation/semantic_encoder/outputs.pt,/root/workspace/LMbot/LLMbot/experiments/calm__dgp_v2_norm_text_following_summary_qwen25__highbase_routed__seed1/seed_1/preparation/semantic_embedding_classifier/outputs.pt \
  --semantic_gate_candidate_names qwen25_peft,dgp_embed_mlp \
  --semantic_gate_feature_family node_attribute \
  --semantic_gate_break_weight 2.0 \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/semantic_gate_nodeattr__dgp_v2_peft_mlp__highbase_routed__seed1 \
  --device 0 \
  --disable_wandb
```

- Result table:

| run | feature family | break weight | valid fix/break/net | routed-test fix/break/net | full-test Acc | full-test Macro-F1 | selected routed-test actions |
|---|---|---:|---:|---:|---:|---:|---|
| probability baseline | probability | 2 | `20 / 14 / +6` | `9 / 12 / -3` | `0.8613693998309383` | `0.8603514893960071` | base `275`, qwen `3`, mlp `18` |
| nodeattr bw2 | node_attribute | 2 | `42 / 41 / +1` | `12 / 26 / -14` | `0.8520710059171598` | `0.8508277340442779` | base `254`, qwen `16`, mlp `26` |
| nodeattr bw4 | node_attribute | 4 | `4 / 3 / +1` | `2 / 8 / -6` | `0.8588334742180896` | `0.8574629896979364` | base `282`, qwen `2`, mlp `12` |
| nodeattr bw8 | node_attribute | 8 | `3 / 2 / +1` | `1 / 4 / -3` | `0.8613693998309383` | `0.8599523001051015` | base `288`, qwen `1`, mlp `7` |

- Diagnostic finding:
  The node-attribute gate does not improve over the probability-only gate.
  At break weight 2 it becomes too aggressive and chooses more candidate
  actions, but the extra actions break many base-correct nodes. Stronger break
  weights reduce break but also collapse fixes, returning to an almost-abstain
  policy.
- Attribute slice observation:
  On routed test, the nodeattr bw2 changed nodes are below train-average in
  `tweet_count_log1p` and `tweet_char_len_log1p` and above average in graph
  follower/following availability. Break nodes have higher
  `graph_following_log1p`, `graph_has_following`, `graph_has_follower`, and
  `graph_follower_log1p` than fix nodes. This confirms the earlier failure
  diagnosis: node attributes identify the dense/sparse hard-node regime, but
  still do not identify whether a semantic candidate can safely correct the
  node.
- Summary:
  Node attributes alone are not enough as a gate. They provide node-type
  context, but in this routed set the same dense/low-text regimes contain both
  fixable wrong nodes and fragile base-correct nodes. The next method should
  add candidate-specific evidence alignment or validation-calibrated
  per-candidate rules, not only more generic attributes.

### mpe__raw_concat_ego_following__qwen3embed__highbase__seed1

- Status: completed on server
- Stage: `joint_router_refinement`
- Scope:
  direct-prompt routed-node refiner using:
  - `expert_ego`: BotSay-style `tweet + metadata`
  - `expert_graph_following`: Glance-style `EGO + HOP1_FOLLOWING + Category?`
  - embedding model: `Qwen3-Embedding-8B`
  - fusion: `raw_concat_ego_following`
- Baseline:
  canonical high-base frozen SimTeG seed 1
  - full-test Acc / Macro-F1:
    `0.863905325443787 / 0.862442357206568`
- Execution note:
  the server `lmbot` env was not usable for this line after the transformers
  upgrade because `torch==2.0.1+cu117` is incompatible with
  `transformers>=4.56` model/optimization imports. This run therefore used the
  server `Qwen` env for both `precompute.py` and `main.py`.
- Routed source:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
- Component caches:
  - ego:
    `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_ego_qwen3embed_routed_glance_botsay_seed1.pt`
  - graph_following:
    `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_graph_following_qwen3embed_routed_glance_seed1.pt`
- Merged bundle used by the refiner:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_ego_following_qwen3embed_routed_glance_botsay_seed1.pt`
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_ego_following__qwen3embed__highbase__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_ego_following__qwen3embed__highbase__seed1/seed_1/stages/joint_router_refinement`
- Completed artifacts:
  - `manifest.json`
  - `metrics.json`
  - `analysis_summary.json`
  - `outputs.pt`
  - `checkpoint.pt`
  - `per_node_test.jsonl`
- Validation-locked comparable result:
  - selected budget:
    `0.10`
  - test Acc / Macro-F1 under the validation-selected budget:
    `0.863905325443787 / 0.8625376651402136`
  - fix / break / net:
    `7 / 7 / 0`
- Post-hoc test-curve best point:
  - full-test Acc / Macro-F1:
    `0.8681318681318682 / 0.8667838952219258`
  - delta vs base:
    `+0.004226542688081203 / +0.004341538015357749`
  - fix / break / net:
    `15 / 10 / +5`
  - routed wrong precision:
    `0.3614864864864865`
  - wrong-node fix rate:
    `0.09316770186335403`
  - correct-node break rate:
    `0.009784735812133072`
- Summary:
  This BotSay-ego + Glance-following line has real correction signal, but it
  does not beat the high-base frozen SimTeG under the validation-locked budget
  protocol: the comparable routed-test result is exactly net-neutral
  (`7 / 7 / 0`). The positive `+5` net gain appears only at the post-hoc best
  test budget, so it should be treated as diagnostic headroom rather than a
  claim-grade improvement.

### mpe__raw_concat_ego_follower__qwen3embed__highbase__seed1

- Status: completed on server
- Stage: `joint_router_refinement`
- Scope:
  same direct-prompt routed-node refiner family as above, but replacing the
  graph branch with `expert_graph_follower` and using
  `raw_concat_ego_follower`
- Baseline:
  canonical high-base frozen SimTeG seed 1
  - full-test Acc / Macro-F1:
    `0.863905325443787 / 0.862442357206568`
- Routed source:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
- Component caches:
  - ego:
    `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_ego_qwen3embed_routed_glance_botsay_seed1.pt`
  - graph_follower:
    `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_graph_follower_qwen3embed_routed_glance_seed1.pt`
- Merged bundle used by the refiner:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_ego_follower_qwen3embed_routed_glance_botsay_seed1.pt`
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_ego_follower__qwen3embed__highbase__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_ego_follower__qwen3embed__highbase__seed1/seed_1/stages/joint_router_refinement`
- Completed artifacts:
  - `manifest.json`
  - `metrics.json`
  - `analysis_summary.json`
  - `outputs.pt`
  - `checkpoint.pt`
  - `per_node_test.jsonl`
- Validation-locked comparable result:
  - selected budget:
    `0.05`
  - test Acc / Macro-F1 under the validation-selected budget:
    `0.863905325443787 / 0.8626744391450274`
  - fix / break / net:
    `7 / 7 / 0`
- Post-hoc test-curve best point:
  - full-test Acc / Macro-F1:
    `0.8647506339814032 / 0.8639807756430157`
  - delta vs base:
    `+0.0008453085376162184 / +0.0015384184364476416`
  - fix / break / net:
    `24 / 23 / +1`
  - routed wrong precision:
    `0.3614864864864865`
  - wrong-node fix rate:
    `0.14906832298136646`
  - correct-node break rate:
    `0.022504892367906065`
- Summary:
  The follower ablation is weaker than the following mainline. It fixes more
  wrong nodes at the post-hoc best point (`24`) but also breaks substantially
  more correct nodes (`23`), leaving only `+1` net. Under the validation-locked
  budget protocol it is again net-neutral (`7 / 7 / 0`), so there is no
  claim-grade improvement over the high-base frozen SimTeG baseline.

### qwen3embed_direct_prompt_routed_glance_botsay__execution_note

- Status: recorded
- Scope:
  execution note for the direct-prompt `Qwen3-Embedding-8B` routed-node line
- Note:
  an initial full-bundle `expert_concat_v1` precompute over
  `ego + graph_following + graph_follower + tweet + conflict` was started on
  the server but then abandoned as an inefficient path for this question. The
  actual downstream experiments in this section consume only:
  - `ego + graph_following`
  - `ego + graph_follower`
  so the final claim-supporting runs were produced from targeted component
  caches plus a minimal merged bundle per fusion mode. This does not change the
  model family; it only removes unused prompt-expert components from the
  runtime path.

### mpe__raw_concat_ego_following_follower__qwen3embed__highbase__seed1

- Status: completed on server
- Stage: `joint_router_refinement`
- Scope:
  direct-prompt routed-node refiner using:
  - `expert_ego`: BotSay-style `tweet + metadata`
  - `expert_graph_following`: Glance-style `EGO + HOP1_FOLLOWING + Category?`
  - `expert_graph_follower`: Glance-style `EGO + HOP1_FOLLOWER + Category?`
  - embedding model: `Qwen3-Embedding-8B`
  - fusion: `raw_concat_ego_following_follower`
- Baseline:
  canonical high-base frozen SimTeG seed 1
  - full-test Acc / Macro-F1:
    `0.863905325443787 / 0.862442357206568`
- Routed source:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
- Merged bundle used by the refiner:
  `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_ego_following_follower_qwen3embed_routed_glance_botsay_seed1.pt`
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_ego_following_follower__qwen3embed__highbase__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/mpe__raw_concat_ego_following_follower__qwen3embed__highbase__seed1/seed_1/stages/joint_router_refinement`
- Completed artifacts:
  - `manifest.json`
  - `metrics.json`
  - `analysis_summary.json`
  - `outputs.pt`
  - `checkpoint.pt`
  - `per_node_test.jsonl`
- Validation-locked comparable result:
  - selected budget:
    `0.05`
  - test Acc / Macro-F1 under the validation-selected budget:
    `0.8630600169061707 / 0.8618875071345797`
  - fix / break / net:
    `7 / 8 / -1`
- Post-hoc test-curve result recorded in `overall_test`:
  - full-test Acc / Macro-F1:
    `0.8622147083685545 / 0.8613392413492079`
  - delta vs base:
    `-0.0016906170752324368 / -0.0011031158573601152`
  - fix / break / net:
    `21 / 23 / -2`
  - routed wrong precision:
    `0.3614864864864865`
  - wrong-node fix rate:
    `0.13043478260869565`
  - correct-node break rate:
    `0.022504892367906065`
- Summary:
  Adding both directional graph experts to the BotSay-style `ego` branch does
  not improve this routed-node refiner line. Compared with the two smaller
  ablations, the three-way raw concat increases recovery on wrong nodes but
  also increases breaks enough to become clearly negative (`21 / 23 / -2` at
  the recorded test curve, `7 / 8 / -1` under the validation-locked budget).
  In this setting, `following` is the safer directional signal; naively adding
  `follower` on top does not yield useful complementarity.

### semantic_gate_localcompetence__dgp_v2_peft_mlp__highbase_routed__seed1

- Status: completed on server
- Stage: `semantic_correction_gate`
- Scope:
  literature-backed local-competence gate over the same two fixed DGP-v2
  semantic candidates used by the probability-only and node-attribute gate
  diagnostics:
  - `qwen25_peft`
  - `dgp_embed_mlp`
- Claim boundary:
  this is a routed-node learning-to-defer / dynamic-selection diagnostic. It
  does not regenerate prompts, update Qwen PEFT, update the embedding MLP,
  change the router, modify graph structure, or retrain frozen SimTeG.
- Literature basis:
  - META-DES: candidate competence should be estimated in a local region, not
    only from global posterior scores.
  - Learning-to-Defer / SelectiveNet: the downstream decision is accept/defer
    against a strong base, not direct semantic replacement.
  - Multicalibration: global calibration can fail on selected subgroups, so
    subgroup/local competence features are a reasonable next diagnostic after
    static node attributes failed.
- Stable routed split:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_gate_localcompetence__dgp_v2_peft_mlp__highbase_routed__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_gate_localcompetence__dgp_v2_peft_mlp__highbase_routed__seed1/seed_1/preparation/semantic_correction_gate`
- Completed artifacts:
  - `manifest.json`
  - `metrics.json`
  - `outputs.pt`
  - `checkpoint.pt`
  - `per_node_test.jsonl`
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
/root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task semantic_correction_gate \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --seeds 1 \
  --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json \
  --semantic_gate_base_outputs_path /root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_classifier_qwen3_20260601/base_outputs_for_routed_classifier.pt \
  --semantic_gate_candidate_output_paths /root/workspace/LMbot/LLMbot/experiments/dgp__v2_qwen25_peft_predictor_norm_text_following_summary__highbase_routed__seed1/seed_1/preparation/semantic_encoder/outputs.pt,/root/workspace/LMbot/LLMbot/experiments/calm__dgp_v2_norm_text_following_summary_qwen25__highbase_routed__seed1/seed_1/preparation/semantic_embedding_classifier/outputs.pt \
  --semantic_gate_candidate_names qwen25_peft,dgp_embed_mlp \
  --semantic_gate_feature_family local_competence \
  --semantic_gate_local_k 25 \
  --semantic_gate_break_weight 2.0 \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/semantic_gate_localcompetence__dgp_v2_peft_mlp__highbase_routed__seed1 \
  --device 0 \
  --disable_wandb
```

- Feature contract:
  - `--semantic_gate_feature_family local_competence`
  - `--semantic_gate_local_k 25`
  - final action feature dim: `64`
  - node-attribute descriptor dim: `52`
  - local competence appended dim: `12`
  - competence source: routed train nodes only
  - nearest-neighbor policy:
    `cosine_topk_on_train_standardized_action_descriptors`
  - self-neighbor policy: excluded for train queries
- Validation-locked result:
  - validation threshold:
    `0.17996224761009216`
  - validation fix / break / net:
    `12 / 7 / +5`
  - routed test fix / break / net:
    `4 / 6 / -2`
  - routed test base Acc / Macro-F1:
    `0.6385135135135135 / 0.6291638858641564`
  - routed test gated Acc / Macro-F1:
    `0.6317567567567568 / 0.625911625911626`
  - canonical full-test gated Acc / Macro-F1:
    `0.8622147083685545 / 0.8609685315567668`
  - canonical high-base frozen SimTeG Acc / Macro-F1:
    `0.863905325443787 / 0.862442357206568`
- Routed test selected action counts:
  - `base`: `247`
  - `qwen25_peft`: `7`
  - `dgp_embed_mlp`: `42`
- Comparison to previous gate diagnostics:

| run | feature family | break weight | valid fix/break/net | test fix/break/net | full-test Acc | full-test Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| probability baseline | probability | 2 | `20 / 14 / +6` | `9 / 12 / -3` | `0.8613693998309383` | `0.8603514893960071` |
| nodeattr bw2 | node_attribute | 2 | `42 / 41 / +1` | `12 / 26 / -14` | `0.8520710059171598` | `0.8508277340442779` |
| nodeattr bw4 | node_attribute | 4 | `4 / 3 / +1` | `2 / 8 / -6` | `0.8588334742180896` | `0.8574629896979364` |
| nodeattr bw8 | node_attribute | 8 | `3 / 2 / +1` | `1 / 4 / -3` | `0.8613693998309383` | `0.8599523001051015` |
| local competence k25 | local_competence | 2 | `12 / 7 / +5` | `4 / 6 / -2` | `0.8622147083685545` | `0.8609685315567668` |

- Interpretation boundary:
  The literature-backed local-competence gate improves the failure mode of the
  node-attribute bw2 run by becoming much more conservative on routed test
  (`10` changed nodes rather than `38`), but it still fails to beat frozen
  SimTeG and does not solve valid-test mismatch. The result supports the
  negative diagnosis that static attributes and train-neighborhood competence
  estimates are still insufficient to reliably identify safe semantic
  corrections under the fixed high-base routed split.

### semantic_gate_defer_breakfirst_localcompetence__dgp_v2_peft_mlp__highbase_routed__seed1

- Status: completed on server
- Stage: `semantic_correction_gate`
- Scope:
  P0/P1 gate optimization over the same fixed DGP-v2 semantic candidates:
  - `qwen25_peft`
  - `dgp_embed_mlp`
- Claim boundary:
  this run changes only the routed-node gate selection policy. It does not
  regenerate prompts, update Qwen PEFT, update the embedding MLP, change the
  router, modify graph structure, or retrain frozen SimTeG.
- Method contract:
  - `--semantic_gate_feature_family local_competence`
  - `--semantic_gate_local_k 25`
  - `--semantic_gate_selection_policy defer_softmax`
  - `--semantic_gate_safety_policy break_first`
  - action space: `{keep_base, qwen25_peft, dgp_embed_mlp}`
  - safety head: per-candidate break-risk, threshold locked on routed validation
- Stable routed split:
  `/root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_gate_defer_breakfirst_localcompetence__dgp_v2_peft_mlp__highbase_routed__seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/semantic_gate_defer_breakfirst_localcompetence__dgp_v2_peft_mlp__highbase_routed__seed1/seed_1/preparation/semantic_correction_gate`
- Completed artifacts:
  - `manifest.json`
  - `metrics.json`
  - `outputs.pt`
  - `checkpoint.pt`
  - `per_node_test.jsonl`
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
/root/mambaforge/envs/lmbot/bin/python -u main.py \
  --experiment_task semantic_correction_gate \
  --dataset TwiBot-20 \
  --reset_split -1 \
  --seeds 1 \
  --routed_nodes_path /root/workspace/LMbot/datasets/TwiBot-20/routed_nodes_highbase_preiter_budget020_seed1_20260529.json \
  --semantic_gate_base_outputs_path /root/workspace/LMbot/LLMbot/server_prompt_expert_v2_clean_routed_classifier_qwen3_20260601/base_outputs_for_routed_classifier.pt \
  --semantic_gate_candidate_output_paths /root/workspace/LMbot/LLMbot/experiments/dgp__v2_qwen25_peft_predictor_norm_text_following_summary__highbase_routed__seed1/seed_1/preparation/semantic_encoder/outputs.pt,/root/workspace/LMbot/LLMbot/experiments/calm__dgp_v2_norm_text_following_summary_qwen25__highbase_routed__seed1/seed_1/preparation/semantic_embedding_classifier/outputs.pt \
  --semantic_gate_candidate_names qwen25_peft,dgp_embed_mlp \
  --semantic_gate_feature_family local_competence \
  --semantic_gate_local_k 25 \
  --semantic_gate_selection_policy defer_softmax \
  --semantic_gate_safety_policy break_first \
  --semantic_gate_break_weight 2.0 \
  --artifact_root /root/workspace/LMbot/LLMbot/experiments/semantic_gate_defer_breakfirst_localcompetence__dgp_v2_peft_mlp__highbase_routed__seed1 \
  --device 0 \
  --disable_wandb
```

- Validation-locked result:
  - validation threshold:
    accept `0.5`, break `0.6399999856948853`
  - validation fix / break / net:
    `5 / 4 / +1`
  - routed test fix / break / net:
    `1 / 3 / -2`
  - canonical full-test gated Acc / Macro-F1:
    `0.8622147083685545 / 0.8608300584959926`
  - canonical high-base frozen SimTeG Acc / Macro-F1:
    `0.863905325443787 / 0.862442357206568`
- Routed test selected action counts:
  - `base`: `292`
  - `dgp_embed_mlp`: `4`
  - `qwen25_peft`: `0`
- Narrow ablations:

| run | selection | safety | break budget | valid fix/break/net | test fix/break/net | full-test Acc | full-test Macro-F1 | selected test actions |
|---|---|---|---:|---:|---:|---:|---:|---|
| probability baseline | independent BCE | none | n/a | `20 / 14 / +6` | `9 / 12 / -3` | `0.8613693998309383` | `0.8603514893960071` | base `275`, qwen `3`, mlp `18` |
| local threshold | independent BCE | none | n/a | `12 / 7 / +5` | `4 / 6 / -2` | `0.8622147083685545` | `0.8609685315567668` | base `247`, qwen `7`, mlp `42` |
| defer only | defer_softmax | none | n/a | `6 / 6 / 0` | `2 / 4 / -2` | `0.8622147083685545` | `0.8608770498249321` | base `290`, mlp `6` |
| defer + break-first | defer_softmax | break_first | -1 | `5 / 4 / +1` | `1 / 3 / -2` | `0.8622147083685545` | `0.8608300584959926` | base `292`, mlp `4` |
| defer + break-first budget0 | defer_softmax | break_first | 0 | `1 / 0 / +1` | `0 / 0 / 0` | `0.863905325443787` | `0.862442357206568` | base `296` |

- Interpretation boundary:
  The action-level learning-to-defer gate and break-first safety head did not
  recover positive routed-test utility. They reduced coverage and break, but
  the accepted candidate actions still had lower test fix than break. With a
  zero-break validation budget the policy collapses to full abstain on routed
  test. This supports the current diagnosis that the fixed semantic candidate
  outputs do not expose a stable enough correction-utility boundary for small
  post-hoc gates to exploit under this high-base routed split.

### hyperscan_labeled_dynamic_neighborloader_originaldetector_step200_20260609_seed1

- Status: completed on server
- Purpose:
  detector/fusion-head ablation for the closest labeled-graph
  NeighborLoader HyperScan-style branch.
- Claim boundary:
  single-seed, fixed 200 optimizer-step ablation. It changes only
  `--hyperscan_detector_style` from the current residual detector to the
  HyperScan-style bidirectional cross-attention + concat/ReLU + linear
  detector. It does not change the labeled split, semantic embedding tensor,
  relation backbone, batch-local KNN construction, NeighborLoader fanout,
  graph batch size, or update budget.
- Artifact root:
  `/root/workspace/LMbot/LLMbot/experiments/hyperscan_labeled_dynamic_neighborloader_originaldetector_step200_20260609_seed1`
- Stage root:
  `/root/workspace/LMbot/LLMbot/experiments/hyperscan_labeled_dynamic_neighborloader_originaldetector_step200_20260609_seed1/seed_1/preparation/graph_detector`
- Completed artifacts:
  - `manifest.json`
  - `selection_metrics.json`
  - `graph_refine_stats.json`
  - `outputs.pt`
  - `checkpoint.pt`
  - `run.log`
- Command:

```bash
cd /root/workspace/LMbot/LLMbot
CUDA_VISIBLE_DEVICES=0 /root/mambaforge/envs/lmbot/bin/python main.py \
  --experiment_task graph_detector_prepare \
  --experiment_name experiments/hyperscan_labeled_dynamic_neighborloader_originaldetector_step200_20260609_seed1 \
  --dataset TwiBot-20 \
  --graph_data_variant labeled \
  --use_GNN \
  --graph_backbone rgcn_hyperscan_routed \
  --embedding_path /root/workspace/LMbot/TwiBot-20_seed_1/intermediate/LM/embeddings_iter_-1.pt \
  --graph_refine_mode hyperscan_neighborloader_batch_local_branch \
  --graph_neighbor_num_neighbors 64 \
  --graph_refine_knn_k 8 \
  --hyperscan_detector_style original_cross_attention \
  --gnn_batch_size 1024 \
  --graph_training_max_steps 200 \
  --seeds 1 \
  --device 0 \
  --disable_wandb
```

- Same-budget ablation:

| run | dynamic KNN | detector | optimizer steps | Val Acc | Val Macro-F1 | Val Loss | node_repr dim |
|---|---|---|---:|---:|---:|---:|---:|
| `hyperscan_labeled_base_neighborloader_step200_20260609_seed1` | no | residual graph baseline | 200 | `0.8596194503` | `0.8567800295` | `0.3191002905` | 128 |
| `hyperscan_labeled_dynamic_neighborloader_step200_20260609_seed1` | yes | residual | 200 | `0.8600422833` | `0.8571940407` | `0.3178465664` | 128 |
| `hyperscan_labeled_dynamic_neighborloader_originaldetector_step200_20260609_seed1` | yes | original cross-attention | 200 | `0.8668076110` | `0.8644175976` | `0.3118747771` | 256 |

- Interpretation boundary:
  Under this single-seed same-budget ablation, swapping in the paper-style
  detector/fusion head gives a clear validation improvement over both the
  residual dynamic branch and the no-dynamic NeighborLoader control. This
  supports testing the paper detector in longer/multi-seed HyperScan-style
  runs, but does not by itself establish a claim-grade reproduction.
