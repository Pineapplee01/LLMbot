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
| clean v2 routed explanations | `/root/workspace/LMbot/datasets/TwiBot-20/prompt_expert_v2_clean_routed_preiter_cache_20260531` | clean sidecars for graph_following, graph_follower, tweet, conflict |
| clean v2 finetuned-RoBERTa cache | `/root/workspace/LMbot/datasets/TwiBot-20/glance_prompt_expert_concat_v2_roberta_finetuned_simteg_clean_routed_highbase_preiter_seed1_20260531.pt` | routed explanation expert cache |
| pretrained roberta-base snapshot | `/root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b` | explicit pretrained encoder path |

## Experiment Registry

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
