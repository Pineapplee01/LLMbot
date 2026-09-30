# Experiment Notes

## TwiBot-20 Seed-1 Paper-Formula / Same-Hyperedge Alignment

Date: 2026-07-12

Purpose: test the manuscript's exact weighted reference-tail router separately
from the implementation's explicit `same_hyperedge` route. The two methods are
not aliases and must remain separate in reports.

Fixed contract: seed 1, fine-tuned RoBERTa input, `K=8`, fanout 64, DHG,
`routed_only` residual consumption, `low_only` non-consumer fallback, 10%
split-wise routing budget, and 200 graph optimizer steps. The exact paper route
uses the canonical train split as its reference pool and excludes a train
target's self-weight.

Operator commands:

```powershell
cd G:\Research\BotDetection\LLMbot
python run_twibot20_seed1_router_alignment_20260712.py --dry-run
python run_twibot20_seed1_router_alignment_20260712.py
```

Artifact root:

```text
LLMbot/experiments/twibot20_seed1_router_alignment_20260712/
```

The runner records the exact commands, environment, source hashes, resolved
same-hyperedge configuration, graph-contract checks, canonical test metrics,
prediction transitions, route overlaps, and error-ranking diagnostics for both
routes on their shared frozen C2 source. The report also includes a clearly
labelled fixed-weight paper-route swap using archived graph weights; that row is
an inference diagnostic, not a retrained model. No result from this single seed
may be promoted to five-seed significance or used to repair mixed mean/std
provenance.

Date: 2026-05-28

All metrics below are test-set percentages. `Std` is the population standard
deviation across 5 seeds.

## Evidence Paths

- BotBR on TwiBot-20:
  - log: `/root/workspace/LMbot/repro_baselines_20260527/logs/botbr_twibot20_seed1_5.log`
  - checkpoints: `/root/workspace/LMbot/botbr/checkpoint/Twitter/`
- BotBR on MGTAB:
  - log: `/root/workspace/LMbot/repro_baselines_20260527/logs/botbr_mgtab_seed1_5.log`
  - parsed seed metrics: `/root/workspace/LMbot/repro_baselines_20260527/results/botbr_mgtab_seed_metrics.json`
- LMBot RGCN baseline:
  - `/root/workspace/LMbot/TwiBot-20_seed_{1..5}/results_GNN.json`
  - `/root/workspace/LMbot/TwiBot-20_seed_{1..5}/results_LM.json`
- LMBot RGT baseline:
  - `/root/workspace/LMbot/TwiBot-20_RGT_seed_{1..5}/results_GNN.json`
  - `/root/workspace/LMbot/TwiBot-20_RGT_seed_{1..5}/results_LM.json`

## Current Frozen SimTeG Baseline For Router/Prompt Work

The canonical frozen SimTeG baseline for the current KNN router and routed
prompt/refiner branch is stored under
`docs/research/baselines/frozen_simteg_fullgraph_20260614/`.

Use this baseline for future `KNN router -> prompt` comparisons:

```text
/root/workspace/LMbot/LLMbot_active_knn_router_20260612/experiments/tmp_iterm1_forward_fullgraph/seed_{1,2,3}/preparation/graph_detector/outputs.pt
/root/workspace/LMbot/LLMbot_active_knn_router_20260612/experiments/tmp_iter_minus1_5seed_fullgraph/seed_{4,5}/preparation/graph_detector/outputs.pt
```

This family uses `contract=frozen_g0_v1`, `backbone=rgcn`,
`graph_data_variant=full_graph_support`, `training_scope=train_only_supervised`,
and frozen `iter_-1` LM embeddings. The source-of-truth seed inputs are:

```text
/root/workspace/LMbot/TwiBot-20_seed_{1..5}/intermediate/LM/embeddings_iter_-1.pt
```

Seed 1-3 detector manifests reference copied dataset files
`../datasets/TwiBot-20/embeddings_iter_-1_seed_{1,2,3}.pt`; seed 4-5 reference
the source `TwiBot-20_seed_{4,5}/intermediate/LM/embeddings_iter_-1.pt`
directly. Do not substitute `finetuned_roberta_embeddings_iter_2_seed1.pt`
for this branch. Replayed full-test metrics are:

| Seed | Accuracy | Macro-F1 | Bot-F1 | Wrong |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.8605 | 0.8591 | 0.8732 | 165 |
| 2 | 0.8605 | 0.8581 | 0.8766 | 165 |
| 3 | 0.8681 | 0.8666 | 0.8809 | 156 |
| 4 | 0.8538 | 0.8528 | 0.8649 | 173 |
| 5 | 0.8656 | 0.8636 | 0.8802 | 159 |
| Mean +/- sample std | 0.8617 +/- 0.0055 | 0.8600 +/- 0.0053 | 0.8752 +/- 0.0065 | 163.6 +/- 6.5422 |

Do not mix this line with historical `0.87`-level artifacts unless the method is
rerun on the same feature/training contract. The audit found that those stronger
runs used different conditions, especially
`finetuned_roberta_embeddings_iter_2_seed1.pt`, `train_supervised_plus_mhlgc`,
labeled-only graph variants, or HyperScan-style detector ablations.

### KNN Router -> Qwen Edge Editing Strict 5-Seed Run

Formal run root:

```text
/root/workspace/LMbot/LLMbot_active_knn_router_20260612/experiments/graphedit_qwen35_edgeuse_iter_minus1_5seed_20260614
```

Protocol:

- Router setting: `samehyperedge_global_tail`, `conformal_knn_k=8`,
  `candidate_scope=hyperscan_full`, `repr_source=x_new`,
  `learning_mode=ncp_local`, `local_calibration_scope=same_hyperedge`.
- LLM setting: Qwen3.5-9B, GraphEdit-style `graphedit_edge_usefulness`
  two-stage edge prompt, strict prompt reuse only by
  `prompt_sha256(prompt_text, prompt_family, prompt_variant, stage, src, dst)`.
- Prompt coverage: final strict cache size `20410`; each seed has
  `4720/4720` sidecar rows; final merge conflict count `0`.
- Prompt reuse is edge/prompt-level, not seed/node-level. Reuse is allowed only
  when the full prompt hash matches; seed changes alter routed nodes, KNN
  support, and second-ring candidates, so seed 4 and seed 5 required additional
  Qwen generation even though the LM input family is the same `iter_-1` branch.

Routed-test aggregate over 5 seeds:

| Mode | Macro-F1 Mean | Acc Mean | Bot-F1 Mean | Wrong Mean | Delta Macro-F1 vs Base-Only | Delta Wrong vs Base-Only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base_only | 0.4421 | 0.4746 | 0.4689 | 31.0 | 0.0000 | 0.0 |
| all_knn | 0.4449 | 0.4576 | 0.4861 | 32.0 | +0.0028 | +1.0 |
| keepdrop | 0.4471 | 0.4678 | 0.5403 | 31.4 | +0.0049 | +0.4 |
| keepdrop_add | 0.4838 | 0.5085 | 0.5652 | 29.0 | +0.0417 | -2.0 |

Full-test aggregate after replacing only routed-test predictions:

| Mode | Full Macro-F1 Mean | Full Acc Mean | Full Bot-F1 Mean | Full Wrong Mean | Delta Macro-F1 vs Raw Frozen SimTeG | Delta Wrong vs Raw Frozen SimTeG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw frozen SimTeG | 0.8600 | 0.8617 | 0.8752 | 163.6 | 0.0000 | 0.0 |
| base_only | 0.8581 | 0.8598 | 0.8736 | 165.8 | -0.0020 | +2.2 |
| all_knn | 0.8572 | 0.8590 | 0.8731 | 166.8 | -0.0028 | +3.2 |
| keepdrop | 0.8576 | 0.8595 | 0.8741 | 166.2 | -0.0025 | +2.6 |
| keepdrop_add | 0.8597 | 0.8615 | 0.8758 | 163.8 | -0.0004 | +0.2 |

Interpretation boundary: `keepdrop_add` is the best LLM-edited KNN consumer in
this run and improves the routed slice relative to the target-only refiner, but
the 5-seed full-test result is still essentially neutral rather than a robust
overall improvement. Seed 4 is a negative outlier, so this run supports
continued routed-node analysis rather than a final performance claim.

## BotBR Reproduction

### BotBR on TwiBot-20

This run uses the BotBR-specific processed feature bundle from
`F:\Twibot20\processed_data`, synchronized to a dedicated remote data root, and
reuses the canonical `train_idx.pt`, `valid_idx.pt`, and `test_idx.pt` split
files from `datasets/TwiBot-20/`.

| Seed | Best Epoch | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 56 | 86.81 | 85.17 | 91.56 | 88.25 |
| 2 | 84 | 87.32 | 85.92 | 91.56 | 88.65 |
| 3 | 68 | 86.81 | 86.01 | 90.31 | 88.11 |
| 4 | 64 | 86.81 | 85.28 | 91.41 | 88.24 |
| 5 | 60 | 86.81 | 84.87 | 92.03 | 88.31 |

| Metric | Mean | Std |
| --- | ---: | ---: |
| Accuracy | 86.91 | 0.20 |
| Precision | 85.45 | 0.44 |
| Recall | 91.37 | 0.57 |
| F1 | 88.31 | 0.18 |

Paper-vs-reproduction comparison on TwiBot-20:

| Metric | Paper | Reproduced | Delta (Reproduced - Paper) |
| --- | ---: | ---: | ---: |
| Accuracy | 87.23 +/- 0.08 | 86.91 +/- 0.20 | -0.32 |
| Precision | 85.25 +/- 0.18 | 85.45 +/- 0.44 | +0.20 |
| Recall | 92.89 +/- 0.19 | 91.37 +/- 0.57 | -1.52 |
| F1 | 88.91 +/- 0.06 | 88.31 +/- 0.18 | -0.60 |

The largest gap is on recall, while precision is slightly higher than the
paper table under the current LMBot-split reproduction.

### HyperScan on TwiBot-20 (LMBot Split)

This line uses the dedicated `HyperScan/Twibot20` adaptation and reuses the
same canonical `train_idx.pt`, `valid_idx.pt`, and `test_idx.pt` files as the
LMBot TwiBot-20 mainline. The table below is extracted from the completed
remote rerun written to
`/root/workspace/LMbot/repro_baselines_20260528/results/hyperscan_twibot20_lmbotsplit_results.csv`.

| Seed | Accuracy | Precision | Recall | F1 | AUCROC | AUCPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 91.27 | 91.46 | 93.47 | 92.46 | 96.35 | 96.79 |
| 2 | 85.87 | 85.74 | 90.42 | 88.02 | 92.99 | 93.81 |
| 3 | 93.35 | 92.98 | 95.62 | 94.28 | 97.35 | 97.57 |
| 4 | 93.60 | 93.22 | 95.80 | 94.49 | 97.52 | 97.80 |
| 5 | 93.57 | 93.27 | 95.68 | 94.46 | 97.66 | 97.96 |

| Metric | Mean | Std |
| --- | ---: | ---: |
| Accuracy | 91.53 | 2.96 |
| Precision | 91.33 | 2.87 |
| Recall | 94.20 | 2.08 |
| F1 | 92.74 | 2.48 |
| AUCROC | 96.37 | 1.75 |
| AUCPR | 96.79 | 1.54 |

Auxiliary retrieval-style diagnostics:

| Metric | Mean | Std |
| --- | ---: | ---: |
| Recall@P80 | 98.98 | 1.12 |
| Recall@P85 | 97.10 | 2.98 |
| Recall@P90 | 93.25 | 7.00 |

Remote evidence:

- log root: `/root/workspace/LMbot/repro_baselines_20260528/logs/`
- result csv: `/root/workspace/LMbot/repro_baselines_20260528/results/hyperscan_twibot20_lmbotsplit_results.csv`

### BotBR on MGTAB

| Seed | Best Epoch | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 160 | 93.33 | 87.27 | 87.27 | 87.27 |
| 2 | 187 | 92.65 | 84.08 | 89.34 | 86.63 |
| 3 | 175 | 92.65 | 84.43 | 89.05 | 86.68 |
| 4 | 140 | 92.25 | 81.98 | 89.23 | 85.45 |
| 5 | 150 | 91.67 | 83.56 | 87.37 | 85.42 |

| Metric | Mean | Std |
| --- | ---: | ---: |
| Accuracy | 92.51 | 0.55 |
| Precision | 84.26 | 1.72 |
| Recall | 88.45 | 0.93 |
| F1 | 86.29 | 0.73 |

## LMBot Reproduction

### LMBot Server-Side Artifact Snapshot

The current remote server already contains two pre-existing 5-seed TwiBot-20
baseline lines:

- `TwiBot-20_seed_{1..5}`: RoBERTa -> RGCN mainline
- `TwiBot-20_RGT_seed_{1..5}`: RoBERTa -> RGT stress-test backbone

For each seed, the server retains:

- `results_GNN.json`
- `results_LM.json`
- `diagnostics/regime_table_{val,test}.csv`
- LM/GNN logits, probabilities, predictions, and entropy tensors under
  `diagnostics/`

### LMBot Overall Branch Comparison on TwiBot-20

| Branch | Accuracy Mean | Accuracy Std | F1 Mean | F1 Std | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| RGCN GNN | 85.11 | 0.39 | 87.23 | 0.42 | Graph detector under `TwiBot-20_seed_{1..5}` |
| RGCN LM | 85.41 | 0.10 | 87.71 | 0.12 | Final LM branch under `TwiBot-20_seed_{1..5}` |
| RGT GNN | 84.01 | 0.19 | 86.18 | 0.22 | Graph detector under `TwiBot-20_RGT_seed_{1..5}` |
| RGT LM | 85.41 | 0.14 | 87.44 | 0.19 | Final LM branch under `TwiBot-20_RGT_seed_{1..5}` |

### LMBot RGCN Baseline on TwiBot-20

| Seed | GNN Acc | GNN F1 | LM Acc | LM F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 85.33 | 87.32 | 85.54 | 87.56 |
| 2 | 84.39 | 86.46 | 85.42 | 87.79 |
| 3 | 85.53 | 87.65 | 85.41 | 87.57 |
| 4 | 85.12 | 87.15 | 85.42 | 87.83 |
| 5 | 85.16 | 87.56 | 85.22 | 87.80 |

| Branch | Accuracy Mean | Accuracy Std | F1 Mean | F1 Std |
| --- | ---: | ---: | ---: | ---: |
| GNN | 85.11 | 0.39 | 87.23 | 0.42 |
| LM | 85.41 | 0.10 | 87.71 | 0.12 |

### LMBot RGCN LM Pretrain Stage on TwiBot-20

| Seed | Pretrain Acc | Pretrain F1 | Final LM Acc | Final LM F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 84.03 | 86.01 | 85.54 | 87.56 |
| 2 | 85.05 | 87.38 | 85.42 | 87.79 |
| 3 | 84.48 | 86.98 | 85.41 | 87.57 |
| 4 | 84.42 | 86.97 | 85.42 | 87.83 |
| 5 | 83.85 | 86.86 | 85.22 | 87.80 |

| Stage | Accuracy Mean | Accuracy Std | F1 Mean | F1 Std |
| --- | ---: | ---: | ---: | ---: |
| LM Pretrain | 84.37 | 0.42 | 86.84 | 0.45 |
| Final LM | 85.41 | 0.10 | 87.71 | 0.12 |

### LMBot RGT Baseline on TwiBot-20

| Seed | GNN Acc | GNN F1 | LM Acc | LM F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 83.84 | 86.04 | 85.41 | 87.29 |
| 2 | 84.02 | 85.83 | 85.17 | 87.24 |
| 3 | 84.19 | 86.31 | 85.37 | 87.54 |
| 4 | 83.74 | 86.35 | 85.56 | 87.37 |
| 5 | 84.25 | 86.40 | 85.53 | 87.77 |

| Branch | Accuracy Mean | Accuracy Std | F1 Mean | F1 Std |
| --- | ---: | ---: | ---: | ---: |
| GNN | 84.01 | 0.19 | 86.18 | 0.22 |
| LM | 85.41 | 0.14 | 87.44 | 0.19 |

### LMBot RGT LM Pretrain Stage on TwiBot-20

| Seed | Pretrain Acc | Pretrain F1 | Final LM Acc | Final LM F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 84.03 | 86.01 | 85.41 | 87.29 |
| 2 | 85.05 | 87.38 | 85.17 | 87.24 |
| 3 | 84.48 | 86.98 | 85.37 | 87.54 |
| 4 | 84.42 | 86.97 | 85.56 | 87.37 |
| 5 | 83.85 | 86.86 | 85.53 | 87.77 |

| Stage | Accuracy Mean | Accuracy Std | F1 Mean | F1 Std |
| --- | ---: | ---: | ---: | ---: |
| LM Pretrain | 84.37 | 0.42 | 86.84 | 0.45 |
| Final LM | 85.41 | 0.14 | 87.44 | 0.19 |

## Scope Notes

- `BotBR on TwiBot-20` and `BotBR on MGTAB` were extracted from this session's
  remote reproduction logs.
- `LMBot` results were extracted from pre-existing server-side 5-seed result
  JSON files rather than rerun during this documentation task.
- `HyperScan on MGTAB` is still in progress on the server and is intentionally
  not summarized here yet.

## Local Routed Residual Gate And K/Fanout Ablation

Date: 2026-06-18

Scope: local seed-1 exploratory evidence for paper-writing direction only. This
run did not use the server official HyperScan nodeinput tensor
`official_hyperscan_x_tweet_num_cat_labeled_prefix.pt`; it used
`datasets/TwiBot-20/embeddings_iter_-1_seed_1.pt` with the
`hyperscan_meta_tweet_proxy` node-input family. Do not mix these numbers with
official-nodeinput-788 server results without rerunning the same settings under
the official tensor and seeds 1/2/3.

Primary metric below is the canonical deduplicated full-test recompute from
`outputs.pt + test_idx.pt`; repeated sampled-subgraph sidecar metrics are
audit-only.

Evidence:

- Detailed record:
  `LLMbot/experiments.md`, section
  `2026-06-18 - Local routed residual gate and K ablation (seed 1)`.
- Summary artifacts:
  `LLMbot/experiments/local_twibot20_routed_residual_20260618_reports/summary_seed1.csv`
  and
  `LLMbot/experiments/local_twibot20_routed_residual_20260618_reports/summary_seed1.json`.

Gate / consumer ablation:

| Setting | Full Macro-F1 | Routed 10% Macro-F1 | Non-routed 10% Macro-F1 |
| --- | ---: | ---: | ---: |
| `low_only` | 0.8741 | 0.7340 | 0.8834 |
| `all_nodes residual` | 0.8732 | 0.7340 | 0.8824 |
| `routed_only residual @10%` | 0.8684 | 0.7627 | 0.8753 |
| `risk_gated_all_nodes residual` | 0.8761 | 0.7485 | 0.8845 |
| `shuffled routed_only residual @10%` | 0.8745 | 0.7771 | 0.8813 |

Gate diagnostic for `risk_gated_all_nodes`: Spearman(`risk`, `gate`) is
approximately 1.0; the learned gate spans a narrow but ordered range
(`min/max = 0.5128 / 0.7289`, bottom/top risk decile mean
`0.5408 / 0.7220`).

K / fanout ablation at routed-only 10%:

| Setting | Full Macro-F1 | Routed 10% Macro-F1 | Non-routed 10% Macro-F1 |
| --- | ---: | ---: | ---: |
| `K=4, fanout=64` | 0.8742 | 0.7917 | 0.8809 |
| `K=8, fanout=64` | 0.8684 | 0.7627 | 0.8753 |
| `K=16, fanout=64` | 0.8782 | 0.7726 | 0.8856 |
| `K=8, fanout=32` | 0.8732 | 0.7485 | 0.8815 |
| `K=8, fanout=128` | 0.8817 | 0.7943 | 0.8875 |

Writing boundary: the local evidence supports writing the risk gate as the
cleaner candidate mechanism than binary routed-only residual consumption in this
local contract. It does not yet prove that conformal-selected routed nodes are
better residual consumers than arbitrary same-budget nodes, because the shuffled
routed-only control is strong. The K/fanout table also shows that the residual
branch is sensitive to high-order neighborhood scale; the current best local
seed-1 setting is `K=8, fanout=128`, but that must be confirmed under the
official server tensor and multiple seeds before a paper-level performance
claim.

## TwiBot-20 Seed1 Component Ablation And Hyperparameter Sensitivity

Date: 2026-06-19

Scope: local TwiBot-20 seed-1 same-contract rerun for the paper route
`risk router -> routed nodes -> residual connection`. Component ablations and
hyperparameter sensitivity are reported separately. `K`, `fanout`, and `budget`
are hyperparameters, not removed model components.

Primary metric: canonical deduplicated full-test recompute from
`outputs.pt + test_idx.pt`. In the tables below, `Acc` is full-test accuracy
and `F1` is full-test Macro-F1.

Shared completed-run contract:

- `graph_backbone=rgcn_hyperscan_dhg_nodeinput`
- `graph_node_input_family=hyperscan_meta_tweet_proxy`
- `graph_training_loader_mode=neighbor_subgraph`
- `graph_second_view_scope=neighborloader_batch`
- `graph_neighborloader_contract=hyperscan_sampled_subgraph`
- `graph_second_view_hypergraph_backend=dhg`
- `graph_second_view_fusion=residual`
- `graph_training_max_steps=200`
- default setting: `K=8`, `fanout=64`, `budget=10%`

Evidence:

- Detailed record:
  `LLMbot/experiments.md`, section
  `2026-06-19 - TwiBot-20 seed1 component ablation and hyperparameter sensitivity`.
- Reports:
  `LLMbot/experiments/twibot20_seed1_component_ablation_20260619_reports/component_ablation_seed1.csv`
  and
  `LLMbot/experiments/twibot20_seed1_component_ablation_20260619_reports/hyperparameter_sensitivity_seed1.csv`.
- Hyperparameter figure:
  `LLMbot/experiments/twibot20_seed1_component_ablation_20260619_reports/twibot20_seed1_hparam_sensitivity.pdf`
  generated from the report CSV in a HyperScan-style line-plot format.
- Contract audit:
  `LLMbot/experiments/twibot20_seed1_component_ablation_20260619_reports/manifest_contract_checks_seed1.json`.

Component ablation:

| Variant | Removed / Changed Component | Status | Acc | F1 |
| --- | --- | --- | ---: | ---: |
| Full method | none: finetuned RoBERTa + conformal x_new router + routed-only residual @10% | completed | 0.8893 | 0.8878 |
| w/o router | all nodes consume residual | completed | 0.8850 | 0.8835 |
| w/o residual | empty routed mask, all nodes take low-only fallback | completed | 0.8774 | 0.8755 |
| w/o LM supervised fine-tuning | raw pretrained roberta-base embedding | missing local raw embedding |  |  |
| Frozen SimTeG -> RGCN/graph detector | replace finetuned embedding with `embeddings_iter_-1_seed_1.pt` | completed | 0.8715 | 0.8684 |
| Random routed residual | shuffled same-budget 10% routed mask | completed | 0.8698 | 0.8677 |

Notes:

- `w/o residual` uses the existing `routed_only + low_only fallback` path with
  an empty routed-node payload, so no node consumes residual output.
- `w/o router` is `all_nodes + residual`.
- The raw RoBERTa row is intentionally not backfilled from historical
  server-side raw RoBERTa results, because those runs used a different graph
  detector contract.
- This component ablation is recorded as the seed-1 table. The paper table
  should be updated with multi-seed mean/std after the same-contract seeds are
  completed.

Hyperparameter sensitivity:

| Hyperparameter | Setting | Acc | F1 |
| --- | ---: | ---: | ---: |
| K | 4 | 0.8859 | 0.8842 |
| K | 8 | 0.8893 | 0.8878 |
| K | 16 | 0.8791 | 0.8771 |
| fanout | 32 | 0.8884 | 0.8863 |
| fanout | 64 | 0.8893 | 0.8878 |
| fanout | 128 | 0.8757 | 0.8742 |
| budget | 5% | 0.8850 | 0.8834 |
| budget | 10% | 0.8893 | 0.8878 |
| budget | 20% | 0.8791 | 0.8774 |

Interpretation boundary: under this seed-1 finetuned-RoBERTa contract, the full
method is strongest on full-test Macro-F1 among the completed component rows.
However, the `w/o LM supervised fine-tuning` component remains incomplete until
a same-contract raw `roberta-base` embedding artifact is generated.
The hyperparameter sensitivity plot follows the HyperScan convention of
plotting `F1-Score` and `Accuracy` against each varied setting. It is used as
the paper's TwiBot-20 seed-1 hyperparameter sensitivity analysis and does not
require multi-seed expansion.
