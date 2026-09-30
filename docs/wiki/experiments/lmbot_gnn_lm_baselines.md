---
type: experiment
node_id: exp:lmbot_gnn_lm_baselines
title: LMbot GNN + LM Baseline Results on TwiBot-20
status: completed
priority: high
created_at: 2026-01-18T00:00:00Z
updated_at: 2026-04-15T00:00:00Z
tags: [baseline, lmbot, gnn, lm, twibot-20, rgcn]
---

# LMbot GNN + LM Baseline Results

## Source

Server: `/root/workspace/LMbot/TwiBot-20_seed_{1-5}/`
Local: `LMBot/results_rgcn/seed_{1-5}/`

## Results (RGCN, 10-round iterative co-distillation, 5 seeds)

| Model | Accuracy (mean) | F1 (mean) |
|-------|-----------------|-----------|
| GNN (RGCN) | 0.8511 | 0.8723 |
| LM (pretrain) | 0.8437 | 0.8676 |
| LM (co-trained) | 0.8540 | 0.8771 |

**Note**: Previously labeled "SimpleHGN" — corrected to RGCN (confirmed from training log: `RGCN(... RGCNConv ...)`). The `--GNN_model` default is `rgcn`.

See also: [exp:lmbot_rgt_baseline](lmbot_rgt_baseline.md) for RGT single-pass results.

## Key Observation

- LM co-trained (F1=0.8771) slightly outperforms GNN (F1=0.8723) after iterative distillation
- Both are strong baselines (~87% F1)
- These are the LMbot (WSDM 2024) reproduction numbers on the unified split

## Frozen SimTeG `iter_-1` Baseline For KNN Router / Prompt Work

**Date**: 2026-06-14

This is the current canonical baseline for the KNN router -> LLM prompt/refiner
branch. The input must be the frozen LM embedding at `iter_-1`, not
`finetuned_roberta_embeddings_iter_2_seed1.pt`.

Source-of-truth seed inputs on the server:

```text
/root/workspace/LMbot/TwiBot-20_seed_{1..5}/intermediate/LM/embeddings_iter_-1.pt
```

Detector outputs:

```text
/root/workspace/LMbot/LLMbot_active_knn_router_20260612/experiments/tmp_iterm1_forward_fullgraph/seed_{1,2,3}/preparation/graph_detector/outputs.pt
/root/workspace/LMbot/LLMbot_active_knn_router_20260612/experiments/tmp_iter_minus1_5seed_fullgraph/seed_{4,5}/preparation/graph_detector/outputs.pt
```

Full-test baseline:

| Seed | Accuracy | Macro-F1 | Bot-F1 | Wrong |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.8605 | 0.8591 | 0.8732 | 165 |
| 2 | 0.8605 | 0.8581 | 0.8766 | 165 |
| 3 | 0.8681 | 0.8666 | 0.8809 | 156 |
| 4 | 0.8538 | 0.8528 | 0.8649 | 173 |
| 5 | 0.8656 | 0.8636 | 0.8802 | 159 |
| Mean +/- sample std | 0.8617 +/- 0.0055 | 0.8600 +/- 0.0053 | 0.8752 +/- 0.0065 | 163.6 +/- 6.5422 |

Formal GraphEdit/Qwen strict sidecar run:

```text
/root/workspace/LMbot/LLMbot_active_knn_router_20260612/experiments/graphedit_qwen35_edgeuse_iter_minus1_5seed_20260614
```

Protocol notes:

- Router: `samehyperedge_global_tail`, `K=8`, `candidate_scope=hyperscan_full`,
  `repr_source=x_new`, `learning_mode=ncp_local`,
  `local_calibration_scope=same_hyperedge`.
- LLM prompt: Qwen3.5-9B with GraphEdit-style `graphedit_edge_usefulness`.
- Prompt reuse is strict hash reuse only:
  `prompt_sha256(prompt_text, prompt_family, prompt_variant, stage, src, dst)`.
  It is not safe to reuse by seed, routed node, or edge id alone because seed
  changes alter routed nodes, KNN support, and second-ring add candidates.
- Final strict cache size: `20410`; per-seed sidecar coverage: `4720/4720`;
  cache merge conflicts: `0`.

Routed-test aggregate over 5 seeds:

| Mode | Macro-F1 Mean | Acc Mean | Bot-F1 Mean | Wrong Mean | Delta Macro-F1 vs Base-Only |
| --- | ---: | ---: | ---: | ---: | ---: |
| base_only | 0.4421 | 0.4746 | 0.4689 | 31.0 | 0.0000 |
| all_knn | 0.4449 | 0.4576 | 0.4861 | 32.0 | +0.0028 |
| keepdrop | 0.4471 | 0.4678 | 0.5403 | 31.4 | +0.0049 |
| keepdrop_add | 0.4838 | 0.5085 | 0.5652 | 29.0 | +0.0417 |

Full-test aggregate after replacing only routed-test predictions:

| Mode | Full Macro-F1 Mean | Full Acc Mean | Full Bot-F1 Mean | Full Wrong Mean | Delta Macro-F1 vs Raw Frozen SimTeG |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw frozen SimTeG | 0.8600 | 0.8617 | 0.8752 | 163.6 | 0.0000 |
| base_only | 0.8581 | 0.8598 | 0.8736 | 165.8 | -0.0020 |
| all_knn | 0.8572 | 0.8590 | 0.8731 | 166.8 | -0.0028 |
| keepdrop | 0.8576 | 0.8595 | 0.8741 | 166.2 | -0.0025 |
| keepdrop_add | 0.8597 | 0.8615 | 0.8758 | 163.8 | -0.0004 |

Interpretation boundary: `keepdrop_add` improves the routed slice relative to a
target-only routed refiner, but the 5-seed full-test effect is essentially
neutral. Seed 4 is a negative outlier, so this is not yet a robust overall
performance claim.
