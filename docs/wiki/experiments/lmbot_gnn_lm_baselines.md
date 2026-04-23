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
