---
type: experiment
node_id: exp:gnn_backbone_comparison
title: GNN Backbone Comparison (RGT vs RGCN vs BotRGCN) on TwiBot-20
status: completed
priority: medium
created_at: 2026-04-16T00:00:00Z
updated_at: 2026-04-16T00:00:00Z
tags: [baseline, gnn, botrgcn, rgcn, rgt, twibot-20, backbone-comparison]
---

# GNN Backbone Comparison: RGT vs RGCN vs BotRGCN

## Purpose

Compare three GNN backbones under identical conditions to determine which relational graph convolution best suits the SEGA framework for social bot detection.

## Setup

- **Dataset**: TwiBot-20 (11826 nodes, 16908 edges, 2 relation types)
- **Input features**: Precomputed RoBERTa LM embeddings (768-dim, from LMbot iter-0)
- **Architecture**: GNN backbone + Linear classifier (no fusion, no VIB)
- **Hidden dim**: 128
- **Training**: Full-batch, Adam (lr=1e-3, wd=1e-4), early stopping (patience=40)
- **Seeds**: 42, 123, 456
- **Server**: RTX 3090, CUDA

## Models

| Model | Description | Params |
|-------|-------------|--------|
| RGT | Relational Graph Transformer (multi-head attention + FFN) | 496,002 |
| RGCN | Relational Graph Conv with basis decomposition + LayerNorm + residual | 164,994 |
| BotRGCN | FACNConv (attention-based edge weighting + RCM) ported from SEBot | 329,986 |

## Results

| Model | Accuracy | Macro-F1 | Precision | Recall |
|-------|----------|----------|-----------|--------|
| RGT | 0.8597±0.0078 | 0.8579±0.0078 | 0.8616±0.0092 | 0.8563±0.0076 |
| RGCN | 0.8656±0.0007 | 0.8635±0.0005 | 0.8690±0.0020 | 0.8613±0.0003 |
| BotRGCN | **0.8679±0.0042** | **0.8660±0.0042** | **0.8703±0.0049** | **0.8640±0.0040** |

## Analysis

1. **BotRGCN > RGCN > RGT** on all metrics. BotRGCN's FACNConv attention + RCM provides ~0.25% F1 gain over plain RGCN.
2. **RGCN is most stable** (lowest std across seeds), likely due to simpler architecture.
3. **RGT underperforms** — full self-attention on the entire batch may oversmooth or overfit with only 16K edges. The transformer layers also have 3x the parameters of RGCN.
4. **BotRGCN cost**: 2x params vs RGCN, ~2.5x training time, but still fast (<4s per seed).

## Caveats

- This is GNN-only (no text expert, no fusion). Rankings may shift when integrated into the full SEGA pipeline.
- Input is 768-dim RoBERTa embeddings, not 4096-dim Qwen. Results may differ with Qwen features.
- Only 3 seeds; consider 5-seed runs for publication.

## Artifacts

- Script: `LLMbot/code/sega/scripts/run_gnn_comparison.py`
- Server log: `/root/workspace/LMbot/gnn_comparison.log`
- Results JSON: `/root/workspace/LMbot/datasets/gnn_comparison_results.json`
- BotRGCN implementation: `LLMbot/code/sega/models/gnn.py` (FACNConv + BotRGCN classes)
