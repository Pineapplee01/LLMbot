---
type: experiment
node_id: exp:lmbot_rgt_baseline
title: LMBot RoBERTa→RGT Baseline Reproduction on TwiBot-20
status: completed
priority: high
seeds: [1, 2, 3, 4, 5]
created_at: 2026-04-15T00:00:00Z
updated_at: 2026-04-15T00:00:00Z
tags: [baseline, lmbot, rgt, roberta, twibot-20, reproduction]
---

# LMBot RoBERTa→RGT Baseline Reproduction

## Purpose

Reproduce the LMBot pipeline with RGT (Relational Graph Transformer) as the GNN backbone, as a strong baseline for the failure-regime-conditioned intervention research line. Single-pass (max_iters=1): pretrain RoBERTa embeddings → train RGT on graph, no iterative co-distillation.

## Code Changes

Original `LMBot/GNNs.py` had undefined `MLP_Model` class in RGT/HGT. Fixed by replacing with `nn.Linear(lm_input_dim, hidden_dim)` and aligning forward signature to `(x, edge_index, edge_type)`. Also fixed missing `--dataset` arg in `parser_args.py`. Also installed `torch_scatter` CUDA version on server (was CPU-only, causing `TransformerConv` crash).

## Configuration

- LM: RoBERTa-base (768-dim), pretrain 4.5 epochs
- GNN: RGT, 2 layers, hidden_dim=128, 8 attention heads, 8 semantic heads
- Training: 200 GNN epochs, AdamW lr=5e-4, dropout=0.4
- Data: TwiBot-20 unified split (train=8278, val=2365, test=1183)
- max_iters=1 (single-pass, no iterative distillation)

## Results

### RGT (single-pass, this experiment)

| Seed | GNN Acc | GNN F1 | LM Acc | LM F1 |
|------|---------|--------|--------|-------|
| 1 | 0.8384 | 0.8604 | 0.8541 | 0.8729 |
| 2 | 0.8402 | 0.8583 | 0.8517 | 0.8724 |
| 3 | 0.8419 | 0.8631 | 0.8537 | 0.8754 |
| 4 | 0.8374 | 0.8635 | 0.8556 | 0.8737 |
| 5 | 0.8425 | 0.8640 | 0.8553 | 0.8777 |
| **Mean** | **0.8401** | **0.8619** | **0.8541** | **0.8744** |

### RGCN (iterative 10-round co-distillation, from server)

| Seed | GNN Acc | GNN F1 | LM Acc | LM F1 |
|------|---------|--------|--------|-------|
| 1 | 0.8533 | 0.8732 | 0.8554 | 0.8756 |
| 2 | 0.8439 | 0.8646 | 0.8542 | 0.8779 |
| 3 | 0.8553 | 0.8765 | 0.8541 | 0.8757 |
| 4 | 0.8512 | 0.8715 | 0.8542 | 0.8783 |
| 5 | 0.8516 | 0.8756 | 0.8522 | 0.8780 |
| **Mean** | **0.8511** | **0.8723** | **0.8540** | **0.8771** |

### LM Pretrain (shared across GNN variants)

| Seed | Pretrain Acc | Pretrain F1 |
|------|-------------|-------------|
| 1 | 0.8403 | 0.8601 |
| 2 | 0.8505 | 0.8738 |
| 3 | 0.8448 | 0.8698 |
| 4 | 0.8442 | 0.8697 |
| 5 | 0.8385 | 0.8686 |

## Key Observations

- RGT single-pass GNN F1 (0.8619) is ~1pp below RGCN iterative (0.8723), expected since RGCN had 10 rounds of co-distillation
- LM performance is nearly identical across GNN variants (~0.8744 vs 0.8771), confirming LM pretrain is the shared foundation
- RGT serves as the base detector for failure-regime diagnostics

## Artifacts

- Server: `/root/workspace/LMbot/TwiBot-20_RGT_seed_{1-5}/`
- Local: `LMBot/results_rgt/seed_{1-5}/`
- Diagnostics: `TwiBot-20_RGT_seed_{1-5}/diagnostics/` (from v2 pipeline)
