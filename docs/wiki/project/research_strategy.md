---
type: project
node_id: project:research_strategy
title: 研究策略与路线图（2026-04-09 确定）
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# 研究策略：Option B — Targeted Slice Improvement

## 核心策略（导师确认）

**Novelty**: LLM 的 calibration uncertainty 信号指导图结构修改（不是简单的 LLM embedding 改图）

**目标**: 在特定 failure slice 上提升 3-5% F1，而非追求 global SOTA

**Why:** TwiBot-20 benchmark 已趋于饱和（2024→2025 提升仅 +0.8-1.2%），在 global F1 上提升 2% 是高风险目标。但 low-degree 和 graph-missing slice 存在明显的 failure mode，有针对性提升空间。

**How to apply:** 所有实验设计以 slice F1 为主要指标，global F1 作为辅助指标（不能显著下降）。

## 目标 Slice 与预期提升

| Slice | 定义 | 预期提升 | 理由 |
|-------|------|---------|------|
| graph_missing | total_degree == 0 | +5-8% | 图完全无用，LLM 是唯一信号 |
| low_degree | total_degree < 10 | +3-5% | 图信号稀疏，LLM 可补充 |
| disagreement | pred_sem ≠ pred_graph | +2-3% | LLM calibration 可判断谁对 |
| high_degree | total_degree ≥ 50 | 0% (不降) | 图信号充足，方法不应损害 |

## Novelty 定位（区别于已有工作）

**不是**:
- ❌ "用 LLM embedding 指导图修改" → ENGINE (IJCAI'24) 已做
- ❌ "用 LLM 预测指导图修改" → LLM-GNN (KDD'24) 已做

**是**:
- ✅ "用 LLM 的 calibration uncertainty 作为图可靠性信号"
- ✅ "semantic-graph disagreement 作为 trigger，而非全局修改"
- ✅ "证明在 graph-sparse 场景下 LLM-guided pruning 优于 graph structure learning"

## Baseline 对比策略

| Baseline | 来源 | 复现方式 | 优先级 |
|----------|------|---------|--------|
| LMbot (2024) | g:/Research/LMBot/LMbot/ | 代码已开源，直接运行 | 🔴 必须 |
| BotBR (AAAI'25) | g:/Research/LMBot/botbr/ | 代码已开源，直接运行 | 🔴 必须 |
| HyperScan (WWW'25) | g:/Research/LMBot/HyperScan/ | 需确认 TwiBot-20 适配 | 🟡 待确认 |

## 数据集策略

- **主数据集**: TwiBot-20（使用现有 split，不重新切片）
- **扩展数据集**: TwiBot-22（在核心方法验证后再做，证明泛化性）
- **Split 来源**: `datasets/TwiBot-20/train_idx.pt` 等（LMbot 生成，BotBR 兼容）

## 论文定位

**Title 候选**:
- "Calibration-Guided Graph Pruning for Social Bot Detection on Sparse Networks"
- "When to Trust LLM over Graph: Selective Graph Pruning for Bot Detection"

**Venue 目标**: WWW (best fit), AAAI (viable), KDD application track

**核心 Claim**:
- "Graph evidence is harmful on sparse/disagreement nodes"
- "LLM calibration uncertainty is a better graph reliability signal than topology features"
- "Targeted test-time pruning outperforms prediction-level deferral on triggered nodes"
