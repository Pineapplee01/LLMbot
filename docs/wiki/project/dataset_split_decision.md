---
type: project
node_id: project:dataset_split_decision
title: TwiBot-20 数据集切片决策
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# 数据集切片决策：使用现有 LMbot Split

## 决策结论

**直接使用 `datasets/TwiBot-20/` 下现有的 train/val/test split，不重新切片。**

**Why:** BotBR (`botbr/Dataset.py:56-58`) 直接加载同一套 `train_idx.pt`/`val_idx.pt`/`test_idx.pt`，三个模型共享同一套 split，保证了公平比较。

**How to apply:** 所有 baseline 复现和消融实验均使用 `datasets/TwiBot-20/` 下的现有 split 文件，禁止重新生成。

## Split 详情

| 文件 | 大小 | 估算节点数 |
|------|------|-----------|
| train_idx.pt | 66923 bytes | ~8278 nodes |
| valid_idx.pt | 19627 bytes | ~2427 nodes |
| test_idx.pt | 10155 bytes | ~1121 nodes |
| **总计** | | **11826 nodes** |

**比例**: 约 70% / 20% / 10%（与 LMbot 的 7:2:1 split 一致）

## Baseline Split 使用情况

| 模型 | Split 来源 | 兼容性 |
|------|-----------|--------|
| LMbot (2024) | 生成了当前 split | ✅ 原始来源 |
| BotBR (AAAI'25) | 加载同一套 .pt 文件 | ✅ 直接兼容 |
| HyperScan (WWW'25) | 为 MGTAB 设计，需适配 | ⚠️ 需要确认 |
| LLMbot (我们的方法) | 使用同一套 .pt 文件 | ✅ 直接兼容 |

## HyperScan 适配问题

HyperScan 原始设计针对 MGTAB 数据集，不是 TwiBot-20。需要在新会话中确认：
- 是否已有 TwiBot-20 适配版本
- 如果没有，是否值得花时间适配
- 备选方案：只用 LMbot + BotBR 作为主要 baseline，HyperScan 作为参考引用
