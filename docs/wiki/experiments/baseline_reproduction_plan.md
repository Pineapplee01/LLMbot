---
type: experiment
node_id: exp:baseline_reproduction
title: Baseline Reproduction Plan — LMbot, BotBR, HyperScan on TwiBot-20
status: ready
priority: high
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-23T00:00:00Z
tags: [baseline, reproduction, twibot-20, slice-evaluation]
---

# Baseline Reproduction Plan

## Status: READY ✅

**Update (2026-04-09 23:30)**: Previous blocker was based on incorrect audit. Configuration is correct, results are valid.

**Verified**:
- ✅ Configuration is correct (LLMbot/code/ uses macro F1 and unified split)
- ✅ rw1/rw4 results are valid and comparable
- ✅ Kill tests B2/B3 pass

**Current Results**:
- rw1: Macro-F1=0.7391
- rw4: Macro-F1=0.7603
- LMbot (already reproduced): Macro-F1=0.8732

**See**: [baseline_audit_correction.md](baseline_audit_correction.md) for details.

---

## Objective

在统一的 split 和 slice evaluation framework 下复现三个 baseline，建立公平的比较基准。

**Why:** 不同 baseline 可能使用不同的评估方式，需要统一后才能公平比较 slice F1。

## Slice Definitions（Pre-registered，不可事后修改）

```python
SLICE_DEFINITIONS = {
    "graph_missing":    lambda deg: deg == 0,
    "low_degree":       lambda deg: 0 < deg < 10,
    "mid_degree":       lambda deg: 10 <= deg < 50,
    "high_degree":      lambda deg: deg >= 50,
    "disagreement":     None,  # pred_semantic != pred_graph (computed at runtime)
}
```

## Metrics Per Slice

- F1, Precision, Recall
- Coverage (% of test set in this slice)
- ECE, Brier score（如果模型输出概率）

## Task 1: LMbot Baseline

**Repo**: `LLMbot/baseline/`  
**Entry**: `LLMbot/baseline/core/main.py`  
**Split**: 使用 `datasets/TwiBot-20/` 下现有 split（需确认 LMbot 是否接受外部 split）  
**Expected global F1**: ~86%  
**Output**: `results/lmbot_baseline/slice_eval.json`

**Key questions to resolve**:
- LMbot 的 `reset_split()` 是否会覆盖现有 split？需要传入 `--no_reset_split` 或类似参数
- LMbot 输出的 embedding 格式是否与 LLMbot 兼容？

## Task 2: BotBR Baseline

**Repo**: `botbr/`  
**Entry**: `Twitter-GNN.py`  
**Split**: 直接加载 `train_idx.pt`/`val_idx.pt`/`test_idx.pt`（已确认兼容）  
**Expected global F1**: ~86.8%  
**Output**: `results/botbr_baseline/slice_eval.json`

**Key questions to resolve**:
- BotBR 需要 `cat_properties_tensor.pt`, `num_properties_tensor.pt`, `des_tensor.pt`, `tweets_tensor.pt`
- 这些文件是否在 `datasets/TwiBot-20/` 下？需要检查

## Task 3: HyperScan Baseline

**Repo**: `HyperScan/`  
**Entry**: `MGTAB/train.py`  
**Status**: ⚠️ 原始设计针对 MGTAB，需确认是否有 TwiBot-20 适配  
**Expected global F1**: ~87.2%（论文报告）  
**Output**: `results/hyperscan_baseline/slice_eval.json`

**Decision**: 如果适配成本 > 3天，改为直接引用论文结果

## Task 4: Unified Slice Evaluation Script

**Output**: `LLMbot/baseline/analysis/slice_evaluation.py`  
**Requirements**:
- 接受任意模型的 `(logits, labels, node_degrees)` 作为输入
- 输出标准化的 slice report（JSON + CSV）
- 所有 baseline 和我们的方法使用同一个脚本

## Success Criteria

- [ ] LMbot 复现 F1 在论文报告 ±0.5% 以内
- [ ] BotBR 复现 F1 在论文报告 ±0.5% 以内
- [ ] 所有方法使用同一套 split 和 slice definitions
- [ ] Slice evaluation report 包含所有 pre-registered slices


---

## Historical Context Update (2026-04-23)

- This plan was authored while the wiki still routed active-method context through the `LLMbot/code` line.
- The rw1/rw4 validation notes above remain historical facts about `LLMbot/code`; they were not reclassified as current `LLMbot/baseline` behavior.
- After the 2026-04-23 mainline switch, current sessions should start from `LLMbot/baseline/` and use `docs/wiki/project/mainline_switch_context_2026-04-23.md` for routing.
- Future reproduction notes should stay explicit about whether they refer to `LLMbot/baseline/core`, `LLMbot/baseline/baselines`, or historical `LLMbot/code` runs.
