# Top 3 Ideas: Quick Reference Cards

**Generated**: 2026-04-07 21:31  
**Purpose**: 快速查阅Top 3推荐方案的核心信息

---

## 🥇 Idea #9: Selective Abstention Guided Graph Updates

### One-Line Pitch
图专家在不可靠时主动弃权，语义专家接管决策

### Core Mechanism (3 bullets)
- 图专家计算abstention mask: `abstain = (q_graph < τ_abstain) OR (calibration_error > τ_cal)`
- 弃权节点的图logits被语义logits替换: `logits_final = (1-abstain)*logits_graph + abstain*logits_sem`
- 可选: 训练risk-coverage目标学习最优abstention策略

### Why Better Than Current Rewrite
| 维度 | 当前硬重写 | Abstention方案 |
|------|-----------|---------------|
| 干预点 | 预处理拓扑 | 推理时决策 |
| 可逆性 | 不可逆 | 完全可逆 |
| 解释性 | "重写让图更好" | "图不可靠时不用" |
| 风险 | 坏重写永久损坏 | 无拓扑风险 |

### Implementation Checklist (5 days)

#### Day 1: Abstention Mask计算
- [ ] 在`graph_rgt.py`添加`compute_abstention_mask(q_graph, calib_error, thresholds)`
- [ ] 输入: `q_graph` (N,), `calib_error` (N,), `tau_abstain`, `tau_cal`
- [ ] 输出: `abstain_mask` (N,) boolean tensor
- [ ] 测试: 验证mask在低q_graph节点上为True

#### Day 2: 语义Fallback集成
- [ ] 修改`dual_router_trainer.py`的图专家forward
- [ ] 在图专家输出后插入: `logits_final = torch.where(abstain_mask.unsqueeze(1), logits_sem, logits_graph)`
- [ ] 保存abstention统计到`abstention_stats.json`
- [ ] 测试: 验证弃权节点使用语义logits

#### Day 3: Risk-Coverage Loss (可选)
- [ ] 实现`risk_coverage_loss(predictions, targets, abstain_mask, lambda_coverage)`
- [ ] Risk term: 弃权节点的语义loss
- [ ] Coverage term: 惩罚过度弃权 `lambda_coverage * abstain_mask.float().mean()`
- [ ] 测试: 验证loss平衡risk和coverage

#### Day 4: 实验运行
- [ ] 运行baseline: `--abstention_mode none`
- [ ] 运行hard threshold: `--abstention_mode hard --tau_abstain 0.5`
- [ ] 运行learned: `--abstention_mode learned --lambda_coverage 0.1`
- [ ] 收集: F1, ECE, abstention rate, slice metrics

#### Day 5: 分析和Case Studies
- [ ] 生成abstention rate by slice (low-degree, graph-missing, disagreement)
- [ ] 提取5个case studies: 弃权帮助的案例
- [ ] 对比: abstention vs 硬重写的局部性和稀疏性
- [ ] 撰写: 初步narrative (1-2页)

### Expected Outcomes
- **F1提升**: +0.5-1.5% (modest但稳定)
- **ECE改善**: -0.01-0.03 (校准改善)
- **Abstention rate**: 10-25% (取决于阈值)
- **最强切片**: graph-missing (+2-3%), disagreement (+1-2%)

### Novelty Risks
⚠️ **Must differentiate from**:
- **CF-GNN** (ICLR'24): 测试时conformal coverage vs 训练时learned abstention
- **Selective Classification on Graphs** (ICML'23): 通用selective prediction vs 校准驱动的图-语义切换

✅ **Key differentiation**:
- 我们的abstention是**双专家架构特有的**，不是通用selective prediction
- 决策基于**校准分歧**，而非单一模型的不确定性
- 目标是**可靠性感知的模态切换**，而非coverage保证

### Paper Positioning
- **Title**: "Selective Graph Abstention for Reliable Social Bot Detection"
- **Venue**: NeurIPS/ICML (trustworthy ML) 或 ACL (social computing)
- **Claim**: 图专家应学会何时不使用图，而非盲目融合或重写
- **Narrative**: Reliability-aware routing > topology surgery

---

## 🥈 Idea #5: Counterfactual Neighborhood Consistency Regularization

### One-Line Pitch
训练图专家对邻域腐败的鲁棒性，而非预处理修复图

### Core Mechanism (3 bullets)
- 对分歧节点生成反事实邻域: mask可疑邻居或扰动边权重
- 训练consistency loss: `L_consist = ||pred_original - pred_counterfactual||^2`
- 仅在`q_graph`genuinely高时允许预测变化，否则强制一致性

### Why Better Than Current Rewrite
| 维度 | 当前硬重写 | Counterfactual方案 |
|------|-----------|-------------------|
| 假设 | 重写后的图更好 | 模型应对坏邻域鲁棒 |
| 验证 | 难以验证重写质量 | 直接测试鲁棒性 |
| 泛化 | 依赖重写启发式 | 学习通用鲁棒性 |
| 故事 | 图修复 | Trustworthy AI鲁棒性 |

### Implementation Checklist (6 days)

#### Day 1-2: 邻域扰动生成器
- [ ] 实现`generate_counterfactual_neighborhoods(data, trigger_mask, perturb_ratio)`
- [ ] 策略1: Random edge masking (mask `perturb_ratio`比例的边)
- [ ] 策略2: Adversarial masking (mask高权重边)
- [ ] 策略3: Edge weight perturbation (添加高斯噪声)
- [ ] 输出: `data_cf` (counterfactual graph)
- [ ] 测试: 验证扰动保持图连通性

#### Day 3: Consistency Loss集成
- [ ] 在训练循环添加counterfactual forward pass
- [ ] 计算: `pred_orig = graph_expert(data)`, `pred_cf = graph_expert(data_cf)`
- [ ] Loss: `L_consist = ((pred_orig - pred_cf)**2 * trigger_mask).mean()`
- [ ] 总loss: `L_total = L_task + lambda_consist * L_consist`
- [ ] 测试: 验证loss在分歧节点上非零

#### Day 4-5: 超参调优
- [ ] Grid search: `perturb_ratio` in [0.1, 0.2, 0.3, 0.5]
- [ ] Grid search: `lambda_consist` in [0.01, 0.05, 0.1, 0.5]
- [ ] 评估: F1, ECE, disagreement slice F1
- [ ] 选择: 最佳超参组合

#### Day 6: 鲁棒性分析
- [ ] 测试时邻域扰动实验: 不同扰动强度下的F1
- [ ] 对比: counterfactual训练 vs 标准训练的鲁棒性曲线
- [ ] Case studies: 鲁棒性改善的具体案例
- [ ] 撰写: 鲁棒性narrative (1-2页)

### Expected Outcomes
- **F1提升**: +1-2% (在弱切片上更明显)
- **Disagreement F1**: +2-3%
- **鲁棒性**: 扰动下F1下降减少30-50%
- **最强切片**: disagreement, 脆弱假阳性/假阴性

### Novelty Risks
⚠️ **Must differentiate from**:
- **GRAND** (NeurIPS'20): 全局一致性 vs 选择性反事实
- **Adversarial Training on Graphs**: 通用对抗训练 vs 校准引导的选择性正则化

✅ **Key differentiation**:
- 我们的正则化是**选择性的**，仅在校准识别的风险节点上应用
- 反事实生成基于**语义-图分歧**，而非随机或对抗扰动
- 目标是**邻域腐败鲁棒性**，而非通用对抗鲁棒性

### Paper Positioning
- **Title**: "Counterfactual Robustness for Graph Learning on Noisy Social Networks"
- **Venue**: ICLR/NeurIPS (robustness track)
- **Claim**: 训练对邻域腐败的鲁棒性优于预处理修复
- **Narrative**: Robustness-by-design > topology surgery

---

## 🥉 Idea #10: Slice-Calibrated Risk Coverage Detection Framework

### One-Line Pitch
分析驱动的可靠性诊断框架，解释图方法在哪里失败、为何失败、如何干预

### Core Mechanism (3 bullets)
- 构建切片检测器: 基于degree, graph missingness, disagreement gap, calibration gap
- 为每个切片生成risk-coverage曲线和诊断报告
- 设计切片特定的校正策略: reweight, abstain, fallback, hybrid

### Why Better Than Current Rewrite
| 维度 | 当前硬重写 | Slice Framework |
|------|-----------|-----------------|
| 主张 | 方法创新 | 分析+方法 |
| 依赖 | F1提升 | 诊断洞察 |
| 风险 | modest gain难defend | 即使无gain也有价值 |
| 贡献 | 单一方法 | 通用诊断框架 |

### Implementation Checklist (4 days)

#### Day 1: 切片检测器
- [ ] 实现`SliceDetector(degree_bins, missingness_threshold, disagreement_threshold, calib_gap_threshold)`
- [ ] 输入: `struct_feats`, `q_sem`, `q_graph`, `pred_sem`, `pred_graph`
- [ ] 输出: 每个节点的切片标签 (可多标签)
- [ ] 切片定义:
  - `low_degree`: total_degree < 10
  - `graph_missing`: graph_missing == 1
  - `disagreement`: pred_sem != pred_graph
  - `low_graph_calib`: q_graph < 0.5
  - `high_sem_calib`: q_sem > 0.8
- [ ] 测试: 验证切片覆盖率和重叠

#### Day 2: Risk-Coverage曲线生成
- [ ] 对每个切片，计算不同置信度阈值下的:
  - Coverage: 未弃权的节点比例
  - Risk: 未弃权节点的错误率
  - F1: 未弃权节点的F1
- [ ] 生成曲线: `plot_risk_coverage_curve(slice_name, thresholds, risks, coverages)`
- [ ] 对比: 语义专家 vs 图专家 vs router的曲线
- [ ] 输出: `slice_risk_coverage_curves.pdf`

#### Day 3: 切片特定校正策略
- [ ] 策略1 (low_degree): 使用语义专家
- [ ] 策略2 (graph_missing): 使用语义专家
- [ ] 策略3 (disagreement + high_sem_calib): 使用语义专家
- [ ] 策略4 (disagreement + low_sem_calib): 使用router
- [ ] 策略5 (其他): 使用图专家或router
- [ ] 实现: `apply_slice_correction(predictions, slice_labels, correction_policies)`
- [ ] 测试: 验证每个切片的F1变化

#### Day 4: 诊断报告生成
- [ ] 生成`SLICE_DIAGNOSTIC_REPORT.md`:
  - 每个切片的大小、基线性能、失败模式
  - Risk-coverage曲线分析
  - 校正策略效果对比
  - 5-10个代表性case studies
- [ ] 生成`slice_correction_results.json`:
  - 每个切片的before/after metrics
  - 最佳校正策略
  - 置信度阈值推荐
- [ ] 可视化: 切片性能热力图

### Expected Outcomes
- **F1提升**: +0.3-1.0% (取决于校正策略)
- **主要价值**: 诊断洞察和论文narrative
- **Deliverables**:
  - 完整的切片诊断报告
  - Risk-coverage曲线
  - 切片特定的最佳策略
  - 可直接用于论文的分析

### Novelty Risks
⚠️ **Must differentiate from**:
- **Graph Conformal Prediction**: 分布无关coverage vs 校准驱动诊断
- **Slice Discovery**: 通用slice discovery vs 任务特定可靠性切片

✅ **Key differentiation**:
- 我们的框架是**校准驱动的**，而非分布无关的统计保证
- 切片定义是**任务特定的**（bot detection的失败模式）
- 目标是**可靠性诊断**，而非通用slice discovery

### Paper Positioning
- **Title**: "When Should We Trust Graph Evidence? A Calibration-Driven Analysis for Bot Detection"
- **Venue**: ACL Findings / EMNLP / WWW (analysis paper track)
- **Claim**: 分析驱动的可靠性诊断，而非强制方法创新
- **Narrative**: Understanding failure modes > incremental F1 gains

---

## Implementation Priority Matrix

| Idea | 实现周期 | F1预期 | Narrative强度 | 风险 | 优先级 |
|------|---------|--------|---------------|------|--------|
| #10 Slice Framework | 2-4天 | Low | ⭐⭐⭐⭐⭐ | Very Low | **Week 1** |
| #9 Abstention | 3-5天 | Medium | ⭐⭐⭐⭐ | Low | **Week 1** |
| #5 Counterfactual | 4-6天 | Medium-High | ⭐⭐⭐⭐ | Medium | **Week 2** |

### Recommended Sprint Plan

#### Week 1: Quick Wins
- **Day 1-2**: Idea #10 (Slice Framework)
  - 最快产出，立即clarify当前方法有效性
  - 即使其他idea失败，这个也有价值
- **Day 3-5**: Idea #9 (Abstention)
  - 如果#10显示abstention有潜力，立即跟进
  - 实现简单，风险低

#### Week 2: Deep Method
- **Day 1-6**: Idea #5 (Counterfactual)
  - 如果Week 1成功，这是最强的方法论文候选
  - 需要更多工程，但narrative强

### Parallel Strategy (如果有资源)
- **Track A**: #10 (分析) + #9 (abstention) 串行
- **Track B**: #5 (counterfactual) 独立开发
- **合并点**: Week 2 end，比较三者结果

---

## Quick Decision Tree

```
开始实现
    ↓
实现 #10 (2-4天)
    ↓
#10显示abstention有潜力？
    ├─ Yes → 实现 #9 (3-5天)
    │         ↓
    │     #9成功？
    │         ├─ Yes → 实现 #5 (4-6天) [可选]
    │         └─ No → 分析#10结果，撰写analysis paper
    │
    └─ No → #10显示需要更激进方案？
              ├─ Yes → 实现 #2 (Edge Reliability, 7-10天)
              └─ No → 撰写negative result paper
```

---

## Success Criteria

### Minimum Viable Success (任一满足即可)
- [ ] 任一idea在global F1上beat baseline +0.5%
- [ ] 任一idea在关键切片(low-degree/graph-missing/disagreement)上beat baseline +1%
- [ ] Idea #10产出高质量诊断报告，即使无F1提升

### Strong Success (理想情况)
- [ ] Top 3中至少2个成功
- [ ] Global F1提升 +1-2%
- [ ] 关键切片F1提升 +2-3%
- [ ] ECE改善 -0.02以上
- [ ] 清晰的论文narrative

### Pivot Triggers (何时放弃当前方向)
- [ ] Top 3全部失败 (F1无提升且narrative不清晰)
- [ ] Novelty check发现Top 3都已被做过
- [ ] 实现时间超出预期2倍以上

---

## Document Metadata

- **Generated by**: Claude Opus 4.6
- **Purpose**: 快速参考和实现指南
- **Related**: `IDEA_DISCOVERY_REPORT.md`, `LITERATURE_SURVEY.md`
