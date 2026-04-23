# Idea Discovery Report: Beyond Calibration-Guided Graph Rewrite

**Direction**: 寻找比当前 dual_router + calibration-guided rewrite 更强、但仍适配现有代码的方案  
**Date**: 2026-04-07  
**Pipeline**: Literature Survey → Idea Generation (GPT-5.4) → Feasibility Ranking → Novelty Check  
**Model**: Claude Opus 4.6 + GPT-5.4 (via Codex MCP)

---

## Executive Summary

基于RESEARCH_BRIEF.md的约束和现有实现，通过文献调研和GPT-5.4生成了10个候选方案。**推荐优先实现Top 3**：

1. **🥇 Selective Abstention Guided Graph Updates** (86/100) — 图专家在不可靠时主动弃权，语义专家接管
2. **🥈 Counterfactual Neighborhood Consistency Regularization** (80/100) — 训练对邻域扰动的鲁棒性
3. **🥉 Slice-Calibrated Risk Coverage Detection Framework** (83/100) — 分析驱动的可靠性诊断框架

这三个方案均比当前硬图重写更有原则性，实现周期2-6天，且与现有calibration模块高度兼容。

---

## Phase 1: Literature Landscape

### 搜索策略
- **关键词组合**:
  - "graph neural network calibration uncertainty 2025 2026"
  - "LLM graph structure learning text-attributed graph"
  - "graph rewiring heterophily social network bot detection"
  - "trustworthy graph neural network noisy topology"
  - "selective graph usage confidence-aware GNN"
  - "adaptive edge weighting heterophilous graph learning"

### 文献综合发现

#### 1. 当前方法的文献对齐度（来自implementation_literature_alignment）

| 论文 | 对齐度 | 仓库证据 | 剩余gap |
|------|--------|----------|---------|
| Label-free Node Classification (ICLR'24) | ✅ Strong | q_sem/u_sem作为语义信任 | - |
| GATS (NeurIPS'22) | ✅ Strong | q_graph作为触发信号而非边oracle | - |
| Locality-Aware Graph Rewiring (ICLR'24) | ✅ Strong | 局部稀疏重写+预算 | 缺少局部性保持的量化证据 |
| ENGINE (IJCAI'24) | ✅ Strong | 离线enrichment+一次性重训 | - |
| Harnessing Explanations (ICLR'24) | ⚠️ Partial | explanation cache存在 | 仍是模板化而非LLM生成 |
| SKETCH (ACL'25) | ⚠️ Partial | 超越q_final→GNN | 语义-结构聚合仍未联合优化 |

#### 2. 识别的结构性gap

- **Gap 1**: 当前硬重写假设"一种干预适配所有可疑节点"，但异质性≠噪声
- **Gap 2**: 图校准(q_graph)是节点级信号，不等于边级信任
- **Gap 3**: 相似度剪枝已被替换，但候选召回仍依赖embedding相似度
- **Gap 4**: 解释增强是基础设施级，非文献完备的语义解释模块
- **Gap 5**: 缺少"何时不应使用图"的显式决策机制

#### 3. 相关前沿工作（需novelty check）

- **UnGSL** (ICLR'25): 不确定性引导的结构学习
- **ADMP-GNN** (ICLR'24): 自适应消息传递深度
- **ProtoGNN** (ICLR'23): 原型引导的图学习
- **GRAND** (NeurIPS'20): 一致性正则化
- **Conformal Prediction for GNNs** (ICLR'24): 图不确定性量化
- **TopGateGNN** (IJCNLP'25): 选择性消息门控（bot detection）

---

## Phase 2: Generated Ideas & Ranking

### 完整候选列表（10个）

| Rank | Idea | Novelty | Feasibility | Narrative | Total | 实现周期 |
|------|------|---------|-------------|-----------|-------|----------|
| 1 | #9 Selective Abstention Guided Graph Updates | 22/30 | 26/30 | 38/40 | **86/100** | 3-5天 |
| 2 | #10 Slice-Calibrated Risk Coverage Detection | 16/30 | 29/30 | 38/40 | **83/100** | 2-4天 |
| 3 | #5 Counterfactual Neighborhood Consistency | 21/30 | 22/30 | 37/40 | **80/100** | 4-6天 |
| 4 | #2 Latent Edge Reliability Posterior | 23/30 | 19/30 | 37/40 | 79/100 | 7-10天 |
| 5 | #7 Structural Regime Routing | 19/30 | 21/30 | 35/40 | 75/100 | 5-7天 |
| 6 | #6 Disagreement-Driven Distillation | 17/30 | 27/30 | 30/40 | 74/100 | 3-4天 |
| 7 | #3 Confidence-Adaptive Propagation Depth | 16/30 | 25/30 | 31/40 | 72/100 | 4-5天 |
| 8 | #8 Signed Evidence Passing | 19/30 | 16/30 | 36/40 | 71/100 | 8-12天 |
| 9 | #4 Semantic Prototype Anchors | 13/30 | 26/30 | 28/40 | 67/100 | 3-4天 |
| 10 | #1 Soft Neighborhood Message Suppression | 10/30 | 27/30 | 26/40 | 63/100 | 2-3天 |

---

## Top 3 Recommended Ideas (详细分析)

### 🥇 Idea #9: Selective Abstention Guided Graph Updates

#### 核心机制
- 图专家在`q_graph`低或校准差时**主动弃权**（abstain）
- 弃权的消息组件或logits由语义引导替换
- 使用risk-coverage目标训练模型学习何时不应使用图证据

#### 为何优于当前重写
- **当前方法**: 编辑拓扑结构，希望预处理能修复图
- **本方法**: 直接将"选择性图不可靠性"建模为决策策略，无需拓扑手术
- **关键优势**: 更有原则，更易解释，避免"重写是否真正有效"的争议

#### 与现有代码的兼容性
- ✅ 直接复用`q_sem`, `q_graph`, `u_sem`
- ✅ 无需修改图结构，仅需在图专家forward中加入abstention逻辑
- ✅ 可与现有router框架无缝集成

#### 预期影响
- **最佳切片**: 分歧案例、高风险预测、图缺失节点
- **F1提升**: 可能modest，但**叙事清晰度大幅提升**
- **论文定位**: "Trustworthy Graph Usage via Calibrated Abstention"

#### 实现路线图（3-5天）
1. **Day 1**: 在`graph_rgt.py`中添加abstention mask计算
2. **Day 2**: 修改`dual_router_trainer.py`，在图专家输出中插入语义fallback
3. **Day 3**: 实现risk-coverage loss（可选，先用hard threshold baseline）
4. **Day 4-5**: 运行实验，生成abstention统计和case studies

#### Novelty Check
- ⚠️ 需与**Conformal Prediction for GNNs**和**CF-GNN**区分
- ✅ 差异点: 本方法是训练时学习abstention策略，而非测试时coverage保证
- ⚠️ 需与**selective prediction**文献对比（ICML'23 Selective Classification on Graphs）

---

### 🥈 Idea #5: Counterfactual Neighborhood Consistency Regularization

#### 核心机制
- 对分歧节点生成**局部反事实邻域**（mask可疑邻居或扰动边权重）
- 训练图专家在这些反事实下保持预测稳定，除非图置信度genuinely高
- 直接训练对观察到的邻域腐败的鲁棒性

#### 为何优于当前重写
- **当前方法**: 希望预处理修复图，但无法保证重写后的图真的更好
- **本方法**: 不改拓扑，而是训练模型对坏邻域的鲁棒性
- **关键优势**: 攻击精确的失败模式（corrupted neighborhoods），给出trustworthy AI鲁棒性故事

#### 与现有代码的兼容性
- ✅ 复用`trigger_score`和`disagreement_mask`识别需要正则化的节点
- ✅ 可在现有训练循环中加入consistency loss
- ⚠️ 需要实现邻域采样扰动逻辑（中等工程量）

#### 预期影响
- **最佳切片**: 分歧案例、脆弱的假阳性/假阴性
- **F1提升**: 预期在弱切片上有明显改善
- **论文定位**: "Robust Graph Learning via Counterfactual Neighborhood Consistency"

#### 实现路线图（4-6天）
1. **Day 1-2**: 实现邻域扰动生成器（mask edges, perturb weights）
2. **Day 3**: 添加consistency regularization loss到训练循环
3. **Day 4-5**: 超参调优（扰动强度、loss权重）
4. **Day 6**: 运行实验，分析鲁棒性改善

#### Novelty Check
- ⚠️ 需与**GRAND** (NeurIPS'20)区分
- ✅ 差异点: GRAND是全局一致性，本方法是**校准引导的选择性**反事实正则化
- ⚠️ 需与**adversarial training on graphs**文献对比

---

### 🥉 Idea #10: Slice-Calibrated Risk Coverage Detection Framework

#### 核心机制
- 构建切片检测器（degree, graph missingness, disagreement gap, calibration gap）
- 应用**切片特定的校正或重加权**
- 主要贡献不是F1，而是**可信切片诊断和选择性校正策略**

#### 为何优于当前重写
- **当前方法**: 如果增益modest，很难defend另一个图编辑启发式
- **本方法**: 即使F1增益有限，也能给出强论文——解释图方法在哪里失败、为何失败、如何可靠干预
- **关键优势**: 方法+分析的混合论文，而非纯方法论文

#### 与现有代码的兼容性
- ✅ 完全复用现有切片定义（low-degree, graph-missing, disagreement）
- ✅ 无需修改模型架构，仅需后处理分析
- ✅ 最快原型路径

#### 预期影响
- **最佳切片**: 所有切片（这是分析框架）
- **F1提升**: 可能最小，但**论文叙事最强**
- **论文定位**: "Reliability-Aware Bot Detection: When and How to Trust Graph Evidence"

#### 实现路线图（2-4天）
1. **Day 1**: 实现切片检测器和coverage curve生成
2. **Day 2**: 为每个切片设计校正策略（reweight, abstain, fallback）
3. **Day 3**: 运行实验，生成诊断报告
4. **Day 4**: 撰写分析narrative（可直接用于论文）

#### Novelty Check
- ⚠️ 需与**graph conformal prediction**和**coverage benchmarking**区分
- ✅ 差异点: 本方法是**校准驱动的切片诊断**，而非分布无关的coverage保证
- ✅ 在bot detection领域，这种reliability分析尚未充分探索

---

## Eliminated Ideas (不推荐)

### ❌ Idea #1: Soft Neighborhood Message Suppression
- **原因**: 与现有`graph_soft_gating`重叠度过高
- **现状**: 代码中已有message-level soft edge gating
- **建议**: 仅作为ablation保留，不作为主方法

### ❌ Idea #4: Semantic Prototype Anchors
- **原因**: 与ProtoGNN和BotHP (arXiv 2506.00989)重叠
- **现状**: 原型方法在bot detection中已有探索
- **建议**: 除非有更sharp的twist，否则难以通过审稿

### ⚠️ Idea #8: Signed Evidence Passing (延后)
- **原因**: 实现风险高（8-12天），架构改动大
- **建议**: 如果Top 3都失败，再考虑这个更激进的方案

---

## Phase 3: Novelty Verification (Deep Check)

### 必须验证的文献对比

| Idea | 需检查的论文 | 差异化要点 |
|------|-------------|-----------|
| #9 Abstention | CF-GNN (ICLR'24), Selective Classification on Graphs (ICML'23) | 训练时学习策略 vs 测试时coverage |
| #5 Counterfactual | GRAND (NeurIPS'20), Adversarial Training on Graphs | 校准引导的选择性 vs 全局一致性 |
| #10 Slice Framework | Graph Conformal Benchmarking (ICLR'24) | 校准驱动诊断 vs 分布无关保证 |
| #2 Edge Reliability | UnGSL (ICLR'25) | 概率后验 vs 不确定性加权 |
| #3 Adaptive Depth | ADMP-GNN (ICLR'24) | 校准分歧驱动 vs 节点自适应 |
| #6 Distillation | Cold Brew (ICLR'24), FedKG (Sensors'24) | 切片选择性 vs 全局蒸馏 |
| #7 Regime Routing | Structural Disparity (NeurIPS'23) | 校准+结构混合 vs 纯结构regime |

### 推荐的novelty check流程
1. **Phase 3.1**: 对Top 3进行深度文献搜索（arXiv + Semantic Scholar）
2. **Phase 3.2**: 使用`/novelty-check` skill验证每个idea
3. **Phase 3.3**: 如果Top 3中有idea被证明不novel，从Rank 4-7中提升替代

---

## Phase 4: Implementation Priority & Timeline

### 推荐实现顺序

#### 🎯 Sprint 1 (Week 1): Quick Wins
1. **Idea #10** (2-4天) — 最快产出，立即clarify当前方法的有效性
2. **Idea #9** (3-5天) — 如果#10显示abstention有潜力，立即跟进

#### 🎯 Sprint 2 (Week 2): Deep Method
3. **Idea #5** (4-6天) — 如果前两个成功，这是最强的方法论文候选

#### 🎯 Backup Plan
- 如果Top 3都不work，考虑**Idea #2** (Latent Edge Reliability) 作为更激进的方案
- 如果时间紧张，**Idea #6** (Distillation) 是最安全的fallback（3-4天）

### 并行策略（如果有多GPU或多人）
- **Track A**: Idea #10 (分析) + Idea #9 (abstention)
- **Track B**: Idea #5 (counterfactual) 独立开发
- **合并点**: Week 2 end，比较三者结果

---

## Phase 5: Paper Narrative Positioning

### 如果Idea #9成功
**Title**: "Selective Graph Abstention for Reliable Social Bot Detection"  
**Claim**: 图专家应学会何时不使用图，而非盲目融合或重写  
**Venue**: NeurIPS/ICML (trustworthy ML track) 或 ACL (social computing)

### 如果Idea #5成功
**Title**: "Counterfactual Robustness for Graph Learning on Noisy Social Networks"  
**Claim**: 训练对邻域腐败的鲁棒性优于预处理修复  
**Venue**: ICLR/NeurIPS (robustness track)

### 如果Idea #10成功（但方法增益modest）
**Title**: "When Should We Trust Graph Evidence? A Calibration-Driven Analysis for Bot Detection"  
**Claim**: 分析驱动的可靠性诊断，而非强制方法创新  
**Venue**: ACL Findings / EMNLP / WWW (analysis paper track)

### 如果多个成功
**Title**: "Reliability-Aware Graph Learning: Abstention, Robustness, and Selective Usage"  
**Claim**: 统一框架，包含abstention + counterfactual + slice diagnostics  
**Venue**: 顶会main track (ICML/NeurIPS/ICLR)

---

## Risks & Mitigation

### Risk 1: Top 3都无法beat现有baseline
**Mitigation**: 
- Idea #10本身就是分析论文，不依赖F1提升
- 如果#9和#5都失败，说明问题不在图使用策略，而在语义或数据质量
- 此时pivot到negative result paper: "Why Graph Rewrite Doesn't Help: A Diagnostic Study"

### Risk 2: Novelty check发现Top 3已被做过
**Mitigation**:
- 立即提升Rank 4-7中的候选
- 优先级: #2 (最novel但难) > #7 (interpretable) > #6 (最安全)

### Risk 3: 实现时间超出预期
**Mitigation**:
- Idea #10只需2-4天，作为保底方案
- 如果Week 1结束仍无进展，立即切换到Idea #6 (distillation, 3-4天)

---

## Next Steps (Immediate Actions)

### ✅ Before Starting Implementation
1. [x] 运行`/novelty-check`验证Top 3的novelty → 见 Phase 3 结果
2. [ ] 阅读标记的必查论文（CF-GNN, GRAND, Conformal Prediction）
3. [ ] 确认现有`rw0-rw4`实验结果（如果还没跑，先跑baseline）

### ✅ Week 1 Sprint Plan
1. [x] **Day 1-2**: Novelty check + research review 完成
2. [ ] **Day 3**: 运行 B2 kill test (rw4_disagreement_local vs same-trigger defer)
3. [ ] **Day 4-5**: 如果 B2 通过，运行 B3-B4 ablations

### ✅ Handoff to Workflow 2
- 当 B2+B3 kill tests 通过后，立即调用`/auto-review-loop`
- 输入文件: 本报告 + RESEARCH_BRIEF.md + 新实验结果
- 目标: 迭代直到submission-ready

---

## Phase 3: Novelty Check Results (2026-04-08)

**Method checked**: Selective Abstention Guided Graph Updates → reframed as Reliability-Guided Test-Time Graph Surgery

### Novelty Scores

| Claim | Novelty | Closest Prior Work |
|-------|---------|-------------------|
| Dual-expert abstention | LOW | Mozannar & Sontag 2020 (L2D), Verma et al. 2023 |
| Calibration-driven trigger | MEDIUM | Calibrated Selective Classification, GATS |
| Learned abstention (risk-coverage) | LOW | SelectiveNet (ICML'19), NCwR (TMLR 2025) |
| Bot detection application | LOW | TopGateGNN (IJCNLP'25), BotLGT |
| No topology modification | LOW | Design choice, not contribution |

**Overall novelty score**: 4/10 as "selective abstention"

### Reframing Decision
After GPT-5.4 deep analysis, the method was reframed from "selective abstention" to **"propagation-level intervention"**:
- Old framing: graph expert abstains → semantic takes over (output-level)
- New framing: expert disagreement triggers local edge pruning + re-propagation (propagation-level)
- Key gap vs NCwR/SelectiveNet/L2D: they operate at prediction level; this operates at message-passing level

### Minimum Delta for Top Venue
1. Same-trigger defer baseline (kill test)
2. Formal proposition: pruning reduces expected risk under monotone aggregation
3. Triggered-node characterization (higher heterophily)
4. 3-5 seeds with significance tests
5. TwiBot-22 or synthetic corruption study

**Full report**: `doc/idea_discovery_2026-04-07_21-31/NOVELTY_CHECK_REPORT.md`

---

## Phase 4: Research Review Results (2026-04-08)

**Reviewer**: GPT-5.4 (xhigh), 3 rounds  
**Initial score**: 3/10 | **Projected with experiments**: 6/10 | **Full package**: 8/10

### Key Accepted Criticisms
- Trigger definition fixed: `pred_sem != pred_graph_raw AND trust_i*(1-q_graph) > τ`
- Same-trigger defer baseline is mandatory kill test
- Claim narrowed to "targeted repair," not "universal superiority"

### Venue Projection
| Outcome | Venue |
|---------|-------|
| Optimistic (B2+B3 pass, 5 seeds) | WWW 2026 best fit, AAAI viable |
| Neutral (weak B2) | TMLR, ECML-PKDD |
| Pessimistic (B2 kill) | Workshop only |

### Paper Title (Optimistic)
"When Disagreement Signals Bad Neighborhoods: Reliability-Guided Test-Time Graph Pruning for Social Bot Detection"

**Full report**: `doc/idea_discovery_2026-04-07_21-31/RESEARCH_REVIEW_REPORT.md`

---

## Phase 4.5: Refined Proposal & Experiment Plan (2026-04-08)

**Final Method Thesis**: Reliability-Guided Test-Time Graph Surgery — use expert disagreement as a structural warning signal to trigger local edge pruning + re-propagation, outperforming same-trigger output-level deferral.

**Verdict**: REVISE — thesis clean; submission depends on B2 kill test

### Deliverables
- `refine-logs/FINAL_PROPOSAL.md` — refined mechanism + claims
- `refine-logs/EXPERIMENT_PLAN.md` — 8 blocks, ~9.5 GPU hours
- `refine-logs/EXPERIMENT_TRACKER.md` — run table with kill test gates
- `refine-logs/PIPELINE_SUMMARY.md` — one-page summary
- `refine-logs/REVIEW_SUMMARY.md` — full review history

### First Commands to Run
```powershell
# 1. rw1_calibration_only seed 42
python code/main.py --system_mode dual_router --report_group_id rw1_calibration_only --seed 42 --epochs 30

# 2. Precompute explanation cache
python code/explanation_cache_precompute.py --artifact_dir saved_artifacts\rw1_calibration_only\seed_42

# 3. rw4_disagreement_local seed 42 (B2 kill test)
python code/main.py --system_mode dual_router --report_group_id rw4_disagreement_local --seed 42 --epochs 30
```

---

## Appendix: Full Idea Descriptions

### Idea #1: Disagreement-Aware Soft Neighborhood Message Suppression
**Core**: 用学习的边/消息权重替代硬剪枝，权重依赖q_sem, q_graph, 语义logit分歧, degree和局部结构。可疑邻居不被删除，而是消息被软衰减。  
**Why better**: 可微且可逆，坏的重写决策不会永久损坏图。针对实际失败模式（有害消息传递）而非拓扑本身。  
**Best for**: 分歧案例和高度污染邻域。  
**Difficulty**: Low-Medium. **Narrative**: Strong.

### Idea #2: Latent Edge Reliability Posterior Learning
**Core**: 将每条边视为具有潜在可靠性变量，先验来自结构线索（互惠性、度数比、共享邻居），证据来自专家分歧和校准。在估计边可靠性和在加权邻接上重训图专家之间交替。  
**Why better**: 给出原则性概率故事而非启发式剪枝/添加规则。分离"边存在"和"边有用"，这正是你的问题。  
**Best for**: 异质性区域的噪声和误导边。  
**Difficulty**: Medium. **Narrative**: Strong.

### Idea #3: Confidence-Adaptive Propagation Depth Per Node
**Core**: 从q_sem, q_graph, degree, missingness和局部同质性估计预测节点级传播深度或teleport系数。可靠图节点使用正常传播，风险节点使用浅/自重传播+更强语义残差。  
**Why better**: 核心问题可能是过度传播到坏邻域，而非仅仅坏边。个性化传播更局部，更符合你关心的失败切片。  
**Best for**: 低度节点、图缺失节点、边界案例。  
**Difficulty**: Low-Medium. **Narrative**: Strong.

### Idea #4: Semantic Prototype Anchors With Uncertainty
**Core**: 构建稳定的语义原型或聚类中心，让不确定节点attend这些原型作为虚拟证据。锚点强度由语义校准控制而非原始成对相似度或硬语义边。  
**Why better**: 原型锚点比成对语义链接噪声更小，当图证据缺失或不可靠时提供更清晰的fallback。这是软记忆机制，而非另一个重写启发式。  
**Best for**: 图缺失节点和稀疏用户。  
**Difficulty**: Low. **Narrative**: Strong.

### Idea #6: Disagreement-Driven Teacher Student Graph Distillation
**Core**: 仅在q_sem >> q_graph的切片、图缺失或度数极低时，使用语义专家作为选择性教师。仅在这些切片上将logits或margins蒸馏到图专家，而保持图强区域不变。  
**Why better**: 直接纠正图归纳偏差而不编辑拓扑。这是targeted supervision，而非通用融合。  
**Best for**: 低度、图缺失、分歧节点。  
**Difficulty**: Low. **Narrative**: Medium-Strong.

### Idea #7: Structural Regime Routing Before Propagation
**Core**: 使用校准+局部结构描述符将每个节点或ego-network分类为regime（可信同质、信息异质、稀疏、腐败）。每个regime使用不同的传播规则、残差强度或损失权重。  
**Why better**: 重写假设一种干预适配所有可疑节点，但异质性并非总是噪声。Regime routing给出更清晰的解释：何时应信任、降权或反转图证据。  
**Best for**: 解释分歧案例（图因不同原因错误）。  
**Difficulty**: Medium. **Narrative**: Strong.

### Idea #8: Signed Evidence Passing For Heterophily
**Core**: 将邻域证据分为支持和矛盾通道，符号从语义分歧、原型距离或校准不匹配推断。图专家在预测前分别聚合正负证据。  
**Why better**: 硬剪枝丢弃矛盾，但在异质性社交图中矛盾本身可能是信息。这将冲突转化为可用信号而非纯噪声。  
**Best for**: bot-human边界节点和异质性邻域。  
**Difficulty**: Medium-High. **Narrative**: Strong.

---

## Document Metadata

- **Generated by**: Claude Opus 4.6 + GPT-5.4 (Codex MCP)
- **Worktree**: lovely-churning-duckling
- **Base commit**: 32691fc
- **Related files**:
  - `LLMbot/RESEARCH_BRIEF.md`
  - `LLMbot/refine-logs/EXPERIMENT_PLAN.md`
  - `LLMbot/refine-logs/CALIBRATION_GRAPH_REWRITE_METHOD.md`
  - `LLMbot/doc/literature_method_review_2026-04-01.md`
  - `LLMbot/doc/implementation_literature_alignment_2026-04-01.md`

---

**Status**: ✅ Ready for Phase 3 (Novelty Check) and Phase 4 (Implementation)  
**Recommended next command**: `/novelty-check "Selective Abstention Guided Graph Updates for social bot detection"`
