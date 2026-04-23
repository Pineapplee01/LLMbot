# Literature Survey Summary

**Date**: 2026-04-07 21:31  
**Task**: 为calibration-guided graph rewrite寻找更强替代方案  
**Search Scope**: 2024-2026年图学习、可信AI、社交机器人检测领域

---

## Search Queries Executed

1. "graph neural network calibration uncertainty 2025 2026"
2. "LLM graph structure learning text-attributed graph 2025 2026"
3. "graph rewiring heterophily social network bot detection 2025 2026"
4. "trustworthy graph neural network noisy topology"
5. "selective graph usage confidence-aware GNN 2025"
6. "adaptive edge weighting heterophilous graph learning"
7. "graph structure refinement semantic guidance neural network 2026"

---

## Key Findings from Existing Documentation

### 从 `literature_method_review_2026-04-01.md` 提取

#### Still-Central Anchor Papers (强对齐)

1. **Label-free Node Classification on Graphs with LLMs** (ICLR'24)
   - 核心观点: LLM作为可信教师或伪标签提供者
   - 当前对齐: ✅ Strong - 使用q_sem/u_sem作为信任信号
   - 启发: 语义专家应作为监督源，而非全图oracle

2. **GATS: What Makes GNNs Miscalibrated?** (NeurIPS'22)
   - 核心观点: 节点级校准≠边级信任
   - 当前对齐: ✅ Strong - q_graph作为触发信号而非边oracle
   - 启发: 校准信号需要显式转换为边级决策

3. **Locality-Aware Graph Rewiring in GNNs** (ICLR'24)
   - 核心观点: 好的重写保持局部性和稀疏性
   - 当前对齐: ✅ Strong - 局部稀疏重写+预算
   - Gap: 缺少局部性保持的量化证据

4. **ENGINE: Efficient Tuning for LLMs on Textual Graphs** (IJCAI'24)
   - 核心观点: 效率和稀疏交互是一等设计目标
   - 当前对齐: ✅ Strong - 离线enrichment+一次性重训
   - 启发: 避免多轮迭代和在线LLM调用

5. **GNN-Ret: Graph Neural Network Enhanced Retrieval** (NAACL'25)
   - 核心观点: 图结构最适合候选召回和重排序
   - 当前对齐: ✅ Strong - 候选召回与最终决策分离
   - 启发: embedding相似度仅用于召回，校准用于决策

#### Partial Match Papers (部分对齐，存在gap)

6. **Harnessing Explanations: LLM-to-LM Interpreter** (ICLR'24)
   - 核心观点: 解释是合法的语义监督通道
   - 当前对齐: ⚠️ Partial - explanation cache存在
   - Gap: 仍是模板化而非LLM生成的rationale

7. **SKETCH: Decoupled Aggregation for TAG Learning** (ACL'25)
   - 核心观点: 固定q_final→GNN管道留下性能空间
   - 当前对齐: ⚠️ Partial - 超越q_final→GNN
   - Gap: 语义-结构聚合仍未联合优化

8. **TANS: Can LLMs Convert Graphs to Text-Attributed Graphs?** (NAACL'25)
   - 核心观点: 拓扑感知的语义增强是合法的
   - 当前对齐: ⚠️ Partial - 拓扑摘要存在
   - Gap: 非真正的LLM生成拓扑感知描述

---

## Identified Structural Gaps in Current Method

### Gap 1: 硬重写的"一刀切"假设
- **问题**: 当前方法假设一种干预适配所有可疑节点
- **文献证据**: 异质性图中，不同类型的"坏边"需要不同处理
- **启发方向**: 
  - Regime-based routing (不同结构模式用不同策略)
  - Selective abstention (学习何时不用图)
  - Signed evidence (区分支持/矛盾证据)

### Gap 2: 节点校准≠边信任
- **问题**: q_graph是节点级信号，直接用于边决策有gap
- **文献证据**: GATS明确指出nodewise calibration不等于edgewise trust
- **启发方向**:
  - Latent edge reliability (显式建模边级可靠性)
  - Pairwise calibration features (当前已部分实现)
  - Edge-level uncertainty quantification

### Gap 3: 候选召回仍依赖embedding相似度
- **问题**: 虽然最终决策用校准，但召回阶段仍是相似度
- **文献证据**: GNN-Ret强调召回与决策应分离但协同
- **启发方向**:
  - Learned retrieval (用校准信号训练召回器)
  - Multi-stage filtering (粗召回+精排序)
  - Prototype-based anchors (减少成对相似度依赖)

### Gap 4: 解释增强的质量瓶颈
- **问题**: 当前explanation cache是模板化+哈希，非LLM生成
- **文献证据**: Harnessing Explanations要求真正的rationale
- **启发方向**:
  - 短期: 接受基础设施级实现，诚实描述
  - 长期: 原型驱动的解释生成
  - 替代: 用解释作为router特征而非图输入

### Gap 5: 缺少"何时不用图"的显式机制
- **问题**: 当前方法编辑图，但不决定"是否用图"
- **文献证据**: Trustworthy AI文献强调selective usage
- **启发方向**:
  - Abstention mechanism (图专家主动弃权)
  - Risk-coverage tradeoff (显式建模可靠性-覆盖率)
  - Slice-specific policies (不同切片不同策略)

---

## Related Frontier Work (需Novelty Check)

### 不确定性引导的结构学习
- **UnGSL** (ICLR'25): Uncertainty-guided structure learning
- **G-DeltaUQ** (ICLR'24): Epistemic uncertainty for GNNs
- **差异点**: 我们的校准是后验的，而非结构学习的先验

### 自适应消息传递
- **ADMP-GNN** (ICLR'24): Adaptive message passing depth
- **差异点**: 我们的深度由校准分歧驱动，而非节点特征

### 原型引导的图学习
- **ProtoGNN** (ICLR'23): Prototype-guided GNN
- **BotHP** (arXiv 2506.00989): Prototype for bot detection
- **差异点**: 我们的原型由不确定性加权，而非纯聚类

### 一致性正则化
- **GRAND** (NeurIPS'20): Graph random neural networks
- **差异点**: 我们是选择性的反事实正则化，而非全局一致性

### 图不确定性量化
- **Conformal Prediction for GNNs** (ICLR'24)
- **CF-GNN** (ICLR'24): Conformal prediction on graphs
- **差异点**: 我们是训练时学习abstention，而非测试时coverage保证

### 选择性消息门控
- **TopGateGNN** (IJCNLP'25): Top-k gating for bot detection
- **差异点**: 我们的门控由校准驱动，而非学习的top-k

### 结构regime分类
- **Structural Disparity** (NeurIPS'23): Handling structural heterogeneity
- **AdaptiveMixGNN** (ICLR'24): Adaptive mixing for heterophily
- **差异点**: 我们混合校准+结构，而非纯结构regime

---

## Literature-Driven Design Principles

基于文献综述，以下设计原则应指导新方案：

### Principle 1: 语义信任作为监督，而非图oracle
- ✅ 当前已遵循: q_sem/u_sem作为信任信号
- 🔄 可改进: 更显式的语义监督机制（如distillation）

### Principle 2: 节点校准需显式转换为边/消息级决策
- ✅ 当前已遵循: pairwise features
- 🔄 可改进: 显式边可靠性建模

### Principle 3: 局部性和稀疏性是一等约束
- ✅ 当前已遵循: 局部重写+预算
- ❌ 缺失: 局部性保持的量化证据

### Principle 4: 效率优先，避免在线LLM调用
- ✅ 当前已遵循: 离线enrichment
- ✅ 保持: 所有新方案应保持此约束

### Principle 5: 异质性≠噪声，需区分处理
- ❌ 当前未充分遵循: 硬重写一刀切
- 🎯 新方案重点: regime routing, signed evidence, selective abstention

### Principle 6: 显式建模"何时不用图"
- ❌ 当前缺失: 无abstention机制
- 🎯 新方案重点: abstention, risk-coverage, slice-specific policies

---

## Recommended Reading List (for Novelty Check)

### Must-Read (Top 3 ideas相关)
1. **CF-GNN** (ICLR'24) - for Idea #9 (Abstention)
2. **GRAND** (NeurIPS'20) - for Idea #5 (Counterfactual)
3. **Graph Conformal Benchmarking** (ICLR'24) - for Idea #10 (Slice Framework)

### Should-Read (Backup ideas相关)
4. **UnGSL** (ICLR'25) - for Idea #2 (Edge Reliability)
5. **ADMP-GNN** (ICLR'24) - for Idea #3 (Adaptive Depth)
6. **Cold Brew** (ICLR'24) - for Idea #6 (Distillation)
7. **Structural Disparity** (NeurIPS'23) - for Idea #7 (Regime Routing)

### Nice-to-Read (背景补充)
8. **Selective Classification on Graphs** (ICML'23)
9. **TopGateGNN** (IJCNLP'25)
10. **ProtoGNN** (ICLR'23)

---

## Cross-Paper Synthesis

### 三个稳定结论

#### 1. 当前仓库在方法论上已coherent
- 不再声称相似度剪枝、节点置信度=边真值、无约束图编辑
- 现在声称: 语义信任监督 + 图校准警告 + 稀疏局部约束编辑
- **这是更强的文献对齐故事**

#### 2. 最大剩余弱点是语义增强质量
- 图重写逻辑比语义增强路径更文献对齐
- 当前explanation enrichment是基础设施级，非文献完备
- **应诚实描述为first implementation，而非完整模块**

#### 3. 主要剩余gap是经验性的，非概念性的
- 当前实现比旧的相似度剪枝计划更接近文献
- 主要未解决问题: 无真实rw*实验artifacts，无实现状态文档，无m5/m7/m9对比
- **下一步应focus on empirical validation**

---

## Document Metadata

- **Generated by**: Claude Opus 4.6
- **Sources**: 
  - `LLMbot/doc/literature_method_review_2026-04-01.md`
  - `LLMbot/doc/implementation_literature_alignment_2026-04-01.md`
  - Web search results (7 queries)
- **Related**: `IDEA_DISCOVERY_REPORT.md`
