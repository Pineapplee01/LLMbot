# Graph-Quality-Guided Local Ego Refinement for Social Bot Detection

Version: 2026-05-10

## 1. Research Goal and Claim Boundary

本文档只描述当前研究的 Stage 3：**Graph-Quality-Guided Local Ego Refinement**。LM + GNN 基础检测器、Stage 1 RoBERTa embedding 生成、基础 GNN 训练和已有 residual-risk selector 已经由现有代码线承担，不在本文档中重新展开。这里关注的是：如何对 hard-node candidates 构造可信任 refined ego-graph，并将其作为 GNN refiner 或 LLM enhancer 的输入。

本阶段目标不是再做一个全局图结构学习模型，也不是把 LLM 变成最终分类器，而是实现一个可复现、成本受控、可消融的局部 refinement 模块：

```text
Stage 1 RoBERTa node embeddings + original graph
  -> post-hoc conformal ego-graph quality estimator
  -> hard-node candidates
  -> SKETCH-style semantic/structural retrieval
  -> GAugLLM/CTGL-style coupled edge modifier
  -> trusted refined ego-graph artifact
  -> evidence ego prompt
  -> optional hard-node LLM embedding / refiner
```

核心研究主张应写成：

> 社交机器人检测中的残余错误集中在局部图质量差、文本-结构冲突或异配伪装节点上。我们用可校准的 post-hoc ego quality estimator 触发局部 refinement，再用文本-图联合的 edge modifier 对已有 ego 边做 soft reweight；低效用或可疑边是否表示为 binary reliability、continuous utility 或 evidence-aware role，需要通过消融确定，但它们不应被简单硬删除。

不要写成：

- "首次进行 social bot 图结构修改"。BotBR 和 BECE 已经覆盖了 edge reliability / graph reliability 空间。
- "首次将 LLM 用于图学习"。LOGIN、GLANCE、GraphText、GraphGPT 等已有大量工作；本研究只声称 selective evidence ego prompt / LLM-as-enhancer 的迁移与组合。
- "证明 LLM 理解了图结构"。TMLR 2024 的 graph prompt 分析反而提醒我们，LLM 往往把图提示当作上下文段落，而不是严格图结构推理。
- "完整因果图修复"。当前方法是 post-hoc、budgeted、local、evidence-oriented 的 soft refinement。

**文献依据与支持方式**

- [BotBR, SIGIR 2025](https://doi.org/10.1145/3726302.3729908) 已经在 social bot detection 中引入 edge detector、reliability-enhanced graph learning 和 consistency contrastive learning，说明本研究不能把"边可靠性"本身作为主要新颖性。
- [BECE, IEEE TNNLS 2025](https://ieeexplore.ieee.org/document/10530431/) 用 edge confidence evaluation 和高斯边表示处理 unreliable/camouflaged edges，说明 social bot 场景确实需要边级置信度，但也提示我们应避免重复做 binary reliable/unreliable edge scoring。
- [GLANCE, ICLR 2026](https://openreview.net/forum?id=oODFyykHF5) 支持只在 GNN 容易失败的节点上选择性调用 LLM/refiner，而不是全节点统一调用。它支持本文档把 LLM 调用放在 hard-node local refinement 之后，而不是基础检测器之前。
- [Graph Meets LLM Survey, IJCAI 2024](https://www.ijcai.org/proceedings/2024/898) 给出 LLM-as-enhancer / predictor / alignment component 的分类。本研究采用 enhancer/refiner 范式，而不是 LLM-as-final-predictor。

## 2. Why Conformal Ego-Graph Estimator Is the Main Post-Hoc Estimator

第一版 estimator 选择 graph conformal prediction set，而不是只做 GATS/CaGCN 式标量校准。原因是我们需要的不只是"置信度是否校准"，还需要一个可操作的 ego-graph 质量信号：

```text
Q_phi(Ego_v, X_R, A_v) -> {
  prediction_set(v),
  set_size(v),
  coverage_margin(v),
  abstain_risk(v),
  calibration_metadata
}
```

其中：

- `prediction_set` 表示 conformal set；
- `set_size` 越大，表示该节点在当前图上下文下越不确定；
- `coverage_margin` 衡量离 conformal threshold 的距离；
- `abstain_risk` 把 prediction-set uncertainty 映射成现有 hard-node router 可消费的标量；
- `calibration_metadata` 必须记录 calibration split，明确 test labels 不参与阈值拟合。

**文献依据与支持方式**

- [CF-GNN, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/54a1495b06c4ee2f07184afb9a37abda-Abstract-Conference.html) 将 conformal prediction 引入 GNN，并输出有覆盖率保证的 prediction set/interval。它直接支持把节点不确定性表示为预测集合，而不是单个置信分数。
- [DAPS/NAPS, ICML 2023](https://proceedings.mlr.press/v202/h-zargarbashi23a.html) 将 node-wise conformity scores 通过图扩散结合邻域信息，说明 post-hoc conformal estimator 可以利用 graph structure，而不是只看 logits。
- [SNAPS, NeurIPS 2024](https://openreview.net/forum?id=iBZSOh027z) 用 feature similarity 和 structural neighborhood 聚合 non-conformity scores，提高 prediction set efficiency。这正好支持后续 v2 把 RoBERTa kNN semantic similarity 接入 estimator。
- [GATS, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5975754c7650dfee0682e06e1fec0522-Abstract-Conference.html) 和 [CaGCN, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/c7a9f13a6c0940277d46706c7ca32601-Abstract.html) 说明 GNN calibration 需要考虑图结构、节点相关性和 topology-aware calibration。它们适合作为强 calibration baseline，但不直接提供 prediction-set-triggered refinement 接口。

**迁移到本研究**

第一版只 port 最小 conformal core：

1. 从基础 GNN logits/probabilities 计算 non-conformity score。
2. 只在 train/valid calibration split 上拟合 conformal threshold。
3. 对所有节点输出 prediction set、set size、coverage margin 和 abstain risk。
4. 用 `abstain_risk` 触发 hard-node local ego refinement。

不在 v1 复现完整 CF-GNN topology-aware correction，也不引入外部依赖。DAPS/NAPS/CF-GNN/SNAPS 是方法边界和复现实验簇，不是直接 vendor 的代码依赖。可参考开源仓库：[conformalized-gnn](https://github.com/snap-stanford/conformalized-gnn)、[DAPS](https://github.com/soroushzargar/DAPS)、[graph_cp](https://github.com/jase-clarkson/graph_cp)、[SNAPS](https://github.com/janqsong/SNAPS)。

## 3. Hard-Node Candidate Router

hard node 不应只等于低置信节点，也不应只等于低同质性节点。第一版 candidate router 使用 estimator 输出和现有 residual-risk selector：

```text
hard_v = Router(
  abstain_risk(v),
  set_size(v),
  coverage_margin(v),
  base_gnn_uncertainty(v),
  residual_risk_manifest(v)
)
```

router 的职责是决定哪些节点进入 Stage 3 refinement，而不是直接决定边怎么改。

**文献依据与支持方式**

- [GLANCE, ICLR 2026](https://openreview.net/forum?id=oODFyykHF5) 表明 GNN 和 LLM 在不同结构模式上有互补性，并用 lightweight router 决定是否查询 LLM。我们迁移的是 selective routing 思想，但触发信号从 local homophily / uncertainty 扩展为 conformal ego quality。
- [LOGIN, WSDM 2025](https://doi.org/10.1145/3701551.3703488) 将 LLM 用作 GNN training 中的 consultant，并对 spotted/uncertain nodes 构造包含语义和拓扑信息的 prompt。我们迁移的是"只处理 spotted hard nodes"和"语义+拓扑证据 prompt"思想，但不在 v1 进行交互式训练闭环。
- [GATS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5975754c7650dfee0682e06e1fec0522-Abstract-Conference.html) 指出 GNN 校准受距离训练节点、邻域相似性、预测分布多样性等因素影响，支持 router 不应只用 MSP/entropy。

**迁移到本研究**

第一版保持保守：

- 复用当前 residual-risk selector 作为候选池；
- 用 conformal `abstain_risk` 和 `set_size` 进行重排序或过滤；
- 记录 `router_provenance`，包括阈值、预算、calibration split、是否使用现有 residual-risk manifest；
- 不让 router 直接使用 test labels，不让 router 成为第二个 bot predictor。

## 4. Local Ego Candidate Retrieval: SKETCH + GAugLLM + CTGL

对 hard node `v`，构造两类候选：

```text
Semantic candidates:
  RoBERTa embedding kNN / semantically similar node texts
  -> virtual context candidates

Structural candidates:
  existing ego edges, k-hop neighbors, PPR/Jaccard/degree-aware local ranking
  -> propagation edge candidates if already in ego graph
  -> virtual context candidates if not already connected
```

候选边分成两种：

- `propagation_edges`: 已存在于 ego graph 的边，只允许 soft reweight，进入 GNN/message passing。
- `virtual_context_edges`: 语义检索或结构检索得到的潜在上下文，只进入 evidence artifact / prompt，不写回主图。

**文献依据与支持方式**

- [SKETCH/Taming, ACL 2025](https://aclanthology.org/2025.acl-long.173/) 提出将 node aggregation 与 graph convolution 解耦，并在文本表示学习中加入 semantic aggregation 和 structural aggregation。我们迁移为双通道 candidate retrieval：语义相关文本检索 + 结构上下文检索。
- [CTGL, COLING 2025](https://aclanthology.org/2025.coling-main.722/) 指出 text-learning model 和 graph-learning model 可能在不同节点上互补，并提出 coupled text-graph augmentation。我们可以将其迁移为 text-graph disagreement-aware candidate generation 和 coupled edge modifier scoring。
- [GAugLLM, KDD 2024](https://arxiv.org/abs/2406.11945) 明确指出文本属性与图结构不天然对齐，edge modification 应结合 structural candidates 和 textual commonality。我们迁移其 collaborative edge modifier 思想。
- [SNAPS, NeurIPS 2024](https://openreview.net/forum?id=iBZSOh027z) 用 feature similarity 和 structural neighborhood 改善 conformal set efficiency，支持语义相似和结构邻域可以共同作为可信候选来源。

**迁移到本研究**

不能使用手工堆叠大量特征，而是把 retrieval 写成清晰的可消融模块：

1. semantic-only retrieval；
2. structural-only retrieval；
3. uncoupled semantic + structural retrieval；
4. coupled retrieval + edge modifier scoring。

这四组直接对应后续消融，避免把所有特征混成一个难解释的 heuristic stack。

## 5. Edge Modifier Design: From Reliability Binary to Evidence-Aware Candidate Roles

social bot graph 中的 bot-human 异配边不一定是噪声。它可能是伪装、协同、关注诱导或攻击路径的证据。硬删除这类边会让 LLM prompt 和诊断分析失去关键上下文。

**文献依据与支持方式**

- [BotBR, SIGIR 2025](https://doi.org/10.1145/3726302.3729908) 用 edge detector 区分高低可靠边，并构造 homophily-based graph 做 contrastive learning。它证明 social bot detection 中 edge reliability 是有效方向，但也占据了 binary reliability + contrastive graph learning 的空间。
- [BECE, TNNLS 2025](https://ieeexplore.ieee.org/document/10530431/) 将边可靠性作为 proxy，并用高斯分布建模 edge representation 和噪声。它支持边级 confidence modeling，但本研究不重复其"不可靠边删除/置信度评估"主线。
- [GAugLLM](https://arxiv.org/abs/2406.11945) 支持结构候选和文本 commonality 联合决定 edge modification。我们迁移为 `z_text(e)` 与 `z_graph(e)` 的 coupled modifier scoring。
- [CTGL](https://aclanthology.org/2025.coling-main.722/) 支持文本学习和图学习并行互补，提示 edge modifier 应考虑 text-graph disagreement，而不只是 node embedding similarity。
- [GNNExplainer, NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html)、[PGExplainer, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/e37b08dd3015330dcbb5d6663667b8b8-Abstract.html) 和 [GraphMask, EMNLP Findings 2021 / arXiv](https://arxiv.org/abs/2010.00577) 都支持用 edge mask / edge importance 解释或压缩 GNN 决策。我们迁移其"边对预测贡献不同"的思想，但 modifier 输出不是 explanation report，而是 propagation weight + evidence role。

**迁移到本研究**

本节把 edge modifier 设计成一个可比较的候选空间。第一版先实现统一接口，再用消融决定最终采用 binary reliability、continuous edge utility，还是 evidence-aware role scheme。

推荐统一接口：

```text
z_text(e)  = TextPairEncoder(X_R[v], X_R[u])
z_graph(e) = LocalGraphPairEncoder(Ego_v, e, X_R, Q_phi)
z_e        = CoupledGate(z_text(e), z_graph(e), Q_phi(Ego_v))

candidate outputs:
  s_reliability(e)       # binary / scalar reliability candidate
  u_edge(e)              # continuous propagation utility candidate
  p_role(e)              # optional evidence-aware role distribution
```

候选 supervision 只在 train/valid 上通过 counterfactual edge intervention 生成，test 阶段不使用真标签。反事实操作不只包括 downweight，也可以包括 remove / upweight / candidate-add 的局部诊断；但 v1 写回主图时仍只允许 soft reweight existing ego edges。

```text
Delta_down(e) = Loss_v(A_v with e downweighted) - Loss_v(A_v)
Delta_drop(e) = Loss_v(A_v with e removed)      - Loss_v(A_v)
Delta_up(e)   = Loss_v(A_v with e upweighted)   - Loss_v(A_v)

candidate mappings:
  binary reliability:
    useful / harmful / ambiguous

  continuous utility:
    u_edge(e) = normalized counterfactual gain or loss

  optional evidence-aware roles:
    supportive / suspicious / neutral / uncertain
```

四类角色目前应写成候选 hypothesis，而不是最终定案：

- `supportive`: 增强传播，作为支持证据；
- `suspicious`: 降低传播权重，但保留为可疑证据；
- `neutral`: 基本保持；
- `uncertain`: 不强改，进入 evidence artifact 或 fallback。

注意：不能用 `same-label edge = reliable` 或 `different-label edge = unreliable` 作为主规则。这个规则在 bot-human camouflage 场景中会误删关键证据。最终采用哪种 edge output，需要由 9.4 的 edge-modifier ablation 决定。

## 6. Soft Graph Rewrite Instead of Hard Graph Rewrite

第一版只对已有 ego edges 做 soft reweight：

```text
w_e_next = w_e * clip(
  1 + lambda * budget_v
      * modifier_score(e)
      * (1 - uncertainty_e),
  1 - lambda,
  1 + lambda
)
```

其中 `modifier_score(e)` 可以来自连续 utility score，也可以来自 role distribution 的投影，例如 `p_supportive(e) - p_suspicious(e)`。该投影只是候选实现，不作为第五节的最终预设。

新增候选边不写回主图，只作为 virtual context edges 进入 evidence artifact / LLM prompt。

**文献依据与支持方式**

- [LDS, ICML 2019](https://proceedings.mlr.press/v97/franceschi19a.html) 学习 edge probability distribution，说明图结构可以被视为可学习概率对象，而不是固定真值。
- [Pro-GNN, KDD 2020](https://www.kdd.org/kdd2020/accepted-papers/view/graph-structure-learning-for-robust-graph-neural-networks.html) 支持从 noisy/corrupted graph 中联合学习鲁棒结构和 GNN，但依赖稀疏、低秩、特征平滑等假设。我们只迁移 budgeted graph repair 思想，不做全图结构学习。
- [GNNGuard, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/690d83983a63aa1818423fd6edd3bfdb-Abstract.html) 学习边权并抑制不相关边，直接支持 soft edge weighting。边界是它主要面向 adversarial defense，且常将低相似边视为坏边；本研究保留 suspicious edge 作为证据。
- [IDGL, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/e05c7ba4e087beea9410929698dc41a6-Abstract.html) 通过 node embeddings 和 graph structure 迭代优化，支持后续 iterative estimator loop，但 v1 不做全局迭代图学习。
- [GNNExplainer](https://papers.nips.cc/paper_files/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html) 和 [GraphMask](https://arxiv.org/abs/2010.00577) 支持 edge mask / differentiable edge dropping 的合理性。我们把 edge mask 从解释用途迁移为 soft modifier 的训练信号和诊断证据。

**迁移到本研究**

选择 soft modifier 的理由：

1. 社交机器人图存在有意义异配边，hard delete 容易误删伪装证据。
2. BotBR/BECE 已经占据较多 binary edge reliability 空间；continuous utility 或 evidence-aware role scheme 更适合作为差异化候选，需要通过消融确定。
3. conformal estimator 输出的是不确定性集合，不适合直接驱动不可逆 hard graph rewrite。
4. soft rewrite 可 rollback、可限制 budget、可解释 weight change，更适合第一版实验。

## 7. Evidence Ego-Graph Artifact and LLM Embedding Branch

研究包含 LLM 调用，但 LLM 不作为唯一最终分类器。v1 可以有两个实现分支：

- `Artifact-only branch`: 只保存 refined ego-graph artifact，并用现有 GNN/refiner 消费；
- `LLM-enhanced branch`: 将 trusted ego-graph artifact 序列化为 evidence ego prompt，只对 hard nodes 调用 LLM 生成 embedding，再交给 refiner。

Evidence ego prompt 不是原始 ego graph dump，而是压缩后的证据化结构：

```text
Target node:
  text summary, base posterior, conformal set, set size, abstain risk

High-utility / supportive evidence:
  top-k high-utility neighbors, edge type, short text summary, weight change

Low-utility / suspicious evidence:
  top-k low-utility neighbors, conflict reason, weight change

Uncertain evidence:
  high-uncertainty edges and why no strong rewrite was applied

Semantic context:
  SKETCH-style semantically retrieved nodes

Structural context:
  refined ego structural neighbors or paths

Graph quality summary:
  before/after estimator scores, budget used, stop/rollback decision
```

LLM 输出不直接作为最终标签，而是进入 refiner：

```text
z_llm(v) = LLM_embed(EvidencePrompt_v)
p_final(v) = Refiner(h_g(v), X_R[v], z_llm(v), Q_phi(Ego_v_final))
```

**文献依据与支持方式**

- [GraphText, 2023](https://arxiv.org/abs/2310.01089) 支持将图结构和节点属性转写为 text sequence 后交给 LLM 处理。它支持 graph-to-text 的可行性，但也提醒 serialization 会丢结构、受上下文长度限制。
- [Can LLMs Effectively Leverage Graph Structural Information through Prompts, TMLR 2024](https://openreview.net/forum?id=L2jRavXRxs) 指出 LLM 往往把结构 prompt 当作上下文段落处理，最有效的部分可能是与标签相关的邻域短语。这支持我们构造"证据化 ego prompt"，而不是原始全 ego dump。
- [LOGIN, WSDM 2025](https://doi.org/10.1145/3701551.3703488) 支持对 spotted nodes 构造包含语义和拓扑信息的 concise prompt，并把 LLM 作为 GNN enhancer/consultant。
- [GLANCE, ICLR 2026](https://openreview.net/forum?id=oODFyykHF5) 支持 selective LLM embedding + refiner，而不是所有节点直接调用 LLM 或 LLM direct prediction。
- [Graph Meets LLM Survey, IJCAI 2024](https://www.ijcai.org/proceedings/2024/898) 支持将本研究归入 LLM-as-enhancer，而非 predictor。

**迁移到本研究**

v1 允许实现 LLM-enhanced branch，但需要把它作为可选分支和消融项，而不是唯一主路径。推荐实验顺序是先验证 artifact-only refinement 是否改善 hard-node slice，再比较 LLM-enhanced branch 是否带来额外收益：

1. refined artifact only；
2. hard-node raw ego LLM embedding；
3. hard-node evidence ego LLM embedding；
4. LLM embedding + refiner；
5. LLM direct label as diagnostic baseline only；
6. LLM call ratio / cost-aware performance。

## 8. Iterative Estimator Loop

迭代只在廉价图侧进行，不每轮调用 LLM：

```text
Ego_v^0
  -> Q_phi(Ego_v^0, X_R, A)
  -> EdgeModifier
  -> Ego_v^1
  -> Q_phi(Ego_v^1, X_R, A)
  -> accept / rollback / stop
```

默认：

```text
max_iter = 1
```

扩展消融：

```text
max_iter = 2
```

超过两轮不作为主方法，因为容易产生 confirmation bias：estimator 自己认为自己修改后的图更可信，但不一定对真实标签更好。

**文献依据与支持方式**

- [IDGL, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/e05c7ba4e087beea9410929698dc41a6-Abstract.html) 支持 embedding 和 graph structure 可以迭代互相优化，并使用 stopping strategy。我们迁移为 ego-level estimator loop，而不是全图 end-to-end graph learning。
- [GLEM, ICLR 2023](https://openreview.net/forum?id=q0nmYciuuZN) 用 EM 思想让 LM 和 GNN 交替增强，支持"语义侧与图侧可以迭代互相校正"的高层思想。我们不做 GLEM 式全模型交替训练，只把它作为 iterative refinement 消融的概念依据。
- [CF-GNN](https://proceedings.neurips.cc/paper_files/paper/2023/hash/54a1495b06c4ee2f07184afb9a37abda-Abstract-Conference.html) 和 [DAPS/NAPS](https://proceedings.mlr.press/v202/h-zargarbashi23a.html) 支持每轮用 post-hoc conformal set 观察 uncertainty/efficiency 变化，作为 accept/rollback 的质量信号。

**迁移到本研究**

accept 条件应保守：

- `set_size` 不变小则拒绝；
- `coverage_margin` 不改善则拒绝；
- `abstain_risk` 不降低则拒绝；
- 改动 budget 超限则 rollback；
- calibration risk 或 validation proxy 变差则 rollback。

## 9. Experimental Plan

### 9.1 Main Baselines

主比较不应只拿弱 baseline：

- current RoBERTa + BotRGCN/RGCN/GNN backbone；
- residual-risk selector + no-op；
- GATS/CaGCN-style graph calibration proxy；
- BotBR-style edge reliability graph learning；
- BECE-style edge confidence evaluation if reproduction surface is available；
- raw ego prompt / SKETCH ego prompt / refined ego artifact variants。

**文献依据与支持方式**

- [BotBR](https://doi.org/10.1145/3726302.3729908) 和 [BECE](https://ieeexplore.ieee.org/document/10530431/) 是 social bot edge reliability 强相关 baseline。
- [GATS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5975754c7650dfee0682e06e1fec0522-Abstract-Conference.html) 和 [CaGCN](https://proceedings.neurips.cc/paper/2021/hash/c7a9f13a6c0940277d46706c7ca32601-Abstract.html) 是 graph calibration baseline。
- [DAPS/NAPS](https://proceedings.mlr.press/v202/h-zargarbashi23a.html)、[CF-GNN](https://proceedings.neurips.cc/paper_files/paper/2023/hash/54a1495b06c4ee2f07184afb9a37abda-Abstract-Conference.html) 和 [SNAPS](https://openreview.net/forum?id=iBZSOh027z) 是 conformal estimator baseline family。

### 9.2 Estimator Ablation

目标：检验 conformal set estimator 是否比单一置信分数更适合作为 hard-node refinement trigger。

```text
E0: MSP / entropy / margin
E1: temperature scaling
E2: GATS/CaGCN-style graph calibration proxy
E3: conformal set estimator without graph diffusion
E4: conformal set estimator with DAPS/NAPS-style graph diffusion
E5: SNAPS-style semantic similarity + structural neighborhood extension
```

指标：

- Accuracy / Macro-F1；
- ECE / Brier；
- AURC；
- coverage；
- average set size；
- singleton hit ratio；
- hard-node slice recall；
- low-quality ego slice performance。

### 9.3 Local Ego Refinement Ablation

目标：检验 SKETCH-style 解耦检索和 GAugLLM/CTGL-style coupled modifier 是否都对 hard-node local refinement 有贡献。

```text
R0: no refinement
R1: raw ego artifact only
R2: semantic-only retrieval
R3: structural-only retrieval
R4: uncoupled semantic + structural retrieval
R5: coupled local ego refinement
R6: coupled refinement + conformal accept/rollback
```

文献支持：

- [SKETCH/Taming](https://aclanthology.org/2025.acl-long.173/) 支持 semantic/structural decoupled aggregation；
- [CTGL](https://aclanthology.org/2025.coling-main.722/) 支持 text and graph models have complementary correct/incorrect regions；
- [GAugLLM](https://arxiv.org/abs/2406.11945) 支持 structural candidates + textual commonality 的 collaborative edge modifier。

### 9.4 Edge Modifier Output Ablation

目标：比较不同 edge modifier 输出形式，而不是预设四角色一定优于 binary reliability。

```text
B0: binary reliable/unreliable edge scoring
B1: continuous edge utility scoring
B2: optional supportive/suspicious/neutral/uncertain role scoring
B3: counterfactual downweight supervision only
B4: counterfactual drop/upweight/downweight supervision
B5: hard delete harmful edges
B6: soft reweight harmful/low-utility edges
B7: soft reweight + retain low-utility/suspicious evidence
```

文献支持：

- [BotBR](https://doi.org/10.1145/3726302.3729908) 和 [BECE](https://ieeexplore.ieee.org/document/10530431/) 提供 binary reliability 的强参照；
- [GNNGuard](https://proceedings.neurips.cc/paper/2020/hash/690d83983a63aa1818423fd6edd3bfdb-Abstract.html) 支持 edge weighting/pruning；
- [GNNExplainer](https://papers.nips.cc/paper_files/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html)、[PGExplainer](https://proceedings.neurips.cc/paper/2020/hash/e37b08dd3015330dcbb5d6663667b8b8-Abstract.html)、[GraphMask](https://arxiv.org/abs/2010.00577) 支持 edge importance/mask 的思想。

### 9.5 Soft vs Hard Rewrite

目标：检验局部 soft modifier 是否比 hard add/delete 更稳健。

```text
S0: no graph rewrite
S1: hard delete low-role edges
S2: hard add semantic/structural candidate edges
S3: existing-edge soft reweight
S4: existing-edge soft reweight + virtual context edges
S5: full graph-level rewrite, diagnostic only
```

文献支持：

- [LDS](https://proceedings.mlr.press/v97/franceschi19a.html) 支持 edge probability；
- [Pro-GNN](https://www.kdd.org/kdd2020/accepted-papers/view/graph-structure-learning-for-robust-graph-neural-networks.html) 支持 graph cleaning / robust structure learning；
- [GNNGuard](https://proceedings.neurips.cc/paper/2020/hash/690d83983a63aa1818423fd6edd3bfdb-Abstract.html) 支持 learned edge weighting；
- [BotBR](https://doi.org/10.1145/3726302.3729908) 提醒 hard reliability graph + contrastive learning 已是强竞争方向。

### 9.6 LLM Usage Ablation

研究包含 LLM 调用；v1 可以先实现 artifact-only branch，也可以实现 optional LLM-enhanced branch。无论是否在 v1 实现 LLM，实验表述都必须把 LLM 使用方式作为消融，而不是把 LLM direct prediction 作为默认终点：

```text
L0: no LLM, refined ego artifact only
L1: all-node raw ego LLM embedding
L2: hard-node raw ego LLM embedding
L3: hard-node SKETCH ego LLM embedding
L4: hard-node refined evidence ego LLM embedding
L5: hard-node refined evidence ego LLM direct prediction
```

推荐主线是 `L4`，并通过 refiner 融合 `h_g(v)`、`X_R[v]`、`z_llm(v)` 与 `Q_phi`；不推荐 `L5` 作为最终判别，只把它作为诊断 baseline。

文献支持：

- [GLANCE](https://openreview.net/forum?id=oODFyykHF5) 支持 hard-node selective LLM embedding + refiner；
- [LOGIN](https://doi.org/10.1145/3701551.3703488) 支持 LLM as consultant/enhancer；
- [GraphText](https://arxiv.org/abs/2310.01089) 支持 graph-to-text；
- [TMLR 2024 graph prompt analysis](https://openreview.net/forum?id=L2jRavXRxs) 支持不要把原始 ego graph 直接 dump 给 LLM，而要做证据化 prompt。

### 9.7 Iteration Ablation

```text
I0: no estimator loop
I1: one-shot estimator-guided refinement
I2: two-step estimator-guided refinement
I3: two-step without rollback constraint
I4: iterative loop + LLM every round, diagnostic only
```

文献支持：

- [IDGL](https://proceedings.neurips.cc/paper_files/paper/2020/hash/e05c7ba4e087beea9410929698dc41a6-Abstract.html) 支持 iterative graph/embedding learning；
- [GLEM](https://openreview.net/forum?id=q0nmYciuuZN) 支持语义模型和图模型交替增强；
- [CF-GNN](https://proceedings.neurips.cc/paper_files/paper/2023/hash/54a1495b06c4ee2f07184afb9a37abda-Abstract-Conference.html) 支持每轮使用 prediction set quality 作为不确定性证据。

## 10. Metrics and Diagnostics

主指标：

- Accuracy；
- Macro-F1。

校准与风险指标：

- ECE；
- Brier；
- AURC；
- conformal coverage；
- average set size；
- singleton hit ratio；
- coverage violation by subgroup。

效率指标：

- hard-node ratio；
- LLM call ratio；
- average ego size；
- average virtual context edge count；
- refinement latency。

切片指标：

- low-quality ego slice；
- high `abstain_risk` slice；
- text-graph disagreement slice；
- low-degree / sparse neighborhood slice；
- heterophily slice；
- suspicious-edge-rich slice；
- bot-human mixed-neighborhood slice。

**文献依据与支持方式**

- [CF-GNN](https://proceedings.neurips.cc/paper_files/paper/2023/hash/54a1495b06c4ee2f07184afb9a37abda-Abstract-Conference.html)、[DAPS/NAPS](https://proceedings.mlr.press/v202/h-zargarbashi23a.html) 和 [SNAPS](https://openreview.net/forum?id=iBZSOh027z) 支持 coverage、set size、singleton hit ratio。
- [GATS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5975754c7650dfee0682e06e1fec0522-Abstract-Conference.html) 和 [CaGCN](https://proceedings.neurips.cc/paper/2021/hash/c7a9f13a6c0940277d46706c7ca32601-Abstract.html) 支持 ECE/Brier 等校准诊断。
- [GLANCE](https://openreview.net/forum?id=oODFyykHF5) 支持 subgroup / hard-node / LLM call ratio 分析，而不是只看 aggregate accuracy。
- [BotBR](https://doi.org/10.1145/3726302.3729908) 和 [BECE](https://ieeexplore.ieee.org/document/10530431/) 支持 social bot graph reliability slice 和 heterophily/camouflage edge slice。

## 11. Implementation Order

### Step 1: Estimator-first

实现 `graph_conformal_set_estimator`：

```text
inputs:
  base logits/probabilities
  calibration labels from train/valid only
  optional graph adjacency for score diffusion

outputs:
  prediction_sets
  set_size
  coverage_margin
  abstain_risk
  calibration_metadata
```

文献支持：[CF-GNN](https://proceedings.neurips.cc/paper_files/paper/2023/hash/54a1495b06c4ee2f07184afb9a37abda-Abstract-Conference.html)、[DAPS/NAPS](https://proceedings.mlr.press/v202/h-zargarbashi23a.html)、[SNAPS](https://openreview.net/forum?id=iBZSOh027z)。

### Step 2: Hard-node artifact routing

用 `abstain_risk` 与现有 residual-risk manifest 选 hard nodes，保存 provenance。

文献支持：[GLANCE](https://openreview.net/forum?id=oODFyykHF5)、[LOGIN](https://doi.org/10.1145/3701551.3703488)。

### Step 3: Local ego retrieval

对 hard nodes 构造 semantic candidates 与 structural candidates，并区分 propagation edges 和 virtual context edges。

文献支持：[SKETCH/Taming](https://aclanthology.org/2025.acl-long.173/)、[CTGL](https://aclanthology.org/2025.coling-main.722/)、[GAugLLM](https://arxiv.org/abs/2406.11945)。

### Step 4: Edge modifier

先实现 deterministic/scaffold edge modifier，再逐步替换为可学习 head。监督信号由 train/valid counterfactual edge intervention 生成；可比较 binary reliability、continuous utility 和 optional evidence-aware role scheme。

文献支持：[GAugLLM](https://arxiv.org/abs/2406.11945)、[GNNExplainer](https://papers.nips.cc/paper_files/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html)、[PGExplainer](https://proceedings.neurips.cc/paper/2020/hash/e37b08dd3015330dcbb5d6663667b8b8-Abstract.html)、[GraphMask](https://arxiv.org/abs/2010.00577)。

### Step 5: Soft rewrite and evidence artifact

只更新已有 ego edges 的 artifact 权重；virtual context edges 不写回主图。保存 before/after estimator outputs、edge utility/role evidence 和 counterfactual provenance。

文献支持：[LDS](https://proceedings.mlr.press/v97/franceschi19a.html)、[Pro-GNN](https://www.kdd.org/kdd2020/accepted-papers/view/graph-structure-learning-for-robust-graph-neural-networks.html)、[GNNGuard](https://proceedings.neurips.cc/paper/2020/hash/690d83983a63aa1818423fd6edd3bfdb-Abstract.html)。

### Step 6: LLM embedding branch

构造 evidence ego prompt，只对 hard nodes 调用 LLM embedding，并用 refiner 融合。该步骤可以作为 v1 optional branch 实现，但必须保留 artifact-only branch 作为成本和贡献归因基线。

文献支持：[GraphText](https://arxiv.org/abs/2310.01089)、[LOGIN](https://doi.org/10.1145/3701551.3703488)、[GLANCE](https://openreview.net/forum?id=oODFyykHF5)、[TMLR 2024 graph prompt analysis](https://openreview.net/forum?id=L2jRavXRxs)。

## 12. Expected Contributions

### Contribution 1: Conformal ego quality as a local refinement trigger

把 prediction set quality 从 uncertainty reporting 迁移到 social bot hard-node local refinement trigger。支持文献是 [CF-GNN](https://proceedings.neurips.cc/paper_files/paper/2023/hash/54a1495b06c4ee2f07184afb9a37abda-Abstract-Conference.html)、[DAPS/NAPS](https://proceedings.mlr.press/v202/h-zargarbashi23a.html)、[SNAPS](https://openreview.net/forum?id=iBZSOh027z)。边界是它不是新的 conformal theory，而是应用到 ego quality guided refinement。

### Contribution 2: Text-graph coupled local ego refinement

把 [SKETCH/Taming](https://aclanthology.org/2025.acl-long.173/) 的 semantic/structural decoupled context、[CTGL](https://aclanthology.org/2025.coling-main.722/) 的 text-graph coupled augmentation 和 [GAugLLM](https://arxiv.org/abs/2406.11945) 的 collaborative edge modifier 迁移到 social bot detection 的 hard ego graph。边界是 v1 只 soft-reweight existing edges，不做全图 add/delete。

### Contribution 3: Evidence-preserving edge modifier

把 BotBR/BECE 的 edge reliability 从单一 binary 删除逻辑扩展为可消融的 edge modifier：binary reliability、continuous utility、optional evidence-aware roles 都作为候选输出；核心约束是低效用或 suspicious bot-human edges 可以被降权，但应保留为证据。支持文献是 [BotBR](https://doi.org/10.1145/3726302.3729908)、[BECE](https://ieeexplore.ieee.org/document/10530431/)、[GNNGuard](https://proceedings.neurips.cc/paper/2020/hash/690d83983a63aa1818423fd6edd3bfdb-Abstract.html)、[GNNExplainer](https://papers.nips.cc/paper_files/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html)。边界是 modifier output 不等价于忠实因果解释；四类角色只是候选实现之一。

### Contribution 4: Selective evidence ego prompt for LLM-as-enhancer

把 refined ego artifact 转为证据化 prompt，并只对 hard nodes 生成 LLM embedding。支持文献是 [GLANCE](https://openreview.net/forum?id=oODFyykHF5)、[LOGIN](https://doi.org/10.1145/3701551.3703488)、[GraphText](https://arxiv.org/abs/2310.01089)、[Graph Meets LLM Survey](https://www.ijcai.org/proceedings/2024/898)。边界是 LLM 不作为唯一主分类器；v1 可以选择实现 LLM branch，但必须以 hard-node selective embedding + refiner 形式出现，并保留 no-LLM artifact-only baseline。

## 13. Risks and Guardrails

### Risk 1: 重复 BotBR/BECE 的 claim space

Guardrail: 不把贡献写成 edge reliability graph learning。写成 conformal-quality-guided, evidence-preserving local ego refinement；role-aware 只作为候选实现，不作为未经验证的最终 claim。

支持文献：[BotBR](https://doi.org/10.1145/3726302.3729908)、[BECE](https://ieeexplore.ieee.org/document/10530431/)。

### Risk 2: counterfactual pseudo supervision 泄漏 test labels

Guardrail: edge modifier 的 counterfactual pseudo supervision 只在 train/valid 上生成；test 阶段只用 learned/deterministic scorer 和 estimator outputs。不要把四类角色当作人工真值标签，也不要用 test loss 选择 modifier scheme。

支持文献：[CF-GNN](https://proceedings.neurips.cc/paper_files/paper/2023/hash/54a1495b06c4ee2f07184afb9a37abda-Abstract-Conference.html)、[DAPS/NAPS](https://proceedings.mlr.press/v202/h-zargarbashi23a.html) 对 calibration validity 的要求。

### Risk 3: hard rewrite 破坏 bot-human suspicious evidence

Guardrail: v1 只 soft reweight existing ego edges；potential new edges 只作为 virtual context，不写回主图。

支持文献：[GNNGuard](https://proceedings.neurips.cc/paper/2020/hash/690d83983a63aa1818423fd6edd3bfdb-Abstract.html)、[LDS](https://proceedings.mlr.press/v97/franceschi19a.html)、[Pro-GNN](https://www.kdd.org/kdd2020/accepted-papers/view/graph-structure-learning-for-robust-graph-neural-networks.html)。

### Risk 4: LLM prompt 变成原始邻居堆叠

Guardrail: prompt 必须 evidence-oriented，至少区分 high-utility、low-utility、uncertain、semantic 和 structural context；如果采用 evidence-aware roles，再映射为 supportive/suspicious/neutral/uncertain，并控制 top-k。

支持文献：[GraphText](https://arxiv.org/abs/2310.01089)、[TMLR 2024 graph prompt analysis](https://openreview.net/forum?id=L2jRavXRxs)、[GLANCE](https://openreview.net/forum?id=oODFyykHF5)。

### Risk 5: 迭代 loop 自我确认

Guardrail: v1 `max_iter=1`；v2 `max_iter=2` 只作为消融；必须有 rollback constraint。

支持文献：[IDGL](https://proceedings.neurips.cc/paper_files/paper/2020/hash/e05c7ba4e087beea9410929698dc41a6-Abstract.html)、[GLEM](https://openreview.net/forum?id=q0nmYciuuZN)。

## 14. Open Design Questions and Systematic Challenges

这些问题不是阻塞项，而是后续实验和写作中必须正面回答的研究边界。

### Challenge 1: Estimator quality 是否真的等价于 ego-graph quality?

Conformal `set_size`、`coverage_margin` 和 `abstain_risk` 衡量的是 prediction uncertainty，不必然等价于 ego-graph 结构质量。需要验证低质量 signal 是否真的来自局部结构污染，而不是语义 embedding 本身、类别边界模糊或基础 GNN 欠拟合。

实验要求：

- 比较 estimator-triggered refinement 与 random hard-node budget；
- 比较 high-uncertainty but structure-clean 节点和 high-uncertainty structure-conflict 节点；
- 报告 low-quality ego slice，而不是只报告全局 Macro-F1。

### Challenge 2: Edge modifier 学到的是结构效用，还是标签泄漏 proxy?

Counterfactual pseudo supervision 使用 train/valid loss 生成，容易学到与标签分布或基础模型错误模式绑定的 proxy。必须严格禁止 test-label fitting，并验证 learned modifier 是否能跨 split、跨 seed、跨数据集泛化。

实验要求：

- train/valid-only counterfactual pseudo supervision；
- test 阶段只用 learned/deterministic scorer；
- seed-level variance；
- no-counterfactual heuristic baseline；
- label-shuffled or split-swapped diagnostic, if feasible。

### Challenge 3: 三种 edge output 哪个才是主方法?

当前保留三种候选：binary reliability、continuous utility、optional evidence-aware roles。它们的研究身份不同：

- binary reliability：最容易实现，但最接近 BotBR/BECE；
- continuous utility：更适合作为主线，因为它自然对应 soft reweight；
- evidence-aware roles：最适合 LLM prompt 和诊断，但不能未经验证就当作最终标签空间。

推荐默认：以 `continuous utility` 作为主实现；把 binary reliability 和 evidence-aware roles 放入 ablation。

### Challenge 4: LLM 到底提供了新信息，还是只是重包装 ego evidence?

如果 LLM branch 只把已有 ego artifact 编码成 embedding，收益可能来自 prompt summarization，而不是 LLM 的图推理能力。必须用 raw ego prompt、evidence prompt、no-LLM artifact-only 和 LLM direct label 做对照。

实验要求：

- artifact-only；
- raw ego prompt；
- evidence ego prompt；
- hard-node selective LLM embedding；
- all-node LLM embedding；
- LLM direct prediction as diagnostic only；
- LLM call ratio 与 latency。

### Challenge 5: soft rewrite 是否真的优于 hard delete/add?

本文档倾向 soft rewrite，但这仍然需要实验支持。尤其在 bot graph 中，一些边可能确实是 noise；soft 保留是否会保留过多污染，需要通过 hard delete/add 对照验证。

实验要求：

- hard delete low-utility existing edges；
- hard add high-similarity virtual context edges；
- soft reweight existing edges；
- soft reweight + virtual context prompt；
- no rewrite baseline。

### Challenge 6: refined ego artifact 是否会膨胀成手工特征拼接?

研究贡献应是结构化数据流和可学习/可消融 modifier，而不是人工堆字段。artifact 可以丰富，但最终进入模型的接口必须简洁，且每个字段都要有 ablation。

实验要求：

- artifact field ablation；
- top-k sensitivity；
- prompt length sensitivity；
- semantic-only / structural-only / coupled retrieval。

### Challenge 7: 和 BotBR/BECE 的差异是否足够清楚?

BotBR/BECE 已经覆盖 social bot edge reliability 和 graph modification。本文档必须坚持差异化：

- 不是全图 reliability graph learning；
- 不是 binary unreliable edge deletion；
- 不是 graph contrastive learning 主线；
- 是 conformal-quality-guided hard-node local ego refinement；
- 是 evidence-preserving soft rewrite + optional LLM enhancer。

## 15. Final Recommended v1 Configuration

```text
Estimator:
  graph_conformal_set_estimator
  outputs: prediction_sets, set_size, coverage_margin, abstain_risk

Router:
  residual-risk candidates + conformal abstain_risk filtering

Retrieval:
  semantic candidates from RoBERTa kNN
  structural candidates from ego/k-hop/PPR-style local ranking

Modifier:
  compare binary reliability, continuous utility, and optional evidence-aware roles
  pseudo supervision from train/valid counterfactual edge interventions

Rewrite:
  existing ego edges only
  soft reweight with budget and uncertainty control
  virtual context edges not written back

Artifact:
  trusted refined ego-graph evidence artifact
  before/after quality scores and provenance

LLM:
  artifact-only branch is required as a no-LLM baseline
  hard-node evidence ego embedding is optional in v1
  LLM direct prediction is diagnostic only

Iteration:
  max_iter = 1 for main method
  max_iter = 2 only as ablation
```

This configuration is the most defensible first version because it:

- uses high-level conformal graph literature for post-hoc uncertainty and coverage;
- uses SKETCH/CTGL/GAugLLM only for text-graph coupled local context and edge modifier transfer;
- avoids duplicating BotBR/BECE's binary reliability claim;
- avoids unsafe full graph rewriting;
- includes LLM calls as a controlled hard-node enhancer branch rather than replacing the GNN/refiner.
