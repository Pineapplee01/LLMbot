# 社交机器人检测：长期探索复盘与 NLPCC 后续重点（2026-09-30）

> 只读复盘：依据仓库文档、实验产物、rebuttal 证据库与 git 历史整理，未重跑实验，未改动已有文件。评估框架为 idea-evaluator（致命缺陷 → 生命周期 → 五维 → 范式 → 可行性 → 结论）。
> 路径均相对仓库根 `G:/Research/BotDetection`。缩写：`L/` = `LLMbot/`，`RB/` = `NLPCC/rebuttal/`，`E-xxx` = `RB/evidence_registry.md` 中的证据编号，hss = `hyperscan_sampled_subgraph` 训练契约（§3.G）。

## 0. 结论先行

一句话："冻结基模型 → 定位 → 纠错" 这条主线在本项目的数据上没有找到稳定的净增益。下一步先回答残差错误还能不能修（P1），主张转向决策有效性（P2），P0 的证据卫生是两者的前提。

- 8 个方案与结论（§3）。A–G 共享一个框架：冻结基检测器，先定位可能错的节点，再对它们做纠正。
  - A（LLM–GNN 深度融合原型）：没有结论文档，不评分；它的内核（微调 LM 表示）留在了 BotRHG 的 iter_2 embedding 里。
  - B（测试期改图与 LLM 改边）、C（失败机制分型与最小干预）、E（专用路由器族）、F（LLM 作为纠错器）：Reject，核心的定位或纠错机制已被本项目自己的对照驳倒。
  - D（保形 ego 质量七段 pipeline）：Reject，只有规格，关键组件被 E、F 的数据驳倒。
  - G（BotRHG，NLPCC 2026 录用）：Reject and Pivot。核心机制被自家 rebuttal 证据驳倒，训练契约还让被采到的 valid/test 真标签进入训练 loss。会议报告照常，扩展版不再以 "路由 + 超图纠错" 为主张；可以带走的是基线资产和 "定位≈MSP、W→R≈R→W" 这条负面结论。
  - H（基线复现与评测协议）：保留并修补，是下一阶段最可复用的资产。
- 跨方案规律（§4）：
  - 各种路由器与触发器的错误定位能力都与 MSP/entropy 同档（TW20 error screening 中 MSP 的 error-AUROC 为 0.8576，§3.E）。
  - 纠错器改对与改错几乎一样多（W→R≈R→W）；同预算的随机或 shuffled 对照经常持平甚至更好。
  - 领先幅度主要来自微调 embedding 和训练契约，不是路由或超图纠错；契约中标签暴露占多少还没有量化。
  - TW22 train 与 test 的 bot 比例是 7.80% 对 29.44%，多个模型因此塌缩到全判 human，但一直没有被当作主问题。
  - 统一假设 H*（未验证）：强基模型在 TW20 上剩下的错误集中在特征不可分或标签有歧义的账号上，事后纠错的上限由数据决定。
- NLPCC 后续重点（§5 评估，§6 计划）：
  - P0 证据卫生（会议 11-03 前）：冻结统一 harness，量化标签暴露，给复现基线补存逐节点概率；SSH 恢复后先同步服务器产物。
  - P1 可纠错性诊断（门控，约 2 周，CPU）：共识错误、kNN 纯度、confident learning 辅助名单、双人盲标注，按预注册阈值决定关闭纠错路线、让 P3 具备启动条件，还是记为未决。评估：Accept with Revisions（§5.2）。
  - P2 决策有效性（主线候选）：TW22 先验偏移审计、bot 流行率的 PPI 区间、FDR 受控的 conformal 标记。评估：Accept with Revisions，依赖 P0（§5.1）。
  - P3 定向纠错器：只在 P1 的第二条判定成立时才具备启动条件，即可纠错子集在诊断上占残差错误的足够比例，且不用 test 标签的对应信号挑选效果在 5 seed 上显著优于 MSP 与 ensemble entropy；是否启动由用户决定。在此之前，作为路由信号的邻域纯度维持 Reject and Pivot（§5.3）。
  - 停止清单：新路由器或风险分数不显著优于 MSP 与 ensemble entropy 就不做；本地没有可复查算力时不做 LLM 改边或 LLM 纠错。每个新方法都要过自动 kill gate。

## 1. 证据口径

| 标记 | 含义 |
|---|---|
| 本地 5 seed | 仓库内有逐 seed 产物，可复算 |
| 本地单 seed | 只有 1 个 seed 的产物，不能写成均值±std |
| 仅文档 | 文档写了数值，本地找不到对应产物 |
| 仅服务器 | 结果只在远程 GPU 服务器；SSH 自 2026-06-18 起不可用，无法核实 |
| 预期值 | 实验前写下的估计，从未运行 |

本文的 "未核实" 都指 "本地找不到产物"，不等于数据有误。

## 2. 时间线（2026）

| 阶段 | 起止 | 核心假设 | 结局 | 方案 |
|---|---|---|---|---|
| LLMbot 原型（cross-attn、CL adapter、uncertainty head） | 01-20~03-25 | LLM 与 GNN 深度融合可纠错 | 转向，无结论文档 | A |
| rw0–rw4 可靠性引导的测试期改图 | 04-01~04-10 | 路由加改图可纠错 | 放弃 | B |
| rw5_repair / UGBP v1 | 04-10~04-15 | 双路由修复 / 不确定性引导传播 | rw5 未运行即删除；UGBP 淘汰 | B |
| FRMI v1（LM/GNN 分歧触发） | 04-15 | 分歧可定位错误 | C1 判死 | C |
| LMBot 复现 + Phase0–5 regime probe | 04-15~04-17 | 错误集中于特定 regime | 基线保留；算子弱 | C/H |
| PROTOCOL_FREEZE（TW22 + GATv2） | 04-20 | 换数据集与骨干 | 被推翻，回到 TW20/RGCN | H |
| FRMI v2（B1/A2/A4，S0 tribunal） | 04-21~04-22 | 按 regime 分型做最小干预 | 停滞，门控未关闭 | C |
| dual-lane harness | 04-23 | 统一 StageRunner | 部分实现 | H |
| Conformal ego-quality v1–v8（七段 pipeline） | 05-09~05-11 | 保形 ego 质量触发局部精修 | 仅规格 | D |
| Router 族（RxD-MCR、GLANCE proxy、graph conformal、MSP-TS、ranker、LOGIN） | 05-14~05-18 | 专用路由器能定位残差错误 | 降级为后验校准加排序 | E |
| LLM 证据图 / Route-B / 改边 | 05-16~05-28 | 经剪枝的 LLM 局部证据图可纠错 | 转向 | B |
| LLM 微调与提示（MPE、BotSay ICL/PEFT、DGP-v2、selector、MoE） | 05-23~06-14 | LLM 证据可纠正路由节点 | 06-18 弃用 | F |
| local_conflict、KNN→Qwen 改边、Conformal-KNN、MH-LGC/GCL | 06-06~06-15 | 局部冲突或对比学习可修复高阶结构 | 中性或负面 | B |
| BotRHG（路由 + 选择性超图残差） | 06-15~06-20 | 只对高风险节点注入超图残差 | 录用 NLPCC 2026 | G |
| 治理、rebuttal、camera-ready | 06-20~09-17 | 无 | 已录用；会议 11-03~05（澳门） | G |

## 3. 方案归纳与局限

九个月的探索可以归成 8 个方案。A–G 都是同一个问题框架：冻结一个基检测器，先定位它可能错的节点，再对这些节点做纠正。H 是支撑它们的基线与评测协议。已被数据驳倒的方案按 idea-evaluator 的短路格式只写第一印象、致命缺陷与结论，行动项统一放到 §6。

### 3.A LLM–GNN 深度融合原型（01-20~03-25）

- 第一印象：想用 cross-attention、对比学习 adapter 与不确定性头把 LLM 文本表示和 GNN 结构表示深度融合，一步到位地提升检测并纠错。
- 证据：只有代码（git `f485e34`→`c733dce`），没有结论文档，也没有可复算的结果。
- 局限：无法评估。这段经验没有沉淀下来，是后面 "记录缺口" 问题的起点。
- 结论：不评分。它的内核（微调 LM 表示）后来以 iter_2 embedding 的形式留在了 BotRHG 里，而且可能正是 BotRHG 增益的主要来源（见 §4.4）。

### 3.B 测试期改图与 LLM 改边（rw0–rw5、UGBP、证据图、local_conflict；04-01~06-15）

- 第一印象：先用可靠性路由找出高风险节点，再改它们的局部图（删边、加边、LLM 重写证据），想把错判改对。
- 关键证据：
  - rw0–rw4：confidence_fallback 0.798 高于 router 0.788（`docs/wiki/log.md` L82-112），Codex 评为 "NOT READY"（L206-245）。
  - rw5：只有标为 "预期性能" 的 0.88–0.93（log.md L47-50），没有运行结果；04-15 清理时被标为 rejected/dead end，同批共删除约 30 个文件（L287-309）。
  - 证据图：用 routed_nodes 训练时 router precision 0.25，refiner 倾向于维持原预测；只用 wrong_nodes 训练时 oracle 修正率 0.78，但出现大量 right→wrong；改边效果 "近似随机扰动"（`docs/Discussion/518.md` L149-171）。
  - GraphEdit 式 Qwen3.5-9B 改边：路由切片从 0.4421 升到 0.4838，全测试集从 0.8600 变为 0.8597（`docs/wiki/experiments/lmbot_gnn_lm_baselines.md` L69-110；`docs/experiment.md` L136-148 称 "essentially neutral"）。
  - local_conflict 为 −0.00169，同预算的 local_rand 为 +0.00169（`docs/research/local_conflict_diagnostic_record_2026-06-06.md` L187-198）。
- 致命缺陷：

| # | 缺陷 | 严重度 | 依据 |
|---|---|---|---|
| F1 | 定位信号不优于 fallback 或随机。precision 0.25 意味着被路由的节点大多本来就是对的，refiner 多数维持原判，改边效果近似随机扰动 | CRITICAL（数据已驳倒） | log.md L82-112；518.md L151-171；local_conflict L187-198 |

- 结论：Reject。"路由 + 改图能净纠错" 这个核心机制已被 fallback 与随机对照驳倒。能迁移的只有一条：改图类算子的收益上限由定位精度决定，定位不过关就不值得做算子。

### 3.C 失败机制分型与最小干预（FRMI v1/v2、Phase0–5；04-15~04-22）

- 第一印象：先判断一个错误属于哪种失败机制（regime），再针对这种机制做最小干预。v1 用 LM 与 GNN 的预测分歧作触发器。
- 关键证据（`refine-logs/` 下的文件在 git 中已暂存删除，但工作区仍保留原文件，也可用 `git show HEAD:<path>` 查看）：
  - FRMI v1：分歧触发 d_i 为 0.6894/0.6921，低于 gnn_entropy 的 0.7188（二分类下与 1−max_prob 相同），C1 判死；entropy+disagree 组合为 0.7327，略高于 entropy，说明分歧只能作补充信号，而且只有 seed42（`refine-logs/frmi_results/e1_trigger_seed42.json`；`docs/reviews/REVIEW_SUMMARY.md` L25-33）。
  - FRMI-3A 为 0.7472，低于基线 S 的 0.7477（REVIEW_SUMMARY L37-43）。
  - Phase0–5：
    - PHASE1 只有 prop_corruption 这一类 regime 的错误富集（1.84x）；entropy+disagree 为 0.820，gnn_entropy 为 0.819（`refine-logs/PHASE1_REGIME_SUMMARY.md` L35, L40-41）。
    - PHASE3 的改图/重训（aux_50）只增 +0.003，落在 ±0.011 的波动内（`PHASE3_REWRITE_RETRAIN_BASELINES.md` L19, L26）。
    - PHASE4 [3s]：repair 为 0.8473±0.013，noop 为 0.8453±0.018；semantic override 只有 α=0.3 为正（0.8681，Acc +2.3 pp），α=0.05 降到 0.8276（`PHASE4_ACTION_OUTCOMES.md` L13-17）。0.3 是事后挑出的最优值，按 val 选出的 α 在各 seed 间为 0.05/0.2/0.3（L36），并不稳定。
    - E4 只是单 seed pilot（seed 42，proxy repair；REVIEW_SUMMARY L35-43）。R012 seed1 中 D3 最好（+0.0018），D7 反而更差，方向与假设相反（`git show 628f054:refine-logs/EXPERIMENT_TRACKER.md` L23）。
  - FRMI v2：stability 4/10，S0 tribunal 没有裁决（`idea-stage/IDEA_REPORT.md` L13-15, L159-171）。PAPER_CHARTER L46 规定 "Any gate fails → immediate claim downgrade"，但门控始终没有关闭。
- 致命缺陷：

| # | 缺陷 | 严重度 | 依据 |
|---|---|---|---|
| F1 | 触发信号不优于 entropy（0.689 对 0.719） | CRITICAL（数据已驳倒） | e1_trigger_seed42.json；REVIEW_SUMMARY L25-33 |
| F2 | 干预算子的增益不稳定：改图/重训 +0.003 在噪声内；只有事后最优的 α=0.3 为正，val 选出的 α 随 seed 变化；3A 低于 S（单 seed） | MAJOR | PHASE3 L19, L26；PHASE4 L13-17, L36；REVIEW_SUMMARY L37-43 |

- 结论：Reject。可保留的有两点：probe 能找出错误富集的 regime（prop_corruption），但算子改不动，这和 §4.1 的规律一致；LMBot 复现（RGCN 0.8723）进了基线库。

### 3.D 保形 ego 质量触发的局部精修（v1–v8 七段 pipeline；05-09~05-11）

- 第一印象：用保形预测估计每个节点 ego 子图的质量，质量低就触发局部检索和 LLM 精修。v8 把它锁定为七段流水线：LM+GNN → 估计器 → router → 检索 → rewriter → 证据图 → LLM。
- 证据：全部停留在规格层面，没有跑过实验。
  - R1 评审提示的是风险：二分类下保形集合只有 {bot}、{human}、{bot,human} 三种，集合几何可能塌缩成近似标量阈值信号（"near-scalar threshold"，`refine-logs/round-1-refinement.md` L44）。
  - R6 接受了三条 P0 批评：没有覆盖定理；存在自证循环；s_het 符号错误，异配边不等于噪声（`refine-logs/round-6-critique-response.md`）。
  - v8-1 评审为 RETHINK（6.675），称其为 "multi-heuristic pipeline tuning"（`refine-logs/round-v8-1-review.md` L22-24）；v8-5 升到 REVISE 8.63、drift NONE，评审仍写 "Not READY. Remaining gap is primarily empirical, not structural"（`round-v8-5-review.md` L17-24）。
  - R4/R6/R8 的规格评分都是 9.2，但标注为 "READY (spec-level; conditional on Pilot Gates A/B)"（`round-4-review.md` L64；`round-8-review.md` L20），Pilot Gates 从未运行。
- 致命缺陷：

| # | 缺陷 | 严重度 | 依据 |
|---|---|---|---|
| F1 | "保形" 只是包装：没有覆盖定理（R6 已接受），二分类下还有塌缩成近似标量阈值的风险（R1） | MAJOR | R1、R6 |
| F2 | 七段中的关键环节后来被方案 E/F 的数据驳倒（router≈MSP，LLM 纠错器净增益为负） | CRITICAL（组件已被驳倒） | §3.E、§3.F |

- 结论：Reject。教训：规格评分不能代替 pilot；pipeline 的段数越多，越要先逐段验证。

### 3.E 专用路由器族（RxD-MCR、GLANCE proxy、graph conformal、MSP-TS、ranker、LOGIN；05-14~05-18）

- 第一印象：专门设计或训练一个路由器来定位基检测器的残差错误，把 "找错" 这一步做强。
- 关键证据：
  - 当时的评审：6.5/10，bot 检测 track 为 borderline accept，conformal track 为 weak reject；"Single-dataset results are a kill" 是 W7，严重度 LOW（`idea-stage/router/ROUTER_REVIEW_FEEDBACK.md` L12, L34-35）。05-15 的 `ROUTER_IDEA_REPORT.md` 分数轨迹为 6.5→5.5→7.0，同时写明 "Pilots remain deferred"（L4, L17-21）；`refine-logs/router/PIPELINE_SUMMARY.md` L5 记为 READY 7.0。两者不矛盾，问题在于 READY 是在 pilot 全部推迟时给出的。之后的实际定位是 "后验校准加排序"（`docs/candidates.md` L9）。
  - 迁移来的方法大多真的跑过，只是没有一个超过 MSP。下面是 TW20 test 上的 error screening（只排序、不改预测），MSP 的 error-AUROC 为 0.8576 [3s]，random 约 0.51–0.54（`results/twibot20_router_*`、`results/login_uncertainty_router_*`、`L/experiments/knn_router_*`、`L/experiments/selective_routeronly_*`）：
    - rxd Mondrian conformal：rel_dir 0.8173，比 MSP 差，还输给 shuffle 对照的 0.8518；class 分箱 0.8583，与 MSP 持平。`rxd_rel_dir_dominant` 与 `rxd_rel_dir_shuffle_matched` 的 budget_curve.csv md5 相同，这组 shuffle 对照实际没有生效。
    - graph conformal 0.8465、posthoc ranker 0.8559、Phase-A 2-hop 0.8400–0.8480，都 ≤MSP。`glance_proxy` 与 `msp_ts` 的 budget_curve.csv md5 相同；二分类下全局温度缩放不改变 MSP 的排序，所以两者都等价于 MSP。
    - LOGIN 方差 0.1821/0.3387，排序方向反了。
    - kNN 超图 router 最好为 0.8597，相对 MSP 的 AURC 降幅 +0.0077，CI 含 0，seed3 为负；NCP 为 0.8590。
    - 学习式 router：reliability_mlp 0.8452、SelectiveNet 0.8153，每个 seed 都低于同一 g0 的 MSP。
  - 接上纠错器之后 [3s]：GLANCE 反事实（本地改写，未用 LLM）ΔmF1 为 −0.0031/−0.0041/+0.0040；strict_glance 五族都没超过 base 0.8694；joint router+refiner 单 seed 为 +0.0062，3 seed 为 −0.0013/−0.0040/−0.0014。
  - CF-GNN/DAPS/SNAPS/RR-GNN 本地没有按方法名区分的产物，是否收在 graph conformal 目录里未核实。`router_knn_evidence_reconfiguration.py` 等 3 个脚本本地没有任何输出。
  - rebuttal 阶段在 TW20 上做了系统对照：
    - 论文用的 WRT 为 0.8525±0.0056，MSP 为 0.8517，两者同档（E-ES1-007）。
    - 五模型 ensemble 的 predictive entropy 为 0.8751/0.8775，高于 WRT；registry 注明它是另一族基检测器，不能算作同一 base 上优于 WRT（E-ES1-005/006）。
    - MC-dropout 0.8577、Laplace 0.8526，也在同档（E-ES1-009/010）。
    - 只算打分这一步，WRT 比 confidence deficit（1−MSP）慢约 460–1581 倍（0.152 s 对 0.00033 s；0.585 s 对 0.00037 s），GPU 峰值 105.7–181.5 MB 对 8.5 MB（E-ES1-008/012）。算上 checkpoint 恢复、建图和前向的端到端耗时，WRT 为 5.74 s，split conformal 为 5.50 s，GPU 同为 1.23 GiB（E-ES3-007）。
- 致命缺陷：

| # | 缺陷 | 严重度 | 依据 |
|---|---|---|---|
| F1 | 同一 base 上，专用或迁移来的风险分数都与 MSP/entropy 同档或更差；打分一步贵两到三个数量级，端到端差距很小 | CRITICAL（数据已驳倒） | E-ES1-007~010、E-ES1-012、E-ES3-007；上列 screening |

- 结论：Reject。在这套数据和基模型上，错误定位信号的天花板就在 MSP/entropy 附近（§4.1）。继续设计新路由器的边际价值很低。

### 3.F LLM 作为纠错器（MPE、BotSay ICL/PEFT、DGP-v2、selector、MoE；05-23~06-14）

- 第一印象：让 Qwen2.5/3.5 读取被路由节点的 profile、推文和邻居证据，通过 ICL、PEFT 或学习型 selector/MoE 给出更正。
- 关键证据（只写行号的指 `L/experiments.md`）：
  - 零样本 Qwen3.5：在 296 个被路由节点上 Macro-F1 为 0.4856，FN=112；同一批节点上 base 的 Macro-F1 为 0.6292、Acc 为 0.6385（L185-202, L593, L639-641）。
  - PEFT：fix/break 为 57/79，全测试集从 0.8624 降到 0.8441（L590-604）。
  - DGP-v2：PEFT 为 47/69/−22；embedding MLP 塌缩成单类，62/118/−56（L3748-3755）。
  - learned selector 为 32/40/−8，能找到可修的节点，但保护不住原本正确的节点（L1920-1927）；utility-gate 净增益为负（L1441-1445）。
  - 高 base 上也有小幅正值，都只有单 seed：固定候选 follower_triplet 为 14/8/+6，但它是事后挑出的最优；按 val 选阈值的置信度 selector 为 14/11/+3（L1905-1931）。oracle 上界为 26/0/+26（单 expert）和 32/0/+32（全部候选；L1913-1919），说明可修的节点存在，只是选不准。
  - MoE selector 实际上是 "globally biased soft fusion head"（L2188-2210）：balance 变体净 +4/+4、temp_t02 净 +1，但 237 个路由节点全部或几乎全部被分给同一个 expert（L2180-2186）。
  - 06-18 的大纲被标为 "LEGACY / DO NOT USE"（`paper/NLPCC/paper_outline_nlpcc2026_legacy_llm_mainline.md`）。
  - BotSay 复现（ICL/PEFT）记录的 "Macro-F1" 0.6913/0.7991 分别等于同行 Precision 与 Recall 的调和平均，实际是 bot 类 F1（`L/experiments.md` L741、L783）。KNN router 接 Qwen3.5-9B 的 GraphEdit 式边裁决 [5s]：routed 切片上 keepdrop_add 比 base_only 高 +0.0417，但全 test 相对 frozen SimTeG 为 −0.0028~−0.0004（`docs/wiki/experiments/lmbot_gnn_lm_baselines.md` L69-110，只有文档记录）。
  - Qwen3 表征当特征或第二意见用 [1s]（以下数字由本地 `outputs.pt` 与 `analysis_summary.json` 重算）：直接作为 G0 特征时 test Macro-F1 为 ego prompt 0.7828、hop1 prompt 0.7580、`qwen3_emb_last` 0.7967，都低于 RoBERTa-iter2 的 0.8693（`L/experiments/direct_gnn_*_seed1_rerun20260526`）。在弱 G0（准确率 0.781，test 判错 259 个）上，通用 prompt expert 净 −7 个节点，按 conflict/follower/following/tweet 切分的 expert 净 +36~+41，纠正后准确率约 0.816，仍低于高 base 的 0.866；接到高 base（判错 158 个）上做 refiner，净值为 −2~0（`L/experiments/server_prompt_expert_*_qwen_20260526`、`L/experiments/prompt_refiner_on_highbase_*_seed1`）。
  - 这批 expert run 的切片审计（`L/server_prompt_expert_slice_audit_seed1.json`）的 follower/following 计数维度把 1183 个 test 节点全部归入 "0" 桶，而 follow_ratio 维度分桶正常，说明这一维统计本身失效，不能用来判断邻居上下文是否起作用。
- 致命缺陷：

| # | 缺陷 | 严重度 | 依据 |
|---|---|---|---|
| F1 | 没有稳健的净纠错：生成式变体（零样本、PEFT、DGP-v2）净值全为负；高 base 上 selector/MoE 净值在 −8~+6 之间，正值都是单 seed 或事后挑选；编码器 expert 只在弱基模型上明显为正 | CRITICAL（数据已驳倒） | 上列行号与路径 |
| F2 | 本地 8 GB 显存且没有 Qwen 权重，服务器不可用，无法复查 | MAJOR | §4.7 |

- 结论：Reject。LLM 在这里是一个比基检测器弱的第二意见：同一批 296 个被路由节点上，零样本 Macro-F1 为 0.49，基检测器为 0.63。把更弱的意见直接用在最难的节点上会净破坏；selector/MoE 最多多修 3–6 个节点，而且只有单 seed。编码器路线的正收益只出现在弱基模型上，换到高 base 就消失。

### 3.G BotRHG：可靠性路由 + 选择性超图残差（06-15~06-20，NLPCC 2026 录用）

- 第一印象：基检测器是 RoBERTa 微调表示加 RGCN。用 weighted reference-tail（WRT）分数选出 β=10% 的高风险节点，只对这些节点注入 HyperScan 式 DHG 超图残差。
- 实现现状：
  - 纠错分支在 `L/code/GNNs.py` L633-715，其中有一个论文没有描述的 2 参数 risk gate，但它只在 consumer_scope 为 `risk_gated_all_nodes` 时生效，其余情况返回全 1（L770-776）。WRT 在 `RB/code/rebuttal_pipeline/routing_scores.py` L217。
  - 归档 run 实际用的是 ncp_* 族（`L/code/estimators.py` L2129），与论文公式的节点重合度 Jaccard 只有 0.3882（E-AUDIT-001）。
  - `NLPCC/code/` 只是 stub，`main.py` 只打印 config。
- 关键证据（TW20，Acc/P/F1）：
  - 主表 88.93/89.18/88.78 对应 seed1 的 NCP run（E-AUDIT-002）。本地 `L/experiments/twibot20_full_selective_residual_5seed_20260620/_reports/full_model_5seed_summary.json` 记为 seed_count=1。本地两份备份不一致：`NLPCC/figures/data/main_results.csv.bak_20260620_113116` 中该行为 seeds=1、没有 std；`…_130744` 改为 seeds=5，acc_std 0.0053、macro_f1_std 0.0048。其中 0.0048 与 3 seed frozen A3（routed_only@10%）的 Full Macro-F1 std 相同（`L/experiments.md` L6443），0.0053 在本地找不到对应产物。用户此前说明主表来自服务器运行，这里只记为本地无法核实。
  - 本地可复算的 5 seed：WRT 为 88.06/88.26/87.91，low-only 已经有 87.90/88.36/87.70，NCP 为 87.25/87.59/87.06（`RB/evidence/twibot20_paper_formula_5seed_20260712`、`twibot20_archived_ncp_5seed_20260712`）。
  - 配对检验：
    - A3 对 A0 为 88.29 对 88.16，p=0.61；对 shuffled 为 88.28，p=0.99。以上是 Acc 的 p 值，Macro-F1 对应为 0.58 和 0.96（E-ES2-003）。
    - WRT 的增益为 +0.17/−0.10/+0.22 pp，未校正的配对 p=0.72/0.86/0.64，不做多重比较校正也不显著，逐 seed Δ 在 −1.10~+1.61 之间（E-ES2-004）。
    - NCP 为 −0.08/+0.11/−0.14（E-ES2-005）。
  - 翻转数：W→R 27、R→W 25（E-ES5-002）；fresh final 为 88.67，低于 88.93，W→R 44、R→W 41（E-ES1-004）。
  - 消融（`L/experiments/twibot20_seed1_component_ablation_20260619_reports/component_ablation_seed1.csv`）：
    - 论文消融表的 "w/o Hypergraph Correction" 行（0.8715/0.8684，`NLPCC/figures/data/twibot20_component_ablation_table.tex` L14）在 CSV 中对应 "Frozen SimTeG -> RGCN/graph detector"，即只把微调 embedding 换成 iter_-1，实际是 A3 routed-only 加 iter_-1 embedding，仍然含超图（`L/experiments.md` L6336, L6543）。
    - 06-18 的 seed1 核心消融（iter_-1 embedding，`L/experiments.md` L6331-6339，Full Acc/Macro-F1）中，A3 routed_only@10% 为 0.8715/0.8684，低于 A0 low_only 的 0.8766/0.8741、A1 all_nodes 的 0.8757/0.8732 和 A6 shuffled 的 0.8757/0.8745，排序与论文叙述相反。
  - 3 seed 的 Full Macro-F1：routed_only@10% 为 0.8678，低于 low_only 的 0.8746，也低于同预算 shuffled 的 0.8722（`L/experiments.md` L6440-6447）。路由切片上的数值（如 A3 为 0.8449）按各自设置的节点集计算，不能跨设置比较。
  - 预算（seed1 单次）：5%/10%/15%/20%/30% 分别为 +1.18/+0.59/+0.68/−0.68/−0.51（E-ES3-005），5% 和 15% 都高于 10%。rebuttal 回应却写 "The β grid peaks at 10%, stays stable from 5%-15%, and degrades beyond 20%"（`RB/response/k9d8_paste_ready.md` L15）。
  - 唯一稳健的正面性质：错误捕获率高于随机，lift 从 5% 预算的 3.72x 降到 30% 的 2.62x，10% 预算捕获 36.02% 的错误（E-ES3-004）。但 MSP 与 WRT 同档（error-AUROC 0.8517 对 0.8525，E-ES1-007），这不是 WRT 独有的性质。
  - 弱基模型上确有增益：68.28 → 70.48，+2.20±0.52（E-ES4-005），但没有 shuffled 和 all-nodes 对照，可能只是一般性的集成效应。
  - TW22：82.41±1.40/73.14/71.31 在本地找不到产物；F1 71.31 低于 BIC 的 73.42（E-PAPER-003）。
  - 训练契约暴露评测标签（代码已确认，规模未量化）：
    - C0–C5 消融都跑在 `hyperscan_sampled_subgraph` 契约下（`L/experiments.md` L6579-6581）。该契约对 NeighborLoader 采到的全部行计算 loss（`L/code/trainer_preparation.py` L1898-1901），`batch.y` 只把 support 行填成 −100（L1264-1275），train/valid/test 行都带真标签（L2819-2840），代码里没有其他 valid/test 掩码。
    - 因此只要 valid/test 节点被采进训练邻域，它们的真标签就进入训练 loss。该契约还按全部采样行做验证选模，允许重复计数。`LLMbot/code` 不在 git 跟踪中，服务器跑的是否同一版本无法确认，但 manifest 的 supervision_scope 字段与此一致。
    - 同一 RGCN/control、seed1 下只切换契约，canonical 去重 test 的 Acc/Macro-F1 从 0.8648/0.8628 升到 0.8774/0.8755（Macro-F1 +1.27 pp，multiattn 为 +0.56 pp；`L/experiments.md` L5879-5893，只有文档记录）。这个差值同时混有监督量和选模口径的变化，泄漏占多少没有拆开。作为量级参照，主表中 BotRHG 领先最强复现基线 BotBR（0.8672）2.06 pp。
  - 其余可核实的问题：
    - 按 E-ES5-002 的翻转数，纠错在同一套权重内只净增 2 个节点（约 0.17 pp Acc）。C0 相对 w/o residual 的 1.23 pp Macro-F1 差距（0.8878 对 0.8755）因此主要来自两次训练的轨迹不同，不是纠错本身。
    - 同一图契约下只换 router（E-ES1-004，单 seed）：论文公式在 10% 预算内捕获 C2 的 59/145 个错误，same-hyperedge 只捕获 20/145；重训后 F1 为 88.58 和 87.71，都低于主表 88.78。捕获的错误多了近 3 倍，纠错净值仍只有 +3（W→R 44、R→W 41）。
    - 单 seed 超参扫描中 K、fanout、budget 的默认值（8、64、10%）恰好都是峰值（`L/experiments.md` L6563-6573），没有多 seed 复核。
    - C2（w/o residual）的 0.8774/0.8755 与契约对照表中 hss 下 RGCN 那一行四位小数完全相同，但两边 manifest 记录的 backbone 与输入张量不同，是否同一次 run 需要到服务器产物上确认。
- 引用缺口：当前 `NLPCC/sections/*.tex` 没有引用 NCP（Ghosh et al., 2023）。06-14 的备份稿（`paper/NLPCC/backup_20260614_162257/`）曾在 introduction.tex L9、method.tex L21（"neighborhood-conformal intuition"）、related_work.tex L13 与 references.bib L29-31 中引用，后来被删掉。WRT 与 NCP 的邻域加权思路非常接近，扩展版必须引用并做对照。
- 致命缺陷：

| # | 缺陷 | 严重度 | 依据 |
|---|---|---|---|
| F1 | 核心机制被自家 rebuttal 证据驳倒：对 shuffled p=0.99（Macro-F1 为 0.96），对 low-only 未校正 p=0.64–0.86，W→R≈R→W | CRITICAL（数据已驳倒） | E-ES2-003/004/005、E-ES5-002 |
| F2 | 训练契约让被采到的 valid/test 真标签进入训练 loss，主表数字不是干净的 test 结果；可核实的证据链也与论文表述不一致（主表为单 seed，消融行标签错误，TW22 行本地无产物） | CRITICAL（泄漏路径代码已确认，规模未量化） | `L/code/trainer_preparation.py` L1898-1901；E-AUDIT-002；上列路径 |

- 结论：Reject and Pivot（与 2026-09-29 的评估一致）。论文已录用，会议报告照常进行，但不宜强调相对基线的领先幅度（F2）；扩展版不应再以 "路由 + 超图纠错" 为主张。可以带走两样东西：基线与协议资产（§3.H），以及 "定位≈MSP、W→R≈R→W" 这条负面结论本身（§4）。

### 3.H 基线复现与评测协议（贯穿全程）

- 目标：把所有方法放在同一套 TW20/TW22 协议下公平比较。
- 已有资产（`repro_baselines_20260614/unified_metrics.json`，Acc / Macro-F1）：

| TW20 | Acc | Macro-F1 | 备注 |
|---|---|---|---|
| BotRGCN | .8568 | .8549 | 唯一干净的官方复现 |
| RGT / BIC / BotMoE | .8639 / .8555 / .8612 | .8620 / .8534 / .8592 | RGT 做了适配；BIC 没有推文序列 |
| SEBot | .8629 | .8610 | 适配版，作为诊断，不进主比较 |
| BotBR | .8691 | .8672 | binary-F1 .8831，原论文为 .8891（`docs/experiment.md` L168-181） |
| LMBot LM / GNN | .8541 / .8511 | .8487 / .8465 | |
| frozen SimTeG | .8631 | .8613 | 3 seed；另一组 5 seed 为 Acc .8617、Macro-F1 .8600 |
| BotHunter / FriendBot | .7528 / .7719 | .7439 / .7663 | Macro-F1 取自 `NLPCC/figures/data/main_results.csv`；FriendBot 比 README 高 1.3 |

- TW22 有两个样本版本，不能混用：
  - T22-A：11826 人，test 为 348 bot / 835 human，多数类基线 .7058。
  - T22-B：11542 人，去掉了 284 个无推文用户，test 为 313/786，多数类基线 .7152（`datasets/TwiBot-22-official-prior-sampled-v1/prepare_manifest.json`）。
  - 最强基线是 BIC，在 T22-A 上 Acc .7856、Macro-F1 .7342。
- 发现的问题：
  - 主表混用口径：
    - SEBot 的 TW22 行 78.43/74.54/70.19 恰好是 seed2 的单次 run。5 seed 均值（Acc/macro-P/Macro-F1）是 .7332/.6367/.4891，其中 s1/s5 全判 human（`repro_baselines_20260614/results/sebot_twibot22_official_prior_sampled_same_protocol_20260716_5seed_combined_summary.json`）。
    - BotBR 的 TW20 行 86.72 与其他行一样是 Macro-F1，但它是由取整后的日志反算的（`docs/research/baselines/social_bot_baseline_reproduction_20260614.md` L40）。
    - `NLPCC/sections/experiments.tex` L8 称所有方法用同一 T22 集，与 A/B 混用的实际情况不符。
    - std 口径（总体与样本）混用；没有 macro-precision 字段；部分 macro 值是从取整后的日志反算的。
    - BotRGCN、RGT、BIC、BotMoE、BotHunter、FriendBot 的 TW20 行，以及 LMBot 的 TW22 行，本地都找不到来源。
  - HyperScan 异常：在 LMBot split 上，TW20 Acc 为 91.53±2.96、F1 为 92.74±2.48，其中 seed2 为离群点 85.87（`docs/experiment.md` L186-209，仅有服务器 csv）。HyperScan 论文自报的 TW20 F1 为 87.2（`log.md` L125；`docs/RELATED_WORK.md` L130），本地值比它和其他所有 TW20 基线（85–87）都高出一截，两边 F1 口径是否一致未核实；这个结果也没有进主表。在 "去泄漏审计" 与 "改写 SOTA 表述" 之间必须二选一。另一条线索：`L/experiments.md` 把 `hyperscan_sampled_subgraph` 记作 HyperScan 的 "faithful" 契约（L5816-5845），而这个契约会让被采到的 valid/test 真标签进入训练 loss（§3.G）。如果 91.53 出自同类训练方式，它可能带有同样的暴露；但仓库里 HyperScan 只剩 submodule 指针、本地没有对象，91.53 由哪条代码路径产生也无法确定，本地无法核实。
  - 完整 TW22 无法运行：本地缺 `user.json`、`node.json` 与推文。`baseline_comparability.md` 只定义了 TW20 协议。
  - 在 `repro_baselines_20260614/` 下没有找到逐节点预测文件，跨方法的错误分析需要先确认或补存。
  - MGTAB：BotBR 5 seed 为 Acc 92.51±0.55、F1 86.29±0.73；HyperScan 的 MGTAB 还没跑完（experiment.md L224-239, L334）。
- 结论：保留并修补。这是下一阶段最可复用的资产，修补成本也清楚（§6 P0）。

## 4. 跨方案规律（局限的共同根源）

### 4.1 定位信号有天花板：各种路由器和触发器都与 MSP/entropy 同档

- PHASE1：entropy+disagree 0.820，gnn_entropy 0.819。
- FRMI 的 d_i 0.689，低于 entropy 0.719；entropy+disagree 为 0.733（单 seed），只比 entropy 高一点。
- 同一 base 上，WRT、MC-dropout、Laplace 都在 0.852–0.858，与 MSP 同档（E-ES1-007~010）。五模型 ensemble entropy 为 0.875，但属于另一族基检测器，不能直接比较（E-ES1-005/006）。
- 错误捕获率是随机的 2.6–3.7 倍，预算越大倍数越低（E-ES3-004），MSP 同档。

### 4.2 纠错器改对与改错几乎一样多

| 来源 | W→R | R→W |
|---|---|---|
| BotRHG（E-ES5-002） | 27 | 25 |
| BotRHG fresh final（E-ES1-004） | 44 | 41 |
| PEFT | 57 | 79 |
| DGP-v2 | 47 | 69 |
| selector | 32 | 40 |

518.md 中只用 wrong_nodes 训练时，oracle 修正率为 0.78，但同样出现大量 right→wrong。

### 4.3 同预算的随机或 shuffled 对照经常持平甚至更好

- fallback 0.798 > router 0.788。
- local_rand +0.00169 > local_conflict −0.00169。
- 3 seed Full Macro-F1：同预算 shuffled 0.8722 > routed_only@10% 0.8678（`L/experiments.md` L6440-6447）。
- E-ES2-003 中对 shuffled 的配对 p 值：Acc 0.9858，Macro-F1 0.9627。
- 即使用 test 标签挑出判错节点再改图（oracle，单 seed），上限也不到 1 pp：RGCN 加 oracle 边只修对 157 个错误中的 4 个、改错 3 个，Macro-F1 +0.0008，随机选节点为 +0.0009（`results/oracle_retrieval_upper_bound_rgcn_gaugllm_proxy_seed1/metrics.csv`）；较弱的 RGT（Macro-F1 0.7668）上 oracle 剪边修对 274 个中的 10 个，+0.0084，随机剪边为 −0.0017（`results/oracle_retrieval_upper_bound_rgt_graph_expert_seed42_synced/metrics.csv`）。

### 4.4 增益主要来自基模型

5 seed 的 low-only（不做纠错）已经达到 87.90，full 为 88.06，Acc 差 +0.17 pp（p=0.72，E-ES2-004），不显著。所以 BotRHG 相对基线的差距大部分在 low-only 阶段就已经存在。现有产物可以粗略拆开：

- 只换 embedding（`L/experiments/embedding_compare_20260527/preiter_vs_iter2_seed1_compat_summary.csv`，5 seed）：iter_-1 → iter_2 让 detector test Macro-F1 提升 +0.0153（RGCN，0.8548 → 0.8701）和 +0.0050（RGT，0.8623 → 0.8673）。同一批 run 里 router 纠错阶段的平均增量为 RGCN −0.0036（5 个 seed 中 4 个为负）、RGT −0.0012。这 5 个 seed 共用同一份 seed1 微调 embedding，embedding 本身的 seed 方差没有计入。
- frozen embedding 下，3 seed 的 routed_only@10% 比 low_only 低 0.0068（`L/experiments.md` L6436-6445）。
- 只切换训练契约就有 +1.27 pp Macro-F1（§3.G），与 BotRHG 领先 BotBR 的 2.06 pp 同一量级。
- 合起来看，领先幅度主要来自微调 embedding 和训练契约，不是路由或超图纠错。契约部分里泄漏占多少，要等 P0 在 seed_only 契约下重跑后才能定量。

### 4.5 TW22 的先验偏移没有被当作主问题

- 官方 split 的 bot 比例：train 7.80%，val 27.96%，test 29.44%。样本版与全量一致（`docs/research/baselines/social_bot_baseline_reproduction_20260614.md` L320-331）。
- 由此出现坍缩（T22-B，`L/experiments/sampled_twibot22_official_prior_base5_20260620/seed_*/preparation/`）：LM 阶段（semantic encoder）在 5 个 seed 中有 4 个 test bot-F1 为 0（s1/s2/s4/s5），test Macro-F1 0.4170 恰好等于全判 human 的值（test human 占比 0.7152），s3 的 bot-F1 也只有 0.0249；graph detector 的 valid Macro-F1 为 0.4175–0.4907，其中 s1/s5 同样等于全判 human。SEBot 的 s1/s5 也全判 human，BotMoE 则反向过度预测 bot。
- 这是数据层面的问题，任何纠错模块都修不了，只能在决策层（阈值或先验）处理。现有方案都没有把它当作主问题。

### 4.6 证据卫生

- 预期值没有被检验：rw5 的 0.88–0.93 在 log.md 中标为 "预期性能"，没有运行就被删除；根目录 `IDEA_REPORT.md` L109 的 0.765→0.775 也只是预期增益，本地没有找到对应的检验结果。
- 单 seed 结论先于多 seed 验证：`docs/proposals/FINAL_PROPOSAL.md` L157 只在 seed 42 上验证，`docs/narratives/SUMMARY.md` L4 已写 "PROCEED TO PUBLICATION"，而 B6 多 seed 还排在后面（`docs/reviews/KILL_TEST_RESULTS.md` L16）。主表 88.93 本地只能核实 seed1。
- 指标混用：PHASE5 L55-56 把 Phase 4 的 Overall Acc 放进 F1 列；BotSay 复现记录的 "Macro-F1" 实际是 bot 类 F1（§3.F）。
- split 混用：T22-A 与 T22-B；`docs/wiki/experiments/configuration_verification_2026-04-09.md` L85-89 记录的 11,899（8,303/2,390/1,206）与 11,826 相差 73 个节点，文档判为 "consistent"，但没有解释差值。

### 4.7 流程失效

- kill gate 写了但不执行：根目录 `IDEA_REPORT.md` L321 的 kill 条件是全局 F1<0.87，L109 预期 0.765→0.775，两处没有说明口径是否一致，也都没有运行核对；PAPER_CHARTER L46 的 "Any gate fails → immediate claim downgrade" 没有执行；router tracker L168-178 全是 pending；CKR-001~015 全是 TODO（`refine-logs/conformal_knn_risk_experiment_tracker_2026_06_09.md` L5-19，未纳入 git）。
- 规格评分虚高：9.2 标注为 "spec-level; conditional on Pilot Gates"，gate 从未运行；router 的 READY 7.0 也是在 pilot 全部推迟时给出的。
- 被禁止的路线反复复活：RESEARCH_BRIEF L8 禁止改图与 LLM，v8-0 又锁定了 LLM。
- 基础设施漂移：数据只在服务器上而 SSH 不可用；有 `.git.corrupt`；删过 30 个文件；log.md 没有 5 月条目；`NLPCC/code` 只是 stub。

### 4.8 统一假设 H*（未验证）

4.1–4.3 看起来是同一个现象的三个侧面：强基模型在 TW20 上剩下的错误，集中在现有特征无法区分、或标签本身有歧义的账号上。如果 H* 成立，所有 "冻结基模型 + 事后纠错" 类方案的上限都由数据决定，而不是由方法决定。H* 是 §6 P1 要回答的门控问题。

## 5. 新方向的 idea-evaluator 评估

§5 评估 §3 之外的新方向；§3 中已判 Reject 的方案不再重复评分。候选三的定位机制已被本项目的数据追平（§5.3 缺陷 2），按短路格式只给第一印象、致命缺陷与结论。

评估口径：
- 用户资源按 §7 的假设：单人，每周约 20 小时，本地 RTX 4060 8 GB，服务器暂不可用。
- 新颖性只依据 2026-09-30 的文献检索。检索只到元数据层面（作者、年份、venue、做了什么），没有读全文，也不引用检索片段中的数字；唯一的例外是 §5.2 的 Hays et al.，已对照全文的 Table 1–2 核对。"没有检索到" 只表示按这些关键词没有找到直接重合的工作，不等于新颖。
- 五个维度都从 5 分起评，有实测数据或站得住的机制论证才上调；只有机制论证的分数标为 "机制推断，未经数据确认"。
- 归因规则：如果支持某一维度的证据也可以归因到别的因素（如基模型或纠错器太弱），在消融把效应隔离出来之前，该维度最多上调一档（本文取 6）。

### 5.1 候选一：决策有效性（§6 P2）

范围：TW22 先验偏移审计、bot 流行率的 PPI 置信区间、FDR 受控的 conformal 标记。三者都只复用检测器输出。

#### 1. 第一印象

- 论文类型：New Setting。把评测目标从单一阈值下的 Acc/F1 换成部署层面的决策有效性。
- 一句话故事：在先验偏移和标注预算有限的条件下，给出有覆盖保证的 bot 流行率区间和 FDR 受控的 bot 标记，并检验 TW22 上的方法排名在先验校正后是否改变。

#### 2. 最接近的工作与致命缺陷

检索词组：PPI/ASI × bot prevalence；prevalence × calibration/quantification；PPI × network/dependent data；conformal selection × FDR；graph/link × conformal FDR；conformal × fraud/anomaly/bot。

| 最接近的工作（元数据） | 不同的轴 |
|---|---|
| Tan et al., "BotPercent: Estimating Bot Populations in Twitter Communities"（2023, Findings of EMNLP） | 对象相同（社区 bot 比例）；机制不同，检索中没有看到 PPI 式的有效置信区间（未读全文，需要核实） |
| Wu & Resnick, "Calibrate-Extrapolate: Rethinking Prevalence Estimation with Black Box Classifiers"（2024, ICWSM） | 机制最接近（黑盒分类器加标注校准）；对象是一般内容，不是 bot 账号 |
| Giroux et al., "Unmasking social bots: how confident are we?"（2025, EPJ Data Science） | 对象相同；从标题看侧重检测置信度，不是 FDR 选择 |
| Marandon, "Conformal link prediction for false discovery rate control"（2024, TEST） | 同样在图上做 FDR 控制，但对象是边，不是节点或账号 |
| Yu et al., "Robust Conformalized Selection with Noisy Responses"（2026, arXiv） | 处理校准标签带噪时的选择；对象是通用数据，不涉及 bot 或图 |

方法来源：PPI（Angelopoulos et al., 2023, Science）、PPI++（Angelopoulos et al., 2023, arXiv）、Active Statistical Inference（Zrnic & Candès, 2024, ICML）、Stratified PPI（Fisch et al., 2024, NeurIPS）；conformal selection（Jin & Candès, 2023, JMLR）、conformal p-value（Bates et al., 2023, Ann. Statist.）。CF-GNN（Huang et al., 2023, NeurIPS）给的是预测集覆盖，不是选择。

检索结论：在 "PPI/ASI × bot 流行率"、"PPI × 社交图依赖"、"conformal selection × bot 账号"、"图节点级 FDR 选择" 这些关键词下，没有检索到直接重合的工作。TW22 先验偏移审计这一部分没有检索，新颖性未核实，需要文献检查。

| # | 缺陷 | 严重度 | 应对 |
|---|---|---|---|
| 1 | 新颖性：流行率估计已有 BotPercent、Calibrate-Extrapolate，bot 检测置信度已有 Giroux et al.；差异只在有限样本覆盖、标注预算与图依赖这几个轴上 | MAJOR | 前两者能复现就作为基线，主打有限样本覆盖与标注预算；如果它们只给点估计，就比较同等标注预算下的点估计误差与实测覆盖率，不比区间宽度；都不占优就降级为评测附录 |
| 2 | 保证的两个前提在现有产物上不成立：(a) hss 契约下被采到的 valid/test 真标签进入过训练 loss（§3.G F2），校准集与测试集不再可交换；(b) "金标准" 是 benchmark 标签，如果 P1 发现大量歧义，区间覆盖的只是标签定义下的流行率 | MAJOR | 只用没有标签暴露的检测器（按 train 标签训练的复现基线，或按 §6 P0 第 5 条用 seed_only 契约重训的模型），hss 与 seed_only 下实测覆盖率的对比本身可以作为一个结果；文中写明估计目标，并用 P1 的盲标注子集做小规模真实金标准 |

没有 CRITICAL：这条线还没有实验，现有数据既不支持也不反驳它的核心机制。

#### 3. 生命周期与能力匹配

| 方面 | 用户情况 | 评估 |
|---|---|---|
| 类别 | 跨学科：统计推断方法迁移到 bot 检测 | 方法来源已成熟（2023–2024），bot 场景的直接应用没有检索到 |
| 窗口期 | 估计 12–18 个月 | 2025–2026 已有 bot 置信度与带噪 conformal selection 的相关工作，窗口在收窄。这是推断，需要用户按自己对子领域的了解复核 |
| 每周有效时间 | 约 20 小时（假设） | P0 前置约 1–2 周，P2 本身约 4–6 周 |
| 匹配度 | 单人、CPU 为主 | Yellow：依赖 P0 的 seed_only 重训与逐节点概率 |

#### 4. 五维评分

| 维度 | 分数 | 依据 | 提升建议 |
|---|---|---|---|
| Higher（效果） | 6 | 机制推断，未经数据确认。TW22 的 LM 阶段有 4/5 个 seed 塌缩到全判 human（§4.5）；如果塌缩模型仍保留排序信息，按 val 先验调阈值可以救回一部分 bot-F1 与 Macro-F1。threshold-free 指标不会因此变化 | 验证实验：动作 1 |
| Faster（效率） | 5 | 没有依据。推断开销可以忽略，但速度不是这条线的卖点 | 无 |
| Stronger（稳健） | 7 | 机制推断，未经数据确认。在可交换假设下，覆盖与 FDR 保证不依赖检测器好坏：检测器越差，区间越宽、标记越少，但保证仍成立。上限受缺陷 2 约束 | 验证实验：动作 2、3 的实测覆盖率与 FDR |
| Cheaper（成本） | 7 | 机制推断，未经数据确认。PPI 用检测器输出减少同等区间宽度所需的人工标注；TW20 的强基模型应能明显收窄区间，TW22 的塌缩模型收益可能接近零 | 验证实验：动作 2 的标注预算曲线 |
| Broader（通用） | 7 | 机制推断，未经数据确认。只需要检测器分数和少量标注，与模型族无关，可以直接用到 TW20、TW22 和 LLM 时代的数据集 | 验证实验：至少 3 个模型族 × 2 个数据集 |

没有维度到 8 分，达不到 Strong Accept 的条件（≥2 个维度 8 分以上）。

#### 5. 范式探针

| 探针 | 是/否 | 理由 |
|---|---|---|
| 第一性原理 | 是 | 挑战了一个隐含假设：单一阈值下的 Acc/F1 能代表部署价值。部署与审计更常用的是流行率和标记的错误率（BotPercent、Varol et al. 都以流行率为目标） |
| 房间里的大象 | 是 | TW22 train 与 test 的 bot 比例是 7.80% 对 29.44%，本项目复现中多个模型因此塌缩，但之前没有被当作主问题（§4.5） |
| 技术周期 | 否 | PPI 与 conformal selection 是迁移来的统计工具；LLM 时代的 bot 让问题更迫切，但这条线并不依赖新技术 |
| Hamming 规则 | 是 | 问题重要且有可行的攻击路径：流行率与标记可靠性是部署方真正要的量，PPI/ASI 与 conformal selection 提供了现成的工具 |

颠覆潜力：可能。"房间里的大象" 有本项目的实测数据（§4.5）；第一性原理与 Hamming 两项只停留在机制层面，没有数据支撑。

#### 6. 可行性

| 风险 | 等级 | 缓解 |
|---|---|---|
| 算力 | 中 | PPI 与 conformal selection 本身只需 CPU；前置的 seed_only 重训能否在 8 GB 内完成没有验证，不行就先用较轻的复现基线 |
| 数据 | 中 | 本地没有 TW22 全量（缺 user.json、node.json、tweets），但样本版与全量先验一致（§4.5），动作 1 在样本版上做；动作 2、3 先在 TW20 上做；真实金标准只有 P1 的盲标注子集 |
| 工程 | 低 | PPI、PPI++ 与 conformal selection（BH 步骤）各几十行，自己实现，不引入新依赖 |
| 时间 | 中 | P0 前置 1–2 周，加 P2 本身 4–6 周；按每周 20 小时，适合期刊扩展的周期（假设） |

#### 7. 结论

Accept with Revisions：通过验证实验后值得推进。0 个 CRITICAL；最高分 7 且都是机制推断，不满足 Strong Accept。需要修改的地方：写明估计目标是 benchmark 标签定义下的流行率；只用没有标签暴露的检测器；如果 BotPercent、Calibrate-Extrapolate 能复现，就作为基线。

优先的三个动作：
1. 先验偏移审计（最便宜）：在按 P0 第 3 条选定的 TW22 样本版上做（现有基线分散在 T22-A 与 T22-B，另一版上的需要重跑）。P0 补存逐节点概率后，对复现基线做按 val 先验调阈值、EM 先验重估、BBSE 与 logit adjustment，报告 bot-F1、Macro-F1 与 AUROC/AUPRC 下的排名变化。kill 条件事先锁定：校正后排名不变，且塌缩模型的 Macro-F1 提升在 seed 间波动之内。
2. PPI 区间：在 TW20 上把 test 标签当作金标准池，按标注预算 n ∈ {50, 100, 200, 500} 重复抽样，比较经典估计、PPI、PPI++ 与按社区分层的 PPI 的区间宽度和实测覆盖率。
3. FDR 标记：seed_only 检测器在 val 上校准，用 conformal selection 在 test 上做目标 FDR q ∈ {0.05, 0.1, 0.2} 的标记，报告实测 FDR 与标记数；再在 hss 契约的模型上重复，量化可交换性被破坏后 FDR 的偏差。如果 val 同时用于早停，要先把 val 拆成选模型与校准两半。动作 2、3 在 TW20 上跑通后，用至少 3 个模型族的检测器在按 P0 第 3 条选定的 TW22 样本版上重复，这是 Broader 一行的验证实验。

### 5.2 候选二：残差错误还可不可纠（§6 P1）

范围：共识错误、kNN 标签纯度、confident learning 的疑似错标分数，加上双人盲标注。回答 §4.8 的 H*。

#### 1. 第一印象

- 论文类型：Novel Problem。不提新检测器，而是给 "冻结基模型 + 事后纠错" 这一类方法定一个由数据决定的上限。
- 一句话故事：§3.B–G 中跑过实验的五类事后定位与纠错方案（B、C、E、F、G）都没有稳定的净增益，D 只有规格、关键组件被 E、F 驳倒；本文在 TwiBot-20/22 上做实例级审计，量化强基模型的残差错误里有多少来自特征不可分或标签歧义，并据此重新解读 TW20 上各方法之间的小幅差距（多数复现基线的 Macro-F1 落在 .85–.87 之间，§3.H）。

#### 2. 最接近的工作与致命缺陷

检索词组：Hays 2023 后续 / TwiBot-22 label quality；noisy-label GNN × bot；confident learning × bot。

| 最接近的工作（元数据） | 不同的轴 |
|---|---|
| Hays et al., "Simplistic Collection and Labeling Practices Limit the Utility of Benchmark Datasets for Twitter Bot Detection"（2023, WWW） | 对象部分相同（分析了 TwiBot-20，没有 TwiBot-22；TwiBot-20 上只看 verified 一个特征的深度 1 决策树，Acc/F1 只比他们引用的 SOTA 低 0.05/0.03）；设定是数据集级的采集与标注批判，不是实例级的标签错误估计 |
| Kolomeets et al., "Experimental Evaluation: Can Humans Recognise Social Media Bots?"（2024, BDCC） | 人类识别实验，不是标签审计 |
| Xu et al., "TRUST: Towards Robust Social Bot Detection via Uncertainty-Guided Pseudo-Labeling and Graph Structure Purification"（2026, Findings of ACL）；Zhang et al., "RABot: Reinforcement-Guided Graph Augmentation for Imbalanced and Noisy Social Bot Detection"（2026, arXiv） | bot 图上的抗噪训练；目标是训练鲁棒性，不是噪声诊断 |
| Yang et al., "FISSION: Label Augmentation for Bot Detection"（2026, arXiv） | 机制是标签扩增，不是审计 |

方法来源：NRGNN（Dai et al., 2021, KDD）；confident learning（Northcutt et al., "Confident Learning: Estimating Uncertainty in Dataset Labels"，2021, JAIR；不在本次检索记录中，需要核实）。

检索结论：在 "confident learning × TwiBot-20/22"、"TwiBot-22 标注质量审计" 这些关键词下，没有检索到直接重合的工作。

| # | 缺陷 | 严重度 | 应对 |
|---|---|---|---|
| 1 | 新颖性："bot 基准标签有问题" 本身不新，Hays et al. 已在数据集层面提出 | MAJOR | 贡献落在实例级审计与纠错上限的关系上：用同一套审计解释 §3.B–G 为什么都失败，而不是泛泛地说 "标签有噪声" |
| 2 | 判定效度：单人标注算不出 κ；如果共识错误来自共用同一 embedding 的基线，它只反映共同盲区，不等于标签错误。另外 oracle 上界（26/0/+26、32/0/+32，§3.F）说明确有一部分残差可修，H* 不能写成 "全部不可修" | MAJOR | 找第二位标注者；基线覆盖至少 3 个特征族，kNN 纯度在 iter_-1 与 iter_2 两种 embedding 下都算；判定阈值按比例预注册（§6 P1） |

没有 CRITICAL：H* 还没有被直接检验过。oracle 上界只说明存在可修的子集，没有反驳 "残差错误集中在不可分或有歧义的账号上"。

#### 3. 生命周期与能力匹配

| 方面 | 用户情况 | 评估 |
|---|---|---|
| 类别 | 数据密集：实例级审计加人工标注 | 需要原始账号内容和第二位标注者 |
| 窗口期 | 估计 12–24 个月 | 取决于 TwiBot-20/22 是否还是主流基准；LLM 时代的新数据集可能缩短窗口。这是推断，需要用户复核 |
| 每周有效时间 | 约 20 小时（假设） | 计算部分约 2 周；盲标注按 150 个账号、每个约 4 分钟粗估，每人约 10 小时 |
| 匹配度 | 单人、CPU 为主 | Yellow：依赖 P0 第 7 条的逐节点概率和第二位标注者 |

#### 4. 五维评分

| 维度 | 分数 | 依据 | 提升建议 |
|---|---|---|---|
| Higher（效果） | 5 | 没有依据。诊断本身不提升指标；只有 P1 找到可纠错子集、进入 P3 后才可能间接提升 | 无 |
| Faster（效率） | 5 | 没有依据 | 无 |
| Stronger（稳健） | 6 | 有部分数据，但归因没有隔离。oracle 加边只修对 RGCN 157 个错误中的 4 个（§4.3），W→R≈R→W（§4.2），都与 H* 一致，但也可能只是纠错器太弱；按 §5 开头的归因规则封顶 6 | 验证实验：动作 2 中错误组与对照组 "原标签存疑" 比例的差异 |
| Cheaper（成本） | 7 | 机制推断，未经数据确认。复用 P0 第 7 条补存的输出，只需 CPU 和两人各约 10 小时的标注；如果 H* 成立，可以避免继续在事后纠错路线上投入（§3.B–G 从 04-01 持续到 06-20） | 验证实验：动作 1 能否在 2 周内完成 |
| Broader（通用） | 7 | 机制推断，未经数据确认。审计协议不依赖具体数据集或模型，结论解释的是一类方法而不是单个 pipeline | 验证实验：在按 P0 第 3 条选定的 TW22 样本版上重复动作 1 的计算诊断 |

没有维度到 8 分，达不到 Strong Accept 的条件。

#### 5. 范式探针

| 探针 | 是/否 | 理由 |
|---|---|---|
| 第一性原理 | 是 | 挑战两个隐含假设：benchmark 标签就是真值；残差错误能靠更好的方法修掉 |
| 房间里的大象 | 是 | Twitter bot 基准的采集与标注做法在数据集层面被批评过（Hays et al.；涵盖 TwiBot-20，不含 TwiBot-22），但按上面的检索，没有找到它对纠错上限影响的实例级量化 |
| 技术周期 | 否 | 这条线不依赖新技术；LLM 代标在本地无法复查（§3.F F2）；§3.F 的零样本 Macro-F1 0.49 是对照 benchmark 标签算的，分不开 LLM 出错与标签歧义，不能用来论证 LLM 不适合标注 |
| Hamming 规则 | 是 | 攻击路径现成（共识错误加盲标注）；如果 TW20 强基模型的残差主要来自标签歧义，各方法之间的小幅差距就需要重新解读 |

颠覆潜力：可能。三个 "是" 都没有本项目的实测数据：第一性原理与 Hamming 两项停留在机制层面，"房间里的大象" 依据的是文献与检索。

#### 6. 可行性

| 风险 | 等级 | 缓解 |
|---|---|---|
| 算力 | 低 | kNN 纯度与 confident learning 只需 CPU；逐节点概率与 §5.1 动作 1 共用 P0 第 7 条补存的同一批结果 |
| 数据 | 中 | 盲标注需要原始简介与推文；TW20 的原始内容是否齐全没有核实（第二数据集只重复计算诊断，不做标注），先核实再抽样 |
| 人员 | 高 | 单人算不出 κ。找不到第二位标注者时，只能报告同一人间隔两周的重测一致性，并把结论标为提示性 |
| 工程 | 低 | confident learning 需要样本外概率：只用 train 标签训练的复现基线的 test 预测或交叉验证预测都可以；hss 契约的模型训练时见过部分 test 标签，不能用 |
| 时间 | 低 | 约 2 周计算，加每人约 10 小时标注；TW20 部分可以排在 P2 之前，TW22 上的重复要等 §5.1 动作 1 的先验校正 |

#### 7. 结论

Accept with Revisions：通过验证实验后值得推进。它是 P3 的门控，P2 的估计目标也依赖它。0 个 CRITICAL；最高分 7 且是机制推断。需要修改的地方：基线覆盖至少 3 个特征族（如元数据属性、文本、图结构）；阈值预注册；找到第二位标注者；只用样本外预测，不用 hss 契约的模型。

优先的三个动作：
1. 计算诊断：P0 补存逐节点概率后，统计 TW20 test 上共识错误占残差错误的比例、它们与 BotRHG 路由集合的重合度、错误节点与正确节点在 iter_-1 与 iter_2 两种 embedding 下的 kNN 标签纯度差异，并给出 confident learning 的疑似错标名单。TW20 跑通后，在按 P0 第 3 条选定的 TW22 样本版上重复这一步（排在 §5.1 动作 1 的先验校正之后，见 §6 P1），这是 Broader 一行的验证实验。
2. 盲标注：按 §6 P1 分层抽样（共识错误与被判对的节点各半），两人独立标注，报告 Cohen's κ，以及错误组与对照组 "原标签存疑" 的比例差。
3. 判定：按 §6 P1 的预注册阈值给出三种结果之一。只满足第一条：关闭事后纠错路线，把 H* 作为下一篇的核心发现。满足第二条：可纠错子集具备转入 P3 的条件（是否启动由用户决定）；两条都满足时 H* 记为部分成立，只满足第二条时 H* 记为未获支持。两条都不满足：记为未决，不启动 P3。判定结果不论正负，都写进下一篇的评测部分。

### 5.3 其余候选（精简评估）

#### 候选三：局部纯度诊断 + 协同证据纠错（短路格式）

第一印象：
- 论文类型：Novel Method。
- 一句话故事：在冻结检测器上做事后的局部纯度诊断，只在纯度低的区域引入行为协同图（而不是关注图）上的证据来修正判定。

| 最接近的工作（元数据） | 不同的轴 |
|---|---|
| Yang et al., "SeBot: Structural Entropy Guided Multi-View Contrastive Learning for Social Bot Detection"（2024, KDD） | 训练期表示学习，不是冻结检测器上的事后诊断 |
| Wu et al., "BotSCL: Heterophily-aware Social Bot Detection with Supervised Contrastive Learning"（2024, ICPR） | 同样在训练期处理异配 |
| He et al., "Boosting Bot Detection via Heterophily-Aware Representation Learning and Prototype-Guided Cluster Discovery"（2025, KDD） | 最接近 "异配 + 群体发现"，但属于端到端表示学习 |
| Pacheco et al., "Uncovering Coordinated Networks on Social Media: Methods and Case Studies"（2021, ICWSM） | 协同网络发现，不用于修正账号级检测器 |
| Gopalakrishnan et al., "Density-aware Walks for Coordinated Campaign Detection"（2025, ECML-PKDD） | 同上，对象是协同活动 |

检索结论：在 "kNN 标签纯度 × bot"、"协同证据 × 冻结检测器修正" 这些关键词下，没有检索到直接重合的工作。

| # | 缺陷 | 严重度 | 应对 |
|---|---|---|---|
| 1 | 新颖性：外部没有检索到重合，但 "冻结检测器 + 邻域信号定位" 本项目已经做过（WRT；NCP，§3.G），剩下的差异只在协同证据这一输入 | MAJOR | 只有缺陷 2 被推翻才有意义 |
| 2 | 定位机制已被数据追平：本项目中的邻域类定位信号都被 MSP 追平或输给随机对照。kNN 超图 router 最好为 0.8597（相对 MSP 的 AURC 降幅 CI 含 0，seed3 为负），NCP 为 0.8590，MSP 为 0.8576（§3.E）；按局部冲突剪边不如同预算随机（§4.3）。表示空间的 k 近邻信号也单独测过（`L/experiments/knn_router_nonmodel_sweep_20260610/`，TW20，32 次运行，k=4/8/16，其中 24 次是 seed 1）：把近邻的预测风险与预测分歧叠加到基分数（posthoc calibrated ranker 的风险分）上，在 tune 集（val 的一半，1182 个节点）上逐次事后取最好的一族，error-AUROC 与基分数之差也只有 −0.0016~+0.0078（均值 +0.0042；seed 2 上不超过 +0.0001）；用带标签近邻非一致性的 NCP 式局部族，均值都低于基分数。没有测过的只剩候选三的确切形式（train 邻居的真标签多数与预测不一致）和协同证据 | CRITICAL（定位机制已被数据追平） | 不辩护；确切形式交给 P1 第二条检验 |

结论：Reject and Pivot。不作为独立方向；邻域纯度降级为 P1 的诊断量，不再当路由信号。只有 P1 的第二条判定成立，才算推翻这一结论，§6 P3 也才具备启动条件（是否启动由用户决定）。

#### 候选四：LLM 时代新数据作分布外验证场

范围：不提新方法，只借新数据检验 §5.1 的先验校正、PPI 区间与 FDR 标记在分布偏移下是否仍成立。检索记录有标题、年份、venue 与论文或仓库链接，没有作者：

| 数据（元数据） | 按标题与链接的性质 |
|---|---|
| Fox8："Anatomy of an AI-powered malicious social botnet"（2023；JQD:DM 2024） | AI 驱动的恶意 botnet，有公开仓库 |
| BotSim-24："BotSim: LLM-Powered Malicious Social Botnet Simulation"（2024, arXiv；正式 venue 未确认） | LLM 驱动的 botnet 模拟，有公开仓库 |
| MisBot："How Do Social Bots Participate in Misinformation Spread?"（2025, EMNLP） | bot 参与虚假信息传播 |
| Chirper.ai："Characterizing LLM-driven Social Network: The Chirper.ai Case"（2025；CSCW 2026） | LLM 驱动的社交网络 |
| "Linguistic Differences between AI and Human Comments in Weibo"（2025, CCL） | 微博上 AI 与人类评论的语言差异 |

主要缺陷（MAJOR）：不是独立贡献；BotSim-24、Chirper.ai 是模拟或 LLM 驱动的环境，分布未必代表真实平台；数据与标签能否直接下载、许可如何都没有核实；本地只有 8 GB。

结论：不作为独立方向评估，也不打五维分，并入 §6 P2 的扩展实验。

## 6. NLPCC 后续探索重点

### P0 证据卫生（会议 11-03 前，约 1–2 周）

1. 会议报告的数字与已录用论文保持一致，但不强调相对基线的领先幅度（训练契约问题见 §3.G F2）。如果被问到显著性，就按 E-ES2-004 如实回答，可以讲 "错误捕获率是随机的 2.6–3.7 倍"，但要同时说明 MSP 同档，重点放在负面发现上。
2. SSH 恢复后，第一件事是把服务器上的产物同步回仓库：主表的 ±std、TW22 的 82.41、HyperScan 的 csv。补不齐来源的数字，后续工作不再引用。
3. 冻结一个统一的评测 harness：
   - 固定数据版本：TW20 用 LMBot split；TW22 只用 T22-A 或 T22-B 中的一个，并把版本号写进产物名。
   - 每个结果跑 5 seed。
   - 同时报告 Acc、Macro-F1、bot-F1、AUROC、AUPRC。
   - 强制输出逐节点概率。
   - 对照组固定为 MSP 路由、同预算 shuffled、all-nodes、low-only。
   - 用配对检验加 Holm 校正。
   - 训练只监督 seed 行，验证和测试只在 canonical 节点上去重计算；harness 断言 valid/test 标签不会进入训练 loss。
4. 在后续论文中修正已知的口径问题（camera-ready 按用户此前的决定不改）：消融 "w/o Hypergraph Correction" 行的命名、SEBot TW22 行（单次 run 而非 5 seed 均值）、BotBR 86.72 由取整日志反算。
5. 量化标签暴露（§3.G F2）：
   - 统计训练采样子图中 valid/test 行的占比，以及它们在 loss 中所占的权重。
   - 在 seed_only 契约下重跑 base 与 C2，和 hss 下的同配置结果对比。C0 是否重跑由用户决定（06-19 已决定不再做纠错分支实验）。
   - 本地重跑前先确认环境：默认 Python 里 `import dhg` 失败，涉及 DHG 的配置需要先装好依赖；iter_2 embedding 本地已有。
6. HyperScan 泄漏审计：检查超边构建是否用到了 test 标签或 test 期的特征统计，并确认官方训练是否也对全部采样行算 loss；复核 seed2 离群点；与论文自报的 TW20 F1 87.2 比较前，先确认两边 F1 口径一致。
7. 给复现基线补存逐节点概率，这是 P1 与 P2 的前置条件。

### P1 可纠错性诊断（门控实验，约 2 周，CPU 即可）

目的是回答 §4.8 的 H*（评估见 §5.2）。

- 共识错误：在 TW20 test 上统计被多个复现基线同时判错的节点，看它们占残差错误的比例，以及与 BotRHG 路由集合的重合度。
- kNN 标签纯度：在 embedding 空间里比较错误节点与正确节点的 k 近邻标签纯度。
- confident learning：用样本外概率（只用 train 标签训练的复现基线，不用 hss 契约的模型）给出疑似错标名单，只作交叉核对，不进入下面的判定。
- 第二数据集：以上三项计算诊断在 TW20 上跑通后，在按 P0 第 3 条选定的 TW22 样本版上重复，作为 §5.2 Broader 一行的验证实验；判定仍以 TW20 为准。TW22 上多个模型塌缩到全判 human（§4.5），它们的错误几乎都是 bot，会混进共识错误，所以这一步排在 §5.1 动作 1 之后，用按 val 先验调阈值后的预测计算，并报告剔除塌缩 seed 前后的结果。与 BotRHG 路由集合的重合度要等 P0 第 2 条同步服务器产物后再算（TW22 的 BotRHG 结果本地没有产物，§3.G）。
- 小规模盲标注：分层抽样 100–200 个账号（共识错误与被判对的节点各半，后者作对照，用来估计 "原标签存疑" 的基础比例），至少 2 人在不看原标签的情况下独立标注，报告 Cohen's κ 和 "原标签存疑" 的比例（"原标签存疑" 指两份盲标一致且与 benchmark 标签相反，或至少一份标为无法判断）；找不到第二位标注者时，按 §5.2 改报同一人间隔两周的重测一致性，结论标为提示性。不用 LLM 代标：本地没有可复查的算力（§3.F F2）。296 个被路由节点上零样本 Macro-F1 为 0.49、低于 base 的 0.63，但这是对照 benchmark 标签算的，分不开 LLM 出错与标签歧义，不作为排除理由。
- 预注册判定（阈值在运行前锁定，下面只是建议值）：
  - 第一条：共识错误占残差错误 ≥50%，且盲标注中共识错误样本被判为 "原标签存疑" 的比例 ≥30%，并高于对照组（比例差的 95% 置信区间不含 0）。只满足这一条时，正式关闭事后纠错路线，把 H* 本身作为下一篇的核心发现。
  - 第二条：存在一个可分的可纠错子集，诊断与部署两个条件都要满足。诊断上，邻域纯度高却被判错的节点（k 近邻 train 节点中 ≥80% 与它的真标签相同，模型却判错）占残差错误 ≥20%；部署上，不用 test 标签的对应信号（train 邻居的多数标签与模型预测不一致）挑出这些节点的效果，在 5 seed 上显著优于 MSP 与 ensemble entropy。满足这一条才具备进入 P3 的条件（是否启动由用户决定），届时重新设计针对性的纠错器，并沿用 P0 的全部对照。
  - 两条都满足：H* 记为部分成立，P3 只处理这个子集。只满足第二条：H* 记为未获支持。两条都不满足：记为未决，不启动 P3，结果作为负面发现写进评测部分。

### P2 主线候选：从 "多 0.2 pp" 转向 "决策有效性"（评估见 §5.1）

- TW22 先验偏移审计：对已复现的基线做 label-shift 校正（按 val 先验调阈值、EM 先验重估、BBSE、logit adjustment），同时报告 threshold-free 指标，看方法排名是否改变。
- 部署层保证：
  - 用少量人工标注加检测器输出，给出 bot 流行率的 PPI 置信区间。
  - 用 conformal selection 做 FDR 受控的 bot 标记。
  - 先在 TW20 上跑通，再用至少 3 个模型族的检测器在选定的 TW22 样本版上重复（§5.1 Broader 一行的验证实验）。
- 两者都只复用检测器输出（P0 第 5 条的 seed_only 重训结果与第 7 条补存的逐节点概率），不需要训练大模型。
- 扩展实验（§5.3 候选四）：挑 1–2 个有公开仓库的 LLM 时代数据（Fox8、BotSim-24；数据构成与标签能否直接下载都未核实），检验先验校正、PPI 区间与 FDR 标记在分布偏移下是否仍成立，不在上面开发新方法。

### P3 可选：仅在 P1 第二条判定成立时

基于邻域纯度和协同行为证据的定向纠错器，只处理 P1 挑出的可纠错子集，必须通过 P0 的全部对照；判定成立后是否启动仍由用户决定（06-19 已决定不再做纠错分支实验）。作为路由信号的邻域纯度已判为 Reject and Pivot（§5.3）；P1 第二条判定成立，就是推翻这一结论所需的新证据。

### 停止清单

- 不再设计新的路由器或风险分数，除非它在 5 seed 上显著优于 MSP 和 ensemble entropy。
- 本地没有可复查的算力时，不做 LLM 改边或 LLM 纠错。
- 不写没有 pilot 的 pipeline 规格，也不给规格打分。
- 表格中不出现单 seed 结果或预期值。
- 已判死的路线（改图、LLM 纠错、分歧触发），没有推翻原结论的新证据就不重启。

### 自动 kill gate（每个新方法都要过）

1. 5 seed，对照组为 MSP 路由、同预算 shuffled、all-nodes、low-only。任何一项配对检验在 Holm 校正后 p ≥ 0.05，结论就降级为 "无显著增益"。
2. McNemar 检验 W→R 显著多于 R→W；不满足就不能写 "纠正"。
3. 以上由脚本输出 pass/fail，不靠人工解读。fail 直接写入负面结果，不继续调参。

## 7. 假设与未核实项

- 假设（用户未说明，需要确认）：
  - 后续目标默认为 NLPCC 论文的期刊扩展，或下一篇会议论文。如果目标是更高的 venue，P2 的新颖性门槛要相应提高。
  - 每周投入默认约 20 小时，单人。
  - 服务器能否恢复不确定。如果不能恢复，所有 "仅服务器" 数字作废，P0 第 2 条改为本地重跑。
  - 本地只有 RTX 4060 8 GB，所以 P0–P2 都按 CPU 或 8 GB 可完成来设计。
- 未核实：
  - TW22 82.41 与主表 ±std 的来源；HyperScan TW20 的原始 csv。
  - §3.E、§3.F、§4.3、§4.4 的数字大多来自一次只读审计。本地逐项复核过的是：budget_curve 的两对 md5（seed_1）、标签暴露的代码路径、契约对照表（文档记录）、TW22 base5 的指标文件、主表 csv 的两份备份、BotSay 的 P/R/F1、Qwen3 G0 与 expert/refiner 的 fix/break、oracle 与 embedding_compare 的 csv、E-ES1-004/E-ES5-002 登记项，以及 E-ES1-005~012 与 E-ES3-004/005/007 登记项、PHASE1/3/4/5 表格与 FRMI E1 的 AUROC、selector/MoE/oracle 的 fix/break 行、R1/R4/R8/v8 评审分数、risk gate 与 WRT 的代码位置。其余 router screening 数字（各 error-AUROC）直接取自审计，没有重算。
  - 本文引用又做过一轮独立抽查，已据此修正数字与出处。抽查本身也有误判：它曾报告 "β peaks at 10%" 在回应稿中找不到，实际在 `RB/response/k9d8_paste_ready.md` L15。
  - 训练采样子图中 valid/test 行的实际占比；服务器上跑的 `LLMbot/code` 是否与本地一致（该目录不在 git 跟踪中）。
  - HyperScan 官方实现是否对全部采样行算 loss（submodule 本地无源码）。
  - C2 与契约对照表中 hss RGCN 行四位小数相同，是否为同一次 run；C0–C5 用的是 labeled 还是 full_graph_support 输入变体。
  - 5 seed 主实验中其余 seed 未完成的原因。审计转述为 OOM 被终止和 `import dhg` 失败，本地 `_reports` 中没有对应日志。
  - LLMbot 原型阶段的结论；Embedding-Dominant 提案（04-19）的内容。
  - DGP-fraud、tropical、shi_2025_hyperbot、BotUMC、BotLGT、GAugLLM、LOGIN_init、botsay_official 的迁移状态。
  - 复现基线是否在别处存有逐节点预测。

