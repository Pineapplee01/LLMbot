# LMBot 项目全景上下文

> 本文档为新 Claude 会话提供完整的项目背景。涵盖项目定位、代码结构、模型架构、训练流水线、数据格式与运行方式。

---

## 1. 项目定位

LMBot 是一个 **Twitter 机器人检测系统**，通过融合语言模型（LM）与图神经网络（GNN）来识别社交平台上的自动化账号。

项目代码：
- **（`LLMbot/` 目录）**：基于预计算 Qwen3-Embedding-8B 嵌入的 SeGA（Structure-Enhanced Graph Attention）框架，采用不确定性感知的多阶段融合训练（**当前活跃开发**）

**所有开发工作都在 `LLMbot/` 目录下进行。**

---

## 2. 代码与文件清单


### 2.1 代码（`LLMbot/`，当前活跃）

核心思路：使用预计算的 Qwen3 嵌入（4096维），通过 RGT 编码图结构，VIB 编码文本不确定性，Gate 网络自适应融合。

| 文件 | 职责 |
|------|------|
| `main.py` | 入口，参数解析，模型构建，4阶段训练编排 |
| `train.py` | `QwenPrecomputedTrainer`：训练循环、冻结策略、评估 |
| `model.py` | `VariationalTextAdapter`, `EvidentialGraphHead`, `ReliabilityAwareFusion` |
| `GNNs.py` | `RGT`, `RGCN`, `SimpleHGN`, `GAT` + 工厂函数 `build_gnn()` |
| `RGT.py` | `SemanticAttention`（node-wise）, `RGTLayer` |
| `utils.py` | `EdlLoss`, `AvUCLoss`, `load_raw_data`, `compute_feature_homophily` 等 |
| `generate_qwen3_embeddings.py` | `Qwen3EmbeddingGenerator`：用 Qwen3-Embedding-8B 生成嵌入 |

### 2.2 子包说明 （开发时忽略）

- **`LLMbot/sega/`**：SeGA 框架的模块化重构版（v2.0），包含 configs/data/models/training/evaluation 子模块。提供 dataclass 配置系统和更清晰的 API，但当前主训练流程仍使用 `LLMbot/main.py`。
- **`LLMbot/llmbot/`**：分阶段 Trainer 包，将训练器按阶段拆分为 `VIBTrainer`, `GNNTrainer`, `GateTrainer`, `FusionTrainer`。是 `train.py` 中 `QwenPrecomputedTrainer` 的重构替代方案。

---

## 3. 核心架构

### 3.1 模型组件

#### RGT（Relation Graph Transformer）

文件：`LLMbot/RGT.py` + `LLMbot/GNNs.py`

RGT 是默认的 GNN backbone，核心思想是对不同关系类型（follower/following）分别做消息传递，再用注意力聚合。

**SemanticAttention**（`RGT.py:6-49`）：
- 输入：`z [N, R, D]`（N=节点数, R=关系类型数, D=隐藏维度）
- 多头注意力，每个头独立计算 node-wise 的关系权重
- `softmax(dim=1)` 在关系维度归一化，每个节点有独立的关系权重分布
- 输出：`[N, D]`

**RGTLayer**（`RGT.py:51-104`）：
- 对每种关系类型独立执行 `TransformerConv`
- 门控残差连接：`Tanh(new) * gate + old * (1-gate)`
- `torch.stack(outputs, dim=1)` → `SemanticAttention` 聚合
- 输入：`features [N, D]`, `edge_index_list [list of [2, E_r]]`
- 输出：`[N, D]`

**RGT 类**（`GNNs.py:160-234`）：
```python
forward(x, edge_index, edge_type) → [N, hidden_dim]
```
- `input_proj`：4096 → hidden_dim（默认512），含 LayerNorm + GELU + Dropout
- `prepare_data_for_RGT`：按 edge_type 拆分为 per-relation edge_index_list
- 堆叠 N 层 RGTLayer（默认2层）
- 返回节点嵌入（不含分类头），供 Fusion 使用

**其他 GNN**（`GNNs.py`）：`RGCN`, `SimpleHGN`, `GAT`，均通过 `build_gnn(gnn_type, config)` 工厂函数构建。

#### VariationalTextAdapter（VIB）

文件：`LLMbot/model.py:51-120`

```python
forward(x, num_samples=10) → (logits, z, alpha, uncertainty, kl_loss)
```

- Encoder：`Linear(4096, hidden) → LayerNorm → ReLU → Dropout`
- 重参数化：`z = mu + eps * exp(0.5 * logvar)`，logvar 被 clamp 到 `[-10, 0]`
- 分类器：`Linear(latent, 2)` → logits
- EDL 证据层：`softplus(Linear(latent, 2))` → evidence → `alpha = evidence + 1`
- 不确定性：`uncertainty = K / S`（K=类别数, S=sum(alpha)）
- KL 散度：标准 VAE KL 正则项

#### EvidentialGraphHead

文件：`LLMbot/model.py:13-49`

```python
forward(x, consistency=None) → (logits, alpha, uncertainty, probs)
```

- 投影 + ReLU → `softplus(evidence_layer) * softplus(evidence_scale)`
- 若提供 consistency 分数，用其校准证据：`evidence *= consistency`
- `alpha = evidence + 1`，`uncertainty = 2.0 / S`

#### ReliabilityAwareFusion

文件：`LLMbot/model.py:122-233`

```python
forward(lm_emb, gnn_emb, homophily, degree, consistency, **kwargs) → dict
```

前向传播流程：
1. **文本分支**：`self.vib(lm_emb)` → `logits_lm, alpha_text, u_text`
2. **图分支**：`self.gnn_proj(gnn_emb)` → `self.gnn_evidence_head(z, consistency)` → `logits_gnn, alpha_gnn, u_graph`
3. **Meta 特征**（`torch.no_grad`，5维）：`[u_text, u_graph, homophily, JSD, conf_gap]`
4. **Gate 网络**：`meta → LayerNorm → Linear(5,32) → ReLU → Linear(32,1) → Sigmoid` → `beta ∈ [0,1]`
5. **融合**：`probs_fused = beta * probs_lm + (1-beta) * probs_gnn`

返回字典包含：`logits`, `probs_fused`, `probs_lm`, `probs_gnn`, `logits_lm`, `logits_gnn`, `gate_value`, `u_text`, `u_graph`, `meta_features`, `alpha_text`, `alpha_gnn`, `kl_loss`

### 3.2 训练流水线（4 阶段）

入口：`LLMbot/main.py`，训练器：`LLMbot/train.py` 中的 `QwenPrecomputedTrainer`

每个阶段独立训练特定模块，通过检查点串联。阶段之间通过 `main.py` 中的 **Weight Surgery** 逻辑移植权重。

#### Stage 1：Text VIB 预训练（`pretrain_text_vib`，`train.py:792-968`）

- **冻结**：GNN, gnn_proj, gate_net, gnn_evidence_head
- **训练**：fusion.vib
- **损失**：
  - Warmup 阶段（前5 epoch）：CrossEntropy + LogitNorm + VIB_KL
  - EDL 阶段（5 epoch 后）：EDL_fit + KL_Dirichlet + MisleadingEvidence + AvUC + VIB_KL
- **检查点**：`best_text_vib.pt`（VIB state_dict）

#### Stage 2：GNN 预训练（`pretrain_gnn_stage`，`train.py:1054-1091`）

- **冻结**：fusion.vib, fusion.gate_net
- **训练**：gnn, fusion.gnn_proj, fusion.gnn_evidence_head
- **损失**：CrossEntropy + 0.2 * EdlLoss
- **检查点**：`best_gnn_seed{seed}.pt`（含 gnn_state_dict + classifier 权重）

#### Stage 3：Gate 预热（`pretrain_gate_stage`，`train.py:1092-1139`）

- **冻结**：gnn, fusion.vib, fusion.gnn_proj, fusion.gnn_evidence_head
- **训练**：fusion.gate_net
- **损失**：
  - Gate BCE 监督：Text 对/GNN 错 → gate=1，GNN 对/Text 错 → gate=0
  - 结构一致性损失：基于 homophily 和 u_text 的动态阈值约束
- **检查点**：`best_gate_seed{seed}.pt`

#### Stage 4：联合微调（`train`，`train.py:1141-1156`）

- **解冻**：所有模块，lr = lr * 0.5
- **损失**：CE + text_aux + graph_aux + EDL(text+graph) + VIB_KL + AvUC + gate_struct + margin
- **检查点**：`best_model_seed{seed}.pt` + `fusion_best_seed{seed}.pt`

### 3.3 损失函数清单

| 损失 | 文件位置 | 用途 |
|------|---------|------|
| `EdlLoss` | `utils.py:273-327` | Dirichlet NLL + KL 退火，强制错误预测高不确定性 |
| `AvUCLoss` | `utils.py:73-115` | 准确率-不确定性校准，确保自信预测是正确的 |
| `SupConLoss` | `utils.py:789-856` | 监督对比学习（温度系数放大困难样本梯度） |
| `kl_divergence_dirichlet` | `utils.py:117-131` | Dirichlet 分布间 KL 散度 |
| 结构一致性损失 | `train.py:316-354` | 基于 homophily 和不确定性的 Gate 正则化 |

### 3.4 数据加载与采样

**NeighborSampler**（`train.py:30-127`）：
- 纯 Python 实现的图邻居采样器，替代 `torch_geometric.loader.NeighborLoader`
- 支持 edge_type 保留（关键：RGT 需要 edge_type 信息）
- BFS 采样，默认 `sizes=[10, 10]`（2层，每层采10个邻居）

**MiniBatch**（`train.py:16-28`）：
- 自定义 batch 容器，包含 `n_id`, `edge_index`, `edge_type`, `batch_size`
- `n_id[:batch_size]` 是目标节点，其余是采样到的邻居

**load_raw_data**（`utils.py:589-717`）：
- 加载 TwiBot-20 数据集，返回 dict 包含 `train_idx`, `valid_idx`, `test_idx`, `labels`, `edge_index`, `edge_type`, `user_text`

---

## 4. 数据格式（TwiBot-20）

### 4.1 数据集文件

| 文件 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `train_idx.pt` | `[8278]` | long | 训练集节点索引，range [0, 8277] |
| `valid_idx.pt` | `[2365]` | long | 验证集节点索引，range [8278, 10642] |
| `test_idx.pt` | `[1183]` | long | 测试集节点索引，range [10643, 11825] |
| `labels.pt` | `[11826, 2]` | float | One-hot 标签（Human: 5237, Bot: 6589） |
| `edge_index.pt` | `[2, 16908]` | long | COO 格式边索引 |
| `edge_type.pt` | `[16908]` | long | 边类型（0=follower, 1=following） |
| `qwen3_emb_last.pt` | `[11826, 4096]` | float | Qwen3-Embedding-8B 预计算嵌入 |
| `norm_user_text.json` | dict | JSON | 用户文本内容（profile + tweets） |

### 4.2 训练中间张量

| 张量 | Shape | 说明 |
|------|-------|------|
| `homophily` | `[N, 1]` | 节点与邻居的特征余弦相似度均值 |
| `node_degrees` | `[N, 1]` | 归一化入度 |
| `consistency` | `[B, 1]` | Batch 内节点的结构一致性（边相似度聚合） |
| `meta_features` | `[B, 5]` | `[u_text, u_graph, homophily, JSD, conf_gap]` |

---

## 5. 运行命令

### 5.1 Embedding 生成（只需运行一次）

```bash
cd LLMbot
python generate_qwen3_embeddings.py \
    --dataset_path ./datasets/TwiBot-20 \
    --model_path Qwen/Qwen3-Embedding-8B \
    --device cuda:0 \
    --batch_size 32 \
    --max_length 512
```

输出：`./datasets/TwiBot-20/qwen3_emb_last.pt`

### 5.2 完整四阶段训练

```bash
cd LLMbot
python main.py \
    --pretrain_llm \
    --pretrain_gnn \
    --pretrain_gate \
    --fusion \
    --gnn_type RGT \
    --dataset_path ./datasets/TwiBot-20 \
    --embeddings_path ./datasets/TwiBot-20/qwen3_emb_last.pt \
    --hidden_dim 512 \
    --heads 4 \
    --pretrain_epochs 15 \
    --epochs 100 \
    --seed 42
```

### 5.3 单阶段训练

```bash
python main.py --pretrain_llm                    # Stage 1: VIB 预训练
python main.py --pretrain_gnn                    # Stage 2: GNN 预训练
python main.py --pretrain_gate                   # Stage 3: Gate 预热
python main.py --fusion                          # Stage 4: 联合微调
```

注意：Stage 3/4 会自动加载 Stage 1/2 的检查点（Weight Surgery）。

### 5.4 错误诊断与消融实验

```bash
python main.py --error_capture                   # 生成错误诊断 CSV
python main.py --ablation                        # 消融实验（Text-only / Graph-only / Full）
```

---

## 6. 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gnn_type` | `RGT` | GNN 类型：RGT / RGCN / SimpleHGN |
| `--hidden_dim` | `512` | GNN 隐藏维度 |
| `--heads` | `4` | TransformerConv 注意力头数 |
| `--n_layers` | `2` | GNN 层数 |
| `--dropout` | `0.3` | Dropout 率 |
| `--fusion_hidden_dim` | `512` | Fusion 隐藏维度 |
| `--lr` | `1e-3` | 学习率 |
| `--weight_decay` | `5e-5` | 权重衰减 |
| `--pretrain_epochs` | `15` | 预训练阶段 epoch 数 |
| `--epochs` | `100` | 联合微调 epoch 数 |
| `--batch_size` | `1024` | 批大小 |
| `--data_loader` | `neighbor` | 数据加载方式：neighbor / random |
| `--lambda_avuc` | `0.1` | AvUC 校准损失权重 |
| `--lambda_struct` | `0.1` | 结构一致性损失权重 |

---

## 7. 检查点约定

### 7.1 目录结构

```
saved_models/{ckpt_dir}/
├── best_text_vib.pt              # Stage 1: VIB state_dict
├── best_gnn_seed{seed}.pt        # Stage 2: {gnn_state_dict, epoch}
├── best_gate_seed{seed}.pt       # Stage 3: {gnn_state_dict, fusion_state_dict, epoch}
├── best_model_seed{seed}.pt      # Stage 4: {gnn_state_dict, fusion_state_dict, epoch}
└── fusion_best_seed{seed}.pt     # Stage 4: 融合专用检查点
```

### 7.2 Weight Surgery（权重移植）

`main.py:286-338` 中，Stage 3/4 启动前会执行权重移植：

1. **Text Expert 移植**：加载 `best_text_vib.pt` → `fusion_model.vib`
   - 使用 `load_weights()` 智能匹配 key（自动处理 `vib.` 前缀）

2. **GNN Expert 移植**：加载 `best_gnn_seed{seed}.pt` → 拆分为：
   - Backbone 权重 → `gnn_model`（过滤掉 classifier/head 相关 key）
   - `classifier.0.*` → `fusion_model.gnn_proj`（投影层）
   - `classifier.3.*` / `evidence_layer.*` → `fusion_model.gnn_evidence_head`（证据头）
   - `evidence_scale` → `fusion_model.gnn_evidence_head.evidence_scale`

---

## 8. 已知问题

### 代码层面

1. **死代码 bug**（`train.py`，不影响主训练路径）：
   - `_forward_gnn(self, batch_nodes)` 缺少默认值，被 `_train_epoch_gnn_full_batch` 无参调用
   - `_train_epoch_gnn_full_batch` 中 `self.edl_loss(alpha, labels_train)` 缺少 `epoch_num` 参数
   - `_compute_fusion_losses` 中 `self.homophily[labels.device.index]` 索引逻辑错误

2. **edge_index_list 重复计算**（`GNNs.py:208-219`）：`prepare_data_for_RGT` 每次 forward 都做 mask slicing，TwiBot-20 规模下影响可忽略。

### 论文层面

3. **BotUMC 对齐**：当前实现是 uncertainty-aware late fusion（双专家 + 不确定性 gating），未实现 BotUMC 的 multi-view causal inference 框架。引用时只能说"借鉴其 uncertainty motivation"。

详细分析见 `docs/CODE_REVIEW_AND_FIX_PLAN.md`。
