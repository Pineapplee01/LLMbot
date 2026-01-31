# QwenPrecomputedTrainer 工程重构指南

> **文档目标**: 深度工程审计、隐患分析、模块化重构方案
> 
> **生成时间**: 2026-01-31  
> **最后更新**: 2026-01-31  
> **当前状态**: ✅ 核心功能完整，采样方法已实现，待模块化重构
> **审计级别**: 🔴 Principal Engineer 级别严格审计

---

## 📋 目录

1. [现状分析](#1-现状分析)
2. [深度工程审计](#2-深度工程审计)
3. [架构梳理](#3-架构梳理)
4. [问题诊断](#4-问题诊断)
5. [模块化重构方案](#5-模块化重构方案)
6. [重构后目录结构](#6-重构后目录结构)
7. [实施路线](#7-实施路线)
8. [当前实现状态](#8-当前实现状态)

---

## 1. 现状分析

### 1.1 文件依赖关系图

```
main_qwen3_precomputed.py (入口)
    ├─> QwenPrecomputedTrainer.py (核心训练器, 1045行)
    │   ├─> utils.py (工具函数)
    │   │   └─> EdlLoss, seed_setting, load_raw_data
    │   ├─> GNNs.py (图神经网络)
    │   │   ├─> RGCN, RGT, SimpleHGN, HGT
    │   │   └─> RGT.py (RGT层实现)
    │   └─> AttentionFusion.py (融合模块)
    │       ├─> ReliabilityAwareFusion (主融合类)
    │       ├─> VariationalTextAdapter (VIB文本专家)
    │       └─> EvidentialGraphHead (图专家)
    └─> 外部依赖:
        ├─> torch, torch_geometric
        ├─> wandb (实验跟踪)
        ├─> sklearn (评估指标)
        └─> pandas, numpy (数据处理)
```

### 1.2 Trainer 当前职责 (违反单一职责原则)

| 职责类别 | 代码行数占比 | 问题严重度 | 当前状态 |
|---------|------------|-----------|---------|
| **数据加载与采样** | ~15% | 🟡 中等 | ✅ 已实现 |
| **模型前向传播** | ~20% | 🟢 轻微 | ✅ 正常 |
| **训练循环逻辑** | ~25% | � 中等 | ✅ 可用 |
| **损失函数计算** | ~10% | 🟡 中等 | ✅ 完整 |
| **评估与诊断** | ~15% | � 中等 | ✅ 可用 |
| **检查点管理** | ~5% | 🟢 轻微 | ✅ 正常 |
| **阶段控制(预训练/融合)** | ~10% | � 中等 | ✅ 可用 |

**总代码量**: 1045行 → **功能完整，但仍需模块化重构**

---

## 2. 深度工程审计

> 🔴 **审计标准**: Principal Software Engineer + 安全专家级别

### 2.1 逻辑缺陷 (Logic Bugs)

| 严重度 | 位置 | 问题描述 | 风险 |
|--------|------|----------|------|
| 🔴 **Critical** | `_forward_gnn()` L478 | 函数签名 `_forward_gnn(self, batch_nodes)` 但 `_train_epoch_gnn_full_batch()` 调用 `_forward_gnn()` 无参数 | **运行时崩溃** |
| 🔴 **Critical** | `main.py` L388 | `args.ckpt_dir` 未定义，ablation 分支会崩溃 | **AttributeError** |
| 🟠 **High** | `NeighborSampler` L87-118 | BFS 采样时 `current_layer_nodes` 可能包含不存在的节点索引 | 潜在越界 |
| 🟠 **High** | `AttentionFusion.py` L205 | `meta_bn` BatchNorm 在 `batch_size=1` 时会报错 | 推理失败 |
| 🟡 **Medium** | `_compute_fusion_losses()` L528 | `self.homophily[labels.device.index]` 使用 device.index 作为索引，应该用节点索引 | 逻辑错误 |
| 🟡 **Medium** | `utils.py` L310-395 | `relation_aware_knn_pruning` 循环效率低 O(N*E) | 大图性能问题 |

### 2.2 安全隐患 (Security Risks)

| 严重度 | 位置 | 问题描述 | 建议 |
|--------|------|----------|------|
| 🟠 **High** | `utils.py` L442 | `torch.load()` 使用 `weights_only=False`，存在反序列化漏洞 | 使用 `weights_only=True` 或 `safetensors` |
| 🟠 **High** | `main.py` L192 | `torch.load(..., weights_only=False)` 加载嵌入 | 同上 |
| 🟡 **Medium** | 全局 | 无输入验证，恶意数据可导致 OOM | 添加数据形状/范围检查 |
| 🟡 **Medium** | `load_raw_data()` | JSON 文件无大小限制加载 | 添加文件大小检查 |

### 2.3 性能瓶颈 (Performance Issues)

| 严重度 | 位置 | 问题描述 | 影响 |
|--------|------|----------|------|
| 🔴 **Critical** | `_unpack_batch()` L252-258 | 每个 batch 都执行全图 GNN 前向 (`h_all = self._call_gnn(...)`) | **10x+ 性能损失** |
| 🟠 **High** | `QwenPrecomputedTrainer` | `self.embeddings` 存 CPU，每次 `.to(device)` 传输 | GPU 带宽瓶颈 |
| 🟠 **High** | `_compute_batch_consistency()` | 每 batch 重复计算，无缓存 | 冗余计算 |
| 🟡 **Medium** | `NeighborSampler` | Python 循环 BFS，无向量化 | 采样速度慢 |
| 🟡 **Medium** | `diagnose_errors()` | 600+ 行诊断方法，每次重新计算所有特征 | 诊断效率低 |

### 2.4 代码异味 (Code Smells)

| 类型 | 位置 | 描述 | 建议 |
|------|------|------|------|
| **God Class** | `QwenPrecomputedTrainer` | 1045 行，35+ 方法，7+ 职责 | 拆分为多个类 |
| **参数爆炸** | `__init__` | 17 个参数 | 使用配置对象 |
| **魔法数字** | 多处 | `0.30`, `0.65`, `0.2`, `0.5`, `0.1` 硬编码 | 提取为常量/配置 |
| **重复代码** | `_train_epoch_*` | 采样分支逻辑重复 5+ 次 | 抽取为统一方法 |
| **隐式依赖** | 全局 | `wandb.run is not None` 检查散布各处 | 封装为 Logger 类 |
| **命名不一致** | `utils.py` | 函数名混用 `snake_case` 和 `camelCase` | 统一风格 |
| **缺乏类型注解** | 全部文件 | 无 type hints | 添加类型注解 |
| **文档缺失** | 多数方法 | 无 docstring 或过时 | 补充文档 |

### 2.5 错误处理缺陷 (Error Handling)

| 位置 | 问题 | 建议 |
|------|------|------|
| `load_raw_data()` | 文件不存在直接崩溃 | 添加 `FileNotFoundError` 处理 |
| `load_checkpoint()` | 通用 `Exception` 捕获 | 细化异常类型 |
| `_create_dataloader()` | 无效 `dataloader` 参数无检查 | 添加参数验证 |
| `build_gnn()` | 未知 GNN 类型抛出 `ValueError` 但无上下文 | 提供可用选项列表 |

---

## 3. 架构梳理

### 3.1 文件依赖关系图

#### 2.1.1 初始化逻辑 (__init__)
```python
输入:
- precomputed_embeddings: 预训练LM嵌入 [N, 4096]
- gnn_model: 图神经网络 (RGCN/RGT/SimpleHGN)
- fusion_model: ReliabilityAwareFusion
- data_dict: 包含 train/val/test 索引, labels, edge_index, edge_type
- dataloader: 'neighbor' 或 'full' (采样策略)

关键状态:
- self.embeddings: CPU张量 (内存优化)
- self.labels: GPU张量
- self.edge_index, self.edge_type: 图结构
- self.homophily, self.node_degrees: 预计算特征
- self.dataloader, self.val_loader, self.test_loader: 数据加载器

问题:
❌ 混合了模型构建、数据加载、特征工程
❌ 缺少配置对象,参数传递混乱 (17个参数)
```

#### 2.1.2 训练流程

```
阶段 1: pretrain_text_vib()  
    └─> 只训练 VIB 文本专家
    └─> 冻结: GNN, Gate, Classifier
    └─> 优化目标: 文本分类 + EDL + KL散度

阶段 2: pretrain_gnn_stage()
    └─> 训练 GNN + Evidence Head
    └─> 冻结: Fusion (除 gnn_proj)
    └─> 优化目标: GNN分类 + EDL

阶段 3: pretrain_gate_stage() 
    └─> 训练 Gate + Classifier
    └─> 冻结: GNN, VIB
    └─> 优化目标: 融合分类

阶段 4: train() [全量微调]
    └─> 联合训练所有模块
    └─> 完整损失函数 (9个损失项)
```

**问题**: 
- ❌ 阶段切换逻辑硬编码在各方法中
- ❌ 冻结/解冻逻辑分散,难以追踪
- ❌ 缺少阶段编排器 (Stage Scheduler)

#### 2.1.3 数据流

```
训练批次:
1. _create_dataloader(indices) 
   ├─> 'neighbor': NeighborSampler → MiniBatch(n_id, edge_index, edge_type, batch_size)
   └─> 'full': TensorDataset → (nodes, labels)

2. _unpack_batch(batch)
   ├─> MiniBatch: 动态加载 embeddings[n_id] (CPU→GPU)
   │   └─> 返回 (x, y, edge_index, edge_type, batch_size, n_id)
   └─> Tuple: 全图前向 → 提取节点特征
       └─> 返回 (h_all, y, edge_index, edge_type, batch_size, nodes)

3. 训练循环:
   for batch in dataloader:
       x, labels, edge_index, edge_type, batch_size, n_id = _unpack_batch(batch)
       
       if 邻居采样 (hasattr(batch, 'n_id')):
           h_sub = _call_gnn(x, edge_index, edge_type)  # 子图前向
           h_gnn_target = h_sub[:batch_size]
       else:
           h_gnn_target = x[n_id]  # 已经是全图GNN输出

4. 损失计算 → 反向传播 → 优化器更新
```

**✅ 已修复的问题**:
- ✅ `_unpack_batch` 统一返回 6 元组 (包含 `edge_type`)
- ✅ `_call_gnn` 支持可选 `edge_type` 参数
- ✅ `_compute_batch_consistency` 使用子图边索引

**⚠️ 待优化**:
- ⚠️ 邻居采样与全图模式的分支逻辑仍在多处重复
- ⚠️ 动态 CPU→GPU 传输可考虑预缓存优化

#### 2.1.4 评估与诊断

```python
evaluate(split='val')
    └─> 返回 F1, Accuracy

diagnose_errors(split='val')
    └─> 生成诊断 DataFrame
    └─> 列: node_idx, label, case_type, pred_full, gate, u_text, u_graph, ...
    └─> 分组统计: Easy/Hard/Text_Only/Graph_Only
```

**问题**:
- ❌ `diagnose_errors` 过于庞大 (100+行)
- ❌ 诊断逻辑应该独立为 `DiagnosticAnalyzer` 类
- ❌ 与训练逻辑耦合 (依赖 self.fusion, self.gnn 状态)

---

## 4. 问题诊断

### 4.1 代码异味 (Code Smells)

| 问题类型 | 具体表现 | 影响 | 优先级 |
|---------|---------|------|-------|
| **上帝类** | Trainer 739行, 30+方法 | 难以维护、测试、扩展 | 🔴 P0 |
| **重复代码** | 邻居采样分支重复5+ | 增加bug风险 | 🟠 P1 |
| **参数过多** | `__init__` 17个参数 | 难以理解、易出错 | 🟠 P1 |
| **深度嵌套** | 训练循环嵌套4层+ | 逻辑难以追踪 | 🟡 P2 |
| **魔法数字** | 0.5, 0.30, 0.65硬编码 | 缺乏语义、难调试 | 🟡 P2 |
| **状态混乱** | pretrain_gnn标志位滥用 | 副作用不可预测 | 🟠 P1 |

### 3.2 架构缺陷

#### 3.2.1 缺少抽象层

```
❌ 当前: Trainer 直接操作 dataloader, gnn, fusion
✅ 应该: 引入 DataModule, ModelWrapper, TrainerEngine
```

#### 3.2.2 配置管理混乱

```python
❌ 当前:
trainer = QwenPrecomputedTrainer(
    precomputed_embeddings, gnn, fusion, data_dict,
    dataloader='neighbor', batch_size=64, device='cuda',
    epochs=100, lr=1e-3, weight_decay=5e-5,
    lambda_avuc=0.1, lambda_struct=0.1, ...  # 17个参数!
)

✅ 应该:
config = TrainerConfig(
    data=DataConfig(...),
    model=ModelConfig(...),
    training=TrainingConfig(...)
)
trainer = QwenPrecomputedTrainer(config)
```

#### 3.2.3 测试困难

```
❌ 当前: Trainer 与 wandb, GPU, 文件系统紧耦合
✅ 应该: 依赖注入 + 接口抽象
```

---

## 5. 模块化重构方案

### 5.1 重构目标

| 目标 | 当前 | 重构后 | 衡量标准 |
|------|------|--------|---------|
| **单一职责** | 1 类 35 方法 | 8+ 类各司其职 | 每类 <200 行 |
| **可测试性** | 0% 覆盖 | 80%+ 覆盖 | pytest --cov |
| **可扩展性** | 修改核心类 | 插件式扩展 | 新功能 <50 行 |
| **类型安全** | 无 type hints | 完整类型注解 | mypy 通过 |
| **配置管理** | 17 参数 | 1 配置对象 | dataclass |

### 5.2 模块拆分策略

#### 阶段 1: 数据层分离 (Phase 1)

**目标**: 解耦数据加载与训练逻辑

```python
# 新建: data/data_module.py
class GraphDataModule:
    """统一管理图数据加载"""
    def __init__(self, data_dict, embeddings, config):
        self.data_dict = data_dict
        self.embeddings = embeddings
        self.config = config
        
    def setup(self):
        """预处理: 计算 homophily, degrees, 构建 loaders"""
        pass
        
    def train_dataloader(self):
        return self._create_loader(self.data_dict['train_idx'], shuffle=True)
        
    def val_dataloader(self):
        return self._create_loader(self.data_dict['valid_idx'], shuffle=False)
        
    def _create_loader(self, indices, shuffle):
        if self.config.sampling == 'neighbor':
            return NeighborSampler(...)
        else:
            return TensorDataset(...)

# 新建: data/batch_processor.py
class BatchProcessor:
    """统一批次解包与特征提取"""
    def __init__(self, embeddings, device):
        self.embeddings = embeddings
        self.device = device
        
    def process(self, batch, mode='neighbor'):
        """返回标准化的 (x, labels, edge_index, batch_size, n_id)"""
        if mode == 'neighbor':
            return self._process_neighbor_batch(batch)
        else:
            return self._process_full_batch(batch)
```

**收益**:
- ✅ Trainer 减少 ~100 行
- ✅ 数据逻辑可独立测试
- ✅ 支持新的采样策略 (不修改 Trainer)

---

#### 阶段 2: 训练流程解耦 (Phase 2)

**目标**: 分离训练循环与模型逻辑

```python
# 新建: training/trainer_engine.py
class TrainerEngine:
    """通用训练引擎 (类似 PyTorch Lightning)"""
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def training_step(self, batch, batch_idx):
        """单步训练 (子类实现)"""
        raise NotImplementedError
        
    def validation_step(self, batch, batch_idx):
        """单步验证"""
        raise NotImplementedError
        
    def fit(self, train_loader, val_loader, epochs):
        """训练循环主逻辑"""
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, epoch)
            val_metrics = self._validate_epoch(val_loader)
            self._log_metrics(epoch, train_loss, val_metrics)

# 新建: training/fusion_trainer.py
class FusionTrainer(TrainerEngine):
    """继承通用引擎, 实现融合训练逻辑"""
    def __init__(self, gnn, fusion, config, ...):
        self.gnn = gnn
        self.fusion = fusion
        self.config = config
        super().__init__(...)
        
    def training_step(self, batch, batch_idx):
        x, labels, edge_index, ... = self.batch_processor.process(batch)
        
        h_gnn = self.gnn(x, edge_index)
        out = self.fusion(lm_emb, h_gnn, ...)
        
        loss = self._compute_loss(out, labels)
        return loss
        
    def _compute_loss(self, out, labels):
        """调用 LossComputer"""
        return self.loss_computer.compute_fusion_loss(out, labels, self.config)
```

**收益**:
- ✅ 训练逻辑可复用 (GNN预训练、融合训练共享基类)
- ✅ 清晰的职责边界
- ✅ 易于添加新训练模式 (对比学习、蒸馏等)

---

#### 阶段 3: 损失函数模块化 (Phase 3)

**目标**: 解耦损失计算逻辑

```python
# 新建: training/loss_computer.py
class LossComputer:
    """统一管理所有损失项"""
    def __init__(self, config):
        self.config = config
        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.edl_loss = EdlLoss()
        
    def compute_fusion_loss(self, out, labels, epoch):
        """计算融合训练的完整损失"""
        losses = {}
        
        # 基础分类损失
        losses['cls'] = self.criterion_cls(out['logits'], labels)
        
        # EDL 损失
        losses['edl_text'] = self.edl_loss(out['alpha_text'], labels, epoch)
        losses['edl_graph'] = self.edl_loss(out['alpha_gnn'], labels, epoch)
        
        # VIB KL 损失
        losses['vib'] = out['kl_loss'] * self._get_kl_weight(epoch)
        
        # AvUC 损失
        losses['avuc'] = self._compute_avuc(out, labels) * self.config.lambda_avuc
        
        # 边界损失
        losses['margin'] = self._compute_margin(out, labels) * self.config.margin_weight
        
        # 总损失
        total_loss = self._weighted_sum(losses)
        return total_loss, losses
        
    def _get_kl_weight(self, epoch):
        """KL 权重 warmup"""
        if epoch < 5:
            return 0.0
        return self.config.lambda_struct * min(1.0, (epoch - 5) / 5)
```

**收益**:
- ✅ 损失计算逻辑集中管理
- ✅ 支持动态调整权重策略
- ✅ 易于添加新损失项

---

#### 阶段 4: 阶段编排器 (Phase 4)

**目标**: 管理预训练阶段切换

```python
# 新建: training/stage_scheduler.py
class TrainingStageScheduler:
    """管理多阶段训练流程"""
    def __init__(self, stages, models):
        self.stages = stages  # [Stage1Config, Stage2Config, ...]
        self.models = models  # {'gnn': ..., 'fusion': ...}
        
    def run(self):
        for stage_config in self.stages:
            print(f"\n{'='*60}")
            print(f"Stage {stage_config.id} | {stage_config.name}")
            print(f"{'='*60}")
            
            # 1. 冻结/解冻模块
            self._configure_trainable_params(stage_config.trainable_modules)
            
            # 2. 构建训练器
            trainer = self._build_trainer(stage_config)
            
            # 3. 训练
            trainer.fit(epochs=stage_config.epochs)
            
            # 4. 保存检查点
            self._save_checkpoint(stage_config.ckpt_name)

# 使用示例:
scheduler = TrainingStageScheduler(
    stages=[
        StageConfig(id=1, name='Text VIB', trainable_modules=['fusion.vib'], epochs=15),
        StageConfig(id=2, name='GNN', trainable_modules=['gnn', 'fusion.gnn_proj'], epochs=15),
        StageConfig(id=3, name='Gate', trainable_modules=['fusion.gate_net'], epochs=10),
        StageConfig(id=4, name='Joint', trainable_modules=['all'], epochs=50)
    ],
    models={'gnn': gnn_model, 'fusion': fusion_model}
)
scheduler.run()
```

**收益**:
- ✅ 阶段配置声明式
- ✅ 易于调整训练流程 (删除/添加阶段)
- ✅ 冻结逻辑集中管理

---

#### 阶段 5: 评估与诊断分离 (Phase 5)

**目标**: 独立诊断分析器

```python
# 新建: evaluation/evaluator.py
class ModelEvaluator:
    """标准评估指标计算"""
    def __init__(self, model, data_module, device):
        self.model = model
        self.data_module = data_module
        self.device = device
        
    def evaluate(self, split='val'):
        """计算 F1, Acc"""
        loader = self.data_module.get_loader(split)
        preds, targets = self._collect_predictions(loader)
        return {
            'f1': f1_score(targets, preds, average='macro'),
            'acc': accuracy_score(targets, preds)
        }

# 新建: evaluation/diagnostic_analyzer.py
class DiagnosticAnalyzer:
    """深度诊断分析 (Bad Case Study)"""
    def __init__(self, model, data_module, config):
        self.model = model
        self.data_module = data_module
        self.config = config
        
    def analyze(self, split='val'):
        """生成诊断 DataFrame"""
        results = []
        loader = self.data_module.get_loader(split)
        
        for batch in loader:
            batch_results = self._analyze_batch(batch)
            results.extend(batch_results)
            
        df = pd.DataFrame(results)
        return self._generate_summary(df)
        
    def _analyze_batch(self, batch):
        """单批次诊断"""
        # 提取: pred_text, pred_graph, pred_full
        # 计算: gate, u_text, u_graph, homophily
        # 分类: Easy/Hard/Text_Only/Graph_Only
        pass
        
    def _generate_summary(self, df):
        """生成统计报告"""
        summary = df.groupby('case_type').agg({
            'node_idx': 'count',
            'gate': 'mean',
            'u_text': 'mean',
            'u_graph': 'mean',
            'is_correct_full': 'mean'
        })
        print(summary)
        return df
```

**收益**:
- ✅ 诊断逻辑可独立运行 (不依赖训练状态)
- ✅ 支持离线分析 (加载检查点后诊断)
- ✅ 易于扩展新诊断指标

---

### 5.6 最终架构

```
project/
├── configs/
│   ├── base_config.py          # 配置基类
│   └── training_config.py      # 训练配置 (超参数)
│
├── data/
│   ├── data_module.py          # 数据模块
│   ├── batch_processor.py      # 批次处理器
│   └── samplers.py             # 邻居采样器
│
├── models/
│   ├── gnn.py                  # GNN 模型 (RGCN, RGT, ...)
│   ├── fusion.py               # 融合模块
│   └── components/             # VIB, Evidence Head, Gate
│
├── training/
│   ├── trainer_engine.py       # 通用训练引擎
│   ├── fusion_trainer.py       # 融合训练器
│   ├── stage_scheduler.py      # 阶段编排器
│   └── loss_computer.py        # 损失计算器
│
├── evaluation/
│   ├── evaluator.py            # 标准评估
│   └── diagnostic_analyzer.py  # 诊断分析
│
├── utils/
│   ├── checkpoint.py           # 检查点管理
│   ├── logger.py               # 日志与 wandb 封装
│   └── metrics.py              # 自定义指标
│
└── main.py                     # 入口脚本
```

---

## 6. 重构后目录结构

### 6.1 完整目录树

```
LLMbot/                              # 项目根目录
├── configs/                         # 📁 配置层
│   ├── __init__.py
│   ├── base.py                      # 配置基类与常量
│   ├── model_config.py              # 模型超参数配置
│   ├── training_config.py           # 训练配置
│   └── experiment_config.py         # 实验配置 (wandb, seed, paths)
│
├── data/                            # 📁 数据层
│   ├── __init__.py
│   ├── datasets/                    # 数据集目录
│   │   └── TwiBot-20/
│   ├── data_module.py               # 统一数据加载接口
│   ├── samplers.py                  # NeighborSampler, FullBatchSampler
│   ├── batch_processor.py           # 批次解包与预处理
│   └── transforms.py                # 数据变换 (pruning, normalization)
│
├── models/                          # 📁 模型层
│   ├── __init__.py
│   ├── gnns/                        # GNN 模型
│   │   ├── __init__.py
│   │   ├── base.py                  # GNN 抽象基类
│   │   ├── rgcn.py                  # RGCN 实现
│   │   ├── rgt.py                   # RGT 实现
│   │   └── simple_hgn.py            # SimpleHGN 实现
│   ├── fusion/                      # 融合模块
│   │   ├── __init__.py
│   │   ├── attention_fusion.py      # ReliabilityAwareFusion
│   │   ├── vib_adapter.py           # VariationalTextAdapter
│   │   └── evidence_head.py         # EvidentialGraphHead
│   └── builder.py                   # 模型工厂 (build_gnn, build_fusion)
│
├── training/                        # 📁 训练层
│   ├── __init__.py
│   ├── base_trainer.py              # 抽象训练器基类
│   ├── pretrainers/                 # 预训练器
│   │   ├── __init__.py
│   │   ├── vib_pretrainer.py        # VIB 文本专家预训练
│   │   └── gnn_pretrainer.py        # GNN 预训练
│   ├── fusion_trainer.py            # 融合训练器
│   ├── loss_computer.py             # 损失计算模块
│   ├── stage_scheduler.py           # 多阶段训练编排
│   └── optimizers.py                # 优化器工厂
│
├── evaluation/                      # 📁 评估层
│   ├── __init__.py
│   ├── evaluator.py                 # 标准评估器 (F1, Acc)
│   ├── diagnostic_analyzer.py       # Bad Case 诊断分析
│   └── visualizer.py                # 可视化工具
│
├── utils/                           # 📁 工具层
│   ├── __init__.py
│   ├── checkpoint.py                # 检查点管理
│   ├── logger.py                    # 日志封装 (wandb wrapper)
│   ├── metrics.py                   # 自定义指标 (AvUC, EdlLoss)
│   ├── seed.py                      # 随机种子设置
│   └── io.py                        # 安全的文件 I/O
│
├── scripts/                         # 📁 脚本层
│   ├── train.py                     # 训练入口
│   ├── evaluate.py                  # 评估入口
│   ├── diagnose.py                  # 诊断入口
│   └── generate_embeddings.py       # 嵌入生成
│
├── tests/                           # 📁 测试层
│   ├── __init__.py
│   ├── test_data_module.py
│   ├── test_loss_computer.py
│   ├── test_models.py
│   └── conftest.py                  # pytest fixtures
│
├── saved_models/                    # 📁 检查点存储
├── logs/                            # 📁 日志存储
├── requirements.txt                 # 依赖
├── pyproject.toml                   # 项目配置
└── README.md                        # 项目文档
```

### 6.2 模块职责详解

#### 6.2.1 配置层 (`configs/`)

| 文件 | 职责 | 主要类/函数 |
|------|------|------------|
| `base.py` | 配置基类、常量定义 | `BaseConfig`, `MAGIC_NUMBERS` |
| `model_config.py` | GNN/Fusion 超参数 | `GNNConfig`, `FusionConfig` |
| `training_config.py` | 训练超参数 | `TrainingConfig`, `LossConfig` |
| `experiment_config.py` | 实验配置 | `ExperimentConfig`, `WandbConfig` |

```python
# 使用示例
from configs import TrainingConfig, GNNConfig

config = TrainingConfig(
    lr=1e-3,
    epochs=100,
    loss=LossConfig(lambda_avuc=0.1, u_low=0.30, u_high=0.65)
)
```

#### 6.2.2 数据层 (`data/`)

| 文件 | 职责 | 主要类/函数 |
|------|------|------------|
| `data_module.py` | 统一数据加载 | `GraphDataModule` |
| `samplers.py` | 采样器实现 | `NeighborSampler`, `MiniBatch` |
| `batch_processor.py` | 批次处理 | `BatchProcessor.process()` |
| `transforms.py` | 数据变换 | `KNNPruning`, `FeatureNormalizer` |

```python
# 使用示例
from data import GraphDataModule

dm = GraphDataModule(data_path='./datasets/TwiBot-20', config=config)
dm.setup()
train_loader = dm.train_dataloader()
```

#### 6.2.3 模型层 (`models/`)

| 文件 | 职责 | 主要类/函数 |
|------|------|------------|
| `gnns/base.py` | GNN 抽象基类 | `BaseGNN` |
| `gnns/rgt.py` | RGT 实现 | `RGT`, `RGTLayer` |
| `fusion/attention_fusion.py` | 融合模块 | `ReliabilityAwareFusion` |
| `builder.py` | 模型工厂 | `build_gnn()`, `build_fusion()` |

```python
# 使用示例
from models import build_gnn, build_fusion

gnn = build_gnn('RGT', config.gnn)
fusion = build_fusion(config.fusion, lm_dim=4096, gnn_dim=512)
```

#### 6.2.4 训练层 (`training/`)

| 文件 | 职责 | 主要类/函数 |
|------|------|------------|
| `base_trainer.py` | 训练器基类 | `BaseTrainer` |
| `fusion_trainer.py` | 融合训练 | `FusionTrainer` |
| `loss_computer.py` | 损失计算 | `LossComputer` |
| `stage_scheduler.py` | 阶段编排 | `StageScheduler` |

```python
# 使用示例
from training import FusionTrainer, StageScheduler

trainer = FusionTrainer(gnn, fusion, config)
scheduler = StageScheduler([
    Stage('vib', epochs=15, trainable=['fusion.vib']),
    Stage('gnn', epochs=15, trainable=['gnn']),
    Stage('fusion', epochs=100, trainable=['all'])
])
scheduler.run(trainer)
```

#### 6.2.5 评估层 (`evaluation/`)

| 文件 | 职责 | 主要类/函数 |
|------|------|------------|
| `evaluator.py` | 标准评估 | `Evaluator.evaluate()` |
| `diagnostic_analyzer.py` | 诊断分析 | `DiagnosticAnalyzer.analyze()` |
| `visualizer.py` | 可视化 | `plot_gate_vs_uncertainty()` |

```python
# 使用示例
from evaluation import Evaluator, DiagnosticAnalyzer

evaluator = Evaluator(models, data_module)
metrics = evaluator.evaluate('test')  # {'f1': 0.85, 'acc': 0.87}

analyzer = DiagnosticAnalyzer(models, data_module)
df = analyzer.analyze('val')  # DataFrame with case_type, gate, u_text, etc.
```

#### 6.2.6 工具层 (`utils/`)

| 文件 | 职责 | 主要类/函数 |
|------|------|------------|
| `checkpoint.py` | 检查点管理 | `CheckpointManager` |
| `logger.py` | 日志封装 | `Logger`, `WandbLogger` |
| `metrics.py` | 自定义指标 | `EdlLoss`, `AvUCLoss` |
| `seed.py` | 随机种子 | `set_seed()` |
| `io.py` | 安全 I/O | `safe_load()`, `safe_save()` |

```python
# 使用示例
from utils import set_seed, CheckpointManager, Logger

set_seed(42)
ckpt_manager = CheckpointManager('./saved_models/exp1')
logger = Logger(use_wandb=True, project='SeGA')
```

### 6.3 模块依赖图

```
┌─────────────┐
│   scripts/  │  ← 入口层 (用户交互)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  training/  │  ← 编排层 (训练流程控制)
└──────┬──────┘
       │
   ┌───┴───┐
   ▼       ▼
┌──────┐ ┌────────────┐
│models│ │ evaluation │  ← 核心层 (模型 & 评估)
└──┬───┘ └─────┬──────┘
   │           │
   └─────┬─────┘
         ▼
    ┌─────────┐
    │  data/  │  ← 数据层 (数据加载 & 处理)
    └────┬────┘
         │
         ▼
  ┌────────────┐
  │  configs/  │  ← 配置层 (超参数 & 常量)
  └────────────┘
         │
         ▼
    ┌─────────┐
    │  utils/ │  ← 基础层 (通用工具)
    └─────────┘
```

**依赖规则**:
- ✅ 上层可以依赖下层
- ❌ 下层不能依赖上层
- ❌ 同层之间尽量避免依赖

---

## 7. 实施路线

### 7.1 迁移路径 (零风险重构)

**策略**: 渐进式重构,保持向后兼容

#### Step 1: 创建新模块 (不破坏现有代码)
```bash
# Week 1: 数据层
mkdir -p data/
touch data/data_module.py
touch data/batch_processor.py

# 实现新接口,不修改 Trainer
```

### 7.2 并行测试
```python
# 在 Trainer 中添加开关
class QwenPrecomputedTrainer:
    def __init__(self, ..., use_new_data_module=False):
        if use_new_data_module:
            self.data_module = GraphDataModule(...)
        else:
            # 保留旧逻辑
            self.dataloader = self._create_dataloader(...)
```

### 7.3 逐步迁移
```python
# Week 2: 损失函数
self.loss_computer = LossComputer(config)
# 替换 _compute_fusion_losses → loss_computer.compute_fusion_loss

# Week 3: 训练引擎
self.trainer_engine = FusionTrainer(...)
# 替换 _train_epoch_fusion → trainer_engine.training_step
```

### 7.4 清理旧代码
```python
# Week 4: 删除废弃方法
# 确认新模块稳定后,移除 Trainer 中的旧实现
```

### 7.5 风险控制

| 风险 | 应对措施 |
|------|---------|
| 新模块引入bug | 单元测试覆盖 + 对比旧实现输出 |
| 性能下降 | Profiling 对比,确保无额外开销 |
| 接口不兼容 | 保留适配器层,渐进迁移 |
| 团队学习成本 | 文档 + Code Review + Pair Programming |

### 7.6 成功指标

| 指标 | 目标 | 衡量方式 |
|------|------|---------|
| **代码复杂度** | Trainer < 300行 | 行数统计 |
| **测试覆盖率** | >80% | pytest --cov |
| **模块耦合度** | <5个交叉引用 | 依赖分析工具 |
| **训练速度** | 无性能下降 | Benchmark 对比 |
| **可扩展性** | 新增功能<50行代码 | 实际案例验证 |

---

### 7.7 快速启动 (Quick Wins)

**无需大规模重构,今天就能做:**

1. **提取配置类**
```python
# 新建: configs/trainer_config.py
@dataclass
class TrainerConfig:
    lr: float = 1e-3
    weight_decay: float = 5e-5
    lambda_avuc: float = 0.1
    lambda_struct: float = 0.1
    u_low: float = 0.30
    u_high: float = 0.65
    
# Trainer 使用:
def __init__(self, config: TrainerConfig, ...):
    self.config = config
    # 用 config.lr 替代 self.lr
```

2. **移除魔法数字**
```python
# ❌ 当前
u_low, u_high = 0.30, 0.65

# ✅ 改为
class UncertaintyConfig:
    CORRECT_UPPER_BOUND = 0.30
    WRONG_LOWER_BOUND = 0.65
```

3. **抽取重复代码**
```python
# ❌ 当前: 邻居采样分支重复 5+ 处
if hasattr(batch, 'n_id'):
    h_sub = self.gnn(x, edge_index)
    h_gnn_target = h_sub[:batch_size]
    ...
else:
    h_gnn_target = x[n_id]
    ...

# ✅ 改为
def _extract_gnn_features(self, batch, x, edge_index):
    if hasattr(batch, 'n_id'):
        return self.gnn(x, edge_index)[:batch.batch_size]
    return x[batch[0]]  # nodes
```

### 7.8 中期目标 (1-2 Weeks)

1. 实现 `GraphDataModule`
2. 实现 `LossComputer`
3. 移除 Trainer 中的数据加载逻辑

### 7.9 长期目标 (1 Month)

1. 完成所有模块拆分
2. 达成 80% 测试覆盖
3. 文档化所有接口

---

## 7. 参考资料

### 7.1 相关设计模式

- **Strategy Pattern**: 用于训练策略切换 (GNN预训练/融合训练)
- **Factory Pattern**: 用于模型构建 (build_gnn, build_fusion)
- **Observer Pattern**: 用于训练回调 (wandb logging, checkpoint saving)
- **Template Method**: 用于训练循环基类

### 7.2 类似项目借鉴

- **PyTorch Lightning**: Trainer 抽象, LightningModule, DataModule
- **HuggingFace Transformers**: TrainingArguments, Trainer
- **FastAI**: Learner, DataLoaders, Callbacks

### 7.3 代码质量工具

```bash
# 复杂度分析
pip install radon
radon cc QwenPrecomputedTrainer.py -a

# 代码风格检查
pip install flake8 black
flake8 QwenPrecomputedTrainer.py
black QwenPrecomputedTrainer.py

# 类型检查
pip install mypy
mypy QwenPrecomputedTrainer.py
```

---

## 9. 总结

### 当前问题核心

1. **上帝类**: Trainer 承担了过多职责
2. **紧耦合**: 数据、模型、训练、评估混在一起
3. **难测试**: 依赖全局状态 (GPU, wandb, 文件系统)
4. **难扩展**: 新增功能需要修改核心类

### 重构收益

1. **可维护性**: 单一职责,模块化
2. **可测试性**: 依赖注入,接口抽象
3. **可扩展性**: 新增功能无需修改核心
4. **可读性**: 清晰的架构,减少认知负担

### 下一步行动

✅ **今天**: 提取 `TrainerConfig`  
✅ **本周**: 实现 `LossComputer`  
✅ **本月**: 完成数据层分离  
✅ **长期**: 全面模块化重构

---

**文档维护**: 随重构进展更新此文档  
**反馈渠道**: 欢迎通过 Issue/PR 提出改进建议

---

## 8. 当前实现状态

> **更新时间**: 2026-01-31

### 8.1 采样方法实现 ✅

#### 8.1.1 NeighborSampler 类 (纯 Python 实现)

**位置**: `QwenPrecomputedTrainer.py` 第 30-126 行

```python
class NeighborSampler(DataLoader):
    """
    [Pure Python] 简易图邻居采样器
    替代 torch_geometric.loader.NeighborLoader，解决缺少 pyg-lib/torch-sparse 的报错问题。
    """
    def __init__(self, edge_index, edge_type, sizes, batch_size, input_nodes, ...):
        # 构建邻接表 (支持 edge_type)
        self.adj = [[] for _ in range(num_nodes)]
        self.adj_t = [[] for _ in range(num_nodes)]  # 存储边类型
        
    def sample_subgraph(self, batch_indices):
        # BFS 多层采样
        for size in self.sizes:
            # 随机采样邻居
            ...
        return MiniBatch(n_id, edge_index, edge_type, batch_size)
```

**功能特性**:
| 特性 | 状态 | 说明 |
|------|------|------|
| BFS 多层采样 | ✅ | 默认 `sizes=[10, 10]` 两层 |
| 异质图支持 | ✅ | 保留 `edge_type` 信息 |
| 随机采样 | ✅ | 当邻居数 > size 时随机选取 |
| 局部索引 | ✅ | 边索引映射到子图本地节点ID |
| GPU 兼容 | ✅ | `MiniBatch.to(device)` 方法 |

#### 8.1.2 MiniBatch 数据结构

```python
class MiniBatch:
    n_id: Tensor        # 子图节点的全局ID [num_sampled_nodes]
    edge_index: Tensor  # 子图边索引 (局部ID) [2, num_edges]
    edge_type: Tensor   # 边类型 [num_edges] (可选)
    batch_size: int     # 目标节点数量 (前 batch_size 个是 seed nodes)
```

#### 8.1.3 数据加载器创建

```python
def _create_dataloader(self, indices, shuffle):
    if self.dataloader == 'neighbor':
        return NeighborSampler(
            edge_index=self.data_dict['edge_index'],
            edge_type=self.edge_type,
            sizes=[10, 10],  # 2层 GNN，每层采10个邻居
            batch_size=self.batch_size,
            input_nodes=indices,
            shuffle=shuffle
        )
    else:
        # Full batch 模式
        dataset = TensorDataset(indices, self.labels[indices])
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
```

### 8.2 批次处理统一 ✅

#### 8.2.1 _unpack_batch 方法

**位置**: 第 240-273 行

```python
def _unpack_batch(self, batch):
    if hasattr(batch, 'n_id'):  # NeighborSampler 返回的 MiniBatch
        n_id = batch.n_id
        batch_size = batch.batch_size
        edge_index = batch.edge_index.to(self.device)
        edge_type = batch.edge_type.to(self.device) if batch.edge_type else None
        
        x = self.embeddings[n_id].to(self.device)
        y = self.labels[n_id[:batch_size]].to(self.device)
        
        return x, y, edge_index, edge_type, batch_size, n_id
    else:  # Full batch 模式
        nodes, labels = batch
        # 全图 GNN 前向
        h_all = self._call_gnn(self.embeddings.to(device), full_edge_index, edge_type)
        return h_all, labels, full_edge_index, edge_type, nodes.size(0), nodes
```

**统一接口**:
- 返回值: `(x, y, edge_index, edge_type, batch_size, n_id)` - 6 元组
- 两种模式返回相同结构，训练代码无需判断

### 8.3 GNN 调用统一 ✅

```python
def _call_gnn(self, x, edge_index, edge_type):
    """统一 GNN 调用接口，自动处理 edge_type"""
    if edge_type is not None:
        return self.gnn(x, edge_index, edge_type)
    else:
        return self.gnn(x, edge_index)
```

### 8.4 一致性计算修复 ✅

**位置**: 第 303-336 行

```python
def _compute_batch_consistency(self, x, edge_index, batch_size):
    """
    计算 batch 内节点的结构一致性
    
    ✅ 修复: 使用传入的子图 edge_index，而非全图 self.edge_index
    ✅ 安全检查: 索引越界保护
    """
    num_nodes = x.size(0)
    
    if edge_index.size(1) == 0:
        return torch.full((batch_size, 1), 0.5, device=x.device)
    
    row, col = edge_index
    
    # 安全检查
    valid_mask = (row < num_nodes) & (col < num_nodes) & (row >= 0) & (col >= 0)
    if valid_mask.sum() == 0:
        return torch.full((batch_size, 1), 0.5, device=x.device)
    
    row, col = row[valid_mask], col[valid_mask]
    
    edge_sim = F.cosine_similarity(x[row], x[col], dim=1)
    consistency_all = scatter(edge_sim, row, dim=0, dim_size=num_nodes, reduce='mean')
    consistency_all = (consistency_all + 1.0) / 2.0
    
    return consistency_all[:batch_size].unsqueeze(1)
```

### 8.5 训练阶段状态

| 训练阶段 | 方法 | 状态 | 验证 |
|---------|------|------|------|
| VIB 预训练 | `pretrain_text_vib()` | ✅ 正常 | wandb 记录完整 |
| GNN 预训练 | `pretrain_gnn_stage()` | ✅ 正常 | 修复了 6 元组解包 |
| Gate 预热 | `pretrain_gate_stage()` | ✅ 正常 | - |
| 融合训练 | `train()` | ✅ 正常 | - |

### 8.6 已修复的 Bug

| Bug | 原因 | 修复 | 日期 |
|-----|------|------|------|
| `index out of bounds` | `_compute_batch_consistency` 使用全图 edge_index | 改用传入的子图 edge_index | 2026-01-31 |
| `too many values to unpack` | `_validate_gnn_stats` 只解包 5 个值 | 改为解包 6 个值 (含 edge_type) | 2026-01-31 |

### 8.7 代码统计

```
QwenPrecomputedTrainer.py
├── 总行数: 1045 行
├── 类定义: 3 个 (MiniBatch, NeighborSampler, QwenPrecomputedTrainer)
├── 方法数: ~35 个
└── 采样相关: ~130 行 (12.4%)
```

### 8.8 待优化项

| 优先级 | 项目 | 状态 | 建议 |
|--------|------|------|------|
| 🟠 P1 | 分支逻辑重复 | ⚠️ 待处理 | 抽取 `_extract_gnn_features()` |
| 🟡 P2 | 配置对象缺失 | ⚠️ 待处理 | 创建 `TrainerConfig` |
| 🟡 P2 | 损失计算耦合 | ⚠️ 待处理 | 抽取 `LossComputer` |
| 🟢 P3 | 诊断分析独立 | ⚠️ 待处理 | 抽取 `DiagnosticAnalyzer` |

---

## 10. 变更日志

| 日期 | 变更内容 |
|------|---------|
| 2026-01-31 | 初始版本，完成架构分析 |
| 2026-01-31 | 更新采样方法实现状态，记录 bug 修复 |
| 2026-01-31 | **深度工程审计**: 添加逻辑缺陷、安全隐患、性能瓶颈分析 |
| 2026-01-31 | **模块化重构方案**: 添加完整目录结构和模块职责详解 |
