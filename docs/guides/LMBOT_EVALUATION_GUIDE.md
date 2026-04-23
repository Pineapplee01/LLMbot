# LMBot蒸馏模型评估指南

## 概览
`evaluate_lmbot.py` 是专门为LMBot蒸馏后的语言模型(LM)和图神经网络(GNN)设计的评估脚本。

## 快速开始

### 1. 批量评估所有种子和模型类型
```bash
# 评估所有5个种子的LM和GNN模型
python evaluate_lmbot.py --base_dir . --seeds 1 2 3 4 5 --models LM GNN --output lmbot_eval_results

# 只评估GNN模型
python evaluate_lmbot.py --models GNN --output gnn_only_results
```

### 2. 单个种子评估
```bash
# 评估特定种子的GNN模型
python evaluate_lmbot.py --mode single --seed_dir TwiBot-20_seed_1 --model_type GNN --output single_eval

# 评估特定种子的LM模型  
python evaluate_lmbot.py --mode single --seed_dir TwiBot-20_seed_1 --model_type LM --output single_eval
```

### 3. 与Qwen嵌入对比
```bash
# 先用新脚本评估LMBot模型
python evaluate_lmbot.py --models GNN --output lmbot_gnn_results

# 然后用原始evaluation.py对比Qwen和LMBot
python evaluation.py --emb1 datasets/TwiBot-20/qwen3_emb_last.pt --emb2 TwiBot-20_seed_1/intermediate/GNN/best_embeddings.pt --out qwen_vs_lmbot_comparison
```

## 输出结果结构

运行后会在输出目录生成：

```
lmbot_eval_results/
├── LM/
│   ├── seed_1/
│   │   ├── summary_LM_seed_1.json    # 评估指标
│   │   └── LM_seed_1_tsne.png        # t-SNE可视化
│   ├── seed_2/ ...
│   └── seed_5/ ...
├── GNN/
│   ├── seed_1/ ...
│   └── seed_5/ ...  
├── LM_cross_seed_stats.json          # 跨种子统计
├── GNN_cross_seed_stats.json
└── all_results.json                  # 完整结果汇总
```

## 主要评估指标

### 对于每个模型：
- **Silhouette Score**: 聚类质量，越高越好 ([-1, 1])
- **Davies-Bouldin Score**: 聚类分离度，越低越好
- **Linear Probe Accuracy**: 线性分类器准确率 (5-fold CV)
- **t-SNE可视化**: 2D降维可视化图

### 跨种子统计：
- 每个指标的平均值、标准差、最小值、最大值
- 帮助评估模型稳定性和一致性

## 常见参数说明

- `--base_dir`: 包含种子文件夹的基础目录 (默认: `.`)
- `--seeds`: 要评估的种子列表 (默认: `[1,2,3,4,5]`)
- `--models`: 模型类型 `LM`, `GNN`, `MLP` (默认: `['LM', 'GNN']`)
- `--epoch`: 选择哪个epoch的模型 (`best`/`last`/具体数字)
- `--dataset`: TwiBot-20数据集路径 (默认: `datasets/TwiBot-20`)
- `--output`: 输出目录

## 与原始evaluation.py的区别

| 特性 | evaluate_lmbot.py | evaluation.py |
|------|------------------|---------------|
| 目标 | LMBot蒸馏模型专用 | 通用嵌入对比 |
| 输入 | 种子目录结构 | 直接的.pt文件 |
| 批处理 | ✅ 支持多种子 | ❌ 单次对比 |
| 模型类型 | LM/GNN/MLP分别评估 | 任意两个嵌入 |
| 跨种子统计 | ✅ 自动计算 | ❌ 需手动汇总 |

## 故障排除

1. **文件未找到错误**：确保种子目录结构正确，包含`intermediate/{LM,GNN,MLP}/`子目录
2. **内存不足**：对于大型嵌入，t-SNE可能消耗大量内存，考虑减少样本数或使用服务器运行
3. **依赖缺失**：确保安装了所需的Python包（numpy, torch, sklearn, matplotlib, scipy）

## 推荐工作流程

1. **批量评估**：首先运行批量模式获得所有模型的基线评估
2. **交叉对比**：使用原始`evaluation.py`对比Qwen与最佳LMBot模型
3. **深度分析**：针对表现最好的种子和模型类型进行单独详细分析
4. **报告生成**：基于跨种子统计制作实验报告

这样可以全面评估LMBot蒸馏后各组件的性能，并与Qwen基线进行有意义的对比。