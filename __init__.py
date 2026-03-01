"""
==============================================================================
llmbot - 社交网络机器人检测多阶段训练框架
==============================================================================

本模块是 SeGA (Selective and Gated Attention) 机器人检测系统的训练框架。
它实现了一种新颖的多模态融合方法，结合了：
- 语言模型 (Language Model) 提取的文本特征
- 图神经网络 (Graph Neural Network) 提取的社交关系特征
- 不确定性估计 (Uncertainty Estimation) 用于可靠的预测

【项目概述】
============

这是一个用于检测社交网络机器人（Bot）的深度学习框架。
机器人检测的目标是区分真实用户（Human）和自动化账户（Bot）。

为什么这个问题重要？
- 社交网络上的机器人可能传播虚假信息
- 机器人可能进行垃圾信息投放、网络诈骗等
- 准确检测机器人有助于维护网络安全

【核心设计理念】
================

1. **多模态信息融合**
   - 文本信息：用户发布的推文内容
   - 图结构信息：用户之间的关注/被关注关系
   - 融合策略：使用门控机制动态平衡两种信息

2. **分阶段训练**
   - 阶段1 (VIB)：预训练文本编码器
   - 阶段2 (GNN)：预训练图神经网络
   - 阶段3 (Gate)：预热门控模块
   - 阶段4 (Fusion)：端到端融合训练

3. **不确定性量化**
   - 使用 Evidential Deep Learning (EDL) 估计预测不确定性
   - 不确定性高的预测可以被标记为"不确定"
   - 提高模型在实际应用中的可靠性

【模块结构】
============

本包 (llmbot) 包含以下模块：

1. base_trainer.py - 基础训练器
   - TrainerConfig: 训练配置数据类
   - MiniBatch: 小批量数据容器
   - NeighborSampler: 邻居采样器（用于GNN的mini-batch训练）
   - BaseTrainer: 所有训练器的基类

2. losses.py - 损失函数
   - EdlLoss: Evidential Deep Learning损失
   - AvUCLoss: 平均不确定性校准损失
   - UncertaintyMarginLoss: 不确定性边际损失
   - SupervisedContrastiveLoss: 有监督对比学习损失
   - LossComputer: 统一的损失计算接口

3. Trainer.py - 阶段训练器
   - VIBTrainer: VIB预训练器
   - GNNTrainer: GNN预训练器
   - GateTrainer: 门控预热训练器
   - FusionTrainer: 融合训练器

4. main.py - 主入口
   - 命令行参数解析
   - 数据加载
   - 模型构建
   - 训练流程编排

【快速开始】
============

>>> # 基本使用方式
>>> from llmbot import main
>>> # 然后运行: python -m llmbot.main --dataset TwiBot-20

>>> # 或者直接导入训练器
>>> from llmbot import VIBTrainer, GNNTrainer, GateTrainer, FusionTrainer

>>> # 导入损失函数
>>> from llmbot import LossComputer, EdlLoss

>>> # 导入配置和工具
>>> from llmbot import TrainerConfig, NeighborSampler, MiniBatch

【适合的读者】
==============

本代码库主要面向：
1. 深度学习入门者 - 代码注释详细，适合学习
2. GNN研究者 - 学习邻居采样和图神经网络训练
3. 多模态融合研究者 - 学习模态融合技术
4. 不确定性量化研究者 - 学习EDL等方法

【参考文档】
============

- QUICKSTART.md: 快速上手教程
- CONCEPTS.md: 核心概念详解
- README.md: 项目说明

【相关论文】
============

如果使用本代码，请引用相关论文（具体引用信息请参见README.md）。

【依赖要求】
============

必须的Python包：
- torch >= 1.9.0        # PyTorch深度学习框架
- torch_geometric       # PyTorch Geometric图神经网络库
- transformers          # Hugging Face的预训练模型库
- scikit-learn          # 机器学习工具库
- wandb                 # 实验追踪工具

可选的Python包：
- numpy                 # 数值计算
- tqdm                  # 进度条

==============================================================================
版权信息和许可证
==============================================================================
"""

# ==============================================================================
# 模块导入
# ==============================================================================

# 从 base_trainer 模块导入
# 这些是基础设施组件，所有训练阶段都会用到
from .base_trainer import (
    TrainerConfig,      # 训练配置数据类，包含学习率、批量大小等超参数
    MiniBatch,          # 小批量数据容器，封装一次训练迭代的所有数据
    NeighborSampler,    # 邻居采样器，用于从大图中采样子图
    BaseTrainer,        # 基础训练器类，提供通用的训练、验证、测试方法
)

# 从 losses 模块导入
# 这些是各种损失函数，用于优化模型
from .losses import (
    EdlLoss,                    # Evidential Deep Learning 损失
    AvUCLoss,                   # 平均不确定性校准 (Average Uncertainty Calibration) 损失
    UncertaintyMarginLoss,      # 不确定性边际损失
    SupervisedContrastiveLoss,  # 有监督对比学习损失
    LossComputer,               # 统一的损失计算接口
)

# 从 Trainer 模块导入
# 这些是各阶段的具体训练器
from .Trainer import (
    VIBTrainer,     # 阶段1: VIB (Variational Information Bottleneck) 预训练器
    GNNTrainer,     # 阶段2: GNN (Graph Neural Network) 预训练器
    GateTrainer,    # 阶段3: Gate (门控模块) 预热训练器
    FusionTrainer,  # 阶段4: Fusion (融合) 训练器
)

# ==============================================================================
# 定义公开API
# ==============================================================================

# __all__ 定义了使用 "from llmbot import *" 时会导入的名称
# 这是Python的一种约定，用于控制模块的公开接口

__all__ = [
    # -------------------- 配置和工具类 --------------------
    # 这些类用于设置训练参数和管理数据
    
    "TrainerConfig",    # 训练配置：包含学习率、批量大小、epoch数等超参数
                        # 使用方法: config = TrainerConfig(lr=1e-3, batch_size=256)
    
    "MiniBatch",        # 小批量数据：封装一次前向传播所需的所有数据
                        # 包含: 节点特征、邻接矩阵、标签等
    
    "NeighborSampler",  # 邻居采样器：从大图中采样固定大小的子图
                        # 用于将大规模图神经网络训练转化为mini-batch训练
    
    # -------------------- 训练器类 --------------------
    # 按训练阶段顺序排列
    
    "BaseTrainer",      # 基础训练器：所有训练器的父类
                        # 提供: train_epoch, evaluate, save_checkpoint 等通用方法
    
    "VIBTrainer",       # 阶段1 - VIB预训练器
                        # 目的: 使用变分信息瓶颈预训练文本编码器
                        # 训练模块: model.lm_encoder, model.lm_classifier
    
    "GNNTrainer",       # 阶段2 - GNN预训练器
                        # 目的: 预训练图神经网络以捕获社交关系
                        # 训练模块: model.graph_model
    
    "GateTrainer",      # 阶段3 - 门控预热器
                        # 目的: 预热门控模块，学习如何融合两种模态
                        # 训练模块: model.gated_attention
    
    "FusionTrainer",    # 阶段4 - 融合训练器
                        # 目的: 端到端微调整个模型
                        # 训练模块: 所有模块联合训练
    
    # -------------------- 损失函数类 --------------------
    # 按使用频率和重要性排列
    
    "LossComputer",     # 统一损失计算器：组合多种损失函数
                        # 使用方法: 
                        # loss_computer = LossComputer(args)
                        # total_loss = loss_computer(output_dict, labels)
    
    "EdlLoss",          # Evidential Deep Learning 损失
                        # 用于不确定性感知的分类
                        # 输出: 分类损失 + KL散度正则项
    
    "AvUCLoss",         # 平均不确定性校准损失
                        # 用于校准模型的不确定性估计
                        # 目标: 高不确定性 ↔ 高误差率
    
    "UncertaintyMarginLoss",    # 不确定性边际损失
                                # 鼓励不同类别间有明显的不确定性区分
    
    "SupervisedContrastiveLoss",    # 有监督对比学习损失
                                    # 用于学习更好的特征表示
                                    # 目标: 同类样本近，异类样本远
]

# ==============================================================================
# 版本信息
# ==============================================================================

__version__ = "1.0.0"
__author__ = "LMBot Team"

# ==============================================================================
# 模块级别的文档字符串补充
# ==============================================================================

# 使用示例 1: 完整的训练流程
USAGE_EXAMPLE_FULL = """
【使用示例 1: 完整训练流程】

# 命令行运行（推荐）
python -m llmbot.main --dataset TwiBot-20 --device cuda:0

# 或者在Python中调用
import subprocess
subprocess.run(["python", "-m", "llmbot.main", "--dataset", "TwiBot-20"])
"""

# 使用示例 2: 自定义训练
USAGE_EXAMPLE_CUSTOM = """
【使用示例 2: 自定义训练】

from llmbot import TrainerConfig, VIBTrainer

# 创建配置
config = TrainerConfig(
    lr=1e-4,
    weight_decay=1e-5,
    batch_size=256,
    epochs=10,
    device='cuda:0'
)

# 假设已有 model, train_loader, valid_loader
trainer = VIBTrainer(config, model, train_loader, valid_loader)
trainer.train()
"""

# 使用示例 3: 使用损失函数
USAGE_EXAMPLE_LOSS = """
【使用示例 3: 使用损失函数】

import torch
from llmbot import EdlLoss

# 创建EDL损失
edl_loss = EdlLoss(num_classes=2)

# 假设 evidence 是模型输出 (batch_size, num_classes)
# labels 是真实标签 (batch_size,)
evidence = torch.rand(32, 2)  # 示例数据
labels = torch.randint(0, 2, (32,))

# 计算损失
loss = edl_loss(evidence, labels)
print(f"EDL Loss: {loss.item()}")
"""

# ==============================================================================
# 便捷函数
# ==============================================================================

def get_version():
    """
    获取当前包的版本号。
    
    返回:
        str: 版本号字符串，如 "1.0.0"
    
    使用示例:
        >>> import llmbot
        >>> print(llmbot.get_version())
        1.0.0
    """
    return __version__


def get_available_trainers():
    """
    获取所有可用的训练器类列表。
    
    返回:
        dict: 训练器名称到类的映射
    
    使用示例:
        >>> import llmbot
        >>> trainers = llmbot.get_available_trainers()
        >>> print(trainers.keys())
        dict_keys(['vib', 'gnn', 'gate', 'fusion'])
    """
    return {
        "vib": VIBTrainer,
        "gnn": GNNTrainer,
        "gate": GateTrainer,
        "fusion": FusionTrainer,
    }


def get_available_losses():
    """
    获取所有可用的损失函数类列表。
    
    返回:
        dict: 损失函数名称到类的映射
    
    使用示例:
        >>> import llmbot
        >>> losses = llmbot.get_available_losses()
        >>> print(losses.keys())
        dict_keys(['edl', 'avuc', 'margin', 'contrastive', 'computer'])
    """
    return {
        "edl": EdlLoss,
        "avuc": AvUCLoss,
        "margin": UncertaintyMarginLoss,
        "contrastive": SupervisedContrastiveLoss,
        "computer": LossComputer,
    }


# ==============================================================================
# 模块加载时的初始化（如果需要）
# ==============================================================================

# 如果需要在模块加载时执行某些操作，可以在这里添加
# 例如：设置随机种子、配置日志等
# 
# 目前不需要任何初始化操作，保持简洁

# ==============================================================================
# 文件结束
# ==============================================================================
