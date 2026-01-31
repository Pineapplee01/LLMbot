"""
Training Configuration Classes
配置类：集中管理所有超参数，避免参数传递混乱
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataConfig:
    """数据加载配置"""
    batch_size: int = 64
    sampling_mode: str = 'full'  # 'neighbor' or 'full'
    neighbor_sizes: List[int] = field(default_factory=lambda: [10, 10])
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """模型结构配置"""
    # GNN
    gnn_type: str = 'RGT'
    gnn_hidden_dim: int = 512
    gnn_n_layers: int = 2
    gnn_heads: int = 4
    
    # Fusion
    fusion_hidden_dim: int = 256
    fusion_heads: int = 4
    fusion_dropout: float = 0.1
    
    # Global
    dropout: float = 0.3


@dataclass
class LossConfig:
    """损失函数权重配置"""
    # 基础权重
    lambda_avuc: float = 0.1
    lambda_struct: float = 0.1
    lambda_supcon: float = 0.1
    
    # 辅助损失权重
    text_aux_weight: float = 0.5
    graph_aux_weight: float = 0.5
    edl_text_weight: float = 0.2
    edl_graph_weight: float = 0.2
    gate_weight: float = 0.1
    
    # 不确定性边界
    u_correct_upper: float = 0.30
    u_wrong_lower: float = 0.65
    
    # KL warmup
    kl_warmup_start: int = 5
    kl_warmup_duration: int = 5
    kl_base_weight: float = 0.01  # For VIB pretraining


@dataclass
class TrainingConfig:
    """训练流程配置"""
    # 基础参数
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 5e-5
    grad_clip: float = 5.0
    
    # 预训练阶段
    pretrain_text_epochs: int = 15
    pretrain_gnn_epochs: int = 15
    pretrain_gate_epochs: int = 10
    
    # 早停
    patience: int = 20
    min_delta: float = 1e-4
    
    # 检查点
    save_top_k: int = 3
    monitor_metric: str = 'val_f1'
    monitor_mode: str = 'max'


@dataclass
class TrainerConfig:
    """完整训练器配置 (主配置类)"""
    # 子配置
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 系统配置
    device: str = 'cuda'
    seed: int = 42
    exp_name: str = 'experiment'
    
    # 日志
    use_wandb: bool = True
    wandb_project: str = 'Uncertainty_Gated_Fusion'
    log_interval: int = 10
    
    # 阶段控制
    pretrain_text: bool = False
    pretrain_gnn: bool = False
    pretrain_gate: bool = False
    run_fusion: bool = True
    
    def __post_init__(self):
        """验证配置合法性"""
        assert self.data.sampling_mode in ['neighbor', 'full'], \
            f"Invalid sampling_mode: {self.data.sampling_mode}"
        assert self.loss.u_correct_upper < self.loss.u_wrong_lower, \
            "u_correct_upper must be less than u_wrong_lower"
