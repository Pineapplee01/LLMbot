"""
Batch Processor: 统一批次解包与特征提取
将数据预处理逻辑从 Trainer 中解耦
"""

import torch
from typing import Tuple, Union


class BatchProcessor:
    """
    批次处理器: 统一处理邻居采样和全图批次
    """
    
    def __init__(self, embeddings, gnn, device):
        """
        Args:
            embeddings: 预计算的 LM 嵌入 [N, D], 存储在 CPU
            gnn: GNN 模型
            device: 目标设备 (cuda/cpu)
        """
        self.embeddings = embeddings
        self.gnn = gnn
        self.device = device
        
    def process(self, batch, mode='full') -> Tuple[torch.Tensor, ...]:
        """
        统一批次处理接口
        
        Args:
            batch: MiniBatch (邻居采样) 或 Tuple (全图)
            mode: 'neighbor' 或 'full'
            
        Returns:
            Tuple containing:
            - x: 节点特征 [B, D] 或 [subgraph_size, D]
            - labels: 标签 [B]
            - edge_index: 边索引 [2, E]
            - batch_size: 目标节点数量
            - n_id: 节点索引 [B]
        """
        if mode == 'neighbor':
            return self._process_neighbor_batch(batch)
        else:
            return self._process_full_batch(batch)
    
    def _process_neighbor_batch(self, batch):
        """
        处理邻居采样批次 (MiniBatch)
        
        batch 包含:
        - n_id: 子图节点索引 (包含 seed + neighbors)
        - edge_index: 子图边
        - edge_type: 边类型 (可选)
        - batch_size: seed 节点数量
        - x: 子图特征 (如果已预加载)
        - y: 标签 (如果已预加载)
        """
        batch = batch.to(self.device)
        
        # 提取特征 (动态从 CPU 加载)
        if hasattr(batch, 'x') and batch.x is not None:
            x = batch.x
        else:
            x = self.embeddings[batch.n_id].to(self.device)
        
        # 提取标签
        if hasattr(batch, 'y') and batch.y is not None:
            labels = batch.y[:batch.batch_size]
        else:
            # 需要外部传入 labels
            raise ValueError("Batch missing labels")
        
        edge_index = batch.edge_index
        batch_size = batch.batch_size
        n_id = batch.n_id[:batch_size]
        
        return x, labels, edge_index, batch_size, n_id
    
    def _process_full_batch(self, batch):
        """
        处理全图批次 (Tuple)
        
        batch = (node_indices, node_labels)
        """
        nodes, labels = batch
        nodes = nodes.to(self.device)
        labels = labels.to(self.device)
        
        # 全图特征
        x = self.embeddings[nodes].to(self.device)
        
        # 全图边索引 (需要外部传入或预存)
        # 简化: 返回 None, 由调用者处理
        edge_index = None
        
        batch_size = nodes.size(0)
        n_id = nodes
        
        return x, labels, edge_index, batch_size, n_id
    
    def extract_gnn_features(self, batch, x, edge_index, mode='neighbor'):
        """
        提取 GNN 特征 (避免重复代码)
        
        Args:
            batch: 原始批次
            x: 节点特征
            edge_index: 边索引
            mode: 'neighbor' 或 'full'
            
        Returns:
            h_gnn_target: 目标节点的 GNN 特征 [B, D]
        """
        if mode == 'neighbor':
            # 子图前向
            h_sub = self.gnn(x, edge_index)
            if isinstance(h_sub, tuple):
                h_sub = h_sub[0]
            return h_sub[:batch.batch_size]
        else:
            # 全图前向 (需要完整图结构)
            # 这里假设 gnn 已经在 __call__ 中处理
            return x  # 占位符, 实际需要外部处理
