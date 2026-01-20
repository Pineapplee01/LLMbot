"""
Qwen3-Embedding-8B Model for Bot Detection
使用Qwen3-Embedding替代RoBERTa作为语言模型backbone
"""
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from torch_geometric.nn.models import MLP
import torch


class Qwen_Model(nn.Module):
    """
    Qwen3-Embedding-8B模型包装类
    支持与原LM_Model相同的接口以便于对比实验
    """
    def __init__(self, model_config):
        super().__init__()
        self.model_name = 'qwen3-embedding'
        
        # 加载Qwen3-Embedding-8B模型
        # 这是一个专门用于生成高质量embeddings的模型
        model_name_or_path = model_config.get('qwen_model_path', 'Qwen/Qwen3-Embedding-8B')
        
        print(f"Loading Qwen3-Embedding-8B model from {model_name_or_path}...")
        self.LM = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if model_config.get('use_fp16', False) else torch.float32
        )
        
        # 获取隐藏层大小（Qwen3-Embedding-8B的hidden_size是8192）
        self.hidden_size = self.LM.config.hidden_size
        
        # 分类器：将embedding映射到2分类
        self.classifier = MLP(
            in_channels=self.hidden_size,
            hidden_channels=model_config['classifier_hidden_dim'],
            out_channels=2,
            num_layers=model_config['classifier_n_layers'],
            act=model_config['activation'],
            dropout=model_config['dropout']
        )
        
        # Dropout配置
        if hasattr(self.LM.config, 'hidden_dropout_prob'):
            self.LM.config.hidden_dropout_prob = model_config['lm_dropout']
        if hasattr(self.LM.config, 'attention_probs_dropout_prob'):
            self.LM.config.attention_probs_dropout_prob = model_config['att_dropout']
        
        print(f"Qwen3-Embedding-8B model loaded. Hidden size: {self.hidden_size}")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, tokenized_tensors):
        """
        前向传播
        Args:
            tokenized_tensors: tokenizer输出的字典，包含input_ids, attention_mask等
        Returns:
            embedding: detached的句子embedding (batch_size, hidden_size)
            logits: 分类logits (batch_size, 2)
        """
        # 获取Qwen3-Embedding的输出
        out = self.LM(**tokenized_tensors)
        
        # Qwen3-Embedding直接返回last_hidden_state，需要进行池化
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = out.last_hidden_state
        
        # 考虑attention_mask进行加权平均池化
        attention_mask = tokenized_tensors['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        
        # 返回detached的embedding和分类logits
        return embedding.detach(), self.classifier(embedding)


class Qwen_Model_Small(nn.Module):
    """
    备用：使用小型Qwen模型（非embedding专用）
    注意：推荐使用Qwen3-Embedding-8B而非此类
    """
    def __init__(self, model_config):
        super().__init__()
        self.model_name = 'qwen3-embedding'
        
        # 默认使用Qwen3-Embedding-8B
        # 如果需要更小的模型，可以指定其他路径
        model_name_or_path = model_config.get('qwen_model_path', 'Qwen/Qwen3-Embedding-8B')
        
        print(f"Loading Qwen3-Embedding model from {model_name_or_path}...")
        self.LM = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if model_config.get('use_fp16', False) else torch.float32
        )
        
        self.hidden_size = self.LM.config.hidden_size
        
        self.classifier = MLP(
            in_channels=self.hidden_size,
            hidden_channels=model_config['classifier_hidden_dim'],
            out_channels=2,
            num_layers=model_config['classifier_n_layers'],
            act=model_config['activation'],
            dropout=model_config['dropout']
        )
        
        if hasattr(self.LM.config, 'hidden_dropout_prob'):
            self.LM.config.hidden_dropout_prob = model_config['lm_dropout']
        if hasattr(self.LM.config, 'attention_probs_dropout_prob'):
            self.LM.config.attention_probs_dropout_prob = model_config['att_dropout']
        
        print(f"Qwen3-Embedding model loaded. Hidden size: {self.hidden_size}")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, tokenized_tensors):
        out = self.LM(**tokenized_tensors)
        last_hidden_state = out.last_hidden_state
        
        attention_mask = tokenized_tensors['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        
        return embedding.detach(), self.classifier(embedding)


def build_qwen_tokenizer(model_config):
    """
    构建Qwen3-Embedding tokenizer
    """
    model_name_or_path = model_config.get('qwen_model_path', 'Qwen/Qwen3-Embedding-8B')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # 添加特殊token（与原项目保持一致）
    special_tokens_dict = {
        'additional_special_tokens': ['DESCRIPTION:', 'METADATA:', 'TWEET:']
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    
    tokens_list = ["@USER", '#HASHTAG', "HTTPURL", 'EMOJI', 'RT', 'None']
    tokenizer.add_tokens(tokens_list)
    
    return tokenizer


class Qwen3_Precomputed_Embedding_Model(nn.Module):
    """
    使用预先生成的Qwen3-Embedding-8B embeddings
    这是推荐的方式：先用generate_qwen3_embeddings.py生成embeddings，然后加载使用
    优势：
    1. 避免重复计算embeddings（多次实验时效率高）
    2. 节省GPU内存（不需要加载完整的8B模型）
    3. 保证实验一致性（embeddings固定）
    """
    def __init__(self, embeddings_path, model_config):
        super().__init__()
        self.model_name = 'qwen3-embedding-precomputed'
        
        # 加载预先生成的embeddings
        print(f"Loading precomputed Qwen3-Embedding-8B embeddings from {embeddings_path}...")
        self.embeddings = torch.load(embeddings_path, weights_only=True)
        
        # embeddings shape: (num_users, embedding_dim)
        self.num_users = self.embeddings.shape[0]
        self.hidden_size = self.embeddings.shape[1]
        
        print(f"Loaded embeddings for {self.num_users} users, dimension: {self.hidden_size}")
        
        # 分类器：将embedding映射到2分类
        self.classifier = MLP(
            in_channels=self.hidden_size,
            hidden_channels=model_config['classifier_hidden_dim'],
            out_channels=2,
            num_layers=model_config['classifier_n_layers'],
            act=model_config['activation'],
            dropout=model_config['dropout']
        )
        
        print(f"Qwen3 precomputed embedding model initialized.")
        print(f"Classifier parameters: {sum(p.numel() for p in self.classifier.parameters()):,}")
    
    def forward(self, indices):
        """
        根据索引返回预先计算的embeddings
        Args:
            indices: 用户索引 (batch_size,)
        Returns:
            embedding: 对应的embeddings (batch_size, hidden_size)
            logits: 分类logits (batch_size, 2)
        """
        # 根据索引提取embeddings
        embedding = self.embeddings[indices]
        
        # 计算分类logits
        logits = self.classifier(embedding)
        
        # 返回detached的embedding和logits
        # 注意：embeddings本身是固定的（frozen），只有classifier会被训练
        return embedding.detach(), logits
    
    def get_embedding_dim(self):
        """返回embedding维度"""
        return self.hidden_size
    
    def to_device(self, device):
        """将模型和embeddings移动到指定设备"""
        self.embeddings = self.embeddings.to(device)
        return self.to(device)
