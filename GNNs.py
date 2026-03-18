import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, HGTConv, TransformerConv, GATConv
from torch_geometric.nn.models import MLP
from RGT import RGTLayer, SemanticAttention

def build_gnn(gnn_type, config):
    if gnn_type == 'RGCN':
        return RGCN(config)
    elif gnn_type == 'RGT':
        return RGT(config)
    elif gnn_type == 'SimpleHGN':
        return SimpleHGN(config)
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")

class BaseGraphEncoder(nn.Module):
    """强制契约：所有图编码器必须返回节点嵌入，并暴露输出维度"""
    @property
    def out_dim(self) -> int:
        raise NotImplementedError
    
    def forward(self, x, edge_index, edge_type=None):
        raise NotImplementedError

class RGCN(BaseGraphEncoder):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = model_config['gnn_hidden_dim']
        self._out_dim = self.hidden_dim
        self.n_layers = model_config['gnn_n_layers']
        
        self.activation_name = model_config.get('activation', 'relu').lower()
        self.activation = F.relu if self.activation_name == 'relu' else F.elu
        
        self.linear_in = nn.Linear(model_config['lm_input_dim'], self.hidden_dim)
        
        # 🚨 [核心修复]: 删除了错误的提前调用，只保留合法初始化
        self.convs = nn.ModuleList([
            RGCNConv(self.hidden_dim, self.hidden_dim, model_config['n_relations'])
            for _ in range(self.n_layers)
        ])
        
        self.dropout = nn.Dropout(model_config.get('dropout', 0.3))

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x, edge_index, edge_type=None):
        x = self.linear_in(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = self.activation(x)
            x = self.dropout(x)
        return x 

class RGT(BaseGraphEncoder):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.get('gnn_hidden_dim', 512)
        # 🚨 [核心修复]: 补充 out_dim 契约，防止 main.py 提取时崩溃
        self._out_dim = self.hidden_dim
        self.n_layers = config.get('gnn_n_layers', 2)
        self.num_relations = config['n_relations']
        
        self.activation = F.relu
        self.input_proj = nn.Linear(config['lm_input_dim'], self.hidden_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(self.n_layers):
            self.convs.append(RGTLayer(
                in_size=self.hidden_dim,
                out_size=self.hidden_dim,
                num_edge_type=self.num_relations,
                layer_num_heads=config.get('heads', 4),
                semantic_head=config.get('heads', 4),
                dropout=config.get('dropout', 0.3),
            ))

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def prepare_data_for_RGT(self, edge_index, edge_type):
        edge_index_list = []
        for i in range(self.num_relations):
            mask = (edge_type == i)
            sub_edges = edge_index[:, mask]
            if sub_edges.size(1) == 0:
                sub_edges = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
            edge_index_list.append(sub_edges)
        return edge_index_list
    
    def forward(self, x, edge_index, edge_type=None):
        x = self.input_proj(x)
        edge_index_list = self.prepare_data_for_RGT(edge_index, edge_type)
        
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index_list)
            x = self.activation(x)
        return x   


class SimpleHGN(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = model_config['hidden_dim']
        self.n_layers = model_config['n_layers']
        self.heads = model_config['att_heads']
        self.edge_res = model_config['SimpleHGN_att_res']


        self.convs = nn.ModuleList([])
        for i in range(self.n_layers):
            if i == 0:
                self.convs.append(SimpleHGNConv(768, self.hidden_dim, model_config['n_relations'], 32,  beta=self.edge_res))
            elif i == self.n_layers - 1:
                self.convs.append(SimpleHGNConv(self.hidden_dim, self.hidden_dim, model_config['n_relations'], 32,  beta=self.edge_res, final_layer=True))
            else:
                self.convs.append(SimpleHGNConv(self.hidden_dim, self.hidden_dim, model_config['n_relations'], 32, beta=self.edge_res))

        self.dropout = nn.Dropout(model_config['dropout'])      
        
        self.activation_name = model_config['activation'].lower()
        if self.activation_name == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif self.activation_name == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_name == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError('Please choose activation function from "leakyrelu", "relu" or "elu".')
        
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, 2)

    def forward(self, x, edge_index, edge_type):
        for i in range(self.n_layers):
            if i == 0:
                x, att_res = self.convs[i](x, edge_index, edge_type)
                x = self.dropout(x)
            else:
                x, att_res = self.convs[i](x, edge_index, edge_type, pre_alpha=att_res)
                x = self.dropout(x)
        x = self.linear_pool(x)
        x = self.dropout(x)
        x = self.activation(x)

        return self.linear_out(x)
    
class HGT(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = model_config['gnn_hidden_dim']
        self.n_layers = model_config['gnn_n_layers']
        self.heads = model_config['att_heads']
        self.metadata = (['user'], [('user', 'follower', 'user'), ('user', 'following', 'user')])
        self.mlp = MLP_Model(model_config)
        

        self.convs = nn.ModuleList([])
        for i in range(self.n_layers):
            self.convs.append(HGTConv(self.hidden_dim, self.hidden_dim, self.metadata, self.heads))

        self.dropout = nn.Dropout(model_config['dropout'])
        
        self.activation_name = model_config['activation'].lower()
        if self.activation_name == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif self.activation_name == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_name == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError('Please choose activation function from "leakyrelu", "relu" or "elu".')
        
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, 2)

    def prepare_data_for_HGT(self, x, edge_index, edge_type):
        x_dict = {self.metadata[0][0]: x}
        edge_index_dict = {}
        for i in range(len(self.metadata[1])):

            edge_index_dict[self.metadata[1][i]] = edge_index[:, edge_type==i]
        return x_dict, edge_index_dict

    def forward(self, x1, x2, x3, edge_index, edge_type):
        x = self.mlp(x1, x2, x3)
        x = self.dropout(x)     
        x_dict, edge_index_dict = self.prepare_data_for_HGT(x, edge_index, edge_type)
        for i in range(self.n_layers):
            x = self.convs[i](x_dict, edge_index_dict)
            x[self.metadata[0][0]] = self.activation(self.dropout(x[self.metadata[0][0]]))
            
        x = self.linear_pool(x[self.metadata[0][0]])
        x = self.dropout(x)
        x = self.activation(x)

        return self.linear_out(x)  

class GAT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_dim = config['in_channels']
        self.hidden_dim = config['hidden_channels']
        self.heads = config['heads']
        self.dropout = config['dropout']
        
        self.input_proj = nn.Linear(self.in_dim, self.hidden_dim)

        self.conv1 = GATConv(self.hidden_dim, self.hidden_dim // self.heads, heads=self.heads, dropout=self.dropout)
        self.conv2 = GATConv(self.hidden_dim, self.hidden_dim // self.heads, heads=self.heads, dropout=self.dropout)

    def forward(self, x, edge_index, edge_type=None):
        x = self.input_proj(x)
        x = F.gelu(x)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x    