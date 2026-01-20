import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, HGTConv, TransformerConv, GATConv
from SimpleHGN import SimpleHGNConv
from RGT import RGTLayer, SemanticAttention
from torch_geometric.nn.models import MLP

def build_gnn(gnn_type, config):
    if gnn_type == 'RGCN':
        return RGCN(config)
    elif gnn_type == 'RGT':
        return RGT(config)
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")



class RGCN(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = model_config['gnn_hidden_dim']
        self.n_layers = model_config['gnn_n_layers']
        self.convs = nn.ModuleList([])
        self.linear_in = nn.Linear(model_config['lm_input_dim'], self.hidden_dim)
  
        for i in range(self.n_layers):
            self.convs.append(RGCNConv(self.hidden_dim, self.hidden_dim, model_config['n_relations']))

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
        x = self.linear_in(x)
        x = self.dropout(x)
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_type)
            x = self.activation(x)
        x = self.linear_pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear_out(x)
    


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
    


class RGT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Config Mapping
        self.in_dim = config['in_channels']           # 4096 (Qwen)
        self.hidden_dim = config['hidden_channels']   # 512
        self.num_relations = config['num_relations']
        self.n_layers = config['n_layers']
        
        self.att_heads = config['heads']              
        self.semantic_heads = 4                       # Default for RGT
        self.dropout_val = config['dropout']

        # --- MODIFICATION START ---
        # Replaced their MLP with a Qwen Projector
        self.input_proj = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_val)
        )
        # --- MODIFICATION END ---

        self.convs = nn.ModuleList()
        for i in range(self.n_layers):
            self.convs.append(
                RGTLayer(
                    num_edge_type=self.num_relations,
                    in_size=self.hidden_dim,
                    out_size=self.hidden_dim,
                    layer_num_heads=self.att_heads,
                    semantic_head=self.semantic_heads,
                    dropout=self.dropout_val
                )
            )

        self.activation = nn.ELU()
        
        # --- MODIFICATION: Removed self.linear_out (Classifier) ---

    def prepare_data_for_RGT(self, edge_index, edge_type):
        """Splits master edge_index into list of edge_indices per relation."""
        edge_index_list = []
        for i in range(self.num_relations):
            mask = (edge_type == i)
            if mask.sum() > 0:
                sub_edges = edge_index[:, mask]
            else:
                # Handle empty relations safely
                sub_edges = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
            edge_index_list.append(sub_edges)
        return edge_index_list
    
    def forward(self, x, edge_index, edge_type):
        # 1. Project Qwen
        x = self.input_proj(x)
        
        # 2. Prepare Heterogeneous Data
        edge_index_list = self.prepare_data_for_RGT(edge_index, edge_type)
        
        # 3. RGT Layers
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index_list)
            x = self.activation(x)
            
        # 4. Return Embedding (For Fusion)
        return x
    
    

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