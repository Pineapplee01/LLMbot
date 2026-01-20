import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from sklearn.manifold import TSNE
from tqdm import tqdm

# Import your project modules
from utils import load_raw_data, seed_setting
from GNNs import build_gnn
from AttentionFusion import CrossAttentionFusion

def visualize_embedding_space(args):
    device = torch.device(args.device)
    seed_setting(args.seed)

    print(f"Loading data from {args.dataset_path}...")
    data_dict = load_raw_data(args.dataset_path, use_GNN=True)
    
    # 1. Setup Inputs
    edge_index = data_dict['edge_index'].to(device)
    edge_type = data_dict.get('edge_type', None)
    if edge_type is not None: edge_type = edge_type.to(device)
    
    # Labels
    labels = data_dict['labels']
    if labels.dim() > 1: labels = labels.argmax(dim=1)
    labels = labels.cpu().numpy()
    
    # Numerical Features (Handle missing)
    if 'num_properties' in data_dict:
        # Apply the same Log-Norm you used in training!
        raw = data_dict['num_properties']
        num_features = torch.log1p(raw)
        num_features = (num_features - num_features.mean(0)) / (num_features.std(0) + 1e-6)
        num_features = num_features.to(device)
        num_feat_dim = num_features.shape[1]
    else:
        num_features = None
        num_feat_dim = 0

    # 2. Load Qwen Embeddings
    print(f"Loading Qwen embeddings from {args.embeddings_path}...")
    qwen_emb = torch.load(args.embeddings_path, map_location='cpu').float()
    if not isinstance(qwen_emb, torch.Tensor): qwen_emb = torch.tensor(qwen_emb)
    qwen_emb = qwen_emb.to(device)
    
    # 3. Build Model
    print("Building SeGA Model...")
    # GNN
    gnn_config = {
        'in_channels': qwen_emb.shape[1],
        'hidden_channels': args.hidden_dim,
        'out_channels': args.hidden_dim,
        'num_relations': int(edge_type.max() + 1) if edge_type is not None else 1,
        'n_layers': args.n_layers,
        'dropout': 0.0, # No dropout for viz
        'heads': args.heads,
        'num_nodes': qwen_emb.shape[0],
        'num_features_dim': num_feat_dim
    }
    gnn = build_gnn(args.gnn_type, gnn_config).to(device)
    
    # Fusion
    fusion = CrossAttentionFusion(
        lm_dim=qwen_emb.shape[1],
        gnn_dim=args.hidden_dim,
        hidden_dim=args.fusion_hidden_dim,
        num_heads=args.fusion_heads,
        dropout=0.0
    ).to(device)

    # 4. Load Checkpoint
    print(f"Loading Checkpoint: {args.checkpoint_path}")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        # Handle cases where checkpoint is nested or direct
        if 'gnn_state_dict' in checkpoint:
            gnn.load_state_dict(checkpoint['gnn_state_dict'])
            fusion.load_state_dict(checkpoint['fusion_state_dict'])
        else:
            print("⚠️ Warning: Checkpoint might be incomplete. Trying to load parts...")
            fusion.load_state_dict(checkpoint) # Fallback
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("Visualization will use RANDOM weights (Noise).")

    # 5. Inference
    gnn.eval()
    fusion.eval()
    
    print("Running Inference...")
    with torch.no_grad():
        # GNN Forward
        if num_features is not None:
             h_gnn = gnn(qwen_emb, edge_index, edge_type, num_features)
        else:
             h_gnn = gnn(qwen_emb, edge_index, edge_type)
        
        # Fusion Forward
        # Returns: logits, z_normalized
        _, z_fused = fusion(qwen_emb, h_gnn)
        
    z_fused = z_fused.cpu().numpy()

    # 6. t-SNE Projection
    print("Running t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
    z_2d = tsne.fit_transform(z_fused)

    # 7. Plotting
    print("Generating Plot...")
    plt.figure(figsize=(12, 10))
    
    # Masks
    human_mask = (labels == 0)
    bot_mask = (labels == 1)
    
    # Scatter Plots
    plt.scatter(z_2d[human_mask, 0], z_2d[human_mask, 1], 
                c='dodgerblue', label='Human', alpha=0.5, s=10, edgecolors='none')
    plt.scatter(z_2d[bot_mask, 0], z_2d[bot_mask, 1], 
                c='crimson', label='Bot', alpha=0.5, s=10, edgecolors='none')
    
    plt.title(f"SeGA Embedding Space (t-SNE)\nCheckpoint: {args.checkpoint_path}", fontsize=15)
    plt.legend(fontsize=12, loc='upper right')
    plt.axis('off') # Remove axes for cleaner look
    
    save_path = "sega_embedding_space.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")
    
    # Optional: Plot Density
    # helps see overlap
    plt.figure(figsize=(12, 10))
    plt.hist2d(z_2d[human_mask, 0], z_2d[human_mask, 1], bins=100, cmap='Blues', alpha=0.6, label='Human Density')
    plt.hist2d(z_2d[bot_mask, 0], z_2d[bot_mask, 1], bins=100, cmap='Reds', alpha=0.4, label='Bot Density')
    plt.title("SeGA Density Map (Overlap Analysis)")
    plt.savefig("sega_density_map.png", dpi=300)
    print("✅ Density map saved to: sega_density_map.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./datasets/TwiBot-20')
    parser.add_argument('--embeddings_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    
    # Model Config (Must match training)
    parser.add_argument('--gnn_type', type=str, default='RGT')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--fusion_hidden_dim', type=int, default=512)
    parser.add_argument('--fusion_heads', type=int, default=4)
    
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    visualize_embedding_space(args)