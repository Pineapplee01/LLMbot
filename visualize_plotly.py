import torch
import torch.nn.functional as F
import numpy as np
import plotly.express as px
import pandas as pd
import argparse
import sys
import time

# Metrics
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Import your project modules
from utils import load_raw_data, seed_setting
from GNNs import build_gnn
from AttentionFusion import CrossAttentionFusion

def calculate_cluster_metrics(embeddings, labels):
    """
    Calculates quantitative metrics to evaluate how well 
    Humans and Bots are separated in the high-dimensional space.
    """
    print("\n--- 📊 Calculating Clustering Metrics (Latent Space) ---")
    
    # 1. KNN Accuracy (Separability)
    # Checks: "If I look at my 5 nearest neighbors, are they the same class as me?"
    knn = KNeighborsClassifier(n_neighbors=5)
    # Use 5-fold cross validation to get a robust accuracy
    knn_acc = cross_val_score(knn, embeddings, labels, cv=5).mean()
    print(f"✅ KNN Classifier Accuracy (k=5): {knn_acc:.4f} (Higher is better)")
    
    # 2. Silhouette Score (Cohesion vs Separation)
    # Checks: "How close am I to my cluster vs the enemy cluster?"
    # Note: Can be slow on huge datasets. TwiBot-20 (11k) is fine.
    try:
        sil = silhouette_score(embeddings, labels, metric='cosine')
        print(f"✅ Silhouette Score (Cosine):    {sil:.4f} (Range -1 to 1)")
    except Exception as e:
        print(f"⚠️  Skipping Silhouette: {e}")

    # 3. Davies-Bouldin Index
    # Ratio of within-cluster scatter to between-cluster separation
    db_index = davies_bouldin_score(embeddings, labels)
    print(f"✅ Davies-Bouldin Index:         {db_index:.4f} (Lower is better)")
    print("-" * 50)
    
    return knn_acc, sil

def visualize_3d(args):
    device = torch.device(args.device)
    seed_setting(args.seed)

    # --- 1. Load Data ---
    print(f"Loading data from {args.dataset_path}...")
    data_dict = load_raw_data(args.dataset_path, use_GNN=True)
    
    edge_index = data_dict['edge_index'].to(device)
    edge_type = data_dict.get('edge_type', None)
    if edge_type is not None: edge_type = edge_type.to(device)
    
    labels = data_dict['labels']
    if labels.dim() > 1: labels = labels.argmax(dim=1)
    labels_np = labels.cpu().numpy()
    
    label_text = ["Human" if l == 0 else "Bot" for l in labels_np]

    # --- 2. Load Embeddings ---
    print(f"Loading embeddings from {args.embeddings_path}...")
    qwen_emb = torch.load(args.embeddings_path, map_location='cpu').float().to(device)
    
    # --- 3. Build Model ---
    num_feat_dim = 0
    if 'num_properties' in data_dict:
        num_feat_dim = data_dict['num_properties'].shape[1]

    gnn_config = {
        'in_channels': qwen_emb.shape[1],
        'hidden_channels': args.hidden_dim,
        'out_channels': args.hidden_dim,
        'num_relations': int(edge_type.max() + 1) if edge_type is not None else 1,
        'n_layers': args.n_layers,
        'dropout': 0.0, 
        'heads': args.heads,
        'num_nodes': qwen_emb.shape[0],
        'num_features_dim': num_feat_dim
    }
    
    gnn = build_gnn(args.gnn_type, gnn_config).to(device)
    fusion = CrossAttentionFusion(
        lm_dim=qwen_emb.shape[1],
        gnn_dim=args.hidden_dim,
        hidden_dim=args.fusion_hidden_dim,
        num_heads=args.fusion_heads,
        dropout=0.0
    ).to(device)

    # --- 4. Load Checkpoint ---
    print(f"Loading Checkpoint: {args.checkpoint_path}")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if 'gnn_state_dict' in checkpoint:
            gnn.load_state_dict(checkpoint['gnn_state_dict'])
            fusion.load_state_dict(checkpoint['fusion_state_dict'])
        else:
            fusion.load_state_dict(checkpoint) 
        print("✅ Weights loaded.")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    # --- 5. Inference ---
    gnn.eval()
    fusion.eval()
    
    with torch.no_grad():
        # Correct Forward Pass (4 args)
        h_gnn = gnn(qwen_emb, edge_index, edge_type)
        _, z_fused = fusion(qwen_emb, h_gnn)
        
    z_fused_np = z_fused.cpu().numpy()

    # --- 6. Calculate Scientific Metrics ---
    # We calculate these on the FULL 512-dim vector, NOT the 3D projection.
    # This tells us if the model actually learned, or if t-SNE is just hallucinating.
    calculate_cluster_metrics(z_fused_np, labels_np)

    # --- 7. 3D t-SNE Projection ---
    print("\nRunning 3D t-SNE (this takes longer than 2D)...")
    tsne = TSNE(n_components=3, perplexity=30, init='pca', learning_rate='auto', random_state=42)
    z_3d = tsne.fit_transform(z_fused_np)

    # --- 8. Generate 3D Plotly ---
    print("Generating 3D Interactive Plot...")
    
    df = pd.DataFrame({
        'x': z_3d[:, 0],
        'y': z_3d[:, 1],
        'z': z_3d[:, 2],
        'Label': label_text,
        'ID': range(len(label_text))
    })

    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='Label',
        color_discrete_map={'Human': 'dodgerblue', 'Bot': 'crimson'},
        hover_data=['ID'],
        title=f"SeGA 3D Embedding Space<br>Model: {args.checkpoint_path}",
        opacity=0.6
    )

    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        legend_title_text='User Class'
    )

    output_file = "sega_3d_space.html"
    fig.write_html(output_file)
    print(f"\n✅ 3D Visualization saved to: {output_file}")
    print("Download and open this file in your browser to rotate/zoom.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./datasets/TwiBot-20')
    parser.add_argument('--embeddings_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    
    # Model Config
    parser.add_argument('--gnn_type', type=str, default='RGT')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--fusion_hidden_dim', type=int, default=512)
    parser.add_argument('--fusion_heads', type=int, default=4)
    
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    visualize_3d(args)