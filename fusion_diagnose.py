import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
from pathlib import Path
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.cluster import KMeans

# Import your project modules
from utils import load_raw_data, seed_setting
from GNNs import build_gnn
from AttentionFusion import CrossAttentionFusion

# ==============================================================================
# 1. DIAGNOSER CLASS
# ==============================================================================
class EmbeddingDiagnoser:
    def __init__(self, data_dict, device='cuda:0'):
        self.device = device
        self.edge_index = data_dict['edge_index'].to(device)
        
        # Handle labels (convert one-hot to indices if needed)
        y = data_dict['labels']
        if y.dim() > 1 and y.shape[1] > 1:
            self.labels = y.argmax(dim=1).cpu().numpy()
        else:
            self.labels = y.cpu().numpy()
            
        # Handle masks/indices
        if 'test_mask' in data_dict:
            self.test_mask = data_dict['test_mask'].cpu().numpy()
        elif 'test_idx' in data_dict:
            # Create boolean mask from indices
            mask = torch.zeros(len(self.labels), dtype=torch.bool)
            mask[data_dict['test_idx']] = True
            self.test_mask = mask.cpu().numpy()
        else:
            print("[Diagnoser] Warning: No test split found. Using all nodes.")
            self.test_mask = np.ones(len(self.labels), dtype=bool)

    def sample_nodes(self, mask, n_samples=5000):
        """Randomly sample nodes to speed up slow metrics like Silhouette."""
        indices = np.where(mask)[0]
        if len(indices) > n_samples:
            return np.random.choice(indices, n_samples, replace=False)
        return indices

    def compute_silhouette(self, embeddings, n_samples=5000):
        """
        Measure of cluster separation. 
        Range: [-1, 1]. High value (>0.4) = Good separation.
        """
        indices = self.sample_nodes(self.test_mask, n_samples)
        emb_subset = embeddings[indices].cpu().numpy()
        labels_subset = self.labels[indices]
        
        try:
            score = silhouette_score(emb_subset, labels_subset)
            return score
        except Exception as e:
            print(f"Warning: Silhouette calculation failed: {e}")
            return 0.0

    def compute_nmi(self, embeddings):
        """
        Normalized Mutual Information (Cluster Purity).
        Range: [0, 1]. High value = Embeddings naturally separate Bots/Humans.
        """
        indices = np.where(self.test_mask)[0]
        emb_subset = embeddings[indices].cpu().numpy()
        labels_subset = self.labels[indices]
        
        # Perform K-Means with K=2 (Bot vs Human)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(emb_subset)
        pred_clusters = kmeans.labels_
        
        return normalized_mutual_info_score(labels_subset, pred_clusters)

    def compute_graph_smoothness(self, embeddings):
        """
        Dirichlet Energy. Measures high-frequency noise on the graph.
        Lower = Smoother. Too High (>1.0) = Noisy/Inconsistent with graph.
        """
        x = F.normalize(embeddings, p=2, dim=-1) # Normalize first
        src, dst = self.edge_index
        
        # Squared Euclidean distance on normalized vectors
        # ||x_i - x_j||^2
        dist = (x[src] - x[dst]).pow(2).sum(dim=-1)
        energy = dist.mean().item()
        return energy

    def compute_neighbor_similarity(self, embeddings):
        """
        Homophily Score (Average Cosine Similarity of neighbors).
        High = Neighbors look similar.
        """
        x = F.normalize(embeddings, p=2, dim=-1)
        src, dst = self.edge_index
        cosine_sim = (x[src] * x[dst]).sum(dim=-1)
        return cosine_sim.mean().item()

# ==============================================================================
# 2. HELPER: REBUILD MODELS
# ==============================================================================
def build_models(args, embedding_dim, num_relations, num_nodes):
    print(f"[Init] Building Models for Eval (LM Dim: {embedding_dim})...")
    
    # GNN Config
    gnn_config = {
        'in_channels': embedding_dim,
        'hidden_channels': args.hidden_dim,
        'out_channels': args.hidden_dim,
        'num_relations': num_relations,
        'n_layers': args.n_layers,
        'dropout': args.dropout,
        'heads': args.heads,
        'num_nodes': num_nodes,
        # Add support for numerical features if your RGT uses them
        'num_features_dim': args.num_features_dim if hasattr(args, 'num_features_dim') else 0
    }
    
    gnn_model = build_gnn(args.gnn_type, gnn_config)
    
    # Fusion Config
    # Check if we are using the simple or optimized Fusion
    # We assume 'CrossAttentionFusion' from your file
    fusion_model = CrossAttentionFusion(
        lm_dim=embedding_dim,
        gnn_dim=args.hidden_dim, # Output of GNN
        hidden_dim=args.fusion_hidden_dim,
        num_heads=args.fusion_heads,
        dropout=args.fusion_dropout
    )
    
    return gnn_model, fusion_model

# ==============================================================================
# 3. MAIN EVALUATION LOOP
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='SeGA Embedding Diagnosis')
    
    # Same args as training to ensure model matches
    parser.add_argument('--dataset_path', type=str, default='./datasets/TwiBot-20')
    parser.add_argument('--embeddings_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to .pt checkpoint')
    
    # Model Config (Must match training!)
    parser.add_argument('--gnn_type', type=str, default='RGT')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    parser.add_argument('--fusion_hidden_dim', type=int, default=512)
    parser.add_argument('--fusion_heads', type=int, default=8)
    parser.add_argument('--fusion_dropout', type=float, default=0.1)
    
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    seed_setting(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 1. Load Data
    print("Loading data...")
    data_dict = load_raw_data(args.dataset_path, use_GNN=True)
    
    # Determine Relations
    if 'edge_type' in data_dict and data_dict['edge_type'] is not None:
        num_relations = int(data_dict['edge_type'].max().item()) + 1
    else:
        num_relations = 1

    # Check for Numerical Features (for Hybrid RGT)
    if 'num_properties' in data_dict:
        args.num_features_dim = data_dict['num_properties'].shape[1]
        num_features = data_dict['num_properties'].to(device)
        print(f"Numerical Features Found: {args.num_features_dim} dims")
    else:
        args.num_features_dim = 0
        num_features = None

    # 2. Load Embeddings
    print(f"Loading embeddings from {args.embeddings_path}...")
    qwen_emb = torch.load(args.embeddings_path, map_location='cpu', weights_only=False)
    
    if not isinstance(qwen_emb, torch.Tensor):
        qwen_emb = torch.tensor(qwen_emb)

    qwen_emb = qwen_emb.float() # Force Float32
    embedding_dim = qwen_emb.shape[1]
    num_nodes = qwen_emb.shape[0]

    # 3. Build & Load Models
    gnn, fusion = build_models(args, embedding_dim, num_relations, num_nodes)
    gnn = gnn.to(device)
    fusion = fusion.to(device)
    
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Handle state dict keys (sometimes saved with 'gnn_state_dict', sometimes direct)
    if 'gnn_state_dict' in checkpoint:
        gnn.load_state_dict(checkpoint['gnn_state_dict'])
        fusion.load_state_dict(checkpoint['fusion_state_dict'])
    else:
        # Assuming the checkpoint is just the fusion model? 
        # Usually we save both. If this fails, check your save structure.
        print("Warning: Checkpoint format unclear. Trying to load keys directly...")
        try:
            fusion.load_state_dict(checkpoint)
        except:
            print("Failed. Ensure checkpoint contains 'gnn_state_dict' and 'fusion_state_dict'.")
            sys.exit(1)

    # 4. Run Inference (Get Latent Representations)
    print("Running Inference to extract latent features...")
    gnn.eval()
    fusion.eval()
    qwen_emb = qwen_emb.to(device)
    edge_index = data_dict['edge_index'].to(device)
    edge_type = data_dict['edge_type'].to(device) if 'edge_type' in data_dict else None
    
    with torch.no_grad():
        # A. GNN Forward
        # Check if GNN accepts num_features
        try:
            h_gnn = gnn(qwen_emb, edge_index, edge_type, num_features)
        except TypeError:
            # Fallback if GNN doesn't support num_features yet
            h_gnn = gnn(qwen_emb, edge_index, edge_type)
            
        # B. Fusion Forward
        # We need h_fused. Ensure AttentionFusion.forward returns (logits, h_fused)
        logits, h_fused = fusion(qwen_emb, h_gnn)

    # 5. Run Diagnosis
    diagnoser = EmbeddingDiagnoser(data_dict, device)
    
    print("\n" + "="*50)
    print(" DIAGNOSIS REPORT")
    print("="*50)
    
    # --- SeGA (Your Model) ---
    print(f"\n[Model: SeGA (Fusion)]")
    sega_sil = diagnoser.compute_silhouette(h_fused)
    sega_nmi = diagnoser.compute_nmi(h_fused)
    sega_smooth = diagnoser.compute_graph_smoothness(h_fused)
    print(f"  Silhouette Score:    {sega_sil:.4f}  (>0.1 is ok, >0.4 is good)")
    print(f"  NMI (Clustering):    {sega_nmi:.4f}  (>0.3 is good)")
    print(f"  Graph Smoothness:    {sega_smooth:.4f} (Lower = Smoother)")
    
    # --- Raw Qwen (Baseline) ---
    print(f"\n[Model: Raw Qwen (Text Only)]")
    # We project Qwen to same dim (512) using the trained projector for fair comparison
    # The projector is inside the Fusion model (lm_proj)
    with torch.no_grad():
        qwen_proj = fusion.lm_proj(qwen_emb)
        
    qwen_sil = diagnoser.compute_silhouette(qwen_proj)
    qwen_nmi = diagnoser.compute_nmi(qwen_proj)
    qwen_smooth = diagnoser.compute_graph_smoothness(qwen_proj)
    print(f"  Silhouette Score:    {qwen_sil:.4f}")
    print(f"  NMI (Clustering):    {qwen_nmi:.4f}")
    print(f"  Graph Smoothness:    {qwen_smooth:.4f}")
    
    # --- GNN Only (Structure Only) ---
    print(f"\n[Model: RGT Only (Structure)]")
    rgt_sil = diagnoser.compute_silhouette(h_gnn)
    rgt_nmi = diagnoser.compute_nmi(h_gnn)
    rgt_smooth = diagnoser.compute_graph_smoothness(h_gnn)
    print(f"  Silhouette Score:    {rgt_sil:.4f}")
    print(f"  NMI (Clustering):    {rgt_nmi:.4f}")
    print(f"  Graph Smoothness:    {rgt_smooth:.4f}")
    
    print("="*50)
    print("Interpretation:")
    if sega_nmi < rgt_nmi:
        print("⚠️  WEAKNESS: SeGA is worse than RGT. Text is diluting the structural signal.")
    if sega_smooth > 1.0:
        print("⚠️  NOISE: SeGA embeddings are too high-frequency (noisy). Needs regularization.")
    if sega_nmi > qwen_nmi and sega_nmi > rgt_nmi:
        print("✅  SUCCESS: SeGA is better than both individual parts!")

if __name__ == "__main__":
    main()