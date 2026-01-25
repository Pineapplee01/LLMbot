"""
Main script for training with Precomputed Qwen3-Embedding-8B embeddings
Structure-Enhanced Graph Attention (SeGA) Training
"""

import wandb
import torch
import argparse
import sys
import os
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

# Ensure we can import from local modules
from utils import relation_aware_knn_pruning, seed_setting, load_raw_data, BadCaseAnalyzer
from GNNs import build_gnn, RGCN, RGT, SimpleHGN, HGT
from AttentionFusion import DegreeAwareConfidenceFusion,DisentangledMetaFusion
from QwenPrecomputedTrainer import QwenPrecomputedTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train SeGA with Precomputed Qwen3 Embeddings')
    
    # Data
    parser.add_argument('--dataset', type=str, default='TwiBot-20',
                        choices=['TwiBot-20', 'Cresci-2015', 'Cresci-2017', 'Midterm-2018'])
    parser.add_argument('--dataset_path', type=str, default='./datasets/TwiBot-20',
                        help='Path to dataset directory')
    parser.add_argument('--embeddings_path', type=str, 
                        default='./datasets/TwiBot-20/qwen3_emb_last.pt',
                        help='Path to precomputed embeddings .pt file')
    parser.add_argument('--num_features_dim', type=int, default=0, 
                        help='Dimensionality of numerical features (0 if none)')
    
    # GNN Config
    parser.add_argument('--sample', type=str, default='random',
                        choices=['random', 'hard'])
    parser.add_argument('--gnn_type', type=str, default='RGT',
                        choices=['RGCN', 'RGT', 'SimpleHGN', 'HGT'])
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=512) 
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--pruning', action='store_true', default=False, help='Whether to apply neighbor pruning')
    parser.add_argument('--neighbor', type=int, default=16,
                        help='Number of neighbors to sample per node (for hard sampling)')

    # Fusion Config
    parser.add_argument('--fusion_hidden_dim', type=int, default=256)
    parser.add_argument('--fusion_heads', type=int, default=4)
    parser.add_argument('--fusion_dropout', type=float, default=0.1)
    
    # Training Config
    parser.add_argument('--pretrain',action='store_true', default=False,
                        help = "Whether to run Stage 1: GNN Pre-training")
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100,
                        help='Epochs for Stage 2 (Fusion)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--eval_patience', type=int, default=20)
    parser.add_argument('--ablation', action='store_true', default=False)
    
    # SupCon Config
    parser.add_argument('--supcon_temp', type=float, default=0.1)
    parser.add_argument('--lambda_supcon', type=float, default=0.1)

    # System
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, default='SeGA_Qwen3')
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='SeGA-Experiment')
    parser.add_argument('--error_capture', action='store_true', default=True)
    
    return parser.parse_args()

def run_ablation_study(trainer, data_dict, device):
    """
    Evaluates the model in 3 modes:
    1. Full Fusion (Normal)
    2. Text Only (Mask out Graph Signal)
    3. Graph Only (Mask out Text Signal)
    """
    print("\n" + "="*60)
    print("🔬 STARTING MODALITY ABLATION STUDY")
    print("="*60)
    
    trainer.model.eval()
    
    # Get Test Data
    mask = data_dict['test_mask']
    labels = data_dict['labels'][mask].cpu().numpy()
    if labels.ndim > 1: labels = labels.argmax(1) # Handle one-hot
    
    # Prepare Inputs
    qwen_emb = trainer.embeddings.to(device)
    edge_index = data_dict['edge_index'].to(device)
    edge_type = data_dict['edge_type'].to(device) if 'edge_type' in data_dict else None
    # Use the normalized features we fixed in Part 1
    num_features = data_dict['num_properties'].to(device) if 'num_properties' in data_dict else None

    # Helper for Inference
    def get_preds(mask_text=False, mask_graph=False):
        with torch.no_grad():
            # 1. Get Graph Embeddings (RGT Output)
            # If we mask GRAPH, we just feed zeros into fusion, 
            # but we still need to run GNN or create dummy shape.
            # Efficient way: Run GNN, then zero out result if needed.
            
            h_gnn = trainer.gnn(qwen_emb, edge_index, edge_type, num_features)
            
            # 2. Apply Masking Logic
            h_gnn_in = torch.zeros( (qwen_emb.size(0), trainer.gnn.out_dim),device=device) if mask_graph else h_gnn
            qwen_in = torch.zeros((qwen_emb.size(0), qwen_emb.size(1)), device=device) if mask_text else qwen_emb
            
            # 3. Fusion
            logits, _ = trainer.fusion(qwen_in, h_gnn_in)
            
            # 4. Metrics
            preds = logits[mask].argmax(dim=1).cpu().numpy()
            return f1_score(labels, preds, average='macro'), accuracy_score(labels, preds)


    # Mode 1: Full Model (Baseline)
    f1_full, acc_full = get_preds(mask_text=False, mask_graph=False)
    print(f"1. [Full SeGA]      F1: {f1_full:.4f} | Acc: {acc_full:.4f}")

    # Mode 2: Text Only (Kill Graph)
    f1_text, acc_text = get_preds(mask_text=False, mask_graph=True)
    delta_text = f1_text - f1_full
    print(f"2. [Text Only]      F1: {f1_text:.4f} | Acc: {acc_text:.4f} | Drop: {delta_text:.4f}")

    # Mode 3: Graph Only (Kill Text)
    f1_graph, acc_graph = get_preds(mask_text=True, mask_graph=False)
    delta_graph = f1_graph - f1_full
    print(f"3. [Graph Only]     F1: {f1_graph:.4f} | Acc: {acc_graph:.4f} | Drop: {delta_graph:.4f}")

def build_models(args, embedding_dim, num_relations, num_nodes):
    """
    Constructs the GNN and Fusion modules.
    """
    print(f"[Init] Building Models (LM Dim: {embedding_dim}, Relations: {num_relations})...")
    
    # 1. GNN Construction
    # Note: RGCN input dim is usually the hidden_dim if we project first, 
    # OR embedding_dim if we pass raw features.
    # In SeGA, we pass raw 4096-dim features to the GNN, so in_channels = embedding_dim.
    gnn_config = {
        'lm_input_dim': embedding_dim,
        'gnn_hidden_dim': args.hidden_dim, # e.g. 512
        'n_relations': num_relations,
        'gnn_n_layers': args.n_layers,
        'dropout': args.dropout,
        'heads': args.heads
    }
    
    gnn_model = build_gnn(args.gnn_type, gnn_config)
    
    # 2. Fusion Construction (SeGA)
    # We instantiate the class directly to ensure we use the Corrected Version
    fusion_model = DisentangledMetaFusion(
        lm_dim=embedding_dim,       # 4096
        gnn_dim=args.hidden_dim,    # 256 (Output of GNN)
        hidden_dim=args.fusion_hidden_dim, # 256
        dropout=args.fusion_dropout
    )
    
    return gnn_model, fusion_model


def main():
    
    args = parse_args()
    seed_setting(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Load Embedding
    print(f"Loading precomputed embeddings from {args.embeddings_path}...")
    precomputed_embeddings = torch.load(args.embeddings_path, map_location='cpu', weights_only=False)

    # Convert to tensor if numpy
    if not isinstance(precomputed_embeddings, torch.Tensor):
        precomputed_embeddings = torch.tensor(precomputed_embeddings)
    
    precomputed_embeddings = precomputed_embeddings.float()
    
    embedding_dim = precomputed_embeddings.shape[1]
    num_nodes = precomputed_embeddings.shape[0]
    print(f"Loaded embeddings: {precomputed_embeddings.shape}")

    # 3. Load Graph Data
    print(f"Loading graph data from {args.dataset_path}...")
    data_dict = load_raw_data(args.dataset_path, use_GNN=True)


    if 'num_properties' in data_dict and data_dict['num_properties'] is not None:
        print(f"[Pre-Process] Raw Numerical Features Mean: {data_dict['num_properties'].mean(dim=0)}")
        
        # A. Log-Transform: Squash power-law distribution (e.g., 10k followers vs 0)
        # log1p(x) = log(x + 1) to handle zeros safely
        num_features = torch.log1p(data_dict['num_properties'])
        

        # B. Z-Score Standardization: Center at 0, Scale to 1
        # This prevents gradients from exploding in the RGT
        mean = num_features.mean(dim=0, keepdim=True)
        std = num_features.std(dim=0, keepdim=True) + 1e-6 # Epsilon for stability
        num_features = (num_features - mean) / std
        
        # C. Update Data Dict in place
        data_dict['num_properties'] = num_features
        args.num_features_dim = num_features.shape[1]
        print(f"[Pre-Process] ✅ Log-Normalized & Standardized. New Mean ~0, Std ~1")
    
    else:
        print("[Pre-Process] ⚠️ No numerical properties found.")
        args.num_features_dim = 0
    
    # FIX: Calculate num_relations for RGCN
    if 'edge_type' in data_dict and data_dict['edge_type'] is not None:
        num_relations = int(data_dict['edge_type'].max().item()) + 1
    else:
        num_relations = 1 # Default for GAT/GCN
    
    print(f"Graph Info: {num_nodes} nodes, {num_relations} relation types")

    if args.pruning:

        print(f"[Pre-Process] Applying Relation-Aware KNN Pruning with k={args.neighbor}...")
        data_dict = relation_aware_knn_pruning(
            data_dict, 
            precomputed_embeddings, 
            k=args.neighbor
        )


    # 4. Build Models
    gnn_config = {
        'lm_input_dim': embedding_dim,       # 4096 from Qwen
        'gnn_hidden_dim': args.hidden_dim,   # 512
        'n_relations': num_relations,        # Calculated dynamically
        'gnn_n_layers': args.n_layers,       # 2
        'dropout': args.dropout,             # 0.3
        'heads': args.heads,                 # 4
        'num_features_dim': args.num_features_dim # Numerical features dim
    }
    
    # Correctly pass the string type and the config dict
    gnn_model = build_gnn(args.gnn_type, gnn_config)
    gnn_model = gnn_model.to(device) 
    
    
    fusion_model = DisentangledMetaFusion(
        lm_dim=embedding_dim,
        gnn_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    # 5. Model Statistics
    gnn_params = sum(p.numel() for p in gnn_model.parameters())
    fusion_params = sum(p.numel() for p in fusion_model.parameters())
    print(f"\nModel Parameters:")
    print(f"  GNN ({args.gnn_type}): {gnn_params:,}")
    print(f"  SeGA Fusion:       {fusion_params:,}")
    print(f"  Total:             {gnn_params + fusion_params:,}")

    # 6. WandB
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=f"{args.exp_name}_seed{args.seed}", config=vars(args))

    # 7. Trainer Setup
    ckpt_dir = Path(f"./saved_models/{args.exp_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_filepath = ckpt_dir / f"best_model_seed{args.seed}.pt"

    # Stage1: GNN pretrain
    if args.pretrain:
        print("\n")
        print(f"[Stage 1] Starting GNN Pre-training for {args.pretrain_epochs} epochs...")
        
        pretrain_ckpt_path = os.path.join(ckpt_dir, f"pretrain_gnn_seed{args.seed}.pt")
        
        # Init Trainer with stage='pretrain'
        pretrainer = QwenPrecomputedTrainer(
            precomputed_embeddings=precomputed_embeddings,
            gnn_model=gnn_model,
            fusion_model=fusion_model, # Passed but frozen inside
            data_dict=data_dict,
            device=device,
            epochs=args.pretrain_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            ckpt_filepath=pretrain_ckpt_path,        
            supcon_temp=args.supcon_temp,
            lambda_supcon=args.lambda_supcon,
            pretrain=args.pretrain,
            sample = args.sample,
            metadata=None
        )
        
        pretrainer.train()
        
        # [CRITICAL] Load best GNN weights before Stage 2
        print(f"✅ Stage 1 Finished. Loading best GNN weights from {pretrain_ckpt_path}")
        checkpoint = torch.load(pretrain_ckpt_path, map_location=device)
        gnn_model.load_state_dict(checkpoint['gnn_state_dict'])
        
        # Free memory
        del pretrainer
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Stage2: Fusion Training
    print("\n")
    print(f"[Stage 2] Starting SeGA Fusion Training for {args.epochs} epochs...")

    best_ckpt_path = os.path.join(ckpt_dir, f"best_model_seed{args.seed}.pt")

    trainer = QwenPrecomputedTrainer(
        precomputed_embeddings=precomputed_embeddings,
        gnn_model=gnn_model,
        fusion_model=fusion_model,
        data_dict=data_dict,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ckpt_filepath=best_ckpt_path,
        supcon_temp=args.supcon_temp,
        lambda_supcon=args.lambda_supcon,
        pretrain= False,
        metadata= None,
        sample= args.sample
    )
    
    # 8. Start Training
    best_f1 = trainer.train()

    # Ensure best checkpoint exists (trainer may already have saved during training)
    # Save again to ensure file present and consistent
    try:
        trainer.save_checkpoint()
    except Exception as e:
        print(f"[Warning] save_checkpoint() raised: {e}")


    # Also write a fusion-only checkpoint that fusion_diagnose.py can load directly.
    fusion_ckpt = {
        'fusion_state_dict': trainer.fusion.state_dict(),
        'gnn_state_dict': trainer.gnn.state_dict(),
        'meta': {
            'exp_name': args.exp_name,
            'seed': args.seed,
            'best_val_f1': float(best_f1)
       }
    }
    fusion_ckpt_path = ckpt_dir / f"fusion_best_seed{args.seed}.pt"
    torch.save(fusion_ckpt, fusion_ckpt_path)
    print(f"Saved fusion checkpoint for diagnosis to: {fusion_ckpt_path}")


    if args.error_capture:

        print("[Bad Case Study] STARTING FULL DATASET DIAGNOSIS")

        trainer.load_checkpoint(best_ckpt_path)
        splits = ['train', 'val', 'test']

        for split in splits:

            print(f"[Bad Case Study] Diagnosing errors for split: {split}")
            df_diagnosis = trainer.diagnose_errors(split=split)

            full_csv_path = ckpt_dir / f"diagnosis_{split}_seed{args.seed}.csv"
            df_diagnosis.to_csv(full_csv_path, index=False)
            print(f"[Bad Case Study] Full Diagnosis saved to {full_csv_path}")

            # 按置信度排序
            df_errors = df_diagnosis[df_diagnosis['is_correct_full'] == False].copy()
            df_errors = df_errors.sort_values(by='conf_full', ascending=False)  # pyright: ignore[reportCallIssue]
            
            error_csv_path = ckpt_dir / f"bad_cases_{split}_seed{args.seed}.csv"
            df_errors.to_csv(error_csv_path, index=False)

            print(f"\n [Bad Case Study] Found {len(df_errors)} error samples, list saved to {error_csv_path}")
        


    # Abation Study
    if args.ablation:

        print("Loading best checkpoint for Ablation Study...")
        checkpoint = torch.load(f"{args.ckpt_dir}/best_model_seed{args.seed}.pt")
        trainer.gnn.load_state_dict(checkpoint['gnn_state_dict'])
        trainer.fusion.load_state_dict(checkpoint['fusion_state_dict'])
        
        # RUN ABLATION
        run_ablation_study(trainer, data_dict, device)

if __name__ == '__main__':
    main()