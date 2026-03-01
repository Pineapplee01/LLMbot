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
from utils import relation_aware_knn_pruning, seed_setting, load_raw_data, load_weights, compute_feature_homophily
from GNNs import build_gnn, RGCN, RGT, SimpleHGN, HGT
from model import ReliabilityAwareFusion, EvidentialGraphHead
from train import QwenPrecomputedTrainer
from torch_geometric.utils import homophily as pyg_homophily

def parse_args():
    parser = argparse.ArgumentParser(description='Train SeGA with Precomputed Qwen3 Embeddings')
    
    # Data
    parser.add_argument('--dataset', type=str, default='TwiBot-20',
                        choices=['TwiBot-20', 'Cresci-2015', 'Cresci-2017', 'Midterm-2018'])
    parser.add_argument('--data_loader', type=str, default='neighbor',
                        choices=['random', 'neighbor'])
    parser.add_argument('--batch_size',type=int, default=1024)
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
    parser.add_argument('--fusion_hidden_dim', type=int, default=512)
    parser.add_argument('--fusion_heads', type=int, default=4)
    parser.add_argument('--fusion_dropout', type=float, default=0.1)
    
    # Training Config
    parser.add_argument('--pretrain_gnn',action='store_true', default=False,
                        help = "Whether to run  GNN Pre-training")
    parser.add_argument('--pretrain_llm', action='store_true', default=False,
                        help='Whether to run Text Expert (VIB) Pre-training')
    parser.add_argument('--pretrain_gate', action='store_true', default=False,
                        help='Whether to run Gating Network Pre-training')
    parser.add_argument('--fusion', action='store_true', default=False,
                        help='Whether to run Joint Fusion Training')
    
    parser.add_argument('--pretrain_epochs', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=100,
                        help='Epochs for Stage 2 (Fusion)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--eval_patience', type=int, default=20)
    parser.add_argument('--ablation', action='store_true', default=False)
    parser.add_argument('--lambda_avuc', type=float, default=0.1, help='Weight for AvUC calibration loss')
    parser.add_argument('--lambda_struct', type=float, default=0.1, help='Weight for structural consistency loss')
    
    # SupCon Config
    parser.add_argument('--supcon_temp', type=float, default=0.1)
    parser.add_argument('--lambda_supcon', type=float, default=0.1)

    # System
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, default='SeGA_Qwen3')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints',
                        help='Path to load checkpoint from (if any)')
    parser.add_argument('--wandb_project', type=str, default='Uncertainty_Gated_Fusion')
    parser.add_argument('--error_capture', action='store_true', default=False)
    
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
    fusion_model = ReliabilityAwareFusion(
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
    
    
    # 2. Load Embedding
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

    # 3.A Pre-Process: Compute Homophily Feature
    print("[Pre-Process] Computing Homophily...")
    if 'homophily' not in data_dict:
        print("[Pre-Process] Computing Feature Homophily...")
        # 这是一个计算密集型操作，确保在 GPU 上算完转回 CPU 或者直接存起来
        # 这里假设 utils.compute_feature_homophily 可用
        from utils import compute_feature_homophily
        edge_index = data_dict['edge_index']
        # 为了计算同质性，我们需要把 embeddings 临时移到 device (如果显存够) 或者分块
        # 简单起见，这里用部分节点计算或者在 CPU 计算
        h_score = compute_feature_homophily(precomputed_embeddings, edge_index)
        data_dict['homophily'] = h_score
        print(f"✅ Homophily computed. Shape: {h_score.shape}")


    if 'num_properties' in data_dict and data_dict['num_properties'] is not None:
        num_features = torch.log1p(data_dict['num_properties'])
        mean = num_features.mean(dim=0, keepdim=True)
        std = num_features.std(dim=0, keepdim=True) + 1e-6
        data_dict['num_properties'] = (num_features - mean) / std
        args.num_features_dim = num_features.shape[1]
    else:
        args.num_features_dim = 0
    
    # FIX: Calculate num_relations for RGCN
    if 'edge_type' in data_dict and data_dict['edge_type'] is not None:
        num_relations = int(data_dict['edge_type'].max().item()) + 1
    else:
        num_relations = 1 # Default for GAT/GCN
    
    print(f"Graph Info: {num_nodes} nodes, {num_relations} relation types")


    # wandb
    wandb.init(project=args.wandb_project, name=f"{args.exp_name}_seed{args.seed}", config=vars(args))

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
    
    
    fusion_model = ReliabilityAwareFusion(
        lm_dim=embedding_dim,
        gnn_dim=args.hidden_dim,
        hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout
    ).to(device)

    # 7. Trainer Setup
    ckpt_dir = Path(f"./saved_models/{args.ckpt_dir}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_text = ckpt_dir / f"best_text_vib.pt"
    ckpt_gnn = ckpt_dir / f"best_gnn_seed{args.seed}.pt"
    ckpt_gate = ckpt_dir / f"best_gate_seed{args.seed}.pt"
    ckpt_fusion = ckpt_dir / f"best_model_seed{args.seed}.pt"

    if args.pretrain_gate or args.error_capture or args.fusion:
        print("\n🔧 [Weight Surgery] Transplanting Expert Weights...")

        # (A) 移植 Text Expert (VIB)
        if ckpt_text.exists():
            print(f"  -> Loading Text from {ckpt_text}")
            state = torch.load(ckpt_text, map_location=device)
            # 使用智能加载器解决 vib. 前缀问题
            load_weights(fusion_model.vib, state, "Text VIB")
        else:
            print("  ⚠️ Text checkpoint not found! Text branch will be RANDOM.")

        # --- B. 加载 GNN Expert (Backbone + Head) ---
        if ckpt_gnn.exists():
            print(f"  -> Loading GNN from {ckpt_gnn}")
            state = torch.load(ckpt_gnn, map_location=device)
            gnn_state = state['gnn_state_dict'] if 'gnn_state_dict' in state else state
            
            # 1. 加载骨干 (GNN Backbone)
            # 过滤掉分类头参数，只留骨干
            backbone_dict = {k: v for k, v in gnn_state.items() if 'classifier' not in k and 'head' not in k}
            load_weights(gnn_model, backbone_dict, "GNN Backbone")

            # 2. 移植分类头 (Head Transplant)
            # 将 Stage 2 的 classifier 映射到 Fusion 的 evidence_head
            proj_dict = {}
            head_dict = {}
            
            print("  -> Transplanting GNN Head...")
            for k, v in gnn_state.items():
                # 映射规则：classifier.0 -> gnn_proj (512->256)
                if 'classifier.0.weight' in k: proj_dict['0.weight'] = v
                elif 'classifier.0.bias' in k: proj_dict['0.bias'] = v
                
                # 映射规则：classifier.3 -> evidence_head (256->2)
                # 注意：可能是 classifier.3 或 classifier.1，根据日志适配
                elif 'classifier' in k and ('weight' in k or 'bias' in k) and '.0.' not in k:
                    # 这是一个简单的启发式规则：除了 .0. 之外的权重层通常是最后一层
                    clean_k = k.split('.')[-1]
                    head_dict[f'evidence_layer.{clean_k}'] = v
                
                # 映射 EDL Scale
                elif 'evidence_scale' in k:
                    head_dict['evidence_scale'] = v
            
            if proj_dict:
                fusion_model.gnn_proj.load_state_dict(proj_dict, strict=False)
                print("    ✅ GNN Projection Layer loaded.")
            if head_dict:
                fusion_model.gnn_evidence_head.load_state_dict(head_dict, strict=False)
                print("    ✅ GNN Evidence Head loaded.")
        else:
            print("  ⚠️ GNN checkpoint not found!")
        
    trainer = QwenPrecomputedTrainer(
        precomputed_embeddings=precomputed_embeddings,
        gnn_model=gnn_model,
        lambda_avuc=args.lambda_avuc,
        lambda_struct=args.lambda_struct,
        fusion_model=fusion_model, 
        pretrain_gnn=args.pretrain_gnn,
        pretrain_llm=args.pretrain_llm,
        pretrain_gate= args.pretrain_gate,
        data_dict=data_dict,
        device=device,
        epochs=args.pretrain_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,     
        supcon_temp=args.supcon_temp,
        lambda_supcon=args.lambda_supcon,
        pretrain=args.pretrain_gnn,
        ckpt_filepath=str(ckpt_fusion),
        sample = args.sample,
        dataloader=args.data_loader,
        batch_size=args.batch_size,
        metadata=None
    )

    # Text Expert Pre-training  
    if args.pretrain_llm:
        print("\n [PreTrainer] Text Expert Pre-training\n" )
        trainer.ckpt_filepath = ckpt_text # Point to Text ckpt
        trainer.pretrain_text_vib(epochs=args.pretrain_epochs)
    
    #  GNN Pre-training
    if args.pretrain_gnn:
        print("\n [PreTrainer] GNN Pre-training\n" )
        trainer.ckpt_filepath = ckpt_gnn 
        trainer.pretrain_gnn_stage(epochs=args.pretrain_epochs)
        
    # Gating Network Pre-training
    if args.pretrain_gate:
        print("\n [PreTrainer] Gate Warmup \n" )
        trainer.ckpt_filepath = ckpt_gate
        trainer.pretrain_gate_stage(epochs=args.pretrain_epochs)
    
    # Joint Fine-tuning
    if args.fusion:
        print("\n [Trainer] Joint Fine-tuning \n" )
        if ckpt_gate.exists():
            state = torch.load(ckpt_gate, map_location=device)
            if 'fusion_state_dict' in state: state = state['fusion_state_dict']
            trainer.fusion.load_state_dict(state, strict=False)
        
        trainer.ckpt_filepath = ckpt_fusion
        trainer.pretrain_gnn = False # 设为 False 开启全量微调

        for param in trainer.gnn.parameters():
            param.requires_grad = True
        for param in trainer.fusion.parameters():
            param.requires_grad = True

        trainer.optimizer = torch.optim.AdamW(
            list(trainer.gnn.parameters()) + list(trainer.fusion.parameters()),
            lr=args.lr * 0.5, # Lower LR for fine-tuning
            weight_decay=args.weight_decay
        )

        best_f1 = trainer.train()

        # Also write a fusion-only checkpoint that fusion_diagnose.py can load directly.
        fusion_ckpt = {
            'fusion_state_dict': trainer.fusion.state_dict(),
            'gnn_state_dict': trainer.gnn.state_dict(),
            'meta': {'seed': args.seed, 'best_val_f1': float(best_f1)}
        }

        fusion_ckpt_path = ckpt_dir / f"fusion_best_seed{args.seed}.pt"
        torch.save(fusion_ckpt, fusion_ckpt_path)
        print(f"Saved fusion checkpoint for diagnosis to: {fusion_ckpt_path}")

    if args.error_capture:

        if ckpt_fusion.exists():
            trainer.load_checkpoint(ckpt_fusion)

        for split in ['train', 'val', 'test']:

            print(f"\n[Bad Case Study] Diagnosing errors for split: {split}")
            df_diagnosis = trainer.diagnose_errors(split=split)
            
            full_csv_path = ckpt_dir / f"diagnosis_{split}_seed{args.seed}.csv"
            df_diagnosis.to_csv(full_csv_path, index=False)
            
            # 筛选错误样本
            target_col = 'is_correct'
            if target_col in df_diagnosis.columns:
                df_errors = df_diagnosis[df_diagnosis[target_col] == False].copy()
                if 'gate' in df_errors.columns:
                    df_errors = df_errors.sort_values(by='gate', ascending=False)
                
                error_csv_path = ckpt_dir / f"bad_cases_{split}_seed{args.seed}.csv"
                df_errors.to_csv(error_csv_path, index=False)
                print(f"❌ Found {len(df_errors)} errors. Saved to {error_csv_path}")
        
    # Abation Study
    if args.ablation:

        print("Loading best checkpoint for Ablation Study...")
        checkpoint = torch.load(f"{args.ckpt_dir}/best_model_seed{args.seed}.pt")
        trainer.gnn.load_state_dict(checkpoint['gnn_state_dict'])
        trainer.fusion.load_state_dict(checkpoint['fusion_state_dict'])
        run_ablation_study(trainer, data_dict, device)

if __name__ == '__main__':
    main()