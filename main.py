"""
main.py
RACE-Bot-D3F: Risk-Aware Calibrated Evidence for Social Bot Detection
with Dependency-Discounted Directional Fusion

Entry point: loads final-layer embeddings + graph data, computes structural
features, then trains RACEBotD3F end-to-end.
"""

import argparse
from pathlib import Path
import torch
import wandb
from torch_geometric.data import Data

from model import RACEBotD3F
from train import RACEBotTrainer
from utils import load_raw_data, seed_setting, compute_directed_structural_features


def parse_args():
    parser = argparse.ArgumentParser(description='RACE-Bot-D3F Training Pipeline')
    parser.add_argument('--dataset_path', type=str, default="../datasets/TwiBot-20/")

    # Embedding (final layer only)
    parser.add_argument('--q_final_path', type=str, required=True,
                        help="Path to final-layer embedding .pt file")

    # Model
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # D3F Fusion hyperparameters
    parser.add_argument('--lambda_dep', type=float, default=0.7,
                        help="Dependency prior coefficient (0~1)")
    parser.add_argument('--tau', type=float, default=1.0,
                        help="Temperature smoothing scale")

    # Text branch config (TextFinalSemanticHead)
    parser.add_argument('--min_temp', type=float, default=1.0,
                        help="Minimum temperature for logit calibration")
    parser.add_argument('--temp_scale', type=float, default=1.0,
                        help="Scale factor for learned temperature")
    parser.add_argument('--concentration_scale', type=float, default=1.0,
                        help="Scale factor for learned concentration")

    # Loss weights
    parser.add_argument('--lambda_text', type=float, default=0.2,
                        help="Text branch auxiliary loss weight")
    parser.add_argument('--lambda_graph', type=float, default=0.2,
                        help="Graph branch auxiliary loss weight")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    return parser.parse_args()


def _load_emb(path):
    raw = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(raw, torch.Tensor):
        raw = torch.tensor(raw)
    return raw.float()


def _precompute_structural_features(q_final, edge_index):
    """
    Compute 5-dim per-node structural features (direction-aware).
    Delegates to utils.compute_directed_structural_features.
    """
    num_nodes = q_final.size(0)
    return compute_directed_structural_features(edge_index, num_nodes, q_final)


def main():
    args = parse_args()
    seed_setting(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ── 1. Load final-layer embedding ──
    print("📥 Loading Qwen3 Final-Layer Embeddings...")
    q_final = _load_emb(args.q_final_path)
    in_dim = q_final.size(1)
    num_nodes = q_final.size(0)
    print(f"   Shape: {q_final.shape}")

    # ── 2. Load graph data ──
    print("📥 Loading Graph Data...")
    data_dict = load_raw_data(args.dataset_path, use_GNN=True)
    labels = data_dict['labels']
    if labels.dim() > 1 and labels.size(1) > 1:
        labels = labels.argmax(dim=1)
    labels = labels.long()

    # ── 3. Pre-compute structural features ──
    print("Computing Structural Features...")
    struct_feats = _precompute_structural_features(q_final, data_dict['edge_index'])

    # ── 4. Assemble PyG Data object ──
    data = Data(
        x=q_final,
        q_final=q_final,
        y=labels,
        edge_index=data_dict['edge_index'],
        edge_type=data_dict.get('edge_type', None),
        struct_feats=struct_feats,
    )

    # ── 5. Config dict ──
    cfg = {
        'text': {
            'min_temp': args.min_temp,
            'temp_scale': args.temp_scale,
            'concentration_scale': args.concentration_scale,
        },
        'loss': {
            'lambda_text': args.lambda_text,
            'lambda_graph': args.lambda_graph,
        },
    }

    # ── 6. Build model ──
    num_relations = 1
    if 'edge_type' in data_dict and data_dict['edge_type'] is not None:
        num_relations = int(data_dict['edge_type'].max().item()) + 1

    print(f"🚀 Initializing RACE-Bot-D3F (in_dim={in_dim}, "
          f"hid={args.hidden_dim}, relations={num_relations})...")

    model = RACEBotD3F(
        in_dim=in_dim,
        hid_dim=args.hidden_dim,
        num_classes=2,
        num_relations=num_relations,
        dropout=args.dropout,
        lambda_dep=args.lambda_dep,
        tau=args.tau,
    ).to(device)

    # Explicitly set D3F hyperparameters
    model.fusion.lambda_dep.data = torch.tensor(args.lambda_dep, device=device)
    model.fusion.tau.data = torch.tensor(args.tau, device=device)

    # ── 7. WandB ──
    wandb.init(
        project="RACE-Bot-D3F",
        name=f"end2end_sd{args.seed}",
        config=vars(args),
    )

    # ── 8. Train ──
    out_ckpt = Path(f'./saved_models/{args.ckpt_dir}/best_RACEBot_seed{args.seed}.pt')
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    print("🔥 Starting End-to-End Training Phase...")
    trainer = RACEBotTrainer(
        data=data,
        data_dict=data_dict,
        model=model,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        cfg=cfg,
        ckpt_filepath=str(out_ckpt),
    )

    trainer.train()


if __name__ == '__main__':
    main()