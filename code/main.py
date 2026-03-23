"""
main.py
Unified entrypoint for:
1) Full LM+Graph training (RACE-Bot-D3F)
2) Standalone Text-only training and experiment matrix
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb

from text_model import TextOnlyClassifier
from text_train import TextTrainer, run_text_experiment_matrix
from text_eval_report import generate_text_eval_report


def parse_args():
    parser = argparse.ArgumentParser(description="RACE-Bot-D3F / Text-only Pipeline")
    parser.add_argument("--dataset_path", type=str, default="../datasets/TwiBot-20/")
    parser.add_argument("--q_final_path", type=str, required=True, help="Path to final-layer embedding .pt")

    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Text-only switches
    parser.add_argument("--text", action="store_true", help="Run standalone text-only classifier")
    parser.add_argument("--text_run_matrix", action="store_true", help="Run text-only experiment matrix (g1-g7)")
    parser.add_argument(
        "--text_loss_mode",
        type=str,
        default="decoupled",
        choices=["plain", "edl", "decoupled", "vib_edl"],
        help="Text-only training loss mode",
    )
    parser.add_argument("--text_lambda_u", type=float, default=0.1, help="EDL weight in decoupled mode")
    parser.add_argument(
        "--text_lambda_u_warmup_epochs",
        type=int,
        default=6,
        help="Warmup epochs for decoupled lambda_u (0 disables warmup)",
    )
    parser.add_argument(
        "--text_eval_prob",
        type=str,
        default="auto",
        choices=["auto", "alpha", "logits"],
        help="Probability source for text-only evaluation",
    )
    parser.add_argument("--text_apply_posthoc", action="store_true", help="Apply post-hoc TS in text-only mode")
    parser.add_argument(
        "--text_calibrators",
        type=str,
        nargs="+",
        default=["ts", "platt", "beta"],
        choices=["ts", "platt", "beta", "isotonic"],
        help="Post-hoc calibrators (binary primary protocol defaults to ts/platt/beta; isotonic is appendix-only by default)",
    )
    parser.add_argument(
        "--text_num_seeds",
        type=int,
        default=5,
        help="Number of matrix seeds to run",
    )
    parser.add_argument(
        "--text_seed_start",
        type=int,
        default=42,
        help="Start seed for matrix multi-seed runs",
    )
    parser.add_argument(
        "--text_report_slices",
        action="store_true",
        help="Generate stress-slice report in matrix aggregation",
    )
    parser.add_argument(
        "--text_ckpt_metric",
        type=str,
        default="nll",
        choices=["acc", "f1", "score", "nll"],
        help="Metric used to select best text checkpoint",
    )
    parser.add_argument(
        "--text_slice_min_n",
        type=int,
        default=30,
        help="Minimum slice size for a per-seed slice to be considered supported",
    )
    parser.add_argument(
        "--text_slice_min_seed_support",
        type=int,
        default=3,
        help="Minimum number of supported seeds required for a slice to enter main analysis",
    )
    parser.add_argument(
        "--text_detach_sem_for_u",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach semantic logits in decoupled EDL branch to avoid gradient conflict",
    )
    parser.add_argument(
        "--text_early_stop_patience",
        type=int,
        default=5,
        help="Early stop patience in epochs for text-only runs (0 disables)",
    )
    parser.add_argument(
        "--text_min_delta",
        type=float,
        default=1e-4,
        help="Minimum score improvement treated as better checkpoint",
    )
    parser.add_argument(
        "--text_scheduler_patience",
        type=int,
        default=2,
        help="ReduceLROnPlateau patience for text-only runs (0 disables scheduler)",
    )
    parser.add_argument(
        "--text_scheduler_factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau factor for text-only runs",
    )
    parser.add_argument("--text_min_lr", type=float, default=1e-6, help="Minimum LR for text-only scheduler")
    parser.add_argument(
        "--text_vib_latent_dim",
        type=int,
        default=256,
        help="Latent dimension for the VIB-EDL text branch",
    )
    parser.add_argument(
        "--text_vib_warmup_epochs",
        type=int,
        default=5,
        help="Epochs of CE warmup before annealing starts",
    )
    parser.add_argument(
        "--text_vib_anneal_epochs",
        type=int,
        default=10,
        help="Annealing horizon for EDL KL and optional auxiliary penalties",
    )
    parser.add_argument(
        "--text_vib_kl_weight",
        type=float,
        default=1e-4,
        help="Weight on the VIB KL term",
    )
    parser.add_argument(
        "--text_vib_logit_l2",
        type=float,
        default=1e-2,
        help="L2 coeff applied to logits during warmup",
    )
    parser.add_argument(
        "--text_vib_dir_kl",
        type=float,
        default=5e-2,
        help="Dirichlet KL weight used during annealing",
    )
    parser.add_argument(
        "--text_vib_mislead",
        type=float,
        default=0.0,
        help="Optional misleading-evidence penalty weight (annealed, disabled by default)",
    )
    parser.add_argument(
        "--text_vib_avuc",
        type=float,
        default=0.0,
        help="Optional AvUC loss weight (annealed, disabled by default)",
    )
    parser.add_argument(
        "--text_vib_scale_max",
        type=float,
        default=3.0,
        help="Clamp limit for the evidence scale parameter",
    )
    parser.add_argument(
        "--text_vary_seed_by_group",
        action="store_true",
        help="If set, matrix runs use seed+i per group (default: same seed for fair comparison)",
    )
    parser.add_argument(
        "--text_ckpt_dir",
        type=str,
        default="text_only",
        help="Checkpoint subdir under ./saved_models for text-only runs",
    )

    # Full model (fusion) hyper-parameters
    parser.add_argument("--lambda_dep", type=float, default=0.7, help="Dependency prior coefficient (0~1)")
    parser.add_argument("--tau", type=float, default=1.0, help="Temperature smoothing scale")

    # Text branch config
    parser.add_argument("--min_temp", type=float, default=1.0, help="Minimum temperature for logit calibration")
    parser.add_argument("--temp_scale", type=float, default=1.0, help="Scale factor for learned temperature")
    parser.add_argument(
        "--concentration_scale", type=float, default=1.0, help="Scale factor for learned concentration"
    )

    # Full model loss weights
    parser.add_argument("--lambda_text", type=float, default=0.2, help="Text branch auxiliary loss weight")
    parser.add_argument("--lambda_graph", type=float, default=0.2, help="Graph branch auxiliary loss weight")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--neighbor_sizes",
        type=int,
        nargs="+",
        default=[10, 10],
        help="Neighbor sampling sizes per hop (e.g. 10 10)",
    )
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    return parser.parse_args()


def _load_emb(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, torch.Tensor):
        raw = torch.tensor(raw)
    return raw.float()


def _seed_setting_fallback(seed_number):
    random.seed(seed_number)
    os.environ["PYTHONHASHSEED"] = str(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def _torch_load_fallback(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_data_dir(dataset_path: str) -> Path:
    data_dir = Path(dataset_path)
    if data_dir.exists():
        return data_dir
    cand = Path("./datasets") / data_dir.name
    if cand.exists():
        return cand
    cleaned = str(data_dir).replace("./datasets/", "").replace("datasets/", "")
    cand2 = Path("./datasets") / cleaned
    return cand2 if cand2.exists() else data_dir


def _load_raw_data_fallback(dataset_path, use_GNN=True):
    data_dir = _resolve_data_dir(dataset_path)
    train_idx = _torch_load_fallback(data_dir / "train_idx.pt")
    valid_idx = _torch_load_fallback(data_dir / "valid_idx.pt")
    test_idx = _torch_load_fallback(data_dir / "test_idx.pt")
    labels = _torch_load_fallback(data_dir / "labels.pt")

    data_dict = {
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "labels": labels,
    }
    if use_GNN:
        data_dict["edge_index"] = _torch_load_fallback(data_dir / "edge_index.pt")
        edge_type_path = data_dir / "edge_type.pt"
        data_dict["edge_type"] = _torch_load_fallback(edge_type_path) if edge_type_path.exists() else None
    return data_dict


def _compute_directed_structural_features_fallback(edge_index, num_nodes, x=None):
    row = edge_index[0].long()
    col = edge_index[1].long()
    in_deg = torch.bincount(col, minlength=num_nodes).float()
    out_deg = torch.bincount(row, minlength=num_nodes).float()
    total_deg = in_deg + out_deg

    def _znorm_log(deg):
        ld = torch.log1p(deg)
        return ((ld - ld.mean()) / (ld.std() + 1e-6)).unsqueeze(1)

    in_log_degree = _znorm_log(in_deg)
    out_log_degree = _znorm_log(out_deg)
    total_log_degree = _znorm_log(total_deg)

    sym_row = torch.cat([row, col], dim=0)
    sym_col = torch.cat([col, row], dim=0)
    neighbor_deg = total_deg[sym_row]

    sum_deg = torch.zeros(num_nodes, dtype=torch.float)
    sum_deg.index_add_(0, sym_col, neighbor_deg)

    cnt = torch.zeros(num_nodes, dtype=torch.float)
    cnt.index_add_(0, sym_col, torch.ones_like(neighbor_deg))

    mean_deg = sum_deg / cnt.clamp(min=1.0)
    sq_diff = (neighbor_deg - mean_deg[sym_col]) ** 2
    var_sum = torch.zeros(num_nodes, dtype=torch.float)
    var_sum.index_add_(0, sym_col, sq_diff)
    neighbor_deg_var = var_sum / cnt.clamp(min=1.0)

    ndv = torch.log1p(neighbor_deg_var)
    neighbor_deg_var_normed = ((ndv - ndv.mean()) / (ndv.std() + 1e-6)).unsqueeze(1)
    graph_missing = (total_deg == 0).float().unsqueeze(1)
    return torch.cat(
        [
            in_log_degree,
            out_log_degree,
            total_log_degree,
            neighbor_deg_var_normed,
            graph_missing,
        ],
        dim=1,
    )


def main():
    args = parse_args()
    if args.text_run_matrix:
        args.text = True

    # Prefer project utils; fallback only for text-only mode.
    try:
        from utils import (
            compute_directed_structural_features as compute_structural_fn,
            load_raw_data as load_raw_data_fn,
            seed_setting as seed_setting_fn,
        )
    except Exception as e:
        if not args.text:
            raise
        print(f"[Warning] Using text-safe local fallback utils due to import error: {e}")
        compute_structural_fn = _compute_directed_structural_features_fallback
        load_raw_data_fn = _load_raw_data_fallback
        seed_setting_fn = _seed_setting_fallback

    seed_setting_fn(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading final-layer embeddings...")
    q_final = _load_emb(args.q_final_path)
    in_dim = q_final.size(1)
    print(f"Embedding shape: {tuple(q_final.shape)}")

    print("Loading dataset...")
    data_dict = load_raw_data_fn(args.dataset_path, use_GNN=True)
    labels = data_dict["labels"]
    if labels.dim() > 1 and labels.size(1) > 1:
        labels = labels.argmax(dim=1)
    labels = labels.long()

    print("Computing structural features...")
    struct_feats = compute_structural_fn(data_dict["edge_index"], int(q_final.size(0)), q_final)

    text_cfg = {
        "min_temp": args.min_temp,
        "temp_scale": args.temp_scale,
        "concentration_scale": args.concentration_scale,
    }
    if args.text_loss_mode == "vib_edl" or args.text_run_matrix:
        text_cfg.update(
            {
                "vib_latent_dim": args.text_vib_latent_dim,
                "vib_warmup_epochs": args.text_vib_warmup_epochs,
                "vib_anneal_epochs": args.text_vib_anneal_epochs,
                "vib_kl_weight": args.text_vib_kl_weight,
                "vib_logit_l2": args.text_vib_logit_l2,
                "vib_dir_kl": args.text_vib_dir_kl,
                "vib_mislead": args.text_vib_mislead,
                "vib_avuc": args.text_vib_avuc,
                "vib_scale_max": args.text_vib_scale_max,
            }
        )

    cfg = {
        "text": text_cfg,
        "loss": {
            "lambda_text": args.lambda_text,
            "lambda_graph": args.lambda_graph,
        },
    }

    # ----------------------------------------------------------------------
    # Text-only branch
    # ----------------------------------------------------------------------
    if args.text:
        num_classes = int(labels.max().item()) + 1
        text_ckpt_root = Path(f"./saved_models/{args.text_ckpt_dir}")
        text_ckpt_root.mkdir(parents=True, exist_ok=True)

        if not args.disable_wandb and not args.text_run_matrix:
            wandb.init(
                project="RACE-Bot-TextOnly",
                name=f"text_only_sd{args.seed}",
                config=vars(args),
            )

        if args.text_run_matrix:
            num_seeds = int(max(1, args.text_num_seeds))
            seeds = [int(args.text_seed_start + i) for i in range(num_seeds)]
            print(f"Running text matrix for seeds: {seeds}")
            for seed_now in seeds:
                run_text_experiment_matrix(
                    q_final=q_final,
                    labels=labels,
                    data_dict=data_dict,
                    struct_feats=struct_feats,
                    device=device,
                    in_dim=in_dim,
                    hid_dim=args.hidden_dim,
                    num_classes=num_classes,
                    dropout=args.dropout,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    batch_size=args.batch_size,
                    cfg_text=cfg["text"],
                    ckpt_root=text_ckpt_root,
                    seed=seed_now,
                    lambda_u=args.text_lambda_u,
                    same_seed_across_groups=not args.text_vary_seed_by_group,
                    early_stop_patience=args.text_early_stop_patience,
                    min_delta=args.text_min_delta,
                    lambda_u_warmup_epochs=args.text_lambda_u_warmup_epochs,
                    scheduler_patience=args.text_scheduler_patience,
                    scheduler_factor=args.text_scheduler_factor,
                    min_lr=args.text_min_lr,
                    checkpoint_metric=args.text_ckpt_metric,
                    detach_semantics_for_u=args.text_detach_sem_for_u,
                    vib_warmup_epochs=args.text_vib_warmup_epochs,
                    vib_anneal_epochs=args.text_vib_anneal_epochs,
                    vib_kl_weight=args.text_vib_kl_weight,
                    vib_logit_l2=args.text_vib_logit_l2,
                    vib_dir_kl=args.text_vib_dir_kl,
                    vib_mislead=args.text_vib_mislead,
                    vib_avuc=args.text_vib_avuc,
                    vib_scale_max=args.text_vib_scale_max,
                    calibrators=args.text_calibrators,
                    valid_split_seed=seed_now,
                )
            report_paths = generate_text_eval_report(
                ckpt_root=text_ckpt_root,
                seeds=seeds,
                report_slices=args.text_report_slices,
                min_n_slice=args.text_slice_min_n,
                min_seed_support=args.text_slice_min_seed_support,
            )
            print("Aggregated text-only report artifacts:")
            for k, v in report_paths.items():
                print(f"  {k}: {v}")
            return

        text_model_kwargs = {
            "in_dim": in_dim,
            "hid_dim": args.hidden_dim,
            "num_classes": num_classes,
            "dropout": args.dropout,
            "text_branch_type": "vib_edl" if args.text_loss_mode == "vib_edl" else "semantic",
        }
        if args.text_loss_mode == "vib_edl":
            text_model_kwargs["vib_latent_dim"] = args.text_vib_latent_dim
        model = TextOnlyClassifier(**text_model_kwargs).to(device)

        text_ckpt = text_ckpt_root / f"best_text_seed{args.seed}.pt"
        trainer_kwargs = {
            "q_final": q_final,
            "labels": labels,
            "data_dict": data_dict,
            "model": model,
            "device": device,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "cfg": {"text": cfg["text"]},
            "ckpt_filepath": str(text_ckpt),
            "struct_feats": struct_feats,
            "loss_mode": args.text_loss_mode,
            "lambda_u": args.text_lambda_u,
            "eval_prob_source": args.text_eval_prob,
            "early_stop_patience": args.text_early_stop_patience,
            "min_delta": args.text_min_delta,
            "lambda_u_warmup_epochs": args.text_lambda_u_warmup_epochs,
            "scheduler_patience": args.text_scheduler_patience,
            "scheduler_factor": args.text_scheduler_factor,
            "min_lr": args.text_min_lr,
            "checkpoint_metric": args.text_ckpt_metric,
            "detach_semantics_for_u": args.text_detach_sem_for_u,
            "valid_split_seed": args.seed,
        }
        if args.text_loss_mode == "vib_edl":
            trainer_kwargs.update(
                {
                    "vib_warmup_epochs": args.text_vib_warmup_epochs,
                    "vib_anneal_epochs": args.text_vib_anneal_epochs,
                    "vib_kl_weight": args.text_vib_kl_weight,
                    "vib_logit_l2": args.text_vib_logit_l2,
                    "vib_dir_kl": args.text_vib_dir_kl,
                    "vib_mislead": args.text_vib_mislead,
                    "vib_avuc": args.text_vib_avuc,
                    "vib_scale_max": args.text_vib_scale_max,
                }
            )
        trainer = TextTrainer(**trainer_kwargs)
        trainer.train(
            run_posthoc=args.text_apply_posthoc,
            save_per_node=True,
            calibrators=args.text_calibrators if args.text_apply_posthoc else [],
        )
        return

    # ----------------------------------------------------------------------
    # Full LM+Graph branch
    # ----------------------------------------------------------------------
    from model import RACEBotD3F
    from train import RACEBotTrainer
    from torch_geometric.data import Data

    data = Data(
        x=q_final,
        q_final=q_final,
        y=labels,
        edge_index=data_dict["edge_index"],
        edge_type=data_dict.get("edge_type", None),
        struct_feats=struct_feats,
    )

    num_relations = 1
    if "edge_type" in data_dict and data_dict["edge_type"] is not None:
        num_relations = int(data_dict["edge_type"].max().item()) + 1

    model = RACEBotD3F(
        in_dim=in_dim,
        hid_dim=args.hidden_dim,
        num_classes=int(labels.max().item()) + 1,
        num_relations=num_relations,
        dropout=args.dropout,
        lambda_dep=args.lambda_dep,
        tau=args.tau,
    ).to(device)

    model.fusion.lambda_dep.data = torch.tensor(args.lambda_dep, device=device)
    model.fusion.tau.data = torch.tensor(args.tau, device=device)

    if not args.disable_wandb:
        wandb.init(
            project="RACE-Bot-D3F",
            name=f"end2end_sd{args.seed}",
            config=vars(args),
        )

    out_ckpt = Path(f"./saved_models/{args.ckpt_dir}/best_RACEBot_seed{args.seed}.pt")
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

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
        neighbor_sizes=args.neighbor_sizes,
    )
    trainer.train()


if __name__ == "__main__":
    main()
