"""Single experiment runner for the rewrite pipeline.

Usage:
    cd g:/Research/LMBot
    python rewrite_pipeline/run.py --seed 42 --feature_source z_sem --rewrite_mode none
    python rewrite_pipeline/run.py --seed 42 --feature_source z_sem --rewrite_mode prune_add
"""
from __future__ import annotations
import argparse, json, sys, time, os
from pathlib import Path

# Set up import paths BEFORE any project imports
# LLMbot/ MUST come before LLMbot/code/ because code/utils.py shadows utils/ package
_ROOT = Path(__file__).resolve().parent.parent
_PIPE = Path(__file__).resolve().parent
os.chdir(str(_ROOT))
_paths = [str(_PIPE), str(_ROOT / "LLMbot"), str(_ROOT / "LLMbot" / "code")]
for p in _paths:
    if p in sys.path:
        sys.path.remove(p)
# Insert in reverse so first entry has highest priority
for p in reversed(_paths):
    sys.path.insert(0, p)

import torch

from config import RewriteConfig
from data_loader import load_dataset
from semantic_provider import get_semantic_outputs
from graph_trainer import train_graph_expert
from rewrite_engine import compute_rewrite
from evaluator import evaluate, save_metrics
from utils.misc import seed_setting, compute_directed_structural_features


def run(cfg: RewriteConfig):
    t0 = time.time()
    seed_setting(cfg.seed)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save()

    print(f"=== Rewrite Pipeline: {cfg.run_tag} ===")
    print(f"  feature={cfg.feature_source}, rewrite={cfg.rewrite_mode}, trigger={cfg.trigger_mode}, epochs={cfg.graph_epochs}")

    # 1. Load dataset
    print("\n[1/6] Loading dataset...")
    data = load_dataset(cfg)
    print(f"  Nodes={data['num_nodes']}, Edges={data['edge_index'].size(1)}, "
          f"Train={data['train_idx'].numel()}, Val_ckpt={data['valid_ckpt'].numel()}, Test={data['test_idx'].numel()}")

    # 2. Get semantic outputs
    print("\n[2/6] Getting semantic outputs...")
    sem = get_semantic_outputs(cfg, data)
    print(f"  z_sem shape={sem['z_sem'].shape}, pred_sem unique={sem['pred_sem'].unique().tolist()}")

    # 3. Select node features
    if cfg.feature_source == "z_sem":
        node_features = sem["z_sem"]
    elif cfg.feature_source == "q_final":
        node_features = data["q_final"]
    else:
        raise ValueError(f"Unknown feature_source: {cfg.feature_source}")
    print(f"  Node features: {cfg.feature_source} [{node_features.shape}]")

    # 4. Train raw graph probe on original graph
    print("\n[3/6] Training raw graph probe (original graph)...")
    raw_out = train_graph_expert(
        node_features=node_features,
        edge_index=data["edge_index"],
        edge_type=data["edge_type"],
        labels=data["labels"],
        train_idx=data["train_idx"],
        valid_ckpt_idx=data["valid_ckpt"],
        valid_cal_idx=data["valid_cal"],
        struct_feats=data["struct_feats"],
        cfg=cfg,
        class_weights=data["class_weights"],
        stage_name="raw_probe",
        save_dir=run_dir,
    )
    raw_metrics = evaluate(raw_out["pred_graph"], raw_out["prob_graph_cal"],
                           data["labels"], data["test_idx"], data["struct_feats"], sem["pred_sem"])
    print(f"  Raw probe: F1={raw_metrics['f1']:.4f}, Acc={raw_metrics['acc']:.4f}, ECE={raw_metrics['ece']:.4f}")

    # 5. Graph rewrite
    if cfg.rewrite_mode == "none":
        print("\n[4/6] Rewrite mode=none, skipping...")
        rewrite_result = {"edge_index": data["edge_index"], "edge_type": data["edge_type"],
                          "stats": {"mode": "none", "edges_before": data["edge_index"].size(1),
                                    "edges_after": data["edge_index"].size(1), "pruned": 0, "added": 0, "triggered": 0}}
        rewritten_out = raw_out
        rewritten_metrics = raw_metrics
    else:
        print(f"\n[4/6] Computing rewrite (mode={cfg.rewrite_mode}, trigger={cfg.trigger_mode})...")
        rewrite_result = compute_rewrite(data, sem, raw_out, cfg)
        stats = rewrite_result["stats"]
        print(f"  Triggered={stats['triggered']}, Pruned={stats['pruned']}, Added={stats['added']}, "
              f"Edges: {stats['edges_before']} -> {stats['edges_after']}")

        # Recompute struct_feats for rewritten graph
        new_struct = compute_directed_structural_features(
            rewrite_result["edge_index"].long().cpu(), data["num_nodes"], data["q_final"]
        ).float().cpu()

        # 6. Train new graph expert on rewritten graph
        print("\n[5/6] Training graph expert on rewritten graph...")
        rewritten_out = train_graph_expert(
            node_features=node_features,
            edge_index=rewrite_result["edge_index"],
            edge_type=rewrite_result["edge_type"],
            labels=data["labels"],
            train_idx=data["train_idx"],
            valid_ckpt_idx=data["valid_ckpt"],
            valid_cal_idx=data["valid_cal"],
            struct_feats=new_struct,
            cfg=cfg,
            class_weights=data["class_weights"],
            stage_name="rewritten",
            save_dir=run_dir,
        )
        rewritten_metrics = evaluate(rewritten_out["pred_graph"], rewritten_out["prob_graph_cal"],
                                      data["labels"], data["test_idx"], new_struct, sem["pred_sem"])
        print(f"  Rewritten: F1={rewritten_metrics['f1']:.4f}, Acc={rewritten_metrics['acc']:.4f}, ECE={rewritten_metrics['ece']:.4f}")

    # Save all results
    print("\n[6/6] Saving results...")
    elapsed = time.time() - t0
    summary = {
        "config": {
            "seed": cfg.seed, "feature_source": cfg.feature_source,
            "rewrite_mode": cfg.rewrite_mode, "trigger_mode": cfg.trigger_mode,
            "graph_epochs": cfg.graph_epochs,
        },
        "raw_probe": raw_metrics,
        "rewritten": rewritten_metrics,
        "delta_f1": rewritten_metrics["f1"] - raw_metrics["f1"],
        "rewrite_stats": rewrite_result["stats"],
        "elapsed_seconds": round(elapsed, 1),
    }
    save_metrics(summary, run_dir / "summary.json")
    save_metrics(raw_metrics, run_dir / "raw_probe_metrics.json")
    save_metrics(rewritten_metrics, run_dir / "rewritten_metrics.json")

    print(f"\n=== Done in {elapsed:.0f}s ===")
    print(f"  Raw:      F1={raw_metrics['f1']:.4f}")
    print(f"  Rewritten: F1={rewritten_metrics['f1']:.4f}")
    print(f"  Delta:    {summary['delta_f1']:+.4f}")
    print(f"  Results:  {run_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Rewrite Pipeline — Single Experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_source", type=str, default="z_sem", choices=["z_sem", "q_final"])
    parser.add_argument("--rewrite_mode", type=str, default="prune_add",
                        choices=["none", "prune_only", "add_only", "prune_add", "random_add"])
    parser.add_argument("--trigger_mode", type=str, default="disagreement",
                        choices=["disagreement", "trust_only", "full", "random"])
    parser.add_argument("--graph_epochs", type=int, default=30)
    parser.add_argument("--results_dir", type=str, default="results/rewrite_pipeline")
    args = parser.parse_args()

    cfg = RewriteConfig(
        seed=args.seed,
        feature_source=args.feature_source,
        rewrite_mode=args.rewrite_mode,
        trigger_mode=args.trigger_mode,
        graph_epochs=args.graph_epochs,
        results_dir=args.results_dir,
    )
    run(cfg)


if __name__ == "__main__":
    main()
