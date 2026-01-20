import os
import json
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def evaluate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    out_dir: str,
    seed: int = 42,
    max_geo_samples: int = 20000,
    max_vis_samples: int = 3000,
    n_kmeans_init: int = 10,
    n_cv_folds: int = 5,
):
    """
    Paper-grade evaluation with Hypersphere Normalization.
    """

    print("\n" + "=" * 76)
    print(f"EVALUATION REPORT: {name}")
    print(f"Embeddings shape: {X.shape}")
    print(f"Labels shape:     {y.shape}")
    print("=" * 76)

    # ------------------------------------------------------------------
    # 0. Sanity & Preprocessing
    # ------------------------------------------------------------------
    if np.isnan(X).any() or np.isinf(X).any():
        print("⚠️  NaN / Inf detected. Cleaning...")
        X = np.nan_to_num(X)

    rng = np.random.default_rng(seed)
    results = {
        "name": name,
        "num_samples": len(X),
        "embedding_dim": X.shape[1],
        "seed": seed,
    }

    # CRITICAL FIX: Create L2-Normalized Copy for Geometric Eval
    # LLM embeddings should be compared via Cosine Similarity.
    # L2-Normalizing X makes Euclidean Distance proportional to Cosine Distance.
    X_norm = normalize(X, norm='l2', axis=1)

    # ------------------------------------------------------------------
    # 1. Geometry & Structural Evaluation (Unsupervised)
    # ------------------------------------------------------------------
    print(f"[{name}] 1. Geometry & Structural Evaluation (Cosine Space)")

    if len(X) > max_geo_samples:
        idx = rng.choice(len(X), max_geo_samples, replace=False)
        # Use Normalized X for Geometry
        X_geo, y_geo = X_norm[idx], y[idx] 
    else:
        X_geo, y_geo = X_norm, y

    # 1.1 Class-conditional Silhouette Score (Cosine)
    try:
        # We use 'euclidean' on L2-normalized data, which equates to Cosine
        sil_gt = silhouette_score(X_geo, y_geo, metric="euclidean")
        results["silhouette_gt"] = float(sil_gt)
        print(
            f"   ► Silhouette (GT labels): {sil_gt:.4f} "
            "(Higher = Better Class Separation)"
        )
    except Exception as e:
        print(f"   ✗ Silhouette computation failed: {e}")

    # 1.2 Spherical KMeans + NMI
    try:
        n_clusters = len(np.unique(y_geo))
        # KMeans on L2-normalized data approximates "Spherical KMeans"
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=n_kmeans_init,
            random_state=seed,
        )
        cluster_ids = kmeans.fit_predict(X_geo)

        nmi = normalized_mutual_info_score(y_geo, cluster_ids)
        results["nmi"] = float(nmi)
        results["n_clusters"] = int(n_clusters)

        print(
            f"   ► NMI (KMeans vs Labels): {nmi:.4f} "
            "(Higher = Better Structure-Semantics Alignment)"
        )
    except Exception as e:
        print(f"   ✗ Clustering evaluation failed: {e}")

    # ------------------------------------------------------------------
    # 2. Supervised Linear Probe (Decodability)
    # ------------------------------------------------------------------
    print(f"[{name}] 2. Linear Probe Evaluation (Raw Feature Space)")

    # NOTE: For Logistic Regression, we use the RAW X (with StandardScaler)
    # because the solver handles magnitude and bias terms.
    clf = make_pipeline(
        StandardScaler(), # Center and scale variance
        LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
        ),
    )

    cv = StratifiedKFold(
        n_splits=n_cv_folds,
        shuffle=True,
        random_state=seed,
    )

    try:
        acc_scores = cross_val_score(
            clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1
        )
        f1_scores = cross_val_score(
            clf, X, y, cv=cv, scoring="f1_macro", n_jobs=-1
        )

        results.update({
            "accuracy_mean": float(acc_scores.mean()),
            "accuracy_std": float(acc_scores.std()),
            "f1_macro_mean": float(f1_scores.mean()),
            "f1_macro_std": float(f1_scores.std()),
        })

        print(
            f"   ► Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}"
        )
        print(
            f"   ► F1-Macro: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}"
        )
    except Exception as e:
        print(f"   ✗ Linear probe evaluation failed: {e}")

    # ------------------------------------------------------------------
    # 3. Visualization
    # ------------------------------------------------------------------
    print(f"[{name}] 3. Visualization (Qualitative)")

    if len(X) > max_vis_samples:
        idx = rng.choice(len(X), max_vis_samples, replace=False)
        # Use Normalized for Visualization too (Standard for NLP)
        X_vis, y_vis = X_norm[idx], y[idx]
    else:
        X_vis, y_vis = X_norm, y

    # PCA
    pca_dim = min(50, X_vis.shape[1])
    X_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(X_vis)

    # t-SNE
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=30,
        metric="cosine", # Explicitly use Cosine for t-SNE
        random_state=seed,
    )
    X_tsne = tsne.fit_transform(X_pca)

    plt.figure(figsize=(9, 7))
    for label, color, label_name in [
        (0, "tab:blue", "Human"),
        (1, "tab:red", "Bot"),
    ]:
        mask = y_vis == label
        plt.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            s=12,
            alpha=0.6, # Slightly higher alpha for visibility
            c=color,
            label=label_name,
            edgecolors="none",
        )

    plt.legend()
    plt.axis("off")
    plt.title(
        f"{name} Embedding Space (Cosine t-SNE)\n"
        f"Silhouette: {results.get('silhouette_gt', 0):.3f} | "
        f"F1: {results.get('f1_macro_mean', 0):.3f}"
    )

    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"{name}_tsne.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   ► Visualization saved to: {fig_path}")

    # Persist
    result_path = os.path.join(out_dir, f"{name}_metrics.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--Qwen_raw', type=str, required=True, help='Path to Qwen embeddings .pt')
    parser.add_argument('--label', type=str, required=True, help='Path to labels .pt')
    parser.add_argument('--out', type=str, default='./evaluation_outputs')
    args = parser.parse_args()

    wandb.init(config=vars(args),
               project="Qwen_Evaluation",
               name="Qwen_Real_Data_Evaluation",
               dir=args.out,
               job_type="evaluation",
               reinit=True)

    # Load Data
    print(f"Loading {args.Qwen_raw}...")
    X = torch.load(args.Qwen_raw, map_location='cpu', weights_only=False)
    if hasattr(X, 'numpy'): X = X.numpy()
    
    print(f"Loading {args.label}...")
    y_raw = torch.load(args.label, map_location='cpu', weights_only=False)
    
    # Handle One-Hot
    if y_raw.dim() > 1 and y_raw.shape[1] > 1:
        y = torch.argmax(y_raw, dim=1).numpy()
    else:
        y = y_raw.numpy()

    # Align lengths
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]

    # Run
    evaluate_metrics(X, y, "Qwen_Real_Data", args.out)