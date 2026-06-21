import csv
import json
from pathlib import Path

from runtime_env import now_iso, read_json_file, resolve_botdetection_root, write_json_file


REPO_ROOT = resolve_botdetection_root()
WORK_DIR = REPO_ROOT / "LLMbot"


def mean_std(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, var ** 0.5


def summarize_sampled_twibot22_base():
    root = WORK_DIR / "experiments" / "sampled_twibot22_official_prior_base5_20260620"
    rows = []
    for seed in range(1, 6):
        metrics_path = root / f"seed_{seed}" / "preparation" / "graph_detector" / "selection_metrics.json"
        semantic_path = root / f"seed_{seed}" / "preparation" / "semantic_encoder" / "metrics.json"
        metrics = read_json_file(metrics_path, default=None)
        semantic = read_json_file(semantic_path, default=None)
        if metrics is None:
            rows.append({"dataset": "sampled_twibot22", "seed": seed, "status": "missing"})
            continue
        row = {
            "dataset": "sampled_twibot22",
            "variant": "base_rgcn_finetuned_roberta",
            "seed": seed,
            "status": "completed",
            "acc": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "loss": metrics.get("loss"),
            "epoch": metrics.get("epoch"),
        }
        if semantic and isinstance(semantic.get("test"), dict):
            row["semantic_test_acc"] = semantic["test"].get("accuracy")
            row["semantic_test_macro_f1"] = semantic["test"].get("macro_f1")
            row["semantic_test_bot_f1"] = semantic["test"].get("bot_f1")
        rows.append(row)
    return rows


def load_existing_twibot20_rows():
    rows = []
    for csv_path in [
        WORK_DIR / "experiments" / "local_twibot20_routed_residual_multiseed_20260619" / "reports" / "summary_by_seed.csv",
    ]:
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
    return rows


def summarize_twibot20_completion_rows():
    root = WORK_DIR / "experiments" / "twibot20_formal5_ablation_completion_20260620"
    rows = []
    variant_dirs = sorted(root.glob("*_seed*")) if root.exists() else []
    for variant_dir in variant_dirs:
        for metrics_path in variant_dir.glob("seed_*/preparation/graph_detector/selection_metrics.json"):
            metrics = read_json_file(metrics_path, default=None)
            if not metrics:
                continue
            seed_dir = metrics_path.parents[2]
            try:
                seed = int(seed_dir.name.split("_")[-1])
            except Exception:
                seed = None
            name = variant_dir.name
            variant = name.rsplit("_seed", 1)[0] if "_seed" in name else name
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "artifact_root": str(variant_dir),
                    "full_acc": metrics.get("accuracy"),
                    "full_macro_f1": metrics.get("macro_f1"),
                    "status": "completed",
                }
            )
    return rows


def normalize_twibot20_variant(name):
    mapping = {
        "A0_low_only": "w/o residual",
        "A1_all_nodes_residual": "w/o router",
        "A3_routed_only_budget100": "main routed residual",
        "A6_shuffled_routed_only_budget100": "same-budget random routed residual",
        "C3_wo_lm_supervised_raw_roberta": "w/o LM supervised fine-tuning",
    }
    return mapping.get(str(name), str(name))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by_variant(rows):
    grouped = {}
    for row in rows:
        if row.get("status") and row.get("status") != "completed":
            continue
        variant = row.get("display_variant") or normalize_twibot20_variant(row.get("variant") or "unknown")
        grouped.setdefault(variant, []).append(row)
    summary = []
    for variant, items in sorted(grouped.items()):
        acc_mean, acc_std = mean_std([item.get("full_acc") or item.get("acc") for item in items])
        f1_mean, f1_std = mean_std([item.get("full_macro_f1") or item.get("macro_f1") for item in items])
        summary.append(
            {
                "variant": variant,
                "n": len(items),
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
            }
        )
    return summary


TARGET_TWIBOT20_VARIANTS = {
    "main routed residual",
    "w/o router",
    "w/o residual",
    "same-budget random routed residual",
    "w/o LM supervised fine-tuning",
}


def main():
    report_dir = WORK_DIR / "experiments" / "formal_result_snapshots_20260620"
    sampled_rows = summarize_sampled_twibot22_base()
    twibot20_rows = load_existing_twibot20_rows() + summarize_twibot20_completion_rows()
    for row in twibot20_rows:
        if "variant" in row:
            row["display_variant"] = normalize_twibot20_variant(row["variant"])
    sampled_summary = aggregate_by_variant(sampled_rows)
    twibot20_summary = aggregate_by_variant(twibot20_rows)
    twibot20_target_rows = [
        row for row in twibot20_rows if row.get("display_variant") in TARGET_TWIBOT20_VARIANTS
    ]
    twibot20_target_summary = aggregate_by_variant(twibot20_target_rows)
    write_csv(report_dir / "sampled_twibot22_base_by_seed.csv", sampled_rows)
    write_csv(report_dir / "sampled_twibot22_base_summary.csv", sampled_summary)
    write_csv(report_dir / "twibot20_ablation_by_seed.csv", twibot20_rows)
    write_csv(report_dir / "twibot20_ablation_summary.csv", twibot20_summary)
    write_csv(report_dir / "twibot20_target_ablation_by_seed.csv", twibot20_target_rows)
    write_csv(report_dir / "twibot20_target_ablation_summary.csv", twibot20_target_summary)
    manifest = {
        "created_at": now_iso(),
        "script": "summarize_formal_runs_20260620.py",
        "report_dir": str(report_dir),
        "sampled_twibot22_completed_seeds": [
            int(row["seed"]) for row in sampled_rows if row.get("status") == "completed"
        ],
        "twibot20_row_count": len(twibot20_rows),
        "known_gap": (
            "Raw RoBERTa extract-only w/o-LM-supervision runs are queued by "
            "run_twibot20_formal5_ablation_completion_20260620.py. Strict paired "
            "finetuned-vs-raw 5-seed comparison still needs finetuned_roberta_embeddings_iter_2_seed2..5."
        ),
    }
    write_json_file(report_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
