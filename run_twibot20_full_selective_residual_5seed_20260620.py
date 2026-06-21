import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from runtime_env import (
    build_offline_model_env,
    mark_queue_manifest_failed,
    now_iso,
    run_manifest_command,
    write_json_file as write_json,
)


REPO_ROOT = Path(r"G:\Research\BotDetection")
ACTIVE_ROOT = REPO_ROOT / "LLMbot"
LEGACY_LMBOT_ROOT = REPO_ROOT / "LMBot"
PYTHON = Path(r"D:\Anaconda\envs\llmbot\python.exe")
DATASET_ROOT = REPO_ROOT / "datasets" / "TwiBot-20"
MODEL_ROOT = REPO_ROOT / "models" / "huggingface"
RUN_ROOT = ACTIVE_ROOT / "experiments" / "twibot20_full_selective_residual_5seed_20260620"
INPUTS_DIR = RUN_ROOT / "_inputs"
REPORT_DIR = RUN_ROOT / "_reports"
LOG_DIR = ACTIVE_ROOT / "server_logs"
QUEUE_MANIFEST = ACTIVE_ROOT / "experiments" / "twibot20_full_selective_residual_5seed_20260620_queue_manifest.json"

HOST_ENV_DEFAULTS = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
    "WANDB_SILENT": "true",
}
SEED1_EXISTING_FULL = (
    ACTIVE_ROOT
    / "experiments"
    / "twibot20_seed1_component_ablation_20260619"
    / "C0_full_finetuned_router_routed_only_residual_budget100"
    / "seed_1"
)
LMBOT_LM_MEMORY_ATTEMPTS = [(4, 8), (2, 16), (1, 32)]


BASE_GRAPH_ARGS = [
    "main.py",
    "--experiment_task",
    "graph_detector_prepare",
    "--dataset",
    "TwiBot-20",
    "--reset_split",
    "-1",
    "--graph_data_variant",
    "labeled",
    "--use_GNN",
    "--graph_backbone",
    "rgcn_hyperscan_dhg_nodeinput",
    "--graph_node_input_family",
    "hyperscan_meta_tweet_proxy",
    "--hidden_dim",
    "788",
    "--graph_training_loader_mode",
    "neighbor_subgraph",
    "--graph_second_view_scope",
    "neighborloader_batch",
    "--graph_neighborloader_contract",
    "hyperscan_sampled_subgraph",
    "--graph_second_view_hypergraph_backend",
    "dhg",
    "--graph_second_view_fusion",
    "residual",
    "--graph_refine_knn_k",
    "8",
    "--graph_neighbor_num_neighbors",
    "64",
    "--gnn_batch_size",
    "512",
    "--graph_training_max_steps",
    "200",
    "--device",
    "0",
    "--disable_wandb",
    "--force_retrain_backbone",
    "--graph_refine_mode",
    "hyperscan_neighborloader_batch_local_branch",
]


def clean_env():
    env = build_offline_model_env(REPO_ROOT)
    env.update(HOST_ENV_DEFAULTS)
    return env


def configure_host_process_env():
    for key, value in HOST_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_logged(command, cwd, log_path, manifest, job_name):
    proc = run_manifest_command(
        command,
        cwd=cwd,
        env=clean_env(),
        log_path=log_path,
        manifest=manifest,
        manifest_path=QUEUE_MANIFEST,
        entry={
            "job": job_name,
            "cwd": str(cwd),
            "command": [str(x) for x in command],
            "log_path": str(log_path),
        },
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{job_name} failed; see {log_path}")


def embedding_file_is_loadable(path):
    if not Path(path).exists() or Path(path).stat().st_size <= 0:
        return False
    try:
        import torch

        tensor = torch.load(path, map_location="cpu")
    except Exception:
        return False
    return hasattr(tensor, "shape") and len(tensor.shape) == 2 and int(tensor.shape[0]) > 0


def run_lmbot_embedding_logged(command, cwd, log_path, manifest, job_name, source_path):
    log_path = Path(log_path)
    source_path = Path(source_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "job": job_name,
        "status": "running",
        "started_at": now_iso(),
        "cwd": str(cwd),
        "command": [str(x) for x in command],
        "log_path": str(log_path),
        "early_success_artifact": str(source_path),
    }
    manifest["runs"].append(entry)
    write_json(QUEUE_MANIFEST, manifest)

    last_size = -1
    stable_polls = 0
    with log_path.open("wb") as handle:
        proc = subprocess.Popen(
            [str(x) for x in command],
            cwd=str(cwd),
            env=clean_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        while True:
            returncode = proc.poll()
            if source_path.exists():
                size = source_path.stat().st_size
                stable_polls = stable_polls + 1 if size == last_size and size > 0 else 0
                last_size = size
                if stable_polls >= 2 and embedding_file_is_loadable(source_path):
                    entry["finished_at"] = now_iso()
                    entry["status"] = "completed_early_on_embedding_artifact"
                    entry["early_stop_reason"] = "embeddings_iter_2.pt materialized and loadable"
                    proc.terminate()
                    try:
                        returncode = proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        returncode = proc.wait(timeout=30)
                    entry["returncode"] = int(returncode)
                    write_json(QUEUE_MANIFEST, manifest)
                    return
            if returncode is not None:
                entry["finished_at"] = now_iso()
                entry["returncode"] = int(returncode)
                entry["status"] = "completed" if returncode == 0 else "failed"
                write_json(QUEUE_MANIFEST, manifest)
                if returncode != 0:
                    raise RuntimeError(f"{job_name} failed; see {log_path}")
                return
            time.sleep(15)


def legacy_embedding_target(seed):
    return DATASET_ROOT / f"finetuned_roberta_embeddings_iter_2_seed{int(seed)}.pt"


def ensure_lmbot_iter2_embedding(seed, manifest):
    seed = int(seed)
    target = legacy_embedding_target(seed)
    if target.exists():
        manifest["embedding_artifacts"][str(seed)] = {
            "status": "existing",
            "path": str(target),
        }
        write_json(QUEUE_MANIFEST, manifest)
        return target

    legacy_results_root = RUN_ROOT / "lmbot_iter2_seed_specific"
    legacy_seed_dir = legacy_results_root / "results_rgcn" / f"seed_{seed}"
    source = legacy_seed_dir / "intermediate" / "LM" / "embeddings_iter_2.pt"
    if not source.exists():
        last_error = None
        for batch_size, accumulation in LMBOT_LM_MEMORY_ATTEMPTS:
            log_path = LOG_DIR / f"twibot20_fullmodel_seed{seed}_lmbot_iter2_b{batch_size}_acc{accumulation}.log"
            command = [
                PYTHON,
                "-m",
                "lmbot_harness",
                "train",
                "--project_name",
                "lmbot",
                "--experiment_name",
                "TwiBot-20_fullmodel_iter2",
                "--dataset",
                "TwiBot-20",
                "--dataset_path",
                str(DATASET_ROOT),
                "--results_root",
                str(legacy_results_root),
                "--device",
                "0",
                "--LM_pretrain_epochs",
                "4.5",
                "--alpha",
                "0.5",
                "--max_iters",
                "3",
                "--batch_size_LM",
                str(batch_size),
                "--LM_accumulation",
                str(accumulation),
                "--LM_eval_patience",
                str(20 * accumulation),
                "--use_GNN",
                "--GNN_model",
                "rgcn",
                "--seeds",
                str(seed),
            ]
            try:
                run_lmbot_embedding_logged(
                    command,
                    LEGACY_LMBOT_ROOT,
                    log_path,
                    manifest,
                    f"seed{seed}_lmbot_iter2_embedding_b{batch_size}_acc{accumulation}",
                    source,
                )
                break
            except RuntimeError as exc:
                last_error = exc
                log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
                if "CUDA out of memory" not in log_text or (batch_size, accumulation) == LMBOT_LM_MEMORY_ATTEMPTS[-1]:
                    raise
                manifest.setdefault("embedding_retry_notes", []).append(
                    {
                        "seed": seed,
                        "failed_batch_size_lm": batch_size,
                        "failed_lm_accumulation": accumulation,
                        "reason": "CUDA out of memory",
                        "next_attempt": {
                            "batch_size_lm": LMBOT_LM_MEMORY_ATTEMPTS[
                                LMBOT_LM_MEMORY_ATTEMPTS.index((batch_size, accumulation)) + 1
                            ][0],
                            "lm_accumulation": LMBOT_LM_MEMORY_ATTEMPTS[
                                LMBOT_LM_MEMORY_ATTEMPTS.index((batch_size, accumulation)) + 1
                            ][1],
                        },
                    }
                )
                write_json(QUEUE_MANIFEST, manifest)
        if not source.exists() and last_error is not None:
            raise last_error
    if not source.exists():
        raise FileNotFoundError(f"Expected seed {seed} LMBot iter2 embedding was not produced: {source}")
    if not embedding_file_is_loadable(source):
        raise RuntimeError(f"Expected seed {seed} LMBot iter2 embedding is not loadable: {source}")
    shutil.copy2(source, target)
    manifest["embedding_artifacts"][str(seed)] = {
        "status": "materialized_from_lmbot_iter2",
        "path": str(target),
        "source": str(source),
    }
    write_json(QUEUE_MANIFEST, manifest)
    return target


def graph_artifact_root(variant, seed):
    return RUN_ROOT / f"{variant}_seed{int(seed)}"


def graph_outputs_root(variant, seed):
    return graph_artifact_root(variant, seed) / f"seed_{int(seed)}"


def graph_detector_outputs_path(variant, seed):
    return graph_outputs_root(variant, seed) / "preparation" / "graph_detector" / "outputs.pt"


def run_graph_variant(seed, variant, embedding_path, extra_args, manifest):
    out_path = graph_detector_outputs_path(variant, seed)
    if out_path.exists():
        return graph_outputs_root(variant, seed)
    command = [
        PYTHON,
        *BASE_GRAPH_ARGS,
        "--embedding_path",
        str(embedding_path),
        "--seeds",
        str(int(seed)),
        "--artifact_root",
        str(graph_artifact_root(variant, seed)),
        *extra_args,
    ]
    run_logged(
        command,
        ACTIVE_ROOT,
        LOG_DIR / f"twibot20_fullmodel_seed{int(seed)}_{variant}.log",
        manifest,
        f"seed{int(seed)}_{variant}",
    )
    if not out_path.exists():
        raise FileNotFoundError(f"Graph detector outputs missing after {variant} seed {seed}: {out_path}")
    return graph_outputs_root(variant, seed)


def run_all_nodes_source(seed, embedding_path, manifest):
    return run_graph_variant(
        seed=seed,
        variant="C1_wo_router_all_nodes_finetuned_residual",
        embedding_path=embedding_path,
        extra_args=["--graph_second_view_consumer_scope", "all_nodes"],
        manifest=manifest,
    )


def router_root(seed):
    return RUN_ROOT / f"router_finetuned_xnew_from_all_nodes_seed{int(seed)}"


def risk_manifest_path(seed):
    return router_root(seed) / f"seed_{int(seed)}" / "stages" / "estimator_ablation" / "risk_manifest.json"


def run_router(seed, external_g0_root, manifest):
    risk_path = risk_manifest_path(seed)
    if risk_path.exists():
        return risk_path
    command = [
        PYTHON,
        "main.py",
        "--experiment_task",
        "estimator_ablation",
        "--dataset",
        "TwiBot-20",
        "--reset_split",
        "-1",
        "--graph_data_variant",
        "labeled",
        "--use_GNN",
        "--graph_backbone",
        "rgcn_hyperscan_dhg_nodeinput",
        "--estimator_mode",
        "conformal_knn_risk_router",
        "--external_frozen_g0_root",
        str(external_g0_root),
        "--conformal_knn_repr_source",
        "x_new",
        "--conformal_knn_candidate_scope",
        "labeled_full",
        "--conformal_knn_k",
        "8",
        "--conformal_knn_learning_mode",
        "ncp_local",
        "--conformal_knn_ncp_lambda",
        "1.0",
        "--conformal_knn_score_family_override",
        "auto",
        "--conformal_knn_target_top_n",
        "200",
        "--risk_budgets",
        "0.05,0.1,0.2",
        "--seeds",
        str(int(seed)),
        "--device",
        "0",
        "--disable_wandb",
        "--artifact_root",
        str(router_root(seed)),
    ]
    run_logged(
        command,
        ACTIVE_ROOT,
        LOG_DIR / f"twibot20_fullmodel_seed{int(seed)}_router_finetuned_xnew.log",
        manifest,
        f"seed{int(seed)}_router_finetuned_xnew",
    )
    if not risk_path.exists():
        raise FileNotFoundError(f"Router risk manifest missing after seed {seed}: {risk_path}")
    return risk_path


def materialize_routed_masks(seed, risk_path):
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = read_json(risk_path)
    selected = payload.get("selected_nodes_by_budget") or {}
    outputs = {}
    for budget_key, source_key in {"050": "budget_050", "100": "budget_100", "200": "budget_200"}.items():
        item = selected.get(source_key)
        if not isinstance(item, dict):
            raise KeyError(f"{risk_path} missing selected_nodes_by_budget.{source_key}")
        out_path = INPUTS_DIR / f"routed_nodes_finetuned_xnew_budget{budget_key}_seed{int(seed)}.json"
        if not out_path.exists():
            routed = {
                "contract": "derived_routed_nodes_from_conformal_knn_risk_router_v1",
                "source_risk_manifest_path": str(risk_path),
                "budget_key": budget_key,
                "budget": float(item.get("budget", 0.0)),
                "counts": dict(item.get("split_counts") or {}),
                "graph_data_variant": "labeled",
                "conformal_knn_repr_source": "x_new",
                "seed": int(seed),
                "train": [int(x) for x in item.get("train", [])],
                "valid": [int(x) for x in item.get("valid", [])],
                "test": [int(x) for x in item.get("test", [])],
                "all": [int(x) for x in item.get("all", [])],
            }
            if not routed["all"]:
                routed["all"] = sorted(set(routed["train"] + routed["valid"] + routed["test"]))
            if not routed["counts"]:
                routed["counts"] = {
                    "train": len(routed["train"]),
                    "valid": len(routed["valid"]),
                    "test": len(routed["test"]),
                    "all": len(routed["all"]),
                }
            write_json(out_path, routed)
        outputs[budget_key] = out_path
    return outputs


def run_full_model(seed, embedding_path, routed_budget100_path, manifest):
    return run_graph_variant(
        seed=seed,
        variant="C0_full_finetuned_router_routed_only_residual_budget100",
        embedding_path=embedding_path,
        extra_args=[
            "--graph_second_view_consumer_scope",
            "routed_only",
            "--graph_second_view_nonconsumer_fallback",
            "low_only",
            "--routed_nodes_path",
            str(routed_budget100_path),
        ],
        manifest=manifest,
    )


def recompute_metrics(rows):
    import torch
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    test_idx = torch.load(DATASET_ROOT / "test_idx.pt", map_location="cpu").view(-1).long()
    metric_rows = []
    for row in rows:
        seed = int(row["seed"])
        outputs_path = Path(row["outputs_path"])
        payload = torch.load(outputs_path, map_location="cpu")
        labels = payload["labels"]
        if labels.dim() == 2:
            y_true = labels.argmax(dim=1)
        else:
            y_true = labels.long()
        pred = payload.get("pred")
        if pred is None:
            logits = payload.get("logits")
            prob = payload.get("prob")
            pred = logits.argmax(dim=1) if logits is not None else prob.argmax(dim=1)
        y = y_true[test_idx].cpu().numpy()
        p = pred[test_idx].long().cpu().numpy()
        cm = confusion_matrix(y, p, labels=[0, 1]).tolist()
        metric_rows.append(
            {
                "seed": seed,
                "artifact_root": str(outputs_path.parents[2]),
                "outputs_path": str(outputs_path),
                "test_count": int(len(test_idx)),
                "acc": float(accuracy_score(y, p)),
                "macro_f1": float(f1_score(y, p, average="macro")),
                "binary_f1": float(f1_score(y, p, pos_label=1, zero_division=0)),
                "macro_precision": float(precision_score(y, p, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(y, p, average="macro", zero_division=0)),
                "bot_precision": float(precision_score(y, p, pos_label=1, zero_division=0)),
                "bot_recall": float(recall_score(y, p, pos_label=1, zero_division=0)),
                "confusion_matrix_rows_true_cols_pred": cm,
                "wrong": int((p != y).sum()),
            }
        )
    return metric_rows


def mean_std(values):
    vals = [float(v) for v in values]
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
    return mean, var ** 0.5


def write_summary(metric_rows):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "test_count",
        "acc",
        "macro_f1",
        "binary_f1",
        "macro_precision",
        "macro_recall",
        "bot_precision",
        "bot_recall",
        "wrong",
        "outputs_path",
    ]
    csv_lines = [",".join(fields)]
    for row in metric_rows:
        csv_lines.append(",".join(str(row.get(field, "")) for field in fields))
    (REPORT_DIR / "full_model_5seed_by_seed.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    summary = {"n": len(metric_rows)}
    for key in [
        "acc",
        "macro_f1",
        "binary_f1",
        "macro_precision",
        "macro_recall",
        "bot_precision",
        "bot_recall",
        "wrong",
    ]:
        mean, std = mean_std([row[key] for row in metric_rows])
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
    payload = {
        "contract": "twibot20_full_selective_residual_5seed_summary_v1",
        "metric_contract": "deduplicated full-test recompute from outputs.pt + test_idx.pt",
        "method": "finetuned RoBERTa + conformal x_new router + routed-only residual @10%",
        "seed_count": len(metric_rows),
        "rows": metric_rows,
        "summary": summary,
        "label_mapping_assumption": {"0": "human", "1": "bot"},
    }
    write_json(REPORT_DIR / "full_model_5seed_summary.json", payload)
    return payload


def main():
    configure_host_process_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,2,3,4,5")
    args = parser.parse_args()
    requested_seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    manifest = {
        "created_at": now_iso(),
        "status": "running",
        "dataset": "TwiBot-20",
        "run_root": str(RUN_ROOT),
        "method": "finetuned RoBERTa + conformal x_new router + routed-only residual @10%",
        "seed1_existing_full_root": str(SEED1_EXISTING_FULL),
        "requested_seeds": requested_seeds,
        "embedding_contract": (
            "Seed 1 reuses existing finetuned_roberta_embeddings_iter_2_seed1.pt. "
            "Seeds 2-5 are materialized from legacy LMBot intermediate/LM/embeddings_iter_2.pt."
        ),
        "runs": [],
        "embedding_artifacts": {},
    }
    write_json(QUEUE_MANIFEST, manifest)

    completed_rows = []
    try:
        for seed in requested_seeds:
            if seed == 1:
                seed1_outputs = SEED1_EXISTING_FULL / "preparation" / "graph_detector" / "outputs.pt"
                if not seed1_outputs.exists():
                    embedding = ensure_lmbot_iter2_embedding(seed, manifest)
                    all_nodes_root = run_all_nodes_source(seed, embedding, manifest)
                    risk_path = run_router(seed, all_nodes_root, manifest)
                    routed = materialize_routed_masks(seed, risk_path)
                    full_root = run_full_model(seed, embedding, routed["100"], manifest)
                    seed1_outputs = full_root / "preparation" / "graph_detector" / "outputs.pt"
                completed_rows.append({"seed": seed, "outputs_path": str(seed1_outputs), "source": "existing_seed1_full"})
                continue

            embedding = ensure_lmbot_iter2_embedding(seed, manifest)
            all_nodes_root = run_all_nodes_source(seed, embedding, manifest)
            risk_path = run_router(seed, all_nodes_root, manifest)
            routed = materialize_routed_masks(seed, risk_path)
            full_root = run_full_model(seed, embedding, routed["100"], manifest)
            completed_rows.append(
                {
                    "seed": seed,
                    "outputs_path": str(full_root / "preparation" / "graph_detector" / "outputs.pt"),
                    "source": "new_full_run",
                }
            )

        metric_rows = recompute_metrics(completed_rows)
        summary = write_summary(metric_rows)
        manifest["status"] = "completed"
        manifest["completed_at"] = now_iso()
        manifest["report_dir"] = str(REPORT_DIR)
        manifest["summary"] = summary["summary"]
        write_json(QUEUE_MANIFEST, manifest)
        print(json.dumps(summary["summary"], indent=2, sort_keys=True))
    except Exception as exc:
        mark_queue_manifest_failed(QUEUE_MANIFEST, exc, manifest=manifest)
        print(str(exc), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
