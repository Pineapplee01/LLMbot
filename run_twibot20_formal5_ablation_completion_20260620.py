import json
import shutil
import sys
import time
from pathlib import Path

from runtime_env import build_offline_model_env, now_iso, run_logged_command, write_json_file

REPO_ROOT = Path(r"G:\Research\BotDetection")
WORK_DIR = REPO_ROOT / "LLMbot"
PYTHON = Path(r"D:\Anaconda\envs\llmbot\python.exe")
FALLBACK_PYTHON = Path(r"D:\Anaconda\python.exe")
LOG_DIR = WORK_DIR / "server_logs"
CURRENT_QUEUE = WORK_DIR / "experiments" / "sampled_twibot22_official_prior_base5_20260620_queue_manifest.json"
EXP_ROOT = r"experiments\twibot20_formal5_ablation_completion_20260620"
DATASET = "TwiBot-20"
FROZEN_COMPLETION_SEEDS = [4, 5]
RAW_ROBERTA_SEEDS = [1, 2, 3, 4, 5]

BASE_ARGS = [
    "main.py", "--experiment_task", "graph_detector_prepare",
    "--dataset", DATASET,
    "--reset_split", "-1",
    "--graph_data_variant", "labeled",
    "--use_GNN",
    "--graph_backbone", "rgcn_hyperscan_dhg_nodeinput",
    "--graph_node_input_family", "hyperscan_meta_tweet_proxy",
    "--hidden_dim", "788",
    "--graph_training_loader_mode", "neighbor_subgraph",
    "--graph_second_view_scope", "neighborloader_batch",
    "--graph_neighborloader_contract", "hyperscan_sampled_subgraph",
    "--graph_second_view_hypergraph_backend", "dhg",
    "--graph_second_view_fusion", "residual",
    "--graph_refine_knn_k", "8",
    "--graph_neighbor_num_neighbors", "64",
    "--gnn_batch_size", "512",
    "--graph_training_max_steps", "200",
    "--device", "0",
    "--disable_wandb",
    "--force_retrain_backbone",
    "--graph_refine_mode", "hyperscan_neighborloader_batch_local_branch",
]


def clean_env():
    return build_offline_model_env(REPO_ROOT)


def write_json(path, payload):
    write_json_file(path, payload)


MANIFEST_PATH = WORK_DIR / "experiments" / "twibot20_formal5_ablation_completion_20260620_queue_manifest.json"


def sampled_queue_finished():
    if not CURRENT_QUEUE.exists():
        return True
    try:
        status = json.loads(CURRENT_QUEUE.read_text(encoding="utf-8")).get("status")
    except Exception:
        return False
    return status in {"completed", "failed"}


def wait_for_current_queue(manifest):
    while not sampled_queue_finished():
        manifest["waiting_for"] = str(CURRENT_QUEUE)
        manifest["last_wait_heartbeat"] = now_iso()
        write_json(MANIFEST_PATH, manifest)
        time.sleep(300)
    if CURRENT_QUEUE.exists():
        status = json.loads(CURRENT_QUEUE.read_text(encoding="utf-8")).get("status")
        manifest["wait_dependency_status"] = status
        write_json(MANIFEST_PATH, manifest)
        if status != "completed":
            raise RuntimeError(f"Dependency queue did not complete successfully: {CURRENT_QUEUE} status={status}")


def run_logged(command, log_path, manifest, job_name):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "job": job_name,
        "status": "running",
        "started_at": now_iso(),
        "command": command,
        "log_path": str(log_path),
    }
    manifest["runs"].append(entry)
    write_json(MANIFEST_PATH, manifest)
    proc = run_logged_command(command, cwd=WORK_DIR, env=clean_env(), log_path=log_path)
    entry["finished_at"] = now_iso()
    entry["returncode"] = int(proc.returncode)
    entry["status"] = "completed" if proc.returncode == 0 else "failed"
    write_json(MANIFEST_PATH, manifest)
    if proc.returncode != 0:
        raise RuntimeError(f"{job_name} failed; see {log_path}")


def graph_cmd(seed, variant, embedding_path, extra_args):
    exp_name = f"{EXP_ROOT}\\{variant}_seed{seed}"
    return [str(PYTHON), *BASE_ARGS, "--embedding_path", str(embedding_path), "--seeds", str(seed), "--experiment_name", exp_name, *extra_args]


def generate_routed_masks(seed, manifest):
    embedding = REPO_ROOT / "datasets" / DATASET / f"embeddings_iter_-1_seed_{seed}.pt"
    smoke_exp = f"{EXP_ROOT}\\smoke_all_nodes_residual_seed{seed}"
    smoke_root = WORK_DIR / smoke_exp / f"seed_{seed}"
    outputs = smoke_root / "preparation" / "graph_detector" / "outputs.pt"
    if not outputs.exists():
        cmd = [
            str(PYTHON), *BASE_ARGS,
            "--embedding_path", str(embedding),
            "--seeds", str(seed),
            "--experiment_name", smoke_exp,
            "--graph_second_view_consumer_scope", "all_nodes",
        ]
        run_logged(cmd, LOG_DIR / f"twibot20_formal5_seed{seed}_smoke_all_nodes_residual.log", manifest, f"seed{seed}_smoke_all_nodes_residual")

    router_exp = f"{EXP_ROOT}\\router_from_smoke_seed{seed}"
    router_root = WORK_DIR / router_exp / f"seed_{seed}"
    risk_manifest = router_root / "stages" / "estimator_ablation" / "risk_manifest.json"
    if not risk_manifest.exists():
        cmd = [
            str(PYTHON), "main.py",
            "--experiment_task", "estimator_ablation",
            "--dataset", DATASET,
            "--reset_split", "-1",
            "--graph_data_variant", "labeled",
            "--use_GNN",
            "--graph_backbone", "rgcn_hyperscan_dhg_nodeinput",
            "--estimator_mode", "conformal_knn_risk_router",
            "--external_frozen_g0_root", str(smoke_root),
            "--conformal_knn_repr_source", "x_new",
            "--conformal_knn_candidate_scope", "labeled_full",
            "--conformal_knn_k", "8",
            "--conformal_knn_learning_mode", "ncp_local",
            "--conformal_knn_ncp_lambda", "1.0",
            "--conformal_knn_score_family_override", "auto",
            "--conformal_knn_target_top_n", "200",
            "--risk_budgets", "0.05,0.10,0.20",
            "--seeds", str(seed),
            "--device", "0",
            "--disable_wandb",
            "--experiment_name", router_exp,
        ]
        run_logged(cmd, LOG_DIR / f"twibot20_formal5_seed{seed}_router_from_smoke.log", manifest, f"seed{seed}_router_from_smoke")

    # Copy or materialize routed-node jsons into a stable inputs directory and create shuffled/random budget100.
    inputs_dir = WORK_DIR / "experiments" / "twibot20_formal5_ablation_completion_20260620_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = router_root / "stages" / "estimator_ablation"
    copied = {}
    for budget in ("050", "100", "200"):
        srcs = list(stage_dir.rglob(f"*budget{budget}*.json")) + list(stage_dir.rglob(f"*{budget}*.json"))
        # Prefer routed_nodes artifacts with split payload.
        srcs = [p for p in srcs if "routed" in p.name.lower() and "shuffled" not in p.name.lower()]
        if not srcs:
            # Fallback: inspect all json files for selected_budget.
            for p in stage_dir.rglob("*.json"):
                try:
                    obj=json.loads(p.read_text(encoding='utf-8'))
                except Exception:
                    continue
                if str(obj.get('selected_budget','')).replace('.','') in {budget, budget.lstrip('0')} or obj.get('budget_name') == budget:
                    srcs.append(p)
        if srcs:
            src = sorted(srcs, key=lambda p: len(str(p)))[0]
            dst = inputs_dir / f"routed_nodes_xnew_budget{budget}_seed{seed}.json"
            if not dst.exists():
                shutil.copy2(src, dst)
            copied[budget] = dst
    # If router code already wrote canonical artifacts elsewhere, also locate them by exact known names.
    for budget in ("050", "100", "200"):
        target = inputs_dir / f"routed_nodes_xnew_budget{budget}_seed{seed}.json"
        if not target.exists():
            matches = list(router_root.rglob(f"routed_nodes*budget{budget}*.json"))
            if matches:
                shutil.copy2(matches[0], target)
                copied[budget] = target
    if any(not (inputs_dir / f"routed_nodes_xnew_budget{budget}_seed{seed}.json").exists() for budget in ("050", "100", "200")):
        materialize_routed_masks_from_risk_manifest(
            risk_manifest=risk_manifest,
            inputs_dir=inputs_dir,
            seed=seed,
        )
    budget100 = inputs_dir / f"routed_nodes_xnew_budget100_seed{seed}.json"
    shuffled = inputs_dir / f"routed_nodes_xnew_budget100_seed{seed}_shuffled.json"
    if budget100.exists() and not shuffled.exists():
        make_shuffled_mask(budget100, shuffled, seed)
    return inputs_dir


def materialize_routed_masks_from_risk_manifest(risk_manifest, inputs_dir, seed):
    payload = json.loads(Path(risk_manifest).read_text(encoding="utf-8"))
    selected = payload.get("selected_nodes_by_budget") or {}
    key_map = {
        "050": "budget_050",
        "100": "budget_100",
        "200": "budget_200",
    }
    for budget_key, risk_key in key_map.items():
        out_path = Path(inputs_dir) / f"routed_nodes_xnew_budget{budget_key}_seed{seed}.json"
        if out_path.exists():
            continue
        item = selected.get(risk_key)
        if not isinstance(item, dict):
            raise KeyError(f"{risk_manifest} missing selected_nodes_by_budget.{risk_key}")
        routed = {
            "contract": "conformal_knn_routed_nodes_v1",
            "source_stage": str(risk_manifest),
            "budget": float(item.get("budget", 0.0)),
            "budget_key": budget_key,
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


def make_empty_mask(path, seed):
    if path.exists():
        return
    payload = {
        "contract": "empty_routed_nodes_lowonly_control_v1",
        "source_stage": "empty_mask_for_wo_residual",
        "budget": 0.0,
        "budget_key": "000",
        "counts": {"train": 0, "valid": 0, "test": 0, "all": 0},
        "graph_data_variant": "labeled",
        "conformal_knn_repr_source": "x_new",
        "seed": int(seed),
        "train": [],
        "valid": [],
        "test": [],
        "all": [],
    }
    write_json(path, payload)


def make_shuffled_mask(src, dst, seed):
    obj = json.loads(src.read_text(encoding="utf-8"))
    import random
    rng = random.Random(int(seed) + 20260620)
    # Preserve split sizes but replace selected ids with same-split random nodes.
    # Use TwiBot-20 split indices from prepared dataset.
    import torch
    data_root = REPO_ROOT / "datasets" / DATASET
    train = torch.load(data_root / "train_idx.pt", map_location="cpu").view(-1).tolist() if (data_root / "train_idx.pt").exists() else None
    valid = torch.load(data_root / "valid_idx.pt", map_location="cpu").view(-1).tolist() if (data_root / "valid_idx.pt").exists() else None
    test = torch.load(data_root / "test_idx.pt", map_location="cpu").view(-1).tolist() if (data_root / "test_idx.pt").exists() else None
    split_pool = {"train": train, "valid": valid, "test": test}
    for key, pool in split_pool.items():
        current = obj.get(key, []) or []
        if pool is None:
            continue
        n = min(len(current), len(pool))
        obj[key] = sorted(rng.sample([int(x) for x in pool], n))
    obj["all"] = sorted(set(int(x) for key in ("train", "valid", "test") for x in obj.get(key, [])))
    obj["counts"] = {
        "train": len(obj.get("train", [])),
        "valid": len(obj.get("valid", [])),
        "test": len(obj.get("test", [])),
        "all": len(obj.get("all", [])),
    }
    obj["source_stage"] = "same_budget_random_shuffled_routed_nodes"
    obj["shuffle_seed"] = int(seed) + 20260620
    write_json(dst, obj)


def run_seed_graphs(seed, manifest, inputs_dir):
    frozen_embedding = REPO_ROOT / "datasets" / DATASET / f"embeddings_iter_-1_seed_{seed}.pt"
    variants = [
        ("A0_low_only", frozen_embedding, ["--graph_second_view_consumer_scope", "routed_only", "--graph_second_view_nonconsumer_fallback", "low_only", "--routed_nodes_path", str(inputs_dir / f"routed_nodes_empty_seed{seed}.json")]),
        ("A1_all_nodes_residual", frozen_embedding, ["--graph_second_view_consumer_scope", "all_nodes"]),
        ("A3_routed_only_budget100", frozen_embedding, ["--graph_second_view_consumer_scope", "routed_only", "--graph_second_view_nonconsumer_fallback", "low_only", "--routed_nodes_path", str(inputs_dir / f"routed_nodes_xnew_budget100_seed{seed}.json")]),
        ("A6_shuffled_routed_only_budget100", frozen_embedding, ["--graph_second_view_consumer_scope", "routed_only", "--graph_second_view_nonconsumer_fallback", "low_only", "--routed_nodes_path", str(inputs_dir / f"routed_nodes_xnew_budget100_seed{seed}_shuffled.json")]),
    ]
    make_empty_mask(inputs_dir / f"routed_nodes_empty_seed{seed}.json", seed)
    for variant, embedding, extra in variants:
        out_manifest = WORK_DIR / EXP_ROOT / f"{variant}_seed{seed}" / f"seed_{seed}" / "preparation" / "graph_detector" / "manifest.json"
        if out_manifest.exists():
            continue
        cmd = graph_cmd(seed, variant, embedding, extra)
        run_logged(cmd, LOG_DIR / f"twibot20_formal5_seed{seed}_{variant}.log", manifest, f"seed{seed}_{variant}")


def run_raw_roberta_graph(seed, manifest, inputs_dir):
    raw_embedding = ensure_raw_roberta_embedding(seed, manifest)
    variant = "C3_wo_lm_supervised_raw_roberta"
    out_manifest = WORK_DIR / EXP_ROOT / f"{variant}_seed{seed}" / f"seed_{seed}" / "preparation" / "graph_detector" / "manifest.json"
    if out_manifest.exists():
        return
    cmd = graph_cmd(
        seed,
        variant,
        raw_embedding,
        [
            "--graph_second_view_consumer_scope",
            "routed_only",
            "--graph_second_view_nonconsumer_fallback",
            "low_only",
            "--routed_nodes_path",
            str(inputs_dir / f"routed_nodes_xnew_budget100_seed{seed}.json"),
        ],
    )
    run_logged(cmd, LOG_DIR / f"twibot20_formal5_seed{seed}_{variant}.log", manifest, f"seed{seed}_{variant}")


def ensure_raw_roberta_embedding(seed, manifest):
    out_dir = WORK_DIR / "experiments" / "twibot20_formal5_ablation_completion_20260620_raw_roberta" / f"seed_{seed}"
    embedding_path = out_dir / "raw_roberta_embeddings.pt"
    if embedding_path.exists():
        return embedding_path
    snapshot_root = REPO_ROOT / "models" / "huggingface" / "hub" / "models--roberta-base" / "snapshots"
    snapshots = sorted(snapshot_root.glob("*")) if snapshot_root.exists() else []
    model_path = snapshots[-1] if snapshots else ""
    cmd = [
        str(PYTHON),
        "extract_raw_roberta_embeddings_20260620.py",
        "--dataset",
        DATASET,
        "--graph_data_variant",
        "labeled",
        "--output_dir",
        str(out_dir),
        "--batch_size",
        "16",
        "--max_length",
        "512",
        "--device",
        "0",
    ]
    if model_path:
        cmd.extend(["--model_path", str(model_path)])
    run_logged(cmd, LOG_DIR / f"twibot20_formal5_seed{seed}_extract_raw_roberta.log", manifest, f"seed{seed}_extract_raw_roberta")
    return embedding_path


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": now_iso(),
        "dataset": DATASET,
        "experiment_root": EXP_ROOT,
        "python": str(PYTHON),
        "model_root": str(REPO_ROOT / "models" / "huggingface"),
        "frozen_completion_seeds": FROZEN_COMPLETION_SEEDS,
        "raw_roberta_seeds": RAW_ROBERTA_SEEDS,
        "status": "running",
        "notes": "Waits for sampled TwiBot-22 base queue, then completes TwiBot-20 frozen/residual/router ablation seeds 4-5 and extract-only raw RoBERTa w/o-LM-supervision runs.",
        "runs": [],
        "pending_known_gap": ["finetuned_roberta_embeddings_iter_2_seed2.pt..seed5.pt absent; raw RoBERTa extract-only w/o-LM-supervision runs are queued, but paired finetuned-vs-raw 5-seed comparison still needs finetuned LM artifacts."],
    }
    write_json(MANIFEST_PATH, manifest)
    wait_for_current_queue(manifest)
    for seed in sorted(set(FROZEN_COMPLETION_SEEDS + RAW_ROBERTA_SEEDS)):
        inputs_dir = generate_routed_masks(seed, manifest)
        if seed in FROZEN_COMPLETION_SEEDS:
            run_seed_graphs(seed, manifest, inputs_dir)
        if seed in RAW_ROBERTA_SEEDS:
            run_raw_roberta_graph(seed, manifest, inputs_dir)
    manifest["status"] = "completed"
    manifest["completed_at"] = now_iso()
    write_json(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        existing = {}
        if MANIFEST_PATH.exists():
            existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        existing["status"] = "failed"
        existing["failed_at"] = now_iso()
        existing["error"] = str(exc)
        write_json(MANIFEST_PATH, existing)
        print(str(exc), file=sys.stderr)
        raise
