import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")


REPO_ROOT = Path(r"G:\Research\BotDetection")
ACTIVE_ROOT = REPO_ROOT / "LLMbot"
DATASET_ROOT = REPO_ROOT / "datasets" / "TwiBot-20"
RUN_ROOT = ACTIVE_ROOT / "experiments" / "twibot20_seed1_router_alignment_20260712"
INPUTS_DIR = RUN_ROOT / "inputs"
REPORT_DIR = RUN_ROOT / "reports"
LOG_DIR = RUN_ROOT / "logs"
MANIFEST_PATH = RUN_ROOT / "experiment_manifest.json"

PYTHON = Path(sys.executable).resolve()
SEED = 1
BUDGET = 0.10
KNN_K = 8
FANOUT = 64
GRAPH_STEPS = 200
FORMULA_LAMBDA = 1.0
FORMULA_BATCH_SIZE = 256

C2_ROOT = (
    ACTIVE_ROOT
    / "experiments"
    / "twibot20_seed1_component_ablation_20260619"
    / "C2_wo_residual_finetuned_low_only_empty_mask"
    / "seed_1"
)
C2_OUTPUTS = C2_ROOT / "preparation" / "graph_detector" / "outputs.pt"
ARCHIVED_NCP_ROOT = (
    ACTIVE_ROOT
    / "experiments"
    / "twibot20_seed1_component_ablation_20260619"
    / "C0_full_finetuned_router_routed_only_residual_budget100"
    / "seed_1"
)
ARCHIVED_NCP_OUTPUTS = ARCHIVED_NCP_ROOT / "preparation" / "graph_detector" / "outputs.pt"
ARCHIVED_NCP_RISK_MANIFEST = (
    ACTIVE_ROOT
    / "experiments"
    / "twibot20_seed1_component_ablation_20260619"
    / "router_finetuned_xnew_from_all_nodes"
    / "seed_1"
    / "stages"
    / "estimator_ablation"
    / "risk_manifest.json"
)
EMBEDDING_PATH = DATASET_ROOT / "finetuned_roberta_embeddings_iter_2_seed1.pt"
SPLIT_PATHS = {
    "train": DATASET_ROOT / "train_idx.pt",
    "valid": DATASET_ROOT / "valid_idx.pt",
    "test": DATASET_ROOT / "test_idx.pt",
}

PAPER_ROUTE_PATH = INPUTS_DIR / "paper_reference_tail_budget100_seed1.json"
SAME_HYPEREDGE_ROUTE_PATH = INPUTS_DIR / "same_hyperedge_budget100_seed1.json"
SAME_HYPEREDGE_ROUTER_ROOT = RUN_ROOT / "same_hyperedge_router_seed1"
SAME_HYPEREDGE_STAGE_ROOT = (
    SAME_HYPEREDGE_ROUTER_ROOT / "seed_1" / "stages" / "estimator_ablation"
)
SAME_HYPEREDGE_RISK_MANIFEST = SAME_HYPEREDGE_STAGE_ROOT / "risk_manifest.json"
SAME_HYPEREDGE_RESOLVED_CONFIG = SAME_HYPEREDGE_STAGE_ROOT / "resolved_config.json"
PAPER_GRAPH_ROOT = RUN_ROOT / "paper_formula_seed1"
SAME_HYPEREDGE_GRAPH_ROOT = RUN_ROOT / "same_hyperedge_seed1"

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
    str(KNN_K),
    "--graph_neighbor_num_neighbors",
    str(FANOUT),
    "--gnn_batch_size",
    "512",
    "--graph_training_max_steps",
    str(GRAPH_STEPS),
    "--device",
    "0",
    "--disable_wandb",
    "--force_retrain_backbone",
    "--graph_refine_mode",
    "hyperscan_neighborloader_batch_local_branch",
]


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    return path


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def artifact_entry(path):
    path = require_file(path)
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def clean_env():
    env = {}
    seen = set()
    for key, value in os.environ.items():
        lower = key.lower()
        if lower in seen:
            continue
        seen.add(lower)
        env["Path" if lower == "path" else key] = value
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["HF_HOME"] = str(REPO_ROOT / "models" / "huggingface")
    env["HF_HUB_CACHE"] = str(REPO_ROOT / "models" / "huggingface" / "hub")
    env["TRANSFORMERS_CACHE"] = str(REPO_ROOT / "models" / "huggingface" / "hub")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    env["WANDB_SILENT"] = "true"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def git_state(root):
    def capture(*args):
        proc = subprocess.run(
            ["git", *args],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""

    return {
        "root": str(Path(root).resolve()),
        "revision": capture("rev-parse", "HEAD"),
        "branch": capture("branch", "--show-current"),
        "dirty": bool(capture("status", "--porcelain")),
    }


def update_manifest(manifest):
    write_json(MANIFEST_PATH, manifest)


def run_logged(command, log_path, manifest, job_name):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "job": job_name,
        "status": "running",
        "started_at": now_iso(),
        "cwd": str(ACTIVE_ROOT),
        "command": [str(item) for item in command],
        "log_path": str(log_path.resolve()),
    }
    manifest["runs"].append(entry)
    update_manifest(manifest)
    print(f"[{job_name}] started; log={log_path}", flush=True)
    with log_path.open("wb") as handle:
        proc = subprocess.run(
            [str(item) for item in command],
            cwd=str(ACTIVE_ROOT),
            env=clean_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    entry["finished_at"] = now_iso()
    entry["returncode"] = int(proc.returncode)
    entry["status"] = "completed" if proc.returncode == 0 else "failed"
    update_manifest(manifest)
    print(f"[{job_name}] {entry['status']} (returncode={proc.returncode})", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{job_name} failed; see {log_path}")


def load_index(path):
    import torch

    return torch.load(require_file(path), map_location="cpu", weights_only=False).view(-1).long()


def load_split_indices(node_count):
    splits = {name: load_index(path) for name, path in SPLIT_PATHS.items()}
    seen = set()
    for name, values in splits.items():
        if values.numel() == 0:
            raise ValueError(f"{name} split is empty")
        if int(values.min()) < 0 or int(values.max()) >= int(node_count):
            raise ValueError(f"{name} split contains indices outside [0, {node_count - 1}]")
        current = set(int(item) for item in values.tolist())
        if seen.intersection(current):
            raise ValueError(f"{name} split overlaps an earlier split")
        seen.update(current)
    if len(seen) != int(node_count):
        raise ValueError(f"Canonical splits cover {len(seen)} nodes, expected {node_count}")
    return splits


def labels_from_payload(payload):
    labels = payload["labels"]
    return labels.argmax(dim=1).long() if labels.dim() == 2 else labels.view(-1).long()


def full_reference_tail_risk(
    normalized_x_new,
    confidence_deficit,
    target_idx,
    reference_idx,
    lambda_value,
    batch_size,
    device,
):
    import torch

    reference_idx = reference_idx.view(-1).long().cpu()
    target_idx = target_idx.view(-1).long().cpu()
    references = normalized_x_new[reference_idx].to(device)
    reference_deficit = confidence_deficit[reference_idx].to(device)
    reference_positions = {int(node): position for position, node in enumerate(reference_idx.tolist())}
    batches = []
    with torch.inference_mode():
        for start in range(0, int(target_idx.numel()), int(batch_size)):
            batch_ids = target_idx[start : start + batch_size]
            query = normalized_x_new[batch_ids].to(device)
            similarity = query @ references.T
            distance = torch.sqrt(
                torch.clamp(2.0 - 2.0 * torch.clamp(similarity, -1.0, 1.0), min=0.0)
            )
            weights = torch.exp(-distance / float(lambda_value))
            for row, node_id in enumerate(batch_ids.tolist()):
                self_position = reference_positions.get(int(node_id))
                if self_position is not None:
                    weights[row, self_position] = 0.0
            target_deficit = confidence_deficit[batch_ids].to(device)
            weight_sum = weights.sum(dim=1)
            tail_weight = (
                weights
                * (reference_deficit.unsqueeze(0) >= target_deficit.unsqueeze(1)).to(weights.dtype)
            ).sum(dim=1)
            tau = (1.0 + tail_weight) / (1.0 + weight_sum)
            batches.append((1.0 - tau).cpu())
    return torch.cat(batches).double()


def select_split_budget(indices, risk_scores, budget):
    import numpy as np

    node_ids = indices.view(-1).cpu().numpy().astype(np.int64, copy=False)
    scores = risk_scores.view(-1).cpu().numpy().astype(np.float64, copy=False)
    if node_ids.size != scores.size:
        raise ValueError("Risk scores and target indices do not have the same length")
    count = int(math.floor(float(budget) * int(node_ids.size)))
    order = np.lexsort((node_ids, -scores))
    selected = node_ids[order[:count]].tolist()
    return [int(item) for item in selected], [float(item) for item in scores.tolist()]


def generate_paper_route(manifest):
    import torch

    payload = torch.load(require_file(C2_OUTPUTS), map_location="cpu", weights_only=False)
    required = {"labels", "logits", "logits_lowonly", "x_new"}
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"C2 outputs are missing required keys: {missing}")
    if not torch.equal(payload["logits"], payload["logits_lowonly"]):
        max_delta = float((payload["logits"] - payload["logits_lowonly"]).abs().max())
        raise ValueError(f"C2 is not a pure low-only source; max logit delta={max_delta}")
    logits = payload["logits"].float().cpu()
    x_new = payload["x_new"].float().cpu()
    labels = labels_from_payload(payload)
    if logits.shape[0] != x_new.shape[0] or logits.shape[0] != labels.shape[0]:
        raise ValueError("C2 tensors do not share the same node axis")
    splits = load_split_indices(int(logits.shape[0]))
    probability = torch.softmax(logits, dim=1)
    confidence_deficit = 1.0 - probability.max(dim=1).values
    normalized_x_new = torch.nn.functional.normalize(x_new, p=2, dim=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    selected = {}
    scores_by_split = {}
    for split_name in ("train", "valid", "test"):
        scores = full_reference_tail_risk(
            normalized_x_new=normalized_x_new,
            confidence_deficit=confidence_deficit,
            target_idx=splits[split_name],
            reference_idx=splits["train"],
            lambda_value=FORMULA_LAMBDA,
            batch_size=FORMULA_BATCH_SIZE,
            device=device,
        )
        selected[split_name], split_scores = select_split_budget(
            splits[split_name], scores, BUDGET
        )
        scores_by_split[split_name] = {
            "node_ids": [int(item) for item in splits[split_name].tolist()],
            "risk": split_scores,
        }
    selected["all"] = selected["train"] + selected["valid"] + selected["test"]
    counts = {name: len(selected[name]) for name in ("train", "valid", "test", "all")}
    expected = {
        "train": int(math.floor(BUDGET * splits["train"].numel())),
        "valid": int(math.floor(BUDGET * splits["valid"].numel())),
        "test": int(math.floor(BUDGET * splits["test"].numel())),
    }
    expected["all"] = sum(expected.values())
    if counts != expected:
        raise ValueError(f"Paper route counts {counts} do not match expected {expected}")
    route_payload = {
        "contract": "paper_weighted_reference_tail_router_v1",
        "claim_scope": "seed_1_diagnostic_exact_paper_formula",
        "dataset": "TwiBot-20",
        "seed": SEED,
        "budget": BUDGET,
        "counts": counts,
        "train": selected["train"],
        "valid": selected["valid"],
        "test": selected["test"],
        "all": selected["all"],
        "selected_node_ids_sha256": sha256_json(selected),
        "formula": {
            "confidence_deficit": "u_i = 1 - max_c p_i(c)",
            "weight": "w_ij = exp(-sqrt(2 - 2*cos(x_i,x_j)) / lambda)",
            "tau": "(1 + sum_j w_ij I[u_j >= u_i]) / (1 + sum_j w_ij)",
            "risk": "1 - tau",
            "lambda": FORMULA_LAMBDA,
            "reference_pool": "canonical_train_split",
            "train_target_self_excluded": True,
            "representation": "C2 outputs.pt:x_new",
            "posterior": "C2 outputs.pt:logits (verified identical to logits_lowonly)",
        },
        "scores_by_split": scores_by_split,
        "source_artifacts": {
            "c2_outputs": artifact_entry(C2_OUTPUTS),
            "train_idx": artifact_entry(SPLIT_PATHS["train"]),
            "valid_idx": artifact_entry(SPLIT_PATHS["valid"]),
            "test_idx": artifact_entry(SPLIT_PATHS["test"]),
        },
    }
    write_json(PAPER_ROUTE_PATH, route_payload)
    manifest["artifacts"]["paper_route"] = artifact_entry(PAPER_ROUTE_PATH)
    manifest["paper_route_counts"] = counts
    update_manifest(manifest)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[paper_route] generated counts={counts} device={device}", flush=True)
    return route_payload


def router_command():
    return [
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
        C2_ROOT,
        "--conformal_knn_repr_source",
        "x_new",
        "--conformal_knn_candidate_scope",
        "labeled_full",
        "--conformal_knn_k",
        str(KNN_K),
        "--conformal_knn_learning_mode",
        "ncp_local",
        "--conformal_knn_local_calibration_scope",
        "same_hyperedge",
        "--conformal_knn_score_family_override",
        "same_hyperedge_selected_tail",
        "--conformal_knn_ncp_lambda",
        str(FORMULA_LAMBDA),
        "--conformal_knn_target_top_n",
        "200",
        "--risk_budgets",
        str(BUDGET),
        "--seeds",
        str(SEED),
        "--device",
        "0",
        "--disable_wandb",
        "--artifact_root",
        SAME_HYPEREDGE_ROUTER_ROOT,
    ]


def graph_command(artifact_root, route_path):
    return [
        PYTHON,
        *BASE_GRAPH_ARGS,
        "--embedding_path",
        EMBEDDING_PATH,
        "--seeds",
        str(SEED),
        "--artifact_root",
        artifact_root,
        "--graph_second_view_consumer_scope",
        "routed_only",
        "--graph_second_view_nonconsumer_fallback",
        "low_only",
        "--routed_nodes_path",
        route_path,
    ]


def validate_same_hyperedge_router():
    risk_manifest = read_json(require_file(SAME_HYPEREDGE_RISK_MANIFEST))
    resolved = read_json(require_file(SAME_HYPEREDGE_RESOLVED_CONFIG))
    config = (risk_manifest.get("calibration_metadata") or {}).get("conformal_knn_config") or {}
    expected_config = {
        "candidate_scope": "labeled_full",
        "repr_source": "x_new",
        "learning_mode": "ncp_local",
        "local_calibration_scope": "same_hyperedge",
        "score_family_override": "same_hyperedge_selected_tail",
    }
    for key, expected in expected_config.items():
        actual = config.get(key)
        if actual != expected:
            raise ValueError(
                f"Risk manifest conformal_knn_config.{key}={actual!r}, expected {expected!r}"
            )
    if risk_manifest.get("selected_score_family") != "same_hyperedge_selected_tail":
        raise ValueError(
            "Risk manifest did not select same_hyperedge_selected_tail: "
            f"{risk_manifest.get('selected_score_family')!r}"
        )
    resolved_expectations = {
        "conformal_knn_candidate_scope": "labeled_full",
        "conformal_knn_repr_source": "x_new",
        "conformal_knn_learning_mode": "ncp_local",
        "conformal_knn_local_calibration_scope": "same_hyperedge",
        "conformal_knn_score_family_override": "same_hyperedge_selected_tail",
    }
    for key, expected in resolved_expectations.items():
        if resolved.get(key) != expected:
            raise ValueError(f"Resolved config {key}={resolved.get(key)!r}, expected {expected!r}")
    return risk_manifest, resolved


def run_same_hyperedge_router(manifest):
    if SAME_HYPEREDGE_RISK_MANIFEST.exists() and SAME_HYPEREDGE_RESOLVED_CONFIG.exists():
        validate_same_hyperedge_router()
        manifest["runs"].append(
            {
                "job": "same_hyperedge_router_seed1",
                "status": "reused_existing_validated_output",
                "risk_manifest": str(SAME_HYPEREDGE_RISK_MANIFEST.resolve()),
            }
        )
        update_manifest(manifest)
    else:
        run_logged(
            router_command(),
            LOG_DIR / "same_hyperedge_router_seed1.log",
            manifest,
            "same_hyperedge_router_seed1",
        )
        validate_same_hyperedge_router()
    manifest["artifacts"]["same_hyperedge_risk_manifest"] = artifact_entry(
        SAME_HYPEREDGE_RISK_MANIFEST
    )
    manifest["artifacts"]["same_hyperedge_resolved_config"] = artifact_entry(
        SAME_HYPEREDGE_RESOLVED_CONFIG
    )
    update_manifest(manifest)


def materialize_same_hyperedge_route(manifest):
    risk_manifest, _resolved = validate_same_hyperedge_router()
    item = (risk_manifest.get("selected_nodes_by_budget") or {}).get("budget_100")
    if not isinstance(item, dict):
        raise KeyError("Risk manifest is missing selected_nodes_by_budget.budget_100")
    selected = {
        name: [int(value) for value in item.get(name, [])]
        for name in ("train", "valid", "test", "all")
    }
    if not selected["all"]:
        selected["all"] = selected["train"] + selected["valid"] + selected["test"]
    counts = {name: len(selected[name]) for name in selected}
    expected = {"train": 827, "valid": 236, "test": 118, "all": 1181}
    if counts != expected:
        raise ValueError(f"Same-hyperedge route counts {counts} do not match expected {expected}")
    route_payload = {
        "contract": "derived_same_hyperedge_selected_tail_router_v1",
        "claim_scope": "seed_1_diagnostic_explicit_same_hyperedge",
        "dataset": "TwiBot-20",
        "seed": SEED,
        "budget": BUDGET,
        "counts": counts,
        "train": selected["train"],
        "valid": selected["valid"],
        "test": selected["test"],
        "all": selected["all"],
        "selected_node_ids_sha256": sha256_json(selected),
        "router_contract": {
            "estimator_mode": "conformal_knn_risk_router",
            "repr_source": "x_new",
            "candidate_scope": "labeled_full",
            "knn_k": KNN_K,
            "learning_mode": "ncp_local",
            "local_calibration_scope": "same_hyperedge",
            "score_family": "same_hyperedge_selected_tail",
            "lambda": FORMULA_LAMBDA,
        },
        "source_risk_manifest": artifact_entry(SAME_HYPEREDGE_RISK_MANIFEST),
        "source_resolved_config": artifact_entry(SAME_HYPEREDGE_RESOLVED_CONFIG),
    }
    write_json(SAME_HYPEREDGE_ROUTE_PATH, route_payload)
    manifest["artifacts"]["same_hyperedge_route"] = artifact_entry(SAME_HYPEREDGE_ROUTE_PATH)
    manifest["same_hyperedge_route_counts"] = counts
    update_manifest(manifest)
    print(f"[same_hyperedge_route] materialized counts={counts}", flush=True)
    return route_payload


def graph_outputs_path(artifact_root):
    return Path(artifact_root) / "seed_1" / "preparation" / "graph_detector" / "outputs.pt"


def graph_manifest_path(artifact_root):
    return Path(artifact_root) / "seed_1" / "preparation" / "graph_detector" / "manifest.json"


def validate_graph_arm(artifact_root, route_path):
    import torch

    output_path = require_file(graph_outputs_path(artifact_root))
    graph_manifest_file = require_file(graph_manifest_path(artifact_root))
    graph_manifest = read_json(graph_manifest_file)
    second_view = graph_manifest.get("second_view") or {}
    graph_refine = graph_manifest.get("graph_refine") or {}
    training_loader = graph_manifest.get("training_loader") or {}
    checks = {
        "seed": graph_manifest.get("seed") == SEED,
        "consumer_scope": second_view.get("consumer_scope") == "routed_only",
        "nonconsumer_fallback": second_view.get("nonconsumer_fallback") == "low_only",
        "fusion": second_view.get("fusion") == "residual",
        "hypergraph_backend": second_view.get("hypergraph_backend") == "dhg",
        "knn_k": int(float(graph_refine.get("knn_k", -1))) == KNN_K,
        "fanout": int(training_loader.get("neighbor_num_neighbors", -1)) == FANOUT,
        "optimizer_steps": int(training_loader.get("optimizer_steps", -1)) == GRAPH_STEPS,
        "route_path": Path(str(second_view.get("consumer_mask_source", ""))).resolve()
        == Path(route_path).resolve(),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Graph arm contract validation failed for {artifact_root}: {failed}")
    route = read_json(route_path)
    expected_ids = set(int(item) for item in route["all"])
    outputs = torch.load(output_path, map_location="cpu", weights_only=False)
    observed_mask = outputs["highorder_consumer_mask"].view(-1).bool()
    observed_ids = set(int(item) for item in observed_mask.nonzero(as_tuple=False).view(-1).tolist())
    if observed_ids != expected_ids:
        raise ValueError(
            f"Observed routed mask differs from route input for {artifact_root}: "
            f"observed={len(observed_ids)} expected={len(expected_ids)}"
        )
    return {
        "checks": checks,
        "outputs": artifact_entry(output_path),
        "graph_manifest": artifact_entry(graph_manifest_file),
        "checkpoint": artifact_entry(output_path.parent / "checkpoint.pt"),
    }


def run_graph_arm(name, artifact_root, route_path, manifest):
    output_path = graph_outputs_path(artifact_root)
    if output_path.exists():
        validation = validate_graph_arm(artifact_root, route_path)
        manifest["runs"].append(
            {
                "job": name,
                "status": "reused_existing_validated_output",
                "outputs_path": str(output_path.resolve()),
            }
        )
    else:
        run_logged(
            graph_command(artifact_root, route_path),
            LOG_DIR / f"{name}.log",
            manifest,
            name,
        )
        validation = validate_graph_arm(artifact_root, route_path)
    manifest["graph_contract_checks"][name] = validation
    update_manifest(manifest)


def binary_metrics(labels, predictions):
    from sklearn.metrics import accuracy_score, f1_score, precision_score

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "wrong": int((predictions != labels).sum()),
    }


def transition_counts(labels, base_predictions, final_predictions):
    base_correct = base_predictions == labels
    final_correct = final_predictions == labels
    return {
        "right_to_right": int((base_correct & final_correct).sum()),
        "right_to_wrong": int((base_correct & ~final_correct).sum()),
        "wrong_to_right": int((~base_correct & final_correct).sum()),
        "wrong_to_wrong": int((~base_correct & ~final_correct).sum()),
    }


def route_overlap(test_route, archived_test_route):
    current = set(int(item) for item in test_route)
    archived = set(int(item) for item in archived_test_route)
    union = current.union(archived)
    return {
        "route_overlap_count_archived_test": len(current.intersection(archived)),
        "route_jaccard_archived_test": float(len(current.intersection(archived)) / len(union))
        if union
        else 1.0,
    }


def route_source_diagnostics(paper_route, same_hyperedge_route):
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

    source = torch.load(require_file(C2_OUTPUTS), map_location="cpu", weights_only=False)
    labels = labels_from_payload(source)
    predictions = source["logits"].argmax(dim=1).long()
    test_idx = load_index(SPLIT_PATHS["test"])
    wrong = predictions != labels
    test_wrong = int(wrong[test_idx].sum())
    base_error_rate = float(test_wrong / int(test_idx.numel()))
    same_risk_manifest = read_json(require_file(SAME_HYPEREDGE_RISK_MANIFEST))
    same_risk = np.asarray(same_risk_manifest["risk_score"], dtype=np.float64)

    def score_row(name, route, test_scores):
        node_ids = np.asarray(route["scores_by_split"]["test"]["node_ids"], dtype=np.int64) if name == "paper_formula" else test_idx.numpy()
        error_labels = wrong[torch.from_numpy(node_ids)].numpy().astype(np.int64, copy=False)
        selected_ids = torch.tensor(route["test"], dtype=torch.long)
        captured = int(wrong[selected_ids].sum())
        routed_count = int(selected_ids.numel())
        routed_precision = float(captured / routed_count) if routed_count else 0.0
        return {
            "error_auroc": float(roc_auc_score(error_labels, test_scores)),
            "error_auprc": float(average_precision_score(error_labels, test_scores)),
            "routed_count": routed_count,
            "routed_error_count": captured,
            "error_capture_at_budget": float(captured / test_wrong) if test_wrong else 0.0,
            "error_precision_at_budget": routed_precision,
            "lift_vs_random": float(routed_precision / base_error_rate) if base_error_rate else 0.0,
        }

    paper_scores = np.asarray(
        paper_route["scores_by_split"]["test"]["risk"], dtype=np.float64
    )
    same_scores = same_risk[test_idx.numpy()]
    paper_test = set(int(item) for item in paper_route["test"])
    same_test = set(int(item) for item in same_hyperedge_route["test"])
    paper_all = set(int(item) for item in paper_route["all"])
    same_all = set(int(item) for item in same_hyperedge_route["all"])
    return {
        "source": "C2 low-only logits and x_new",
        "source_test_count": int(test_idx.numel()),
        "source_test_error_count": test_wrong,
        "source_test_error_rate": base_error_rate,
        "paper_formula": score_row("paper_formula", paper_route, paper_scores),
        "same_hyperedge": score_row("same_hyperedge", same_hyperedge_route, same_scores),
        "paper_vs_same_hyperedge_route_overlap": {
            "test_overlap_count": len(paper_test.intersection(same_test)),
            "test_jaccard": float(len(paper_test.intersection(same_test)) / len(paper_test.union(same_test))),
            "all_overlap_count": len(paper_all.intersection(same_all)),
            "all_jaccard": float(len(paper_all.intersection(same_all)) / len(paper_all.union(same_all))),
        },
    }


def output_rows(label, outputs_path, archived_test_route, source_route_path=None):
    import numpy as np
    import torch

    payload = torch.load(require_file(outputs_path), map_location="cpu", weights_only=False)
    labels = labels_from_payload(payload)
    test_idx = load_index(SPLIT_PATHS["test"])
    base_pred = payload["logits_lowonly"].argmax(dim=1).long()
    final_pred = payload["logits"].argmax(dim=1).long()
    observed_mask = payload["highorder_consumer_mask"].view(-1).bool()
    test_mask = observed_mask[test_idx]
    test_ids = test_idx.cpu().numpy()
    route_ids = test_ids[test_mask.cpu().numpy()].tolist()
    y = labels[test_idx].cpu().numpy()
    base = base_pred[test_idx].cpu().numpy()
    final = final_pred[test_idx].cpu().numpy()
    nonrouted_changes = int((base[~test_mask.cpu().numpy()] != final[~test_mask.cpu().numpy()]).sum())
    if nonrouted_changes != 0:
        raise ValueError(f"{label} changed {nonrouted_changes} non-routed test predictions")
    overlap = route_overlap(route_ids, archived_test_route)
    final_row = {
        "source": label,
        "row_kind": "final",
        **binary_metrics(y, final),
        **transition_counts(y, base, final),
        "routed_count_test": int(test_mask.sum()),
        "routed_count_all": int(observed_mask.sum()),
        "nonrouted_prediction_changes": nonrouted_changes,
        "outputs_path": str(Path(outputs_path).resolve()),
        "route_path": str(Path(source_route_path).resolve()) if source_route_path else "",
        **overlap,
    }
    low_row = {
        "source": label,
        "row_kind": "low_only",
        **binary_metrics(y, base),
        **transition_counts(y, base, base),
        "routed_count_test": int(test_mask.sum()),
        "routed_count_all": int(observed_mask.sum()),
        "nonrouted_prediction_changes": 0,
        "outputs_path": str(Path(outputs_path).resolve()),
        "route_path": str(Path(source_route_path).resolve()) if source_route_path else "",
        **overlap,
    }
    return [final_row, low_row], payload, labels, test_idx


def fixed_weight_paper_route_swap_row(archived_payload, labels, test_idx, paper_route, archived_test_route):
    import torch

    route_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    route_mask[torch.tensor(paper_route["all"], dtype=torch.long)] = True
    swapped_logits = torch.where(
        route_mask.unsqueeze(1),
        archived_payload["logits_hnn"],
        archived_payload["logits_lowonly"],
    )
    base_pred = archived_payload["logits_lowonly"].argmax(dim=1).long()[test_idx].cpu().numpy()
    swapped_pred = swapped_logits.argmax(dim=1).long()[test_idx].cpu().numpy()
    y = labels[test_idx].cpu().numpy()
    test_route = [int(item) for item in paper_route["test"]]
    return {
        "source": "fixed_weight_paper_route_swap",
        "row_kind": "diagnostic_no_retraining",
        **binary_metrics(y, swapped_pred),
        **transition_counts(y, base_pred, swapped_pred),
        "routed_count_test": len(test_route),
        "routed_count_all": len(paper_route["all"]),
        "nonrouted_prediction_changes": 0,
        "outputs_path": str(ARCHIVED_NCP_OUTPUTS.resolve()),
        "route_path": str(PAPER_ROUTE_PATH.resolve()),
        **route_overlap(test_route, archived_test_route),
    }


def write_reports(manifest, paper_route):
    import torch

    archived_payload = torch.load(
        require_file(ARCHIVED_NCP_OUTPUTS), map_location="cpu", weights_only=False
    )
    archived_test_idx = load_index(SPLIT_PATHS["test"])
    archived_mask = archived_payload["highorder_consumer_mask"].view(-1).bool()
    archived_test_route = [
        int(item)
        for item in archived_test_idx[archived_mask[archived_test_idx]].view(-1).tolist()
    ]
    rows = []
    archived_rows, archived_payload, labels, test_idx = output_rows(
        "archived_ncp",
        ARCHIVED_NCP_OUTPUTS,
        archived_test_route,
    )
    rows.extend(archived_rows)
    rows.append(
        fixed_weight_paper_route_swap_row(
            archived_payload, labels, test_idx, paper_route, archived_test_route
        )
    )
    paper_rows, _paper_payload, _paper_labels, _paper_test_idx = output_rows(
        "paper_formula",
        graph_outputs_path(PAPER_GRAPH_ROOT),
        archived_test_route,
        PAPER_ROUTE_PATH,
    )
    same_rows, _same_payload, _same_labels, _same_test_idx = output_rows(
        "same_hyperedge",
        graph_outputs_path(SAME_HYPEREDGE_GRAPH_ROOT),
        archived_test_route,
        SAME_HYPEREDGE_ROUTE_PATH,
    )
    rows.extend(paper_rows)
    rows.extend(same_rows)
    same_hyperedge_route = read_json(SAME_HYPEREDGE_ROUTE_PATH)
    source_diagnostics = route_source_diagnostics(paper_route, same_hyperedge_route)

    by_key = {(row["source"], row["row_kind"]): row for row in rows}
    archived_final = by_key[("archived_ncp", "final")]
    for row in rows:
        row["delta_accuracy_pp_vs_archived_ncp"] = 100.0 * (
            row["accuracy"] - archived_final["accuracy"]
        )
        row["delta_macro_precision_pp_vs_archived_ncp"] = 100.0 * (
            row["macro_precision"] - archived_final["macro_precision"]
        )
        row["delta_macro_f1_pp_vs_archived_ncp"] = 100.0 * (
            row["macro_f1"] - archived_final["macro_f1"]
        )
        own_low = by_key.get((row["source"], "low_only"))
        row["delta_accuracy_pp_vs_own_low_only"] = (
            100.0 * (row["accuracy"] - own_low["accuracy"]) if own_low else None
        )
        row["delta_macro_f1_pp_vs_own_low_only"] = (
            100.0 * (row["macro_f1"] - own_low["macro_f1"]) if own_low else None
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "seed1_router_alignment.csv"
    json_path = REPORT_DIR / "seed1_router_alignment.json"
    fields = [
        "source",
        "row_kind",
        "accuracy",
        "macro_precision",
        "macro_f1",
        "wrong",
        "right_to_right",
        "right_to_wrong",
        "wrong_to_right",
        "wrong_to_wrong",
        "routed_count_test",
        "routed_count_all",
        "nonrouted_prediction_changes",
        "route_overlap_count_archived_test",
        "route_jaccard_archived_test",
        "delta_accuracy_pp_vs_archived_ncp",
        "delta_macro_precision_pp_vs_archived_ncp",
        "delta_macro_f1_pp_vs_archived_ncp",
        "delta_accuracy_pp_vs_own_low_only",
        "delta_macro_f1_pp_vs_own_low_only",
        "outputs_path",
        "route_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    report = {
        "contract": "twibot20_seed1_router_alignment_report_v1",
        "claim_scope": "seed_1_diagnostic_only_not_five_seed_main_result_repair",
        "metric_contract": "canonical deduplicated test_idx.pt; macro precision and macro F1",
        "dataset": "TwiBot-20",
        "seed": SEED,
        "fixed_contract": manifest["fixed_contract"],
        "route_source_diagnostics": source_diagnostics,
        "rows": rows,
        "critical_boundaries": [
            "P-Paper and P-Hyperedge are different routing methods and remain separate rows.",
            "fixed_weight_paper_route_swap changes the archived inference mask without retraining and is diagnostic only.",
            "A single seed cannot establish five-seed significance or repair the published mean/std provenance.",
        ],
        "source_artifacts": {
            "archived_ncp_outputs": artifact_entry(ARCHIVED_NCP_OUTPUTS),
            "archived_ncp_risk_manifest": artifact_entry(ARCHIVED_NCP_RISK_MANIFEST),
            "paper_route": artifact_entry(PAPER_ROUTE_PATH),
            "same_hyperedge_route": artifact_entry(SAME_HYPEREDGE_ROUTE_PATH),
            "paper_outputs": artifact_entry(graph_outputs_path(PAPER_GRAPH_ROOT)),
            "same_hyperedge_outputs": artifact_entry(graph_outputs_path(SAME_HYPEREDGE_GRAPH_ROOT)),
        },
    }
    write_json(json_path, report)
    manifest["artifacts"]["comparison_csv"] = artifact_entry(csv_path)
    manifest["artifacts"]["comparison_json"] = artifact_entry(json_path)
    manifest["result_rows"] = rows
    manifest["route_source_diagnostics"] = source_diagnostics
    update_manifest(manifest)
    print(json.dumps(rows, indent=2), flush=True)
    return report


def initial_manifest(dry_run):
    import torch

    previous_invocations = []
    if MANIFEST_PATH.exists():
        previous = read_json(MANIFEST_PATH)
        previous_invocations.extend(previous.get("previous_invocations", []))
        previous_invocations.append(
            {
                "created_at": previous.get("created_at"),
                "completed_at": previous.get("completed_at"),
                "failed_at": previous.get("failed_at"),
                "status": previous.get("status"),
                "error_type": previous.get("error_type"),
                "error": previous.get("error"),
                "runs": previous.get("runs", []),
            }
        )
    required = [
        C2_OUTPUTS,
        ARCHIVED_NCP_OUTPUTS,
        ARCHIVED_NCP_RISK_MANIFEST,
        EMBEDDING_PATH,
        *SPLIT_PATHS.values(),
    ]
    for path in required:
        require_file(path)
    return {
        "contract": "twibot20_seed1_router_alignment_experiment_v1",
        "status": "dry_run_initializing" if dry_run else "running",
        "created_at": now_iso(),
        "claim_scope": "seed_1_diagnostic_only_not_five_seed_main_result_repair",
        "run_root": str(RUN_ROOT.resolve()),
        "historical_artifacts_read_only": True,
        "python_environment": {
            "executable": str(PYTHON),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "kmp_duplicate_lib_ok": "TRUE",
            "openmp_provenance_note": (
                "The historical llmbot conda environment fails with duplicate libiomp5md initialization. "
                "This isolated run uses the working system Anaconda interpreter and records the standard "
                "KMP_DUPLICATE_LIB_OK workaround used by historical runners."
            ),
        },
        "git": {
            "workspace": git_state(REPO_ROOT),
            "llmbot": git_state(ACTIVE_ROOT),
        },
        "fixed_contract": {
            "dataset": "TwiBot-20 canonical labeled split",
            "seed": SEED,
            "embedding": str(EMBEDDING_PATH.resolve()),
            "graph_backbone": "rgcn_hyperscan_dhg_nodeinput",
            "graph_node_input_family": "hyperscan_meta_tweet_proxy",
            "knn_k": KNN_K,
            "fanout": FANOUT,
            "hypergraph_backend": "dhg",
            "consumer_scope": "routed_only",
            "nonconsumer_fallback": "low_only",
            "fusion": "residual",
            "routing_budget": BUDGET,
            "graph_optimizer_steps": GRAPH_STEPS,
        },
        "source_artifacts": {
            "c2_outputs": artifact_entry(C2_OUTPUTS),
            "archived_ncp_outputs": artifact_entry(ARCHIVED_NCP_OUTPUTS),
            "archived_ncp_risk_manifest": artifact_entry(ARCHIVED_NCP_RISK_MANIFEST),
            "embedding": artifact_entry(EMBEDDING_PATH),
            "train_idx": artifact_entry(SPLIT_PATHS["train"]),
            "valid_idx": artifact_entry(SPLIT_PATHS["valid"]),
            "test_idx": artifact_entry(SPLIT_PATHS["test"]),
        },
        "planned_commands": {
            "same_hyperedge_router": [str(item) for item in router_command()],
            "paper_graph": [str(item) for item in graph_command(PAPER_GRAPH_ROOT, PAPER_ROUTE_PATH)],
            "same_hyperedge_graph": [
                str(item)
                for item in graph_command(SAME_HYPEREDGE_GRAPH_ROOT, SAME_HYPEREDGE_ROUTE_PATH)
            ],
        },
        "previous_invocations": previous_invocations,
        "runs": [],
        "artifacts": {},
        "graph_contract_checks": {},
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the isolated TwiBot-20 seed-1 paper-formula and same-hyperedge alignment experiment."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and validate the exact paper route and command plan without launching router/training jobs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = initial_manifest(args.dry_run)
    update_manifest(manifest)
    try:
        paper_route = generate_paper_route(manifest)
        if args.dry_run:
            manifest["status"] = "dry_run_completed"
            manifest["completed_at"] = now_iso()
            manifest["dry_run_checks"] = {
                "paper_route_counts": paper_route["counts"],
                "paper_route_train_reference_pool": True,
                "paper_route_train_self_exclusion": True,
                "same_hyperedge_command_explicit": True,
                "graph_output_roots_isolated": all(
                    str(path.resolve()).startswith(str(RUN_ROOT.resolve()))
                    for path in (PAPER_GRAPH_ROOT, SAME_HYPEREDGE_GRAPH_ROOT)
                ),
            }
            update_manifest(manifest)
            print(json.dumps(manifest["dry_run_checks"], indent=2), flush=True)
            return

        run_same_hyperedge_router(manifest)
        materialize_same_hyperedge_route(manifest)
        run_graph_arm(
            "paper_formula_graph_seed1", PAPER_GRAPH_ROOT, PAPER_ROUTE_PATH, manifest
        )
        run_graph_arm(
            "same_hyperedge_graph_seed1",
            SAME_HYPEREDGE_GRAPH_ROOT,
            SAME_HYPEREDGE_ROUTE_PATH,
            manifest,
        )
        write_reports(manifest, paper_route)
        manifest["status"] = "completed"
        manifest["completed_at"] = now_iso()
        update_manifest(manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = now_iso()
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        update_manifest(manifest)
        raise


if __name__ == "__main__":
    main()
