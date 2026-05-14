"""Public API for the LLMbot utils package."""
import csv
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from utils.calibration import ATSCalibrator, SemanticCalibrator, build_account_text, encode_accounts
from utils.losses import EdlLoss, SupConLoss, kl_divergence_dirichlet, r_edl_loss
from utils.metrics import compute_aurc, compute_brier, compute_ece, compute_ece_from_confidence, compute_nll
from utils.misc import (
    compute_directed_structural_features,
    configure_disabled_env,
    enable_wandb,
    jsd_probs,
    weighted_belief_fusion,
)

try:
    import wandb
except ImportError:
    wandb = None


def safe_torch_load(path, *args, weights_only=True, **kwargs):
    """Load torch artifacts with PyTorch's safer weights-only default when available."""
    try:
        return torch.load(path, *args, weights_only=weights_only, **kwargs)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, *args, **kwargs)


def tensor_sha256(value):
    tensor = value.detach().cpu().contiguous() if torch.is_tensor(value) else torch.as_tensor(value).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_tensor_to_python(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): safe_tensor_to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_tensor_to_python(item) for item in value]
    return value


def write_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(safe_tensor_to_python(payload), handle, indent=2, sort_keys=True)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_torch(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)


def read_torch(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return safe_torch_load(path, map_location="cpu")


def write_text(path, text):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(text))


def write_csv_rows(path, fieldnames, rows):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: safe_tensor_to_python(value) for key, value in row.items()})


def _run_git(command, cwd):
    return subprocess.check_output(
        command,
        cwd=cwd,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _format_git_error(exc):
    details = [str(exc)]
    for attr in ("output", "stdout", "stderr"):
        value = getattr(exc, attr, None)
        if value:
            details.append(str(value))
    return " | ".join(details)


def capture_code_metadata(cwd=None):
    cwd = cwd or os.getcwd()
    metadata = {
        "commit": "unknown",
        "dirty": None,
        "cwd": str(cwd),
        "git_error": None,
    }
    try:
        metadata["commit"] = _run_git(["git", "rev-parse", "HEAD"], cwd).strip()
        try:
            _run_git(["git", "diff", "--quiet"], cwd)
            _run_git(["git", "diff", "--cached", "--quiet"], cwd)
            metadata["dirty"] = False
        except subprocess.CalledProcessError:
            metadata["dirty"] = True
        return metadata
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        metadata["git_error"] = _format_git_error(exc)
        return metadata


def capture_code_commit(cwd=None):
    return capture_code_metadata(cwd)["commit"]


def build_experiment_root(args, seed):
    if getattr(args, "artifact_root", None):
        base = Path(args.artifact_root)
    else:
        base = Path(args.experiment_name)
    return ensure_dir(base / f"seed_{seed}")


def build_stage_dir(experiment_root, stage):
    return ensure_dir(Path(experiment_root) / "stages" / stage)


def save_stage_artifacts(stage_dir, artifact_bundle):
    stage_dir = ensure_dir(stage_dir)
    for name, payload in artifact_bundle.items():
        if payload is None:
            continue
        if name.endswith(".pt"):
            write_torch(stage_dir / name, payload)
        elif torch.is_tensor(payload):
            write_torch(stage_dir / f"{name}.pt", payload)
        elif isinstance(payload, dict):
            write_json(stage_dir / f"{name}.json", payload)
        else:
            write_json(stage_dir / f"{name}.json", {"value": payload})


def load_stage_json(stage_dir, name, default=None):
    return read_json(Path(stage_dir) / f"{name}.json", default=default)


def load_stage_tensor(stage_dir, name, default=None):
    return read_torch(Path(stage_dir) / f"{name}.pt", default=default)


class NullRun:
    def log(self, *args, **kwargs):
        return None

    def finish(self):
        return None


def seed_setting(seed_number):
    random.seed(seed_number)
    os.environ["PYTHONHASHSEED"] = str(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def setup_wandb(args, seed):
    if getattr(args, "disable_wandb", False) or not getattr(args, "project_name", None) or wandb is None:
        return NullRun()
    return wandb.init(
        project=args.project_name,
        name=f"{args.experiment_name}_seed_{seed}",
        config=vars(args),
    )


def resolve_dataset_path(dataset):
    dataset = str(dataset)
    aliases = [dataset]
    if dataset.lower() == "twibot-20":
        aliases.extend(["TwiBot-20", "Twibot-20"])
    candidate_roots = [
        Path("./datasets"),
        Path("../datasets"),
        Path(__file__).resolve().parents[2] / "datasets",
        Path(__file__).resolve().parents[3] / "datasets",
    ]
    candidates = [root / alias for root in candidate_roots for alias in aliases]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for root in candidate_roots:
        if not root.exists():
            continue
        lowered = {child.name.lower(): child for child in root.iterdir() if child.is_dir()}
        matched = lowered.get(dataset.lower())
        if matched is not None:
            return matched
    return candidates[0]


def load_raw_data(dataset, use_GNN):
    data_filepath = resolve_dataset_path(dataset)
    print(f"Loading data from {data_filepath} ...")
    train_idx = safe_torch_load(data_filepath / "train_idx.pt")
    valid_idx = safe_torch_load(data_filepath / "valid_idx.pt")
    test_idx = safe_torch_load(data_filepath / "test_idx.pt")
    with open(data_filepath / "norm_user_text.json", "r", encoding="utf-8") as handle:
        user_text = json.load(handle)
    labels = safe_torch_load(data_filepath / "labels.pt")
    if use_GNN:
        edge_index = safe_torch_load(data_filepath / "edge_index.pt")
        edge_type = safe_torch_load(data_filepath / "edge_type.pt")
        return {
            "dataset_path": data_filepath,
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "test_idx": test_idx,
            "user_text": user_text,
            "labels": labels,
            "edge_index": edge_index,
            "edge_type": edge_type,
        }
    return {
        "dataset_path": data_filepath,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "user_text": user_text,
        "labels": labels,
    }


def load_distilled_knowledge(from_which_model, intermediate_data_filepath, iter):
    if from_which_model == "LM":
        embeddings = safe_torch_load(intermediate_data_filepath / f"embeddings_iter_{iter}.pt")
        soft_labels = safe_torch_load(intermediate_data_filepath / f"soft_labels_iter_{iter}.pt")
        return embeddings, soft_labels
    if from_which_model in {"GNN", "MLP"}:
        return safe_torch_load(intermediate_data_filepath / f"soft_labels_iter_{iter}.pt")
    raise ValueError('"from_which_model" should be "LM", "GNN" or "MLP".')


def prepare_path(experiment_name):
    experiment_path = Path(experiment_name)
    ckpt_filepath = experiment_path / "checkpoints"
    mlp_ckpt_filepath = ckpt_filepath / "MLP"
    lm_ckpt_filepath = ckpt_filepath / "LM"
    gnn_ckpt_filepath = ckpt_filepath / "GNN"
    mlp_kd_ckpt_filepath = experiment_path / "MLP_KD"
    lm_prt_ckpt_filepath = ckpt_filepath / "LM_pretrain"
    gnn_prt_ckpt_filepath = ckpt_filepath / "GNN_pretrain"

    for path in [
        lm_prt_ckpt_filepath,
        gnn_prt_ckpt_filepath,
        lm_ckpt_filepath,
        gnn_ckpt_filepath,
        mlp_kd_ckpt_filepath,
        mlp_ckpt_filepath,
    ]:
        path.mkdir(exist_ok=True, parents=True)

    lm_intermediate_data_filepath = experiment_path / "intermediate" / "LM"
    gnn_intermediate_data_filepath = experiment_path / "intermediate" / "GNN"
    mlp_intermediate_data_filepath = experiment_path / "intermediate" / "MLP"
    for path in [lm_intermediate_data_filepath, gnn_intermediate_data_filepath, mlp_intermediate_data_filepath]:
        path.mkdir(exist_ok=True, parents=True)

    return (
        lm_prt_ckpt_filepath,
        gnn_prt_ckpt_filepath,
        mlp_kd_ckpt_filepath,
        lm_ckpt_filepath,
        gnn_ckpt_filepath,
        mlp_ckpt_filepath,
        lm_intermediate_data_filepath,
        gnn_intermediate_data_filepath,
        mlp_intermediate_data_filepath,
    )


def reset_split(n_nodes, ratio):
    idx = torch.randperm(n_nodes)
    split = list(map(int, ratio.split(",")))
    train_ratio = split[0] / sum(split)
    valid_ratio = split[1] / sum(split)

    train_idx = idx[: int(train_ratio * n_nodes)]
    valid_idx = idx[int(train_ratio * n_nodes) : int((train_ratio + valid_ratio) * n_nodes)]
    test_idx = idx[int((train_ratio + valid_ratio) * n_nodes) :]
    return train_idx, valid_idx, test_idx


__all__ = [
    "ATSCalibrator",
    "EdlLoss",
    "SemanticCalibrator",
    "SupConLoss",
    "build_account_text",
    "build_experiment_root",
    "build_stage_dir",
    "capture_code_commit",
    "capture_code_metadata",
    "compute_aurc",
    "compute_brier",
    "compute_directed_structural_features",
    "compute_ece",
    "compute_ece_from_confidence",
    "compute_nll",
    "configure_disabled_env",
    "enable_wandb",
    "encode_accounts",
    "ensure_dir",
    "jsd_probs",
    "kl_divergence_dirichlet",
    "load_distilled_knowledge",
    "load_raw_data",
    "load_stage_json",
    "load_stage_tensor",
    "prepare_path",
    "r_edl_loss",
    "read_json",
    "read_torch",
    "reset_split",
    "resolve_dataset_path",
    "safe_tensor_to_python",
    "safe_torch_load",
    "save_stage_artifacts",
    "seed_setting",
    "setup_wandb",
    "tensor_sha256",
    "wandb",
    "weighted_belief_fusion",
    "write_csv_rows",
    "write_json",
    "write_text",
    "write_torch",
]
