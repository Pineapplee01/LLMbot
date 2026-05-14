import importlib.util
import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from time import perf_counter
from model_building import (
    PhaseAInputAdapter,
    build_GNN_model,
    build_LM_model,
    build_estimator,
    mode_metadata,
    _idx_tensor,
    _labels_to_index,
    _score_logits,
    phase_a_project_dim,
    phase_a_projector,
    resolve_g0_feature_bundle,
)
from dataloader import build_LM_dataloader, build_GNN_dataloader, build_MLP_dataloader
import os
import json
import math
from pathlib import Path
from types import SimpleNamespace
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn.models import MLP
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils import (
    build_experiment_root,
    build_stage_dir,
    capture_code_metadata,
    ensure_dir,
    save_stage_artifacts,
    safe_torch_load,
    tensor_sha256,
    write_csv_rows,
    write_json,
    write_text,
    write_torch,
)
from subgroups import build_subgroup_manifests
from utils import load_distilled_knowledge, prepare_path, read_json
from subgroups import structural_features
from estimators import (
    CalibratedMultiViewResidualRouter,
    CalibratedResidualRiskHardNodeSelector,
    FINAL_OUTPUT_RANKER_BUDGETS,
    GETSStylePostHocResidualRiskSelector,
    GlanceForContextResidualRiskSelector,
    RESIDUAL_RISK_PAPER_BUDGETS,
    build_probe_features,
    fit_temperature_scaling,
    paired_bootstrap_residual_risk_delta,
    parse_budget_list,
    random_expected_residual_risk_budget_curve,
    residual_risk_budget_curve,
    residual_risk_metrics,
    risk_metrics,
    router_budget_curve,
    stage2_acceptance_gate,
)


PHASE_A_CONTRACT = "phase_a_semantic_source_x_gnn_backbone"
PHASE_A_DISABLED_COMPONENTS = {
    "estimator_mode": "none",
    "semantic_mode": "off",
    "repair_mode": "noop",
    "selector_mode": "none",
    "appendix_mode": "none",
}

FROZEN_G0_CONTRACT = "frozen_g0_v1"
GATS_GATE_NAME = "gats_faithful"
GATS_CONTRACT = "faithful_gats_anchor_v1"

FORMAL_STAGE_GATES = {
    "estimator_matrix": [GATS_GATE_NAME],
    "semantic_matrix": ["lagnn_near_faithful"],
    "semantic_source_matrix": ["lagnn_near_faithful"],
    "repair_matrix": ["gnnguard_local", "cs_conditional_supporting"],
    "selector_matrix": ["gnnguard_local", "cs_conditional_supporting"],
    "positioning_matrix": ["gnnguard_local", "cs_conditional_supporting"],
    "backbone_stress": ["gnnguard_local", "cs_conditional_supporting"],
}


class MissingFrozenArtifactError(RuntimeError):
    pass


def run_phase_a_matrix(args, seed, data, stage_dir, base_bundle):
    """Phase A matrix: semantic source 鑴?GNN backbone comparison.

    This is a placeholder 閳?the full matrix is driven by main.py's
    module-controls entry (--emb_path + --use_GNN) which calls
    build_or_load_frozen_g0 per cell. The StageRunner path delegates here
    but the actual cell execution happens in main.py.
    """
    raise NotImplementedError(
        "Phase A matrix via StageRunner is not yet implemented. "
        "Use main.py module-controls entry: "
        "python main.py --emb_path /path/to/emb.pt "
        "--use_GNN --GNN_model rgcn --seeds 1"
    )


def _as_long_cpu_tensor(idx):
    if idx is None:
        return torch.empty(0, dtype=torch.long)
    if torch.is_tensor(idx):
        return idx.detach().cpu().long().reshape(-1)
    return torch.tensor(idx, dtype=torch.long).reshape(-1)


def count_trainable_parameters(module):
    if module is None:
        return 0
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def split_provenance(data):
    return {
        "train": {
            "size": int(_idx_tensor(data["train_idx"]).numel()),
            "sha256": tensor_sha256(_idx_tensor(data["train_idx"])),
        },
        "valid": {
            "size": int(_idx_tensor(data["valid_idx"]).numel()),
            "sha256": tensor_sha256(_idx_tensor(data["valid_idx"])),
        },
        "test": {
            "size": int(_idx_tensor(data["test_idx"]).numel()),
            "sha256": tensor_sha256(_idx_tensor(data["test_idx"])),
        },
    }


def frozen_g0_dir(experiment_root):
    return ensure_dir(Path(experiment_root) / "frozen" / "g0")


def _is_better(candidate, incumbent):
    if incumbent is None:
        return True
    if candidate["macro_f1"] > incumbent["macro_f1"]:
        return True
    if candidate["macro_f1"] == incumbent["macro_f1"] and candidate["loss"] < incumbent["loss"]:
        return True
    return False


def _model_config(args, features, device):
    return {
        "GNN_model": getattr(args, "GNN_model", "rgcn"),
        "optimizer": getattr(args, "optimizer_GNN", "adamw"),
        "gnn_n_layers": getattr(args, "n_layers", 2),
        "n_layers": getattr(args, "n_layers", 2),
        "n_relations": getattr(args, "n_relations", 2),
        "activation": getattr(args, "activation", "leakyrelu"),
        "dropout": getattr(args, "GNN_dropout", 0.4),
        "gnn_hidden_dim": getattr(args, "hidden_dim", 128),
        "hidden_dim": getattr(args, "hidden_dim", 128),
        "lm_input_dim": int(features.shape[1]),
        "SimpleHGN_att_res": getattr(args, "SimpleHGN_att_res", 0.2),
        "att_heads": getattr(args, "att_heads", 8),
        "RGT_semantic_heads": getattr(args, "RGT_semantic_heads", 8),
        "device": device,
    }


def train_frozen_g0(args, seed, data, experiment_root):
    out_dir = frozen_g0_dir(experiment_root)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    feature_bundle = resolve_g0_feature_bundle(args, data)
    features = feature_bundle["features"]
    raw_features = feature_bundle["raw_features"]
    feature_manifest = feature_bundle["feature_manifest"]
    projector_state = feature_bundle["projector_state"]
    labels = _labels_to_index(data["labels"])
    if features.shape[0] != labels.numel():
        raise ValueError(f"G0 feature rows ({features.shape[0]}) must match labels ({labels.numel()}).")

    device = getattr(args, "device", torch.device("cpu"))
    if not isinstance(device, torch.device):
        device = torch.device("cpu" if int(device) < 0 or not torch.cuda.is_available() else f"cuda:{int(device)}")

    x_projected = features.to(device)
    x_raw = raw_features.to(device)
    y = labels.to(device)
    edge_index = data["edge_index"].to(device)
    edge_type = data["edge_type"].to(device)
    train_idx = _idx_tensor(data["train_idx"]).to(device)
    valid_idx = _idx_tensor(data["valid_idx"]).to(device)

    config = _model_config(args, features, device)
    model = build_GNN_model(config)
    input_adapter = None
    if bool(getattr(args, "peft", False)):
        input_adapter = PhaseAInputAdapter(
            raw_dim=int(raw_features.shape[1]),
            projected_dim=int(features.shape[1]),
            rank=int(getattr(args, "peft_rank", 8)),
            alpha=float(getattr(args, "peft_alpha", 16.0)),
        ).to(device)
        feature_manifest["peft"]["trainable_parameter_count"] = count_trainable_parameters(input_adapter)

    trainable_parameters = list(model.parameters())
    if input_adapter is not None:
        trainable_parameters += list(input_adapter.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(getattr(args, "lr_GNN", 5e-4)),
        weight_decay=float(getattr(args, "weight_decay_GNN", 1e-5)),
    )
    max_epochs = int(getattr(args, "g0_epochs", 0) or getattr(args, "GNN_epochs_per_iter", 1))
    max_epochs = max(max_epochs, 1)
    best_state = None
    best_metrics = None

    for epoch in range(max_epochs):
        model.train()
        if input_adapter is not None:
            input_adapter.train()
        optimizer.zero_grad()
        x = input_adapter(x_projected, x_raw) if input_adapter is not None else x_projected
        logits = model(x, edge_index, edge_type)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        if input_adapter is not None:
            input_adapter.eval()
        with torch.no_grad():
            x_eval = input_adapter(x_projected, x_raw) if input_adapter is not None else x_projected
            logits_eval = model(x_eval, edge_index, edge_type)
        val_metrics = _score_logits(logits_eval.detach().cpu(), labels, valid_idx.cpu())
        val_metrics["epoch"] = int(epoch)
        if _is_better(val_metrics, best_metrics):
            best_metrics = val_metrics
            best_state = {
                "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "input_adapter": (
                    {key: value.detach().cpu().clone() for key, value in input_adapter.state_dict().items()}
                    if input_adapter is not None
                    else None
                ),
            }

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        if input_adapter is not None and best_state["input_adapter"] is not None:
            input_adapter.load_state_dict(best_state["input_adapter"])
    model.eval()
    if input_adapter is not None:
        input_adapter.eval()
    with torch.no_grad():
        x = input_adapter(x_projected, x_raw) if input_adapter is not None else x_projected
        outputs = model.forward_outputs(x, edge_index, edge_type)
    outputs = {
        "logits": outputs["logits"].detach().cpu(),
        "prob": outputs["prob"].detach().cpu(),
        "pred": outputs["prob"].argmax(dim=1).detach().cpu(),
        "labels": data["labels"].detach().cpu() if torch.is_tensor(data["labels"]) else torch.tensor(data["labels"]),
        "node_repr": outputs["node_repr"].detach().cpu(),
        "aux_features": outputs.get("aux_features", {}),
    }

    checkpoint = {
        "model": (best_state or {}).get("model")
        if best_state is not None
        else {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "input_adapter": {
            "enabled": input_adapter is not None,
            "state_dict": (
                (best_state or {}).get("input_adapter")
                if best_state is not None
                else (
                    {key: value.detach().cpu().clone() for key, value in input_adapter.state_dict().items()}
                    if input_adapter is not None
                    else None
                )
            ),
            "config": feature_manifest.get("peft", {}),
        },
        "input_pipeline": {
            "feature_manifest": feature_manifest,
            "projector_state": projector_state,
        },
        "model_config": {key: str(value) if isinstance(value, torch.device) else value for key, value in config.items()},
        "selection_metrics": best_metrics or {},
    }
    write_torch(out_dir / "checkpoint.pt", checkpoint)
    write_torch(out_dir / "outputs.pt", outputs)
    write_json(out_dir / "selection_metrics.json", best_metrics or {})
    write_json(
        out_dir / "manifest.json",
        {
            "contract": FROZEN_G0_CONTRACT,
            "seed": int(seed),
            "backbone": getattr(args, "GNN_model", "rgcn"),
            "training_scope": "train_only_supervised",
            "pseudo_label_policy": "disabled_by_default",
            "checkpoint_selection": {
                "primary": "validation_macro_f1",
                "tie_breaker": "validation_loss",
            },
            "feature_manifest": feature_manifest,
            "input_projection": {
                "enabled": feature_manifest.get("projection_applied", False),
                "method": feature_manifest.get("projector"),
                "target_dim": feature_manifest.get("projected_dim"),
                "fit_scope": feature_manifest.get("fit_scope"),
                "raw_dim": feature_manifest.get("raw_dim"),
                "projected_dim": feature_manifest.get("projected_dim"),
            },
            "input_adapter": feature_manifest.get("peft", {"enabled": False}),
            "split_provenance": split_provenance(data),
            "node_id_manifest": {
                "num_nodes": int(labels.numel()),
                "labels_sha256": tensor_sha256(data["labels"]),
            },
        },
    )
    return load_frozen_g0(experiment_root)


def load_frozen_g0(experiment_root):
    out_dir = Path(experiment_root) / "frozen" / "g0"
    required = {
        "manifest": out_dir / "manifest.json",
        "outputs": out_dir / "outputs.pt",
        "checkpoint": out_dir / "checkpoint.pt",
        "selection_metrics": out_dir / "selection_metrics.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    manifest = read_json(required["manifest"])
    if missing or manifest is None:
        raise MissingFrozenArtifactError(
            f"Missing frozen G0 artifact(s) under {out_dir}: {', '.join(missing) or 'manifest'}. "
            "Run --stage frozen_g0 first."
        )
    if manifest.get("contract") != FROZEN_G0_CONTRACT:
        raise MissingFrozenArtifactError(f"Invalid frozen G0 contract under {out_dir}.")
    return {
        "manifest": manifest,
        "outputs": safe_torch_load(required["outputs"], map_location="cpu"),
        "selection_metrics": read_json(required["selection_metrics"], default={}),
        "checkpoint_path": str(required["checkpoint"]),
        "artifact_dir": str(out_dir),
        "dir": out_dir,
    }


def _requested_feature_path(args):
    path = getattr(args, "emb_path", None) or getattr(args, "g0_feature_path", None)
    return str(Path(path)) if path else None


def _existing_g0_matches_request(args, manifest):
    if manifest.get("contract") != FROZEN_G0_CONTRACT:
        return False
    if manifest.get("backbone", "").lower() != str(getattr(args, "GNN_model", "rgcn")).lower():
        return False

    requested_path = _requested_feature_path(args)
    feature_manifest = manifest.get("feature_manifest", {})
    if requested_path is not None and feature_manifest.get("path") != requested_path:
        return False

    if requested_path is None:
        return True

    if int(feature_manifest.get("projected_dim", -1)) != phase_a_project_dim(args):
        return False
    if str(feature_manifest.get("projector", "")).lower() != phase_a_projector(args):
        return False

    peft_manifest = feature_manifest.get("peft", {})
    if bool(peft_manifest.get("enabled", False)) != bool(getattr(args, "peft", False)):
        return False
    if bool(getattr(args, "peft", False)):
        if int(peft_manifest.get("rank", -1)) != int(getattr(args, "peft_rank", 8)):
            return False
        if float(peft_manifest.get("alpha", -1.0)) != float(getattr(args, "peft_alpha", 16.0)):
            return False
    return True


def build_or_load_frozen_g0(args, seed, data, experiment_root):
    out_dir = Path(experiment_root) / "frozen" / "g0"
    force = bool(getattr(args, "force_retrain_backbone", False))
    manifest = read_json(out_dir / "manifest.json", default=None)
    if (
        not force
        and manifest is not None
        and (out_dir / "outputs.pt").exists()
        and _existing_g0_matches_request(args, manifest)
    ):
        return load_frozen_g0(experiment_root)
    return train_frozen_g0(args, seed, data, experiment_root)


def gate_dir(experiment_root, gate_name):
    return ensure_dir(Path(experiment_root) / "frozen" / "gates" / gate_name)


def _graph_feature_matrix(logits_graph, prob_graph, struct_feats):
    entropy_graph = -(prob_graph.clamp(min=1e-8) * torch.log(prob_graph.clamp(min=1e-8))).sum(dim=1, keepdim=True)
    return torch.cat([logits_graph, prob_graph, entropy_graph, struct_feats], dim=1)


class _GraphCalibrationMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


class GraphGATSCalibrator:
    """Post-hoc GATS-style correctness calibrator for frozen graph predictions."""

    def __init__(self, hidden_dim=32, dropout=0.1, max_iter=300, lr=1e-2, device=None):
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.device = device or torch.device("cpu")
        self.model = None

    def fit(self, logits_graph, prob_graph, struct_feats, pred_graph, labels):
        x = _graph_feature_matrix(
            logits_graph.detach().float().to(self.device),
            prob_graph.detach().float().to(self.device),
            struct_feats.detach().float().to(self.device),
        )
        pred_graph = pred_graph.detach().long().view(-1).to(self.device)
        labels = labels.detach().long().view(-1).to(self.device)
        y = pred_graph.eq(labels).float()
        if x.numel() == 0:
            raise ValueError("Empty graph calibration inputs.")

        self.model = _GraphCalibrationMLP(
            in_dim=int(x.size(1)),
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            pos_logit = self.model(x)
            loss = F.binary_cross_entropy_with_logits(pos_logit, y.float())
            loss.backward()
            optimizer.step()
        return self

    @torch.no_grad()
    def apply(self, logits_graph, prob_graph, struct_feats, pred_graph):
        pred_graph = pred_graph.detach().long().view(-1).cpu()
        if self.model is None:
            q_graph = prob_graph.detach().float().gather(1, pred_graph.unsqueeze(1)).squeeze(1)
        else:
            x = _graph_feature_matrix(
                logits_graph.detach().float().to(self.device),
                prob_graph.detach().float().to(self.device),
                struct_feats.detach().float().to(self.device),
            )
            q_graph = torch.sigmoid(self.model(x)).cpu()
        conf = 0.5 + 0.5 * q_graph.clamp(min=0.0, max=1.0)
        prob_graph_cal = torch.empty_like(prob_graph.detach().float().cpu())
        graph_mask = pred_graph.bool()
        prob_graph_cal[graph_mask, 0] = 1.0 - conf[graph_mask]
        prob_graph_cal[graph_mask, 1] = conf[graph_mask]
        prob_graph_cal[~graph_mask, 0] = conf[~graph_mask]
        prob_graph_cal[~graph_mask, 1] = 1.0 - conf[~graph_mask]
        return {
            "prob_graph_cal": prob_graph_cal,
            "q_graph": q_graph,
        }


def _structural_tensor(data):
    labels = _labels_to_index(data["labels"])
    features = structural_features(data["edge_index"], data["edge_type"], int(labels.numel()))
    table = np.column_stack(
        [
            features["in_degree"],
            features["out_degree"],
            features["total_degree"],
            features["relation_skew"],
        ]
    ).astype(np.float32)
    return torch.from_numpy(table)


def load_gate_manifest(experiment_root, gate_name):
    path = Path(experiment_root) / "frozen" / "gates" / gate_name / "manifest.json"
    manifest = read_json(path)
    if manifest is None:
        raise MissingFrozenArtifactError(
            f"Missing faithful gate '{gate_name}' at {path}. Generate the frozen gate before this stage."
        )
    if manifest.get("gate_name") != gate_name or manifest.get("status") != "available":
        raise MissingFrozenArtifactError(f"Faithful gate '{gate_name}' is not available at {path}.")
    return manifest


def require_stage_gates(experiment_root, stage_name):
    manifests = {}
    for gate_name in FORMAL_STAGE_GATES.get(stage_name, []):
        manifests[gate_name] = load_gate_manifest(experiment_root, gate_name)
    return manifests


def load_gats_outputs(experiment_root):
    out_dir = Path(experiment_root) / "frozen" / "gates" / GATS_GATE_NAME
    manifest = load_gate_manifest(experiment_root, GATS_GATE_NAME)
    outputs_path = out_dir / "outputs.pt"
    if not outputs_path.exists():
        raise MissingFrozenArtifactError(f"Missing faithful GATS outputs at {outputs_path}.")
    return {
        "manifest": manifest,
        "outputs": safe_torch_load(outputs_path, map_location="cpu"),
        "artifact_dir": str(out_dir),
        "dir": out_dir,
    }


def build_or_load_faithful_gats(args, seed, data, experiment_root, g0_context):
    out_dir = gate_dir(experiment_root, GATS_GATE_NAME)
    if (
        (out_dir / "manifest.json").exists()
        and (out_dir / "outputs.pt").exists()
        and not getattr(args, "force_retrain_backbone", False)
    ):
        return load_gats_outputs(experiment_root)

    g0_outputs = g0_context["outputs"]
    labels = _labels_to_index(data["labels"])
    valid_idx = _idx_tensor(data["valid_idx"])
    struct_feats = _structural_tensor(data)
    pred = g0_outputs["pred"].long()

    torch.manual_seed(int(seed))
    max_iter = int(getattr(args, "gats_max_iter", 300))
    calibrator = GraphGATSCalibrator(max_iter=max_iter, device=torch.device("cpu"))
    calibrator.fit(
        logits_graph=g0_outputs["logits"][valid_idx],
        prob_graph=g0_outputs["prob"][valid_idx],
        struct_feats=struct_feats[valid_idx],
        pred_graph=pred[valid_idx],
        labels=labels[valid_idx],
    )
    outputs = calibrator.apply(
        logits_graph=g0_outputs["logits"],
        prob_graph=g0_outputs["prob"],
        struct_feats=struct_feats,
        pred_graph=pred,
    )
    write_torch(out_dir / "outputs.pt", outputs)
    write_json(
        out_dir / "manifest.json",
        {
            "contract": GATS_CONTRACT,
            "gate_name": GATS_GATE_NAME,
            "status": "available",
            "seed": int(seed),
            "source_impl": "trainer.GraphGATSCalibrator",
            "training_scope": "validation_only_calibration",
            "calibration_idx_sha256": tensor_sha256(valid_idx),
            "input_g0_contract": g0_context["manifest"].get("contract"),
            "input_g0_outputs_sha256": tensor_sha256(g0_outputs["pred"]),
        },
    )
    return load_gats_outputs(experiment_root)


def _safe_pseudo_label_training_index(train_idx, valid_idx, test_idx, n_total, pl_ratio, pseudo_label_pool_idx=None):
    """Build train + explicit pseudo-label indices without ever using val/test by complement."""
    train_idx = _as_long_cpu_tensor(train_idx)
    valid_idx = _as_long_cpu_tensor(valid_idx)
    test_idx = _as_long_cpu_tensor(test_idx)
    pool_idx = _as_long_cpu_tensor(pseudo_label_pool_idx)
    n_train = int(train_idx.numel())
    if pool_idx.numel() == 0 or n_train == 0 or float(pl_ratio) <= 0.0:
        is_pl = torch.zeros_like(train_idx, dtype=torch.int64)
        return train_idx, is_pl, torch.empty(0, dtype=torch.long)

    val_test = set(valid_idx.tolist()) | set(test_idx.tolist())
    train_set = set(train_idx.tolist())
    pool_set = set(pool_idx.tolist())
    if pool_set & val_test:
        raise ValueError("pseudo_label_pool_idx must be disjoint from validation and test splits.")
    if pool_set & train_set:
        raise ValueError("pseudo_label_pool_idx must be disjoint from train_idx.")
    if pool_idx.min().item() < 0 or pool_idx.max().item() >= int(n_total):
        raise ValueError("pseudo_label_pool_idx contains node ids outside the graph.")

    n_pl = min(int(n_train * float(pl_ratio)), int(pool_idx.numel()))
    if n_pl <= 0:
        is_pl = torch.zeros_like(train_idx, dtype=torch.int64)
        return train_idx, is_pl, torch.empty(0, dtype=torch.long)
    chosen = pool_idx[torch.randperm(pool_idx.numel())[:n_pl]]
    train_idx_all = torch.cat((train_idx, chosen))
    is_pl = torch.ones_like(train_idx_all, dtype=torch.int64)
    is_pl[:n_train] = 0
    return train_idx_all, is_pl, chosen


class LM_Trainer:
    def __init__(
            self,
            model_name,
            classifier_n_layers,
            classifier_hidden_dim,
            device,
            pretrain_epochs,
            optimizer_name,
            lr,
            weight_decay,
            dropout,
            att_dropout,
            lm_dropout,
            activation,
            warmup,
            label_smoothing_factor,
            pl_weight,
            max_length,
            batch_size,
            grad_accumulation,
            lm_epochs_per_iter,
            temperature,
            pl_ratio,
            eval_patience,
            intermediate_data_filepath,
            ckpt_filepath,
            pretrain_ckpt_filepath,
            raw_data_filepath,
            train_idx,
            valid_idx,
             test_idx,
             hard_labels,
             user_seq,
             run,
             pseudo_label_pool_idx=None,
             peft_rank=8,
             peft_alpha=16.0,
             qwen_model_path=None,
             qwen_trust_remote_code=False):

        self.model_name = model_name
        self._peft_rank = peft_rank
        self._peft_alpha = peft_alpha
        self._qwen_model_path = qwen_model_path
        self._qwen_trust_remote_code = qwen_trust_remote_code
        self.device = device
        self.pretrain_epochs = pretrain_epochs
        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.att_dropout = att_dropout
        self.lm_dropout = lm_dropout
        self.warmup = warmup
        self.label_smoothing_factor = label_smoothing_factor
        self.pl_weight = pl_weight
        self.max_length = max_length
        self.batch_size = batch_size
        self.grad_accumulation = grad_accumulation
        self.lm_epochs_per_iter = lm_epochs_per_iter
        self.temperature = temperature
        self.pl_ratio = pl_ratio
        self.eval_patience = eval_patience
        self.intermediate_data_filepath = intermediate_data_filepath
        self.ckpt_filepath = ckpt_filepath
        self.pretrain_ckpt_filepath = pretrain_ckpt_filepath
        self.raw_data_filepath = Path(raw_data_filepath)
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.hard_labels = hard_labels
        self.user_seq = user_seq
        self.run = run
        self.pseudo_label_pool_idx = pseudo_label_pool_idx
        self.do_mlm_task = False

        self.iter = 0
        self.best_iter = 0
        self.best_valid_acc = 0
        self.best_valid_f1 = -float("inf")
        self.best_epoch = 0
        self.criterion = CrossEntropyLoss(label_smoothing=label_smoothing_factor)
        self.KD_criterion = KLDivLoss(log_target=False, reduction='batchmean')
        self.results = {}


        self.get_train_idx_all()
        self.pretrain_steps_per_epoch = self.train_idx.shape[0] // self.batch_size + 1
        self.pretrain_steps = int(self.pretrain_steps_per_epoch * self.pretrain_epochs)
        self.train_steps_per_iter = (self.train_idx_all.shape[0] // self.batch_size + 1) * self.lm_epochs_per_iter
        self.optimizer_args = dict(lr=lr, weight_decay=weight_decay)

        self.model_config = {
            'lm_model': model_name,
            'dropout': dropout,
            'att_dropout': att_dropout,
            'lm_dropout': self.lm_dropout,
            'classifier_n_layers': classifier_n_layers,
            'classifier_hidden_dim': classifier_hidden_dim,
            'activation': activation,
            'device': device,
            'return_mlm_loss': True if self.do_mlm_task else False,
            'qwen_model_path': self._qwen_model_path,
            'peft_rank': getattr(self, '_peft_rank', 8),
            'peft_alpha': getattr(self, '_peft_alpha', 16.0),
            'qwen_trust_remote_code': getattr(self, '_qwen_trust_remote_code', False),
            }

        self.dataloader_config = {
            'batch_size': batch_size,
            'pl_ratio': pl_ratio
            }


    def build_model(self):
        self.model, self.tokenizer = build_LM_model(self.model_config)
        self.DESCRIPTION_id = self.tokenizer.convert_tokens_to_ids('DESCRIPTION:')
        self.TWEET_id = self.tokenizer.convert_tokens_to_ids('TWEET:')
        self.METADATA_id = self.tokenizer.convert_tokens_to_ids('METADATA:')

    def get_optimizer(self, parameters):

        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(parameters, **self.optimizer_args)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, **self.optimizer_args)
        elif self.optimizer_name == "adadelta":
            optimizer = torch.optim.Adadelta(parameters, **self.optimizer_args)
        elif self.optimizer_name == "radam":
            optimizer = torch.optim.RAdam(parameters, **self.optimizer_args)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        return optimizer

    def get_scheduler(self, optimizer, mode='train'):
        if mode == 'pretrain':
            return get_cosine_schedule_with_warmup(optimizer, self.pretrain_steps_per_epoch * self.warmup, self.pretrain_steps)
        else:
            return CosineAnnealingLR(optimizer, T_max=self.train_steps_per_iter, eta_min=0)

    def get_initial_embeddings(self):

        if not os.path.exists(self.raw_data_filepath / f'embeddings_{self.model_name.lower()}.pt'):
            print('Generating initial GNN embeddings...')
            self.infer(True)

        embeddings = safe_torch_load(self.raw_data_filepath / f'embeddings_{self.model_name.lower()}.pt')
        return embeddings

    def pretrain(self):
        print('LM pretraining start!')
        optimizer = self.get_optimizer(self.model.parameters())
        scheduler = self.get_scheduler(optimizer, 'pretrain')
        best_pretrain_ckpt = self.pretrain_ckpt_filepath / 'best.pkl'
        if best_pretrain_ckpt.exists() and os.path.exists(self.intermediate_data_filepath / 'embeddings_iter_-1.pt'):
            print('Pretrain checkpoint exists, loading from checkpoint...')
            print('Please make sure you use the same parameter setting as the one of the pretrain checkpoint!')
            ckpt = safe_torch_load(best_pretrain_ckpt)
            self.model.load_state_dict(ckpt['model'])
            # self.optimizer.load_state_dict(ckpt['optimizer'])
            # self.scheduler.load_state_dict(ckpt['scheduler'])
            # embeddings = safe_torch_load(self.intermediate_data_filepath / 'embeddings_iter_-1.pt')
            valid_acc, valid_f1 = self.eval('valid')
            self.results['pretrain valid accuracy'] = valid_acc
            self.results['pretrain valid f1'] = valid_f1

        else:
            step = 0
            valid_f1_best = -float("inf")
            valid_step_best = 0

            torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, self.pretrain_ckpt_filepath / 'best.pkl')

            train_loader = build_LM_dataloader(self.dataloader_config, self.train_idx, self.user_seq, self.hard_labels, 'pretrain')

            for epoch in range(int(self.pretrain_epochs)+1):
                self.model.train()
                print(f'------LM Pretraining Epoch: {epoch}/{int(self.pretrain_epochs)}------')
                for batch in tqdm(train_loader):
                    step += 1
                    if step >= self.pretrain_steps:
                        break
                    tokenized_tensors, labels, _ = self.batch_to_tensor(batch)

                    _, output = self.model(tokenized_tensors)
                    loss = self.criterion(output, labels)
                    loss /= self.grad_accumulation
                    loss.backward()
                    self.run.log({'LM Pretrain Loss': loss.item()})

                    if step % self.grad_accumulation == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    scheduler.step()

                    if step % self.eval_patience == 0:
                        valid_acc, valid_f1 = self.eval()

                        print(f'LM Pretrain Valid Accuracy = {valid_acc}')
                        print(f'LM Pretrain Valid F1 = {valid_f1}')
                        self.run.log({'LM Pretrain Valid Accuracy': valid_acc})
                        self.run.log({'LM Pretrain Valid F1': valid_f1})

                        if valid_f1 > valid_f1_best:
                            valid_f1_best = valid_f1
                            valid_step_best = step

                            torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, self.pretrain_ckpt_filepath / 'best.pkl')


            print(f'The highest pretrain valid macro-F1 is {valid_f1_best}!')
            print(f'Load model from step {valid_step_best}')
            self.model.eval()
            all_outputs = []
            all_labels = []
            embeddings = []
            infer_loader = build_LM_dataloader(self.dataloader_config, None, self.user_seq, self.hard_labels, mode='infer')
            with torch.no_grad():
                ckpt = safe_torch_load(self.pretrain_ckpt_filepath / 'best.pkl')
                self.model.load_state_dict(ckpt['model'])
                optimizer.load_state_dict(ckpt['optimizer'])
                scheduler.load_state_dict(ckpt['scheduler'])
                for batch in tqdm(infer_loader):
                    tokenized_tensors, labels, _ = self.batch_to_tensor(batch)
                    embedding, output = self.model(tokenized_tensors)
                    embeddings.append(embedding.cpu())
                    all_outputs.append(output.cpu())
                    all_labels.append(labels.cpu())

                all_outputs = torch.cat(all_outputs, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                embeddings = torch.cat(embeddings, dim=0)
                soft_labels = torch.softmax(all_outputs / self.temperature, dim=1)
                soft_labels[self.train_idx] = all_labels[self.train_idx]

                valid_predictions = torch.argmax(all_outputs[self.valid_idx], dim=1).numpy()
                valid_labels = torch.argmax(all_labels[self.valid_idx], dim=1).numpy()
                torch.save(embeddings, self.intermediate_data_filepath / 'embeddings_iter_-1.pt')
                torch.save(soft_labels, self.intermediate_data_filepath / 'soft_labels_iter_-1.pt')

                valid_acc = accuracy_score(valid_labels, valid_predictions)
                valid_f1 = f1_score(valid_labels, valid_predictions, average="macro", zero_division=0)
                self.results['pretrain valid accuracy'] = valid_acc
                self.results['pretrain valid f1'] = valid_f1


        print(f'LM Pretrain Valid Accuracy = {valid_acc}')
        print(f'LM Pretrain Valid F1 = {valid_f1}')
        self.run.log({'LM Pretrain Valid Accuracy': valid_acc})
        self.run.log({'LM Pretrain Valid F1': valid_f1})




    def train(self, soft_labels):
        for param in self.model.classifier.parameters():
            param.requires_grad = False
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.get_optimizer(parameters)
        scheduler = self.get_scheduler(optimizer)

        early_stop_flag = True
        print('LM training start!')
        step = 0
        train_loader = build_LM_dataloader(self.dataloader_config, self.train_idx_all, self.user_seq, soft_labels, 'train', self.is_pl)


        for epoch in range(self.lm_epochs_per_iter):
            self.model.train()
            print(f'This is iter {self.iter} epoch {epoch}/{self.lm_epochs_per_iter-1}')

            for batch in tqdm(train_loader):
                step += 1

                tokenized_tensors, labels, is_pl = self.batch_to_tensor(batch)

                _, output = self.model(tokenized_tensors)

                pl_idx = torch.nonzero(is_pl == 1).squeeze()
                rl_idx = torch.nonzero(is_pl == 0).squeeze()

                if pl_idx.numel() == 0:
                    loss = self.criterion(output[rl_idx], labels[rl_idx])
                elif rl_idx.numel() == 0:
                    loss = self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), labels[pl_idx])
                else:
                    loss_KD = self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), labels[pl_idx])
                    loss_H = self.criterion(output[rl_idx], labels[rl_idx])
                    self.run.log({'loss_KD': loss_KD.item()})
                    self.run.log({'loss_H': loss_H.item()})
                    loss = self.pl_weight * loss_KD + (1 - self.pl_weight) * loss_H

                loss /= self.grad_accumulation
                loss.backward()
                self.run.log({'LM Train Loss': loss.item()})

                if step % self.grad_accumulation == 0:

                    optimizer.step()
                    optimizer.zero_grad()
                scheduler.step()
                if step % self.eval_patience == 0:
                    valid_acc, valid_f1 = self.eval()

                    print(f'LM Valid Accuracy = {valid_acc}')
                    print(f'LM Valid F1 = {valid_f1}')
                    self.run.log({'LM Valid Accuracy': valid_acc})
                    self.run.log({'LM Valid F1': valid_f1})

                    if valid_f1 > self.best_valid_f1 or (valid_f1 == self.best_valid_f1 and valid_acc > self.best_valid_acc):
                        early_stop_flag = False
                        self.best_valid_acc = valid_acc
                        self.best_valid_f1 = valid_f1
                        self.best_iter = self.iter
                        self.best_epoch = epoch
                        torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, self.ckpt_filepath / 'best.pkl')

        print(f'The highest valid macro-F1 is {self.best_valid_f1}!')
        return early_stop_flag

    def infer(self, provide_embeddings_for_GNN_pretraining=False):
        self.model.eval()
        infer_loader = build_LM_dataloader(self.dataloader_config, None, self.user_seq, self.hard_labels, mode='infer')
        all_outputs = []
        all_labels = []
        embeddings = []
        with torch.no_grad():
            if provide_embeddings_for_GNN_pretraining:
                for batch in tqdm(infer_loader):
                    tokenized_tensors, labels, _ = self.batch_to_tensor(batch)
                    embedding, _ = self.model(tokenized_tensors)
                    embeddings.append(embedding.cpu())
                embeddings = torch.cat(embeddings, dim=0)
                torch.save(embeddings, self.raw_data_filepath / f'embeddings_{self.model_name.lower()}.pt')

            else:
                ckpt = safe_torch_load(self.ckpt_filepath / 'best.pkl')
                self.model.load_state_dict(ckpt['model'])
                # self.optimizer.load_state_dict(ckpt['optimizer'])
                # self.scheduler.load_state_dict(ckpt['scheduler'])
                for batch in tqdm(infer_loader):
                    tokenized_tensors, labels, _ = self.batch_to_tensor(batch)

                    embedding, output = self.model(tokenized_tensors)
                    embeddings.append(embedding.cpu())
                    all_outputs.append(output.cpu())
                    all_labels.append(labels.cpu())

                all_outputs = torch.cat(all_outputs, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                embeddings = torch.cat(embeddings, dim=0)

                soft_labels = torch.softmax(all_outputs / self.temperature, dim=1)
                soft_labels[self.train_idx] = all_labels[self.train_idx]

                torch.save(soft_labels, self.intermediate_data_filepath / f'soft_labels_iter_{self.iter}.pt')

                torch.save(embeddings, self.intermediate_data_filepath / f'embeddings_iter_{self.iter}.pt')

                self.iter += 1

    def eval(self, mode='valid'):
        if mode == 'valid':
            eval_loader =  build_LM_dataloader(self.dataloader_config, self.valid_idx, self.user_seq, self.hard_labels, mode='eval')
        elif mode == 'test':
            eval_loader =  build_LM_dataloader(self.dataloader_config, self.test_idx, self.user_seq, self.hard_labels, mode='eval')
        self.model.eval()

        valid_predictions = []
        valid_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_loader):
                tokenized_tensors, labels, _ = self.batch_to_tensor(batch)

                _, output = self.model(tokenized_tensors)

                valid_predictions.append(torch.argmax(output, dim=1).cpu().numpy())
                valid_labels.append(torch.argmax(labels, dim=1).cpu().numpy())

            valid_predictions = np.concatenate(valid_predictions)
            valid_labels = np.concatenate(valid_labels)
            valid_acc = accuracy_score(valid_labels, valid_predictions)
            valid_f1 = f1_score(valid_labels, valid_predictions, average="macro", zero_division=0)

            return valid_acc, valid_f1


    def test(self):
        print('Computing test accuracy and f1 for LM...')
        ckpt = safe_torch_load(self.ckpt_filepath / 'best.pkl')
        self.model.load_state_dict(ckpt['model'])
        test_acc, test_f1 = self.eval('test')
        print(f'LM Test Accuracy = {test_acc}')
        print(f'LM Test F1 = {test_f1}')
        self.run.log({'LM Test Accuracy': test_acc})
        self.run.log({'LM Test F1': test_f1})
        self.results['accuracy'] = test_acc
        self.results['f1'] = test_f1

    def batch_to_tensor(self, batch):

        tokenized_tensors = self.tokenizer(text=batch[0], return_tensors='pt', max_length=self.max_length, truncation=True, padding='longest', add_special_tokens=False)
        for key in tokenized_tensors.keys():
            tokenized_tensors[key] = tokenized_tensors[key].to(self.device)
        labels = batch[1].to(self.device)

        if len(batch) == 3:
            is_pl = batch[2].to(self.device)
            return tokenized_tensors, labels, is_pl
        else:
            return tokenized_tensors, labels, None

    def load_embedding(self, iter):
        embeddings = safe_torch_load(self.intermediate_data_filepath / f'embeddings_iter_{iter}.pt')
        return embeddings

    def predict_all(self, load_best=True):
        self.model.eval()
        infer_loader = build_LM_dataloader(self.dataloader_config, None, self.user_seq, self.hard_labels, mode='infer')
        if load_best and (self.ckpt_filepath / 'best.pkl').exists():
            ckpt = safe_torch_load(self.ckpt_filepath / 'best.pkl', map_location=self.device)
            self.model.load_state_dict(ckpt['model'])

        all_embeddings = []
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(infer_loader):
                tokenized_tensors, labels, _ = self.batch_to_tensor(batch)
                embedding, output = self.model(tokenized_tensors)
                all_embeddings.append(embedding.cpu())
                all_logits.append(output.cpu())
                all_labels.append(labels.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        probs = torch.softmax(logits, dim=1)
        return {
            'embeddings': embeddings,
            'logits': logits,
            'prob': probs,
            'pred': probs.argmax(dim=1),
            'labels': labels,
        }

    def save_results(self, path):
        json.dump(self.results, open(path, 'w'), indent=4)

    def get_train_idx_all(self):
        n_total = self.hard_labels.shape[0]
        self.train_idx_all, self.is_pl, self.pl_idx = _safe_pseudo_label_training_index(
            train_idx=self.train_idx,
            valid_idx=self.valid_idx,
            test_idx=self.test_idx,
            n_total=n_total,
            pl_ratio=self.pl_ratio,
            pseudo_label_pool_idx=self.pseudo_label_pool_idx,
        )


def _resolve_device(device_arg):
    if isinstance(device_arg, torch.device):
        return device_arg
    if isinstance(device_arg, str):
        return torch.device(device_arg)
    if isinstance(device_arg, int):
        if device_arg < 0 or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(f"cuda:{device_arg}")
    return torch.device("cpu")


def _set_cuda_device(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.set_device(device)
    return torch.cuda.current_device()


def _reset_cuda_peak_memory_stats(device):
    cuda_index = _set_cuda_device(device)
    if cuda_index is not None:
        torch.cuda.reset_peak_memory_stats(cuda_index)


def _max_cuda_memory_allocated(device):
    cuda_index = _set_cuda_device(device)
    if cuda_index is None:
        return 0
    return int(torch.cuda.max_memory_allocated(cuda_index))


def _idx_numpy(idx):
    if torch.is_tensor(idx):
        return idx.detach().cpu().numpy().astype(np.int64)
    return np.asarray(idx, dtype=np.int64)


def _mask_from_idx(num_nodes, idx):
    mask = np.zeros(num_nodes, dtype=bool)
    mask[_idx_numpy(idx)] = True
    return mask


def _subset_scores(labels, pred, idx_mask):
    labels_np = labels if isinstance(labels, np.ndarray) else np.asarray(labels)
    pred_np = pred if isinstance(pred, np.ndarray) else np.asarray(pred)
    if idx_mask.sum() == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "bot_f1": 0.0, "count": 0}
    return {
        "accuracy": float(accuracy_score(labels_np[idx_mask], pred_np[idx_mask])),
        "macro_f1": float(f1_score(labels_np[idx_mask], pred_np[idx_mask], average="macro", zero_division=0)),
        "bot_f1": float(f1_score(labels_np[idx_mask], pred_np[idx_mask], average="binary", zero_division=0)),
        "count": int(idx_mask.sum()),
    }


def _delta_table(base_pred, new_pred, labels, idx_mask):
    labels_np = np.asarray(labels)
    base_pred = np.asarray(base_pred)
    new_pred = np.asarray(new_pred)
    affected = idx_mask.astype(bool)
    base_correct = base_pred == labels_np
    new_correct = new_pred == labels_np
    fix = int((~base_correct & new_correct & affected).sum())
    broke = int((base_correct & ~new_correct & affected).sum())
    return {
        "fix": fix,
        "broke": broke,
        "net": fix - broke,
        "touched": int(affected.sum()),
    }


def _score_all(labels, pred):
    labels_np = np.asarray(labels)
    pred_np = np.asarray(pred)
    return {
        "accuracy": float(accuracy_score(labels_np, pred_np)),
        "macro_f1": float(f1_score(labels_np, pred_np, average="macro", zero_division=0)),
        "bot_f1": float(f1_score(labels_np, pred_np, average="binary", zero_division=0)),
        "count": int(labels_np.shape[0]),
    }


def _semantic_backbone_to_lm_model(args):
    backbone = str(getattr(args, "semantic_backbone", "auto")).lower()
    if backbone == "auto":
        return str(getattr(args, "LM_model", "roberta")).lower()
    mapping = {
        "roberta": "roberta",
        "roberta_finetuned": "roberta_finetuned",
        "qwen3_peft": "qwen3_peft",
    }
    if backbone not in mapping:
        raise ValueError(
            f"--stage semantic_finetune does not train semantic_backbone={backbone}. "
            "Use qwen3_peft for real Qwen LoRA PEFT or pass --emb_path for frozen embeddings."
        )
    return mapping[backbone]


def _semantic_command(args):
    keys = [
        "stage",
        "dataset",
        "reset_split",
        "seeds",
        "semantic_backbone",
        "qwen_model_path",
        "qwen_trust_remote_code",
        "peft_rank",
        "peft_alpha",
        "batch_size_LM",
        "max_length",
        "semantic_train_limit",
        "semantic_max_steps",
        "experiment_name",
        "artifact_root",
        "device",
        "disable_wandb",
    ]
    parts = ["python", "main.py"]
    for key in keys:
        if not hasattr(args, key):
            continue
        value = getattr(args, key)
        if isinstance(value, bool):
            if value:
                parts.append(f"--{key}")
            continue
        if value is not None:
            parts.extend([f"--{key}", str(value)])
    return " ".join(parts)


def _semantic_tokenize(tokenizer, texts, max_length, device):
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
    )
    return {key: value.to(device) for key, value in tokenized.items()}


def _classification_metrics_from_logits(logits, labels, idx):
    idx = _as_long_cpu_tensor(idx)
    if idx.numel() == 0:
        return {"loss": float("inf"), "accuracy": 0.0, "macro_f1": 0.0, "bot_f1": 0.0, "count": 0}
    logits_idx = logits[idx]
    labels_idx = labels[idx]
    pred = logits_idx.argmax(dim=1)
    scores = _score_all(labels_idx.numpy(), pred.numpy())
    scores["loss"] = float(F.cross_entropy(logits_idx, labels_idx).item())
    return scores


def _select_semantic_train_idx(train_idx, limit, seed):
    train_idx = _as_long_cpu_tensor(train_idx)
    if int(limit or 0) <= 0 or int(limit) >= int(train_idx.numel()):
        return train_idx
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    order = torch.randperm(train_idx.numel(), generator=generator)
    return train_idx[order[: int(limit)]]


def run_semantic_finetune_seed(args, seed, data, experiment_root, run):
    stage_dir = build_stage_dir(experiment_root, "semantic_finetune")
    code_provenance = capture_code_metadata(Path(__file__).resolve().parents[2])
    manifest = {
        "contract": "semantic_finetune_v1",
        "status": "started",
        "seed": int(seed),
        "semantic_backbone": getattr(args, "semantic_backbone", "auto"),
        "dataset": getattr(args, "dataset", "unknown"),
        "dataset_path": str(data.get("dataset_path", "")),
        "training_scope": "train_idx_supervised_semantic_backbone",
        "notes": "Exploratory semantic finetune unless promoted by later full validation.",
        "command": _semantic_command(args),
        "code_commit": code_provenance["commit"],
        "code_provenance": code_provenance,
    }
    write_json(stage_dir / "manifest.json", manifest)
    write_text(stage_dir / "command.txt", manifest["command"] + "\n")

    try:
        device = _resolve_device(getattr(args, "device", -1))
        if device.type == "cuda":
            _reset_cuda_peak_memory_stats(device)
        labels = _labels_to_index(data["labels"]).cpu()
        user_text = list(data["user_text"])
        train_idx = _select_semantic_train_idx(
            data["train_idx"],
            getattr(args, "semantic_train_limit", 0),
            seed,
        )
        valid_idx = _as_long_cpu_tensor(data["valid_idx"])
        test_idx = _as_long_cpu_tensor(data["test_idx"])
        if train_idx.numel() == 0:
            raise ValueError("semantic_finetune requires a non-empty train_idx.")

        lm_model = _semantic_backbone_to_lm_model(args)
        model_config = {
            "lm_model": lm_model,
            "qwen_model_path": getattr(args, "qwen_model_path", None),
            "dropout": getattr(args, "dropout", 0.4),
            "att_dropout": getattr(args, "LM_att_dropout", 0.1),
            "lm_dropout": getattr(args, "LM_dropout", 0.1),
            "classifier_n_layers": getattr(args, "LM_classifier_n_layers", 2),
            "classifier_hidden_dim": getattr(args, "LM_classifier_hidden_dim", 128),
            "activation": getattr(args, "activation", "leakyrelu"),
            "device": device,
            "peft_rank": getattr(args, "peft_rank", 8),
            "peft_alpha": getattr(args, "peft_alpha", 16.0),
            "qwen_trust_remote_code": getattr(args, "qwen_trust_remote_code", False),
            "detach_embeddings": False,
        }
        model, tokenizer = build_LM_model(model_config)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(getattr(args, "lr_LM", 1e-5)),
            weight_decay=float(getattr(args, "weight_decay_LM", 0.01)),
        )
        batch_size = max(int(getattr(args, "batch_size_LM", 1)), 1)
        max_length = int(getattr(args, "max_length", 512))
        max_steps = int(getattr(args, "semantic_max_steps", 0) or np.ceil(train_idx.numel() / batch_size))
        max_steps = max(max_steps, 1)
        losses = []
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        order = train_idx[torch.randperm(train_idx.numel(), generator=generator)]

        for step in range(max_steps):
            start = (step * batch_size) % int(order.numel())
            if start == 0 and step > 0:
                order = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
            batch_idx = order[start : start + batch_size]
            if batch_idx.numel() < batch_size:
                extra = order[: batch_size - batch_idx.numel()]
                batch_idx = torch.cat([batch_idx, extra])
            texts = [user_text[int(idx)] for idx in batch_idx]
            y = labels[batch_idx].to(device)
            tokenized = _semantic_tokenize(tokenizer, texts, max_length, device)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(tokenized)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            run.log({"semantic_finetune_loss": loss_value, "semantic_finetune_step": step + 1})

        model.eval()
        all_embeddings = []
        all_logits = []
        with torch.no_grad():
            for start in range(0, len(user_text), batch_size):
                texts = user_text[start : start + batch_size]
                tokenized = _semantic_tokenize(tokenizer, texts, max_length, device)
                embeddings, logits = model(tokenized)
                all_embeddings.append(embeddings.cpu())
                all_logits.append(logits.cpu())
        embeddings = torch.cat(all_embeddings, dim=0)
        logits = torch.cat(all_logits, dim=0)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        outputs = {"logits": logits, "prob": prob, "pred": pred, "labels": labels}

        write_torch(stage_dir / "embeddings.pt", embeddings)
        write_torch(stage_dir / "outputs.pt", outputs)
        write_torch(stage_dir / "classifier.pt", model.classifier.state_dict())
        if lm_model == "qwen3_peft":
            adapter_dir = stage_dir / "adapter"
            model.LM.save_pretrained(adapter_dir)
            model_artifact = {"adapter_dir": str(adapter_dir)}
        else:
            write_torch(stage_dir / "model.pt", {"model": model.state_dict(), "model_config": model_config})
            model_artifact = {"model_path": str(stage_dir / "model.pt")}

        metrics = {
            "losses": losses,
            "final_loss": losses[-1],
            "train": _classification_metrics_from_logits(logits, labels, train_idx),
            "validation": _classification_metrics_from_logits(logits, labels, valid_idx),
            "test": _classification_metrics_from_logits(logits, labels, test_idx),
            "trainable_model_params": int(sum(param.numel() for param in model.LM.parameters() if param.requires_grad)),
            "trainable_classifier_params": int(sum(param.numel() for param in model.classifier.parameters() if param.requires_grad)),
            "cuda_max_memory_allocated": _max_cuda_memory_allocated(device),
        }
        write_json(stage_dir / "metrics.json", metrics)
        manifest.update(
            {
                "status": "completed",
                "lm_model": lm_model,
                "qwen_model_path": getattr(args, "qwen_model_path", None),
                "qwen_trust_remote_code": bool(getattr(args, "qwen_trust_remote_code", False)),
                "train_limit": int(train_idx.numel()),
                "max_steps": int(max_steps),
                "batch_size": int(batch_size),
                "max_length": int(max_length),
                "embeddings_path": str(stage_dir / "embeddings.pt"),
                "outputs_path": str(stage_dir / "outputs.pt"),
                "classifier_path": str(stage_dir / "classifier.pt"),
                "metrics_path": str(stage_dir / "metrics.json"),
                **model_artifact,
            }
        )
        write_json(stage_dir / "manifest.json", manifest)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {"stage": "semantic_finetune", "stage_dir": str(stage_dir), "metrics": metrics}
    except Exception as exc:
        manifest.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        write_json(stage_dir / "manifest.json", manifest)
        raise


def _compute_budget_curve(
    labels,
    base_pred,
    new_pred,
    risk_score,
    eval_mask,
    hcw_mask=None,
    budgets=(0.05, 0.10, 0.15, 0.20),
):
    labels_np = np.asarray(labels)
    base_pred = np.asarray(base_pred)
    new_pred = np.asarray(new_pred)
    risk_score = np.asarray(risk_score, dtype=np.float32)
    eval_mask = np.asarray(eval_mask, dtype=bool)
    hcw_mask = np.asarray(hcw_mask, dtype=bool) if hcw_mask is not None else np.zeros_like(eval_mask, dtype=bool)

    candidate_idx = np.flatnonzero(eval_mask)
    if candidate_idx.size == 0:
        return []
    ranked_idx = candidate_idx[np.argsort(-risk_score[candidate_idx])]
    base_correct = base_pred == labels_np
    new_correct = new_pred == labels_np

    rows = []
    for budget in budgets:
        k = max(int(candidate_idx.size * budget), 1)
        current_idx = ranked_idx[:k]
        current_mask = np.zeros_like(eval_mask, dtype=bool)
        current_mask[current_idx] = True
        fix = int((~base_correct & new_correct & current_mask).sum())
        broke = int((base_correct & ~new_correct & current_mask).sum())
        net = fix - broke
        touched = int(current_mask.sum())
        wrong_rate = float((~base_correct[current_idx]).mean()) if current_idx.size else 0.0
        screened_acc = float(new_correct[current_idx].mean()) if current_idx.size else 0.0
        hcw_hits = int((hcw_mask & current_mask).sum())
        hcw_total = int((hcw_mask & eval_mask).sum())
        rows.append(
            {
                "budget": float(budget),
                "triggered": int(current_idx.size),
                "touched": touched,
                "fix": fix,
                "broke": broke,
                "net": net,
                "utility": wrong_rate,
                "screened_set_accuracy": screened_acc,
                "hcw_capture": float(hcw_hits / max(hcw_total, 1)),
            }
        )
    return rows


def _paired_bootstrap_delta(labels, base_pred, new_pred, eval_mask, n_samples=512, seed=0):
    labels_np = np.asarray(labels)
    base_pred = np.asarray(base_pred)
    new_pred = np.asarray(new_pred)
    eval_idx = np.flatnonzero(np.asarray(eval_mask, dtype=bool))
    if eval_idx.size == 0:
        return {}

    rng = np.random.default_rng(seed)
    macro_delta = []
    acc_delta = []
    bot_delta = []
    for _ in range(int(n_samples)):
        sample_idx = rng.choice(eval_idx, size=eval_idx.size, replace=True)
        y = labels_np[sample_idx]
        base_s = base_pred[sample_idx]
        new_s = new_pred[sample_idx]
        acc_delta.append(accuracy_score(y, new_s) - accuracy_score(y, base_s))
        macro_delta.append(
            f1_score(y, new_s, average="macro", zero_division=0) - f1_score(y, base_s, average="macro", zero_division=0)
        )
        bot_delta.append(
            f1_score(y, new_s, average="binary", zero_division=0) - f1_score(y, base_s, average="binary", zero_division=0)
        )

    def _ci(values):
        values = np.asarray(values, dtype=np.float32)
        return {
            "mean": float(values.mean()),
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
        }

    return {
        "accuracy_delta": _ci(acc_delta),
        "macro_f1_delta": _ci(macro_delta),
        "bot_f1_delta": _ci(bot_delta),
    }


def _empty_gain_cost_report():
    return {
        "peak_gpu_memory": None,
        "wall_clock_time": None,
        "trainable_parameter_count": None,
        "per_triggered_node_latency": None,
        "embedding_cache_size": None,
        "precompute_cost": None,
    }


class _LinearSemanticSourceHead(nn.Module):
    def __init__(self, in_dim, num_classes=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        h = self.dropout(self.norm(x))
        logits = self.classifier(h)
        prob = torch.softmax(logits, dim=-1)
        alpha = prob * float(self.num_classes) + 1.0
        uncertainty = self.num_classes / alpha.sum(dim=-1, keepdim=True)
        return {
            "z_sem": h,
            "logits_sem": logits,
            "alpha_sem": alpha,
            "prob_sem": prob,
            "u_sem": uncertainty,
        }


class _AdapterSemanticSourceHead(nn.Module):
    def __init__(self, in_dim, rank=64, num_classes=2, dropout=0.1):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, in_dim, bias=False)
        self.norm_out = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        h = self.norm_in(x)
        adapted = x + self.up(F.gelu(self.down(h)))
        z_sem = self.dropout(self.norm_out(adapted))
        logits = self.classifier(z_sem)
        prob = torch.softmax(logits, dim=-1)
        alpha = prob * float(self.num_classes) + 1.0
        uncertainty = self.num_classes / alpha.sum(dim=-1, keepdim=True)
        return {
            "z_sem": z_sem,
            "logits_sem": logits,
            "alpha_sem": alpha,
            "prob_sem": prob,
            "u_sem": uncertainty,
        }


_SEMANTIC_IB_EDL_SYMBOLS = None


def _load_semantic_ib_edl_symbols():
    global _SEMANTIC_IB_EDL_SYMBOLS
    if _SEMANTIC_IB_EDL_SYMBOLS is not None:
        return _SEMANTIC_IB_EDL_SYMBOLS

    module_path = Path(__file__).resolve().parents[2] / "code" / "semantic_ib_edl_head.py"
    if not module_path.exists():
        _SEMANTIC_IB_EDL_SYMBOLS = (None, None)
        return _SEMANTIC_IB_EDL_SYMBOLS

    spec = importlib.util.spec_from_file_location("_semantic_ib_edl_head", module_path)
    if spec is None or spec.loader is None:
        _SEMANTIC_IB_EDL_SYMBOLS = (None, None)
        return _SEMANTIC_IB_EDL_SYMBOLS

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _SEMANTIC_IB_EDL_SYMBOLS = (
        getattr(module, "SemanticIBEDLHead", None),
        getattr(module, "r_edl_loss", None),
    )
    return _SEMANTIC_IB_EDL_SYMBOLS


def run_legacy_graph_seed(args, seed, data, run, runtime_paths=None):
    runtime_paths = runtime_paths or prepare_path(args.experiment_name + f'_seed_{seed}')
    (
        lm_prt_ckpt_filepath,
        gnn_prt_ckpt_filepath,
        mlp_kd_ckpt_filepath,
        lm_ckpt_filepath,
        gnn_ckpt_filepath,
        _mlp_ckpt_filepath,
        lm_intermediate_data_filepath,
        gnn_intermediate_data_filepath,
        _mlp_intermediate_data_filepath,
    ) = runtime_paths

    device = _resolve_device(args.device)
    lm_trainer = LM_Trainer(
        model_name=args.LM_model,
        classifier_n_layers=args.LM_classifier_n_layers,
        classifier_hidden_dim=args.LM_classifier_hidden_dim,
        device=device,
        pretrain_epochs=args.LM_pretrain_epochs,
        optimizer_name=args.optimizer_LM,
        lr=args.lr_LM,
        weight_decay=args.weight_decay_LM,
        dropout=args.dropout,
        att_dropout=args.LM_att_dropout,
        lm_dropout=args.LM_dropout,
        activation=args.activation,
        warmup=args.warmup,
        label_smoothing_factor=args.label_smoothing_factor,
        pl_weight=args.alpha,
        max_length=args.max_length,
        batch_size=args.batch_size_LM,
        grad_accumulation=args.LM_accumulation,
        lm_epochs_per_iter=args.LM_epochs_per_iter,
        temperature=args.temperature,
        pl_ratio=args.pl_ratio_LM,
        eval_patience=args.LM_eval_patience,
        intermediate_data_filepath=lm_intermediate_data_filepath,
        ckpt_filepath=lm_ckpt_filepath,
        pretrain_ckpt_filepath=lm_prt_ckpt_filepath,
        raw_data_filepath=args.raw_data_filepath,
        train_idx=data['train_idx'],
        valid_idx=data['valid_idx'],
        test_idx=data['test_idx'],
        hard_labels=data['labels'],
        user_seq=data['user_text'],
        run=run,
        peft_rank=getattr(args, 'peft_rank', 8),
        peft_alpha=getattr(args, 'peft_alpha', 16.0),
        qwen_model_path=getattr(args, 'qwen_model_path', None),
        qwen_trust_remote_code=getattr(args, 'qwen_trust_remote_code', False),
    )
    lm_trainer.build_model()
    lm_trainer.pretrain()

    gnn_trainer = GNN_Trainer(
        model_name=args.GNN_model,
        device=device,
        optimizer_name=args.optimizer_GNN,
        lr=args.lr_GNN,
        weight_decay=args.weight_decay_GNN,
        dropout=args.GNN_dropout,
        pl_weight=args.beta,
        batch_size=args.batch_size_GNN,
        gnn_n_layers=args.n_layers,
        n_relations=args.n_relations,
        activation=args.activation,
        gnn_epochs_per_iter=args.GNN_epochs_per_iter,
        temperature=args.temperature,
        pl_ratio=args.pl_ratio_GNN,
        intermediate_data_filepath=gnn_intermediate_data_filepath,
        ckpt_filepath=gnn_ckpt_filepath,
        pretrain_ckpt_filepath=gnn_prt_ckpt_filepath,
        train_idx=data['train_idx'],
        valid_idx=data['valid_idx'],
        test_idx=data['test_idx'],
        hard_labels=data['labels'],
        edge_index=data['edge_index'],
        edge_type=data['edge_type'],
        run=run,
        SimpleHGN_att_res=args.SimpleHGN_att_res,
        att_heads=args.att_heads,
        RGT_semantic_heads=args.RGT_semantic_heads,
        gnn_hidden_dim=args.hidden_dim,
        lm_name=args.LM_model,
    )
    gnn_trainer.build_model()

    gnn_best_exists = (gnn_ckpt_filepath / 'best.pkl').exists()
    lm_best_exists = (lm_ckpt_filepath / 'best.pkl').exists()
    if not (getattr(args, "reuse_existing_artifacts", False) and gnn_best_exists and lm_best_exists) or getattr(args, "force_retrain_backbone", False):
        for iter_idx in range(args.max_iters):
            print(f'------Iter: {iter_idx}/{args.max_iters-1}------')
            embeddings_lm, soft_labels_lm = load_distilled_knowledge('LM', lm_intermediate_data_filepath, iter_idx - 1)
            gnn_early_stop = gnn_trainer.train(embeddings_lm, soft_labels_lm)
            gnn_trainer.infer(embeddings_lm)
            if gnn_early_stop:
                print(f'Early stop by GNN at iter {iter_idx}!')
                break

            soft_labels_gnn = load_distilled_knowledge('GNN', gnn_intermediate_data_filepath, iter_idx)
            lm_early_stop = lm_trainer.train(soft_labels_gnn)
            lm_trainer.infer()
            if lm_early_stop:
                print(f'Early stop by LM at iter {iter_idx}!')
                break

    embedding_files = sorted(lm_intermediate_data_filepath.glob('embeddings_iter_*.pt'))
    if not embedding_files:
        raise FileNotFoundError(f'No LM embeddings found under {lm_intermediate_data_filepath}')
    latest_embedding = embedding_files[-1]
    best_iter = max(getattr(gnn_trainer, "best_iter", 0) - 1, -1)
    best_embedding_path = lm_intermediate_data_filepath / f'embeddings_iter_{best_iter}.pt'
    if best_embedding_path.exists():
        embeddings_best = safe_torch_load(best_embedding_path, map_location='cpu')
    else:
        embeddings_best = safe_torch_load(latest_embedding, map_location='cpu')

    return {
        "lm_trainer": lm_trainer,
        "gnn_trainer": gnn_trainer,
        "embeddings_best": embeddings_best,
        "runtime_paths": runtime_paths,
    }


class StageRunner:
    def __init__(self, args, seed, data, run):
        self.args = args
        self.seed = seed
        self.data = data
        self.run_handle = run
        self.device = _resolve_device(args.device)
        self.experiment_root = build_experiment_root(args, seed)
        self.runtime_root = self.experiment_root / "runtime"
        self.labels = _labels_to_index(data["labels"]).cpu().numpy()
        self.train_mask = _mask_from_idx(len(self.labels), data["train_idx"])
        self.val_mask = _mask_from_idx(len(self.labels), data["valid_idx"])
        self.test_mask = _mask_from_idx(len(self.labels), data["test_idx"])
        self.base_context = None
        self.requested_stage = getattr(args, "requested_stage", args.stage)

    def ensure_backbone_context(self):
        if self.base_context is not None:
            return self.base_context
        frozen = load_frozen_g0(self.experiment_root)
        gnn_outputs = frozen["outputs"]
        self.base_context = {
            "frozen_g0": frozen,
            "runtime_dir": frozen["dir"],
            "gnn_outputs": gnn_outputs,
        }
        return self.base_context

    def _base_artifact_bundle(self):
        code_provenance = capture_code_metadata(Path(__file__).resolve().parents[2])
        return {
            "resolved_config": vars(self.args),
            "split_manifest": {
                "seed": self.seed,
                "train_size": int(self.train_mask.sum()),
                "valid_size": int(self.val_mask.sum()),
                "test_size": int(self.test_mask.sum()),
            },
            "fold_manifest": {"type": "single_split", "seed": self.seed},
            "node_id_manifest": {"num_nodes": int(len(self.labels))},
            "code_commit": code_provenance["commit"],
            "code_provenance": code_provenance,
        }

    def _read_stage_json(self, stage_name, filename):
        path = self.experiment_root / "stages" / stage_name / f"{filename}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _require_stage_json(self, stage_name, filename):
        path = self.experiment_root / "stages" / stage_name / f"{filename}.json"
        if not path.exists():
            raise MissingFrozenArtifactError(
                f"Stage '{self.args.stage}' requires frozen dependency {path}. "
                f"Run --stage {stage_name} first; strict stages never recompute upstream artifacts."
            )
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _require_stage_tensor(self, stage_name, filename):
        path = self.experiment_root / "stages" / stage_name / f"{filename}.pt"
        if not path.exists():
            raise MissingFrozenArtifactError(
                f"Stage '{self.args.stage}' requires frozen dependency {path}. "
                f"Run --stage {stage_name} first; strict stages never recompute upstream artifacts."
            )
        return safe_torch_load(path, map_location="cpu")

    def _load_vertical_minimal_dependency(self):
        return {
            "risk_manifest": self._require_stage_json("vertical_minimal", "risk_manifest"),
            "structural_manifest": self._require_stage_json("vertical_minimal", "structural_manifest"),
            "subgroup_manifest_operational": self._require_stage_json("vertical_minimal", "subgroup_manifest_operational"),
            "subgroup_manifest_analysis": self._require_stage_json("vertical_minimal", "subgroup_manifest_analysis"),
            "survivor_manifest": self._require_stage_json("vertical_minimal", "survivor_manifest"),
            "all_node_outputs": self._require_stage_tensor("vertical_minimal", "all_node_outputs"),
        }

    def _load_final_output_dependency(self):
        return {
            "all_node_outputs": self._require_stage_tensor("vertical_minimal", "all_node_outputs"),
            "survivor_manifest": self._require_stage_json("vertical_minimal", "survivor_manifest"),
        }

    def _operational_groups(self, risk_bundle):
        operational = risk_bundle["subgroup_manifest_operational"]
        return operational.get("groups", operational)

    def _analysis_groups(self, risk_bundle):
        analysis = risk_bundle["subgroup_manifest_analysis"]
        return analysis.get("groups", analysis)

    def _validation_frozen_high_mask(self, risk_bundle):
        operational = risk_bundle["subgroup_manifest_operational"]
        groups = self._operational_groups(risk_bundle)
        strata = groups.get("validation_frozen_risk_strata", operational.get("validation_frozen_risk_strata", {}))
        high_group = strata.get("high") or strata.get("top_15") or {}
        return np.asarray(high_group.get("mask", np.zeros(len(self.labels), dtype=bool)), dtype=bool)

    def _policy_masks(self, risk_bundle):
        operational = risk_bundle["subgroup_manifest_operational"]
        if operational.get("selector_safe") is not True:
            raise ValueError("Policy masks require selector-safe operational subgroup manifest.")
        groups = self._operational_groups(risk_bundle)
        selector_views = operational.get("selector_safe_views", {})
        sparse_group = groups.get("degree_buckets", {}).get("low", {})
        repair_group = groups.get("neigh_inconsistency", {})
        sparse_mask = np.asarray(
            selector_views.get(
                "sparse_mask",
                operational.get("policy_tags", {}).get("sparse", sparse_group).get("mask", np.zeros(len(self.labels), dtype=bool)),
            ),
            dtype=bool,
        )
        repair_focus = np.asarray(
            selector_views.get(
                "prop_mask",
                operational.get("policy_tags", {}).get("repair_focus", repair_group).get("mask", np.zeros(len(self.labels), dtype=bool)),
            ),
            dtype=bool,
        )
        high_risk = self._validation_frozen_high_mask(risk_bundle)
        return sparse_mask, repair_focus, high_risk

    def _embedding_cache_size(self, context=None):
        context = context or self.ensure_backbone_context()
        outputs = context["gnn_outputs"]
        tensor = outputs.get("node_repr")
        if tensor is None:
            tensor = outputs.get("logits")
        if tensor is None:
            return 0
        return int(tensor.numel() * tensor.element_size())

    def _count_trainable_parameters(self, context=None):
        return 0

    def _find_qwen_embedding_path(self):
        dataset_path = Path(self.data["dataset_path"])
        direct = dataset_path / "qwen3_emb_last.pt"
        if direct.exists():
            return direct
        for pattern in ("**/qwen3_emb_last.pt", "**/qwen3_emb_L-1.pt"):
            candidates = sorted(dataset_path.glob(pattern))
            if candidates:
                return candidates[0]
        return None

    def _router_requested(self):
        estimator_mode = str(getattr(self.args, "estimator_mode", "none")).lower()
        return (
            estimator_mode
            in {
                "calibrated_multiview_router",
                "calibrated_residual_risk_selector",
                "glance_for_context_residual_risk_selector",
            }
            or str(getattr(self.args, "router_mode", "off")).lower() == "multiview"
        )

    def _final_output_ranker_requested(self):
        return str(getattr(self.args, "estimator_mode", "none")).lower() == "final_output_ranker"

    def _dual_posterior_ranker_requested(self):
        return str(getattr(self.args, "estimator_mode", "none")).lower() == "dual_posterior_ranker"

    def _login_uncertainty_router_requested(self):
        return str(getattr(self.args, "estimator_mode", "none")).lower() == "login_uncertainty_router"

    def _graph_conformal_estimator_requested(self):
        return str(getattr(self.args, "estimator_mode", "none")).lower() in {
            "graph_conformal_set_estimator",
            "gnn_2hop_conformal",
        }

    def _router_budgets(self):
        return parse_budget_list(getattr(self.args, "risk_budgets", None) or getattr(self.args, "router_budgets", None), default=RESIDUAL_RISK_PAPER_BUDGETS)

    def _final_output_ranker_budgets(self):
        return parse_budget_list(getattr(self.args, "risk_budgets", None) or getattr(self.args, "router_budgets", None), default=FINAL_OUTPUT_RANKER_BUDGETS)

    def _router_oof_dir(self):
        return Path(self.experiment_root) / "frozen" / "router_oof"

    def _load_router_oof_artifact(self):
        if str(getattr(self.args, "router_oof_mode", "artifact")).lower() != "artifact":
            raise MissingFrozenArtifactError("Stage 2 router only supports --router_oof_mode artifact in this implementation.")
        oof_dir = self._router_oof_dir()
        manifest_path = oof_dir / "manifest.json"
        outputs_path = oof_dir / "outputs.pt"
        if not manifest_path.exists() or not outputs_path.exists():
            raise MissingFrozenArtifactError(
                "calibrated_multiview_router requires frozen train-split OOF artifacts under "
                f"{oof_dir}. Expected manifest.json and outputs.pt; direct train correctness is not allowed."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = safe_torch_load(outputs_path, map_location="cpu")
        required = {"oof_idx", "lm_prob_cal", "gnn_prob_cal"}
        missing = sorted(required - set(outputs))
        if missing:
            raise MissingFrozenArtifactError(f"Router OOF outputs missing required keys: {missing}")
        for key in ("lm_prob_cal", "gnn_prob_cal"):
            tensor = outputs[key]
            if not torch.is_tensor(tensor) or tensor.dim() != 2 or tensor.size(1) != 2:
                raise MissingFrozenArtifactError(f"Router OOF '{key}' must be a [num_nodes, 2] probability tensor.")
            if int(tensor.shape[0]) != int(len(self.labels)):
                raise MissingFrozenArtifactError(
                    f"Router OOF '{key}' must be a full-node tensor aligned with Stage 1 outputs."
                )
        oof_idx = _idx_numpy(outputs["oof_idx"])
        if oof_idx.size == 0:
            raise MissingFrozenArtifactError("Router OOF 'oof_idx' must contain train-split held-out nodes.")
        if int(oof_idx.min()) < 0 or int(oof_idx.max()) >= int(len(self.labels)):
            raise MissingFrozenArtifactError("Router OOF 'oof_idx' contains node ids outside the dataset range.")
        train_idx = set(int(item) for item in _idx_numpy(self.data["train_idx"]).tolist())
        if not set(int(item) for item in oof_idx.tolist()).issubset(train_idx):
            raise MissingFrozenArtifactError(
                "Router OOF 'oof_idx' must be a subset of train_idx; validation/test correctness is not allowed for router training."
            )
        return {"manifest": manifest, "outputs": outputs, "dir": oof_dir}

    def _lm_only_head_dir(self):
        return Path(self.experiment_root) / "frozen" / "lm_only_head"

    def _load_lm_only_head(self):
        out_dir = self._lm_only_head_dir()
        manifest_path = out_dir / "manifest.json"
        outputs_path = out_dir / "outputs.pt"
        if manifest_path.exists() and outputs_path.exists():
            return {
                "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
                "outputs": safe_torch_load(outputs_path, map_location="cpu"),
                "dir": out_dir,
                "source": "existing_frozen_lm_only_head",
            }
        return None

    def _build_or_load_lm_only_head(self):
        existing = self._load_lm_only_head()
        if existing is not None:
            return existing
        if bool(getattr(self.args, "claim_grade", False)):
            raise MissingFrozenArtifactError(
                "claim_grade dual_posterior_ranker requires frozen/lm_only_head artifacts; "
                "implicit upstream recompute is disallowed."
            )

        context = self.ensure_backbone_context()
        feature_manifest = context["frozen_g0"]["manifest"].get("feature_manifest", {})
        feature_path = feature_manifest.get("path")
        if not feature_path or not Path(feature_path).exists():
            raise MissingFrozenArtifactError(
                "Stage 2 router requires LM-only probabilities. No frozen LM-only head exists and "
                "frozen G0 does not record a readable semantic embedding path for a lightweight LM-only head."
            )

        feature_args = SimpleNamespace(**vars(self.args))
        feature_args.emb_path = str(feature_path)
        feature_args.g0_feature_path = str(feature_path)
        if feature_manifest.get("projected_dim") is not None:
            feature_args.phase_a_project_dim = int(feature_manifest["projected_dim"])
        if feature_manifest.get("projector") is not None:
            feature_args.phase_a_projector = str(feature_manifest["projector"])
        feature_args.peft = False
        feature_bundle = resolve_g0_feature_bundle(feature_args, self.data)
        features = feature_bundle["features"].detach().cpu().float().numpy()
        labels = np.asarray(self.labels, dtype=np.int64)
        train_idx = _idx_numpy(self.data["train_idx"])
        valid_idx = _idx_numpy(self.data["valid_idx"])

        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train_idx])
        x_all = scaler.transform(features)
        classes = np.unique(labels[train_idx])
        if classes.size < 2:
            prob = np.zeros((len(labels), 2), dtype=np.float32)
            prob[:, int(classes[0])] = 1.0
            classifier_payload = {"type": "constant", "class": int(classes[0])}
        else:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(x_train, labels[train_idx])
            prob = clf.predict_proba(x_all).astype(np.float32)
            if prob.shape[1] == 1:
                fixed = np.zeros((len(labels), 2), dtype=np.float32)
                fixed[:, int(clf.classes_[0])] = prob[:, 0]
                prob = fixed
            elif list(clf.classes_) != [0, 1]:
                fixed = np.zeros((len(labels), 2), dtype=np.float32)
                for col, cls in enumerate(clf.classes_):
                    fixed[:, int(cls)] = prob[:, col]
                prob = fixed
            classifier_payload = {
                "type": "logistic_regression",
                "classes": [int(item) for item in getattr(clf, "classes_", [0, 1])],
                "coef": getattr(clf, "coef_", np.zeros((1, features.shape[1]))).tolist(),
                "intercept": getattr(clf, "intercept_", np.zeros(1)).tolist(),
            }

        logits = torch.log(torch.tensor(prob, dtype=torch.float32).clamp_min(1e-8))
        temperature = 1.0
        if valid_idx.size > 0 and len(np.unique(labels[valid_idx])) > 1:
            temperature = fit_temperature_scaling(logits[valid_idx], labels[valid_idx])
        prob_cal = torch.softmax(logits / float(temperature), dim=1)
        outputs = {
            "logits": logits,
            "prob": torch.tensor(prob, dtype=torch.float32),
            "prob_cal": prob_cal.detach().cpu(),
            "pred": prob_cal.argmax(dim=1).detach().cpu(),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        out_dir = self._lm_only_head_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(outputs, out_dir / "outputs.pt")
        torch.save(
            {
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "classifier": classifier_payload,
                "temperature": float(temperature),
            },
            out_dir / "checkpoint.pt",
        )
        manifest = {
            "contract": "lm_only_head_v1",
            "training_scope": "train_idx_supervised_lightweight_head",
            "calibration_scope": "valid_idx_temperature_scaling",
            "input_source": "cached_semantic_embedding_from_frozen_g0_manifest",
            "feature_manifest": feature_bundle["feature_manifest"],
            "temperature": float(temperature),
            "split_counts": {
                "train": int(train_idx.size),
                "valid": int(valid_idx.size),
                "test": int(_idx_numpy(self.data["test_idx"]).size),
            },
        }
        write_json(out_dir / "manifest.json", manifest)
        return {"manifest": manifest, "outputs": outputs, "dir": out_dir, "source": "derived_from_frozen_g0_feature_manifest"}

    def _build_multiview_router_bundle(self, gats_bundle):
        oof = self._load_router_oof_artifact()
        lm_bundle = self._build_or_load_lm_only_head()
        context = self.ensure_backbone_context()
        gnn_outputs = context["gnn_outputs"]
        gats_outputs = gats_bundle["outputs"]
        gnn_prob_cal = gats_outputs.get("prob_graph_cal", gnn_outputs["prob"]).detach().cpu().float()
        lm_prob_cal = lm_bundle["outputs"].get("prob_cal", lm_bundle["outputs"]["prob"]).detach().cpu().float()
        budgets = self._router_budgets()
        estimator_mode = str(getattr(self.args, "estimator_mode", "calibrated_multiview_router")).lower()
        risk_variant = getattr(self.args, "risk_variant", None) or getattr(self.args, "router_variant", "router_4_view_ensemble")
        risk_variant_key = str(risk_variant).lower()
        selector_modes = {"calibrated_residual_risk_selector", "glance_for_context_residual_risk_selector"}
        if estimator_mode in selector_modes:
            if estimator_mode == "glance_for_context_residual_risk_selector" or risk_variant_key in {
                "glance_for_context",
                "glance_residual_risk_selector",
                "glance_router",
            }:
                estimator = GlanceForContextResidualRiskSelector(
                    budgets=budgets,
                    training_objective=getattr(self.args, "glance_training_objective", "residual_error"),
                    llm_query_cost=getattr(self.args, "glance_llm_query_cost", 0.2),
                )
                paper_identity = "GLANCE-inspired Node-Aware Residual-Risk Selector"
            elif risk_variant_key in {"gets_posthoc", "posthoc_gets", "gets_style_posthoc"}:
                estimator = GETSStylePostHocResidualRiskSelector(risk_variant=risk_variant, budgets=budgets)
                paper_identity = "Calibrated Residual-Risk Hard-Node Selector"
            else:
                estimator = CalibratedResidualRiskHardNodeSelector(
                    risk_variant=risk_variant,
                    lambda_rank=getattr(self.args, "router_lambda_rank", 0.2),
                    rank_margin=getattr(self.args, "router_rank_margin", 0.1),
                    budgets=budgets,
                )
                paper_identity = "Calibrated Residual-Risk Hard-Node Selector"
            artifact_mode = estimator_mode
        else:
            estimator = CalibratedMultiViewResidualRouter(
                variant=getattr(self.args, "router_variant", "router_4_view_ensemble"),
                lambda_rank=getattr(self.args, "router_lambda_rank", 0.2),
                rank_margin=getattr(self.args, "router_rank_margin", 0.1),
                budgets=budgets,
            )
            artifact_mode = "calibrated_multiview_router"
            paper_identity = "legacy calibrated multi-view residual-risk selector"
        if estimator_mode in selector_modes:
            legacy_variant = getattr(estimator, "variant", str(risk_variant))
        else:
            legacy_variant = getattr(self.args, "router_variant", "router_4_view_ensemble")
        oof_outputs = oof["outputs"]
        estimator.fit(
            logits=torch.log(oof_outputs["gnn_prob_cal"].float().clamp_min(1e-8)),
            probs=oof_outputs["gnn_prob_cal"].float(),
            labels=torch.tensor(self.labels, dtype=torch.long),
            train_idx=self.data["train_idx"],
            val_idx=self.data["valid_idx"],
            edge_index=self.data["edge_index"],
            edge_type=self.data["edge_type"],
            lm_probs=oof_outputs["lm_prob_cal"].float(),
            fit_idx=oof_outputs["oof_idx"],
            oof_metadata=oof["manifest"],
            training_objective=getattr(self.args, "glance_training_objective", "residual_error"),
            llm_query_cost=getattr(self.args, "glance_llm_query_cost", 0.2),
        )
        risk_manifest = estimator.build_manifest(
            logits=gnn_outputs["logits"],
            probs=gnn_prob_cal,
            labels=torch.tensor(self.labels, dtype=torch.long),
            val_idx=self.data["valid_idx"],
            test_idx=self.data["test_idx"],
            edge_index=self.data["edge_index"],
            edge_type=self.data["edge_type"],
            lm_probs=lm_prob_cal,
            budgets=budgets,
        )
        risk_manifest["calibration_metadata"] = {
            **risk_manifest.get("calibration_metadata", {}),
            "source": artifact_mode,
            "paper_identity": paper_identity,
            "risk_variant": str(risk_variant),
            "legacy_router_variant": str(legacy_variant),
            "lm_only_head_manifest": str(self._lm_only_head_dir() / "manifest.json"),
            "router_oof_manifest": str(self._router_oof_dir() / "manifest.json"),
        }
        bundle = self._bundle_from_risk_manifest(artifact_mode, risk_manifest)
        test_idx = _idx_numpy(self.data["test_idx"])
        wrong = (gnn_prob_cal.argmax(dim=1).numpy() != self.labels).astype(np.int32)
        router_budget_rows = router_budget_curve(
            wrong[test_idx],
            np.asarray(risk_manifest["risk_score"], dtype=np.float32)[test_idx],
            budgets=budgets,
        )
        router_metrics = {
            "validation": risk_manifest.get("validation_metrics", {}),
            "test": risk_manifest.get("test_metrics", {}),
            "budget_curve": router_budget_rows,
            "variant": str(legacy_variant),
        }
        router_manifest = {
            "contract": "calibrated_multiview_residual_router_v1",
            "estimator_mode": "calibrated_multiview_router",
            "router_mode": "multiview",
            "variant": str(legacy_variant),
            "training_label": "z_i = 1[base_gnn_prediction != y_i]",
            "target": "Stage 1 base detector failure probability",
            "input_boundary": estimator.training_summary.get("forbidden_inputs_audit", {}),
            "oof_required": True,
            "oof_folds": int(getattr(self.args, "router_oof_folds", 5)),
            "oof_mode": getattr(self.args, "router_oof_mode", "artifact"),
            "view_names": estimator.training_summary.get(
                "feature_family",
                ["calibrated_prediction", "semantic_graph_disagreement", "graph_context_consistency"],
            ),
        }
        router_artifacts = {
            "router_manifest": router_manifest,
            "router_metrics": router_metrics,
            "router_oof_manifest": oof["manifest"],
            "router_scores.pt": torch.tensor(risk_manifest["risk_score"], dtype=torch.float32),
            "router_view_weights.pt": estimator.last_view_weights if estimator.last_view_weights is not None else torch.empty(0),
            "router_view_scores.pt": estimator.last_view_scores if estimator.last_view_scores is not None else torch.empty(0),
            "router_checkpoint.pt": estimator.state_dict_payload(),
            "lm_only_head_manifest": lm_bundle["manifest"],
        }
        if artifact_mode in {"calibrated_residual_risk_selector", "glance_for_context_residual_risk_selector"}:
            selector_manifest = {
                "contract": "calibrated_residual_risk_hard_node_selector_v1",
                "paper_name": paper_identity,
                "estimator_mode": artifact_mode,
                "risk_variant": str(risk_variant),
                "legacy_router_variant": str(legacy_variant),
                "training_label": "z_i = 1[base_gnn_prediction != y_i]",
                "target": "P(Stage1 prediction is wrong | Stage1 outputs/features)",
                "training_objective": estimator.training_summary.get("training_objective", "residual_error"),
                "target_semantics": estimator.training_summary.get("target_semantics", "1[Stage1 prediction != y]"),
                "input_boundary": estimator.training_summary.get("forbidden_inputs_audit", {}),
                "oof_required": True,
                "oof_folds": int(getattr(self.args, "router_oof_folds", 5)),
                "screening_only": True,
                "diagnosis_or_action": False,
                "positioning": "calibration/ranking selector; not a Stage 3 action router",
                "view_names": estimator.training_summary.get(
                    "feature_family",
                    ["calibrated_prediction", "semantic_graph_disagreement", "graph_context_consistency"],
                ),
            }
            router_artifacts.update(
                {
                    "residual_risk_selector_manifest": selector_manifest,
                    "residual_risk_selector_metrics": router_metrics,
                    "residual_risk_scores.pt": torch.tensor(risk_manifest["risk_score"], dtype=torch.float32),
                    "residual_risk_view_weights.pt": estimator.last_view_weights if estimator.last_view_weights is not None else torch.empty(0),
                    "residual_risk_view_scores.pt": estimator.last_view_scores if estimator.last_view_scores is not None else torch.empty(0),
                    "residual_risk_selector_checkpoint.pt": estimator.state_dict_payload(),
                }
            )
        return bundle, router_artifacts, router_budget_rows

    @staticmethod
    def _first_available_tensor(mapping, keys):
        for key in keys:
            value = mapping.get(key)
            if value is not None:
                return value
        return None

    def _build_login_uncertainty_router_bundle(self):
        context = self.ensure_backbone_context()
        frozen = context["frozen_g0"]
        gnn_outputs = context["gnn_outputs"]
        budgets = self._router_budgets()
        estimator = build_estimator("login_uncertainty_router")
        mc_dropout_logits = self._first_available_tensor(
            gnn_outputs,
            ("mc_dropout_logits", "dropout_logits", "logits_mc", "mc_logits"),
        )
        mc_dropout_probs = self._first_available_tensor(
            gnn_outputs,
            ("mc_dropout_probs", "dropout_probs", "prob_mc", "mc_probs"),
        )
        labels = torch.tensor(self.labels, dtype=torch.long)
        estimator.fit(
            logits=gnn_outputs.get("logits"),
            probs=gnn_outputs.get("prob"),
            labels=labels,
            train_idx=self.data["train_idx"],
            val_idx=self.data["valid_idx"],
            test_idx=self.data["test_idx"],
            mc_dropout_logits=mc_dropout_logits,
            mc_dropout_probs=mc_dropout_probs,
        )
        risk_manifest = estimator.build_manifest(
            logits=gnn_outputs.get("logits"),
            probs=gnn_outputs.get("prob"),
            labels=labels,
            val_idx=self.data["valid_idx"],
            test_idx=self.data["test_idx"],
            mc_dropout_logits=mc_dropout_logits,
            mc_dropout_probs=mc_dropout_probs,
            budgets=budgets,
        )
        frozen_manifest = frozen.get("manifest", {})
        feature_manifest = frozen_manifest.get("feature_manifest", {})
        risk_manifest["calibration_metadata"] = {
            **risk_manifest.get("calibration_metadata", {}),
            "source": "login_uncertainty_router",
            "paper_identity": "LOGIN-style Uncertainty Hard-Node Router",
            "paper_identity_risk": "medium_until_official_code_verified",
            "promotion_rule": "ablation_only_never_canonical",
            "simteg_like_input": "phase_a_cached_semantic_embedding_to_frozen_gnn",
            "semantic_source": feature_manifest.get("source", "unknown"),
            "semantic_feature_path": feature_manifest.get("path"),
            "embedding_manifest": feature_manifest,
            "gnn_backbone": frozen_manifest.get("backbone", getattr(self.args, "GNN_model", "unknown")),
            "frozen_g0_manifest": str(Path(frozen["dir"]) / "manifest.json"),
            "split_provenance": frozen_manifest.get("split_provenance", {}),
            "mc_dropout_available": bool(estimator.mc_dropout_available),
            "official_code_verified": False,
            "official_code_source": "https://github.com/QiaoYRan/LOGIN_init",
            "official_code_verification_note": "Repository URL was provided, but implementation files were not locally available in this run.",
        }
        bundle = self._bundle_from_risk_manifest("login_uncertainty_router", risk_manifest)
        test_idx = _idx_numpy(self.data["test_idx"])
        probs = gnn_outputs["prob"].detach().cpu()
        wrong = (probs.argmax(dim=1).numpy() != self.labels).astype(np.int32)
        risk_score = np.asarray(risk_manifest["risk_score"], dtype=np.float32)
        login_budget_rows = router_budget_curve(wrong[test_idx], risk_score[test_idx], budgets=budgets)
        login_metrics = {
            "validation": risk_manifest.get("validation_metrics", {}),
            "test": risk_manifest.get("test_metrics", {}),
            "budget_curve": login_budget_rows,
            "mc_dropout_available": bool(estimator.mc_dropout_available),
            "official_code_verified": False,
        }
        login_manifest = {
            "contract": "login_uncertainty_router_v1",
            "estimator_mode": "login_uncertainty_router",
            "target": "Stage 1 base detector failure probability",
            "training_label": "z_i = 1[base_gnn_prediction != y_i]",
            "input_boundary": risk_manifest["calibration_metadata"].get("input_boundary", {}),
            "simteg_like_input": "phase_a_cached_semantic_embedding_to_frozen_gnn",
            "semantic_source": risk_manifest["calibration_metadata"].get("semantic_source"),
            "gnn_backbone": risk_manifest["calibration_metadata"].get("gnn_backbone"),
            "screening_only": True,
            "diagnosis_or_action": False,
            "llm_call": False,
            "login_scope": "hard_node_selection_only",
            "mc_dropout_available": bool(estimator.mc_dropout_available),
            "official_code_verified": False,
            "official_code_source": "https://github.com/QiaoYRan/LOGIN_init",
            "official_code_verification_note": "Repository URL was provided, but implementation files were not locally available in this run.",
            "frozen_g0_manifest": str(Path(frozen["dir"]) / "manifest.json"),
            "budgets": [float(item) for item in budgets],
        }
        login_artifacts = {
            "login_router_manifest": login_manifest,
            "login_router_metrics": login_metrics,
            "login_router_scores.pt": torch.tensor(risk_score, dtype=torch.float32),
            "login_router_checkpoint.pt": estimator.state_dict_payload(),
        }
        return bundle, login_artifacts, login_budget_rows

    def _build_graph_conformal_estimator_bundle(self, estimator_mode):
        context = self.ensure_backbone_context()
        gnn_outputs = context["gnn_outputs"]
        budgets = self._router_budgets()
        estimator = build_estimator(estimator_mode)
        labels = gnn_outputs.get("labels")
        if labels is None:
            labels = torch.tensor(self.labels, dtype=torch.long)
        estimator.fit(
            logits=gnn_outputs.get("logits"),
            probs=gnn_outputs.get("prob"),
            labels=labels,
            train_idx=self.data["train_idx"],
            val_idx=self.data["valid_idx"],
            edge_index=self.data["edge_index"],
            edge_type=self.data["edge_type"],
            node_repr=gnn_outputs.get("node_repr"),
            budgets=budgets,
        )
        risk_manifest = estimator.build_manifest(
            logits=gnn_outputs.get("logits"),
            probs=gnn_outputs.get("prob"),
            labels=labels,
            val_idx=self.data["valid_idx"],
            test_idx=self.data["test_idx"],
            edge_index=self.data["edge_index"],
            edge_type=self.data["edge_type"],
            node_repr=gnn_outputs.get("node_repr"),
            budgets=budgets,
        )
        risk_manifest["calibration_metadata"] = {
            **risk_manifest.get("calibration_metadata", {}),
            "source": estimator_mode,
            "paper_identity": getattr(estimator, "metadata", {}).get(
                "paper_identity",
                "Graph Conformal Prediction-Set Ego Quality Estimator",
            ),
            "stage_role": getattr(estimator, "metadata", {}).get("stage2_role", "hard_node_quality_gate"),
            "simteg_like_input": "phase_a_cached_semantic_embedding_to_frozen_gnn",
            "router_boundary": (
                "post-hoc prediction-set quality gate; no learned router head, no LLM output, "
                "and no LM-GNN disagreement signal"
            ),
        }
        return self._bundle_from_risk_manifest(estimator_mode, risk_manifest)

    def _run_graph_conformal_estimator_matrix(self, stage_dir, base_bundle):
        requested_mode = str(getattr(self.args, "estimator_mode", "none")).lower()
        risk_bundle = self._build_graph_conformal_estimator_bundle(requested_mode)
        context = self.ensure_backbone_context()
        base_pred = context["gnn_outputs"]["pred"].detach().cpu().numpy()
        triggered_mask = self._validation_frozen_high_mask(risk_bundle) & self.test_mask
        budgets = self._router_budgets()
        metrics_payload, budget_curve = self._build_metrics_payload(
            risk_bundle,
            base_pred,
            base_pred,
            self.test_mask,
            triggered_mask,
            extra_metrics={"best_p_proxy": requested_mode},
            budgets=budgets,
        )
        paper_bundle = self._build_stage2_paper_report(
            requested_mode,
            risk_bundle,
            context["gnn_outputs"],
            budgets,
        )
        paper_bundle["report"]["claim_status"] = "exploratory_conformal_hard_node_probe"
        paper_bundle["acceptance_gate"]["claim_status"] = "exploratory_conformal_hard_node_probe"
        self._write_stage2_paper_artifacts(stage_dir, paper_bundle)
        metrics_payload["stage2_acceptance_gate"] = paper_bundle["acceptance_gate"]
        metrics_payload["stage2_claim_status"] = paper_bundle["report"]["claim_status"]
        notes = [
            f"claim tested: {requested_mode} hard-node selection from frozen Phase A GNN posterior",
            "scope: screening only; no LLM call, no graph rewrite, no GNN retrain",
            "boundary: conformal-style post-hoc quality proxy; no coverage theorem is claimed after graph edits",
        ]
        self._write_stage_outputs(
            stage_dir,
            risk_bundle,
            metrics_payload,
            budget_curve,
            notes,
            extra_artifacts={
                **base_bundle,
                "dependency_manifest": {
                    "stage_contract": "strict_frozen_dependency_load_v1",
                    "reads": ["frozen/g0/manifest.json", "frozen/g0/outputs.pt"],
                    "recomputed_upstream": False,
                },
                "summary": {requested_mode: risk_bundle["risk_manifest"].get("validation_metrics", {})},
                "risk_manifest": risk_bundle["risk_manifest"],
                "stage2_paper_report": paper_bundle["report"],
                "stage2_acceptance_gate": paper_bundle["acceptance_gate"],
                "stage2_hard_node_subgroups": paper_bundle["subgroup_report"],
                "structural_manifest": risk_bundle["structural_manifest"],
                "subgroup_manifest_operational": risk_bundle["subgroup_manifest_operational"],
                "subgroup_manifest_analysis": risk_bundle["subgroup_manifest_analysis"],
                "survivor_manifest": {
                    "best_p_proxy": requested_mode,
                    "proxy_suite": [requested_mode],
                    "stage2_paper_report": "stage2_paper_report.json",
                    "stage2_acceptance_gate": "stage2_acceptance_gate.json",
                    "claim_status": paper_bundle["report"]["claim_status"],
                },
            },
            subgroup_extra={"best_mode": requested_mode, "estimator_suite": [requested_mode]},
        )
        return {"stage": self.args.stage, "best_mode": requested_mode}

    def _run_login_uncertainty_router_matrix(self, stage_dir, base_bundle):
        risk_bundle, login_artifacts, login_budget_rows = self._build_login_uncertainty_router_bundle()
        context = self.ensure_backbone_context()
        base_pred = context["gnn_outputs"]["pred"].detach().cpu().numpy()
        triggered_mask = self._validation_frozen_high_mask(risk_bundle) & self.test_mask
        budgets = self._router_budgets()
        metrics_payload, budget_curve = self._build_metrics_payload(
            risk_bundle,
            base_pred,
            base_pred,
            self.test_mask,
            triggered_mask,
            extra_metrics={"best_p_proxy": "login_uncertainty_router"},
            budgets=budgets,
        )
        paper_bundle = self._build_stage2_paper_report(
            "login_uncertainty_router",
            risk_bundle,
            context["gnn_outputs"],
            budgets,
        )
        claim_boundary = {
            "official_code_verified": False,
            "official_code_source": "https://github.com/QiaoYRan/LOGIN_init",
            "implementation_status": "paper_faithful_uncertainty_router_without_official_code_parity",
            "allowed_use": "engineering_ablation_and_smoke_only",
            "forbidden_claims": [
                "official_LOGIN_reproduction",
                "full_LOGIN_loop",
                "SOTA_or_paper_claim",
                "LLM_feedback_or_graph_refinement_effect",
            ],
        }
        paper_bundle["report"]["claim_status"] = "ablation_only_official_login_code_unverified"
        paper_bundle["report"]["claim_boundary"] = claim_boundary
        paper_bundle["acceptance_gate"]["claim_status"] = "ablation_only_official_login_code_unverified"
        paper_bundle["acceptance_gate"]["claim_boundary"] = claim_boundary
        self._write_stage2_paper_artifacts(stage_dir, paper_bundle)
        metrics_payload["stage2_acceptance_gate"] = paper_bundle["acceptance_gate"]
        metrics_payload["stage2_claim_status"] = paper_bundle["report"]["claim_status"]
        notes = [
            "claim tested: LOGIN-style uncertainty hard-node selection from frozen Phase A GNN posterior",
            "scope: screening only; no LLM call, no semantic feature update, no graph rewrite, no GNN retrain",
            "official LOGIN_init code status: not locally verified in this run",
        ]
        self._write_stage_outputs(
            stage_dir,
            risk_bundle,
            metrics_payload,
            budget_curve,
            notes,
            extra_artifacts={
                **base_bundle,
                "dependency_manifest": {
                    "stage_contract": "strict_frozen_dependency_load_v1",
                    "reads": ["frozen/g0/manifest.json", "frozen/g0/outputs.pt"],
                    "recomputed_upstream": False,
                },
                "summary": {"login_uncertainty_router": risk_bundle["risk_manifest"].get("validation_metrics", {})},
                "risk_manifest": risk_bundle["risk_manifest"],
                "stage2_paper_report": paper_bundle["report"],
                "stage2_acceptance_gate": paper_bundle["acceptance_gate"],
                "stage2_hard_node_subgroups": paper_bundle["subgroup_report"],
                "structural_manifest": risk_bundle["structural_manifest"],
                "subgroup_manifest_operational": risk_bundle["subgroup_manifest_operational"],
                "subgroup_manifest_analysis": risk_bundle["subgroup_manifest_analysis"],
                "survivor_manifest": {
                    "best_p_proxy": "login_uncertainty_router",
                    "proxy_suite": ["login_uncertainty_router"],
                    "stage2_paper_report": "stage2_paper_report.json",
                    "stage2_acceptance_gate": "stage2_acceptance_gate.json",
                    "claim_status": paper_bundle["report"]["claim_status"],
                },
                **login_artifacts,
            },
            subgroup_extra={"best_mode": "login_uncertainty_router", "estimator_suite": ["login_uncertainty_router"]},
        )
        write_csv_rows(
            stage_dir / "login_router_budget_curve.csv",
            ["budget", "selected_count", "error_recall", "precision", "lift", "remaining_risk"],
            login_budget_rows,
        )
        return {"stage": self.args.stage, "best_mode": "login_uncertainty_router"}

    def _fit_semantic_source_head(self, embeddings, variant):
        self._strict_recompute_forbidden("_fit_semantic_source_head")

    def _semantic_result_from_source(self, risk_bundle, semantic_mode, source_variant, source_bundle):
        self._strict_recompute_forbidden("_semantic_result_from_source")

    def _compact_result_summary(self, results):
        compact = {}
        for name, result in results.items():
            compact[name] = {
                "metadata": result.get("metadata"),
                "available": result.get("available"),
                "status": result.get("status", "executed" if result.get("pred") is not None else "unknown"),
                "metrics": result.get("metrics"),
                "delta": result.get("delta"),
                "gain_cost_report": result.get("gain_cost_report"),
                "source_metrics": result.get("source_metrics"),
            }
        return compact

    def _preferred_semantic_source_name(self):
        manifest = self._read_stage_json("semantic_source_matrix", "survivor_manifest") or {}
        return manifest.get("best_semantic_source", "S0_roberta_local_source")

    def _resolve_best_semantic_action_result(self, risk_bundle, semantic_results):
        self._strict_recompute_forbidden("_resolve_best_semantic_action_result")

    def _stage_subgroup_report(self, risk_bundle, stage_extra=None):
        if risk_bundle is None:
            return {"stage_extra": stage_extra or {}}
        operational = risk_bundle["subgroup_manifest_operational"]
        analysis = risk_bundle["subgroup_manifest_analysis"]
        operational_groups = self._operational_groups(risk_bundle)
        analysis_groups = self._analysis_groups(risk_bundle)
        return {
            "degree_buckets": operational_groups.get("degree_buckets"),
            "local_consistency_or_homophily": {
                "operational": operational_groups.get("neigh_pred_agreement", operational.get("local_consistency")),
                "analysis": analysis_groups.get("oracle_homophily"),
            },
            "atomic_relation_skew": operational_groups.get("relation_skew_high"),
            "atomic_neighbor_inconsistency": operational_groups.get("neigh_inconsistency"),
            "camouflage_heavy": analysis_groups.get("oracle_camouflage_heavy"),
            "harmful_propagation": analysis_groups.get("harmful_propagation"),
            "operational": operational,
            "analysis": analysis,
            "stage_summary": stage_extra or {},
        }

    def _build_gain_cost_report(self, elapsed=None, triggered_mask=None, extra=None):
        report = _empty_gain_cost_report()
        report["embedding_cache_size"] = self._embedding_cache_size()
        report["trainable_parameter_count"] = self._count_trainable_parameters()
        if elapsed is not None:
            report["wall_clock_time"] = float(elapsed)
            triggered_count = int(np.asarray(triggered_mask, dtype=bool).sum()) if triggered_mask is not None else 0
            report["per_triggered_node_latency"] = float(elapsed / max(triggered_count, 1))
        if torch.cuda.is_available() and self.device.type == "cuda":
            report["peak_gpu_memory"] = _max_cuda_memory_allocated(self.device)
        if extra:
            report.update(extra)
        return report

    def _risk_manifest_from_score(self, risk_score, calibration_metadata=None):
        context = self.ensure_backbone_context()
        gnn_outputs = context["gnn_outputs"]
        risk_score = np.asarray(risk_score, dtype=np.float32)
        pred = gnn_outputs["pred"].detach().cpu().numpy()
        wrong = (pred != self.labels).astype(np.int32)
        val_idx = _idx_numpy(self.data["valid_idx"])
        test_idx = _idx_numpy(self.data["test_idx"])
        if val_idx.size:
            budget = max(int(val_idx.size * 0.15), 1)
            validation_threshold = float(np.sort(risk_score[val_idx])[-budget])
        else:
            validation_threshold = float("inf")
        confidence = gnn_outputs["prob"].detach().cpu().max(dim=1)[0].numpy()
        hcw_mask = ((1.0 - confidence) < 0.1) & (wrong == 1)
        return {
            "risk_score": risk_score,
            "thresholds": {"validation_risk_threshold": validation_threshold},
            "calibration_metadata": calibration_metadata or {},
            "validation_metrics": risk_metrics(wrong[val_idx], risk_score[val_idx]) if val_idx.size else {},
            "test_metrics": risk_metrics(wrong[test_idx], risk_score[test_idx]) if test_idx.size else {},
            "hcw_mask": hcw_mask.tolist(),
        }

    def _bundle_from_risk_manifest(self, mode, risk_manifest, structural_manifest=None, subgroup_op=None, subgroup_analysis=None):
        context = self.ensure_backbone_context()
        gnn_outputs = context["gnn_outputs"]
        probe_features = build_probe_features(
            logits=gnn_outputs["logits"],
            probs=gnn_outputs["prob"],
            edge_index=self.data["edge_index"],
            edge_type=self.data["edge_type"],
            node_repr=gnn_outputs["node_repr"],
        )
        if structural_manifest is None or subgroup_op is None or subgroup_analysis is None:
            structural_manifest, subgroup_op, subgroup_analysis = build_subgroup_manifests(
                edge_index=self.data["edge_index"],
                edge_type=self.data["edge_type"],
                num_nodes=len(self.labels),
                risk_manifest=risk_manifest,
                labels=self.labels,
                predictions=gnn_outputs["pred"].numpy(),
                probe_features=probe_features,
                degree_quantile=self.args.sparse_degree_quantile,
                propagation_quantile=self.args.propagation_quantile,
                train_idx=self.data["train_idx"],
            )
        return {
            "metadata": mode_metadata(mode),
            "risk_manifest": risk_manifest,
            "structural_manifest": structural_manifest,
            "subgroup_manifest_operational": subgroup_op,
            "subgroup_manifest_analysis": subgroup_analysis,
            "probe_features": probe_features,
        }

    def _bundle_from_final_output_ranker_manifest(self, risk_manifest, mode="final_output_ranker"):
        return {
            "metadata": mode_metadata(mode),
            "risk_manifest": risk_manifest,
        }

    def _ranked_budget_mask(self, risk_score, eval_mask, budget):
        risk_score = np.asarray(risk_score, dtype=np.float32)
        eval_mask = np.asarray(eval_mask, dtype=bool)
        candidate_idx = np.flatnonzero(eval_mask)
        mask = np.zeros_like(eval_mask, dtype=bool)
        if candidate_idx.size == 0:
            return mask
        ranked_idx = candidate_idx[np.argsort(-risk_score[candidate_idx])]
        k = min(max(int(candidate_idx.size * float(budget)), 1), candidate_idx.size)
        mask[ranked_idx[:k]] = True
        return mask

    def _build_final_output_candidate_manifest(self, risk_manifest, final_outputs, budgets):
        risk_score = np.asarray(risk_manifest["risk_score"], dtype=np.float32)
        probs = final_outputs.get("prob")
        if probs is None:
            probs = torch.softmax(final_outputs["logits"].float(), dim=1)
        probs_np = probs.detach().cpu().numpy()
        pred_np = final_outputs.get("pred")
        if pred_np is None:
            pred_np = probs.argmax(dim=1)
        pred_np = pred_np.detach().cpu().numpy()
        candidate_idx = np.flatnonzero(self.test_mask)
        ranked_idx = candidate_idx[np.argsort(-risk_score[candidate_idx])] if candidate_idx.size else np.asarray([], dtype=np.int64)
        candidate_sets = {}
        for budget in budgets:
            budget_key = f"top_{int(round(float(budget) * 100))}pct"
            k = min(max(int(candidate_idx.size * float(budget)), 1), candidate_idx.size) if candidate_idx.size else 0
            selected = ranked_idx[:k]
            candidate_sets[budget_key] = {
                "budget": float(budget),
                "selected_count": int(selected.size),
                "node_ids": [int(item) for item in selected.tolist()],
                "risk_scores": [float(risk_score[item]) for item in selected.tolist()],
                "final_predictions": [int(pred_np[item]) for item in selected.tolist()],
                "final_confidences": [float(probs_np[item].max()) for item in selected.tolist()],
            }
        return {
            "contract": "final_output_candidate_manifest_v1",
            "estimator_mode": "final_output_ranker",
            "canonical_stage2": True,
            "candidate_source": "stage1_final_outputs",
            "budget_scope": "test_mask",
            "budgets": [float(item) for item in budgets],
            "ranking_key": "risk_score_desc",
            "ranked_node_ids": [int(item) for item in ranked_idx.tolist()],
            "candidate_sets": candidate_sets,
            "input_boundary": risk_manifest.get("calibration_metadata", {}).get("input_boundary", {}),
            "screening_only": True,
            "diagnosis_or_action": False,
        }

    def _build_final_output_ranker_bundle(self, final_outputs):
        budgets = self._final_output_ranker_budgets()
        estimator = build_estimator("final_output_ranker")
        labels = final_outputs.get("labels")
        if labels is None:
            labels = torch.tensor(self.labels, dtype=torch.long)
        estimator.fit(
            logits=final_outputs.get("logits"),
            probs=final_outputs.get("prob"),
            labels=labels,
            train_idx=self.data["train_idx"],
            val_idx=self.data["valid_idx"],
        )
        risk_manifest = estimator.build_manifest(
            logits=final_outputs.get("logits"),
            probs=final_outputs.get("prob"),
            labels=labels,
            val_idx=self.data["valid_idx"],
            test_idx=self.data["test_idx"],
            budgets=budgets,
        )
        risk_manifest["calibration_metadata"] = {
            **risk_manifest.get("calibration_metadata", {}),
            "source": "final_output_ranker",
            "canonical_stage2": True,
            "canonical_budgets": [float(item) for item in budgets],
        }
        ranker_manifest = {
            "contract": "final_output_ranker_v1",
            "estimator_mode": "final_output_ranker",
            "canonical_stage2": True,
            "target": "Stage 1 base detector residual failure screening",
            "method_positioning": "GETS-inspired calibrated residual-risk ranking, not a GETS router or Graph-MoE router",
            "training_label": "z_i = 1[stage1_final_prediction != y_i]",
            "screening_only": True,
            "diagnosis_or_action": False,
            "budgets": [float(item) for item in budgets],
            "input_boundary": risk_manifest["calibration_metadata"].get("input_boundary", {}),
            "calibration_metadata": risk_manifest["calibration_metadata"],
        }
        candidate_manifest = self._build_final_output_candidate_manifest(risk_manifest, final_outputs, budgets)
        return (
            self._bundle_from_final_output_ranker_manifest(risk_manifest),
            ranker_manifest,
            candidate_manifest,
            torch.tensor(risk_manifest["risk_score"], dtype=torch.float32),
        )

    def _build_dual_posterior_candidate_manifest(self, risk_manifest, final_outputs, lm_probs, budgets):
        risk_score = np.asarray(risk_manifest["risk_score"], dtype=np.float32)
        probs = final_outputs.get("prob")
        if probs is None:
            probs = torch.softmax(final_outputs["logits"].float(), dim=1)
        probs_np = probs.detach().cpu().numpy()
        lm_probs_np = lm_probs.detach().cpu().numpy()
        pred_np = final_outputs.get("pred")
        if pred_np is None:
            pred_np = probs.argmax(dim=1)
        pred_np = pred_np.detach().cpu().numpy()
        lm_pred_np = lm_probs.argmax(dim=1).detach().cpu().numpy()
        candidate_idx = np.flatnonzero(self.test_mask)
        ranked_idx = candidate_idx[np.argsort(-risk_score[candidate_idx])] if candidate_idx.size else np.asarray([], dtype=np.int64)
        candidate_sets = {}
        for budget in budgets:
            budget_key = f"top_{int(round(float(budget) * 100))}pct"
            k = min(max(int(candidate_idx.size * float(budget)), 1), candidate_idx.size) if candidate_idx.size else 0
            selected = ranked_idx[:k]
            candidate_sets[budget_key] = {
                "budget": float(budget),
                "selected_count": int(selected.size),
                "node_ids": [int(item) for item in selected.tolist()],
                "risk_scores": [float(risk_score[item]) for item in selected.tolist()],
                "final_predictions": [int(pred_np[item]) for item in selected.tolist()],
                "final_confidences": [float(probs_np[item].max()) for item in selected.tolist()],
                "lm_only_predictions": [int(lm_pred_np[item]) for item in selected.tolist()],
                "lm_only_confidences": [float(lm_probs_np[item].max()) for item in selected.tolist()],
            }
        return {
            "contract": "dual_posterior_candidate_manifest_v1",
            "estimator_mode": "dual_posterior_ranker",
            "canonical_stage2": False,
            "candidate_source": "stage1_final_outputs_plus_lm_only_posterior",
            "budget_scope": "test_mask",
            "budgets": [float(item) for item in budgets],
            "ranking_key": "risk_score_desc",
            "ranked_node_ids": [int(item) for item in ranked_idx.tolist()],
            "candidate_sets": candidate_sets,
            "input_boundary": risk_manifest.get("calibration_metadata", {}).get("input_boundary", {}),
            "screening_only": True,
            "diagnosis_or_action": False,
        }

    def _build_dual_posterior_ranker_bundle(self, final_outputs):
        budgets = self._final_output_ranker_budgets()
        lm_bundle = self._build_or_load_lm_only_head()
        lm_outputs = lm_bundle["outputs"]
        lm_probs = lm_outputs.get("prob_cal", lm_outputs.get("prob"))
        if lm_probs is None:
            raise MissingFrozenArtifactError("dual_posterior_ranker requires frozen/lm_only_head outputs with prob_cal or prob.")
        lm_probs = lm_probs.detach().cpu().float()
        estimator = build_estimator("dual_posterior_ranker")
        labels = final_outputs.get("labels")
        if labels is None:
            labels = torch.tensor(self.labels, dtype=torch.long)
        estimator.fit(
            logits=final_outputs.get("logits"),
            probs=final_outputs.get("prob"),
            labels=labels,
            train_idx=self.data["train_idx"],
            val_idx=self.data["valid_idx"],
            lm_probs=lm_probs,
        )
        risk_manifest = estimator.build_manifest(
            logits=final_outputs.get("logits"),
            probs=final_outputs.get("prob"),
            labels=labels,
            val_idx=self.data["valid_idx"],
            test_idx=self.data["test_idx"],
            lm_probs=lm_probs,
            budgets=budgets,
        )
        risk_manifest["calibration_metadata"] = {
            **risk_manifest.get("calibration_metadata", {}),
            "source": "dual_posterior_ranker",
            "lm_only_head_manifest": str(self._lm_only_head_dir() / "manifest.json"),
            "canonical_stage2": False,
            "canonical_reference": "final_output_ranker",
            "canonical_budgets": [float(item) for item in budgets],
        }
        ranker_manifest = {
            "contract": "dual_posterior_ranker_v1",
            "estimator_mode": "dual_posterior_ranker",
            "canonical_stage2": False,
            "ablation_only": True,
            "target": "Stage 1 base detector residual failure screening",
            "training_label": "z_i = 1[stage1_final_prediction != y_i]",
            "screening_only": True,
            "diagnosis_or_action": False,
            "budgets": [float(item) for item in budgets],
            "input_boundary": risk_manifest["calibration_metadata"].get("input_boundary", {}),
            "calibration_metadata": risk_manifest["calibration_metadata"],
            "lm_only_head_manifest": str(self._lm_only_head_dir() / "manifest.json"),
            "lm_posterior_source": "frozen_lm_only_head_outputs_prob_cal_or_prob",
        }
        candidate_manifest = self._build_dual_posterior_candidate_manifest(risk_manifest, final_outputs, lm_probs, budgets)
        return (
            self._bundle_from_final_output_ranker_manifest(risk_manifest, mode="dual_posterior_ranker"),
            ranker_manifest,
            candidate_manifest,
            torch.tensor(risk_manifest["risk_score"], dtype=torch.float32),
            lm_bundle,
        )

    def _stage2_probabilities(self, final_outputs):
        probs = final_outputs.get("prob")
        if probs is None:
            probs = torch.softmax(final_outputs["logits"].float(), dim=1)
        return probs.detach().cpu().float()

    def _stage2_predictions(self, final_outputs):
        pred = final_outputs.get("pred")
        if pred is None:
            pred = self._stage2_probabilities(final_outputs).argmax(dim=1)
        return pred.detach().cpu().long().numpy()

    def _stage2_logits(self, final_outputs):
        logits = final_outputs.get("logits")
        return logits.detach().cpu().float() if logits is not None else torch.log(self._stage2_probabilities(final_outputs).clamp_min(1e-8))

    @staticmethod
    def _minmax_score(values):
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return values
        lo = float(np.min(values))
        hi = float(np.max(values))
        if hi - lo <= 1e-12:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - lo) / (hi - lo)).astype(np.float32)

    def _logit_risk_calibrator_score(self, final_outputs, y_wrong):
        probs = self._stage2_probabilities(final_outputs).numpy()
        logits = self._stage2_logits(final_outputs).numpy()
        valid_idx = _idx_numpy(self.data["valid_idx"])
        if valid_idx.size < 3 or np.unique(y_wrong[valid_idx]).size < 2:
            return None, {"status": "not_available", "reason": "validation split lacks both residual-error classes"}
        top2_prob = np.sort(probs, axis=1)[:, -2:]
        top2_logits = np.sort(logits, axis=1)[:, -2:]
        entropy = (-probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1) / max(math.log(probs.shape[1]), 1e-8)
        features = np.column_stack(
            [
                1.0 - probs.max(axis=1),
                entropy,
                1.0 - (top2_prob[:, 1] - top2_prob[:, 0]),
                top2_logits[:, 1] - top2_logits[:, 0],
                logits.max(axis=1),
            ]
        ).astype(np.float32)
        scaler = StandardScaler()
        x_val = scaler.fit_transform(features[valid_idx])
        x_all = scaler.transform(features)
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(x_val, y_wrong[valid_idx])
        return model.predict_proba(x_all)[:, 1].astype(np.float32), {
            "status": "available",
            "fit_scope": "validation_split_residual_labels_only",
            "feature_family": ["msp", "entropy", "margin_risk", "logit_gap", "max_logit"],
        }

    def _gets_lite_score(self, final_outputs):
        probs = self._stage2_probabilities(final_outputs).numpy()
        pred = probs.argmax(axis=1)
        msp = 1.0 - probs.max(axis=1)
        entropy = (-probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1) / max(math.log(probs.shape[1]), 1e-8)
        edge_index = _idx_numpy(self.data["edge_index"])
        num_nodes = int(len(self.labels))
        if edge_index.size == 0:
            return (0.65 * msp + 0.35 * entropy).astype(np.float32), {"status": "available_no_edges"}
        src, dst = edge_index
        degree = np.zeros(num_nodes, dtype=np.float32)
        same_pred = np.zeros(num_nodes, dtype=np.float32)
        np.add.at(degree, dst, 1.0)
        np.add.at(same_pred, dst, (pred[src] == pred[dst]).astype(np.float32))
        local_consistency = same_pred / np.clip(degree, a_min=1.0, a_max=None)
        sparse_score = 1.0 - self._minmax_score(np.log1p(degree))
        score = 0.45 * msp + 0.25 * entropy + 0.20 * (1.0 - local_consistency) + 0.10 * sparse_score
        return np.clip(score, 0.0, 1.0).astype(np.float32), {
            "status": "available",
            "feature_family": ["msp", "entropy", "local_prediction_consistency", "degree"],
            "positioning": "GETS-lite graph-aware calibration/ranking ablation; not an action router",
        }

    def _stage2_baseline_suite(self, final_outputs, y_wrong, lm_probs=None):
        probs = self._stage2_probabilities(final_outputs).numpy()
        logits = self._stage2_logits(final_outputs)
        top2 = np.sort(probs, axis=1)[:, -2:]
        entropy = (-probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1) / max(math.log(probs.shape[1]), 1e-8)
        scores = {
            "msp": {
                "score": (1.0 - probs.max(axis=1)).astype(np.float32),
                "metadata": {"status": "available", "baseline": "MSP risk = 1 - max p_i"},
            },
            "entropy": {
                "score": np.clip(entropy, 0.0, 1.0).astype(np.float32),
                "metadata": {"status": "available", "baseline": "normalized predictive entropy"},
            },
            "margin": {
                "score": np.clip(1.0 - (top2[:, 1] - top2[:, 0]), 0.0, 1.0).astype(np.float32),
                "metadata": {"status": "available", "baseline": "1 - top1/top2 probability margin"},
            },
            "seeded_random": {
                "score": np.random.default_rng(self.seed).random(len(self.labels), dtype=np.float32),
                "metadata": {"status": "available", "baseline": "seeded random ranking"},
            },
        }
        valid_idx = _idx_numpy(self.data["valid_idx"])
        if valid_idx.size:
            temperature = fit_temperature_scaling(logits[valid_idx], torch.tensor(self.labels[valid_idx], dtype=torch.long))
            prob_ts = torch.softmax(logits / float(temperature), dim=1).numpy()
            scores["msp_ts"] = {
                "score": (1.0 - prob_ts.max(axis=1)).astype(np.float32),
                "metadata": {"status": "available", "temperature": float(temperature), "fit_scope": "validation_split_labels_only"},
            }
        calibrator_score, calibrator_meta = self._logit_risk_calibrator_score(final_outputs, y_wrong)
        if calibrator_score is not None:
            scores["logit_risk_calibrator"] = {"score": calibrator_score, "metadata": calibrator_meta}
        else:
            scores["logit_risk_calibrator"] = {"score": None, "metadata": calibrator_meta}
        gets_score, gets_meta = self._gets_lite_score(final_outputs)
        scores["gets_lite_graph_aware"] = {"score": gets_score, "metadata": gets_meta}
        if lm_probs is not None:
            lm_probs_np = lm_probs.detach().cpu().float().numpy()
            lm_conf = lm_probs_np.max(axis=1)
            gnn_conf = probs.max(axis=1)
            lm_pred = lm_probs_np.argmax(axis=1)
            gnn_pred = probs.argmax(axis=1)
            disagreement = (lm_pred != gnn_pred).astype(np.float32)
            score = 0.50 * (1.0 - gnn_conf) + 0.20 * (1.0 - lm_conf) + 0.20 * disagreement + 0.10 * np.abs(lm_conf - gnn_conf)
            scores["dual_posterior_ranker"] = {
                "score": np.clip(score, 0.0, 1.0).astype(np.float32),
                "metadata": {"status": "available", "baseline": "LM-only posterior plus final GNN posterior disagreement"},
            }
        else:
            scores["dual_posterior_ranker"] = {
                "score": None,
                "metadata": {"status": "not_available", "reason": "LM-only posterior artifact not loaded for this Stage 2 run"},
            }
        return scores

    def _stage2_neighbor_label_masks(self, eval_mask):
        edge_index = _idx_numpy(self.data["edge_index"])
        if edge_index.size == 0:
            return None, None
        labels = np.asarray(self.labels, dtype=np.int64)
        num_nodes = int(len(labels))
        src, dst = edge_index
        degree = np.zeros(num_nodes, dtype=np.float32)
        human_neighbors = np.zeros(num_nodes, dtype=np.float32)
        same_label = np.zeros(num_nodes, dtype=np.float32)
        np.add.at(degree, dst, 1.0)
        np.add.at(human_neighbors, dst, (labels[src] == 0).astype(np.float32))
        np.add.at(same_label, dst, (labels[src] == labels[dst]).astype(np.float32))
        valid = degree > 0
        if not (valid & eval_mask).any():
            return None, None
        human_ratio = human_neighbors / np.clip(degree, a_min=1.0, a_max=None)
        homophily = same_label / np.clip(degree, a_min=1.0, a_max=None)
        high_human_cut = float(np.quantile(human_ratio[valid & eval_mask], 0.75))
        low_homophily_cut = float(np.quantile(homophily[valid & eval_mask], 0.25))
        return (valid & (human_ratio >= high_human_cut)), (valid & (homophily <= low_homophily_cut))

    def _stage2_hard_node_subgroup_report(self, final_outputs, y_wrong, risk_score, eval_mask, budgets, lm_probs=None):
        eval_mask = np.asarray(eval_mask, dtype=bool)
        probs = self._stage2_probabilities(final_outputs).numpy()
        pred = probs.argmax(axis=1)
        edge_index = _idx_numpy(self.data["edge_index"])
        num_nodes = int(len(self.labels))
        hcw_mask = (probs.max(axis=1) >= 0.90) & (y_wrong == 1)
        low_degree_mask = None
        disagreement_mask = None
        if edge_index.size:
            src, dst = edge_index
            degree = np.zeros(num_nodes, dtype=np.float32)
            np.add.at(degree, src, 1.0)
            np.add.at(degree, dst, 1.0)
            cutoff = float(np.quantile(degree[eval_mask], getattr(self.args, "sparse_degree_quantile", 0.25))) if eval_mask.any() else 0.0
            low_degree_mask = degree <= cutoff
        if lm_probs is not None:
            disagreement_mask = lm_probs.detach().cpu().float().argmax(dim=1).numpy() != pred
        high_human_mask, low_homophily_mask = self._stage2_neighbor_label_masks(eval_mask)

        candidate_idx = np.flatnonzero(eval_mask)
        ranked_idx = candidate_idx[np.argsort(-np.asarray(risk_score, dtype=np.float32)[candidate_idx])] if candidate_idx.size else np.asarray([], dtype=np.int64)

        def _group(name, mask, metric_prefix="ErrRecall", analysis_only=False):
            if mask is None:
                return {"status": "not_available", "reason": "required feature is unavailable"}
            mask = np.asarray(mask, dtype=bool)
            group_errors = mask & eval_mask & (y_wrong == 1)
            total = int(group_errors.sum())
            rows = []
            for budget in budgets:
                key = int(round(float(budget) * 100))
                k = min(max(int(candidate_idx.size * float(budget)), 1), candidate_idx.size) if candidate_idx.size else 0
                selected = ranked_idx[:k]
                hits = int(group_errors[selected].sum()) if selected.size else 0
                rows.append(
                    {
                        "budget": float(budget),
                        f"{metric_prefix}@{key}": float(hits / max(total, 1)),
                        "selected_group_error_count": hits,
                        "group_error_count": total,
                    }
                )
            return {
                "status": "available",
                "analysis_only_oracle_labels": bool(analysis_only),
                "group_node_count": int((mask & eval_mask).sum()),
                "group_error_count": total,
                "budget_rows": rows,
            }

        report = {
            "contract": "stage2_hard_node_subgroup_report_v1",
            "budget_scope": "test_mask",
            "high_confidence_wrong_nodes": _group("high_confidence_wrong_nodes", hcw_mask, metric_prefix="HCWRecall"),
            "low_degree_sparse_profile": _group("low_degree_sparse_profile", low_degree_mask),
            "lm_gnn_disagreement_nodes": _group("lm_gnn_disagreement_nodes", disagreement_mask),
            "high_human_neighbor_ratio_camouflage": _group("high_human_neighbor_ratio_camouflage", high_human_mask, analysis_only=True),
            "local_heterophily_low_homophily": _group("local_heterophily_low_homophily", low_homophily_mask, analysis_only=True),
            "claim_cautions": [],
        }
        hcw40 = None
        for row in report["high_confidence_wrong_nodes"].get("budget_rows", []):
            if "HCWRecall@40" in row:
                hcw40 = row["HCWRecall@40"]
        if hcw40 is not None and hcw40 < 0.20:
            report["claim_cautions"].append("HCWRecall@40 < 0.20; do not claim high-confidence hard-error identification.")
        overall40 = residual_risk_metrics(y_wrong[eval_mask], np.asarray(risk_score)[eval_mask], budgets=budgets).get("ErrRecall@40")
        if overall40 is not None:
            for name, group in report.items():
                if not isinstance(group, dict) or group.get("status") != "available":
                    continue
                for row in group.get("budget_rows", []):
                    group40 = row.get("ErrRecall@40")
                    if group40 is not None and float(overall40) - float(group40) > 0.15:
                        report["claim_cautions"].append(f"{name} ErrRecall@40 trails overall by >15 points; downgrade subgroup claim.")
        return report

    def _build_stage2_paper_report(self, selector_name, risk_bundle, final_outputs, budgets, lm_probs=None):
        budgets = parse_budget_list(budgets, default=RESIDUAL_RISK_PAPER_BUDGETS)
        risk_score = np.asarray(risk_bundle["risk_manifest"]["risk_score"], dtype=np.float32)
        base_pred = self._stage2_predictions(final_outputs)
        y_wrong = (base_pred != self.labels).astype(np.int32)
        test_idx = np.flatnonzero(self.test_mask)
        selector_metrics = residual_risk_metrics(y_wrong[test_idx], risk_score[test_idx], budgets=budgets) if test_idx.size else {}
        selector_curve = residual_risk_budget_curve(y_wrong[test_idx], risk_score[test_idx], budgets=budgets) if test_idx.size else []
        baseline_scores = self._stage2_baseline_suite(final_outputs, y_wrong, lm_probs=lm_probs)
        baseline_reports = {
            "random_expected": {
                "metadata": {"status": "available", "baseline": "expected random top-B selection"},
                "budget_curve": random_expected_residual_risk_budget_curve(y_wrong[test_idx], budgets=budgets),
                "metrics": {"base_error_rate": float(y_wrong[test_idx].mean()) if test_idx.size else 0.0},
            }
        }
        baseline_metrics = {}
        for name, payload in baseline_scores.items():
            score = payload.get("score")
            if score is None:
                baseline_reports[name] = {"metadata": payload.get("metadata", {"status": "not_available"})}
                continue
            metrics = residual_risk_metrics(y_wrong[test_idx], np.asarray(score)[test_idx], budgets=budgets) if test_idx.size else {}
            baseline_metrics[name] = metrics
            baseline_reports[name] = {
                "metadata": payload.get("metadata", {}),
                "metrics": metrics,
                "budget_curve": residual_risk_budget_curve(y_wrong[test_idx], np.asarray(score)[test_idx], budgets=budgets) if test_idx.size else [],
            }

        aurc_relative_reduction = {}
        selector_aurc = float(selector_metrics.get("aurc", float("nan"))) if selector_metrics else float("nan")
        for name, metrics in baseline_metrics.items():
            baseline_aurc = float(metrics.get("aurc", float("nan")))
            if math.isfinite(selector_aurc) and math.isfinite(baseline_aurc) and baseline_aurc > 0:
                aurc_relative_reduction[name] = float((baseline_aurc - selector_aurc) / baseline_aurc)

        bootstrap = {}
        for baseline_name in ("msp", "entropy"):
            payload = baseline_scores.get(baseline_name, {})
            score = payload.get("score")
            if score is None or not test_idx.size:
                continue
            bootstrap[baseline_name] = paired_bootstrap_residual_risk_delta(
                y_wrong[test_idx],
                risk_score[test_idx],
                np.asarray(score)[test_idx],
                budgets=budgets,
                n_samples=getattr(self.args, "bootstrap_samples", 512),
                seed=self.seed,
            )

        calibration_reference = baseline_metrics.get("msp_ts") or baseline_metrics.get("msp")
        gate = stage2_acceptance_gate(selector_metrics, baselines=baseline_metrics, calibration_reference=calibration_reference)
        subgroup_report = self._stage2_hard_node_subgroup_report(final_outputs, y_wrong, risk_score, self.test_mask, budgets, lm_probs=lm_probs)
        report = {
            "contract": "stage2_calibrated_residual_risk_report_v1",
            "paper_name": "Calibrated Residual-Risk Hard-Node Selector",
            "selector_name": selector_name,
            "task": "P(Stage1 prediction is wrong | Stage1 outputs/features)",
            "screening_only": True,
            "diagnosis_or_action": False,
            "budgets": [float(item) for item in budgets],
            "main_table_budgets": [0.20, 0.30, 0.40],
            "base_error_rate": float(y_wrong[test_idx].mean()) if test_idx.size else 0.0,
            "test_count": int(test_idx.size),
            "test_error_count": int(y_wrong[test_idx].sum()) if test_idx.size else 0,
            "selector_metrics": selector_metrics,
            "selector_budget_curve": selector_curve,
            "baseline_suite": baseline_reports,
            "aurc_relative_reduction_vs_baselines": aurc_relative_reduction,
            "paired_bootstrap_vs_baselines": bootstrap,
            "acceptance_gate": gate,
            "claim_status": gate["claim_status"],
            "hard_node_subgroups": subgroup_report,
        }
        reliability_rows = selector_metrics.get("reliability_diagram", {}).get("fixed_bins", []) if selector_metrics else []
        return {
            "report": report,
            "acceptance_gate": gate,
            "reliability_rows": reliability_rows,
            "subgroup_report": subgroup_report,
        }

    def _write_stage2_paper_artifacts(self, stage_dir, paper_bundle):
        write_json(stage_dir / "stage2_paper_report.json", paper_bundle["report"])
        write_json(stage_dir / "stage2_acceptance_gate.json", paper_bundle["acceptance_gate"])
        write_json(stage_dir / "stage2_hard_node_subgroups.json", paper_bundle["subgroup_report"])
        write_csv_rows(
            stage_dir / "stage2_reliability_diagram.csv",
            ["bin", "lower", "upper", "count", "fraction", "mean_predicted_risk", "empirical_error_rate", "empirical_accuracy", "calibration_gap"],
            paper_bundle["reliability_rows"],
        )

    def _build_p0_risk_bundle(self):
        context = self.ensure_backbone_context()
        gnn_outputs = context["gnn_outputs"]
        estimator = build_estimator("msp_ts")
        estimator.fit(
            logits=gnn_outputs["logits"],
            probs=gnn_outputs["prob"],
            labels=gnn_outputs["labels"],
            train_idx=self.data["train_idx"],
            val_idx=self.data["valid_idx"],
            edge_index=self.data["edge_index"],
            edge_type=self.data["edge_type"],
            node_repr=gnn_outputs["node_repr"],
        )
        risk_manifest = estimator.build_manifest(
            logits=gnn_outputs["logits"],
            probs=gnn_outputs["prob"],
            labels=gnn_outputs["labels"],
            val_idx=self.data["valid_idx"],
            test_idx=self.data["test_idx"],
            edge_index=self.data["edge_index"],
            edge_type=self.data["edge_type"],
            node_repr=gnn_outputs["node_repr"],
        )
        risk_manifest["calibration_metadata"] = {
            **risk_manifest.get("calibration_metadata", {}),
            "source": "p0_msp_temperature_from_frozen_g0",
            "training_scope": "validation_only_temperature_scaling",
        }
        return self._bundle_from_risk_manifest("msp_ts", risk_manifest)

    def _build_gats_risk_bundle(self, gats_bundle):
        outputs = gats_bundle["outputs"]
        if "q_graph" not in outputs:
            raise MissingFrozenArtifactError("Faithful GATS outputs must include q_graph.")
        risk_score = 1.0 - outputs["q_graph"].detach().cpu().numpy().astype(np.float32)
        risk_manifest = self._risk_manifest_from_score(
            risk_score,
            calibration_metadata={
                "source": "faithful_gats_anchor",
                "training_scope": gats_bundle["manifest"].get("training_scope"),
                "gate_contract": gats_bundle["manifest"].get("contract"),
            },
        )
        return self._bundle_from_risk_manifest("gats_faithful", risk_manifest)

    def _pick_best_risk_bundle(self, risk_bundles):
        def _score(item):
            mode, bundle = item
            metrics = bundle["risk_manifest"].get("validation_metrics", {})
            return (float(metrics.get("utility_at_15", 0.0)), mode)

        return max(risk_bundles.items(), key=_score)

    def _build_gain_explain(self, risk_bundle, base_pred, new_pred, eval_mask, triggered_mask):
        risk_score = np.asarray(risk_bundle["risk_manifest"]["risk_score"], dtype=np.float32)
        empirical_gain = ((np.asarray(new_pred) == self.labels).astype(np.int32) - (np.asarray(base_pred) == self.labels).astype(np.int32)).astype(np.float32)
        eval_mask = np.asarray(eval_mask, dtype=bool)
        triggered_mask = np.asarray(triggered_mask, dtype=bool)
        if eval_mask.sum() > 1:
            predicted = risk_score[eval_mask]
            empirical = empirical_gain[eval_mask]
            corr = float(np.corrcoef(predicted, empirical)[0, 1]) if np.std(predicted) > 0 and np.std(empirical) > 0 else 0.0
        else:
            corr = 0.0
        sparse_mask, repair_focus, high_risk = self._policy_masks(risk_bundle)
        return {
            "predicted_gain_vs_empirical_gain": {
                "correlation": corr,
                "eval_count": int(eval_mask.sum()),
            },
            "triggered_node_composition": {
                "triggered_count": int((triggered_mask & self.test_mask).sum()),
                "sparse": int((triggered_mask & sparse_mask & self.test_mask).sum()),
                "repair_focus": int((triggered_mask & repair_focus & self.test_mask).sum()),
                "high_risk": int((triggered_mask & high_risk & self.test_mask).sum()),
            },
        }

    def _build_metrics_payload(
        self,
        risk_bundle,
        base_pred,
        new_pred,
        eval_mask,
        triggered_mask,
        gain_cost_report=None,
        extra_metrics=None,
        budgets=None,
    ):
        base_pred = np.asarray(base_pred)
        new_pred = np.asarray(new_pred)
        eval_mask = np.asarray(eval_mask, dtype=bool)
        triggered_mask = np.asarray(triggered_mask, dtype=bool)
        risk_manifest = risk_bundle["risk_manifest"]
        budgets = parse_budget_list(budgets, default=(0.05, 0.10, 0.15, 0.20)) if budgets is not None else (0.05, 0.10, 0.15, 0.20)
        budget_curve = _compute_budget_curve(
            labels=self.labels,
            base_pred=base_pred,
            new_pred=new_pred,
            risk_score=risk_manifest["risk_score"],
            eval_mask=eval_mask,
            hcw_mask=risk_manifest.get("hcw_mask"),
            budgets=budgets,
        )
        targeted = _subset_scores(self.labels, new_pred, eval_mask)
        overall = _score_all(self.labels[self.test_mask], new_pred[self.test_mask])
        delta = _delta_table(base_pred, new_pred, self.labels, eval_mask)
        easy_mask = self.test_mask & ~triggered_mask
        base_correct = base_pred == self.labels
        new_correct = new_pred == self.labels
        easy_preservation = float((base_correct & new_correct & easy_mask).sum() / max(int((base_correct & easy_mask).sum()), 1))
        budget_lookup = {int(round(row["budget"] * 100)): row for row in budget_curve}
        hcw_budget = 15 if 15 in budget_lookup else (int(round(float(budgets[0]) * 100)) if budgets else 0)
        metrics = {
            "accuracy": overall["accuracy"],
            "macro_f1": overall["macro_f1"],
            "bot_f1": overall["bot_f1"],
            "screened_set_accuracy": targeted["accuracy"],
            "fix": delta["fix"],
            "broke": delta["broke"],
            "net": delta["net"],
            "HCW_capture": budget_lookup.get(hcw_budget, budget_curve[-1] if budget_curve else {"hcw_capture": 0.0})["hcw_capture"] if budget_curve else 0.0,
            "easy_node_preservation": easy_preservation,
            "touched_node_ratio": float((triggered_mask & self.test_mask).sum() / max(int(self.test_mask.sum()), 1)),
            "paired_bootstrap": _paired_bootstrap_delta(
                labels=self.labels,
                base_pred=base_pred,
                new_pred=new_pred,
                eval_mask=eval_mask,
                n_samples=getattr(self.args, "bootstrap_samples", 512),
                seed=self.seed,
            ),
        }
        for budget in budgets:
            budget_key = int(round(float(budget) * 100))
            metrics[f"utility_at_{budget_key}"] = budget_lookup.get(budget_key, {"utility": 0.0})["utility"]
            metrics[f"utility@{budget_key}"] = metrics[f"utility_at_{budget_key}"]
        metrics["fix/broke/net"] = {
            "fix": metrics["fix"],
            "broke": metrics["broke"],
            "net": metrics["net"],
        }
        metrics.update(gain_cost_report or _empty_gain_cost_report())
        if extra_metrics:
            metrics.update(extra_metrics)
        return metrics, budget_curve

    def _write_stage_outputs(self, stage_dir, risk_bundle, metrics_payload, budget_curve, notes_lines, extra_artifacts=None, gain_cost_report=None, gain_explain=None, subgroup_extra=None):
        write_json(stage_dir / "metrics.json", metrics_payload)
        write_csv_rows(
            stage_dir / "budget_curve.csv",
            ["budget", "triggered", "touched", "fix", "broke", "net", "utility", "screened_set_accuracy", "hcw_capture"],
            budget_curve,
        )
        write_json(stage_dir / "subgroup_report.json", self._stage_subgroup_report(risk_bundle, subgroup_extra))
        write_text(stage_dir / "notes.md", "\n".join(notes_lines).strip() + "\n")
        if gain_cost_report is not None:
            write_json(stage_dir / "gain_cost_report.json", gain_cost_report)
        if gain_explain is not None:
            write_json(stage_dir / "gain_explain.json", gain_explain)
        if extra_artifacts:
            save_stage_artifacts(stage_dir, extra_artifacts)

    def _ensure_diagnostic_baseline_artifacts(self, risk_bundle):
        context = self.ensure_backbone_context()
        runtime_dir = context["runtime_dir"]
        diag_dir = runtime_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        gnn_outputs = context["gnn_outputs"]
        lm_outputs = context["lm_outputs"]
        probs = gnn_outputs["prob"]
        entropy = (-probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).cpu()
        write_json(diag_dir / "diagnostics_manifest.json", {"generated_by": "StageRunner", "seed": self.seed})
        torch.save(gnn_outputs["logits"], diag_dir / "gnn_logits_all.pt")
        torch.save(gnn_outputs["prob"], diag_dir / "gnn_probs_all.pt")
        torch.save(gnn_outputs["pred"], diag_dir / "gnn_pred_all.pt")
        torch.save(entropy, diag_dir / "gnn_entropy_all.pt")
        torch.save(lm_outputs["pred"][self.data["test_idx"]], diag_dir / "lm_pred_test.pt")

        edge_index = _idx_numpy(self.data["edge_index"])
        src, dst = edge_index
        entropy_np = entropy.numpy()
        neighbor_entropy_sum = np.zeros(len(self.labels), dtype=np.float32)
        neighbor_count = np.zeros(len(self.labels), dtype=np.float32)
        np.add.at(neighbor_entropy_sum, dst, entropy_np[src])
        np.add.at(neighbor_count, dst, 1.0)
        neighbor_entropy_mean = neighbor_entropy_sum / np.clip(neighbor_count, a_min=1.0, a_max=None)

        structural = risk_bundle["structural_manifest"]["features"]
        sparse_mask, repair_focus, _ = self._policy_masks(risk_bundle)

        def _rows(node_idx):
            rows = []
            for nid in _idx_numpy(node_idx):
                rows.append(
                    {
                        "node_id": int(nid),
                        "total_degree": float(structural["total_degree"][nid]),
                        "neighbor_entropy_mean": float(neighbor_entropy_mean[nid]),
                        "sparse_evidence": int(sparse_mask[nid]),
                        "prop_corruption_flag": int(repair_focus[nid]),
                        "graph_missing": int(structural["total_degree"][nid] == 0),
                    }
                )
            return rows

        write_csv_rows(diag_dir / "regime_table_val.csv", ["node_id", "total_degree", "neighbor_entropy_mean", "sparse_evidence", "prop_corruption_flag", "graph_missing"], _rows(self.data["valid_idx"]))
        write_csv_rows(diag_dir / "regime_table_test.csv", ["node_id", "total_degree", "neighbor_entropy_mean", "sparse_evidence", "prop_corruption_flag", "graph_missing"], _rows(self.data["test_idx"]))
        return runtime_dir

    def _strict_recompute_forbidden(self, component):
        raise MissingFrozenArtifactError(
            f"Strict frozen StageRunner forbids recomputing '{component}'. "
            "Load the required frozen dependency artifact instead."
        )

    def _build_risk_suite(self, estimator_modes=None):
        self._strict_recompute_forbidden("_build_risk_suite")

    def _semantic_results(self, risk_bundle, semantic_modes=None, focal_mode="sparse"):
        self._strict_recompute_forbidden("_semantic_results")

    def _lift_test_only_result(self, result_dict, fallback_pred):
        self._strict_recompute_forbidden("_lift_test_only_result")

    def _repair_results(self, risk_bundle, repair_modes=None):
        self._strict_recompute_forbidden("_repair_results")

    def _selector_results(self, risk_bundle, semantic_results, repair_results):
        self._strict_recompute_forbidden("_selector_results")

    def _semantic_source_results(self, risk_bundle, semantic_results):
        self._strict_recompute_forbidden("_semantic_source_results")

    def _positioning_results(self, risk_bundle, semantic_results, repair_results):
        self._strict_recompute_forbidden("_positioning_results")

    def _load_survivor_manifest(self, default_p_mode, default_sem_mode, default_rep_mode):
        est = self._read_stage_json("estimator_matrix", "survivor_manifest") or {"best_p_proxy": default_p_mode}
        sem = self._read_stage_json("semantic_matrix", "survivor_manifest") or {"best_semantic_operator": default_sem_mode}
        sem_source = self._read_stage_json("semantic_source_matrix", "survivor_manifest") or {"best_semantic_source": "S0_roberta_local_source"}
        rep = self._read_stage_json("repair_matrix", "survivor_manifest") or {"best_repair_single_action": default_rep_mode}
        sel = self._read_stage_json("selector_matrix", "survivor_manifest") or {"best_selector": "true three-action"}
        return {
            "p_mode": est.get("best_p_proxy", default_p_mode),
            "semantic_mode": sem.get("best_semantic_operator", default_sem_mode),
            "semantic_source": sem_source.get("best_semantic_source", "S0_roberta_local_source"),
            "repair_mode": rep.get("best_repair_single_action", default_rep_mode),
            "selector_name": sel.get("best_selector", "true three-action"),
        }

    def _run_vertical_minimal(self, stage_dir, base_bundle):
        best_mode = "msp_ts"
        risk_bundle = self._build_p0_risk_bundle()
        context = self.ensure_backbone_context()
        base_pred = context["gnn_outputs"]["pred"].numpy()
        eval_mask = self.test_mask
        triggered_mask = self._validation_frozen_high_mask(risk_bundle) & self.test_mask
        metrics_payload, budget_curve = self._build_metrics_payload(risk_bundle, base_pred, base_pred, eval_mask, triggered_mask)
        self._write_stage_outputs(
            stage_dir,
            risk_bundle,
            metrics_payload,
            budget_curve,
            [
                "claim tested: harness smoke and canonical risk-manifest export",
                "keep / kill: keep if unified formal-stage artifacts are produced",
                "enter next main table: yes if smoke succeeds",
            ],
            extra_artifacts={
                **base_bundle,
                "dependency_manifest": {
                    "stage_contract": "strict_frozen_dependency_load_v1",
                    "reads": ["frozen/g0/manifest.json", "frozen/g0/outputs.pt"],
                    "recomputed_upstream": False,
                },
                "risk_manifest": risk_bundle["risk_manifest"],
                "structural_manifest": risk_bundle["structural_manifest"],
                "subgroup_manifest_operational": risk_bundle["subgroup_manifest_operational"],
                "subgroup_manifest_analysis": risk_bundle["subgroup_manifest_analysis"],
                "all_node_outputs.pt": context["gnn_outputs"],
                "survivor_manifest": {"best_p_proxy": best_mode},
            },
            subgroup_extra={"best_mode": best_mode},
        )
        return {"stage": self.args.stage, "best_mode": best_mode}

    def _run_final_output_ranker_matrix(self, stage_dir, base_bundle):
        dependency = self._load_final_output_dependency()
        final_outputs = dependency["all_node_outputs"]
        risk_bundle, ranker_manifest, candidate_manifest, screening_scores = self._build_final_output_ranker_bundle(final_outputs)
        budgets = self._final_output_ranker_budgets()
        base_pred = final_outputs.get("pred")
        if base_pred is None:
            probs = final_outputs.get("prob")
            if probs is None:
                probs = torch.softmax(final_outputs["logits"].float(), dim=1)
            base_pred = probs.argmax(dim=1)
        base_pred = base_pred.detach().cpu().numpy()
        triggered_mask = self._ranked_budget_mask(
            risk_bundle["risk_manifest"]["risk_score"],
            self.test_mask,
            budgets[0],
        )
        metrics_payload, budget_curve = self._build_metrics_payload(
            risk_bundle,
            base_pred,
            base_pred,
            self.test_mask,
            triggered_mask,
            extra_metrics={
                "canonical_stage2": "final_output_ranker",
                "screening_role": "high_recall_residual_candidate_ranking",
            },
            budgets=budgets,
        )
        paper_bundle = self._build_stage2_paper_report(
            "final_output_ranker",
            risk_bundle,
            final_outputs,
            budgets,
        )
        self._write_stage2_paper_artifacts(stage_dir, paper_bundle)
        metrics_payload["stage2_acceptance_gate"] = paper_bundle["acceptance_gate"]
        metrics_payload["stage2_claim_status"] = paper_bundle["report"]["claim_status"]
        notes = [
            "claim tested: canonical Stage 2 high-recall residual screening from Stage 1 final outputs only",
            "keep / kill: fixed canonical residual-risk ranker; graph-aware selectors remain calibration/ranking ablations",
            "selector scope: screening only, no attribution, no action selection, no graph context, no LLM call",
        ]
        write_json(stage_dir / "metrics.json", metrics_payload)
        write_csv_rows(
            stage_dir / "budget_curve.csv",
            ["budget", "triggered", "touched", "fix", "broke", "net", "utility", "screened_set_accuracy", "hcw_capture"],
            budget_curve,
        )
        write_text(stage_dir / "notes.md", "\n".join(notes).strip() + "\n")
        save_stage_artifacts(
            stage_dir,
            {
                "dependency_manifest": {
                    "stage_contract": "final_output_ranker_dependency_load_v1",
                    "reads": [
                        "stages/vertical_minimal/all_node_outputs.pt",
                        "stages/vertical_minimal/survivor_manifest.json",
                    ],
                    "recomputed_upstream": False,
                    "gats_required": False,
                    "router_oof_required": False,
                },
                "summary": {"final_output_ranker": risk_bundle["risk_manifest"].get("validation_metrics", {})},
                "risk_manifest": risk_bundle["risk_manifest"],
                "stage2_paper_report": paper_bundle["report"],
                "stage2_acceptance_gate": paper_bundle["acceptance_gate"],
                "stage2_hard_node_subgroups": paper_bundle["subgroup_report"],
                "final_output_ranker_manifest": ranker_manifest,
                "candidate_manifest": candidate_manifest,
                "screening_scores.pt": screening_scores,
                "survivor_manifest": {
                    "canonical_stage2": "final_output_ranker",
                    "canonical_candidate_manifest": "candidate_manifest.json",
                    "canonical_budgets": [float(item) for item in budgets],
                    "stage2_paper_report": "stage2_paper_report.json",
                    "stage2_acceptance_gate": "stage2_acceptance_gate.json",
                    "claim_status": paper_bundle["report"]["claim_status"],
                    "proxy_suite": ["final_output_ranker"],
                    "ablations": ["msp_ts", "gats_faithful", "dual_posterior_ranker", "calibrated_multiview_router"],
                },
            },
        )
        return {"stage": self.args.stage, "best_mode": "final_output_ranker"}

    def _run_dual_posterior_ranker_matrix(self, stage_dir, base_bundle):
        dependency = self._load_final_output_dependency()
        final_outputs = dependency["all_node_outputs"]
        risk_bundle, ranker_manifest, candidate_manifest, screening_scores, lm_bundle = self._build_dual_posterior_ranker_bundle(final_outputs)
        budgets = self._final_output_ranker_budgets()
        base_pred = final_outputs.get("pred")
        if base_pred is None:
            probs = final_outputs.get("prob")
            if probs is None:
                probs = torch.softmax(final_outputs["logits"].float(), dim=1)
            base_pred = probs.argmax(dim=1)
        base_pred = base_pred.detach().cpu().numpy()
        triggered_mask = self._ranked_budget_mask(
            risk_bundle["risk_manifest"]["risk_score"],
            self.test_mask,
            budgets[0],
        )
        metrics_payload, budget_curve = self._build_metrics_payload(
            risk_bundle,
            base_pred,
            base_pred,
            self.test_mask,
            triggered_mask,
            extra_metrics={
                "stage2_ablation": "dual_posterior_ranker",
                "screening_role": "high_recall_residual_candidate_ranking",
            },
            budgets=budgets,
        )
        lm_outputs = lm_bundle["outputs"]
        lm_probs = lm_outputs.get("prob_cal", lm_outputs.get("prob")).detach().cpu().float()
        paper_bundle = self._build_stage2_paper_report(
            "dual_posterior_ranker",
            risk_bundle,
            final_outputs,
            budgets,
            lm_probs=lm_probs,
        )
        self._write_stage2_paper_artifacts(stage_dir, paper_bundle)
        metrics_payload["stage2_acceptance_gate"] = paper_bundle["acceptance_gate"]
        metrics_payload["stage2_claim_status"] = paper_bundle["report"]["claim_status"]
        notes = [
            "claim tested: ablation-only Stage 2 residual screening from calibrated LM-only and final GNN posteriors",
            "keep / kill: compare against canonical final_output_ranker before any paper-facing promotion",
            "selector scope: screening only, no attribution, no action selection, no graph context, no LLM call",
        ]
        write_json(stage_dir / "metrics.json", metrics_payload)
        write_csv_rows(
            stage_dir / "budget_curve.csv",
            ["budget", "triggered", "touched", "fix", "broke", "net", "utility", "screened_set_accuracy", "hcw_capture"],
            budget_curve,
        )
        write_text(stage_dir / "notes.md", "\n".join(notes).strip() + "\n")
        save_stage_artifacts(
            stage_dir,
            {
                "dependency_manifest": {
                    "stage_contract": "dual_posterior_ranker_dependency_load_v1",
                    "reads": [
                        "stages/vertical_minimal/all_node_outputs.pt",
                        "stages/vertical_minimal/survivor_manifest.json",
                        "frozen/lm_only_head/manifest.json",
                        "frozen/lm_only_head/outputs.pt",
                    ],
                    "recomputed_upstream": lm_bundle.get("source") != "existing_frozen_lm_only_head",
                    "gats_required": False,
                    "router_oof_required": False,
                    "lm_only_head_required": True,
                    "lm_only_head_source": lm_bundle.get("source"),
                    "lm_only_head_manifest": str(self._lm_only_head_dir() / "manifest.json"),
                },
                "summary": {"dual_posterior_ranker": risk_bundle["risk_manifest"].get("validation_metrics", {})},
                "risk_manifest": risk_bundle["risk_manifest"],
                "stage2_paper_report": paper_bundle["report"],
                "stage2_acceptance_gate": paper_bundle["acceptance_gate"],
                "stage2_hard_node_subgroups": paper_bundle["subgroup_report"],
                "dual_posterior_ranker_manifest": ranker_manifest,
                "candidate_manifest": candidate_manifest,
                "dual_posterior_scores.pt": screening_scores,
                "lm_only_head_manifest": lm_bundle["manifest"],
                "survivor_manifest": {
                    "canonical_stage2": "final_output_ranker",
                    "ablation_stage2": "dual_posterior_ranker",
                    "canonical_budgets": [float(item) for item in budgets],
                    "stage2_paper_report": "stage2_paper_report.json",
                    "stage2_acceptance_gate": "stage2_acceptance_gate.json",
                    "claim_status": paper_bundle["report"]["claim_status"],
                    "proxy_suite": ["dual_posterior_ranker"],
                    "ablations": ["dual_posterior_ranker", "calibrated_multiview_router"],
                },
            },
        )
        return {"stage": self.args.stage, "best_mode": "dual_posterior_ranker"}

    def _run_estimator_matrix(self, stage_dir, base_bundle):
        if self._final_output_ranker_requested():
            return self._run_final_output_ranker_matrix(stage_dir, base_bundle)
        if self._dual_posterior_ranker_requested():
            return self._run_dual_posterior_ranker_matrix(stage_dir, base_bundle)
        if self._login_uncertainty_router_requested():
            return self._run_login_uncertainty_router_matrix(stage_dir, base_bundle)
        if self._graph_conformal_estimator_requested():
            return self._run_graph_conformal_estimator_matrix(stage_dir, base_bundle)

        require_stage_gates(self.experiment_root, "estimator_matrix")
        dependency = self._load_vertical_minimal_dependency()
        p0_bundle = self._bundle_from_risk_manifest(
            "msp_ts",
            dependency["risk_manifest"],
            structural_manifest=dependency["structural_manifest"],
            subgroup_op=dependency["subgroup_manifest_operational"],
            subgroup_analysis=dependency["subgroup_manifest_analysis"],
        )
        gats_bundle = self._build_gats_risk_bundle(load_gats_outputs(self.experiment_root))
        suite = {
            "msp_ts": p0_bundle,
            "gats_faithful": gats_bundle,
        }
        router_artifacts = {}
        router_budget_rows = None
        if self._router_requested():
            router_bundle, router_artifacts, router_budget_rows = self._build_multiview_router_bundle(load_gats_outputs(self.experiment_root))
            router_mode_name = router_bundle["risk_manifest"].get("calibration_metadata", {}).get("source", "calibrated_multiview_router")
            suite[str(router_mode_name)] = router_bundle

        requested_mode = str(getattr(self.args, "estimator_mode", "none")).lower()
        if requested_mode in {
            "calibrated_multiview_router",
            "calibrated_residual_risk_selector",
            "glance_for_context_residual_risk_selector",
        }:
            best_mode, risk_bundle = requested_mode, suite[requested_mode]
        else:
            best_mode, risk_bundle = self._pick_best_risk_bundle(suite)
        context = self.ensure_backbone_context()
        base_pred = context["gnn_outputs"]["pred"].numpy()
        triggered_mask = self._validation_frozen_high_mask(risk_bundle) & self.test_mask
        metrics_payload, budget_curve = self._build_metrics_payload(
            risk_bundle,
            base_pred,
            base_pred,
            self.test_mask,
            triggered_mask,
            extra_metrics={"best_p_proxy": best_mode},
            budgets=self._router_budgets(),
        )
        lm_probs_for_report = None
        if self._router_requested():
            lm_outputs_path = self._lm_only_head_dir() / "outputs.pt"
            if lm_outputs_path.exists():
                lm_outputs = safe_torch_load(lm_outputs_path, map_location="cpu")
                lm_probs_loaded = lm_outputs.get("prob_cal", lm_outputs.get("prob"))
                if lm_probs_loaded is not None:
                    lm_probs_for_report = lm_probs_loaded.detach().cpu().float()
        paper_bundle = self._build_stage2_paper_report(
            best_mode,
            risk_bundle,
            context["gnn_outputs"],
            self._router_budgets(),
            lm_probs=lm_probs_for_report,
        )
        self._write_stage2_paper_artifacts(stage_dir, paper_bundle)
        metrics_payload["stage2_acceptance_gate"] = paper_bundle["acceptance_gate"]
        metrics_payload["stage2_claim_status"] = paper_bundle["report"]["claim_status"]
        notes = [
            "claim tested: Stage 2 residual-risk estimation from Stage 1 posterior behavior",
            f"keep / kill: current stage-local risk anchor is {best_mode}",
            "enter next main table: only if top-budget error coverage beats uncertainty baselines across seeds",
        ]
        if best_mode in {
            "calibrated_multiview_router",
            "calibrated_residual_risk_selector",
            "glance_for_context_residual_risk_selector",
        }:
            notes.append("selector scope: no local rewrite, no LLM call, no raw embedding input, and no action utility in this stage")
        else:
            notes.append("selector scope: OOF graph-aware residual-risk selector not selected for this run")
        self._write_stage_outputs(
            stage_dir,
            risk_bundle,
            metrics_payload,
            budget_curve,
            notes,
            extra_artifacts={
                **base_bundle,
                "dependency_manifest": {
                    "stage_contract": "strict_frozen_dependency_load_v1",
                    "reads": [
                        "stages/vertical_minimal/risk_manifest.json",
                        "stages/vertical_minimal/subgroup_manifest_operational.json",
                        "frozen/gates/gats_faithful/manifest.json",
                        "frozen/gates/gats_faithful/outputs.pt",
                        *(
                            ["frozen/router_oof/manifest.json", "frozen/router_oof/outputs.pt", "frozen/lm_only_head/manifest.json"]
                            if self._router_requested()
                            else []
                        ),
                    ],
                    "recomputed_upstream": False,
                },
                "summary": {mode: suite[mode]["risk_manifest"]["validation_metrics"] for mode in suite},
                "risk_manifest": suite[best_mode]["risk_manifest"],
                "stage2_paper_report": paper_bundle["report"],
                "stage2_acceptance_gate": paper_bundle["acceptance_gate"],
                "stage2_hard_node_subgroups": paper_bundle["subgroup_report"],
                "structural_manifest": suite[best_mode]["structural_manifest"],
                "subgroup_manifest_operational": suite[best_mode]["subgroup_manifest_operational"],
                "subgroup_manifest_analysis": suite[best_mode]["subgroup_manifest_analysis"],
                "survivor_manifest": {
                    "best_p_proxy": best_mode,
                    "proxy_suite": list(suite.keys()),
                    "stage2_paper_report": "stage2_paper_report.json",
                    "stage2_acceptance_gate": "stage2_acceptance_gate.json",
                    "claim_status": paper_bundle["report"]["claim_status"],
                },
                **router_artifacts,
            },
            subgroup_extra={"best_mode": best_mode, "estimator_suite": list(suite.keys())},
        )
        if router_budget_rows is not None:
            write_csv_rows(
                stage_dir / "router_budget_curve.csv",
                ["budget", "selected_count", "error_recall", "precision", "lift", "remaining_risk"],
                router_budget_rows,
            )
            if best_mode in {"calibrated_residual_risk_selector", "glance_for_context_residual_risk_selector"}:
                write_csv_rows(
                    stage_dir / "residual_risk_budget_curve.csv",
                    ["budget", "selected_count", "error_recall", "precision", "lift", "remaining_risk"],
                    router_budget_rows,
                )
        return {"stage": self.args.stage, "best_mode": best_mode}

    def _run_semantic_matrix(self, stage_dir, base_bundle, risk_bundle):
        self._strict_recompute_forbidden("_run_semantic_matrix")

    def _run_semantic_source_matrix(self, stage_dir, base_bundle, risk_bundle, semantic_results):
        return run_phase_a_matrix(self.args, self.seed, self.data, stage_dir, base_bundle)

    def _run_repair_matrix(self, stage_dir, base_bundle, risk_bundle, semantic_results):
        self._strict_recompute_forbidden("_run_repair_matrix")

    def _run_selector_matrix(self, stage_dir, base_bundle, risk_bundle, semantic_results, repair_results):
        self._strict_recompute_forbidden("_run_selector_matrix")

    def _run_positioning_matrix(self, stage_dir, base_bundle, risk_bundle, semantic_results, repair_results):
        self._strict_recompute_forbidden("_run_positioning_matrix")

    def _run_backbone_stress(self, stage_dir, base_bundle, default_p_mode, default_sem_mode, default_rep_mode):
        self._strict_recompute_forbidden("_run_backbone_stress")

    def _run_appendix(self, stage_dir, base_bundle, risk_bundle):
        self._strict_recompute_forbidden("_run_appendix")

    def _run_eqc_v8_matrix(self, stage_dir, base_bundle):
        """Driver for ``--stage eqc_v8_matrix``.

        Executes the v8 EQC pipeline (see ``refine-logs/FINAL_PROPOSAL.md``)
        and dispatches to one of the pre-registered experimental blocks via
        ``--eqc_v8_block``. Stages 1-5 are constructed from the frozen Stage-0
        backbone outputs (``h_gnn``, ``p_gnn``) and the frozen LM-head
        outputs (``p_lm``); Stage-6 delegates to :mod:`llm_evidence_refiner`.

        The ``sanity`` block runs a single-seed plumbing check (Stage-1..6
        with the E-null variant, zero LLM calls); all other blocks implement
        the falsification experiments described in the FINAL_PROPOSAL
        Block-E / Block-P / Block-R / Block-L / Block-CT acceptance criteria.
        """
        from estimators import (
            TwoTermEgoUncertaintyEstimator,
            CalibratedExpectedErrorProxy,
        )
        from operators import (
            IterativeStructuralTextConfirmRetriever,
            EvidenceGraphRewriter,
        )
        from llm_evidence_refiner import (
            LLMEvidenceRefiner,
            build_block_e_encoder_specs,
            assemble_block_e_acceptance_report,
        )

        block = str(getattr(self.args, "eqc_v8_block", "sanity"))
        context = self.ensure_backbone_context()
        gnn_outputs = context.get("gnn_outputs", {})
        lm_outputs_path = self._lm_only_head_dir() / "outputs.pt"
        if not lm_outputs_path.exists():
            raise MissingFrozenArtifactError(
                "v8 EQC pipeline requires a frozen LM-only head. "
                f"Expected outputs at {lm_outputs_path}; run --stage semantic_finetune first."
            )
        lm_outputs = safe_torch_load(lm_outputs_path, map_location="cpu")

        p_gnn = gnn_outputs.get("prob")
        logits_gnn = gnn_outputs.get("logits")
        if p_gnn is None and logits_gnn is None:
            raise MissingFrozenArtifactError("Frozen GNN backbone must provide 'prob' or 'logits'.")
        p_lm = lm_outputs.get("prob_cal", lm_outputs.get("prob"))
        if p_lm is None:
            raise MissingFrozenArtifactError("Frozen LM-only head must provide 'prob' or 'prob_cal'.")

        labels = self.data["labels"]
        train_idx = self.data.get("train_idx")
        val_idx = self.data["valid_idx"]
        test_idx = self.data["test_idx"]

        estimator = TwoTermEgoUncertaintyEstimator(alpha=float(self.args.eqc_v8_alpha))
        estimator.fit(
            logits=logits_gnn,
            probs=p_gnn,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            lm_probs=p_lm,
        )
        manifest = estimator.build_manifest(
            logits=logits_gnn,
            probs=p_gnn,
            labels=labels,
            val_idx=val_idx,
            test_idx=test_idx,
            lm_probs=p_lm,
        )

        q_composite = np.asarray(manifest["q_composite"], dtype=np.float64)
        set_size = np.asarray(manifest["set_size"], dtype=np.float64)
        budget_frac = float(getattr(self.args, "eqc_v8_hard_node_budget", 0.15))
        num_nodes = int(q_composite.shape[0])
        valid_mask = np.zeros(num_nodes, dtype=bool)
        valid_mask[np.asarray(val_idx).reshape(-1)] = True
        target_count = max(int(round(float(valid_mask.sum()) * budget_frac)), 1)
        order = np.argsort(-q_composite[valid_mask])
        threshold_value = float(q_composite[valid_mask][order[target_count - 1]])
        hard_mask_all = q_composite >= threshold_value - 1e-12

        # Stage-4 accept-rule: calibrated expected-error proxy (5-fold CV on valid_cal).
        posterior_gnn_np = np.asarray(
            p_gnn.detach().cpu().numpy() if hasattr(p_gnn, "detach") else p_gnn,
            dtype=np.float64,
        )
        val_idx_np = np.asarray(val_idx, dtype=np.int64).reshape(-1)
        labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
        preds_np = posterior_gnn_np.argmax(axis=1)
        calibrator = CalibratedExpectedErrorProxy(
            n_folds=int(getattr(self.args, "eqc_v8_calibrator_folds", 5)),
            seed=int(self.seed),
        )
        calibrator.fit(
            posterior_val=posterior_gnn_np[val_idx_np],
            q_composite_val=q_composite[val_idx_np],
            set_size_val=set_size[val_idx_np],
            labels_val=labels_np[val_idx_np],
            preds_val=preds_np[val_idx_np],
        )

        stage_report = {
            "contract": "eqc_v8_matrix_stage_report_v1",
            "block": block,
            "estimator_fit_summary": estimator.fit_summary,
            "calibrator_fit_summary": calibrator.fit_summary,
            "hard_node_budget": float(budget_frac),
            "hard_node_count": int(hard_mask_all.sum()),
            "threshold_q": float(threshold_value),
        }

        if block == "sanity":
            retriever = IterativeStructuralTextConfirmRetriever(
                max_iterations=int(self.args.eqc_v8_t_ret),
                k_struct=int(self.args.eqc_v8_k_struct),
                epsilon=float(self.args.eqc_v8_epsilon_stop),
                single_pass_budget_matched=bool(self.args.eqc_v8_single_pass_budget_matched),
            )
            node_repr_source = next(
                (v for v in [gnn_outputs.get("node_repr"), gnn_outputs.get("h_gnn"), context.get("x_roberta")] if v is not None),
                None,
            )
            if node_repr_source is None:
                raise MissingFrozenArtifactError("Stage-3 retrieval requires a node embedding context.")
            retriever.fit(node_repr=node_repr_source, val_idx=val_idx)
            rewriter = EvidenceGraphRewriter(
                lambda_reweight=float(self.args.eqc_v8_lambda_reweight),
                rho_stability_cap=float(self.args.eqc_v8_rho_stability_cap),
                token_budget=int(self.args.eqc_v8_token_budget),
            )
            sanity_probe = LLMEvidenceRefiner(
                spec=build_block_e_encoder_specs(include_shuffle_controls=False, include_mistral=False)[0],
                h_gnn_dim=int(context.get("h_gnn_dim", 256)),
                x_roberta_dim=int(context.get("x_roberta_dim", 768)),
                seed=int(self.seed),
            )
            stage_report.update({
                "retriever_fit_summary": retriever.fit_summary,
                "rewriter_schema_version": rewriter.schema_version,
                "stage6_probe_variant": sanity_probe.spec.name,
                "sanity_status": "plumbing_initialized_no_linear_probe_fit",
                "smoke_note": (
                    "Stage-1..Stage-5 fits executed; Stage-6 linear probe is left unfit to keep "
                    "the sanity runner compute-free. Full falsification experiments live in the "
                    "block_e / block_p / block_r / block_l / block_ct branches."
                ),
            })
        elif block == "block_e":
            stage_report["block_e_acceptance_criterion"] = assemble_block_e_acceptance_report
            stage_report["block_e_variants_pending"] = [
                spec.name for spec in build_block_e_encoder_specs()
            ]
            stage_report["block_e_note"] = (
                "Block-E execution is driven by an external sweep driver that varies "
                "--eqc_v8_llm_backbone / --eqc_v8_schema_mode across the seven canonical "
                "encoder specs; this stage call materializes only the Stage-1..5 artifacts "
                "consumed by that sweep."
            )
        else:
            stage_report["pending_block"] = block
            stage_report["status"] = (
                f"Block '{block}' orchestration is declared in refine-logs/FINAL_PROPOSAL.md "
                "and scheduled for driver implementation in the experiment-queue / "
                "run-experiment layer; Stage-1..5 artifacts persist in stage_report."
            )

        self._write_stage_outputs(
            stage_dir,
            None,
            stage_report,
            [],
            notes_lines=[
                "claim tested: v8 EQC Stage-1..Stage-6 plumbing + Block selection",
                f"v8 block: {block}",
                "stage-5 canonical schema: v1.0 (target + supportive + suspicious + uncertain + quality_summary)",
            ],
            extra_artifacts={
                **base_bundle,
                "eqc_v8_stage_report": stage_report,
                "eqc_v8_estimator_manifest": {
                    "risk_score": manifest["risk_score"],
                    "prediction_sets": manifest["prediction_sets"],
                    "set_size": manifest["set_size"],
                    "thresholds": manifest["thresholds"],
                    "calibration_metadata": manifest["calibration_metadata"],
                    "q_composite_threshold": float(threshold_value),
                    "hard_node_mask_count": int(hard_mask_all.sum()),
                },
            },
            subgroup_extra={"eqc_v8_block": block},
        )
        return {"stage": self.args.stage, "eqc_v8_block": block}

    def _fail_outside_phase0_scope(self):
        require_stage_gates(self.experiment_root, self.args.stage)
        dependency_map = {
            "semantic_matrix": [("estimator_matrix", "survivor_manifest")],
            "repair_matrix": [("estimator_matrix", "survivor_manifest")],
            "selector_matrix": [("repair_matrix", "survivor_manifest")],
            "positioning_matrix": [("selector_matrix", "survivor_manifest")],
            "backbone_stress": [("selector_matrix", "survivor_manifest")],
            "appendix": [("estimator_matrix", "survivor_manifest")],
        }
        for stage_name, filename in dependency_map.get(self.args.stage, []):
            self._require_stage_json(stage_name, filename)
        raise MissingFrozenArtifactError(
            f"Stage '{self.args.stage}' is outside the Phase 0 validation scope. "
            "Strict frozen harness refuses to execute semantic, repair, selector, "
            "positioning, appendix, or backbone-stress recomputation paths in this pass."
        )

    def run(self):
        stage_dir = build_stage_dir(self.experiment_root, self.args.stage)
        base_bundle = self._base_artifact_bundle()

        if self.args.stage == "vertical_minimal":
            return self._run_vertical_minimal(stage_dir, base_bundle)

        if self.args.stage == "estimator_matrix":
            return self._run_estimator_matrix(stage_dir, base_bundle)

        if self.args.stage == "eqc_v8_matrix":
            return self._run_eqc_v8_matrix(stage_dir, base_bundle)

        if self.args.stage in {
            "semantic_matrix",
            "repair_matrix",
            "selector_matrix",
            "positioning_matrix",
            "backbone_stress",
            "appendix",
        }:
            return self._fail_outside_phase0_scope()

        if self.args.stage == "semantic_source_matrix":
            return self._run_semantic_source_matrix(stage_dir, base_bundle, None, None)

        raise ValueError(f"Unsupported stage: {self.args.stage}")

class GNN_Trainer:
    def __init__(
        self,
        model_name,
        device,
        optimizer_name,
        lr,
        weight_decay,
        dropout,
        pl_weight,
        batch_size,
        gnn_n_layers,
        n_relations,
        activation,
        gnn_epochs_per_iter,
        temperature,
        pl_ratio,
        intermediate_data_filepath,
        ckpt_filepath,
        pretrain_ckpt_filepath,
        train_idx,
        valid_idx,
        test_idx,
        hard_labels,
        edge_index,
        edge_type,
        run,
        SimpleHGN_att_res,
        att_heads,
        RGT_semantic_heads,
        gnn_hidden_dim,
        lm_name,
        pseudo_label_pool_idx=None
        ):

        self.model_name = model_name
        self.device = device
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.pl_weight = pl_weight
        self.dropout = dropout
        self.batch_size = batch_size
        self.gnn_n_layers = gnn_n_layers
        self.n_relations = n_relations
        self.activation = activation
        self.gnn_epochs_per_iter = gnn_epochs_per_iter
        self.temperature = temperature
        self.pl_ratio = pl_ratio
        self.intermediate_data_filepath = intermediate_data_filepath
        self.ckpt_filepath = ckpt_filepath
        self.pretrain_ckpt_filepath = pretrain_ckpt_filepath
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.hard_labels = hard_labels
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.run = run
        self.pseudo_label_pool_idx = pseudo_label_pool_idx
        self.SimpleHGN_att_res = SimpleHGN_att_res
        self.att_heads = att_heads
        self.RGT_semantic_heads = RGT_semantic_heads
        self.gnn_hidden_dim = gnn_hidden_dim
        self.lm_input_dim = 1024 if lm_name.lower() in ['roberta-large'] else 768
        self.iter = 0
        self.best_iter = 0
        self.best_valid_acc = 0
        self.best_valid_f1 = -float("inf")
        self.best_valid_epoch = 0
        self.criterion = CrossEntropyLoss()
        self.KD_criterion = KLDivLoss(log_target=False, reduction='batchmean')


        self.results = {}
        self.get_train_idx_all()
        self.optimizer_args = dict(lr=lr, weight_decay=weight_decay)

        self.model_config = {
            'GNN_model': model_name,
            'optimizer': optimizer_name,
            'gnn_n_layers': gnn_n_layers,
            'n_relations': n_relations,
            'activation': activation,
            'dropout': dropout,
            'gnn_hidden_dim': gnn_hidden_dim,
            'lm_input_dim': self.lm_input_dim,
            'SimpleHGN_att_res': SimpleHGN_att_res,
            'att_heads': att_heads,
            'RGT_semantic_heads': RGT_semantic_heads,
            'device': device
            }

        self.dataloader_config = {
            'batch_size': batch_size,
            'n_layers': gnn_n_layers
            }



    def build_model(self):
        self.model = build_GNN_model(self.model_config)

    def get_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.gnn_epochs_per_iter, eta_min=0)


    def get_optimizer(self):

        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "adadelta":
            optimizer = torch.optim.Adadelta(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "radam":
            optimizer = torch.optim.RAdam(self.model.parameters(), **self.optimizer_args)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        return optimizer

    def train(self, embeddings_LM, soft_labels):
        early_stop_flag = True

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        print('GNN training start!')
        print(f'This is iter {self.iter}')

        train_loader = build_GNN_dataloader(self.dataloader_config, self.train_idx_all, embeddings_LM, soft_labels, self.edge_index, self.edge_type, mode='train', is_pl=self.is_pl)

        for epoch in tqdm(range(self.gnn_epochs_per_iter)):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                batch_size = batch.batch_size
                x_batch = batch.x.to(self.device)

                edge_index_batch = batch.edge_index.to(self.device)
                edge_type_batch = batch.edge_type.to(self.device)
                is_pl = batch.is_pl[0: batch_size].to(self.device)
                labels = batch.labels[0: batch_size].to(self.device)

                output = self.model(x_batch, edge_index_batch, edge_type_batch)
                output = output[0: batch_size]

                pl_idx = torch.nonzero(is_pl == 1).squeeze()
                rl_idx = torch.nonzero(is_pl == 0).squeeze()


                if pl_idx.numel() == 0:
                    loss = self.criterion(output[rl_idx], labels[rl_idx])
                elif rl_idx.numel() == 0:
                    loss = self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), labels[pl_idx])
                else:
                    # loss = self.pl_weight * self.criterion(output[pl_idx], labels[pl_idx]) + (1 - self.pl_weight) * self.criterion(output[rl_idx], labels[rl_idx])
                    loss = self.pl_weight * self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), labels[pl_idx]) + (1 - self.pl_weight) * self.criterion(output[rl_idx], labels[rl_idx])

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.run.log({'GNN Train Loss': loss.item()})


            valid_acc, valid_f1 = self.eval(embeddings_LM)

            self.run.log({'GNN Valid Accuracy': valid_acc})
            self.run.log({'GNN Valid F1': valid_f1})

            if valid_f1 > self.best_valid_f1 or (valid_f1 == self.best_valid_f1 and valid_acc > self.best_valid_acc):
                early_stop_flag = False
                self.best_valid_acc = valid_acc
                self.best_valid_f1 = valid_f1
                self.best_epoch = epoch
                self.best_iter = self.iter
                torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'model_config': self.model_config}, self.ckpt_filepath / 'best.pkl')
        print(f'The highest valid macro-F1 is {self.best_valid_f1}!')
        return early_stop_flag

    def infer(self, embeddings_LM):
        self.model.eval()
        infer_loader = build_GNN_dataloader(self.dataloader_config, None, embeddings_LM, self.hard_labels, self.edge_index, self.edge_type, mode='infer')

        all_outputs = []
        all_labels = []
        with torch.no_grad():
            ckpt = safe_torch_load(self.ckpt_filepath / 'best.pkl')
            self.model.load_state_dict(ckpt['model'])
            # self.optimizer.load_state_dict(ckpt['optimizer'])
            # self.scheduler.load_state_dict(ckpt['scheduler'])
            for batch in infer_loader:
                batch_size = batch.batch_size
                x_batch = batch.x.to(self.device)

                edge_index_batch = batch.edge_index.to(self.device)
                edge_type_batch = batch.edge_type.to(self.device)
                labels = batch.labels[0: batch_size].to(self.device)

                output = self.model(x_batch, edge_index_batch, edge_type_batch)
                output = output[0: batch_size]

                all_outputs.append(output.cpu())
                all_labels.append(labels.cpu())

            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            soft_labels = torch.softmax(all_outputs / self.temperature, dim=1)
            soft_labels[self.train_idx] = all_labels[self.train_idx]

            torch.save(soft_labels, self.intermediate_data_filepath / f'soft_labels_iter_{self.iter}.pt')

            self.iter += 1


    def eval(self, embeddings_LM,  mode='valid'):
        if mode == 'valid':
            eval_loader =  build_GNN_dataloader(self.dataloader_config, self.valid_idx, embeddings_LM,  self.hard_labels, self.edge_index, self.edge_type, mode='eval')
        elif mode == 'test':
            eval_loader =  build_GNN_dataloader(self.dataloader_config, self.test_idx, embeddings_LM, self.hard_labels, self.edge_index, self.edge_type, mode='eval')
        self.model.eval()

        valid_predictions = []
        valid_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                batch_size = batch.batch_size
                x_batch = batch.x.to(self.device)
                edge_index_batch = batch.edge_index.to(self.device)
                edge_type_batch = batch.edge_type.to(self.device)
                labels = batch.labels[0: batch_size].to(self.device)

                output = self.model(x_batch, edge_index_batch, edge_type_batch)
                output = output[0: batch_size]

                valid_predictions.append(torch.argmax(output, dim=1).cpu().numpy())
                valid_labels.append(torch.argmax(labels, dim=1).cpu().numpy())

            valid_predictions = np.concatenate(valid_predictions)
            valid_labels = np.concatenate(valid_labels)
            valid_acc = accuracy_score(valid_labels, valid_predictions)
            valid_f1 = f1_score(valid_labels, valid_predictions, average="macro", zero_division=0)

            return valid_acc, valid_f1



    def test(self, embeddings_LM):
        print('Computing test accuracy and f1 for GNN...')
        ckpt = safe_torch_load(self.ckpt_filepath / 'best.pkl')
        self.model.load_state_dict(ckpt['model'])
        test_acc, test_f1 = self.eval(embeddings_LM, 'test')
        print(f'GNN Test Accuracy = {test_acc}')
        print(f'GNN Test F1 = {test_f1}')
        self.run.log({'GNN Test Accuracy': test_acc})
        self.run.log({'GNN Test F1': test_f1})
        self.results['accuracy'] = test_acc
        self.results['f1'] = test_f1

    def load_soft_labels(self, iter):
        soft_labels = safe_torch_load(self.intermediate_data_filepath / f'soft_labels_iter_{iter}.pt')
        return soft_labels

    def predict_all(self, embeddings_LM, load_best=True):
        self.model.eval()
        if load_best and (self.ckpt_filepath / 'best.pkl').exists():
            ckpt = safe_torch_load(self.ckpt_filepath / 'best.pkl', map_location=self.device)
            self.model.load_state_dict(ckpt['model'])

        with torch.no_grad():
            x = embeddings_LM.to(self.device)
            edge_index = self.edge_index.to(self.device)
            edge_type = self.edge_type.to(self.device)
            if hasattr(self.model, 'forward_outputs'):
                out = self.model.forward_outputs(x, edge_index, edge_type)
                logits = out['logits'].cpu()
                prob = out['prob'].cpu()
                node_repr = out['node_repr'].cpu()
                aux_features = out.get('aux_features', {})
            else:
                logits = self.model(x, edge_index, edge_type).cpu()
                prob = torch.softmax(logits, dim=1)
                node_repr = logits
                aux_features = {}

        return {
            'logits': logits,
            'prob': prob,
            'pred': prob.argmax(dim=1),
            'labels': self.hard_labels.cpu(),
            'node_repr': node_repr,
            'aux_features': aux_features,
        }

    def save_results(self, path):
        json.dump(self.results, open(path, 'w'), indent=4)

    def get_train_idx_all(self):
        n_total = self.hard_labels.shape[0]
        self.train_idx_all, self.is_pl, self.pl_idx = _safe_pseudo_label_training_index(
            train_idx=self.train_idx,
            valid_idx=self.valid_idx,
            test_idx=self.test_idx,
            n_total=n_total,
            pl_ratio=self.pl_ratio,
            pseudo_label_pool_idx=self.pseudo_label_pool_idx,
        )

class MLP_Trainer:
    def __init__(
        self,
        device,
        optimizer_name,
        lr,
        weight_decay,
        dropout,
        pl_weight,
        batch_size,
        n_layers,
        hidden_dim,
        activation,
        glnn_epochs,
        mlp_epochs_per_iter,
        temperature,
        pl_ratio,
        intermediate_data_filepath,
        ckpt_filepath,
        KD_ckpt_filepath,
        train_idx,
        valid_idx,
        test_idx,
        hard_labels,
        run,
        seed,
        use_gnn,
        pseudo_label_pool_idx=None):

        self.device = device
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.pl_weight = pl_weight
        self.dropout = dropout
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.mlp_epochs_per_iter = mlp_epochs_per_iter
        self.glnn_epochs = glnn_epochs
        self.temperature = temperature
        self.pl_ratio = pl_ratio
        self.intermediate_data_filepath = intermediate_data_filepath
        self.ckpt_filepath = ckpt_filepath
        self.KD_ckpt_filepath = KD_ckpt_filepath
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.hard_labels = hard_labels
        self.run = run
        self.seed = seed
        self.use_gnn = use_gnn
        self.pseudo_label_pool_idx = pseudo_label_pool_idx
        self.iter = 0
        self.best_iter = 0
        self.best_valid_acc = 0
        self.best_valid_f1 = -float("inf")
        self.best_valid_epoch = 0
        self.criterion = CrossEntropyLoss()
        self.KD_criterion = KLDivLoss(log_target=False, reduction='batchmean')


        self.get_train_idx_all()
        self.results = {}

        self.dataloader_config = {
            'batch_size': batch_size
            }

        self.optimizer_args = dict(lr=lr, weight_decay=weight_decay)

    def get_scheduler(self, optimizer, T_max):
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)


    def get_optimizer(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "adadelta":
            optimizer = torch.optim.Adadelta(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "radam":
            optimizer = torch.optim.RAdam(self.model.parameters(), **self.optimizer_args)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        return optimizer

    def build_model(self):
        if self.use_gnn:
            self.model = MLP(in_channels=768, hidden_channels=self.hidden_dim, out_channels=2, dropout=self.dropout, act=self.activation, num_layers=self.n_layers).to(self.device)
        else:
            ckpt = safe_torch_load(self.KD_ckpt_filepath / f'seed_{self.seed}_best.pkl')
            self.model = MLP(**ckpt['model_params']).to(self.device)
            # self.model.load_state_dict(ckpt['model'])


    def KD_GLNN(self, LM_embeddings, soft_labels):
        print('Distilling from GNN to GLNN')
        train_loader = build_MLP_dataloader(self.dataloader_config, self.train_idx_all, LM_embeddings, soft_labels, mode='train', is_pl=self.is_pl)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer, self.glnn_epochs)

        for epoch in tqdm(range(self.glnn_epochs)):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                LM_embedding, label, is_pl = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                output = self.model(LM_embedding)

                pl_idx = torch.nonzero(is_pl == 1).squeeze()
                rl_idx = torch.nonzero(is_pl == 0).squeeze()

                if pl_idx.numel() == 0:
                    loss = self.criterion(output[rl_idx], label[rl_idx])
                elif rl_idx.numel() == 0:
                    loss = self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), label[pl_idx])
                else:
                    # loss = self.pl_weight * self.criterion(output[pl_idx], labels[pl_idx]) + (1 - self.pl_weight) * self.criterion(output[rl_idx], labels[rl_idx])
                    loss = self.pl_weight * self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), label[pl_idx]) + (1 - self.pl_weight) * self.criterion(output[rl_idx], label[rl_idx])

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.run.log({'GLNN KD Train Loss': loss.item()})


            valid_acc, valid_f1 = self.eval(LM_embeddings)

            self.run.log({'GLNN KD Valid Accuracy': valid_acc})
            self.run.log({'GLNN KD Valid F1': valid_f1})

            if valid_f1 > self.best_valid_f1 or (valid_f1 == self.best_valid_f1 and valid_acc > self.best_valid_acc):
                self.best_valid_acc = valid_acc
                self.best_valid_f1 = valid_f1
                self.best_epoch = epoch
                torch.save({'model': self.model.state_dict(), 'model_params': {'num_layers': self.n_layers, 'hidden_channels': self.hidden_dim, 'dropout': self.dropout, 'act': self.activation, 'in_channels': 768, 'out_channels': 2}}, self.KD_ckpt_filepath / f'seed_{self.seed}_best.pkl')

        print(f'The highest valid macro-F1 is {self.best_valid_f1}!')
        print(f'Save model from epoch {self.best_epoch}')

    def train(self, LM_embeddings, soft_labels):
        print('MLP training start!')
        print(f'This is iter {self.iter}')
        train_loader = build_MLP_dataloader(self.dataloader_config, self.train_idx_all, LM_embeddings, soft_labels, mode='train', is_pl=self.is_pl)
        early_stop_flag = True
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer, self.glnn_epochs)
        for epoch in tqdm(range(self.mlp_epochs_per_iter)):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                LM_embedding, label, is_pl = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                output = self.model(LM_embedding)

                pl_idx = torch.nonzero(is_pl == 1).squeeze()
                rl_idx = torch.nonzero(is_pl == 0).squeeze()

                if pl_idx.numel() == 0:
                    loss = self.criterion(output[rl_idx], label[rl_idx])
                elif rl_idx.numel() == 0:
                    loss = self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), label[pl_idx])
                else:
                    loss = self.pl_weight * self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), label[pl_idx]) + (1 - self.pl_weight) * self.criterion(output[rl_idx], label[rl_idx])

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.run.log({'MLP Train Loss': loss.item()})



            valid_acc, valid_f1 = self.eval(LM_embeddings)

            self.run.log({'MLP Valid Accuracy': valid_acc})
            self.run.log({'MLP Valid F1': valid_f1})

            if valid_f1 > self.best_valid_f1 or (valid_f1 == self.best_valid_f1 and valid_acc > self.best_valid_acc):
                early_stop_flag = False
                self.best_valid_acc = valid_acc
                self.best_valid_f1 = valid_f1
                self.best_epoch = epoch
                self.best_iter = self.iter
                torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, self.ckpt_filepath / 'best.pkl')
        print(f'The highest valid macro-F1 is {self.best_valid_f1}!')
        return early_stop_flag

    def infer(self, LM_embeddings):
        self.model.eval()
        infer_loader = build_MLP_dataloader(self.dataloader_config, None, LM_embeddings,  self.hard_labels, mode='infer')
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            ckpt = safe_torch_load(self.ckpt_filepath / 'best.pkl')
            self.model.load_state_dict(ckpt['model'])
            for batch in infer_loader:
                LM_embedding, label = batch[0].to(self.device), batch[1].to(self.device)
                output = self.model(LM_embedding)
                all_outputs.append(output.cpu())
                all_labels.append(label.cpu())

            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            soft_labels = torch.softmax(all_outputs / self.temperature, dim=1)
            soft_labels[self.train_idx] = all_labels[self.train_idx]

            torch.save(soft_labels, self.intermediate_data_filepath / f'soft_labels_iter_{self.iter}.pt')

            self.iter += 1

    def eval(self, LM_embeddings, mode='valid'):
        if mode == 'valid':
            eval_loader =  build_MLP_dataloader(self.dataloader_config, self.valid_idx, LM_embeddings, self.hard_labels, mode='eval')
        elif mode == 'test':
            eval_loader =  build_MLP_dataloader(self.dataloader_config, self.test_idx, LM_embeddings, self.hard_labels, mode='eval')
        self.model.eval()

        valid_predictions = []
        valid_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                LM_embedding, label = batch[0].to(self.device), batch[1].to(self.device)
                output = self.model(LM_embedding)

                valid_predictions.append(torch.argmax(output, dim=1).cpu().numpy())
                valid_labels.append(torch.argmax(label, dim=1).cpu().numpy())

            valid_predictions = np.concatenate(valid_predictions)
            valid_labels = np.concatenate(valid_labels)
            valid_acc = accuracy_score(valid_labels, valid_predictions)
            valid_f1 = f1_score(valid_labels, valid_predictions, average="macro", zero_division=0)

            return valid_acc, valid_f1

    def test(self, LM_embeddings):
        print('Computing test accuracy and f1 for MLP...')
        ckpt = safe_torch_load(self.ckpt_filepath / 'best.pkl')
        self.model.load_state_dict(ckpt['model'])
        test_acc, test_f1 = self.eval(LM_embeddings, 'test')
        print(f'MLP Test Accuracy = {test_acc}')
        print(f'MLP Test F1 = {test_f1}')
        self.run.log({'MLP Test Accuracy': test_acc})
        self.run.log({'MLP Test F1': test_f1})
        self.results['accuracy'] = test_acc
        self.results['f1'] = test_f1

    def save_results(self, path):
        json.dump(self.results, open(path, 'w'), indent=4)

    def get_train_idx_all(self):
        n_total = self.hard_labels.shape[0]
        self.train_idx_all, self.is_pl, self.pl_idx = _safe_pseudo_label_training_index(
            train_idx=self.train_idx,
            valid_idx=self.valid_idx,
            test_idx=self.test_idx,
            n_total=n_total,
            pl_ratio=self.pl_ratio,
            pseudo_label_pool_idx=self.pseudo_label_pool_idx,
        )
