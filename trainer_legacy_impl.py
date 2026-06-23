import importlib.util
import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, roc_auc_score
import numpy as np
from time import perf_counter
from model_building import (
    PhaseAInputAdapter,
    _runtime_concat_full_graph_support_embeddings,
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
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn.models import MLP
from torch_geometric.nn import RGCNConv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils import (
    build_experiment_root,
    build_preparation_dir,
    build_stage_dir,
    capture_code_metadata,
    ensure_dir,
    resolve_dataset_path,
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
from stage_registry import resolve_stage_name
from router import (
    GlanceReliabilityRouterMLP,
    build_reliability_router_feature_bundle,
    fit_reliability_temperature,
)
from estimators import (
    GlanceForContextResidualRiskSelector,
    RESIDUAL_RISK_PAPER_BUDGETS,
    build_glance_for_context_router_features,
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
from trainer_distillation_metrics import (
    _compute_budget_curve,
    _empty_gain_cost_report,
    _paired_bootstrap_delta,
)
from trainer_indexing import _safe_pseudo_label_training_index
from artifact_contracts import (
    MissingFrozenArtifactError,
    PHASE_A_CONTRACT,
    PHASE_A_DISABLED_COMPONENTS,
    canonical_stage_name as _canonical_stage_name,
    canonical_stage_dir as _canonical_stage_dir,
    stage_artifact_reference as _stage_artifact_reference,
    preparation_artifact_reference as _preparation_artifact_reference,
    split_provenance,
    frozen_g0_dir,
)

try:
    from transformers.optimization import get_cosine_schedule_with_warmup
except Exception:
    from torch.optim.lr_scheduler import LambdaLR

    def get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps,
        num_training_steps,
        num_cycles=0.5,
        last_epoch=-1,
    ):
        """Compatibility fallback for environments where transformers schedulers fail to import."""

        num_warmup_steps = max(int(num_warmup_steps), 0)
        num_training_steps = max(int(num_training_steps), 1)
        num_cycles = float(num_cycles)

        def lr_lambda(current_step):
            current_step = int(current_step)
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)))

        return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
from runtime_env import (
    _max_cuda_memory_allocated as _runtime_max_cuda_memory_allocated,
    _reset_cuda_peak_memory_stats as _runtime_reset_cuda_peak_memory_stats,
    _resolve_device as _runtime_resolve_device,
    _set_cuda_device as _runtime_set_cuda_device,
)
from trainer_preparation import (
    build_or_load_faithful_gats as _prep_build_or_load_faithful_gats,
    build_or_load_frozen_g0 as _prep_build_or_load_frozen_g0,
    load_frozen_g0 as _prep_load_frozen_g0,
    load_gats_outputs as _prep_load_gats_outputs,
    require_stage_gates as _prep_require_stage_gates,
)
from trainer_semantic import (
    _classification_metrics_from_logits as _semantic_classification_metrics_from_logits,
    run_semantic_finetune_seed as _semantic_run_semantic_finetune_seed,
)


FROZEN_G0_CONTRACT = "frozen_g0_v1"
GATS_GATE_NAME = "gats_faithful"
GATS_CONTRACT = "faithful_gats_anchor_v1"

FORMAL_STAGE_GATES = {
    "estimator_ablation": [GATS_GATE_NAME],
    "semantic_operator_ablation": ["lagnn_near_faithful"],
    "semantic_source_ablation": ["lagnn_near_faithful"],
    "repair_operator_ablation": ["gnnguard_local", "cs_conditional_supporting"],
    "selector_ablation": ["gnnguard_local", "cs_conditional_supporting"],
    "positioning_ablation": ["gnnguard_local", "cs_conditional_supporting"],
    "backbone_stress_test": ["gnnguard_local", "cs_conditional_supporting"],
    "local_conformal_diagnostic": [],
    "local_conflict_prune_diag": [],
    "joint_router_refinement": [],
}

CANONICAL_TO_LEGACY_STAGE = {
    "distillation_pipeline": "legacy_distill",
    "semantic_encoder_finetune": "semantic_finetune",
    "graph_detector_prepare": "frozen_g0",
    "graph_calibration_prepare": "frozen_gats",
    "local_conformal_diagnostic": "local_conformal_prune_diag",
    "joint_router_refinement": "glance_joint_router_refine",
    "minimal_pipeline": "vertical_minimal",
    "estimator_ablation": "estimator_matrix",
    "semantic_operator_ablation": "semantic_matrix",
    "semantic_source_ablation": "semantic_source_matrix",
    "repair_operator_ablation": "repair_matrix",
    "selector_ablation": "selector_matrix",
    "positioning_ablation": "positioning_matrix",
    "backbone_stress_test": "backbone_stress",
    "appendix_ablation": "appendix",
    "glance_oracle_refinement_internal": "glance_oracle_refine",
    "glance_full_graph_refinement_internal": "glance_full_graph_refine",
    "glance_counterfactual_router_internal": "glance_counterfactual_router",
    "glance_budgeted_refinement_internal": "glance_budgeted_refine",
    "glance_refiner_analysis_internal": "glance_refiner_analysis",
    "phase_a_single_cell_internal": "phase_a_single_cell_internal",
}


class StructuralConflictRGCN(nn.Module):
    """Structure-only auxiliary view used by local_conflict_prune_diag."""

    def __init__(self, num_nodes, node_emb_dim=64, hidden_dim=64, n_relations=2, n_layers=2, dropout=0.1):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.node_emb_dim = int(node_emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_relations = int(n_relations)
        self.n_layers = int(n_layers)
        self.node_table = nn.Embedding(self.num_nodes, self.node_emb_dim)
        self.linear_in = nn.Linear(self.node_emb_dim, self.hidden_dim)
        self.convs = nn.ModuleList(
            [RGCNConv(self.hidden_dim, self.hidden_dim, self.n_relations) for _ in range(self.n_layers)]
        )
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, 2)
        self.dropout = nn.Dropout(float(dropout))
        self.activation = nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_table.weight)
        nn.init.xavier_uniform_(self.linear_in.weight)
        nn.init.zeros_(self.linear_in.bias)
        for conv in self.convs:
            conv.reset_parameters()
        nn.init.xavier_uniform_(self.linear_pool.weight)
        nn.init.zeros_(self.linear_pool.bias)
        nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

    def forward_outputs(self, edge_index, edge_type):
        node_ids = torch.arange(self.num_nodes, device=self.node_table.weight.device, dtype=torch.long)
        x = self.node_table(node_ids)
        x = self.linear_in(x)
        x = self.dropout(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = self.activation(x)
        hidden = self.linear_pool(x)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        logits = self.linear_out(hidden)
        prob = torch.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "prob": prob,
            "pred": prob.argmax(dim=1),
            "node_repr": hidden,
        }


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
        "python main.py --embedding_path /path/to/emb.pt "
        "--use_GNN --graph_backbone rgcn --seeds 1"
    )


def _legacy_stage_name(stage_name):
    resolved = resolve_stage_name(stage_name)
    return CANONICAL_TO_LEGACY_STAGE.get(resolved, resolved)


def _legacy_stage_dir(root, stage_name):
    legacy_stage = _legacy_stage_name(stage_name)
    return Path(root) / "stages" / legacy_stage


def _stage_dir_read_candidates(root, stage_name):
    canonical_dir = _canonical_stage_dir(root, stage_name)
    legacy_dir = _legacy_stage_dir(root, stage_name)
    candidates = [canonical_dir]
    if legacy_dir != canonical_dir:
        candidates.append(legacy_dir)
    return candidates


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
    refine_request = _graph_refine_request(args)
    labels = _labels_to_index(data["labels"])
    if features.shape[0] != labels.numel():
        raise ValueError(f"G0 feature rows ({features.shape[0]}) must match labels ({labels.numel()}).")

    device = getattr(args, "device", torch.device("cpu"))
    if not isinstance(device, torch.device):
        device = torch.device("cpu" if int(device) < 0 or not torch.cuda.is_available() else f"cuda:{int(device)}")

    x_projected = features.to(device)
    x_raw = raw_features.to(device)
    y = labels.to(device)
    edge_index = data["edge_index"]
    edge_type = data["edge_type"]
    graph_refine_stats = None
    if refine_request["mode"] != "none":
        edge_index, edge_type, graph_refine_stats = _directional_relation_aware_prune(
            edge_index=edge_index,
            edge_type=edge_type,
            raw_features=raw_features,
            budget=refine_request["budget"],
            mode=refine_request["mode"],
            seed=seed,
            labels=labels,
        )
        write_torch(out_dir / "pruned_edge_index.pt", edge_index)
        write_torch(out_dir / "pruned_edge_type.pt", edge_type)
        write_json(out_dir / "graph_refine_stats.json", graph_refine_stats)
    else:
        for stale_path in (
            out_dir / "pruned_edge_index.pt",
            out_dir / "pruned_edge_type.pt",
            out_dir / "graph_refine_stats.json",
        ):
            if stale_path.exists():
                stale_path.unlink()
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
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
        raw_outputs = model.forward_outputs(x, edge_index, edge_type)
    outputs = {
        "logits": raw_outputs["logits"].detach().cpu(),
        "prob": raw_outputs["prob"].detach().cpu(),
        "pred": raw_outputs["prob"].argmax(dim=1).detach().cpu(),
        "labels": data["labels"].detach().cpu() if torch.is_tensor(data["labels"]) else torch.tensor(data["labels"]),
        "node_repr": raw_outputs["node_repr"].detach().cpu(),
        "fused_x": (
            raw_outputs["fused_x"].detach().cpu()
            if torch.is_tensor(raw_outputs.get("fused_x"))
            else raw_outputs["node_repr"].detach().cpu()
        ),
        "aux_features": raw_outputs.get("aux_features", {}),
    }
    if torch.is_tensor(raw_outputs.get("x_low")):
        outputs["x_low"] = raw_outputs["x_low"].detach().cpu()
    if torch.is_tensor(raw_outputs.get("x_new")):
        outputs["x_new"] = raw_outputs["x_new"].detach().cpu()

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
            "canonical_task_name": "graph_detector_prepare",
            "legacy_task_name_used": getattr(args, "legacy_task_name_used", None),
            "deprecated_cli_flags": list(getattr(args, "deprecated_cli_flags", [])),
            "stage_visibility": "public",
            "artifact_namespace": "preparation/graph_detector",
            "invocation": {
                "requested_task": getattr(args, "requested_experiment_task", getattr(args, "experiment_task", None)),
                "resolved_task": "graph_detector_prepare",
            },
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
            **(
                {
                    "graph_refine": {
                        **graph_refine_stats,
                        "embedding_source": feature_manifest.get("path"),
                    }
                }
                if graph_refine_stats is not None
                else {}
            ),
        },
    )
    return load_frozen_g0(experiment_root)


def load_frozen_g0(experiment_root):
    experiment_root = Path(experiment_root)
    canonical_dir = experiment_root / "preparation" / "graph_detector"
    legacy_dir = Path(experiment_root) / "frozen" / "g0"
    out_dir = canonical_dir if (canonical_dir / "manifest.json").exists() else legacy_dir
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
            "Run graph_detector_prepare first."
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
    path = (
        getattr(args, "embedding_path", None)
        or getattr(args, "emb_path", None)
        or getattr(args, "g0_feature_path", None)
    )
    return str(Path(path)) if path else None


def _graph_refine_request(args):
    mode = str(getattr(args, "graph_refine_mode", "none") or "none").lower()
    budget = float(getattr(args, "graph_refine_budget", 0.0) or 0.0)
    graph_refine_positioning = str(getattr(args, "graph_refine_positioning", "none") or "none").strip().lower()
    graph_refine_control_only = bool(getattr(args, "graph_refine_control_only", False))
    if mode == "none":
        return {
            "mode": "none",
            "budget": 0.0,
            "degree_guard_enabled": False,
            "oracle_uses_labels": False,
            "diagnostic_only": False,
            "positioning": graph_refine_positioning,
            "control_only": graph_refine_control_only,
        }
    if mode not in {
        "directional_relation_aware_prune",
        "directional_relation_aware_random_prune",
        "directional_relation_aware_oracle_prune",
        "directional_relation_aware_oracle_prune_no_hetero_priority",
    }:
        raise ValueError(f"Unknown graph_refine_mode: {mode}")
    if not (0.0 < budget < 1.0):
        raise ValueError(
            "graph_refine_budget must be in (0, 1) when --graph_refine_mode is not none."
        )
    return {
        "mode": mode,
        "budget": budget,
        "degree_guard_enabled": True,
        "oracle_uses_labels": mode in {
            "directional_relation_aware_oracle_prune",
            "directional_relation_aware_oracle_prune_no_hetero_priority",
        },
        "diagnostic_only": mode in {
            "directional_relation_aware_oracle_prune",
            "directional_relation_aware_oracle_prune_no_hetero_priority",
        },
        "hetero_priority": mode == "directional_relation_aware_oracle_prune",
        "positioning": graph_refine_positioning,
        "control_only": graph_refine_control_only,
    }


def _directional_relation_aware_prune(edge_index, edge_type, raw_features, budget, mode, seed=None, labels=None):
    edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
    edge_type_cpu = edge_type.detach().cpu().long() if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long)
    raw_features_cpu = raw_features.detach().cpu().float() if torch.is_tensor(raw_features) else torch.tensor(raw_features, dtype=torch.float32)

    if edge_index_cpu.dim() != 2 or edge_index_cpu.size(0) != 2:
        raise ValueError("edge_index must be shaped [2, num_edges] for directional_relation_aware_prune.")
    if edge_type_cpu.numel() != edge_index_cpu.size(1):
        raise ValueError("edge_type must align with edge_index for directional_relation_aware_prune.")
    if raw_features_cpu.dim() != 2:
        raise ValueError("raw_features must be a 2D tensor for directional_relation_aware_prune.")
    if mode in {
        "directional_relation_aware_oracle_prune",
        "directional_relation_aware_oracle_prune_no_hetero_priority",
    }:
        if labels is None:
            raise ValueError("labels are required for directional_relation_aware_oracle_prune variants.")
        labels_cpu = labels.detach().cpu().long() if torch.is_tensor(labels) else torch.tensor(labels, dtype=torch.long)
        if labels_cpu.dim() == 2:
            labels_cpu = labels_cpu.argmax(dim=1)
        labels_cpu = labels_cpu.view(-1)
    else:
        labels_cpu = None

    num_nodes = int(raw_features_cpu.size(0))
    num_edges_before = int(edge_index_cpu.size(1))
    oracle_uses_labels = mode in {
        "directional_relation_aware_oracle_prune",
        "directional_relation_aware_oracle_prune_no_hetero_priority",
    }
    hetero_priority = mode == "directional_relation_aware_oracle_prune"
    diagnostic_only = oracle_uses_labels
    if mode == "directional_relation_aware_random_prune":
        score_source = "random_per_relation_seeded"
    elif oracle_uses_labels:
        score_source = "oracle_heterophily_then_raw_semantic_cosine" if hetero_priority else "oracle_raw_semantic_cosine"
    else:
        score_source = "raw_semantic_cosine"
    if num_edges_before == 0:
        stats = {
            "mode": mode,
            "budget_requested": float(budget),
            "budget_achieved_global": 0.0,
            "budget_achieved_by_relation": {},
            "num_edges_before": 0,
            "num_edges_after": 0,
            "num_edges_pruned": 0,
            "mean_cos_pruned": None,
            "mean_cos_kept": None,
            "isolated_in_nodes_after": int(num_nodes),
            "isolated_out_nodes_after": int(num_nodes),
            "degree_guard_enabled": True,
            "score_source": score_source,
            "oracle_uses_labels": oracle_uses_labels,
            "diagnostic_only": diagnostic_only,
            "hetero_priority": hetero_priority,
        }
        return edge_index_cpu.contiguous(), edge_type_cpu.contiguous(), stats

    normalized = F.normalize(raw_features_cpu, p=2, dim=1, eps=1e-12)
    src = edge_index_cpu[0]
    dst = edge_index_cpu[1]
    cosine = (normalized[src] * normalized[dst]).sum(dim=1)
    prune_score = 1.0 - cosine
    heterophilic = None
    if labels_cpu is not None:
        heterophilic = (labels_cpu[src] != labels_cpu[dst]).to(torch.long)

    keep_mask = torch.ones(num_edges_before, dtype=torch.bool)
    out_degree = torch.bincount(src, minlength=num_nodes).to(torch.long)
    in_degree = torch.bincount(dst, minlength=num_nodes).to(torch.long)
    relation_stats = {}
    rng = np.random.default_rng(int(seed)) if mode == "directional_relation_aware_random_prune" else None

    for relation_id in sorted(int(rel) for rel in edge_type_cpu.unique().tolist()):
        relation_indices = torch.nonzero(edge_type_cpu == relation_id, as_tuple=False).view(-1)
        relation_total = int(relation_indices.numel())
        target_prune = int(math.floor(float(budget) * relation_total))
        pruned_in_relation = 0
        if target_prune > 0 and relation_total > 0:
            if mode == "directional_relation_aware_random_prune":
                relation_positions = relation_indices.numpy().copy()
                rng.shuffle(relation_positions)
                ordered = torch.tensor(relation_positions, dtype=torch.long)
            elif mode in {
                "directional_relation_aware_oracle_prune",
                "directional_relation_aware_oracle_prune_no_hetero_priority",
            }:
                relation_hetero = heterophilic[relation_indices]
                relation_scores = prune_score[relation_indices]
                if hetero_priority:
                    priority = torch.stack((relation_hetero.float(), relation_scores), dim=1)
                    ordered_ids = sorted(
                        range(relation_total),
                        key=lambda idx: (
                            float(priority[idx, 0].item()),
                            float(priority[idx, 1].item()),
                        ),
                        reverse=True,
                    )
                else:
                    ordered_ids = sorted(
                        range(relation_total),
                        key=lambda idx: float(relation_scores[idx].item()),
                        reverse=True,
                    )
                ordered = relation_indices[torch.tensor(ordered_ids, dtype=torch.long)]
            else:
                relation_scores = prune_score[relation_indices]
                ordered = relation_indices[torch.argsort(relation_scores, descending=True)]
            for edge_pos in ordered.tolist():
                u = int(src[edge_pos])
                v = int(dst[edge_pos])
                if out_degree[u] <= 1 or in_degree[v] <= 1:
                    continue
                keep_mask[edge_pos] = False
                out_degree[u] -= 1
                in_degree[v] -= 1
                pruned_in_relation += 1
                if pruned_in_relation >= target_prune:
                    break
        achieved = float(pruned_in_relation / relation_total) if relation_total else 0.0
        relation_stats[str(relation_id)] = {
            "num_edges_before": relation_total,
            "num_edges_pruned": int(pruned_in_relation),
            "budget_requested": float(budget),
            "budget_achieved": achieved,
        }

    pruned_edge_index = edge_index_cpu[:, keep_mask].contiguous()
    pruned_edge_type = edge_type_cpu[keep_mask].contiguous()
    num_edges_after = int(pruned_edge_index.size(1))
    num_edges_pruned = int((~keep_mask).sum().item())
    mean_cos_pruned = float(cosine[~keep_mask].mean().item()) if num_edges_pruned else None
    mean_cos_kept = float(cosine[keep_mask].mean().item()) if num_edges_after else None
    isolated_in_nodes_after = int((in_degree == 0).sum().item())
    isolated_out_nodes_after = int((out_degree == 0).sum().item())
    stats = {
        "mode": mode,
        "budget_requested": float(budget),
        "budget_achieved_global": (float(num_edges_pruned) / float(num_edges_before)) if num_edges_before else 0.0,
        "budget_achieved_by_relation": relation_stats,
        "num_edges_before": num_edges_before,
        "num_edges_after": num_edges_after,
        "num_edges_pruned": num_edges_pruned,
        "mean_cos_pruned": mean_cos_pruned,
        "mean_cos_kept": mean_cos_kept,
        "isolated_in_nodes_after": isolated_in_nodes_after,
        "isolated_out_nodes_after": isolated_out_nodes_after,
        "degree_guard_enabled": True,
        "score_source": score_source,
        "oracle_uses_labels": oracle_uses_labels,
        "diagnostic_only": diagnostic_only,
    }
    return pruned_edge_index, pruned_edge_type, stats


def _resolve_optional_graph_override_paths(args):
    edge_index_path = getattr(args, "external_graph_edge_index_path", None)
    edge_type_path = getattr(args, "external_graph_edge_type_path", None)
    if not edge_index_path and not edge_type_path:
        return None
    if not edge_index_path or not edge_type_path:
        raise MissingFrozenArtifactError(
            "Both --external_graph_edge_index_path and --external_graph_edge_type_path are required together."
        )
    return {
        "edge_index_path": Path(edge_index_path),
        "edge_type_path": Path(edge_type_path),
    }


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

    refine_request = _graph_refine_request(args)
    refine_manifest = manifest.get("graph_refine")
    if refine_request["mode"] == "none":
        return refine_manifest in (None, {}, {"mode": "none"})
    if not isinstance(refine_manifest, dict):
        return False
    if str(refine_manifest.get("mode", "")).lower() != refine_request["mode"]:
        return False
    if not math.isclose(float(refine_manifest.get("budget_requested", -1.0)), float(refine_request["budget"]), rel_tol=0.0, abs_tol=1e-12):
        return False
    if bool(refine_manifest.get("degree_guard_enabled", False)) != bool(refine_request["degree_guard_enabled"]):
        return False
    if bool(refine_manifest.get("oracle_uses_labels", False)) != bool(refine_request["oracle_uses_labels"]):
        return False
    if bool(refine_manifest.get("diagnostic_only", False)) != bool(refine_request["diagnostic_only"]):
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
    if gate_name == GATS_GATE_NAME:
        return ensure_dir(build_preparation_dir(experiment_root, "graph_calibrator"))
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
    canonical_path = gate_dir(experiment_root, gate_name) / "manifest.json"
    legacy_path = Path(experiment_root) / "frozen" / "gates" / gate_name / "manifest.json"
    path = canonical_path if canonical_path.exists() else legacy_path
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
    canonical_stage_name = _canonical_stage_name(stage_name)
    for gate_name in FORMAL_STAGE_GATES.get(canonical_stage_name, []):
        manifests[gate_name] = load_gate_manifest(experiment_root, gate_name)
    return manifests


def load_gats_outputs(experiment_root):
    canonical_dir = Path(build_preparation_dir(experiment_root, "graph_calibrator"))
    legacy_dir = Path(experiment_root) / "frozen" / "gates" / GATS_GATE_NAME
    out_dir = canonical_dir if (canonical_dir / "manifest.json").exists() else legacy_dir
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


def _gats_manifest_matches_request(args, data, manifest, g0_context):
    if not isinstance(manifest, dict):
        return False
    if manifest.get("contract") != GATS_CONTRACT:
        return False
    frozen_manifest = g0_context.get("manifest", {}) or {}
    feature_manifest = frozen_manifest.get("feature_manifest", {}) or {}
    requested_backbone = str(getattr(args, "graph_backbone", getattr(args, "GNN_model", ""))).lower()
    manifest_backbone = str(manifest.get("gnn_backbone", "")).lower()
    if manifest_backbone and requested_backbone and manifest_backbone != requested_backbone:
        return False
    if manifest.get("input_g0_contract") != frozen_manifest.get("contract"):
        return False
    current_pred_sha = tensor_sha256(g0_context["outputs"]["pred"])
    if manifest.get("input_g0_outputs_sha256") != current_pred_sha:
        return False
    current_feature_path = str(feature_manifest.get("path", "") or "")
    if str(manifest.get("input_feature_path", "") or "") != current_feature_path:
        return False
    current_split = split_provenance(data)
    manifest_split = manifest.get("split_provenance", {}) or {}
    for split_name in ("train", "valid", "test"):
        current_meta = current_split.get(split_name, {})
        manifest_meta = manifest_split.get(split_name, {})
        if manifest_meta.get("size") != current_meta.get("size"):
            return False
        if manifest_meta.get("sha256") != current_meta.get("sha256"):
            return False
    return True


def build_or_load_faithful_gats(args, seed, data, experiment_root, g0_context):
    out_dir = gate_dir(experiment_root, GATS_GATE_NAME)
    existing_manifest = read_json(out_dir / "manifest.json", default=None)
    if (
        existing_manifest is not None
        and (out_dir / "outputs.pt").exists()
        and not getattr(args, "force_retrain_backbone", False)
        and _gats_manifest_matches_request(args, data, existing_manifest, g0_context)
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
            "canonical_task_name": "graph_calibration_prepare",
            "legacy_task_name_used": getattr(args, "legacy_task_name_used", None),
            "deprecated_cli_flags": list(getattr(args, "deprecated_cli_flags", [])),
            "stage_visibility": "public",
            "artifact_namespace": "preparation/graph_calibrator",
            "invocation": {
                "requested_task": getattr(args, "requested_experiment_task", getattr(args, "experiment_task", None)),
                "resolved_task": "graph_calibration_prepare",
            },
            "gate_name": GATS_GATE_NAME,
            "status": "available",
            "seed": int(seed),
            "source_impl": "trainer.GraphGATSCalibrator",
            "training_scope": "validation_only_calibration",
            "calibration_idx_sha256": tensor_sha256(valid_idx),
            "input_g0_contract": g0_context["manifest"].get("contract"),
            "input_g0_outputs_sha256": tensor_sha256(g0_outputs["pred"]),
            "input_feature_path": g0_context["manifest"].get("feature_manifest", {}).get("path"),
            "gnn_backbone": g0_context["manifest"].get("backbone"),
            "split_provenance": split_provenance(data),
        },
    )
    return load_gats_outputs(experiment_root)


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


def _js_divergence_from_probs(p, q, eps=1e-8):
    p_t = p.detach().cpu().float() if torch.is_tensor(p) else torch.tensor(p, dtype=torch.float32)
    q_t = q.detach().cpu().float() if torch.is_tensor(q) else torch.tensor(q, dtype=torch.float32)
    p_t = p_t.clamp_min(eps)
    p_t = p_t / p_t.sum(dim=-1, keepdim=True).clamp_min(eps)
    q_t = q_t.clamp_min(eps)
    q_t = q_t / q_t.sum(dim=-1, keepdim=True).clamp_min(eps)
    m_t = 0.5 * (p_t + q_t)
    kl_pm = torch.sum(p_t * (torch.log(p_t) - torch.log(m_t.clamp_min(eps))), dim=-1)
    kl_qm = torch.sum(q_t * (torch.log(q_t) - torch.log(m_t.clamp_min(eps))), dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def _semantic_backbone_to_lm_model(args):
    backbone = str(getattr(args, "semantic_encoder", getattr(args, "semantic_backbone", "auto"))).lower()
    if backbone == "auto":
        return str(getattr(args, "text_encoder", getattr(args, "LM_model", "roberta"))).lower()
    mapping = {
        "roberta": "roberta",
        "roberta_finetuned": "roberta_finetuned",
        "qwen3_peft": "qwen3_peft",
    }
    if backbone not in mapping:
        raise ValueError(
            f"--experiment_task semantic_encoder_finetune does not train --semantic_encoder {backbone}. "
            "Use qwen3_peft for real Qwen LoRA PEFT or pass --embedding_path for frozen embeddings."
        )
    return mapping[backbone]


def _semantic_command(args):
    keys = [
        "experiment_task",
        "dataset",
        "reset_split",
        "seeds",
        "semantic_encoder",
        "qwen_model_path",
        "qwen_trust_remote_code",
        "peft_rank",
        "peft_alpha",
        "lm_batch_size",
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


def _normalize_glance_advantage_target(advantage):
    if advantage.numel() == 0:
        return advantage
    centered = advantage - advantage.mean()
    scale = centered.std(unbiased=False).clamp_min(1e-6)
    return centered / scale


def _scale_glance_advantage_target(advantage):
    if advantage.numel() == 0:
        return advantage
    scale = advantage.std(unbiased=False).clamp_min(1e-6)
    return advantage / scale


def _glance_oracle_topk_membership(advantage, k):
    if advantage.numel() == 0:
        return torch.zeros_like(advantage, dtype=torch.float32)
    k = min(max(int(k), 1), int(advantage.numel()))
    target = torch.zeros_like(advantage, dtype=torch.float32)
    top_idx = torch.topk(advantage.detach(), k=k, largest=True, sorted=False).indices
    target[top_idx] = 1.0
    return target


def _glance_pairwise_ranking_loss(scores, targets):
    if scores.numel() < 2:
        return scores.sum() * 0.0
    score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
    pair_mask = torch.triu(torch.ones_like(target_diff, dtype=torch.bool), diagonal=1)
    pair_mask = pair_mask & (target_diff.abs() > 1e-6)
    if not bool(pair_mask.any().item()):
        return scores.sum() * 0.0
    signed_margin = torch.sign(target_diff[pair_mask]) * score_diff[pair_mask]
    pair_weight = target_diff[pair_mask].abs()
    loss = F.softplus(-signed_margin)
    return (loss * pair_weight).sum() / pair_weight.sum().clamp_min(1e-6)


def _safe_score_corr(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return None
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _safe_binary_score_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size or labels.size == 0:
        return {
            "positive_rate": 0.0,
            "mean_score": 0.0,
            "auroc": None,
            "auprc": None,
        }
    metrics = {
        "positive_rate": float(labels.mean()),
        "mean_score": float(scores.mean()),
        "auroc": None,
        "auprc": None,
    }
    if np.unique(labels).size < 2:
        return metrics
    try:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
    except Exception:
        metrics["auroc"] = None
    try:
        metrics["auprc"] = float(average_precision_score(labels, scores))
    except Exception:
        metrics["auprc"] = None
    return metrics


def _safe_advantage_router_metrics(advantage, scores, routed_mask=None):
    advantage = np.asarray(advantage, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if advantage.size != scores.size or advantage.size == 0:
        return {
            "positive_rate": 0.0,
            "mean_advantage": 0.0,
            "score_adv_corr": None,
            "auroc": None,
            "auprc": None,
        }
    labels = (advantage > 0.0).astype(np.int64)
    metrics = {
        "positive_rate": float(labels.mean()),
        "mean_advantage": float(advantage.mean()),
        "score_adv_corr": _safe_score_corr(scores, advantage),
        "auroc": None,
        "auprc": None,
    }


def _labeled_prefix_array(values, labeled_count):
    values_np = np.asarray(values)
    return values_np[: int(labeled_count)]
    if np.unique(labels).size >= 2:
        try:
            metrics["auroc"] = float(roc_auc_score(labels, scores))
        except Exception:
            metrics["auroc"] = None
        try:
            metrics["auprc"] = float(average_precision_score(labels, scores))
        except Exception:
            metrics["auprc"] = None
    if routed_mask is not None:
        routed_mask = np.asarray(routed_mask, dtype=bool).reshape(-1)
        if routed_mask.size == labels.size and routed_mask.any():
            routed_labels = labels[routed_mask]
            metrics["routed_positive_precision"] = float(routed_labels.mean())
            metrics["routed_mean_advantage"] = float(advantage[routed_mask].mean())
            positive_count = int(labels.sum())
            metrics["routed_positive_coverage"] = float(routed_labels.sum() / positive_count) if positive_count > 0 else 0.0
            budget = int(routed_mask.sum())
            oracle_idx = np.argsort(-advantage)[:budget]
            routed_idx = np.flatnonzero(routed_mask)
            metrics["routed_oracle_topk_overlap"] = float(
                len(set(routed_idx.tolist()).intersection(set(oracle_idx.tolist()))) / max(budget, 1)
            )
        else:
            metrics["routed_positive_precision"] = 0.0
            metrics["routed_mean_advantage"] = 0.0
            metrics["routed_positive_coverage"] = 0.0
            metrics["routed_oracle_topk_overlap"] = 0.0
    return metrics


def _infer_embedding_source_identity(path):
    path = Path(path)
    stem = path.stem.lower()
    name = path.name.lower()
    token = f"{stem} {name}"
    if "qwen" in token:
        return {
            "semantic_backbone": "qwen_cached",
            "lm_model": "qwen_cached",
            "source_identity": "qwen_cached_embedding",
        }
    if "roberta" in token:
        return {
            "semantic_backbone": "roberta_finetuned",
            "lm_model": "roberta-f",
            "source_identity": "roberta_finetuned_embedding",
        }
    return {
        "semantic_backbone": "external_embedding_unknown",
        "lm_model": "external_embedding_unknown",
        "source_identity": "external_embedding_unknown",
    }


def _directional_neighbor_views(semantic_embeddings, edge_index, edge_type):
    x_sem = semantic_embeddings.detach().cpu().float()
    num_nodes = int(x_sem.shape[0])
    zero = torch.zeros_like(x_sem)
    zero_counts = np.zeros(num_nodes, dtype=np.int64)
    if edge_index is None or edge_type is None:
        return {
            "ego": x_sem,
            "following": zero.clone(),
            "follower": zero.clone(),
            "count_following": zero_counts.copy(),
            "count_follower": zero_counts.copy(),
            "has_following": zero_counts.copy(),
            "has_follower": zero_counts.copy(),
        }
    edge_index_t = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
    edge_type_t = edge_type.detach().cpu().long() if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long)
    if edge_index_t.dim() != 2 or edge_index_t.size(0) != 2:
        raise ValueError("edge_index must be shaped [2, num_edges] for directional semantic views.")
    if edge_type_t.numel() != edge_index_t.size(1):
        raise ValueError("edge_type must align with edge_index for directional semantic views.")

    src = edge_index_t[0].numpy()
    dst = edge_index_t[1].numpy()
    rel = edge_type_t.numpy()
    following_bucket = [[] for _ in range(num_nodes)]
    follower_bucket = [[] for _ in range(num_nodes)]
    for s, d, r in zip(src.tolist(), dst.tolist(), rel.tolist()):
        if not (0 <= s < num_nodes and 0 <= d < num_nodes):
            continue
        if int(r) == 1:
            following_bucket[s].append(d)
        elif int(r) == 0:
            follower_bucket[d].append(s)

    def _pool(bucket):
        view_tensor = torch.zeros_like(x_sem)
        count_arr = np.zeros(num_nodes, dtype=np.int64)
        has_arr = np.zeros(num_nodes, dtype=np.int64)
        for node_idx in range(num_nodes):
            neighbors = sorted(set(int(n) for n in bucket[node_idx] if int(n) != node_idx))
            count_arr[node_idx] = len(neighbors)
            has_arr[node_idx] = 1 if neighbors else 0
            if neighbors:
                view_tensor[node_idx] = x_sem[torch.tensor(neighbors, dtype=torch.long)].mean(dim=0)
        return view_tensor, count_arr, has_arr

    following_view, count_following, has_following = _pool(following_bucket)
    follower_view, count_follower, has_follower = _pool(follower_bucket)
    return {
        "ego": x_sem,
        "following": following_view,
        "follower": follower_view,
        "count_following": count_following,
        "count_follower": count_follower,
        "has_following": has_following,
        "has_follower": has_follower,
    }


def _weighted_directional_neighbor_views(semantic_embeddings, edge_index, edge_type, include_2hop=False):
    x_sem = semantic_embeddings.detach().cpu().float()
    num_nodes = int(x_sem.shape[0])
    zero = torch.zeros_like(x_sem)
    zero_counts = np.zeros(num_nodes, dtype=np.int64)
    if edge_index is None or edge_type is None:
        outputs = {
            "ego": x_sem,
            "in_rel0": zero.clone(),
            "in_rel1": zero.clone(),
            "out_rel0": zero.clone(),
            "out_rel1": zero.clone(),
            "count_in_rel0": zero_counts.copy(),
            "count_in_rel1": zero_counts.copy(),
            "count_out_rel0": zero_counts.copy(),
            "count_out_rel1": zero_counts.copy(),
            "has_in_rel0": zero_counts.copy(),
            "has_in_rel1": zero_counts.copy(),
            "has_out_rel0": zero_counts.copy(),
            "has_out_rel1": zero_counts.copy(),
            "count_1hop": zero_counts.copy(),
            "count_2hop": zero_counts.copy(),
        }
        if include_2hop:
            for key in ("in_rel0_2hop", "in_rel1_2hop", "out_rel0_2hop", "out_rel1_2hop"):
                outputs[key] = zero.clone()
                outputs[f"count_{key}"] = zero_counts.copy()
                outputs[f"has_{key}"] = zero_counts.copy()
        return outputs
    base_views = _directional_neighbor_views(semantic_embeddings, edge_index, edge_type)
    edge_index_t = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
    edge_type_t = edge_type.detach().cpu().long() if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long)
    src = edge_index_t[0].numpy()
    dst = edge_index_t[1].numpy()
    rel = edge_type_t.numpy()
    rel_buckets = {
        "in_rel0": [[] for _ in range(num_nodes)],
        "in_rel1": [[] for _ in range(num_nodes)],
        "out_rel0": [[] for _ in range(num_nodes)],
        "out_rel1": [[] for _ in range(num_nodes)],
    }
    for s, d, r in zip(src.tolist(), dst.tolist(), rel.tolist()):
        if not (0 <= s < num_nodes and 0 <= d < num_nodes):
            continue
        if int(r) == 0:
            rel_buckets["in_rel0"][d].append(s)
            rel_buckets["out_rel0"][s].append(d)
        elif int(r) == 1:
            rel_buckets["in_rel1"][d].append(s)
            rel_buckets["out_rel1"][s].append(d)

    def _pool(bucket):
        view_tensor = torch.zeros_like(x_sem)
        count_arr = np.zeros(num_nodes, dtype=np.int64)
        has_arr = np.zeros(num_nodes, dtype=np.int64)
        for node_idx in range(num_nodes):
            neighbors = sorted(set(int(n) for n in bucket[node_idx] if int(n) != node_idx))
            count_arr[node_idx] = len(neighbors)
            has_arr[node_idx] = 1 if neighbors else 0
            if neighbors:
                view_tensor[node_idx] = x_sem[torch.tensor(neighbors, dtype=torch.long)].mean(dim=0)
        return view_tensor, count_arr, has_arr

    outputs = dict(base_views)
    for key in ("in_rel0", "in_rel1", "out_rel0", "out_rel1"):
        view_tensor, count_arr, has_arr = _pool(rel_buckets[key])
        outputs[key] = view_tensor
        outputs[f"count_{key}"] = count_arr
        outputs[f"has_{key}"] = has_arr
    outputs["count_1hop"] = np.asarray(
        [len(set(rel_buckets["in_rel0"][i]) | set(rel_buckets["in_rel1"][i]) | set(rel_buckets["out_rel0"][i]) | set(rel_buckets["out_rel1"][i])) for i in range(num_nodes)],
        dtype=np.int64,
    )
    outputs["count_2hop"] = zero_counts.copy()
    if include_2hop:
        for key in ("in_rel0", "in_rel1", "out_rel0", "out_rel1"):
            parent_bucket = rel_buckets[key]
            view_tensor = torch.zeros_like(x_sem)
            count_arr = np.zeros(num_nodes, dtype=np.int64)
            has_arr = np.zeros(num_nodes, dtype=np.int64)
            for node_idx in range(num_nodes):
                n1 = sorted(set(int(n) for n in parent_bucket[node_idx] if int(n) != node_idx))
                n1_set = set(n1)
                n2 = set()
                for neigh in n1:
                    for cand in parent_bucket[neigh]:
                        cand = int(cand)
                        if cand == node_idx or cand in n1_set:
                            continue
                        n2.add(cand)
                neighbors = sorted(n2)
                count_arr[node_idx] = len(neighbors)
                has_arr[node_idx] = 1 if neighbors else 0
                if neighbors:
                    view_tensor[node_idx] = x_sem[torch.tensor(neighbors, dtype=torch.long)].mean(dim=0)
            outputs[f"{key}_2hop"] = view_tensor
            outputs[f"count_{key}_2hop"] = count_arr
            outputs[f"has_{key}_2hop"] = has_arr
    return outputs


def _semantic_view_count_lookup(semantic_views, key, default_key=None):
    if key in semantic_views:
        return semantic_views[key]
    if default_key is not None and default_key in semantic_views:
        return semantic_views[default_key]
    first_value = next(iter(semantic_views.values()))
    return np.zeros_like(first_value[:, 0].detach().cpu().numpy(), dtype=np.int64)


def _direct_embedding_manifest(path):
    identity = _infer_embedding_source_identity(path)
    return {
        "contract": "direct_embedding_fallback_v2",
        "status": "completed_external_embedding",
        "embeddings_path": str(Path(path)),
        "source": "direct_embedding_path",
        **identity,
    }


def _normalize_semantic_payload(payload):
    payload_keys = []
    precomputed_views = None
    prompt_expert_bundle = None
    embeddings = payload
    if isinstance(payload, dict):
        payload_keys = sorted(str(key) for key in payload.keys())
        direct_views = {}
        for view_name in ("ego", "hop1", "hop2"):
            if view_name not in payload:
                continue
            view_value = payload[view_name]
            if not torch.is_tensor(view_value):
                try:
                    view_value = torch.as_tensor(view_value)
                except Exception:
                    continue
            if view_value.dim() != 2:
                continue
            direct_views[view_name] = view_value.detach().cpu().float()
        if len(direct_views) == 3:
            precomputed_views = direct_views
        expert_component_tensors = {}
        for component_name in ("ego", "graph_following", "graph_follower", "tweet", "conflict"):
            if component_name not in payload:
                continue
            component_value = payload[component_name]
            if not torch.is_tensor(component_value):
                try:
                    component_value = torch.as_tensor(component_value)
                except Exception:
                    continue
            if component_value.dim() != 2:
                continue
            expert_component_tensors[component_name] = component_value.detach().cpu().float()
        expert_scalar_tensors = {}
        for scalar_key in (
            "count_following",
            "count_follower",
            "has_following",
            "has_follower",
            "rt_ratio",
            "url_ratio",
            "hashtag_ratio",
        ):
            if scalar_key not in payload:
                continue
            scalar_value = payload[scalar_key]
            if not torch.is_tensor(scalar_value):
                try:
                    scalar_value = torch.as_tensor(scalar_value)
                except Exception:
                    continue
            if scalar_value.dim() == 2 and int(scalar_value.shape[1]) == 1:
                scalar_value = scalar_value.view(-1)
            if scalar_value.dim() != 1:
                continue
            expert_scalar_tensors[scalar_key] = scalar_value.detach().cpu().float().view(-1)
        payload_semantic_view_mode = str(payload.get("semantic_view_mode", "") or "").strip().lower()
        has_nonlegacy_expert_component = any(
            name in expert_component_tensors
            for name in ("graph_following", "graph_follower", "tweet", "conflict")
        )
        if has_nonlegacy_expert_component or expert_scalar_tensors or payload_semantic_view_mode == "prompt_expert_bundle_v1":
            active_components = payload.get("active_components")
            if active_components is None:
                active_components = list(expert_component_tensors.keys())
            elif isinstance(active_components, str):
                active_components = [active_components]
            else:
                active_components = [str(item) for item in active_components]
            prompt_expert_bundle = {
                "component_tensors": expert_component_tensors,
                "scalar_tensors": expert_scalar_tensors,
                "active_components": list(active_components),
                "semantic_view_mode": "prompt_expert_bundle_v1",
            }
        for candidate_key in ("embeddings", "features", "x"):
            if candidate_key in payload:
                embeddings = payload[candidate_key]
                break
        else:
            if precomputed_views is not None:
                embeddings = precomputed_views["ego"]
    if not torch.is_tensor(embeddings):
        embeddings = torch.as_tensor(embeddings)
    embeddings = embeddings.detach().cpu().float()
    if precomputed_views is not None:
        expected_nodes = int(embeddings.shape[0]) if embeddings.dim() == 2 else int(precomputed_views["ego"].shape[0])
        for view_name, view_tensor in precomputed_views.items():
            if view_tensor.dim() != 2:
                raise MissingFrozenArtifactError(
                    f"Semantic payload view {view_name} must be 2-D, got {tuple(view_tensor.shape)}."
                )
            if int(view_tensor.shape[0]) != expected_nodes:
                raise MissingFrozenArtifactError(
                    f"Semantic payload view {view_name} has {int(view_tensor.shape[0])} nodes, expected {expected_nodes}."
                )
    if prompt_expert_bundle is not None:
        expected_nodes = int(embeddings.shape[0]) if embeddings.dim() == 2 else None
        if expected_nodes is None:
            expert_components = prompt_expert_bundle["component_tensors"]
            if expert_components:
                expected_nodes = int(next(iter(expert_components.values())).shape[0])
        if expected_nodes is None:
            raise MissingFrozenArtifactError(
                "Prompt-expert semantic payload must provide either a 2-D embeddings tensor or at least one expert component tensor."
            )
        for component_name, component_tensor in prompt_expert_bundle["component_tensors"].items():
            if component_tensor.dim() != 2:
                raise MissingFrozenArtifactError(
                    f"Prompt-expert component {component_name} must be 2-D, got {tuple(component_tensor.shape)}."
                )
            if int(component_tensor.shape[0]) != expected_nodes:
                raise MissingFrozenArtifactError(
                    f"Prompt-expert component {component_name} has {int(component_tensor.shape[0])} nodes, expected {expected_nodes}."
                )
        for scalar_key, scalar_tensor in prompt_expert_bundle["scalar_tensors"].items():
            if scalar_tensor.dim() != 1:
                raise MissingFrozenArtifactError(
                    f"Prompt-expert scalar {scalar_key} must be 1-D, got {tuple(scalar_tensor.shape)}."
                )
            if int(scalar_tensor.shape[0]) != expected_nodes:
                raise MissingFrozenArtifactError(
                    f"Prompt-expert scalar {scalar_key} has {int(scalar_tensor.shape[0])} nodes, expected {expected_nodes}."
                )
    return embeddings, precomputed_views, prompt_expert_bundle, payload_keys


class GlanceRefinerMLP(nn.Module):
    """GLANCE-inspired post-hoc refiner over frozen GNN + semantic embeddings."""

    def __init__(self, input_dim, hidden_dim=128, activation="leakyrelu", dropout=0.1):
        super().__init__()
        activation = str(activation).lower()
        if activation == "leakyrelu":
            act = nn.LeakyReLU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == "elu":
            act = nn.ELU()
        else:
            raise ValueError(f"Unsupported refiner activation: {activation}")
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            act,
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, x):
        return self.net(x)


class GatedGlanceRefinerMLP(nn.Module):
    """Strict joint refiner with an explicit keep/change gate over base and semantic paths."""

    def __init__(self, input_dim, hidden_dim=128, activation="leakyrelu", dropout=0.1):
        super().__init__()
        activation = str(activation).lower()
        if activation == "leakyrelu":
            act_factory = nn.LeakyReLU
        elif activation == "relu":
            act_factory = nn.ReLU
        elif activation == "elu":
            act_factory = nn.ELU
        else:
            raise ValueError(f"Unsupported refiner activation: {activation}")
        self.shared = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            act_factory(),
            nn.Dropout(float(dropout)),
        )
        self.classifier = nn.Linear(int(hidden_dim), 2)
        self.gate_head = nn.Linear(int(hidden_dim), 1)

    def forward(self, x):
        hidden = self.shared(x)
        logits = self.classifier(hidden)
        gate_logits = self.gate_head(hidden).squeeze(-1)
        gate_prob = torch.sigmoid(gate_logits)
        return logits, gate_logits, gate_prob


class DirectionalGatedGlanceRefinerMLP(nn.Module):
    """Directional strict refiner with optional global semantic-view gates."""

    def __init__(
        self,
        z_gnn_dim,
        semantic_dim,
        use_presence_features=False,
        activation="leakyrelu",
        dropout=0.1,
    ):
        super().__init__()
        activation = str(activation).lower()
        if activation == "leakyrelu":
            act = nn.LeakyReLU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == "elu":
            act = nn.ELU()
        else:
            raise ValueError(f"Unsupported refiner activation: {activation}")
        self.z_gnn_dim = int(z_gnn_dim)
        self.semantic_dim = int(semantic_dim)
        self.use_presence_features = bool(use_presence_features)
        self.presence_dim = 4 if self.use_presence_features else 0
        self.view_gates = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        input_dim = self.z_gnn_dim + self.semantic_dim * 3 + self.presence_dim
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), 128),
            act,
            nn.Dropout(float(dropout)),
            nn.Linear(128, 2),
        )

    def forward(self, z_gnn, semantic_views, presence_features=None):
        gates = torch.sigmoid(self.view_gates).to(z_gnn.device)
        gated_views = [
            semantic_views[0] * gates[0],
            semantic_views[1] * gates[1],
            semantic_views[2] * gates[2],
        ]
        parts = [z_gnn] + gated_views
        if self.use_presence_features:
            if presence_features is None:
                raise ValueError("presence_features is required when use_presence_features=True")
            parts.append(presence_features)
        x = torch.cat(parts, dim=1)
        return self.net(x)


class WeightedDirectionalGlanceRefinerMLP(nn.Module):
    """Node-conditioned weighted directional refiner with fixed-width projected views."""

    def __init__(
        self,
        z_gnn_dim,
        semantic_dim,
        view_count,
        structural_dim=4,
        proj_dim=128,
        hidden_dim=128,
        activation="leakyrelu",
        dropout=0.1,
    ):
        super().__init__()
        activation = str(activation).lower()
        if activation == "leakyrelu":
            act_factory = nn.LeakyReLU
        elif activation == "relu":
            act_factory = nn.ReLU
        elif activation == "elu":
            act_factory = nn.ELU
        else:
            raise ValueError(f"Unsupported refiner activation: {activation}")
        self.z_gnn_dim = int(z_gnn_dim)
        self.semantic_dim = int(semantic_dim)
        self.view_count = int(view_count)
        self.proj_dim = int(proj_dim)
        self.structural_dim = int(structural_dim)
        self.view_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.semantic_dim, self.proj_dim),
                    act_factory(),
                    nn.Dropout(float(dropout)),
                )
                for _ in range(self.view_count)
            ]
        )
        gate_input_dim = self.z_gnn_dim + self.proj_dim + self.structural_dim
        self.gate_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(gate_input_dim, int(hidden_dim)),
                    act_factory(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden_dim), 1),
                )
                for _ in range(self.view_count)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.z_gnn_dim + self.proj_dim + self.structural_dim, int(hidden_dim)),
            act_factory(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, z_gnn, semantic_views, structural_features):
        projected = [projector(semantic_views[idx]) for idx, projector in enumerate(self.view_projectors)]
        gate_inputs = []
        for proj in projected:
            gate_inputs.append(torch.cat([z_gnn, proj, structural_features], dim=1))
        gate_logits = torch.cat([head(inp) for head, inp in zip(self.gate_heads, gate_inputs)], dim=1)
        gate_weights = torch.softmax(gate_logits, dim=1)
        fused = torch.zeros_like(projected[0])
        for idx, proj in enumerate(projected):
            fused = fused + proj * gate_weights[:, idx : idx + 1]
        x = torch.cat([z_gnn, fused, structural_features], dim=1)
        logits = self.classifier(x)
        return logits, gate_weights


class RelationAwareSemanticFusionMLP(nn.Module):
    """Project and fuse relation-aware semantic views before final classification."""

    def __init__(self, z_gnn_dim, semantic_dim, view_count, structural_dim=8, proj_dim=128, activation="leakyrelu", dropout=0.1):
        super().__init__()
        activation = str(activation).lower()
        if activation == "leakyrelu":
            act_factory = nn.LeakyReLU
        elif activation == "relu":
            act_factory = nn.ReLU
        elif activation == "elu":
            act_factory = nn.ELU
        else:
            raise ValueError(f"Unsupported refiner activation: {activation}")
        self.view_count = int(view_count)
        self.proj_dim = int(proj_dim)
        self.z_gnn_dim = int(z_gnn_dim)
        self.semantic_dim = int(semantic_dim)
        self.structural_dim = int(structural_dim)
        self.view_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.semantic_dim, self.proj_dim),
                    act_factory(),
                    nn.Dropout(float(dropout)),
                )
                for _ in range(self.view_count)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.z_gnn_dim + self.view_count * self.proj_dim + self.structural_dim, 128),
            act_factory(),
            nn.Dropout(float(dropout)),
            nn.Linear(128, 2),
        )

    def forward(self, z_gnn, semantic_views, structural_features):
        projected = []
        for idx, projector in enumerate(self.view_projectors):
            projected.append(projector(semantic_views[idx]))
        fused_semantic = torch.cat(projected, dim=1)
        x = torch.cat([z_gnn, fused_semantic, structural_features], dim=1)
        return self.classifier(x)


class MultiSourceRelationAwareSemanticFusionMLP(nn.Module):
    """Project and fuse an ordered list of semantic views before final classification."""

    def __init__(self, z_gnn_dim, semantic_dims, structural_dim=8, proj_dim=128, activation="leakyrelu", dropout=0.1):
        super().__init__()
        activation = str(activation).lower()
        if activation == "leakyrelu":
            act_factory = nn.LeakyReLU
        elif activation == "relu":
            act_factory = nn.ReLU
        elif activation == "elu":
            act_factory = nn.ELU
        else:
            raise ValueError(f"Unsupported refiner activation: {activation}")
        self.semantic_dims = [int(item) for item in semantic_dims]
        self.view_count = int(len(self.semantic_dims))
        self.proj_dim = int(proj_dim)
        self.z_gnn_dim = int(z_gnn_dim)
        self.structural_dim = int(structural_dim)
        self.view_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(view_dim), self.proj_dim),
                    act_factory(),
                    nn.Dropout(float(dropout)),
                )
                for view_dim in self.semantic_dims
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.z_gnn_dim + self.view_count * self.proj_dim + self.structural_dim, 128),
            act_factory(),
            nn.Dropout(float(dropout)),
            nn.Linear(128, 2),
        )

    def forward(self, z_gnn, semantic_views, structural_features):
        projected = []
        for idx, projector in enumerate(self.view_projectors):
            projected.append(projector(semantic_views[idx]))
        fused_semantic = torch.cat(projected, dim=1)
        x = torch.cat([z_gnn, fused_semantic, structural_features], dim=1)
        return self.classifier(x)


class PromptExpertBundleRefinerMLP(nn.Module):
    """Strict refiner head for the prompt-expert semantic bundle v1."""

    COMPONENT_ORDER = ("ego", "graph_following", "graph_follower", "tweet", "conflict")

    def __init__(
        self,
        z_gnn_dim,
        component_dims,
        proj_dim=256,
        hidden_dim=128,
        structural_dim=4,
        activation="leakyrelu",
        dropout=0.1,
    ):
        super().__init__()
        activation = str(activation).lower()
        if activation == "leakyrelu":
            act_factory = nn.LeakyReLU
        elif activation == "relu":
            act_factory = nn.ReLU
        elif activation == "elu":
            act_factory = nn.ELU
        else:
            raise ValueError(f"Unsupported refiner activation: {activation}")
        self.component_dims = {name: int(component_dims[name]) for name in self.COMPONENT_ORDER}
        self.z_gnn_dim = int(z_gnn_dim)
        self.proj_dim = int(proj_dim)
        self.structural_dim = int(structural_dim)
        self.projectors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.component_dims[name], self.proj_dim),
                    act_factory(),
                    nn.Dropout(float(dropout)),
                )
                for name in self.COMPONENT_ORDER
            }
        )
        self.graph_gate = nn.Linear(self.structural_dim, 2)
        total_proj_slots = 6  # ego + following + follower + fused_graph + tweet + conflict
        self.classifier = nn.Sequential(
            nn.Linear(self.z_gnn_dim + total_proj_slots * self.proj_dim + self.structural_dim, int(hidden_dim)),
            act_factory(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, z_gnn, semantic_views, structural_features):
        projected = {
            name: self.projectors[name](semantic_views[name])
            for name in self.COMPONENT_ORDER
        }
        gate_weights = torch.softmax(self.graph_gate(structural_features), dim=1)
        graph_fused = (
            gate_weights[:, 0:1] * projected["graph_following"]
            + gate_weights[:, 1:2] * projected["graph_follower"]
        )
        x = torch.cat(
            [
                z_gnn,
                projected["ego"],
                projected["graph_following"],
                projected["graph_follower"],
                graph_fused,
                projected["tweet"],
                projected["conflict"],
                structural_features,
            ],
            dim=1,
        )
        return self.classifier(x)


def _refiner_feature_lookup(refiner_features, batch_idx, semantic_view_mode, device):
    batch_idx = batch_idx.to(device=device)
    if isinstance(refiner_features, dict) and refiner_features.get("feature_kind") == "prompt_expert_bundle_v1":
        return {
            "feature_kind": "prompt_expert_bundle_v1",
            "z_gnn": refiner_features["z_gnn"][batch_idx].to(device),
            "semantic_views": {
                key: value[batch_idx].to(device)
                for key, value in refiner_features["semantic_views"].items()
                if torch.is_tensor(value) and value.dim() == 2
            },
            "structural_features": refiner_features["structural_features"][batch_idx].to(device),
        }
    feature_kind = refiner_features.get("feature_kind") if isinstance(refiner_features, dict) else None
    if feature_kind in {"directional_gated", "directional_weighted_gated", "directional_weighted_projected"}:
        return {
            "z_gnn": refiner_features["z_gnn"][batch_idx].to(device),
            "semantic_views": [view[batch_idx].to(device) for view in refiner_features["semantic_views"]],
            **(
                {"presence_features": refiner_features["presence_features"][batch_idx].to(device)}
                if "presence_features" in refiner_features
                else {}
            ),
            **(
                {"structural_features": refiner_features["structural_features"][batch_idx].to(device)}
                if "structural_features" in refiner_features
                else {}
            ),
        }
    if semantic_view_mode in {"relation_aware_1hop", "relation_aware_1hop_2hop"} and feature_kind == "relation_projected":
        return {
            "z_gnn": refiner_features["z_gnn"][batch_idx].to(device),
            "semantic_views": [view[batch_idx].to(device) for view in refiner_features["semantic_views"]],
            "structural_features": refiner_features["structural_features"][batch_idx].to(device),
        }
    return refiner_features[batch_idx].to(device)


class GlanceHomophilyQMLP(nn.Module):
    """Lightweight auxiliary classifier used to estimate GLANCE-style soft homophily."""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, x):
        return self.net(x)


def _select_semantic_train_idx(train_idx, limit, seed):
    train_idx = _as_long_cpu_tensor(train_idx)
    if int(limit or 0) <= 0 or int(limit) >= int(train_idx.numel()):
        return train_idx
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    order = torch.randperm(train_idx.numel(), generator=generator)
    return train_idx[order[: int(limit)]]


def run_semantic_finetune_seed(args, seed, data, experiment_root, run):
    stage_dir = build_preparation_dir(experiment_root, "semantic_encoder")
    code_provenance = capture_code_metadata(Path(__file__).resolve().parents[2])
    manifest = {
        "contract": "semantic_finetune_v1",
        "status": "started",
        "seed": int(seed),
        "canonical_task_name": "semantic_encoder_finetune",
        "semantic_backbone": getattr(args, "semantic_encoder", getattr(args, "semantic_backbone", "auto")),
        "dataset": getattr(args, "dataset", "unknown"),
        "dataset_path": str(data.get("dataset_path", "")),
        "training_scope": "train_idx_supervised_semantic_backbone",
        "notes": "Exploratory semantic finetune unless promoted by later full validation.",
        "command": _semantic_command(args),
        "code_commit": code_provenance["commit"],
        "code_provenance": code_provenance,
        "deprecated_cli_flags": list(getattr(args, "deprecated_cli_flags", [])),
        "stage_visibility": "public",
        "artifact_namespace": "preparation/semantic_encoder",
        "invocation": {
            "requested_task": getattr(args, "requested_experiment_task", getattr(args, "experiment_task", None)),
            "resolved_task": "semantic_encoder_finetune",
        },
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
        return {"stage": "semantic_encoder_finetune", "stage_dir": str(stage_dir), "metrics": metrics}
    except Exception as exc:
        manifest.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        write_json(stage_dir / "manifest.json", manifest)
        raise


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


# Compatibility rebinding: active preparation and semantic owners now live in
# dedicated modules. Keep legacy names stable for StageRunner while routing
# runtime behavior through the extracted implementations.
_resolve_device = _runtime_resolve_device
_set_cuda_device = _runtime_set_cuda_device
_reset_cuda_peak_memory_stats = _runtime_reset_cuda_peak_memory_stats
_max_cuda_memory_allocated = _runtime_max_cuda_memory_allocated
load_frozen_g0 = _prep_load_frozen_g0
build_or_load_frozen_g0 = _prep_build_or_load_frozen_g0
build_or_load_faithful_gats = _prep_build_or_load_faithful_gats
require_stage_gates = _prep_require_stage_gates
load_gats_outputs = _prep_load_gats_outputs
run_semantic_finetune_seed = _semantic_run_semantic_finetune_seed
_classification_metrics_from_logits = _semantic_classification_metrics_from_logits


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
        self.execution_stage = resolve_stage_name(getattr(args, "execution_task", args.stage))
        self.legacy_execution_stage = _legacy_stage_name(self.execution_stage)
        self.requested_task = getattr(args, "requested_experiment_task", getattr(args, "experiment_task", self.requested_stage))

    def ensure_backbone_context(self):
        if self.base_context is not None:
            return self.base_context
        external_root = getattr(self.args, "external_frozen_g0_root", None)
        if self.execution_stage == "joint_router_refinement" and external_root:
            if not getattr(self.args, "joint_refiner_embedding_path", None):
                raise MissingFrozenArtifactError(
                    "joint_router_refinement now requires the current run's own preparation/graph_detector artifact. "
                    "Cross-root --external_frozen_g0_root reuse is only allowed together with "
                    "--joint_refiner_embedding_path for refiner-only semantic override ablations."
                )
        frozen_root = Path(external_root) if external_root else self.experiment_root
        frozen = load_frozen_g0(frozen_root)
        self._validate_external_frozen_g0(frozen, frozen_root)
        gnn_outputs = frozen["outputs"]
        graph_bundle = self._resolve_stage_graph_bundle(frozen_root, frozen)
        self.data["edge_index"] = graph_bundle["edge_index"]
        self.data["edge_type"] = graph_bundle["edge_type"]
        self.base_context = {
            "frozen_g0": frozen,
            "runtime_dir": frozen["dir"],
            "gnn_outputs": gnn_outputs,
            "source_experiment_root": str(frozen_root),
            "graph_bundle": graph_bundle,
        }
        return self.base_context

    def _strict_joint_feature_manifest(self):
        context = self.ensure_backbone_context()
        manifest = context["frozen_g0"].get("manifest", {}) or {}
        feature_manifest = manifest.get("feature_manifest", {}) or {}
        feature_path = feature_manifest.get("path")
        if not feature_path:
            raise MissingFrozenArtifactError(
                "joint_router_refinement requires graph_detector_prepare to record feature_manifest.path in the "
                "current run's preparation/graph_detector manifest."
            )
        return feature_manifest, Path(feature_path)

    def _validate_external_frozen_g0(self, frozen, frozen_root):
        manifest = frozen.get("manifest", {}) or {}
        node_manifest = manifest.get("node_id_manifest", {}) or {}
        split_manifest = manifest.get("split_provenance", {}) or {}
        current_labels_sha = tensor_sha256(self.data["labels"])
        if int(node_manifest.get("num_nodes", -1)) != int(len(self.labels)):
            raise MissingFrozenArtifactError(
                f"External frozen_g0 at {frozen_root} has num_nodes={node_manifest.get('num_nodes')} "
                f"but current dataset has {len(self.labels)}."
            )
        if node_manifest.get("labels_sha256") and str(node_manifest.get("labels_sha256")) != str(current_labels_sha):
            raise MissingFrozenArtifactError(
                f"External frozen_g0 at {frozen_root} has mismatched label hash."
            )
        expected_split = split_provenance(self.data)
        for split_name in ("train", "valid", "test"):
            current = expected_split.get(split_name, {})
            external = split_manifest.get(split_name, {})
            if external.get("size") is not None and int(external.get("size")) != int(current.get("size", -1)):
                raise MissingFrozenArtifactError(
                    f"External frozen_g0 at {frozen_root} has mismatched {split_name} size."
                )
            if external.get("sha256") and str(external.get("sha256")) != str(current.get("sha256")):
                raise MissingFrozenArtifactError(
                    f"External frozen_g0 at {frozen_root} has mismatched {split_name} split hash."
                )
        backbone = str(manifest.get("backbone", "")).lower()
        requested_backbone = str(getattr(self.args, "GNN_model", "")).lower()
        if backbone and requested_backbone and backbone != requested_backbone:
            raise MissingFrozenArtifactError(
                f"External frozen_g0 at {frozen_root} uses backbone={backbone}, "
                f"but the current run requests backbone={requested_backbone}."
            )

    def _resolve_stage_graph_bundle(self, frozen_root, frozen):
        override_paths = _resolve_optional_graph_override_paths(self.args)
        source_stage = None
        if override_paths is not None:
            edge_index_path = override_paths["edge_index_path"]
            edge_type_path = override_paths["edge_type_path"]
            if not edge_index_path.exists() or not edge_type_path.exists():
                raise MissingFrozenArtifactError(
                    "External graph override paths must both exist before graph-aware refiner stages can run."
                )
            edge_index = safe_torch_load(edge_index_path, map_location="cpu")
            edge_type = safe_torch_load(edge_type_path, map_location="cpu")
            source = "explicit_external_graph_paths"
            source_root = str(edge_index_path.parent)
            source_stage = "external_override"
        else:
            frozen_dir = Path(frozen["dir"])
            pruned_edge_index_path = frozen_dir / "pruned_edge_index.pt"
            pruned_edge_type_path = frozen_dir / "pruned_edge_type.pt"
            if pruned_edge_index_path.exists() and pruned_edge_type_path.exists():
                edge_index = safe_torch_load(pruned_edge_index_path, map_location="cpu")
                edge_type = safe_torch_load(pruned_edge_type_path, map_location="cpu")
                source = "frozen_g0_pruned_graph"
                source_root = str(frozen_dir)
                source_stage = "graph_detector_prepare"
            else:
                edge_index = self.data.get("edge_index")
                edge_type = self.data.get("edge_type")
                source = "dataset_original_graph"
                source_root = str(frozen_root)
                source_stage = "dataset"

        if edge_index is None or edge_type is None:
            raise MissingFrozenArtifactError("Graph-aware stage requires edge_index and edge_type.")
        edge_index = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
        edge_type = edge_type.detach().cpu().long() if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long)
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise MissingFrozenArtifactError("Resolved graph edge_index must have shape [2, num_edges].")
        if edge_type.numel() != edge_index.size(1):
            raise MissingFrozenArtifactError("Resolved graph edge_type must align with edge_index.")
        relation_cardinality = int(edge_type.max().item()) + 1 if edge_type.numel() else 0
        num_nodes = int(getattr(self, "graph_node_count", len(self.labels)))
        num_edges = int(edge_index.size(1))
        return {
            "edge_index": edge_index.contiguous(),
            "edge_type": edge_type.contiguous(),
            "source": source,
            "source_root": source_root,
            "source_stage": source_stage,
            "graph_input_provenance": {
                "edge_index_sha256": tensor_sha256(edge_index),
                "edge_type_sha256": tensor_sha256(edge_type),
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "relation_cardinality": relation_cardinality,
                "source": source,
                "source_root": source_root,
                "source_stage": source_stage,
                "dataset_node_count_match": bool(num_nodes == int(getattr(self, "graph_node_count", len(self.labels)))),
            },
        }

    def _active_graph_tensors(self):
        context = self.ensure_backbone_context()
        graph_bundle = context.get("graph_bundle", {})
        edge_index = graph_bundle.get("edge_index")
        edge_type = graph_bundle.get("edge_type")
        if edge_index is None or edge_type is None:
            raise MissingFrozenArtifactError("Active graph bundle is missing edge tensors.")
        return edge_index, edge_type

    def _load_semantic_finetune_artifact(self):
        stage_dirs = [
            self.experiment_root / "preparation" / "semantic_encoder",
            self.experiment_root / "stages" / "semantic_encoder_finetune",
            self.experiment_root / "stages" / "semantic_finetune",
        ]
        stage_dir = None
        manifest_path = None
        embeddings_path = None
        outputs_path = None
        for candidate_dir in stage_dirs:
            candidate_manifest = candidate_dir / "manifest.json"
            candidate_embeddings = candidate_dir / "embeddings.pt"
            candidate_outputs = candidate_dir / "outputs.pt"
            if candidate_manifest.exists() and candidate_embeddings.exists() and candidate_outputs.exists():
                stage_dir = candidate_dir
                manifest_path = candidate_manifest
                embeddings_path = candidate_embeddings
                outputs_path = candidate_outputs
                break
        if stage_dir is None:
            fallback_path = getattr(self.args, "embedding_path", None) or getattr(self.args, "emb_path", None) or getattr(self.args, "g0_feature_path", None)
            if not fallback_path:
                raise MissingFrozenArtifactError(
                    "Strict GLANCE stages require same-seed semantic_encoder_finetune artifacts or an explicit "
                    "--embedding_path tensor. Implicit seed_1 fallback is disabled."
                )
            fallback_path = Path(fallback_path)
            if not fallback_path.exists():
                raise MissingFrozenArtifactError(
                    "Strict GLANCE stages require semantic_encoder_finetune artifacts or an explicit "
                    f"--embedding_path tensor. Resolved path does not exist: {fallback_path}."
                )
            embeddings = safe_torch_load(fallback_path, map_location="cpu")
            if not torch.is_tensor(embeddings) or embeddings.dim() != 2:
                raise MissingFrozenArtifactError(
                    f"Fallback semantic embedding at {fallback_path} must be a 2D tensor."
                )
            if int(embeddings.shape[0]) != int(len(self.labels)):
                raise MissingFrozenArtifactError(
                    "Fallback semantic embedding does not align with the current dataset node count."
                )
            return {
                "dir": fallback_path.parent,
                "manifest": _direct_embedding_manifest(fallback_path),
                "embeddings": embeddings.detach().cpu().float(),
                "outputs": None,
            }
        manifest = read_json(manifest_path, default={}) or {}
        semantic_backbone = str(manifest.get("semantic_backbone", "")).lower()
        if semantic_backbone not in {"roberta_finetuned", "roberta-f"}:
            raise MissingFrozenArtifactError(
                "glance_oracle_refine requires semantic_finetune artifacts produced by "
                "--semantic_backbone roberta_finetuned."
            )
        embeddings = safe_torch_load(embeddings_path, map_location="cpu")
        outputs = safe_torch_load(outputs_path, map_location="cpu")
        if not torch.is_tensor(embeddings) or embeddings.dim() != 2:
            raise MissingFrozenArtifactError(
                f"semantic_finetune embeddings at {embeddings_path} must be a 2D tensor."
            )
        if int(embeddings.shape[0]) != int(len(self.labels)):
            raise MissingFrozenArtifactError(
                "semantic_finetune embeddings do not align with the current dataset node count."
            )
        return {
            "dir": stage_dir,
            "manifest": manifest,
            "embeddings": embeddings.detach().cpu().float(),
            "outputs": outputs,
        }

    def _load_glance_semantic_source_bundle(self):
        feature_manifest, feature_path = self._strict_joint_feature_manifest()
        override_path = getattr(self.args, "joint_refiner_embedding_path", None)
        requested_path = (
            getattr(self.args, "embedding_path", None)
            or getattr(self.args, "emb_path", None)
            or getattr(self.args, "g0_feature_path", None)
        )
        if override_path:
            emb_path = Path(override_path)
            if requested_path:
                requested_path = Path(requested_path)
                if requested_path != feature_path:
                    raise MissingFrozenArtifactError(
                        "joint_router_refinement keeps backbone provenance pinned to the current run's "
                        "graph_detector_prepare artifact even when --joint_refiner_embedding_path is used. "
                        f"Requested --embedding_path {requested_path} does not match frozen_g0 feature_manifest.path {feature_path}."
                    )
            provenance_binding = "joint_refiner_override_same_backbone"
            manifest_source = "joint_refiner_embedding_override"
            semantic_override_role = "refiner_only"
        else:
            if requested_path:
                requested_path = Path(requested_path)
                if requested_path != feature_path:
                    raise MissingFrozenArtifactError(
                        "joint_router_refinement requires semantic input provenance to match the current run's "
                        "graph_detector_prepare artifact. "
                        f"Requested --embedding_path {requested_path} does not match frozen_g0 feature_manifest.path {feature_path}."
                    )
            emb_path = feature_path
            provenance_binding = "strict_same_root_graph_detector_prepare"
            manifest_source = "graph_detector_prepare_feature_manifest"
            semantic_override_role = "shared_backbone_and_refiner"
        if not emb_path.exists():
            raise MissingFrozenArtifactError(
                "joint_router_refinement could not find the requested semantic embedding tensor at "
                f"{emb_path}."
            )
        payload = safe_torch_load(emb_path, map_location="cpu")
        embeddings, precomputed_views, prompt_expert_bundle, payload_keys = _normalize_semantic_payload(payload)
        if embeddings.dim() != 2:
            raise MissingFrozenArtifactError(
                f"joint_router_refinement expects --embedding_path to contain a 2-D tensor, got {tuple(embeddings.shape)}."
            )
        if int(embeddings.shape[0]) != int(len(self.labels)):
            raise MissingFrozenArtifactError(
                "joint_router_refinement semantic embeddings do not align with the current dataset node count."
            )
        if precomputed_views is not None:
            for view_name, view_tensor in precomputed_views.items():
                if int(view_tensor.shape[0]) != int(len(self.labels)):
                    raise MissingFrozenArtifactError(
                        f"joint_router_refinement semantic view {view_name} does not align with the current dataset node count."
                    )
        if prompt_expert_bundle is not None:
            semantic_view_mode = "prompt_expert_bundle_v1"
        elif precomputed_views is not None:
            semantic_view_mode = "precomputed_prompt_views"
        else:
            semantic_view_mode = "legacy_inbound_khop"
        primary = {
            "dir": emb_path.parent,
            "manifest": {
                **_direct_embedding_manifest(emb_path),
                "source": manifest_source,
                "feature_manifest": feature_manifest,
                "provenance_binding": provenance_binding,
                "semantic_override_role": semantic_override_role,
                "payload_keys": payload_keys,
                "semantic_view_mode": semantic_view_mode,
                "active_components": list(prompt_expert_bundle.get("active_components", [])) if prompt_expert_bundle else [],
            },
            "embeddings": embeddings,
            "precomputed_views": precomputed_views,
            "prompt_expert_bundle": prompt_expert_bundle,
            "semantic_view_mode": semantic_view_mode,
            "outputs": None,
        }
        bundle = {
            "mode": "single_source",
            "primary": primary,
            "secondary": None,
            "sources": [primary],
        }
        return bundle

    def _strict_glance_train_idx(self, train_idx, cap=3000):
        train_idx = np.asarray(train_idx, dtype=np.int64).reshape(-1)
        cap = int(cap)
        if cap <= 0 or train_idx.size <= cap:
            return train_idx
        rng = np.random.default_rng(int(self.seed))
        selected = np.sort(rng.choice(train_idx, size=cap, replace=False))
        return selected.astype(np.int64)

    def _strict_glance_original_node_features(self):
        context = self.ensure_backbone_context()
        frozen_manifest = context["frozen_g0"].get("manifest", {}) or {}
        feature_manifest = frozen_manifest.get("feature_manifest", {}) or {}
        checkpoint = safe_torch_load(context["frozen_g0"]["checkpoint_path"], map_location="cpu")
        input_pipeline = checkpoint.get("input_pipeline", {}) if isinstance(checkpoint, dict) else {}
        checkpoint_feature_manifest = input_pipeline.get("feature_manifest", {}) or {}
        feature_path = checkpoint_feature_manifest.get("path") or feature_manifest.get("path")
        if not feature_path:
            raise MissingFrozenArtifactError(
                "glance_joint_router_refine requires frozen_g0 to record an original node-feature path. "
                "The current frozen_g0 artifact does not contain input_pipeline.feature_manifest.path."
            )
        feature_path = Path(feature_path)
        if not feature_path.exists():
            raise MissingFrozenArtifactError(
                f"glance_joint_router_refine could not read the frozen_g0 original node features at {feature_path}."
            )
        if str(getattr(self, "graph_data_variant", "labeled")).lower() == "full_graph_support":
            concat_bundle = _runtime_concat_full_graph_support_embeddings(self.args, self.data, feature_path)
            raw_features = concat_bundle["features"].detach().cpu().float()
        else:
            raw_features = safe_torch_load(feature_path, map_location="cpu")
            if isinstance(raw_features, dict):
                for key in ("embeddings", "features", "x"):
                    if key in raw_features:
                        raw_features = raw_features[key]
                        break
            if not torch.is_tensor(raw_features):
                raw_features = torch.tensor(raw_features)
            raw_features = raw_features.detach().cpu().float()
        if raw_features.dim() != 2:
            raise MissingFrozenArtifactError(
                f"glance_joint_router_refine expects original node features to be 2-D, got {tuple(raw_features.shape)}."
            )
        expected_nodes = int(getattr(self, "graph_node_count", len(self.labels)))
        if int(raw_features.shape[0]) != expected_nodes:
            raise MissingFrozenArtifactError(
                "glance_joint_router_refine original node features do not align with the current dataset node count."
            )
        return {
            "features": raw_features,
            "path": str(feature_path),
            "feature_manifest": checkpoint_feature_manifest or feature_manifest,
        }

    def _strict_glance_backbone_input_features(self):
        context = self.ensure_backbone_context()
        checkpoint = safe_torch_load(context["frozen_g0"]["checkpoint_path"], map_location="cpu")
        input_pipeline = checkpoint.get("input_pipeline", {}) if isinstance(checkpoint, dict) else {}
        feature_manifest = input_pipeline.get("feature_manifest", {}) or context["frozen_g0"].get("manifest", {}).get("feature_manifest", {})
        feature_args = SimpleNamespace(**vars(self.args))
        feature_path = feature_manifest.get("path")
        if feature_path:
            feature_args.emb_path = str(feature_path)
            feature_args.g0_feature_path = str(feature_path)
        if feature_manifest.get("projected_dim") is not None:
            feature_args.phase_a_project_dim = int(feature_manifest["projected_dim"])
        if feature_manifest.get("projector") is not None:
            feature_args.phase_a_projector = str(feature_manifest["projector"])
        feature_args.peft = False
        feature_bundle = resolve_g0_feature_bundle(feature_args, self.data)
        projected = feature_bundle["features"].detach().cpu().float()
        raw = feature_bundle["raw_features"].detach().cpu().float()
        input_adapter_payload = checkpoint.get("input_adapter", {}) if isinstance(checkpoint, dict) else {}
        if bool(input_adapter_payload.get("enabled", False)):
            adapter_config = input_adapter_payload.get("config", {}) or {}
            adapter_state = input_adapter_payload.get("state_dict")
            if adapter_state is None:
                raise MissingFrozenArtifactError(
                    "glance_joint_router_refine expected frozen_g0 checkpoint.pt to include input_adapter.state_dict when input_adapter.enabled is true."
                )
            adapter = PhaseAInputAdapter(
                raw_dim=int(raw.shape[1]),
                projected_dim=int(projected.shape[1]),
                rank=int(adapter_config.get("rank", 8)),
                alpha=float(adapter_config.get("alpha", 16.0)),
            )
            adapter.load_state_dict(adapter_state)
            adapter.eval()
            with torch.no_grad():
                projected = adapter(projected, raw).detach().cpu().float()
        return projected

    def _fit_strict_glance_q_probs(self, node_features, train_idx, valid_idx):
        train_idx = np.asarray(train_idx, dtype=np.int64).reshape(-1)
        valid_idx = np.asarray(valid_idx, dtype=np.int64).reshape(-1)
        if train_idx.size == 0:
            raise MissingFrozenArtifactError("glance_joint_router_refine requires a non-empty strict train split for Q fitting.")
        x_all = node_features.detach().cpu().float().numpy()
        y_all = np.asarray(self.labels, dtype=np.int64)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_all[train_idx])
        x_full = scaler.transform(x_all)
        classes = np.unique(y_all[train_idx])
        if classes.size < 2:
            q_probs = np.zeros((x_all.shape[0], 2), dtype=np.float32)
            q_probs[:, int(classes[0])] = 1.0
            q_logits = torch.log(torch.tensor(q_probs, dtype=torch.float32).clamp_min(1e-8))
            classifier_payload = {
                "type": "constant",
                "class": int(classes[0]),
            }
        else:
            q_device = self.device if isinstance(self.device, torch.device) else torch.device("cpu")
            x_train_t = torch.tensor(x_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_all[train_idx], dtype=torch.long)
            x_full_t = torch.tensor(x_full, dtype=torch.float32)
            valid_labels_t = (
                torch.tensor(y_all[valid_idx], dtype=torch.long)
                if valid_idx.size > 0
                else torch.empty(0, dtype=torch.long)
            )
            class_counts = np.bincount(y_train_t.numpy(), minlength=2).astype(np.float32)
            class_weights = np.zeros(2, dtype=np.float32)
            nonzero = class_counts > 0
            class_weights[nonzero] = float(y_train_t.numel()) / (2.0 * class_counts[nonzero])
            q_model = GlanceHomophilyQMLP(
                input_dim=int(x_train_t.shape[1]),
                hidden_dim=128,
                dropout=0.1,
            ).to(q_device)
            optimizer = torch.optim.AdamW(q_model.parameters(), lr=1e-3, weight_decay=1e-4)
            loss_fn = CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=q_device))
            batch_size = max(32, min(256, int(x_train_t.shape[0])))
            max_epochs = 50
            patience = 5
            generator = torch.Generator()
            generator.manual_seed(int(self.seed))
            train_loader = DataLoader(
                TensorDataset(x_train_t, y_train_t),
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
            )
            best_state = None
            best_epoch = 0
            best_score = float("inf")
            wait = 0
            final_train_loss = float("inf")

            for epoch in range(max_epochs):
                q_model.train()
                train_loss_sum = 0.0
                train_count = 0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(q_device)
                    batch_y = batch_y.to(q_device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = q_model(batch_x)
                    loss = loss_fn(logits, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss_sum += float(loss.detach().cpu().item()) * int(batch_y.numel())
                    train_count += int(batch_y.numel())
                final_train_loss = train_loss_sum / max(train_count, 1)

                q_model.eval()
                with torch.no_grad():
                    logits_full_t = q_model(x_full_t.to(q_device)).detach().cpu()
                if valid_idx.size > 0:
                    valid_loss = float(
                        F.cross_entropy(logits_full_t[valid_idx], valid_labels_t).detach().cpu().item()
                    )
                    selection_score = valid_loss
                else:
                    valid_loss = final_train_loss
                    selection_score = final_train_loss

                if selection_score < best_score:
                    best_score = selection_score
                    best_epoch = int(epoch + 1)
                    best_state = {key: value.detach().cpu().clone() for key, value in q_model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break

            if best_state is None:
                raise MissingFrozenArtifactError("glance_joint_router_refine failed to train the auxiliary Q MLP.")
            q_model.load_state_dict(best_state)
            q_model.eval()
            with torch.no_grad():
                q_logits = q_model(x_full_t.to(q_device)).detach().cpu()
            q_probs = torch.softmax(q_logits, dim=1).numpy().astype(np.float32)
            classifier_payload = {
                "type": "mlp",
                "input_dim": int(x_train.shape[1]),
                "hidden_dim": 128,
                "dropout": 0.1,
                "batch_size": int(batch_size),
                "max_epochs": int(max_epochs),
                "patience": int(patience),
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "best_epoch": int(best_epoch),
                "final_train_loss": float(final_train_loss),
                "best_selection_loss": float(best_score),
                "class_weighting": "inverse_frequency_balanced",
            }
        temperature = 1.0
        if valid_idx.size > 0 and len(np.unique(y_all[valid_idx])) > 1:
            temperature = fit_temperature_scaling(q_logits[valid_idx], y_all[valid_idx])
        q_probs_cal = torch.softmax(q_logits / float(temperature), dim=1).detach().cpu().float()
        return {
            "q_probs": q_probs_cal,
            "temperature": float(temperature),
            "classifier": classifier_payload,
            "scaler": {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            },
            "train_count": int(train_idx.size),
            "valid_count": int(valid_idx.size),
        }

    def _strict_glance_mc_dropout_uncertainty(self, node_features):
        context = self.ensure_backbone_context()
        checkpoint = safe_torch_load(context["frozen_g0"]["checkpoint_path"], map_location="cpu")
        model_state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        model_config = dict((checkpoint.get("model_config") or {})) if isinstance(checkpoint, dict) else {}
        if not model_state or not model_config:
            raise MissingFrozenArtifactError(
                "glance_joint_router_refine requires frozen_g0 checkpoint.pt to include model weights and model_config for MC-dropout uncertainty."
            )
        device = self.device
        edge_index, edge_type = self._active_graph_tensors()
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
        x = node_features.detach().cpu().float().to(device)
        mc_config = dict(model_config)
        mc_config["device"] = device
        model = build_GNN_model(mc_config)
        model.load_state_dict(model_state)
        model.to(device)
        logits_list = []
        with torch.no_grad():
            for _ in range(5):
                model.train()
                outputs = model.forward_outputs(x, edge_index, edge_type)
                logits_list.append(outputs["logits"].detach().cpu().float())
        logits_stack = torch.stack(logits_list, dim=0)
        probs_stack = torch.softmax(logits_stack, dim=2)
        uncertainty = probs_stack.var(dim=0).sum(dim=1).detach().cpu().float()
        return {
            "uncertainty": uncertainty,
            "num_passes": 5,
            "source": "mc_dropout",
            "logits_shape": [int(item) for item in logits_stack.shape],
        }

    def _build_strict_glance_router_features(
        self,
        logits_gnn,
        p_gnn,
        z_gnn,
        original_node_features,
        q_bundle,
        mc_bundle,
        calibration_temperature=1.0,
    ):
        edge_index, edge_type = self._active_graph_tensors()
        try:
            feature_bundle = build_reliability_router_feature_bundle(
                logits_gnn=logits_gnn,
                z_gnn=z_gnn,
                original_node_features=original_node_features,
                q_probs=q_bundle["q_probs"],
                uncertainty=mc_bundle["uncertainty"],
                edge_index=edge_index,
                edge_type=edge_type,
                calibration_temperature=calibration_temperature,
            )
        except ValueError as exc:
            raise MissingFrozenArtifactError(str(exc)) from exc
        feature_bundle["full_feature_bundle"]["used_edge_type"] = edge_type is not None
        feature_bundle["full_feature_bundle"]["used_q_probs"] = True
        feature_bundle["full_feature_bundle"]["used_mc_dropout_uncertainty"] = True
        feature_bundle["full_feature_bundle"]["node_feature_dim"] = int(original_node_features.shape[1])
        feature_bundle["full_feature_bundle"]["z_gnn_dim"] = int(z_gnn.shape[1])
        feature_bundle["full_feature_bundle"]["feature_dim"] = int(feature_bundle["features"].shape[1])
        return feature_bundle

    def _glance_refiner_source_stage(self):
        source = str(getattr(self.args, "glance_refiner_source", "oracle")).lower()
        if source == "oracle":
            return "glance_oracle_refine"
        if source == "full_graph":
            return "glance_full_graph_refine"
        raise ValueError(f"Unsupported --glance_refiner_source: {source}")

    def _load_glance_refiner_artifact(self):
        stage_name = self._glance_refiner_source_stage()
        manifest = self._require_stage_json(stage_name, "manifest")
        metrics = self._require_stage_json(stage_name, "metrics")
        outputs = self._require_stage_tensor(stage_name, "outputs")
        checkpoint = self._require_stage_tensor(stage_name, "checkpoint")
        per_node_rows = self._require_stage_jsonl(stage_name, "per_node_test")
        analysis_summary = self._read_stage_json(stage_name, "analysis_summary")
        return {
            "stage_name": stage_name,
            "manifest": manifest,
            "metrics": metrics,
            "outputs": outputs,
            "checkpoint": checkpoint,
            "per_node_rows": per_node_rows,
            "analysis_summary": analysis_summary,
        }

    def _oracle_wrong_node_masks(self, base_pred):
        base_pred = np.asarray(base_pred, dtype=np.int64).reshape(-1)
        wrong = base_pred != self.labels
        return {
            "train": wrong & self.train_mask,
            "valid": wrong & self.val_mask,
            "test": wrong & self.test_mask,
            "all": wrong,
        }

    def _build_k_hop_semantic_views(self, semantic_embeddings, edge_index):
        direct_views = None
        if isinstance(semantic_embeddings, dict):
            maybe_direct = {}
            for view_name in ("ego", "hop1", "hop2"):
                if view_name not in semantic_embeddings:
                    maybe_direct = None
                    break
                view_tensor = semantic_embeddings[view_name]
                if not torch.is_tensor(view_tensor):
                    view_tensor = torch.as_tensor(view_tensor)
                maybe_direct[view_name] = view_tensor.detach().cpu().float()
            if maybe_direct is not None:
                direct_views = maybe_direct
                x_sem = direct_views["ego"]
            else:
                base_tensor = None
                for key in ("embeddings", "features", "x"):
                    if key in semantic_embeddings:
                        base_tensor = semantic_embeddings[key]
                        break
                if base_tensor is None:
                    raise ValueError("semantic_embeddings dict must provide ego/hop1/hop2 or embeddings/features/x.")
                if not torch.is_tensor(base_tensor):
                    base_tensor = torch.as_tensor(base_tensor)
                x_sem = base_tensor.detach().cpu().float()
        else:
            x_sem = semantic_embeddings.detach().cpu().float()
        num_nodes = int(x_sem.shape[0])
        if edge_index is None:
            zero = torch.zeros_like(x_sem)
            counts = np.zeros(num_nodes, dtype=np.int64)
            return {
                "ego": x_sem,
                "hop1": direct_views["hop1"].clone() if direct_views is not None else zero.clone(),
                "hop2": direct_views["hop2"].clone() if direct_views is not None else zero.clone(),
                "count_1hop": counts.copy(),
                "count_2hop": counts.copy(),
            }

        edge_index_t = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
        if edge_index_t.dim() != 2 or edge_index_t.size(0) != 2:
            raise ValueError("edge_index must be shaped [2, num_edges] for glance_oracle_refine.")
        if edge_index_t.numel() == 0:
            zero = torch.zeros_like(x_sem)
            counts = np.zeros(num_nodes, dtype=np.int64)
            return {
                "ego": x_sem,
                "hop1": direct_views["hop1"].clone() if direct_views is not None else zero.clone(),
                "hop2": direct_views["hop2"].clone() if direct_views is not None else zero.clone(),
                "count_1hop": counts.copy(),
                "count_2hop": counts.copy(),
            }

        src = edge_index_t[0].numpy()
        dst = edge_index_t[1].numpy()
        neighbors_in = [[] for _ in range(num_nodes)]
        for s, d in zip(src.tolist(), dst.tolist()):
            if 0 <= s < num_nodes and 0 <= d < num_nodes:
                neighbors_in[d].append(s)

        hop1 = direct_views["hop1"].clone() if direct_views is not None else torch.zeros_like(x_sem)
        hop2 = direct_views["hop2"].clone() if direct_views is not None else torch.zeros_like(x_sem)
        count_1hop = np.zeros(num_nodes, dtype=np.int64)
        count_2hop = np.zeros(num_nodes, dtype=np.int64)

        for node_idx in range(num_nodes):
            n1 = sorted(set(int(n) for n in neighbors_in[node_idx] if int(n) != node_idx))
            count_1hop[node_idx] = len(n1)
            if direct_views is None and n1:
                hop1[node_idx] = x_sem[torch.tensor(n1, dtype=torch.long)].mean(dim=0)

            n1_set = set(n1)
            n2 = set()
            for neigh in n1:
                for cand in neighbors_in[neigh]:
                    cand = int(cand)
                    if cand == node_idx or cand in n1_set:
                        continue
                    n2.add(cand)
            n2 = sorted(n2)
            count_2hop[node_idx] = len(n2)
            if direct_views is None and n2:
                hop2[node_idx] = x_sem[torch.tensor(n2, dtype=torch.long)].mean(dim=0)

        return {
            "ego": x_sem,
            "hop1": hop1,
            "hop2": hop2,
            "count_1hop": count_1hop,
            "count_2hop": count_2hop,
        }

    def _build_prompt_expert_semantic_views(self, expert_bundle, z_gnn):
        if not expert_bundle:
            raise MissingFrozenArtifactError(
                "prompt_expert_bundle_v1 requires expert_bundle payload metadata from the semantic cache."
            )
        num_nodes = int(z_gnn.shape[0])
        component_tensors = dict(expert_bundle.get("component_tensors", {}))
        base_component = next(iter(component_tensors.values()), None)
        if base_component is None:
            raise MissingFrozenArtifactError(
                "prompt_expert_bundle_v1 requires at least one of ego/graph_following/graph_follower/tweet/conflict in the prompt cache payload."
            )
        semantic_dim = int(base_component.shape[1])
        zero_component = torch.zeros((num_nodes, semantic_dim), dtype=torch.float32)
        semantic_views = {}
        for component_name in PromptExpertBundleRefinerMLP.COMPONENT_ORDER:
            component_value = component_tensors.get(component_name)
            if component_value is None:
                semantic_views[component_name] = zero_component.clone()
                continue
            if int(component_value.shape[0]) != num_nodes:
                raise MissingFrozenArtifactError(
                    f"Prompt-expert component {component_name} has {int(component_value.shape[0])} nodes, expected {num_nodes}."
                )
            semantic_views[component_name] = component_value.detach().cpu().float()

        scalar_tensors = dict(expert_bundle.get("scalar_tensors", {}))

        def _scalar(name):
            value = scalar_tensors.get(name)
            if value is None:
                return torch.zeros(num_nodes, dtype=torch.float32)
            value = value.detach().cpu().float().view(-1)
            if int(value.shape[0]) != num_nodes:
                raise MissingFrozenArtifactError(
                    f"Prompt-expert scalar {name} has {int(value.shape[0])} nodes, expected {num_nodes}."
                )
            return value

        count_following = _scalar("count_following")
        count_follower = _scalar("count_follower")
        has_following = _scalar("has_following")
        has_follower = _scalar("has_follower")
        structural_features = torch.stack(
            [
                torch.log1p(count_following),
                torch.log1p(count_follower),
                has_following,
                has_follower,
            ],
            dim=1,
        ).float()

        analysis_views = dict(semantic_views)
        analysis_views.update(
            {
                "count_following": count_following.numpy().astype(np.int64),
                "count_follower": count_follower.numpy().astype(np.int64),
                "has_following": has_following.numpy().astype(np.int64),
                "has_follower": has_follower.numpy().astype(np.int64),
                "count_1hop": (count_following + count_follower).numpy().astype(np.int64),
                "count_2hop": np.zeros(num_nodes, dtype=np.int64),
                "rt_ratio": _scalar("rt_ratio").numpy().astype(np.float32),
                "url_ratio": _scalar("url_ratio").numpy().astype(np.float32),
                "hashtag_ratio": _scalar("hashtag_ratio").numpy().astype(np.float32),
            }
        )
        refiner_features = {
            "feature_kind": "prompt_expert_bundle_v1",
            "z_gnn": z_gnn.detach().cpu().float(),
            "semantic_views": semantic_views,
            "structural_features": structural_features,
            "active_components": list(expert_bundle.get("active_components", [])),
        }
        return refiner_features, analysis_views

    def _build_relation_aware_1hop_semantic_views(self, semantic_embeddings, edge_index, edge_type, include_2hop=False):
        x_sem = semantic_embeddings.detach().cpu().float()
        num_nodes = int(x_sem.shape[0])
        zero = torch.zeros_like(x_sem)
        zero_counts = np.zeros(num_nodes, dtype=np.int64)
        if edge_index is None or edge_type is None:
            return {
                "ego": x_sem,
                "in_rel0": zero.clone(),
                "in_rel1": zero.clone(),
                "out_rel0": zero.clone(),
                "out_rel1": zero.clone(),
                "in_rel0_2hop": zero.clone(),
                "in_rel1_2hop": zero.clone(),
                "out_rel0_2hop": zero.clone(),
                "out_rel1_2hop": zero.clone(),
                "count_in_rel0": zero_counts.copy(),
                "count_in_rel1": zero_counts.copy(),
                "count_out_rel0": zero_counts.copy(),
                "count_out_rel1": zero_counts.copy(),
                "count_in_rel0_2hop": zero_counts.copy(),
                "count_in_rel1_2hop": zero_counts.copy(),
                "count_out_rel0_2hop": zero_counts.copy(),
                "count_out_rel1_2hop": zero_counts.copy(),
                "has_in_rel0": zero_counts.copy(),
                "has_in_rel1": zero_counts.copy(),
                "has_out_rel0": zero_counts.copy(),
                "has_out_rel1": zero_counts.copy(),
                "has_in_rel0_2hop": zero_counts.copy(),
                "has_in_rel1_2hop": zero_counts.copy(),
                "has_out_rel0_2hop": zero_counts.copy(),
                "has_out_rel1_2hop": zero_counts.copy(),
                "count_1hop": zero_counts.copy(),
                "count_2hop": zero_counts.copy(),
            }

        edge_index_t = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
        edge_type_t = edge_type.detach().cpu().long() if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long)
        if edge_index_t.dim() != 2 or edge_index_t.size(0) != 2:
            raise ValueError("edge_index must be shaped [2, num_edges] for relation-aware semantic views.")
        if edge_type_t.numel() != edge_index_t.size(1):
            raise ValueError("edge_type must align with edge_index for relation-aware semantic views.")

        src = edge_index_t[0].numpy()
        dst = edge_index_t[1].numpy()
        rel = edge_type_t.numpy()
        buckets = {
            "in_rel0": [[] for _ in range(num_nodes)],
            "in_rel1": [[] for _ in range(num_nodes)],
            "out_rel0": [[] for _ in range(num_nodes)],
            "out_rel1": [[] for _ in range(num_nodes)],
        }
        for s, d, r in zip(src.tolist(), dst.tolist(), rel.tolist()):
            if not (0 <= s < num_nodes and 0 <= d < num_nodes):
                continue
            if int(r) == 0:
                buckets["in_rel0"][d].append(s)
                buckets["out_rel0"][s].append(d)
            elif int(r) == 1:
                buckets["in_rel1"][d].append(s)
                buckets["out_rel1"][s].append(d)

        outputs = {"ego": x_sem}
        union_1hop = [set() for _ in range(num_nodes)]
        for key in ("in_rel0", "in_rel1", "out_rel0", "out_rel1"):
            view_tensor = torch.zeros_like(x_sem)
            count_arr = np.zeros(num_nodes, dtype=np.int64)
            has_arr = np.zeros(num_nodes, dtype=np.int64)
            for node_idx in range(num_nodes):
                neighbors = sorted(set(int(n) for n in buckets[key][node_idx] if int(n) != node_idx))
                union_1hop[node_idx].update(neighbors)
                count_arr[node_idx] = len(neighbors)
                has_arr[node_idx] = 1 if neighbors else 0
                if neighbors:
                    view_tensor[node_idx] = x_sem[torch.tensor(neighbors, dtype=torch.long)].mean(dim=0)
            outputs[key] = view_tensor
            outputs[f"count_{key}"] = count_arr
            outputs[f"has_{key}"] = has_arr
        outputs["count_1hop"] = np.asarray([len(item) for item in union_1hop], dtype=np.int64)
        outputs["count_2hop"] = zero_counts.copy()
        if include_2hop:
            key_map = {
                "in_rel0_2hop": "in_rel0",
                "in_rel1_2hop": "in_rel1",
                "out_rel0_2hop": "out_rel0",
                "out_rel1_2hop": "out_rel1",
            }
            union_2hop = [set() for _ in range(num_nodes)]
            for key, parent_key in key_map.items():
                view_tensor = torch.zeros_like(x_sem)
                count_arr = np.zeros(num_nodes, dtype=np.int64)
                has_arr = np.zeros(num_nodes, dtype=np.int64)
                parent_bucket = buckets[parent_key]
                for node_idx in range(num_nodes):
                    n1 = sorted(set(int(n) for n in parent_bucket[node_idx] if int(n) != node_idx))
                    n1_set = set(n1)
                    n2 = set()
                    for neigh in n1:
                        for cand in parent_bucket[neigh]:
                            cand = int(cand)
                            if cand == node_idx or cand in n1_set:
                                continue
                            n2.add(cand)
                    neighbors = sorted(n2)
                    union_2hop[node_idx].update(neighbors)
                    count_arr[node_idx] = len(neighbors)
                    has_arr[node_idx] = 1 if neighbors else 0
                    if neighbors:
                        view_tensor[node_idx] = x_sem[torch.tensor(neighbors, dtype=torch.long)].mean(dim=0)
                outputs[key] = view_tensor
                outputs[f"count_{key}"] = count_arr
                outputs[f"has_{key}"] = has_arr
            outputs["count_2hop"] = np.asarray([len(item) for item in union_2hop], dtype=np.int64)
        return outputs

    def _train_glance_refiner(self, model, train_features, train_labels, valid_features, valid_labels):
        train_features = train_features.detach().cpu().float()
        valid_features = valid_features.detach().cpu().float()
        train_labels = train_labels.detach().cpu().long()
        valid_labels = valid_labels.detach().cpu().long()

        if train_features.size(0) == 0:
            raise ValueError(f"{self.args.stage} requires at least one train node.")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(getattr(self.args, "lr_LM", 1e-5)),
            weight_decay=float(getattr(self.args, "weight_decay_LM", 0.01)),
        )
        batch_size = max(min(int(getattr(self.args, "batch_size_LM", 32)), int(train_features.size(0))), 1)
        loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)
        max_epochs = max(int(getattr(self.args, "LM_pretrain_epochs", 5)), 1)
        patience = max(int(getattr(self.args, "LM_eval_patience", 20)), 1)

        best_state = None
        best_metrics = None
        best_score = (-1.0, float("inf"))
        wait = 0
        train_losses = []

        model.to(self.device)
        for epoch in range(max_epochs):
            model.train()
            epoch_losses = []
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = F.cross_entropy(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            train_losses.append(train_loss)

            model.eval()
            with torch.no_grad():
                train_logits = model(train_features.to(self.device)).cpu()
                if valid_features.size(0) > 0:
                    valid_logits = model(valid_features.to(self.device)).cpu()
                else:
                    valid_logits = torch.empty((0, 2), dtype=torch.float32)
            train_metrics = _classification_metrics_from_logits(train_logits, train_labels.cpu(), torch.arange(train_labels.numel()))
            if valid_features.size(0) > 0:
                valid_metrics = _classification_metrics_from_logits(valid_logits, valid_labels.cpu(), torch.arange(valid_labels.numel()))
                current_score = (float(valid_metrics["macro_f1"]), -float(valid_metrics["loss"]))
            else:
                valid_metrics = {"loss": float("inf"), "accuracy": 0.0, "macro_f1": 0.0, "bot_f1": 0.0, "count": 0}
                current_score = (float(train_metrics["macro_f1"]), -float(train_metrics["loss"]))

            if current_score > best_score:
                best_score = current_score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_metrics = {
                    "epoch": epoch + 1,
                    "train": train_metrics,
                    "valid": valid_metrics,
                    "train_loss": train_loss,
                }
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is None:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = {
                "epoch": 0,
                "train": _classification_metrics_from_logits(
                    model(train_features.to(self.device)).detach().cpu(),
                    train_labels.cpu(),
                    torch.arange(train_labels.numel()),
                ),
                "valid": {"loss": float("inf"), "accuracy": 0.0, "macro_f1": 0.0, "bot_f1": 0.0, "count": int(valid_labels.numel())},
                "train_loss": 0.0,
            }

        model.load_state_dict(best_state)
        model.to("cpu")
        return model, {
            "best_epoch": int(best_metrics["epoch"]),
            "train_losses": train_losses,
            "best_train": best_metrics["train"],
            "best_valid": best_metrics["valid"],
            "hidden_dim": 128,
            "dropout": 0.1,
            "activation": str(getattr(self.args, "activation", "leakyrelu")).lower(),
        }

    @staticmethod
    def _refiner_bucket_summary(rows, key):
        buckets = {
            "0": lambda x: x == 0,
            "1": lambda x: x == 1,
            "2-5": lambda x: 2 <= x <= 5,
            "6-10": lambda x: 6 <= x <= 10,
            "11+": lambda x: x >= 11,
        }
        result = {}
        for name, fn in buckets.items():
            subset = [row for row in rows if fn(int(row[key]))]
            if not subset:
                continue
            base_wrong = sum(1 for row in subset if row["was_wrong_base"])
            fixed = sum(1 for row in subset if row["was_wrong_base"] and not row["is_wrong_final"])
            base_correct = sum(1 for row in subset if not row["was_wrong_base"])
            broken = sum(1 for row in subset if (not row["was_wrong_base"]) and row["is_wrong_final"])
            result[name] = {
                "count": int(len(subset)),
                "base_wrong": int(base_wrong),
                "fixed": int(fixed),
                "fix_rate": float(fixed / base_wrong) if base_wrong else None,
                "base_correct": int(base_correct),
                "broken": int(broken),
                "break_rate": float(broken / base_correct) if base_correct else None,
            }
        return result

    def _build_refiner_analysis_summary(self, per_node_rows, mode_name):
        rows = list(per_node_rows)
        base_wrong = [row for row in rows if row["was_wrong_base"]]
        base_correct = [row for row in rows if not row["was_wrong_base"]]
        fixed = [row for row in base_wrong if not row["is_wrong_final"]]
        still_wrong = [row for row in base_wrong if row["is_wrong_final"]]
        broken = [row for row in base_correct if row["is_wrong_final"]]
        preserved = [row for row in base_correct if not row["is_wrong_final"]]
        changed = [row for row in rows if row["base_pred"] != row["final_pred"]]
        routed = [row for row in rows if bool(row.get("routed", True))]
        routed_wrong = [row for row in routed if row["was_wrong_base"]]
        routed_correct = [row for row in routed if not row["was_wrong_base"]]
        routed_fixed = [row for row in routed_wrong if not row["is_wrong_final"]]
        routed_broken = [row for row in routed_correct if row["is_wrong_final"]]

        def _avg(items, key):
            if not items:
                return 0.0
            return float(sum(float(row[key]) for row in items) / len(items))

        directional_bucket_analysis = None
        if rows and any(
            any(key in row for key in ("has_following", "has_follower", "neighbor_count_following", "neighbor_count_follower"))
            for row in rows
        ):
            directional_bucket_analysis = {}
            directional_buckets = {
                "no_directional_neighbors": lambda row: (int(row.get("has_following", 0)) == 0) and (int(row.get("has_follower", 0)) == 0),
                "following_only": lambda row: (int(row.get("has_following", 0)) == 1) and (int(row.get("has_follower", 0)) == 0),
                "follower_only": lambda row: (int(row.get("has_following", 0)) == 0) and (int(row.get("has_follower", 0)) == 1),
                "both_present": lambda row: (int(row.get("has_following", 0)) == 1) and (int(row.get("has_follower", 0)) == 1),
            }
            for name, fn in directional_buckets.items():
                subset = [row for row in rows if fn(row)]
                base_wrong_subset = [row for row in subset if row["was_wrong_base"]]
                base_correct_subset = [row for row in subset if not row["was_wrong_base"]]
                fixed_subset = [row for row in base_wrong_subset if not row["is_wrong_final"]]
                broken_subset = [row for row in base_correct_subset if row["is_wrong_final"]]
                directional_bucket_analysis[name] = {
                    "count": int(len(subset)),
                    "base_wrong": int(len(base_wrong_subset)),
                    "fixed": int(len(fixed_subset)),
                    "fix_rate": float(len(fixed_subset) / len(base_wrong_subset)) if base_wrong_subset else None,
                    "base_correct": int(len(base_correct_subset)),
                    "broken": int(len(broken_subset)),
                    "break_rate": float(len(broken_subset) / len(base_correct_subset)) if base_correct_subset else None,
                }

        return {
            "mode": mode_name,
            "total_test_nodes": int(len(rows)),
            "base_wrong_count": int(len(base_wrong)),
            "base_correct_count": int(len(base_correct)),
            "fixed_count": int(len(fixed)),
            "still_wrong_count": int(len(still_wrong)),
            "broken_count": int(len(broken)),
            "preserved_correct_count": int(len(preserved)),
            "improved_count": int(len(fixed)),
            "degraded_count": int(len(broken)),
            "net_gain": int(len(fixed) - len(broken)),
            "changed_prediction_count": int(len(changed)),
            "unchanged_prediction_count": int(len(rows) - len(changed)),
            "wrong_node_fix_rate": float(len(fixed) / len(base_wrong)) if base_wrong else 0.0,
            "correct_node_break_rate": float(len(broken) / len(base_correct)) if base_correct else 0.0,
            "wrong_nodes_changed_rate": float(sum(1 for row in base_wrong if row["base_pred"] != row["final_pred"]) / len(base_wrong)) if base_wrong else 0.0,
            "correct_nodes_changed_rate": float(sum(1 for row in base_correct if row["base_pred"] != row["final_pred"]) / len(base_correct)) if base_correct else 0.0,
            "routed_count": int(len(routed)),
            "routed_wrong_count": int(len(routed_wrong)),
            "routed_correct_count": int(len(routed_correct)),
            "routed_wrong_coverage": float(len(routed_wrong) / len(base_wrong)) if base_wrong else 0.0,
            "routed_wrong_precision": float(len(routed_wrong) / len(routed)) if routed else 0.0,
            "conditional_fix_rate_on_selected_wrong": float(len(routed_fixed) / len(routed_wrong)) if routed_wrong else 0.0,
            "conditional_break_rate_on_selected_correct": float(len(routed_broken) / len(routed_correct)) if routed_correct else 0.0,
            "avg_neighbor_counts": {
                "fixed": {
                    "hop1": _avg(fixed, "neighbor_count_1hop"),
                    "hop2": _avg(fixed, "neighbor_count_2hop"),
                },
                "still_wrong": {
                    "hop1": _avg(still_wrong, "neighbor_count_1hop"),
                    "hop2": _avg(still_wrong, "neighbor_count_2hop"),
                },
                "broken": {
                    "hop1": _avg(broken, "neighbor_count_1hop"),
                    "hop2": _avg(broken, "neighbor_count_2hop"),
                },
                "preserved": {
                    "hop1": _avg(preserved, "neighbor_count_1hop"),
                    "hop2": _avg(preserved, "neighbor_count_2hop"),
                },
            },
            "bucket_analysis": {
                "neighbor_count_1hop": self._refiner_bucket_summary(rows, "neighbor_count_1hop"),
                "neighbor_count_2hop": self._refiner_bucket_summary(rows, "neighbor_count_2hop"),
            },
            "directional_bucket_analysis": directional_bucket_analysis,
            "examples": {
                "fixed": fixed[:10],
                "broken": broken[:10],
            },
        }

    @staticmethod
    def _budget_row_score(row, count_key="routed_count"):
        return (
            float(row.get("delta_macro_f1", 0.0)),
            int(row.get("net", 0)),
            -int(row.get(count_key, row.get("routed_count", 0))),
        )

    def _select_best_budget_row(self, rows, count_key="routed_count"):
        if not rows:
            return None, None, {}
        best_row = max(rows, key=lambda item: self._budget_row_score(item, count_key))
        rows_by_key = {str(self._budget_key(item["budget"])): item for item in rows}
        best_key = str(self._budget_key(best_row["budget"]))
        return best_row, best_key, rows_by_key

    @staticmethod
    def _standardize_glance_router_features(feature_matrix, fit_idx):
        feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
        if feature_matrix.ndim != 2:
            raise ValueError("GLANCE router feature matrix must be 2-D.")
        fit_idx = np.asarray(fit_idx, dtype=np.int64).reshape(-1)
        fit_matrix = feature_matrix[fit_idx] if fit_idx.size else feature_matrix
        scaler = StandardScaler()
        scaler.fit(fit_matrix)
        scaled = scaler.transform(feature_matrix).astype(np.float32)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(scaled, dtype=torch.float32), {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        }

    @staticmethod
    def _iter_glance_batches(idx, batch_size, shuffle=False, seed=0):
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive for GLANCE routing batches.")
        if shuffle:
            rng = np.random.default_rng(int(seed))
            idx = idx[rng.permutation(idx.size)]
        for start in range(0, idx.size, int(batch_size)):
            yield idx[start : start + int(batch_size)]

    @staticmethod
    def _glance_training_top_k(batch_size, epoch_index, decay_factor=0.5):
        batch_size = max(int(batch_size), 1)
        k_start = batch_size
        k_end = max(int(round(batch_size / 4.0)), 1)
        value = k_end + (k_start - k_end) * (float(decay_factor) ** int(epoch_index))
        return max(min(int(round(value)), batch_size), 1)

    @staticmethod
    def _glance_refiner_forward(refiner_model, batch_refiner, batch_base_logits=None):
        if isinstance(batch_refiner, dict):
            feature_kind = batch_refiner.get("feature_kind")
            if feature_kind == "prompt_expert_bundle_v1":
                outputs = refiner_model(
                    batch_refiner["z_gnn"],
                    batch_refiner["semantic_views"],
                    batch_refiner["structural_features"],
                )
            elif feature_kind == "directional_gated":
                outputs = refiner_model(
                    batch_refiner["z_gnn"],
                    batch_refiner["semantic_views"],
                    batch_refiner.get("presence_features"),
                )
            elif feature_kind in {"directional_weighted_gated", "directional_weighted_projected", "relation_projected"}:
                outputs = refiner_model(
                    batch_refiner["z_gnn"],
                    batch_refiner["semantic_views"],
                    batch_refiner["structural_features"],
                )
            else:
                raise ValueError(f"Unsupported refiner feature kind: {feature_kind}")
        else:
            outputs = refiner_model(batch_refiner)
        if isinstance(outputs, tuple):
            if len(outputs) == 3:
                refiner_logits, gate_logits, gate_prob = outputs
            elif len(outputs) == 2:
                refiner_logits, gate_prob = outputs
                gate_logits = torch.logit(gate_prob.clamp_min(1e-6).clamp_max(1.0 - 1e-6))
            else:
                raise ValueError("Unsupported gated refiner output arity.")
        else:
            refiner_logits = outputs
            gate_logits = None
            gate_prob = None
        mixed_logits = refiner_logits
        if gate_prob is not None:
            if batch_base_logits is None:
                raise ValueError("Gated refiner requires batch_base_logits for mixed inference.")
            base_prob = torch.softmax(batch_base_logits, dim=1)
            ref_prob = torch.softmax(refiner_logits, dim=1)
            mixed_prob = (1.0 - gate_prob.unsqueeze(1)) * base_prob + gate_prob.unsqueeze(1) * ref_prob
            mixed_prob = mixed_prob.clamp_min(1e-8)
            mixed_prob = mixed_prob / mixed_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
            mixed_logits = torch.log(mixed_prob)
        return {
            "refiner_logits": refiner_logits,
            "mixed_logits": mixed_logits,
            "gate_logits": gate_logits,
            "gate_prob": gate_prob,
        }

    @staticmethod
    def _glance_refiner_sample_weights(base_wrong_target, utility_target, mode, base_wrong_weight=2.0, utility_weight=3.0):
        mode = str(mode or "off").lower()
        weights = torch.ones_like(base_wrong_target, dtype=torch.float32)
        if mode in {"base_wrong", "base_wrong_plus_utility"}:
            weights = weights + base_wrong_target.float() * max(float(base_wrong_weight) - 1.0, 0.0)
        if mode in {"utility_positive", "base_wrong_plus_utility"}:
            weights = weights + utility_target.float() * max(float(utility_weight) - 1.0, 0.0)
        return weights

    def _apply_glance_joint_policy(
        self,
        router_model,
        refiner_model,
        router_features,
        refiner_features,
        semantic_view_mode,
        base_logits,
        base_prob,
        base_pred,
        labels_t,
        beta,
        split_idx,
        batch_size,
        top_k,
    ):
        split_idx = np.asarray(split_idx, dtype=np.int64).reshape(-1)
        final_logits = base_logits.clone()
        final_prob = base_prob.clone()
        final_pred = base_pred.clone()
        routed_mask = torch.zeros(base_pred.numel(), dtype=torch.bool)
        router_prob_all = torch.zeros(base_pred.numel(), dtype=torch.float32)
        router_score_all = torch.zeros(base_pred.numel(), dtype=torch.float32)
        oracle_advantage_all = torch.zeros(base_pred.numel(), dtype=torch.float32)
        if split_idx.size == 0:
            return {
                "logits": final_logits,
                "prob": final_prob,
                "pred": final_pred,
                "routed_mask": routed_mask,
                "router_prob": router_prob_all,
                "router_score": router_score_all,
                "oracle_advantage": oracle_advantage_all,
                "routed_count": 0,
            }

        router_model.eval()
        refiner_model.eval()
        router_device = next(router_model.parameters()).device
        refiner_device = next(refiner_model.parameters()).device
        with torch.no_grad():
            for batch_idx_np in self._iter_glance_batches(split_idx, batch_size, shuffle=False):
                batch_idx = torch.tensor(batch_idx_np, dtype=torch.long)
                batch_score, batch_prob = router_model(router_features[batch_idx].to(router_device))
                batch_score = batch_score.detach().cpu()
                batch_prob = batch_prob.detach().cpu()
                router_score_all[batch_idx] = batch_score
                router_prob_all[batch_idx] = batch_prob
                k = min(max(int(top_k), 1), int(batch_idx.numel()))
                routed_rel = torch.topk(batch_score, k=k, largest=True, sorted=False).indices
                routed_idx = batch_idx[routed_rel]
                routed_mask[routed_idx] = True
                routed_forward = self._glance_refiner_forward(
                    refiner_model,
                    _refiner_feature_lookup(refiner_features, routed_idx, semantic_view_mode, refiner_device),
                    batch_base_logits=base_logits[routed_idx].to(refiner_device),
                )
                final_logits[routed_idx] = routed_forward["mixed_logits"].cpu()
                final_prob[routed_idx] = torch.softmax(routed_forward["mixed_logits"].cpu(), dim=1)
                final_pred[routed_idx] = routed_forward["mixed_logits"].cpu().argmax(dim=1)
                full_forward = self._glance_refiner_forward(
                    refiner_model,
                    _refiner_feature_lookup(refiner_features, batch_idx, semantic_view_mode, refiner_device),
                    batch_base_logits=base_logits[batch_idx].to(refiner_device),
                )
                full_ref_loss = F.cross_entropy(
                    full_forward["mixed_logits"].cpu(),
                    labels_t[batch_idx].to(refiner_device).cpu(),
                    reduction="none",
                )
                base_loss = F.cross_entropy(
                    base_logits[batch_idx],
                    labels_t[batch_idx],
                    reduction="none",
                )
                oracle_advantage_all[batch_idx] = base_loss.detach().cpu() - full_ref_loss.detach().cpu() - float(beta)

        return {
            "logits": final_logits,
            "prob": final_prob,
            "pred": final_pred,
            "routed_mask": routed_mask,
            "router_prob": router_prob_all,
            "router_score": router_score_all,
            "oracle_advantage": oracle_advantage_all,
            "routed_count": int(routed_mask[torch.tensor(split_idx, dtype=torch.long)].sum().item()),
        }

    def _collect_glance_joint_full_split_outputs(
        self,
        router_model,
        refiner_model,
        router_features,
        refiner_features,
        semantic_view_mode,
        base_logits,
        base_prob,
        base_pred,
        labels_t,
        beta,
        split_idx,
        batch_size,
    ):
        split_idx = np.asarray(split_idx, dtype=np.int64).reshape(-1)
        refiner_logits_all = base_logits.clone()
        refiner_prob_all = base_prob.clone()
        refiner_pred_all = base_pred.clone()
        router_prob_all = torch.zeros(base_pred.numel(), dtype=torch.float32)
        router_score_all = torch.zeros(base_pred.numel(), dtype=torch.float32)
        oracle_advantage_all = torch.zeros(base_pred.numel(), dtype=torch.float32)
        if split_idx.size == 0:
            return {
                "refiner_logits": refiner_logits_all,
                "refiner_prob": refiner_prob_all,
                "refiner_pred": refiner_pred_all,
                "router_prob": router_prob_all,
                "router_score": router_score_all,
                "oracle_advantage": oracle_advantage_all,
            }

        router_model.eval()
        refiner_model.eval()
        router_device = next(router_model.parameters()).device
        refiner_device = next(refiner_model.parameters()).device
        with torch.no_grad():
            for batch_idx_np in self._iter_glance_batches(split_idx, batch_size, shuffle=False):
                batch_idx = torch.tensor(batch_idx_np, dtype=torch.long)
                batch_score, batch_prob = router_model(router_features[batch_idx].to(router_device))
                batch_score = batch_score.detach().cpu()
                batch_prob = batch_prob.detach().cpu()
                router_score_all[batch_idx] = batch_score
                router_prob_all[batch_idx] = batch_prob

                full_forward = self._glance_refiner_forward(
                    refiner_model,
                    _refiner_feature_lookup(refiner_features, batch_idx, semantic_view_mode, refiner_device),
                    batch_base_logits=base_logits[batch_idx].to(refiner_device),
                )
                full_ref_logits = full_forward["mixed_logits"].detach().cpu()
                full_ref_prob = torch.softmax(full_ref_logits, dim=1)
                refiner_logits_all[batch_idx] = full_ref_logits
                refiner_prob_all[batch_idx] = full_ref_prob
                refiner_pred_all[batch_idx] = full_ref_logits.argmax(dim=1)

                full_ref_loss = F.cross_entropy(
                    full_ref_logits,
                    labels_t[batch_idx],
                    reduction="none",
                )
                base_loss = F.cross_entropy(
                    base_logits[batch_idx],
                    labels_t[batch_idx],
                    reduction="none",
                )
                oracle_advantage_all[batch_idx] = base_loss.detach().cpu() - full_ref_loss.detach().cpu() - float(beta)

        return {
            "refiner_logits": refiner_logits_all,
            "refiner_prob": refiner_prob_all,
            "refiner_pred": refiner_pred_all,
            "router_prob": router_prob_all,
            "router_score": router_score_all,
            "oracle_advantage": oracle_advantage_all,
        }

    @staticmethod
    def _parse_float_candidate_csv(raw_value, default_values):
        if raw_value is None:
            return tuple(float(item) for item in default_values)
        tokens = [token.strip() for token in str(raw_value).split(",") if token.strip()]
        if not tokens:
            return tuple(float(item) for item in default_values)
        return tuple(float(token) for token in tokens)

    @staticmethod
    def _parse_int_candidate_csv(raw_value, default_values):
        if raw_value is None:
            return tuple(int(item) for item in default_values)
        tokens = [token.strip() for token in str(raw_value).split(",") if token.strip()]
        if not tokens:
            return tuple(int(item) for item in default_values)
        return tuple(int(token) for token in tokens)

    def _build_glance_joint_per_node_rows(
        self,
        split_idx,
        pred_gnn,
        final_pred,
        labels_np,
        routed_mask,
        router_prob,
        router_score,
        oracle_advantage,
        semantic_views,
        mode_name,
        route_prob_by_epoch=None,
        route_score_by_epoch=None,
        refiner_pred_by_epoch=None,
        base_pred_by_epoch=None,
    ):
        split_idx = np.asarray(split_idx, dtype=np.int64).reshape(-1)
        rows = []
        final_pred_np = final_pred.detach().cpu().numpy().astype(np.int64)
        pred_gnn_np = pred_gnn.detach().cpu().numpy().astype(np.int64)
        routed_mask_np = routed_mask.detach().cpu().numpy().astype(bool)
        router_prob_np = router_prob.detach().cpu().numpy().astype(np.float32)
        router_score_np = router_score.detach().cpu().numpy().astype(np.float32) if router_score is not None else None
        oracle_advantage_np = oracle_advantage.detach().cpu().numpy().astype(np.float32) if oracle_advantage is not None else None
        count_1hop = semantic_views.get("count_1hop")
        count_2hop = semantic_views.get("count_2hop")
        if count_1hop is None:
            count_1hop = semantic_views.get("count_following")
        if count_2hop is None:
            count_2hop = semantic_views.get("count_follower")
        for node_idx in split_idx.tolist():
            row = {
                "node_id": int(node_idx),
                "base_pred": int(pred_gnn_np[node_idx]),
                "final_pred": int(final_pred_np[node_idx]),
                "label": int(labels_np[node_idx]),
                "routed": bool(routed_mask_np[node_idx]),
                "router_prob": float(router_prob_np[node_idx]),
                "router_score": float(router_score_np[node_idx]) if router_score_np is not None else 0.0,
                "oracle_advantage": float(oracle_advantage_np[node_idx]) if oracle_advantage_np is not None else 0.0,
                "was_wrong_base": bool(pred_gnn_np[node_idx] != labels_np[node_idx]),
                "is_wrong_final": bool(final_pred_np[node_idx] != labels_np[node_idx]),
                "neighbor_count_1hop": int(count_1hop[node_idx]) if count_1hop is not None else 0,
                "neighbor_count_2hop": int(count_2hop[node_idx]) if count_2hop is not None else 0,
                "neighbor_count_following": int(semantic_views["count_following"][node_idx]) if "count_following" in semantic_views else 0,
                "neighbor_count_follower": int(semantic_views["count_follower"][node_idx]) if "count_follower" in semantic_views else 0,
                "has_following": int(semantic_views["has_following"][node_idx]) if "has_following" in semantic_views else 0,
                "has_follower": int(semantic_views["has_follower"][node_idx]) if "has_follower" in semantic_views else 0,
            }
            if route_prob_by_epoch is not None:
                row["router_prob_by_epoch"] = [float(v[node_idx]) for v in route_prob_by_epoch]
            if route_score_by_epoch is not None:
                row["router_score_by_epoch"] = [float(v[node_idx]) for v in route_score_by_epoch]
            if refiner_pred_by_epoch is not None:
                row["refiner_pred_by_epoch"] = [int(v[node_idx]) for v in refiner_pred_by_epoch]
            if base_pred_by_epoch is not None:
                row["base_pred_by_epoch"] = [int(v[node_idx]) for v in base_pred_by_epoch]
            rows.append(row)
        return {
            "rows": rows,
            "analysis": self._build_refiner_analysis_summary(rows, mode_name),
        }

    def _train_glance_joint_candidate(
        self,
        *,
        labels_t,
        labels_np,
        train_idx,
        valid_idx,
        test_idx,
        logits_gnn,
        p_gnn,
        pred_gnn,
        router_features,
        refiner_features,
        semantic_views,
        semantic_view_mode,
        activation,
        batch_size,
        max_epochs,
        patience,
        decay_factor,
        entropy_weight,
        router_weight,
        router_regression_weight,
        router_ranking_weight,
        router_calibration_weight,
        router_reliability_weight,
        learning_rate,
        weight_decay,
        beta,
        eval_top_k,
        refiner_explicit_gate=False,
        refiner_target_mode="predict",
        refiner_weight_mode="off",
        refiner_base_wrong_weight=2.0,
        refiner_utility_weight=3.0,
        refiner_gate_weight=0.5,
    ):
        candidate_seed = int(self.seed) * 100000 + int(round(float(beta) * 1000)) * 10 + int(eval_top_k)
        torch.manual_seed(candidate_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(candidate_seed)
        np.random.seed(candidate_seed % (2**32 - 1))
        router_model = GlanceReliabilityRouterMLP(
            input_dim=int(router_features.shape[1]),
            hidden_dim=128,
            dropout=0.1,
        )
        refiner_feature_kind = refiner_features.get("feature_kind") if isinstance(refiner_features, dict) else None
        if refiner_feature_kind == "prompt_expert_bundle_v1":
            component_dims = {
                name: int(refiner_features["semantic_views"][name].shape[1])
                for name in PromptExpertBundleRefinerMLP.COMPONENT_ORDER
            }
            refiner_model = PromptExpertBundleRefinerMLP(
                z_gnn_dim=int(refiner_features["z_gnn"].shape[1]),
                component_dims=component_dims,
                proj_dim=256,
                hidden_dim=128,
                structural_dim=int(refiner_features["structural_features"].shape[1]),
                activation=activation,
                dropout=0.1,
            )
        elif bool(refiner_explicit_gate):
            refiner_model = GatedGlanceRefinerMLP(
                input_dim=int(refiner_features.shape[1]),
                hidden_dim=128,
                activation=activation,
                dropout=0.1,
            )
        else:
            refiner_model = GlanceRefinerMLP(
                input_dim=int(refiner_features.shape[1]),
                hidden_dim=128,
                activation=activation,
                dropout=0.1,
            )
        router_model.to(self.device)
        refiner_model.to(self.device)
        optimizer = torch.optim.AdamW(
            list(router_model.parameters()) + list(refiner_model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        best_state = None
        best_epoch_summary = None
        best_score = (-1.0, float("inf"))
        wait = 0
        epoch_history = []
        epoch_component_curves = []
        train_wrong = (pred_gnn.numpy()[train_idx] != labels_np[train_idx]).astype(np.int64)
        valid_wrong = (pred_gnn.numpy()[valid_idx] != labels_np[valid_idx]).astype(np.int64)
        test_wrong = (pred_gnn.numpy()[test_idx] != labels_np[test_idx]).astype(np.int64)

        for epoch in range(max_epochs):
            router_model.train()
            refiner_model.train()
            train_loss_history = []
            train_pred_history = []
            train_route_history = []
            train_router_regression_history = []
            train_router_ranking_history = []
            train_router_selection_history = []
            train_router_reliability_history = []
            train_adv_corr_history = []
            train_gate_history = []
            train_routed_weight_history = []
            train_k = self._glance_training_top_k(
                batch_size,
                epoch,
                decay_factor=decay_factor,
            )

            for batch_idx_np in self._iter_glance_batches(
                train_idx,
                batch_size,
                shuffle=True,
                seed=int(self.seed) * 1000 + int(epoch) + int(round(float(beta) * 1000)) + int(eval_top_k),
            ):
                batch_idx = torch.tensor(batch_idx_np, dtype=torch.long)
                batch_router_x = router_features[batch_idx].to(self.device)
                batch_refiner = _refiner_feature_lookup(refiner_features, batch_idx, semantic_view_mode, self.device)
                batch_labels = labels_t[batch_idx].to(self.device)
                batch_base_logits = logits_gnn[batch_idx].to(self.device)
                batch_base_pred = pred_gnn[batch_idx].to(self.device)
                batch_base_loss = F.cross_entropy(
                    batch_base_logits,
                    batch_labels,
                    reduction="none",
                ).detach()

                route_score, route_prob = router_model(batch_router_x)
                k = min(max(int(train_k), 1), int(batch_idx.numel()))
                routed_rel = torch.topk(route_score, k=k, largest=True, sorted=False).indices
                routed_mask = torch.zeros(batch_labels.size(0), dtype=torch.bool, device=self.device)
                routed_mask[routed_rel] = True

                full_forward = self._glance_refiner_forward(
                    refiner_model,
                    batch_refiner,
                    batch_base_logits=batch_base_logits,
                )
                base_wrong_target = (batch_base_pred != batch_labels).float()
                full_refiner_loss = F.cross_entropy(
                    full_forward["mixed_logits"],
                    batch_labels,
                    reduction="none",
                )
                utility_target = ((batch_base_loss - full_refiner_loss.detach() - float(beta)) > 0.0).float()

                routed_labels = batch_labels[routed_mask]
                if int(routed_labels.numel()) > 0:
                    routed_forward = self._glance_refiner_forward(
                        refiner_model,
                        _refiner_feature_lookup(refiner_features, batch_idx[routed_mask], semantic_view_mode, self.device),
                        batch_base_logits=batch_base_logits[routed_mask],
                    )
                    routed_sample_weights = self._glance_refiner_sample_weights(
                        base_wrong_target[routed_mask],
                        utility_target[routed_mask],
                        refiner_weight_mode,
                        base_wrong_weight=refiner_base_wrong_weight,
                        utility_weight=refiner_utility_weight,
                    )
                    if str(refiner_target_mode).lower() == "keep_change":
                        routed_target = (batch_base_pred[routed_mask] != routed_labels).long()
                        routed_refiner_loss_raw = F.cross_entropy(
                            routed_forward["refiner_logits"],
                            routed_target,
                            reduction="none",
                        )
                    else:
                        routed_refiner_loss_raw = F.cross_entropy(
                            routed_forward["mixed_logits"],
                            routed_labels,
                            reduction="none",
                        )
                    routed_refiner_loss = routed_refiner_loss_raw * routed_sample_weights
                    if routed_forward["gate_logits"] is not None:
                        gate_target = ((batch_base_pred[routed_mask] != routed_labels) | (utility_target[routed_mask] > 0.0)).float()
                        gate_loss = F.binary_cross_entropy_with_logits(
                            routed_forward["gate_logits"],
                            gate_target,
                            weight=routed_sample_weights,
                        )
                    else:
                        gate_loss = routed_refiner_loss.sum() * 0.0
                else:
                    routed_refiner_loss = torch.empty((0,), device=self.device)
                    routed_sample_weights = torch.empty((0,), device=self.device)
                    gate_loss = batch_base_loss.sum() * 0.0
                refiner_loss = batch_base_loss.detach().clone()
                refiner_loss[routed_mask] = routed_refiner_loss
                pred_loss = (
                    batch_base_loss[~routed_mask].sum()
                    + routed_refiner_loss.sum()
                ) / float(batch_labels.size(0))

                oracle_advantage = (batch_base_loss - full_refiner_loss.detach() - float(beta)).detach()
                route_loss = batch_base_loss.sum() * 0.0
                router_regression_loss = batch_base_loss.sum() * 0.0
                router_ranking_loss = _glance_pairwise_ranking_loss(route_score, base_wrong_target)
                pos_count = int(base_wrong_target.sum().detach().cpu().item())
                neg_count = int(base_wrong_target.numel() - pos_count)
                if 0 < pos_count < int(base_wrong_target.numel()):
                    pos_weight = torch.tensor(
                        float(neg_count / max(pos_count, 1)),
                        dtype=route_score.dtype,
                        device=route_score.device,
                    )
                    router_selection_loss = F.binary_cross_entropy_with_logits(
                        route_score,
                        base_wrong_target,
                        pos_weight=pos_weight,
                    )
                else:
                    router_selection_loss = route_score.sum() * 0.0
                utility_consistency_loss = batch_base_loss.sum() * 0.0
                total_loss = (
                    pred_loss
                    + float(refiner_gate_weight) * gate_loss
                    + float(router_ranking_weight) * router_ranking_loss
                    + float(router_calibration_weight) * router_selection_loss
                )

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

                train_loss_history.append(float(total_loss.detach().cpu().item()))
                train_pred_history.append(float(pred_loss.detach().cpu().item()))
                train_route_history.append(float(route_loss.detach().cpu().item()))
                train_router_regression_history.append(float(router_regression_loss.detach().cpu().item()))
                train_router_ranking_history.append(float(router_ranking_loss.detach().cpu().item()))
                train_router_selection_history.append(float(router_selection_loss.detach().cpu().item()))
                train_router_reliability_history.append(float(utility_consistency_loss.detach().cpu().item()))
                train_gate_history.append(float(gate_loss.detach().cpu().item()))
                train_routed_weight_history.append(float(routed_sample_weights.mean().detach().cpu().item()) if int(routed_sample_weights.numel()) > 0 else 0.0)
                train_adv_corr_history.append(
                    float(
                        _safe_score_corr(
                            route_score.detach().cpu().numpy(),
                            oracle_advantage.detach().cpu().numpy(),
                        )
                        or 0.0
                    )
                )

            router_model.to("cpu")
            refiner_model.to("cpu")
            train_outputs = self._apply_glance_joint_policy(
                router_model,
                refiner_model,
                router_features,
                refiner_features,
                semantic_view_mode,
                logits_gnn,
                p_gnn,
                pred_gnn,
                labels_t,
                beta,
                train_idx,
                batch_size,
                eval_top_k,
            )
            valid_outputs = self._apply_glance_joint_policy(
                router_model,
                refiner_model,
                router_features,
                refiner_features,
                semantic_view_mode,
                logits_gnn,
                p_gnn,
                pred_gnn,
                labels_t,
                beta,
                valid_idx,
                batch_size,
                eval_top_k,
            )
            test_outputs = self._apply_glance_joint_policy(
                router_model,
                refiner_model,
                router_features,
                refiner_features,
                semantic_view_mode,
                logits_gnn,
                p_gnn,
                pred_gnn,
                labels_t,
                beta,
                test_idx,
                batch_size,
                eval_top_k,
            )
            valid_metrics = _classification_metrics_from_logits(
                valid_outputs["logits"],
                labels_t,
                torch.tensor(valid_idx, dtype=torch.long),
            )
            valid_metrics["routed_count"] = int(valid_outputs["routed_count"])
            valid_metrics["query_rate"] = float(valid_outputs["routed_count"] / max(int(valid_idx.size), 1))
            train_metrics_epoch = _classification_metrics_from_logits(
                train_outputs["logits"],
                labels_t,
                torch.tensor(train_idx, dtype=torch.long),
            )
            train_metrics_epoch["routed_count"] = int(train_outputs["routed_count"])
            train_metrics_epoch["query_rate"] = float(train_outputs["routed_count"] / max(int(train_idx.size), 1))
            train_router_diag_epoch = _safe_binary_score_metrics(train_wrong, train_outputs["router_score"][torch.tensor(train_idx, dtype=torch.long)].numpy())
            valid_router_diag_epoch = _safe_binary_score_metrics(valid_wrong, valid_outputs["router_score"][torch.tensor(valid_idx, dtype=torch.long)].numpy())
            test_router_diag_epoch = _safe_binary_score_metrics(test_wrong, test_outputs["router_score"][torch.tensor(test_idx, dtype=torch.long)].numpy())
            train_adv_diag_epoch = _safe_advantage_router_metrics(
                train_outputs["oracle_advantage"][torch.tensor(train_idx, dtype=torch.long)].numpy(),
                train_outputs["router_score"][torch.tensor(train_idx, dtype=torch.long)].numpy(),
                train_outputs["routed_mask"][torch.tensor(train_idx, dtype=torch.long)].numpy(),
            )
            valid_adv_diag_epoch = _safe_advantage_router_metrics(
                valid_outputs["oracle_advantage"][torch.tensor(valid_idx, dtype=torch.long)].numpy(),
                valid_outputs["router_score"][torch.tensor(valid_idx, dtype=torch.long)].numpy(),
                valid_outputs["routed_mask"][torch.tensor(valid_idx, dtype=torch.long)].numpy(),
            )
            test_adv_diag_epoch = _safe_advantage_router_metrics(
                test_outputs["oracle_advantage"][torch.tensor(test_idx, dtype=torch.long)].numpy(),
                test_outputs["router_score"][torch.tensor(test_idx, dtype=torch.long)].numpy(),
                test_outputs["routed_mask"][torch.tensor(test_idx, dtype=torch.long)].numpy(),
            )
            train_delta_epoch = _delta_table(pred_gnn.numpy(), train_outputs["pred"].numpy(), labels_np, _mask_from_idx(labels_np.shape[0], train_idx))
            valid_delta_epoch = _delta_table(pred_gnn.numpy(), valid_outputs["pred"].numpy(), labels_np, _mask_from_idx(labels_np.shape[0], valid_idx))
            test_delta_epoch = _delta_table(pred_gnn.numpy(), test_outputs["pred"].numpy(), labels_np, _mask_from_idx(labels_np.shape[0], test_idx))

            epoch_summary = {
                "epoch": int(epoch + 1),
                "beta": float(beta),
                "eval_top_k": int(eval_top_k),
                "train_top_k": int(train_k),
                "avg_total_loss": float(np.mean(train_loss_history)) if train_loss_history else 0.0,
                "avg_pred_loss": float(np.mean(train_pred_history)) if train_pred_history else 0.0,
                "avg_route_loss": float(np.mean(train_route_history)) if train_route_history else 0.0,
                "avg_router_regression_loss": float(np.mean(train_router_regression_history)) if train_router_regression_history else 0.0,
                "avg_router_ranking_loss": float(np.mean(train_router_ranking_history)) if train_router_ranking_history else 0.0,
                "avg_router_selection_loss": float(np.mean(train_router_selection_history)) if train_router_selection_history else 0.0,
                "avg_router_calibration_loss": float(np.mean(train_router_selection_history)) if train_router_selection_history else 0.0,
                "avg_router_reliability_loss": float(np.mean(train_router_reliability_history)) if train_router_reliability_history else 0.0,
                "avg_refiner_gate_loss": float(np.mean(train_gate_history)) if train_gate_history else 0.0,
                "avg_routed_sample_weight": float(np.mean(train_routed_weight_history)) if train_routed_weight_history else 0.0,
                "avg_router_advantage_corr": float(np.mean(train_adv_corr_history)) if train_adv_corr_history else 0.0,
                "train": train_metrics_epoch,
                "valid": valid_metrics,
                "router_diagnostics": {
                    "train": train_router_diag_epoch,
                    "valid": valid_router_diag_epoch,
                    "test": test_router_diag_epoch,
                },
                "advantage_router_diagnostics": {
                    "train": train_adv_diag_epoch,
                    "valid": valid_adv_diag_epoch,
                    "test": test_adv_diag_epoch,
                },
                "component_deltas": {
                    "train": train_delta_epoch,
                    "valid": valid_delta_epoch,
                    "test": test_delta_epoch,
                },
            }
            epoch_history.append(epoch_summary)
            epoch_component_curves.append({
                "epoch": int(epoch + 1),
                "beta": float(beta),
                "eval_top_k": int(eval_top_k),
                "train_top_k": int(train_k),
                "router": {
                    "train": train_router_diag_epoch,
                    "valid": valid_router_diag_epoch,
                    "test": test_router_diag_epoch,
                },
                "refiner": {
                    "train": {
                        "wrong_node_fix_rate": float(train_delta_epoch["fix"] / max(int((pred_gnn.numpy()[train_idx] != labels_np[train_idx]).sum()), 1)),
                        "correct_node_break_rate": float(train_delta_epoch["broke"] / max(int(train_idx.size - (pred_gnn.numpy()[train_idx] != labels_np[train_idx]).sum()), 1)),
                        "net_gain": int(train_delta_epoch["net"]),
                    },
                    "valid": {
                        "wrong_node_fix_rate": float(valid_delta_epoch["fix"] / max(int((pred_gnn.numpy()[valid_idx] != labels_np[valid_idx]).sum()), 1)),
                        "correct_node_break_rate": float(valid_delta_epoch["broke"] / max(int(valid_idx.size - (pred_gnn.numpy()[valid_idx] != labels_np[valid_idx]).sum()), 1)),
                        "net_gain": int(valid_delta_epoch["net"]),
                    },
                    "test": {
                        "wrong_node_fix_rate": float(test_delta_epoch["fix"] / max(int((pred_gnn.numpy()[test_idx] != labels_np[test_idx]).sum()), 1)),
                        "correct_node_break_rate": float(test_delta_epoch["broke"] / max(int(test_idx.size - (pred_gnn.numpy()[test_idx] != labels_np[test_idx]).sum()), 1)),
                        "net_gain": int(test_delta_epoch["net"]),
                    },
                },
            })
            current_score = (float(valid_metrics["macro_f1"]), -float(valid_metrics["loss"]))
            if current_score > best_score:
                best_score = current_score
                best_state = {
                    "router": {key: value.detach().cpu().clone() for key, value in router_model.state_dict().items()},
                    "refiner": {key: value.detach().cpu().clone() for key, value in refiner_model.state_dict().items()},
                }
                best_epoch_summary = epoch_summary
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

            router_model.to(self.device)
            refiner_model.to(self.device)

        if best_state is None or best_epoch_summary is None:
            raise MissingFrozenArtifactError("glance_joint_router_refine failed to produce a valid checkpoint.")

        router_model.load_state_dict(best_state["router"])
        refiner_model.load_state_dict(best_state["refiner"])
        router_model.to("cpu")
        refiner_model.to("cpu")

        topk_train_outputs = self._apply_glance_joint_policy(
            router_model,
            refiner_model,
            router_features,
            refiner_features,
            semantic_view_mode,
            logits_gnn,
            p_gnn,
            pred_gnn,
            labels_t,
            beta,
            train_idx,
            batch_size,
            eval_top_k,
        )
        topk_valid_outputs = self._apply_glance_joint_policy(
            router_model,
            refiner_model,
            router_features,
            refiner_features,
            semantic_view_mode,
            logits_gnn,
            p_gnn,
            pred_gnn,
            labels_t,
            beta,
            valid_idx,
            batch_size,
            eval_top_k,
        )
        topk_test_outputs = self._apply_glance_joint_policy(
            router_model,
            refiner_model,
            router_features,
            refiner_features,
            semantic_view_mode,
            logits_gnn,
            p_gnn,
            pred_gnn,
            labels_t,
            beta,
            test_idx,
            batch_size,
            eval_top_k,
        )

        topk_train_metrics = _classification_metrics_from_logits(
            topk_train_outputs["logits"],
            labels_t,
            torch.tensor(train_idx, dtype=torch.long),
        )
        topk_valid_metrics = _classification_metrics_from_logits(
            topk_valid_outputs["logits"],
            labels_t,
            torch.tensor(valid_idx, dtype=torch.long),
        )
        topk_test_metrics = _classification_metrics_from_logits(
            topk_test_outputs["logits"],
            labels_t,
            torch.tensor(test_idx, dtype=torch.long),
        )
        for split_metrics, split_outputs, split_idx in (
            (topk_train_metrics, topk_train_outputs, train_idx),
            (topk_valid_metrics, topk_valid_outputs, valid_idx),
            (topk_test_metrics, topk_test_outputs, test_idx),
        ):
            split_metrics["routed_count"] = int(split_outputs["routed_count"])
            split_metrics["query_rate"] = float(split_outputs["routed_count"] / max(int(split_idx.size), 1))

        topk_train_rows = self._build_glance_joint_per_node_rows(
            train_idx,
            pred_gnn,
            topk_train_outputs["pred"],
            labels_np,
            topk_train_outputs["routed_mask"],
            topk_train_outputs["router_prob"],
            topk_train_outputs["router_score"],
            topk_train_outputs["oracle_advantage"],
            semantic_views,
            "glance_joint_router_refine_train",
        )
        topk_valid_rows = self._build_glance_joint_per_node_rows(
            valid_idx,
            pred_gnn,
            topk_valid_outputs["pred"],
            labels_np,
            topk_valid_outputs["routed_mask"],
            topk_valid_outputs["router_prob"],
            topk_valid_outputs["router_score"],
            topk_valid_outputs["oracle_advantage"],
            semantic_views,
            "glance_joint_router_refine_valid",
        )
        topk_test_rows = self._build_glance_joint_per_node_rows(
            test_idx,
            pred_gnn,
            topk_test_outputs["pred"],
            labels_np,
            topk_test_outputs["routed_mask"],
            topk_test_outputs["router_prob"],
            topk_test_outputs["router_score"],
            topk_test_outputs["oracle_advantage"],
            semantic_views,
            "glance_joint_router_refine_test",
        )
        train_wrong = (pred_gnn.numpy()[train_idx] != labels_np[train_idx]).astype(np.int64)
        valid_wrong = (pred_gnn.numpy()[valid_idx] != labels_np[valid_idx]).astype(np.int64)
        test_wrong = (pred_gnn.numpy()[test_idx] != labels_np[test_idx]).astype(np.int64)
        topk_train_router_diag = _safe_binary_score_metrics(train_wrong, topk_train_outputs["router_score"][torch.tensor(train_idx, dtype=torch.long)].numpy())
        topk_valid_router_diag = _safe_binary_score_metrics(valid_wrong, topk_valid_outputs["router_score"][torch.tensor(valid_idx, dtype=torch.long)].numpy())
        topk_test_router_diag = _safe_binary_score_metrics(test_wrong, topk_test_outputs["router_score"][torch.tensor(test_idx, dtype=torch.long)].numpy())
        topk_train_adv_diag = _safe_advantage_router_metrics(
            topk_train_outputs["oracle_advantage"][torch.tensor(train_idx, dtype=torch.long)].numpy(),
            topk_train_outputs["router_score"][torch.tensor(train_idx, dtype=torch.long)].numpy(),
            topk_train_outputs["routed_mask"][torch.tensor(train_idx, dtype=torch.long)].numpy(),
        )
        topk_valid_adv_diag = _safe_advantage_router_metrics(
            topk_valid_outputs["oracle_advantage"][torch.tensor(valid_idx, dtype=torch.long)].numpy(),
            topk_valid_outputs["router_score"][torch.tensor(valid_idx, dtype=torch.long)].numpy(),
            topk_valid_outputs["routed_mask"][torch.tensor(valid_idx, dtype=torch.long)].numpy(),
        )
        topk_test_adv_diag = _safe_advantage_router_metrics(
            topk_test_outputs["oracle_advantage"][torch.tensor(test_idx, dtype=torch.long)].numpy(),
            topk_test_outputs["router_score"][torch.tensor(test_idx, dtype=torch.long)].numpy(),
            topk_test_outputs["routed_mask"][torch.tensor(test_idx, dtype=torch.long)].numpy(),
        )
        component_curve_summary = {
            "best_epoch": int(best_epoch_summary["epoch"]),
            "curve": [
                {
                    "epoch": int(item["epoch"]),
                    "beta": float(item["beta"]),
                    "train_top_k": int(item["train_top_k"]),
                    "eval_top_k": int(item["eval_top_k"]),
                    "router_train_auroc": item["router"]["train"].get("auroc"),
                    "router_valid_auroc": item["router"]["valid"].get("auroc"),
                    "router_test_auroc": item["router"]["test"].get("auroc"),
                    "router_valid_adv_auroc": item.get("advantage_router_diagnostics", {}).get("valid", {}).get("auroc"),
                    "router_test_adv_auroc": item.get("advantage_router_diagnostics", {}).get("test", {}).get("auroc"),
                    "router_valid_auprc": item["router"]["valid"].get("auprc"),
                    "router_valid_adv_auprc": item.get("advantage_router_diagnostics", {}).get("valid", {}).get("auprc"),
                    "refiner_valid_fix_rate": item["refiner"]["valid"].get("wrong_node_fix_rate"),
                    "refiner_valid_break_rate": item["refiner"]["valid"].get("correct_node_break_rate"),
                    "refiner_test_fix_rate": item["refiner"]["test"].get("wrong_node_fix_rate"),
                    "refiner_test_break_rate": item["refiner"]["test"].get("correct_node_break_rate"),
                }
                for item in epoch_component_curves
            ],
        }

        full_train_outputs = self._collect_glance_joint_full_split_outputs(
            router_model,
            refiner_model,
            router_features,
            refiner_features,
            semantic_view_mode,
            logits_gnn,
            p_gnn,
            pred_gnn,
            labels_t,
            beta,
            train_idx,
            batch_size,
        )
        full_valid_outputs = self._collect_glance_joint_full_split_outputs(
            router_model,
            refiner_model,
            router_features,
            refiner_features,
            semantic_view_mode,
            logits_gnn,
            p_gnn,
            pred_gnn,
            labels_t,
            beta,
            valid_idx,
            batch_size,
        )
        full_test_outputs = self._collect_glance_joint_full_split_outputs(
            router_model,
            refiner_model,
            router_features,
            refiner_features,
            semantic_view_mode,
            logits_gnn,
            p_gnn,
            pred_gnn,
            labels_t,
            beta,
            test_idx,
            batch_size,
        )
        budget_values = self._router_budgets()
        train_budget_rows, train_budget_payloads = self._build_glance_joint_budget_rows(
            budget_values,
            train_idx,
            labels_np,
            labels_t,
            logits_gnn,
            p_gnn,
            pred_gnn,
            full_train_outputs["refiner_logits"],
            full_train_outputs["refiner_prob"],
            full_train_outputs["refiner_pred"],
            full_train_outputs["router_score"],
            full_train_outputs["router_prob"],
            full_train_outputs["oracle_advantage"],
        )
        valid_budget_rows, valid_budget_payloads = self._build_glance_joint_budget_rows(
            budget_values,
            valid_idx,
            labels_np,
            labels_t,
            logits_gnn,
            p_gnn,
            pred_gnn,
            full_valid_outputs["refiner_logits"],
            full_valid_outputs["refiner_prob"],
            full_valid_outputs["refiner_pred"],
            full_valid_outputs["router_score"],
            full_valid_outputs["router_prob"],
            full_valid_outputs["oracle_advantage"],
        )
        test_budget_rows, test_budget_payloads = self._build_glance_joint_budget_rows(
            budget_values,
            test_idx,
            labels_np,
            labels_t,
            logits_gnn,
            p_gnn,
            pred_gnn,
            full_test_outputs["refiner_logits"],
            full_test_outputs["refiner_prob"],
            full_test_outputs["refiner_pred"],
            full_test_outputs["router_score"],
            full_test_outputs["router_prob"],
            full_test_outputs["oracle_advantage"],
        )
        selected_valid_budget_row, selected_budget_key, _ = self._select_best_budget_row(valid_budget_rows)
        if selected_valid_budget_row is None or selected_budget_key is None:
            raise MissingFrozenArtifactError("glance_joint_router_refine failed to select a validation budget.")
        if selected_budget_key not in train_budget_payloads or selected_budget_key not in valid_budget_payloads or selected_budget_key not in test_budget_payloads:
            raise MissingFrozenArtifactError(
                "glance_joint_router_refine budget selection produced inconsistent train/valid/test payloads."
            )

        train_outputs = train_budget_payloads[selected_budget_key]["outputs"]
        valid_outputs = valid_budget_payloads[selected_budget_key]["outputs"]
        test_outputs = test_budget_payloads[selected_budget_key]["outputs"]

        def _row_to_metrics(row, split_size):
            return {
                "loss": float(row["loss"]),
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
                "bot_f1": float(row["bot_f1"]),
                "count": int(split_size),
                "routed_count": int(row["routed_count"]),
                "query_rate": float(row["query_rate"]),
            }

        train_metrics = _row_to_metrics(train_budget_payloads[selected_budget_key]["row"], train_idx.size)
        valid_metrics = _row_to_metrics(valid_budget_payloads[selected_budget_key]["row"], valid_idx.size)
        test_metrics = _row_to_metrics(test_budget_payloads[selected_budget_key]["row"], test_idx.size)

        train_rows = self._build_glance_joint_per_node_rows(
            train_idx,
            pred_gnn,
            train_outputs["pred"],
            labels_np,
            train_outputs["routed_mask"],
            train_outputs["router_prob"],
            train_outputs["router_score"],
            train_outputs["oracle_advantage"],
            semantic_views,
            "glance_joint_router_refine_train",
        )
        valid_rows = self._build_glance_joint_per_node_rows(
            valid_idx,
            pred_gnn,
            valid_outputs["pred"],
            labels_np,
            valid_outputs["routed_mask"],
            valid_outputs["router_prob"],
            valid_outputs["router_score"],
            valid_outputs["oracle_advantage"],
            semantic_views,
            "glance_joint_router_refine_valid",
        )
        test_rows = self._build_glance_joint_per_node_rows(
            test_idx,
            pred_gnn,
            test_outputs["pred"],
            labels_np,
            test_outputs["routed_mask"],
            test_outputs["router_prob"],
            test_outputs["router_score"],
            test_outputs["oracle_advantage"],
            semantic_views,
            "glance_joint_router_refine_test",
        )
        train_router_diag = _safe_binary_score_metrics(train_wrong, train_outputs["router_score"][torch.tensor(train_idx, dtype=torch.long)].numpy())
        valid_router_diag = _safe_binary_score_metrics(valid_wrong, valid_outputs["router_score"][torch.tensor(valid_idx, dtype=torch.long)].numpy())
        test_router_diag = _safe_binary_score_metrics(test_wrong, test_outputs["router_score"][torch.tensor(test_idx, dtype=torch.long)].numpy())
        train_adv_diag = _safe_advantage_router_metrics(
            train_outputs["oracle_advantage"][torch.tensor(train_idx, dtype=torch.long)].numpy(),
            train_outputs["router_score"][torch.tensor(train_idx, dtype=torch.long)].numpy(),
            train_outputs["routed_mask"][torch.tensor(train_idx, dtype=torch.long)].numpy(),
        )
        valid_adv_diag = _safe_advantage_router_metrics(
            valid_outputs["oracle_advantage"][torch.tensor(valid_idx, dtype=torch.long)].numpy(),
            valid_outputs["router_score"][torch.tensor(valid_idx, dtype=torch.long)].numpy(),
            valid_outputs["routed_mask"][torch.tensor(valid_idx, dtype=torch.long)].numpy(),
        )
        test_adv_diag = _safe_advantage_router_metrics(
            test_outputs["oracle_advantage"][torch.tensor(test_idx, dtype=torch.long)].numpy(),
            test_outputs["router_score"][torch.tensor(test_idx, dtype=torch.long)].numpy(),
            test_outputs["routed_mask"][torch.tensor(test_idx, dtype=torch.long)].numpy(),
        )

        return {
            "beta": float(beta),
            "eval_top_k": int(eval_top_k),
            "best_epoch": int(best_epoch_summary["epoch"]),
            "router_state": best_state["router"],
            "refiner_state": best_state["refiner"],
            "learned_view_gates": None,
            "fit_summary": {
                "best_epoch": int(best_epoch_summary["epoch"]),
                "epoch_history": epoch_history,
                "epoch_component_curves": epoch_component_curves,
                "component_curve_summary": component_curve_summary,
                "selected_train": train_metrics,
                "selected_valid": valid_metrics,
                "selected_test": test_metrics,
                "selected_budget_source": "validation",
                "selected_budget": float(selected_valid_budget_row["budget"]),
                "selected_budget_key": selected_budget_key,
                "router_weight": float(router_weight),
                "router_regression_weight": float(router_regression_weight),
                "router_ranking_weight": float(router_ranking_weight),
                "router_selection_weight": float(router_calibration_weight),
                "router_calibration_weight": float(router_calibration_weight),
                "entropy_weight": float(entropy_weight),
                "eval_top_k": int(eval_top_k),
                "router_diagnostics": {
                    "train": train_router_diag,
                    "valid": valid_router_diag,
                    "test": test_router_diag,
                },
                "advantage_router_diagnostics": {
                    "train": train_adv_diag,
                    "valid": valid_adv_diag,
                    "test": test_adv_diag,
                },
                "topk_reference": {
                    "train_metrics": topk_train_metrics,
                    "valid_metrics": topk_valid_metrics,
                    "test_metrics": topk_test_metrics,
                    "router_diagnostics": {
                        "train": topk_train_router_diag,
                        "valid": topk_valid_router_diag,
                        "test": topk_test_router_diag,
                    },
                    "advantage_router_diagnostics": {
                        "train": topk_train_adv_diag,
                        "valid": topk_valid_adv_diag,
                        "test": topk_test_adv_diag,
                    },
                    "analysis": {
                        "train": topk_train_rows["analysis"],
                        "valid": topk_valid_rows["analysis"],
                        "test": topk_test_rows["analysis"],
                    },
                },
            },
            "train_outputs": train_outputs,
            "valid_outputs": valid_outputs,
            "test_outputs": test_outputs,
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "train_rows": train_rows,
            "valid_rows": valid_rows,
            "test_rows": test_rows,
            "component_curves": epoch_component_curves,
            "component_curve_summary": component_curve_summary,
            "selected_budget": float(selected_valid_budget_row["budget"]),
            "selected_budget_key": selected_budget_key,
            "selected_budget_source": "validation",
            "selected_budget_metrics": {
                "train": train_metrics,
                "valid": valid_metrics,
                "test": test_metrics,
            },
            "selected_valid_budget_metrics": dict(selected_valid_budget_row),
            "selected_test_budget_metrics_under_valid_choice": dict(test_budget_payloads[selected_budget_key]["row"]),
            "train_budget_curve": train_budget_rows,
            "valid_budget_curve": valid_budget_rows,
            "test_budget_curve": test_budget_rows,
        }

    def _base_artifact_bundle(self):
        code_provenance = capture_code_metadata(Path(__file__).resolve().parents[2])
        return {
            "resolved_config": self._resolved_config_snapshot(),
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

    def _stage_dir_candidates(self, stage_name):
        return _stage_dir_read_candidates(self.experiment_root, stage_name)

    def _resolved_config_snapshot(self):
        snapshot = dict(vars(self.args))
        stage_spec = snapshot.get("stage_spec")
        if stage_spec is not None:
            snapshot["stage_spec"] = {
                "canonical_name": getattr(stage_spec, "canonical_name", None),
                "legacy_names": list(getattr(stage_spec, "legacy_names", ()) or ()),
                "visibility": getattr(stage_spec, "visibility", None),
                "family": getattr(stage_spec, "family", None),
                "runner_kind": getattr(stage_spec, "runner_kind", None),
                "claim_grade_allowed": getattr(stage_spec, "claim_grade_allowed", None),
                "requires_canonical_split": getattr(stage_spec, "requires_canonical_split", None),
                "forces_use_gnn": getattr(stage_spec, "forces_use_gnn", None),
                "graph_data_mode": getattr(stage_spec, "graph_data_mode", None),
                "artifact_namespace": getattr(stage_spec, "artifact_namespace", None),
            }
        return snapshot

    def _manifest_provenance(self, *, artifact_namespace=None, visibility=None, resolved_task=None):
        return {
            "canonical_task_name": resolved_task or self.execution_stage,
            "legacy_task_name_used": getattr(self.args, "legacy_task_name_used", None),
            "deprecated_cli_flags": list(getattr(self.args, "deprecated_cli_flags", [])),
            "stage_visibility": visibility or getattr(self.args, "stage_visibility", "public"),
            "artifact_namespace": artifact_namespace
            or f"stages/{self.execution_stage}",
            "invocation": {
                "requested_task": self.requested_task,
                "resolved_task": resolved_task or self.execution_stage,
            },
        }

    def _write_stage_manifest(self, stage_dir, manifest, *, artifact_namespace=None, visibility=None, resolved_task=None):
        payload = dict(manifest)
        payload.update(
            {
                key: value
                for key, value in self._manifest_provenance(
                    artifact_namespace=artifact_namespace,
                    visibility=visibility,
                    resolved_task=resolved_task,
                ).items()
                if key not in payload
            }
        )
        write_json(Path(stage_dir) / "manifest.json", payload)
        return payload

    def _read_stage_json(self, stage_name, filename):
        for candidate in self._stage_dir_candidates(stage_name):
            path = candidate / f"{filename}.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
        return None

    def _require_stage_json(self, stage_name, filename):
        candidates = [candidate / f"{filename}.json" for candidate in self._stage_dir_candidates(stage_name)]
        for path in candidates:
            if path.exists():
                with open(path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
        raise MissingFrozenArtifactError(
            f"Stage '{self.execution_stage}' requires frozen dependency {candidates[0]}. "
            f"Run the upstream task for '{stage_name}' first; strict stages never recompute upstream artifacts."
        )

    def _require_stage_tensor(self, stage_name, filename):
        candidates = [candidate / f"{filename}.pt" for candidate in self._stage_dir_candidates(stage_name)]
        for path in candidates:
            if path.exists():
                return safe_torch_load(path, map_location="cpu")
        raise MissingFrozenArtifactError(
            f"Stage '{self.execution_stage}' requires frozen dependency {candidates[0]}. "
            f"Run the upstream task for '{stage_name}' first; strict stages never recompute upstream artifacts."
        )

    def _require_stage_jsonl(self, stage_name, filename):
        candidates = [candidate / f"{filename}.jsonl" for candidate in self._stage_dir_candidates(stage_name)]
        for path in candidates:
            if path.exists():
                with open(path, "r", encoding="utf-8") as handle:
                    return [json.loads(line) for line in handle if line.strip()]
        raise MissingFrozenArtifactError(
            f"Stage '{self.execution_stage}' requires frozen dependency {candidates[0]}. "
            f"Run the upstream task for '{stage_name}' first; strict stages never recompute upstream artifacts."
        )

    def _load_vertical_minimal_dependency(self):
        return {
            "risk_manifest": self._require_stage_json("minimal_pipeline", "risk_manifest"),
            "structural_manifest": self._require_stage_json("minimal_pipeline", "structural_manifest"),
            "subgroup_manifest_operational": self._require_stage_json("minimal_pipeline", "subgroup_manifest_operational"),
            "subgroup_manifest_analysis": self._require_stage_json("minimal_pipeline", "subgroup_manifest_analysis"),
            "survivor_manifest": self._require_stage_json("minimal_pipeline", "survivor_manifest"),
            "all_node_outputs": self._require_stage_tensor("minimal_pipeline", "all_node_outputs"),
        }

    def _load_final_output_dependency(self):
        return {
            "all_node_outputs": self._require_stage_tensor("minimal_pipeline", "all_node_outputs"),
            "survivor_manifest": self._require_stage_json("minimal_pipeline", "survivor_manifest"),
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

    def _stage2_structural_graph(self):
        if str(getattr(self, "graph_data_variant", "labeled")).lower() != "full_graph_support":
            return self.data["edge_index"], self.data["edge_type"], int(len(self.labels))
        dataset_path = resolve_dataset_path(self.args.dataset)
        edge_index = safe_torch_load(dataset_path / "edge_index.pt", map_location="cpu")
        edge_type = safe_torch_load(dataset_path / "edge_type.pt", map_location="cpu")
        return edge_index, edge_type, int(len(self.labels))

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
        return estimator_mode == "glance_for_context_residual_risk_selector"

    def _login_uncertainty_router_requested(self):
        return str(getattr(self.args, "estimator_mode", "none")).lower() == "login_uncertainty_router"

    def _login_uncertainty_aux_root(self, stage_dir):
        return ensure_dir(Path(stage_dir) / "aux_login_uncertainty_runs")

    def _build_or_load_login_uncertainty_logits_list(self, stage_dir):
        aux_root = self._login_uncertainty_aux_root(stage_dir)
        context = self.ensure_backbone_context()
        frozen_manifest = context["frozen_g0"].get("manifest", {})
        feature_manifest = frozen_manifest.get("feature_manifest", {})
        feature_path = feature_manifest.get("path")
        projected_dim = feature_manifest.get("projected_dim")
        projector = feature_manifest.get("projector")
        peft_manifest = feature_manifest.get("peft", {})
        drop_out_list = [0.5, 0.5, 0.5, 0.5, 0.5]
        logits_list = []
        aux_runs = []
        force_retrain = bool(getattr(self.args, "force_retrain_backbone", False))
        for run_idx, dropout in enumerate(drop_out_list):
            run_root = ensure_dir(aux_root / f"run_{run_idx}")
            aux_args = SimpleNamespace(**vars(self.args))
            aux_args.GNN_dropout = float(dropout)
            aux_args.force_retrain_backbone = force_retrain
            if feature_path:
                aux_args.emb_path = str(feature_path)
                aux_args.g0_feature_path = str(feature_path)
            if projected_dim is not None:
                aux_args.phase_a_project_dim = int(projected_dim)
            if projector is not None:
                aux_args.phase_a_projector = str(projector)
            aux_args.peft = bool(peft_manifest.get("enabled", False))
            if aux_args.peft:
                aux_args.peft_rank = int(peft_manifest.get("rank", getattr(self.args, "peft_rank", 8)))
                aux_args.peft_alpha = float(peft_manifest.get("alpha", getattr(self.args, "peft_alpha", 16.0)))
            train_seed = int(self.seed) * 100 + int(run_idx)
            had_existing = (run_root / "frozen" / "g0" / "manifest.json").exists() and not force_retrain
            frozen = build_or_load_frozen_g0(aux_args, train_seed, self.data, run_root)
            logits = frozen["outputs"].get("logits")
            if logits is None or not torch.is_tensor(logits) or logits.dim() != 2:
                raise MissingFrozenArtifactError(
                    f"LOGIN uncertainty aux run {run_idx} under {run_root} did not produce full-node logits."
                )
            logits_list.append(logits.detach().cpu().float())
            aux_runs.append(
                {
                    "run_index": int(run_idx),
                    "dropout": float(dropout),
                    "training_seed": int(train_seed),
                    "source": "existing_frozen_g0" if had_existing else "trained_frozen_g0",
                    "root": str(run_root),
                    "frozen_g0_dir": str(frozen["dir"]),
                    "manifest_path": str(Path(frozen["dir"]) / "manifest.json"),
                    "outputs_path": str(Path(frozen["dir"]) / "outputs.pt"),
                    "checkpoint_path": str(Path(frozen["dir"]) / "checkpoint.pt"),
                    "selection_metrics_path": str(Path(frozen["dir"]) / "selection_metrics.json"),
                    "selection_metrics": frozen.get("selection_metrics", {}),
                }
            )
        logits_stack = torch.stack(logits_list, dim=0)
        aux_manifest = {
            "contract": "login_uncertainty_aux_runs_v1",
            "official_repo_path": r"G:\Research\BotDetection\LOGIN_init",
            "official_submodule_scope": "node_selection_uncertainty_only",
            "official_drop_out_list": list(drop_out_list),
            "official_pl_rate": 0.1,
            "run_count": int(len(drop_out_list)),
            "run_shape": [int(item) for item in logits_stack.shape],
            "dataset": str(getattr(self.args, "dataset", "unknown")),
            "gnn_backbone": str(getattr(self.args, "GNN_model", "unknown")),
            "feature_path": str(feature_path or getattr(self.args, "emb_path", getattr(self.args, "g0_feature_path", "")) or ""),
            "split_provenance": split_provenance(self.data),
            "force_retrain_backbone": force_retrain,
            "local_seed_policy": "base_seed_x100_plus_run_idx_for_independent_aux_training",
            "note": (
                "Official LOGIN uncertainty code trains five GNNs inside one loop. "
                "Local LLMbot reproduction uses deterministic seed offsets so the five aux runs remain independent "
                "while preserving the same dataset split, backbone family, and input features."
            ),
            "runs": aux_runs,
        }
        return logits_stack, aux_manifest

    def _graph_conformal_estimator_requested(self):
        return str(getattr(self.args, "estimator_mode", "none")).lower() in {
            "msp_ts",
            "posthoc_calibrated_ranker",
            "calibrated_local_risk_router",
            "conformal_knn_risk_router",
            "graph_conformal_set_estimator",
            "gnn_2hop_conformal",
        }

    @staticmethod
    def _index_sha256(idx):
        arr = np.asarray(idx, dtype=np.int64).reshape(-1)
        digest = hashlib.sha256()
        digest.update(str(tuple(arr.shape)).encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(arr.tobytes())
        return digest.hexdigest()

    def _split_valid_tune_cal(self, labels):
        labels_np = _labels_to_index(labels).detach().cpu().numpy()
        valid_idx = _idx_numpy(self.data["valid_idx"]).astype(np.int64)
        valid_idx = valid_idx[(valid_idx >= 0) & (valid_idx < labels_np.shape[0])]
        rng = np.random.default_rng(int(self.seed))
        tune_parts = []
        cal_parts = []
        fallback = False
        for class_id in sorted(np.unique(labels_np[valid_idx]).astype(int).tolist()) if valid_idx.size else []:
            class_idx = valid_idx[labels_np[valid_idx] == int(class_id)]
            class_idx = np.asarray(class_idx, dtype=np.int64)
            rng.shuffle(class_idx)
            if class_idx.size < 2:
                fallback = True
                continue
            split = class_idx.size // 2
            tune_parts.append(np.sort(class_idx[:split]))
            cal_parts.append(np.sort(class_idx[split:]))
        if not tune_parts or not cal_parts:
            fallback = True
        tune_idx = np.sort(np.concatenate(tune_parts)) if tune_parts else np.asarray([], dtype=np.int64)
        cal_idx = np.sort(np.concatenate(cal_parts)) if cal_parts else np.asarray([], dtype=np.int64)
        if fallback or tune_idx.size == 0 or cal_idx.size == 0:
            ordered = np.sort(valid_idx)
            tune_idx = ordered[::2]
            cal_idx = ordered[1::2]
            if tune_idx.size == 0 and ordered.size:
                tune_idx = ordered[:1]
            if cal_idx.size == 0 and ordered.size > 1:
                cal_idx = ordered[1:]
            elif cal_idx.size == 0:
                cal_idx = ordered[:1]
        metadata = {
            "split_policy": "label_stratified_50_50_valid_split",
            "split_fallback": bool(fallback),
            "split_seed": int(self.seed),
            "valid_count": int(valid_idx.size),
            "tune_count": int(tune_idx.size),
            "calibration_count": int(cal_idx.size),
            "valid_idx_sha256": self._index_sha256(valid_idx),
            "tune_idx_sha256": self._index_sha256(tune_idx),
            "cal_idx_sha256": self._index_sha256(cal_idx),
            "tune_cal_disjoint": bool(set(tune_idx.tolist()).isdisjoint(set(cal_idx.tolist()))),
            "test_labels_used_for_threshold": False,
        }
        return torch.tensor(tune_idx, dtype=torch.long), torch.tensor(cal_idx, dtype=torch.long), metadata

    @staticmethod
    def _split_metrics_from_outputs(outputs, labels_np, valid_mask, test_mask):
        pred = outputs.get("pred")
        if pred is None:
            raise MissingFrozenArtifactError("Expected outputs.pt to include 'pred' for local_conformal_prune_diag.")
        pred_np = pred.detach().cpu().numpy().astype(np.int64) if torch.is_tensor(pred) else np.asarray(pred, dtype=np.int64)
        labels_np = np.asarray(labels_np, dtype=np.int64)
        return {
            "valid": _score_all(labels_np[valid_mask], pred_np[valid_mask]) if valid_mask.any() else {"accuracy": 0.0, "macro_f1": 0.0, "bot_f1": 0.0, "count": 0},
            "test": _score_all(labels_np[test_mask], pred_np[test_mask]) if test_mask.any() else {"accuracy": 0.0, "macro_f1": 0.0, "bot_f1": 0.0, "count": 0},
            "pred": pred_np,
        }

    @staticmethod
    def _degree_arrays(edge_index, num_nodes):
        edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
        src = edge_index_cpu[0]
        dst = edge_index_cpu[1]
        out_degree = torch.bincount(src, minlength=int(num_nodes)).cpu().numpy().astype(np.int64)
        in_degree = torch.bincount(dst, minlength=int(num_nodes)).cpu().numpy().astype(np.int64)
        return in_degree, out_degree

    @staticmethod
    def _remove_edge_at_position(edge_index, edge_type, edge_pos):
        edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
        edge_type_cpu = edge_type.detach().cpu().long() if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long)
        keep_mask = torch.ones(edge_index_cpu.size(1), dtype=torch.bool)
        keep_mask[int(edge_pos)] = False
        return edge_index_cpu[:, keep_mask].contiguous(), edge_type_cpu[keep_mask].contiguous()

    @staticmethod
    def _pruned_graph_from_removed_positions(edge_index, edge_type, removed_positions):
        edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
        edge_type_cpu = edge_type.detach().cpu().long() if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long)
        if not removed_positions:
            return edge_index_cpu.contiguous(), edge_type_cpu.contiguous()
        keep_mask = torch.ones(edge_index_cpu.size(1), dtype=torch.bool)
        keep_mask[torch.tensor(sorted(int(pos) for pos in removed_positions), dtype=torch.long)] = False
        return edge_index_cpu[:, keep_mask].contiguous(), edge_type_cpu[keep_mask].contiguous()

    def _fit_local_conformal_router(self, logits_gnn, prob_gnn, labels_t, edge_index, edge_type, node_repr):
        train_idx = _idx_tensor(self.data["train_idx"])
        valid_idx = _idx_tensor(self.data["valid_idx"])
        test_idx = _idx_tensor(self.data["test_idx"])
        tune_idx, cal_idx, split_metadata = self._split_valid_tune_cal(labels_t)
        estimator = build_estimator("gnn_2hop_conformal")
        routed_budget = 0.10
        estimator.fit(
            logits_gnn,
            prob_gnn,
            labels_t,
            train_idx=train_idx,
            val_idx=valid_idx,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            tune_idx=tune_idx,
            cal_idx=cal_idx,
            tune_cal_split_metadata=split_metadata,
            budgets=[routed_budget],
            direction_mode="incoming",
            relation_mode="agnostic",
        )
        risk_manifest = estimator.build_manifest(
            logits_gnn,
            prob_gnn,
            labels_t,
            val_idx=valid_idx,
            test_idx=test_idx,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            budgets=[routed_budget],
            direction_mode="incoming",
            relation_mode="agnostic",
        )
        q_score = np.asarray(risk_manifest.get("router_score"), dtype=np.float32)
        threshold = float((risk_manifest.get("thresholds") or {}).get("validation_risk_threshold", float("inf")))
        routed_mask = q_score >= (threshold - 1e-12)
        return {
            "estimator": estimator,
            "risk_manifest": risk_manifest,
            "q_score": q_score,
            "threshold": threshold,
            "routed_mask": routed_mask,
            "routed_budget": routed_budget,
            "tune_idx": tune_idx,
            "cal_idx": cal_idx,
            "split_metadata": split_metadata,
        }

    def _rerun_frozen_g0_with_edge_override(self, stage_dir, run_name, edge_index, edge_type):
        rerun_root = ensure_dir(Path(stage_dir) / "reruns" / run_name)
        rerun_args = SimpleNamespace(**vars(self.args))
        rerun_args.experiment_task = "graph_detector_prepare"
        rerun_args.requested_experiment_task = "graph_detector_prepare"
        rerun_args.stage = "graph_detector_prepare"
        rerun_args.requested_stage = "graph_detector_prepare"
        rerun_args.force_retrain_backbone = True
        rerun_args.graph_refine_mode = "none"
        rerun_args.graph_refine_budget = 0.0
        rerun_args.external_frozen_g0_root = None
        rerun_data = dict(self.data)
        rerun_data["edge_index"] = edge_index.detach().cpu().long()
        rerun_data["edge_type"] = edge_type.detach().cpu().long()
        return build_or_load_frozen_g0(rerun_args, self.seed, rerun_data, rerun_root)

    def _router_budgets(self):
        return parse_budget_list(getattr(self.args, "risk_budgets", None) or getattr(self.args, "router_budgets", None), default=RESIDUAL_RISK_PAPER_BUDGETS)

    def _router_oof_dir(self):
        return ensure_dir(Path(self.experiment_root) / "runtime" / "router_oof")

    def _router_oof_dir_candidates(self):
        canonical_dir = Path(self.experiment_root) / "runtime" / "router_oof"
        legacy_dir = Path(self.experiment_root) / "frozen" / "router_oof"
        candidates = [canonical_dir]
        if legacy_dir != canonical_dir:
            candidates.append(legacy_dir)
        return candidates

    def _load_router_oof_artifact(self):
        if str(getattr(self.args, "router_oof_mode", "artifact")).lower() != "artifact":
            raise MissingFrozenArtifactError("Stage 2 router only supports --router_oof_mode artifact in this implementation.")
        selected_dir = None
        manifest_path = None
        outputs_path = None
        for candidate_dir in self._router_oof_dir_candidates():
            candidate_manifest = candidate_dir / "manifest.json"
            candidate_outputs = candidate_dir / "outputs.pt"
            if candidate_manifest.exists() and candidate_outputs.exists():
                selected_dir = candidate_dir
                manifest_path = candidate_manifest
                outputs_path = candidate_outputs
                break
        if selected_dir is None:
            oof_dir = self._router_oof_dir()
            raise MissingFrozenArtifactError(
                "glance_for_context_residual_risk_selector requires frozen train-split OOF artifacts under "
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
        return {"manifest": manifest, "outputs": outputs, "dir": selected_dir}

    def _lm_only_head_dir(self):
        return ensure_dir(Path(self.experiment_root) / "runtime" / "lm_only_head")

    def _lm_only_head_dir_candidates(self):
        canonical_dir = Path(self.experiment_root) / "runtime" / "lm_only_head"
        legacy_dir = Path(self.experiment_root) / "frozen" / "lm_only_head"
        candidates = [canonical_dir]
        if legacy_dir != canonical_dir:
            candidates.append(legacy_dir)
        return candidates

    def _load_lm_only_head(self):
        for out_dir in self._lm_only_head_dir_candidates():
            manifest_path = out_dir / "manifest.json"
            outputs_path = out_dir / "outputs.pt"
            if manifest_path.exists() and outputs_path.exists():
                return {
                    "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
                    "outputs": safe_torch_load(outputs_path, map_location="cpu"),
                    "dir": out_dir,
                    "source": "existing_semantic_head_runtime_artifact",
                }
        return None

    def _build_or_load_lm_only_head(self):
        existing = self._load_lm_only_head()
        if existing is not None:
            return existing
        if bool(getattr(self.args, "claim_grade", False)):
            raise MissingFrozenArtifactError(
                "claim_grade GLANCE context selector requires frozen/lm_only_head artifacts; "
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

    def _build_glance_context_router_bundle(self, gats_bundle):
        oof = self._load_router_oof_artifact()
        lm_bundle = self._build_or_load_lm_only_head()
        context = self.ensure_backbone_context()
        gnn_outputs = context["gnn_outputs"]
        gats_outputs = gats_bundle["outputs"]
        gnn_prob_cal = gats_outputs.get("prob_graph_cal", gnn_outputs["prob"]).detach().cpu().float()
        lm_prob_cal = lm_bundle["outputs"].get("prob_cal", lm_bundle["outputs"]["prob"]).detach().cpu().float()
        budgets = self._router_budgets()
        estimator_mode = str(getattr(self.args, "estimator_mode", "none")).lower()
        if estimator_mode != "glance_for_context_residual_risk_selector":
            raise ValueError(f"Unsupported router estimator mode after cleanup: {estimator_mode}")
        estimator = GlanceForContextResidualRiskSelector(
            budgets=budgets,
            training_objective=getattr(self.args, "glance_training_objective", "residual_error"),
            llm_query_cost=getattr(self.args, "glance_llm_query_cost", 0.2),
        )
        paper_identity = "GLANCE-inspired Node-Aware Residual-Risk Selector"
        artifact_mode = estimator_mode
        glance_variant = "glance_for_context"
        legacy_variant = getattr(estimator, "variant", str(glance_variant))
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
            "glance_variant": str(glance_variant),
            "glance_legacy_variant": str(legacy_variant),
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
            "contract": "glance_context_residual_router_v1",
            "estimator_mode": artifact_mode,
            "router_mode": "off",
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
        selector_manifest = {
            "contract": "glance_context_residual_risk_selector_v1",
            "paper_name": paper_identity,
            "estimator_mode": artifact_mode,
            "glance_variant": str(glance_variant),
            "glance_legacy_variant": str(legacy_variant),
            "training_label": "z_i = 1[base_gnn_prediction != y_i]",
            "target": "P(Stage1 prediction is wrong | Stage1 outputs/features)",
            "training_objective": estimator.training_summary.get("training_objective", "residual_error"),
            "target_semantics": estimator.training_summary.get("target_semantics", "1[Stage1 prediction != y]"),
            "input_boundary": estimator.training_summary.get("forbidden_inputs_audit", {}),
            "oof_required": True,
            "oof_folds": int(getattr(self.args, "router_oof_folds", 5)),
            "screening_only": True,
            "diagnosis_or_action": False,
            "positioning": "GLANCE-inspired calibration/ranking selector; not a Stage 3 action router",
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

    def _fit_local_conflict_structural_view(self, edge_index, edge_type, labels_t):
        train_idx = _idx_tensor(self.data["train_idx"])
        valid_idx = _idx_tensor(self.data["valid_idx"])
        test_idx = _idx_tensor(self.data["test_idx"])
        if train_idx.numel() == 0 or valid_idx.numel() == 0:
            raise MissingFrozenArtifactError(
                "local_conflict_prune_diag requires non-empty train_idx and valid_idx for structural-view training."
            )

        num_nodes = int(labels_t.numel())
        relation_cardinality = int(edge_type.max().item()) + 1 if edge_type.numel() else int(getattr(self.args, "n_relations", 2))
        model = StructuralConflictRGCN(
            num_nodes=num_nodes,
            node_emb_dim=64,
            hidden_dim=64,
            n_relations=max(int(relation_cardinality), 1),
            n_layers=2,
            dropout=0.1,
        ).to(self.device)
        edge_index_dev = edge_index.to(self.device)
        edge_type_dev = edge_type.to(self.device)
        labels_dev = labels_t.to(self.device)
        train_idx_dev = train_idx.to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        patience = 10
        max_epochs = 100
        best_state = None
        best_metrics = None
        history = []
        wait = 0

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            train_outputs = model.forward_outputs(edge_index_dev, edge_type_dev)
            loss = F.cross_entropy(train_outputs["logits"][train_idx_dev], labels_dev[train_idx_dev])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                eval_outputs = model.forward_outputs(edge_index_dev, edge_type_dev)
            valid_metrics = _score_logits(eval_outputs["logits"].detach().cpu(), labels_t.detach().cpu(), valid_idx.cpu())
            valid_metrics["epoch"] = int(epoch)
            history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(loss.detach().cpu().item()),
                    "valid_accuracy": float(valid_metrics["accuracy"]),
                    "valid_macro_f1": float(valid_metrics["macro_f1"]),
                    "valid_loss": float(valid_metrics["loss"]),
                }
            )
            if _is_better(valid_metrics, best_metrics):
                best_metrics = valid_metrics
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is None:
            raise MissingFrozenArtifactError("local_conflict_prune_diag failed to train a structural-only auxiliary view.")

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            final_outputs = model.forward_outputs(edge_index_dev, edge_type_dev)
        return {
            "model": model,
            "outputs": {key: value.detach().cpu() for key, value in final_outputs.items()},
            "checkpoint": {key: value.detach().cpu().clone() for key, value in best_state.items()},
            "fit_summary": {
                "max_epochs": int(max_epochs),
                "patience": int(patience),
                "best_epoch": int(best_metrics.get("epoch", -1) if best_metrics else -1),
                "history": history,
            },
            "metrics": {
                "train": _score_logits(final_outputs["logits"].detach().cpu(), labels_t.detach().cpu(), train_idx.cpu()),
                "valid": _score_logits(final_outputs["logits"].detach().cpu(), labels_t.detach().cpu(), valid_idx.cpu()),
                "test": _score_logits(final_outputs["logits"].detach().cpu(), labels_t.detach().cpu(), test_idx.cpu()),
            },
            "model_config": {
                "type": "structural_conflict_rgcn",
                "node_emb_dim": 64,
                "hidden_dim": 64,
                "n_layers": 2,
                "n_relations": max(int(relation_cardinality), 1),
                "dropout": 0.1,
                "optimizer": "adamw",
                "learning_rate": 1e-3,
                "weight_decay": 1e-5,
                "checkpoint_selection": {
                    "primary": "validation_macro_f1",
                    "tie_breaker": "validation_loss",
                },
            },
        }

    def _run_structural_view_forward(self, model, edge_index, edge_type):
        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            outputs = model.forward_outputs(edge_index.to(self.device), edge_type.to(self.device))
        return {key: value.detach().cpu() for key, value in outputs.items()}

    def _fit_local_conflict_router(self, semantic_prob, structural_prob):
        valid_idx = _idx_numpy(self.data["valid_idx"])
        test_idx = _idx_numpy(self.data["test_idx"])
        budget = float(getattr(self.args, "conflict_router_budget", 0.10) or 0.10)
        if not (0.0 < budget < 1.0):
            raise ValueError("--conflict_router_budget must be in (0, 1) for local_conflict_prune_diag.")
        conflict_score = _js_divergence_from_probs(structural_prob, semantic_prob).detach().cpu().numpy().astype(np.float32)
        valid_scores = conflict_score[valid_idx]
        if valid_scores.size == 0:
            raise MissingFrozenArtifactError("local_conflict_prune_diag requires non-empty valid conflict scores.")
        routed_count_valid = max(int(round(valid_scores.size * budget)), 1)
        ranked_valid = np.sort(valid_scores)[::-1]
        threshold = float(ranked_valid[min(routed_count_valid - 1, ranked_valid.size - 1)])
        routed_mask = conflict_score >= (threshold - 1e-12)
        return {
            "conflict_score": conflict_score,
            "threshold": threshold,
            "routed_mask": routed_mask,
            "budget": budget,
            "routed_count_valid_target": int(routed_count_valid),
            "test_conflict_summary": {
                "mean": float(conflict_score[test_idx].mean()) if test_idx.size else 0.0,
                "median": float(np.median(conflict_score[test_idx])) if test_idx.size else 0.0,
                "max": float(conflict_score[test_idx].max()) if test_idx.size else 0.0,
            },
        }

    @staticmethod
    def _first_available_tensor(mapping, keys):
        for key in keys:
            value = mapping.get(key)
            if value is not None:
                return value
        return None

    def _build_login_uncertainty_router_bundle(self, stage_dir):
        context = self.ensure_backbone_context()
        frozen = context["frozen_g0"]
        gnn_outputs = context["gnn_outputs"]
        budgets = self._router_budgets()
        estimator = build_estimator("login_uncertainty_router")
        logits_list, aux_runs_manifest = self._build_or_load_login_uncertainty_logits_list(stage_dir)
        labels = torch.tensor(self.labels, dtype=torch.long)
        estimator.fit(
            logits=gnn_outputs.get("logits"),
            probs=gnn_outputs.get("prob"),
            labels=labels,
            train_idx=self.data["train_idx"],
            val_idx=self.data["valid_idx"],
            test_idx=self.data["test_idx"],
            logits_list=logits_list,
            budgets=budgets,
        )
        risk_manifest = estimator.build_manifest(
            logits=gnn_outputs.get("logits"),
            probs=gnn_outputs.get("prob"),
            labels=labels,
            val_idx=self.data["valid_idx"],
            test_idx=self.data["test_idx"],
            logits_list=logits_list,
            budgets=budgets,
        )
        frozen_manifest = frozen.get("manifest", {})
        feature_manifest = frozen_manifest.get("feature_manifest", {})
        risk_manifest["calibration_metadata"] = {
            **risk_manifest.get("calibration_metadata", {}),
            "source": "login_uncertainty_router",
            "paper_identity": "LOGIN_init Node-Selection Uncertainty Router",
            "paper_identity_risk": "uncertainty_submodule_only",
            "promotion_rule": "ablation_only_never_canonical",
            "simteg_like_input": "phase_a_cached_semantic_embedding_to_frozen_gnn",
            "semantic_source": feature_manifest.get("source", "unknown"),
            "semantic_feature_path": feature_manifest.get("path"),
            "embedding_manifest": feature_manifest,
            "gnn_backbone": frozen_manifest.get("backbone", getattr(self.args, "GNN_model", "unknown")),
            "frozen_g0_manifest": str(Path(frozen["dir"]) / "manifest.json"),
            "split_provenance": frozen_manifest.get("split_provenance", {}),
            "official_code_verified": False,
            "paper_aligned": True,
            "repo_locally_verified": False,
            "verified_scope": "node_selection_uncertainty_only",
            "official_repo_path": r"G:\Research\BotDetection\LOGIN_init",
            "official_submodule_scope": "node_selection_uncertainty_only",
            "official_formula": "var(stack(logits_list), dim=0).sum(dim=1)",
            "official_selection_contract": "global_topk_by_pl_rate",
            "official_drop_out_list": [0.5, 0.5, 0.5, 0.5, 0.5],
            "official_pl_rate": 0.1,
            "full_LOGIN_loop": False,
            "prompt_llm": False,
            "feature_update": False,
            "edge_pruning": False,
            "analysis_reference_posterior_source": "frozen_g0_current_run",
            "best_index_unused": True,
            "auxiliary_budget_analysis": True,
            "auxiliary_runs_manifest_path": str(Path(stage_dir) / "login_uncertainty_aux_runs.json"),
            "logits_list_shape": [int(item) for item in logits_list.shape],
        }
        bundle = self._bundle_from_risk_manifest("login_uncertainty_router", risk_manifest)
        test_idx = _idx_numpy(self.data["test_idx"])
        probs = gnn_outputs["prob"].detach().cpu()
        wrong = (probs.argmax(dim=1).numpy() != self.labels).astype(np.int32)
        risk_score = np.asarray(risk_manifest["risk_score"], dtype=np.float32)
        official_pl_mask = np.asarray(risk_manifest.get("official_pl_mask", np.zeros(len(self.labels), dtype=bool)), dtype=bool)
        login_budget_rows = router_budget_curve(wrong[test_idx], risk_score[test_idx], budgets=budgets)
        official_test_mask = official_pl_mask & self.test_mask
        official_test_selected = int(official_test_mask.sum())
        base_wrong_total = int(wrong[test_idx].sum())
        official_wrong_hits = int(wrong[official_test_mask].sum()) if official_test_selected else 0
        official_precision = float(official_wrong_hits / official_test_selected) if official_test_selected else 0.0
        random_error_rate = float(wrong[test_idx].mean()) if test_idx.size else 0.0
        official_error_recall = float(official_wrong_hits / base_wrong_total) if base_wrong_total else 0.0
        official_lift = float(official_precision / random_error_rate) if random_error_rate > 0 else 0.0
        login_metrics = {
            "validation": risk_manifest.get("validation_metrics", {}),
            "test": risk_manifest.get("test_metrics", {}),
            "budget_curve": login_budget_rows,
            "official_selection": {
                "selection_contract": "global_topk_by_pl_rate",
                "pl_rate": 0.1,
                "global_selected_count": int(official_pl_mask.sum()),
                "test_selected_count": official_test_selected,
                "test_wrong_hits": official_wrong_hits,
                "test_error_recall": official_error_recall,
                "test_precision": official_precision,
                "test_lift": official_lift,
            },
            "official_code_verified": False,
            "paper_aligned": True,
            "repo_locally_verified": False,
            "verified_scope": "node_selection_uncertainty_only",
            "auxiliary_budget_analysis": True,
        }
        login_manifest = {
            "contract": "login_uncertainty_router_v2",
            "estimator_mode": "login_uncertainty_router",
            "target": "official_LOGIN_init_uncertainty_score",
            "training_label": "none_official_logit_variance_selector",
            "input_boundary": risk_manifest["calibration_metadata"].get("input_boundary", {}),
            "simteg_like_input": "phase_a_cached_semantic_embedding_to_frozen_gnn",
            "semantic_source": risk_manifest["calibration_metadata"].get("semantic_source"),
            "gnn_backbone": risk_manifest["calibration_metadata"].get("gnn_backbone"),
            "screening_only": True,
            "diagnosis_or_action": False,
            "llm_call": False,
            "login_scope": "node_selection_uncertainty_only",
            "official_code_verified": False,
            "paper_aligned": True,
            "repo_locally_verified": False,
            "verified_scope": "node_selection_uncertainty_only",
            "official_repo_path": r"G:\Research\BotDetection\LOGIN_init",
            "official_submodule_scope": "node_selection_uncertainty_only",
            "official_formula": "var(stack(logits_list), dim=0).sum(dim=1)",
            "official_selection_contract": "global_topk_by_pl_rate",
            "official_drop_out_list": [0.5, 0.5, 0.5, 0.5, 0.5],
            "official_pl_rate": 0.1,
            "full_LOGIN_loop": False,
            "prompt_llm": False,
            "feature_update": False,
            "edge_pruning": False,
            "auxiliary_budget_analysis": True,
            "analysis_reference_posterior_source": "frozen_g0_current_run",
            "best_index_unused": True,
            "frozen_g0_manifest": str(Path(frozen["dir"]) / "manifest.json"),
            "budgets": [float(item) for item in budgets],
            "official_pl_mask_count": int(official_pl_mask.sum()),
            "official_pl_mask_test_count": official_test_selected,
        }
        login_artifacts = {
            "login_router_manifest": login_manifest,
            "login_router_metrics": login_metrics,
            "login_router_scores.pt": torch.tensor(risk_score, dtype=torch.float32),
            "login_router_checkpoint.pt": estimator.state_dict_payload(),
            "login_uncertainty_logits_list.pt": logits_list.detach().cpu().float(),
            "login_uncertainty_official_mask.pt": torch.tensor(official_pl_mask, dtype=torch.bool),
            "login_uncertainty_aux_runs": aux_runs_manifest,
        }
        return bundle, login_artifacts, login_budget_rows

    def _frozen_g0_second_view_config(self):
        context = self.ensure_backbone_context()
        manifest = context.get("frozen_g0", {}).get("manifest", {}) or {}
        graph_refine = manifest.get("graph_refine", {}) if isinstance(manifest.get("graph_refine", {}), dict) else {}
        second_view = manifest.get("second_view", {}) if isinstance(manifest.get("second_view", {}), dict) else {}
        detector = manifest.get("detector", {}) if isinstance(manifest.get("detector", {}), dict) else {}
        mode = str(graph_refine.get("mode", "") or "").strip().lower()
        scope = str(
            second_view.get("scope", graph_refine.get("second_view_scope", ""))
            or ""
        ).strip().lower()
        candidate_scope = str(
            second_view.get("candidate_scope", graph_refine.get("candidate_scope", ""))
            or ""
        ).strip().lower()
        training_geometry = str(
            second_view.get(
                "training_geometry",
                graph_refine.get("training_geometry", graph_refine.get("training_loader_mode", "")),
            )
            or ""
        ).strip().lower()
        active_modes = {
            "hyperscan_knn_hypergraph_proxy_augment",
            "relation_overlap_knn_proxy_augment",
            "relation_overlap_knn_repr_prefit_augment",
            "routed_dynamic_hyperscan_branch",
            "hyperscan_neighborloader_batch_local_branch",
        }
        active_scopes = {"labeled_prefix", "routed_nodes", "neighborloader_batch"}
        router_candidate_scope = None
        candidate_alignment_note = ""
        if candidate_scope in {"labeled_relation_1hop", "undirected_relation_1hop"}:
            router_candidate_scope = candidate_scope
            candidate_alignment_note = "router uses the same relation-local candidate scope as the Hyper KNN branch"
        elif candidate_scope == "batch_local_subgraph_knn" or mode == "hyperscan_neighborloader_batch_local_branch":
            router_candidate_scope = "labeled_full"
            candidate_alignment_note = (
                "training branch uses NeighborLoader batch-local KNN; post-hoc router cannot replay stochastic "
                "sampled batches, so it inherits k/x_new and uses deterministic labeled_full support for the "
                "same labeled graph"
            )
        elif mode == "hyperscan_knn_hypergraph_proxy_augment":
            router_candidate_scope = "hyperscan_full"
            candidate_alignment_note = "proxy hypergraph augmentation used full feature-pool KNN"
        knn_k = graph_refine.get("knn_k", second_view.get("knn_k"))
        return {
            "active": bool(mode in active_modes or scope in active_scopes),
            "mode": mode,
            "scope": scope,
            "candidate_scope": candidate_scope,
            "router_candidate_scope": router_candidate_scope,
            "candidate_alignment_note": candidate_alignment_note,
            "training_geometry": training_geometry,
            "knn_k": knn_k,
            "hypergraph_backend": str(
                second_view.get(
                    "hypergraph_backend",
                    detector.get("graph_second_view_hypergraph_backend", detector.get("hypergraph_backend", "")),
                )
                or ""
            ),
            "fusion": str(
                second_view.get(
                    "fusion",
                    detector.get("graph_second_view_fusion", detector.get("hyperscan_detector_style", "")),
                )
                or ""
            ),
            "artifact_dir": str(context.get("frozen_g0", {}).get("dir", "")),
        }

    def _effective_conformal_knn_config(self, estimator_mode):
        config = {
            "conformal_knn_k": int(getattr(self.args, "conformal_knn_k", 8)),
            "conformal_knn_candidate_scope": str(getattr(self.args, "conformal_knn_candidate_scope", "labeled_full")),
            "conformal_knn_target_top_n": int(getattr(self.args, "conformal_knn_target_top_n", 200)),
            "conformal_knn_shrinkage_tau": float(getattr(self.args, "conformal_knn_shrinkage_tau", 3.0)),
            "conformal_knn_ncp_lambda": float(getattr(self.args, "conformal_knn_ncp_lambda", 1.0)),
            "conformal_knn_neighbor_mode": str(getattr(self.args, "conformal_knn_neighbor_mode", "standard") or "standard"),
            "conformal_knn_similarity_threshold": float(
                getattr(self.args, "conformal_knn_similarity_threshold", -1.0)
            ),
            "conformal_knn_min_support": int(getattr(self.args, "conformal_knn_min_support", 1)),
            "conformal_knn_adaptive_max_k": int(getattr(self.args, "conformal_knn_adaptive_max_k", 0)),
            "conformal_knn_hubness_correction": str(
                getattr(self.args, "conformal_knn_hubness_correction", "none") or "none"
            ),
            "conformal_knn_learning_mode": str(getattr(self.args, "conformal_knn_learning_mode", "fixed") or "fixed"),
            "conformal_knn_score_family_override": str(
                getattr(self.args, "conformal_knn_score_family_override", "auto") or "auto"
            ),
            "conformal_knn_repr_source": str(getattr(self.args, "conformal_knn_repr_source", "x_new") or "x_new"),
            "conformal_knn_local_calibration_scope": str(
                getattr(self.args, "conformal_knn_local_calibration_scope", "independent_knn") or "independent_knn"
            ),
        }
        metadata = {
            "source": str(getattr(self.args, "conformal_knn_config_source", "explicit_cli_or_router_defaults")),
            "parser_inherited_from_second_view": bool(
                getattr(self.args, "conformal_knn_inherited_from_second_view", False)
            ),
            "explicit_cli_flags": list(getattr(self.args, "explicit_cli_flags", []) or []),
            "inherited_fields": [],
            "frozen_g0_second_view": None,
        }
        if estimator_mode != "conformal_knn_risk_router":
            return config, metadata

        explicit_flags = set(getattr(self.args, "explicit_cli_flags", []) or [])
        second_view_config = self._frozen_g0_second_view_config()
        metadata["frozen_g0_second_view"] = second_view_config
        if second_view_config.get("active"):
            inherited_fields = []
            if "--conformal_knn_k" not in explicit_flags and second_view_config.get("knn_k") is not None:
                config["conformal_knn_k"] = int(second_view_config["knn_k"])
                inherited_fields.append("k")
            if "--conformal_knn_candidate_scope" not in explicit_flags and second_view_config.get("router_candidate_scope"):
                config["conformal_knn_candidate_scope"] = str(second_view_config["router_candidate_scope"])
                inherited_fields.append("candidate_scope")
            if "--conformal_knn_repr_source" not in explicit_flags and second_view_config.get("mode") in {
                "routed_dynamic_hyperscan_branch",
                "hyperscan_neighborloader_batch_local_branch",
                "hyperscan_knn_hypergraph_proxy_augment",
            }:
                config["conformal_knn_repr_source"] = "x_new"
                inherited_fields.append("repr_source")
            if (
                "--conformal_knn_local_calibration_scope" not in explicit_flags
                and config["conformal_knn_candidate_scope"] == "hyperscan_full"
            ):
                config["conformal_knn_local_calibration_scope"] = "same_hyperedge"
                inherited_fields.append("local_calibration_scope")
            if inherited_fields:
                metadata["source"] = "frozen_g0_hyper_knn_second_view"
                metadata["inherited_fields"] = inherited_fields

        allowed_scopes = {"labeled_full", "hyperscan_full", "labeled_relation_1hop", "undirected_relation_1hop"}
        if config["conformal_knn_candidate_scope"] not in allowed_scopes:
            raise ValueError(
                "--conformal_knn_candidate_scope must resolve to one of "
                "{labeled_full, hyperscan_full, labeled_relation_1hop, undirected_relation_1hop}."
            )
        if config["conformal_knn_repr_source"] not in {"fused_x", "node_repr", "x_low", "x_new", "x_high"}:
            raise ValueError(
                "--conformal_knn_repr_source must resolve to one of {fused_x, node_repr, x_low, x_new, x_high}."
            )
        if config["conformal_knn_learning_mode"] not in {"fixed", "ncp_local", "learned_logistic"}:
            raise ValueError("--conformal_knn_learning_mode must resolve to one of {fixed, ncp_local, learned_logistic}.")
        if config["conformal_knn_local_calibration_scope"] not in {"independent_knn", "same_hyperedge"}:
            raise ValueError(
                "--conformal_knn_local_calibration_scope must resolve to one of "
                "{independent_knn, same_hyperedge}."
            )
        if config["conformal_knn_neighbor_mode"] not in {
            "standard",
            "mutual",
            "threshold",
            "adaptive",
            "mutual_adaptive",
        }:
            raise ValueError(
                "--conformal_knn_neighbor_mode must resolve to one of "
                "{standard, mutual, threshold, adaptive, mutual_adaptive}."
            )
        if config["conformal_knn_hubness_correction"] not in {"none", "degree"}:
            raise ValueError("--conformal_knn_hubness_correction must resolve to one of {none, degree}.")
        if config["conformal_knn_min_support"] < 0:
            raise ValueError("--conformal_knn_min_support must be non-negative.")
        if config["conformal_knn_adaptive_max_k"] < 0:
            raise ValueError("--conformal_knn_adaptive_max_k must be non-negative.")
        return config, metadata

    def _conformal_knn_router_repr(self, gnn_outputs, estimator_mode, repr_source=None):
        node_repr = None if estimator_mode == "posthoc_calibrated_ranker" else gnn_outputs.get("node_repr")
        if estimator_mode != "conformal_knn_risk_router":
            return node_repr, None
        repr_source = str(repr_source or getattr(self.args, "conformal_knn_repr_source", "x_new") or "x_new").strip().lower()
        if repr_source not in {"fused_x", "node_repr", "x_low", "x_new", "x_high"}:
            raise ValueError("--conformal_knn_repr_source must be one of {fused_x, node_repr, x_low, x_new, x_high}.")

        def _to_cpu_float_view(value, name):
            if value is None:
                return None
            tensor = value.detach().cpu().float() if torch.is_tensor(value) else torch.tensor(value, dtype=torch.float32)
            if tensor.dim() != 2:
                raise MissingFrozenArtifactError(
                    f"conformal_knn {name} expects a 2-D tensor, got {tuple(tensor.shape)}."
                )
            return tensor

        fused_x_export = _to_cpu_float_view(gnn_outputs.get("fused_x"), "fused_x")
        node_repr_tensor = _to_cpu_float_view(node_repr, "node_repr")
        x_low_export = _to_cpu_float_view(gnn_outputs.get("x_low"), "x_low")
        x_new_export = _to_cpu_float_view(gnn_outputs.get("x_new"), "x_new")
        x_high_export = _to_cpu_float_view(gnn_outputs.get("x_high"), "x_high")
        metadata = {
            "requested_repr_source": repr_source,
            "available_repr_fields": {
                "fused_x": bool(fused_x_export is not None),
                "node_repr": bool(node_repr_tensor is not None),
                "x_low": bool(x_low_export is not None),
                "x_new": bool(x_new_export is not None),
                "x_high": bool(x_high_export is not None),
            },
        }
        if repr_source in {"fused_x", "node_repr"}:
            fused_tensor = fused_x_export if fused_x_export is not None else node_repr_tensor
            if fused_tensor is None:
                raise MissingFrozenArtifactError(
                    "conformal_knn_risk_router requires frozen G0 fused_x or legacy node_repr."
                )
            metadata.update(
                {
                    "effective_repr_source": "fused_x",
                    "repr_resolution": (
                        "direct_fused_x"
                        if fused_x_export is not None
                        else "legacy_node_repr_alias_for_fused_x"
                    ),
                    "repr_shape": [int(fused_tensor.shape[0]), int(fused_tensor.shape[1])],
                    "hyperscan_alignment_note": (
                        "Router consumed the explicit detector fused_x hidden."
                        if fused_x_export is not None
                        else "Frozen G0 did not export fused_x, so router fell back to legacy node_repr as the fused hidden alias."
                    ),
                }
            )
            return fused_tensor, metadata

        x_low = x_low_export
        x_low_source = "frozen_g0.x_low"
        if x_low is None:
            if node_repr_tensor is None:
                raise MissingFrozenArtifactError(
                    "conformal_knn_risk_router requires frozen G0 node_repr or exported x_low."
                )
            x_low = node_repr_tensor
            x_low_source = "frozen_g0.node_repr_legacy_proxy"
        metadata["x_low_source"] = x_low_source
        metadata["x_low_shape"] = [int(x_low.shape[0]), int(x_low.shape[1])]

        if repr_source == "x_low":
            metadata.update(
                {
                    "effective_repr_source": "x_low",
                    "repr_resolution": (
                        "exported_x_low" if x_low_source == "frozen_g0.x_low" else "legacy_node_repr_proxy_for_x_low"
                    ),
                    "repr_shape": [int(x_low.shape[0]), int(x_low.shape[1])],
                    "hyperscan_alignment_note": (
                        "Router consumed the relation-view x_low directly from frozen G0."
                        if x_low_source == "frozen_g0.x_low"
                        else "Frozen G0 did not export x_low, so router fell back to node_repr as an x_low proxy."
                    ),
                }
            )
            return x_low, metadata

        if repr_source == "x_new":
            if x_new_export is not None:
                metadata.update(
                    {
                        "effective_repr_source": "x_new",
                        "repr_resolution": "exported_x_new",
                        "x_new_source": "frozen_g0.x_new",
                        "repr_shape": [int(x_new_export.shape[0]), int(x_new_export.shape[1])],
                        "hyperscan_alignment_note": (
                            "Router consumed the exact HyperScan x_new exported by frozen G0, instead of reconstructing "
                            "it from node_repr."
                        ),
                    }
                )
                return x_new_export, metadata

            raise MissingFrozenArtifactError(
                "conformal_knn_risk_router with --conformal_knn_repr_source x_new requires frozen G0 outputs['x_new']. "
                "HyperScan-aligned KNN support must consume the forward-native cat(x_low, x_in) tensor; do not "
                "approximate it from legacy node_repr. Regenerate graph_detector_prepare with a backbone/export path "
                "that writes x_new, or run an explicit non-HyperScan control with --conformal_knn_repr_source fused_x/node_repr."
            )

        if repr_source == "x_high":
            if x_high_export is not None:
                metadata.update(
                    {
                        "effective_repr_source": "x_high",
                        "repr_resolution": "exported_x_high",
                        "x_high_source": "frozen_g0.x_high",
                        "repr_shape": [int(x_high_export.shape[0]), int(x_high_export.shape[1])],
                        "hyperscan_alignment_note": (
                            "Router consumed the exported HyperScan HGNN high-order branch x_high before detector fusion."
                        ),
                    }
                )
                return x_high_export, metadata

            raise MissingFrozenArtifactError(
                "conformal_knn_risk_router with --conformal_knn_repr_source x_high requires frozen G0 outputs['x_high']. "
                "Regenerate graph_detector_prepare with a HyperScan-style backbone/export path that writes x_high."
            )

        raise ValueError(f"Unsupported conformal KNN representation source after normalization: {repr_source!r}.")

    def _build_graph_conformal_estimator_bundle(self, estimator_mode):
        context = self.ensure_backbone_context()
        gnn_outputs = context["gnn_outputs"]
        budgets = self._router_budgets()
        estimator = build_estimator(estimator_mode)
        labels = gnn_outputs.get("labels")
        if labels is None:
            labels = torch.tensor(self.labels, dtype=torch.long)
        tune_idx, cal_idx, split_metadata = self._split_valid_tune_cal(labels)
        scalar_only = estimator_mode == "posthoc_calibrated_ranker"
        edge_index = None if scalar_only else self.data["edge_index"]
        edge_type = None if scalar_only else self.data["edge_type"]
        conformal_knn_config, conformal_knn_config_metadata = self._effective_conformal_knn_config(estimator_mode)
        node_repr, conformal_knn_repr_metadata = self._conformal_knn_router_repr(
            gnn_outputs,
            estimator_mode,
            repr_source=conformal_knn_config.get("conformal_knn_repr_source"),
        )
        estimator.fit(
            logits=gnn_outputs.get("logits"),
            probs=gnn_outputs.get("prob"),
            labels=labels,
            train_idx=self.data["train_idx"],
            val_idx=cal_idx if estimator_mode == "msp_ts" else self.data["valid_idx"],
            tune_idx=tune_idx,
            cal_idx=cal_idx,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            budgets=budgets,
            tune_cal_split_metadata=split_metadata,
            **conformal_knn_config,
        )
        risk_manifest = estimator.build_manifest(
            logits=gnn_outputs.get("logits"),
            probs=gnn_outputs.get("prob"),
            labels=labels,
            train_idx=self.data["train_idx"],
            val_idx=self.data["valid_idx"],
            test_idx=self.data["test_idx"],
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            budgets=budgets,
            **conformal_knn_config,
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
            "tune_cal_split_metadata": {
                **dict(risk_manifest.get("calibration_metadata", {}).get("tune_cal_split_metadata", {})),
                **split_metadata,
            },
            "conformal_knn_config": {
                "k": int(conformal_knn_config["conformal_knn_k"]),
                "candidate_scope": str(conformal_knn_config["conformal_knn_candidate_scope"]),
                "target_top_n": int(conformal_knn_config["conformal_knn_target_top_n"]),
                "anchor_top_n": int(conformal_knn_config["conformal_knn_target_top_n"]),
                "shrinkage_tau": float(conformal_knn_config["conformal_knn_shrinkage_tau"]),
                "ncp_lambda": float(conformal_knn_config["conformal_knn_ncp_lambda"]),
                "neighbor_mode": str(conformal_knn_config["conformal_knn_neighbor_mode"]),
                "similarity_threshold": float(conformal_knn_config["conformal_knn_similarity_threshold"]),
                "min_support": int(conformal_knn_config["conformal_knn_min_support"]),
                "adaptive_max_k": int(conformal_knn_config["conformal_knn_adaptive_max_k"]),
                "hubness_correction": str(conformal_knn_config["conformal_knn_hubness_correction"]),
                "learning_mode": str(conformal_knn_config["conformal_knn_learning_mode"]),
                "local_calibration_scope": str(conformal_knn_config["conformal_knn_local_calibration_scope"]),
                "score_family_override": str(conformal_knn_config["conformal_knn_score_family_override"]),
                "repr_source": str(conformal_knn_config["conformal_knn_repr_source"]),
                "repr_metadata": dict(conformal_knn_repr_metadata or {}),
                "config_source": dict(conformal_knn_config_metadata),
                "hyper_knn_alignment": {
                    "router_reuses_hyper_knn_config_by_default": bool(
                        conformal_knn_config_metadata.get("inherited_fields")
                    ),
                    "explicit_router_flags_override_inheritance": True,
                    "candidate_alignment_note": str(
                        (conformal_knn_config_metadata.get("frozen_g0_second_view") or {}).get(
                            "candidate_alignment_note",
                            "",
                        )
                        or ""
                    ),
                    "k_semantics": (
                        "hyperscan_full uses k as hyperedge size including center; "
                        "relation-local scopes use k as the maximum non-center support-neighbor count"
                    ),
                    "local_calibration_scope_note": (
                        "same_hyperedge reuses the target-centered HyperScan KNN hyperedge for local calibration; "
                        "independent_knn queries calibration neighbors separately in the same representation space"
                    ),
                },
                "target_selection_contract": "target_node_to_knn_support_group_risk_v1",
                "support_group_role": "evidence_only_not_routed_outputs",
            }
            if estimator_mode == "conformal_knn_risk_router"
            else None,
        }
        return self._bundle_from_risk_manifest(estimator_mode, risk_manifest)

    def _run_graph_conformal_estimator_matrix(self, stage_dir, base_bundle):
        requested_mode = str(getattr(self.args, "estimator_mode", "none")).lower()
        risk_bundle = self._build_graph_conformal_estimator_bundle(requested_mode)
        context = self.ensure_backbone_context()
        base_pred = _labeled_prefix_array(context["gnn_outputs"]["pred"].detach().cpu().numpy(), len(self.labels))
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
                    "reads": [
                        _preparation_artifact_reference("graph_detector", "manifest.json"),
                        _preparation_artifact_reference("graph_detector", "outputs.pt"),
                    ],
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
        risk_bundle, login_artifacts, login_budget_rows = self._build_login_uncertainty_router_bundle(stage_dir)
        context = self.ensure_backbone_context()
        base_pred = context["gnn_outputs"]["pred"].detach().cpu().numpy()
        official_pl_mask = np.asarray(
            risk_bundle["risk_manifest"].get("official_pl_mask", np.zeros(len(self.labels), dtype=bool)),
            dtype=bool,
        )
        triggered_mask = official_pl_mask & self.test_mask
        budgets = self._router_budgets()
        metrics_payload, budget_curve = self._build_metrics_payload(
            risk_bundle,
            base_pred,
            base_pred,
            self.test_mask,
            triggered_mask,
            extra_metrics={
                "best_p_proxy": "login_uncertainty_router",
                "selection_contract": "official_global_topk_by_pl_rate",
                "official_pl_rate": 0.1,
                "official_selected_count_global": int(official_pl_mask.sum()),
                "official_selected_count_test": int(triggered_mask.sum()),
                "auxiliary_budget_analysis": True,
                "official_selection": login_artifacts["login_router_metrics"]["official_selection"],
            },
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
            "paper_aligned": True,
            "repo_locally_verified": False,
            "verified_scope": "node_selection_uncertainty_only",
            "official_repo_path": r"G:\Research\BotDetection\LOGIN_init",
            "official_submodule_scope": "node_selection_uncertainty_only",
            "implementation_status": "official_uncertainty_module_reproduction_only",
            "allowed_use": "official_node_selection_uncertainty_module_plus_local_auxiliary_budget_analysis",
            "forbidden_claims": [
                "full_LOGIN_reproduction",
                "full_LOGIN_loop",
                "LLM_feedback_or_graph_refinement_effect",
                "best_index_prompt_llm_feature_update_edge_pruning_reproduction",
            ],
        }
        paper_bundle["report"]["claim_status"] = "official_uncertainty_module_only_not_full_login_reproduction"
        paper_bundle["report"]["claim_boundary"] = claim_boundary
        paper_bundle["acceptance_gate"]["claim_status"] = "official_uncertainty_module_only_not_full_login_reproduction"
        paper_bundle["acceptance_gate"]["claim_boundary"] = claim_boundary
        self._write_stage2_paper_artifacts(stage_dir, paper_bundle)
        metrics_payload["stage2_acceptance_gate"] = paper_bundle["acceptance_gate"]
        metrics_payload["stage2_claim_status"] = paper_bundle["report"]["claim_status"]
        notes = [
            "claim tested: official LOGIN_init uncertainty node-selection submodule only",
            "official formula: variance over five independently trained GNN logits, summed over class dimension",
            "scope: selection only; prompt_LLM, feature update, edge pruning, and full LOGIN loop remain out of scope",
            "budget_curve.csv and validation threshold are auxiliary local analysis outputs, not the official LOGIN selection contract",
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
                    "reads": [
                        _preparation_artifact_reference("graph_detector", "manifest.json"),
                        _preparation_artifact_reference("graph_detector", "outputs.pt"),
                    ],
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
        pred_labeled = _labeled_prefix_array(pred, len(self.labels))
        wrong = (pred_labeled != self.labels).astype(np.int32)
        val_idx = _idx_numpy(self.data["valid_idx"])
        test_idx = _idx_numpy(self.data["test_idx"])
        if val_idx.size:
            budget = max(int(val_idx.size * 0.15), 1)
            validation_threshold = float(np.sort(risk_score[val_idx])[-budget])
        else:
            validation_threshold = float("inf")
        confidence = gnn_outputs["prob"].detach().cpu().max(dim=1)[0].numpy()
        confidence_labeled = _labeled_prefix_array(confidence, len(self.labels))
        hcw_mask = ((1.0 - confidence_labeled) < 0.1) & (wrong == 1)
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
        labeled_count = int(len(self.labels))
        risk_manifest_local = dict(risk_manifest)
        if "risk_score" in risk_manifest_local:
            risk_manifest_local["risk_score"] = _labeled_prefix_array(risk_manifest_local["risk_score"], labeled_count).astype(np.float32)
        if "hcw_mask" in risk_manifest_local:
            risk_manifest_local["hcw_mask"] = _labeled_prefix_array(risk_manifest_local["hcw_mask"], labeled_count).astype(bool).tolist()
        edge_index_stage2, edge_type_stage2, num_nodes_stage2 = self._stage2_structural_graph()
        probe_features = build_probe_features(
            logits=gnn_outputs["logits"][:labeled_count],
            probs=gnn_outputs["prob"][:labeled_count],
            edge_index=edge_index_stage2,
            edge_type=edge_type_stage2,
            node_repr=gnn_outputs["node_repr"][:labeled_count],
        )
        if structural_manifest is None or subgroup_op is None or subgroup_analysis is None:
            structural_manifest, subgroup_op, subgroup_analysis = build_subgroup_manifests(
                edge_index=edge_index_stage2,
                edge_type=edge_type_stage2,
                num_nodes=num_nodes_stage2,
                risk_manifest=risk_manifest_local,
                labels=self.labels,
                predictions=gnn_outputs["pred"][:labeled_count].numpy(),
                probe_features=probe_features,
                degree_quantile=self.args.sparse_degree_quantile,
                propagation_quantile=self.args.propagation_quantile,
                train_idx=self.data["train_idx"],
            )
        return {
            "metadata": mode_metadata(mode),
            "risk_manifest": risk_manifest_local,
            "structural_manifest": structural_manifest,
            "subgroup_manifest_operational": subgroup_op,
            "subgroup_manifest_analysis": subgroup_analysis,
            "probe_features": probe_features,
        }

    def _stage2_probabilities(self, final_outputs):
        probs = final_outputs.get("prob")
        if probs is None:
            probs = torch.softmax(final_outputs["logits"].float(), dim=1)
        probs = probs.detach().cpu().float()
        return probs[: int(len(self.labels))]

    def _stage2_predictions(self, final_outputs):
        pred = final_outputs.get("pred")
        if pred is None:
            pred = self._stage2_probabilities(final_outputs).argmax(dim=1)
        return pred.detach().cpu().long().numpy()[: int(len(self.labels))]

    def _stage2_logits(self, final_outputs):
        logits = final_outputs.get("logits")
        logits_t = logits.detach().cpu().float() if logits is not None else torch.log(self._stage2_probabilities(final_outputs).clamp_min(1e-8))
        return logits_t[: int(len(self.labels))]

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
        edge_index, _edge_type, num_nodes = self._stage2_structural_graph()
        edge_index = _idx_numpy(edge_index)
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
        return scores

    def _stage2_neighbor_label_masks(self, eval_mask):
        edge_index, _edge_type, _num_nodes = self._stage2_structural_graph()
        edge_index = _idx_numpy(edge_index)
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
        edge_index, _edge_type, num_nodes = self._stage2_structural_graph()
        edge_index = _idx_numpy(edge_index)
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
        manifest = {
            "contract": "stage_outputs_v1",
            "status": "completed",
            "stage": self.execution_stage,
            "best_mode": metrics_payload.get("best_p_proxy"),
            "risk_summary": {
                "validation_metrics": risk_bundle.get("risk_manifest", {}).get("validation_metrics", {}),
                "test_metrics": risk_bundle.get("risk_manifest", {}).get("test_metrics", {}),
            },
        }
        self._write_stage_manifest(stage_dir, manifest)
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

    def _run_glance_oracle_refine(self, stage_dir, base_bundle):
        context = self.ensure_backbone_context()
        semantic = self._load_semantic_finetune_artifact()
        active_edge_index, active_edge_type = self._active_graph_tensors()
        gnn_outputs = context["gnn_outputs"]
        p_gnn = gnn_outputs.get("prob")
        logits_gnn = gnn_outputs.get("logits")
        pred_gnn = gnn_outputs.get("pred")
        z_gnn = gnn_outputs.get("node_repr")
        if any(item is None for item in (p_gnn, logits_gnn, pred_gnn, z_gnn)):
            raise MissingFrozenArtifactError(
                "glance_oracle_refine requires frozen_g0 outputs with prob/logits/pred/node_repr."
            )

        p_gnn = p_gnn.detach().cpu().float()
        logits_gnn = logits_gnn.detach().cpu().float()
        pred_gnn = pred_gnn.detach().cpu().long()
        z_gnn = z_gnn.detach().cpu().float()
        labels_t = torch.tensor(self.labels, dtype=torch.long)

        routed_masks = self._oracle_wrong_node_masks(pred_gnn.numpy())
        semantic_views = self._build_k_hop_semantic_views(
            semantic["embeddings"],
            active_edge_index,
        )
        refiner_features = torch.cat(
            [z_gnn, semantic_views["ego"], semantic_views["hop1"], semantic_views["hop2"]],
            dim=1,
        )

        train_idx = np.flatnonzero(routed_masks["train"])
        valid_idx = np.flatnonzero(routed_masks["valid"])
        test_idx = np.flatnonzero(routed_masks["test"])
        if train_idx.size == 0:
            raise MissingFrozenArtifactError(
                "glance_oracle_refine found zero oracle-routed train nodes. "
                "This upper-bound stage requires at least one train error node."
            )

        model = GlanceRefinerMLP(
            input_dim=int(refiner_features.shape[1]),
            hidden_dim=128,
            activation=str(getattr(self.args, "activation", "leakyrelu")).lower(),
            dropout=0.1,
        )
        model, fit_summary = self._train_glance_refiner(
            model,
            refiner_features[train_idx],
            labels_t[train_idx],
            refiner_features[valid_idx],
            labels_t[valid_idx],
        )

        model.eval()
        with torch.no_grad():
            if train_idx.size > 0:
                train_logits_ref = model(refiner_features[train_idx]).cpu()
            else:
                train_logits_ref = torch.empty((0, 2), dtype=torch.float32)
            if valid_idx.size > 0:
                valid_logits_ref = model(refiner_features[valid_idx]).cpu()
            else:
                valid_logits_ref = torch.empty((0, 2), dtype=torch.float32)
            if test_idx.size > 0:
                test_logits_ref = model(refiner_features[test_idx]).cpu()
            else:
                test_logits_ref = torch.empty((0, 2), dtype=torch.float32)

        final_logits = logits_gnn.clone()
        final_prob = p_gnn.clone()
        final_pred = pred_gnn.clone()
        if train_idx.size > 0:
            final_logits[train_idx] = train_logits_ref
            final_prob[train_idx] = torch.softmax(train_logits_ref, dim=1)
            final_pred[train_idx] = train_logits_ref.argmax(dim=1)
        if valid_idx.size > 0:
            final_logits[valid_idx] = valid_logits_ref
            final_prob[valid_idx] = torch.softmax(valid_logits_ref, dim=1)
            final_pred[valid_idx] = valid_logits_ref.argmax(dim=1)
        if test_idx.size > 0:
            final_logits[test_idx] = test_logits_ref
            final_prob[test_idx] = torch.softmax(test_logits_ref, dim=1)
            final_pred[test_idx] = test_logits_ref.argmax(dim=1)

        test_delta = _delta_table(pred_gnn.numpy(), final_pred.numpy(), self.labels, self.test_mask)
        routed_test_ratio = float(test_idx.size / max(int(self.test_mask.sum()), 1))
        non_routed_test_mask = self.test_mask & ~routed_masks["test"]
        preservation = float(
            (final_pred.numpy()[non_routed_test_mask] == pred_gnn.numpy()[non_routed_test_mask]).mean()
        ) if non_routed_test_mask.any() else 1.0

        overall_test = _score_all(self.labels[self.test_mask], final_pred.numpy()[self.test_mask])
        routed_test = _score_all(self.labels[test_idx], final_pred.numpy()[test_idx]) if test_idx.size else {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "bot_f1": 0.0,
            "count": 0,
        }
        base_test = _score_all(self.labels[self.test_mask], pred_gnn.numpy()[self.test_mask])
        base_wrong_count = int(test_idx.size)
        base_correct_count = int(int(self.test_mask.sum()) - base_wrong_count)

        metrics_payload = {
            "contract": "glance_oracle_refine_metrics_v1",
            "routing_mode": "oracle_wrong_nodes",
            "upper_bound_only": True,
            "not_deployable": True,
            "overall_test": overall_test,
            "routed_test": routed_test,
            "base_test": base_test,
            "non_routed_test_preservation_accuracy": preservation,
            "routed_test_ratio": routed_test_ratio,
            "fixed_wrong_to_right": int(test_delta["fix"]),
            "fixed_right_to_wrong": int(test_delta["broke"]),
            "net_gain_on_routed": int(test_delta["net"]),
            "base_wrong_count": base_wrong_count,
            "base_correct_count": base_correct_count,
            "wrong_node_fix_rate": float(test_delta["fix"] / base_wrong_count) if base_wrong_count else 0.0,
            "correct_node_break_rate": 0.0,
            "overall_delta_vs_base_gnn": {
                "accuracy": float(overall_test["accuracy"] - base_test["accuracy"]),
                "macro_f1": float(overall_test["macro_f1"] - base_test["macro_f1"]),
                "bot_f1": float(overall_test["bot_f1"] - base_test["bot_f1"]),
            },
            "oracle_routed_counts": {
                "train": int(train_idx.size),
                "valid": int(valid_idx.size),
                "test": int(test_idx.size),
                "all": int(routed_masks["all"].sum()),
            },
            "fit_summary": fit_summary,
        }

        manifest = {
            "contract": "glance_oracle_refine_v1",
            "status": "completed",
            "routing_mode": "oracle_wrong_nodes",
            "upper_bound_only": True,
            "not_deployable": True,
            "research_positioning": "error_only_oracle_upper_bound_for_correction_refiner",
            "no_router_training": True,
            "no_llm_generation": True,
            "semantic_source": "semantic_finetune_roberta_finetuned",
            "semantic_stage_dir": str(semantic["dir"]),
            "semantic_manifest": semantic["manifest"],
            "frozen_g0_dir": str(context["frozen_g0"]["dir"]),
            "frozen_g0_manifest": context["frozen_g0"]["manifest"],
            "graph_source": context["graph_bundle"]["source"],
            "graph_source_root": context["graph_bundle"]["source_root"],
            "text_encoder_identity": str(semantic["manifest"].get("lm_model", semantic["manifest"].get("semantic_backbone", "roberta_finetuned"))),
            "refiner_architecture": {
                "type": "oracle_glance_refiner_mlp",
                "input": "[z_gnn || z_sem_0 || z_sem_1 || z_sem_2]",
                "hidden_dim": 128,
                "activation": str(getattr(self.args, "activation", "leakyrelu")).lower(),
                "dropout": 0.1,
                "output_dim": 2,
                "hard_switch": True,
            },
            "oracle_routed_counts": metrics_payload["oracle_routed_counts"],
        }
        self._write_stage_manifest(
            stage_dir,
            manifest,
            artifact_namespace="stages/local_conformal_diagnostic",
            visibility="public",
            resolved_task="local_conformal_diagnostic",
        )
        write_json(stage_dir / "metrics.json", metrics_payload)

        per_node_rows = []
        for node_idx in np.flatnonzero(self.test_mask):
            per_node_rows.append({
                "node_id": int(node_idx),
                "base_pred": int(pred_gnn[node_idx].item()),
                "final_pred": int(final_pred[node_idx].item()),
                "label": int(self.labels[node_idx]),
                "routed": bool(routed_masks["test"][node_idx]),
                "was_wrong_base": bool(pred_gnn[node_idx].item() != self.labels[node_idx]),
                "is_wrong_final": bool(final_pred[node_idx].item() != self.labels[node_idx]),
                "neighbor_count_1hop": int(semantic_views["count_1hop"][node_idx]),
                "neighbor_count_2hop": int(semantic_views["count_2hop"][node_idx]),
            })
        write_text(
            stage_dir / "per_node_test.jsonl",
            "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in per_node_rows) + "\n",
        )
        analysis_summary = self._build_refiner_analysis_summary(per_node_rows, "oracle_wrong_nodes")
        write_json(stage_dir / "analysis_summary.json", analysis_summary)

        save_stage_artifacts(
            stage_dir,
            {
                "outputs.pt": {
                    "logits": final_logits,
                    "prob": final_prob,
                    "pred": final_pred,
                    "labels": labels_t,
                    "base_pred": pred_gnn,
                    "routed_masks": {key: torch.tensor(value, dtype=torch.bool) for key, value in routed_masks.items()},
                    "neighbor_count_1hop": torch.tensor(semantic_views["count_1hop"], dtype=torch.long),
                    "neighbor_count_2hop": torch.tensor(semantic_views["count_2hop"], dtype=torch.long),
                },
                "checkpoint.pt": {
                    "model": model.state_dict(),
                    "fit_summary": fit_summary,
                },
                "oracle_summary": {
                    "routing_mode": "oracle_wrong_nodes",
                    "warning": "upper bound only; routed masks use ground-truth errors in every split",
                },
                "analysis_summary": analysis_summary,
                **base_bundle,
            },
        )

        notes = [
            "# GLANCE oracle upper bound",
            "",
            "This stage is an oracle upper-bound ablation.",
            "Routed nodes are defined by ground-truth base-GNN errors within each split.",
            "No router is trained and no generative LLM is queried.",
            "Routed-node refinement uses semantic_finetune roberta-f embeddings with ego/1-hop/2-hop mean pooling.",
        ]
        write_text(stage_dir / "notes.md", "\n".join(notes) + "\n")
        return {
            "stage": "glance_oracle_refine",
            "stage_dir": str(stage_dir),
            "metrics": metrics_payload,
        }

    def _run_glance_full_graph_refine(self, stage_dir, base_bundle):
        context = self.ensure_backbone_context()
        semantic = self._load_semantic_finetune_artifact()
        active_edge_index, active_edge_type = self._active_graph_tensors()
        gnn_outputs = context["gnn_outputs"]
        p_gnn = gnn_outputs.get("prob")
        logits_gnn = gnn_outputs.get("logits")
        pred_gnn = gnn_outputs.get("pred")
        z_gnn = gnn_outputs.get("node_repr")
        if any(item is None for item in (p_gnn, logits_gnn, pred_gnn, z_gnn)):
            raise MissingFrozenArtifactError(
                "glance_full_graph_refine requires frozen_g0 outputs with prob/logits/pred/node_repr."
            )

        p_gnn = p_gnn.detach().cpu().float()
        logits_gnn = logits_gnn.detach().cpu().float()
        pred_gnn = pred_gnn.detach().cpu().long()
        z_gnn = z_gnn.detach().cpu().float()
        labels_t = torch.tensor(self.labels, dtype=torch.long)

        semantic_views = self._build_k_hop_semantic_views(
            semantic["embeddings"],
            active_edge_index,
        )
        refiner_features = torch.cat(
            [z_gnn, semantic_views["ego"], semantic_views["hop1"], semantic_views["hop2"]],
            dim=1,
        )

        train_idx = _idx_numpy(self.data["train_idx"])
        valid_idx = _idx_numpy(self.data["valid_idx"])
        test_idx = _idx_numpy(self.data["test_idx"])
        if train_idx.size == 0:
            raise MissingFrozenArtifactError("glance_full_graph_refine requires a non-empty train_idx.")

        model = GlanceRefinerMLP(
            input_dim=int(refiner_features.shape[1]),
            hidden_dim=128,
            activation=str(getattr(self.args, "activation", "leakyrelu")).lower(),
            dropout=0.1,
        )
        model, fit_summary = self._train_glance_refiner(
            model,
            refiner_features[train_idx],
            labels_t[train_idx],
            refiner_features[valid_idx],
            labels_t[valid_idx],
        )

        model.eval()
        with torch.no_grad():
            final_logits = model(refiner_features).cpu()
            final_prob = torch.softmax(final_logits, dim=1)
            final_pred = final_logits.argmax(dim=1)

        test_delta = _delta_table(pred_gnn.numpy(), final_pred.numpy(), self.labels, self.test_mask)
        overall_test = _score_all(self.labels[self.test_mask], final_pred.numpy()[self.test_mask])
        base_test = _score_all(self.labels[self.test_mask], pred_gnn.numpy()[self.test_mask])
        lm_only_test = None
        overall_delta_vs_lm_only = None
        lm_only_pred = None
        if semantic.get("outputs") is not None:
            lm_outputs = semantic["outputs"]
            lm_prob = lm_outputs.get("prob_cal", lm_outputs.get("prob"))
            lm_pred_t = lm_outputs.get("pred")
            if lm_pred_t is None and lm_prob is not None:
                lm_pred_t = lm_prob.argmax(dim=1)
            if lm_pred_t is not None:
                lm_only_pred = lm_pred_t.detach().cpu().long() if torch.is_tensor(lm_pred_t) else torch.tensor(lm_pred_t, dtype=torch.long)
                lm_only_test = _score_all(self.labels[self.test_mask], lm_only_pred.numpy()[self.test_mask])
                overall_delta_vs_lm_only = {
                    "accuracy": float(overall_test["accuracy"] - lm_only_test["accuracy"]),
                    "macro_f1": float(overall_test["macro_f1"] - lm_only_test["macro_f1"]),
                    "bot_f1": float(overall_test["bot_f1"] - lm_only_test["bot_f1"]),
                }

        base_wrong_count = int((pred_gnn.numpy()[self.test_mask] != self.labels[self.test_mask]).sum())
        base_correct_count = int(int(self.test_mask.sum()) - base_wrong_count)

        metrics_payload = {
            "contract": "glance_full_graph_refine_metrics_v1",
            "routing_mode": "full_graph_supervised_refiner",
            "upper_bound_only": False,
            "not_deployable": False,
            "overall_test": overall_test,
            "base_test": base_test,
            "lm_only_test": lm_only_test,
            "overall_delta_vs_base_gnn": {
                "accuracy": float(overall_test["accuracy"] - base_test["accuracy"]),
                "macro_f1": float(overall_test["macro_f1"] - base_test["macro_f1"]),
                "bot_f1": float(overall_test["bot_f1"] - base_test["bot_f1"]),
            },
            "overall_delta_vs_lm_only": overall_delta_vs_lm_only,
            "improved_count": int(test_delta["fix"]),
            "degraded_count": int(test_delta["broke"]),
            "net_gain": int(test_delta["net"]),
            "count": int(test_delta["touched"]),
            "base_wrong_count": base_wrong_count,
            "base_correct_count": base_correct_count,
            "wrong_node_fix_rate": float(test_delta["fix"] / base_wrong_count) if base_wrong_count else 0.0,
            "correct_node_break_rate": float(test_delta["broke"] / base_correct_count) if base_correct_count else 0.0,
            "fit_summary": fit_summary,
            "split_scope": {
                "train": int(train_idx.size),
                "valid": int(valid_idx.size),
                "test": int(test_idx.size),
            },
        }

        manifest = {
            "contract": "glance_full_graph_refine_v1",
            "status": "completed",
            "routing_mode": "full_graph_supervised_refiner",
            "upper_bound_only": False,
            "not_deployable": False,
            "research_positioning": "no_router_lower_bound_router_ablation_for_refiner_diagnosis",
            "no_router_training": True,
            "no_llm_generation": True,
            "semantic_source": "semantic_finetune_roberta_finetuned",
            "semantic_stage_dir": str(semantic["dir"]),
            "semantic_manifest": semantic["manifest"],
            "frozen_g0_dir": str(context["frozen_g0"]["dir"]),
            "frozen_g0_manifest": context["frozen_g0"]["manifest"],
            "graph_source": context["graph_bundle"]["source"],
            "graph_source_root": context["graph_bundle"]["source_root"],
            "text_encoder_identity": str(semantic["manifest"].get("lm_model", semantic["manifest"].get("semantic_backbone", "roberta_finetuned"))),
            "refiner_architecture": {
                "type": "glance_full_graph_refiner_mlp",
                "input": "[z_gnn || z_sem_0 || z_sem_1 || z_sem_2]",
                "hidden_dim": 128,
                "activation": str(getattr(self.args, "activation", "leakyrelu")).lower(),
                "dropout": 0.1,
                "output_dim": 2,
                "full_graph": True,
            },
            "split_scope": metrics_payload["split_scope"],
            "lm_only_reference_available": bool(lm_only_test is not None),
            "seed_scope_note": "current local validation path is seed_1; multi-seed requires matching frozen_g0 per seed",
        }
        self._write_stage_manifest(
            stage_dir,
            manifest,
            artifact_namespace="stages/local_conflict_prune_diag",
            visibility="public",
            resolved_task="local_conflict_prune_diag",
        )
        write_json(stage_dir / "metrics.json", metrics_payload)

        per_node_rows = []
        for node_idx in np.flatnonzero(self.test_mask):
            row = {
                "node_id": int(node_idx),
                "base_pred": int(pred_gnn[node_idx].item()),
                "final_pred": int(final_pred[node_idx].item()),
                "label": int(self.labels[node_idx]),
                "routed": True,
                "neighbor_count_1hop": int(semantic_views["count_1hop"][node_idx]),
                "neighbor_count_2hop": int(semantic_views["count_2hop"][node_idx]),
                "was_wrong_base": bool(pred_gnn[node_idx].item() != self.labels[node_idx]),
                "is_wrong_final": bool(final_pred[node_idx].item() != self.labels[node_idx]),
            }
            if lm_only_pred is not None:
                row["lm_only_pred"] = int(lm_only_pred[node_idx].item())
            per_node_rows.append(row)
        write_text(
            stage_dir / "per_node_test.jsonl",
            "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in per_node_rows) + "\n",
        )
        analysis_summary = self._build_refiner_analysis_summary(per_node_rows, "full_graph_supervised_refiner")
        write_json(stage_dir / "analysis_summary.json", analysis_summary)

        save_stage_artifacts(
            stage_dir,
            {
                "outputs.pt": {
                    "logits": final_logits,
                    "prob": final_prob,
                    "pred": final_pred,
                    "labels": labels_t,
                    "base_pred": pred_gnn,
                    "neighbor_count_1hop": torch.tensor(semantic_views["count_1hop"], dtype=torch.long),
                    "neighbor_count_2hop": torch.tensor(semantic_views["count_2hop"], dtype=torch.long),
                },
                "checkpoint.pt": {
                    "model": model.state_dict(),
                    "fit_summary": fit_summary,
                },
                "refiner_summary": {
                    "scope": "full_graph_supervised_refiner",
                    "warning": "seed_1 local validation path uses shared LM embedding and per-seed frozen_g0 backbone",
                },
                "analysis_summary": analysis_summary,
                **base_bundle,
            },
        )

        notes = [
            "# GLANCE full-graph refiner",
            "",
            "This stage trains a full-graph supervised refiner on train_idx and selects the best epoch on valid_idx.",
            "The refiner input is [z_gnn || z_sem_0 || z_sem_1 || z_sem_2] with semantic embeddings from semantic_finetune or explicit --emb_path.",
            "No router is trained and no generative LLM is queried.",
            "Research positioning: no-router lower-bound / router ablation for diagnosing where the refiner helps or harms.",
            "Current local validation path is seed_1 only; multi-seed requires matching frozen_g0 artifacts per seed.",
        ]
        write_text(stage_dir / "notes.md", "\n".join(notes) + "\n")
        return {
            "stage": "glance_full_graph_refine",
            "stage_dir": str(stage_dir),
            "metrics": metrics_payload,
        }

    def _run_local_conformal_prune_diag(self, stage_dir, base_bundle):
        context = self.ensure_backbone_context()
        gnn_outputs = context["gnn_outputs"]
        logits_gnn = gnn_outputs.get("logits")
        prob_gnn = gnn_outputs.get("prob")
        pred_gnn = gnn_outputs.get("pred")
        node_repr = gnn_outputs.get("node_repr")
        if any(item is None for item in (logits_gnn, prob_gnn, pred_gnn, node_repr)):
            raise MissingFrozenArtifactError(
                "local_conformal_prune_diag requires frozen_g0 outputs with logits/prob/pred/node_repr."
            )

        edge_index = self.data.get("edge_index")
        edge_type = self.data.get("edge_type")
        if edge_index is None or edge_type is None:
            raise MissingFrozenArtifactError("local_conformal_prune_diag requires graph edge_index and edge_type.")

        logits_gnn = logits_gnn.detach().cpu().float()
        prob_gnn = prob_gnn.detach().cpu().float()
        pred_gnn = pred_gnn.detach().cpu().long()
        node_repr = node_repr.detach().cpu().float()
        labels_t = torch.tensor(self.labels, dtype=torch.long)
        labels_np = labels_t.numpy()
        semantic_bundle = resolve_g0_feature_bundle(self.args, self.data)
        semantic_raw = semantic_bundle["raw_features"].detach().cpu().float()
        semantic_norm = F.normalize(semantic_raw, p=2, dim=1, eps=1e-12)
        similarity_gate_threshold = getattr(self.args, "local_conf_similarity_gate_threshold", None)
        if similarity_gate_threshold is not None:
            similarity_gate_threshold = float(similarity_gate_threshold)
        degree_guard_enabled = not bool(getattr(self.args, "local_conf_disable_degree_guard", False))

        router_bundle = self._fit_local_conformal_router(
            logits_gnn=logits_gnn,
            prob_gnn=prob_gnn,
            labels_t=labels_t,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
        )
        estimator = router_bundle["estimator"]
        risk_manifest = router_bundle["risk_manifest"]
        q_score = router_bundle["q_score"]
        routed_mask = router_bundle["routed_mask"]
        routed_budget = float(router_bundle["routed_budget"])
        threshold = float(router_bundle["threshold"])

        edge_index_cpu = edge_index.detach().cpu().long()
        edge_type_cpu = edge_type.detach().cpu().long()
        src_np = edge_index_cpu[0].numpy().astype(np.int64)
        dst_np = edge_index_cpu[1].numpy().astype(np.int64)
        rel_np = edge_type_cpu.numpy().astype(np.int64)
        num_nodes = int(labels_t.numel())
        num_edges = int(edge_index_cpu.size(1))
        base_in_degree, base_out_degree = self._degree_arrays(edge_index_cpu, num_nodes)

        routed_node_ids = np.flatnonzero(routed_mask)
        routed_node_ids = np.asarray(
            sorted(routed_node_ids.tolist(), key=lambda node_id: (-float(q_score[int(node_id)]), int(node_id))),
            dtype=np.int64,
        )
        incident_edges = [[] for _ in range(num_nodes)]
        for edge_pos, (u, v) in enumerate(zip(src_np.tolist(), dst_np.tolist())):
            incident_edges[int(u)].append(int(edge_pos))
            if int(v) != int(u):
                incident_edges[int(v)].append(int(edge_pos))

        edge_q_after_cache = {}
        bucket_rows = {}
        candidate_rows = []
        for target_id in routed_node_ids.tolist():
            for relation_id in (0, 1):
                rows = []
                for edge_pos in incident_edges[int(target_id)]:
                    if int(rel_np[edge_pos]) != int(relation_id):
                        continue
                    if edge_pos not in edge_q_after_cache:
                        cf_edge_index, cf_edge_type = self._remove_edge_at_position(edge_index_cpu, edge_type_cpu, edge_pos)
                        q_after_all = estimator.score(
                            logits_gnn,
                            prob_gnn,
                            edge_index=cf_edge_index,
                            edge_type=cf_edge_type,
                            node_repr=node_repr,
                            direction_mode="incoming",
                            relation_mode="agnostic",
                        )
                        edge_q_after_cache[int(edge_pos)] = {
                            int(src_np[edge_pos]): float(q_after_all[int(src_np[edge_pos])]),
                            int(dst_np[edge_pos]): float(q_after_all[int(dst_np[edge_pos])]),
                        }
                    q_after = float(edge_q_after_cache[int(edge_pos)][int(target_id)])
                    cosine_uv = float((semantic_norm[int(src_np[edge_pos])] * semantic_norm[int(dst_np[edge_pos])]).sum().item())
                    row = {
                        "target_id": int(target_id),
                        "edge": [int(src_np[edge_pos]), int(dst_np[edge_pos])],
                        "relation": int(relation_id),
                        "edge_pos": int(edge_pos),
                        "q_before": float(q_score[int(target_id)]),
                        "q_after": q_after,
                        "delta_q": float(q_after - q_score[int(target_id)]),
                        "cosine": cosine_uv,
                        "similarity_gate_pass": True if similarity_gate_threshold is None else bool(cosine_uv <= similarity_gate_threshold),
                        "selected": False,
                        "selection_reason": None,
                        "guard_blocked": False,
                    }
                    rows.append(row)
                    candidate_rows.append(row)
                bucket_rows[(int(target_id), int(relation_id))] = rows

        conf_removed_positions = set()
        conf_bucket_budget = {}
        conf_in_degree = base_in_degree.copy()
        conf_out_degree = base_out_degree.copy()
        duplicate_attempts = 0
        for target_id in routed_node_ids.tolist():
            for relation_id in (0, 1):
                rows = bucket_rows.get((int(target_id), int(relation_id)), [])
                harmful_rows = sorted(
                    [
                        row
                        for row in rows
                        if float(row["delta_q"]) < 0.0 and bool(row["similarity_gate_pass"])
                    ],
                    key=lambda row: (float(row["delta_q"]), int(row["edge_pos"])),
                )
                selected_count = 0
                for row in harmful_rows:
                    edge_pos = int(row["edge_pos"])
                    u, v = row["edge"]
                    if edge_pos in conf_removed_positions:
                        row["selection_reason"] = "duplicate_already_selected"
                        duplicate_attempts += 1
                        continue
                    if degree_guard_enabled and (conf_out_degree[int(u)] <= 1 or conf_in_degree[int(v)] <= 1):
                        row["guard_blocked"] = True
                        row["selection_reason"] = "degree_guard_blocked"
                        continue
                    conf_removed_positions.add(edge_pos)
                    if degree_guard_enabled:
                        conf_out_degree[int(u)] -= 1
                        conf_in_degree[int(v)] -= 1
                    row["selected"] = True
                    row["selection_reason"] = "topk_harmful"
                    selected_count = 1
                    break
                conf_bucket_budget[(int(target_id), int(relation_id))] = int(selected_count)
        for row in candidate_rows:
            if row["selection_reason"] is None:
                if float(row["delta_q"]) >= 0.0:
                    row["selection_reason"] = "non_harmful_delta_q"
                elif not bool(row["similarity_gate_pass"]):
                    row["selection_reason"] = "similarity_gate_filtered"
                else:
                    row["selection_reason"] = "not_topk_harmful"

        rng = np.random.default_rng(int(self.seed))
        rand_removed_positions = set()
        rand_in_degree = base_in_degree.copy()
        rand_out_degree = base_out_degree.copy()
        random_selected_rows = []
        for target_id in routed_node_ids.tolist():
            for relation_id in (0, 1):
                desired = int(conf_bucket_budget.get((int(target_id), int(relation_id)), 0))
                if desired <= 0:
                    continue
                rows = list(bucket_rows.get((int(target_id), int(relation_id)), []))
                if not rows:
                    continue
                order = rng.permutation(len(rows))
                picked = 0
                for row_idx in order.tolist():
                    row = rows[int(row_idx)]
                    edge_pos = int(row["edge_pos"])
                    u, v = row["edge"]
                    if edge_pos in rand_removed_positions:
                        continue
                    if degree_guard_enabled and (rand_out_degree[int(u)] <= 1 or rand_in_degree[int(v)] <= 1):
                        continue
                    rand_removed_positions.add(edge_pos)
                    if degree_guard_enabled:
                        rand_out_degree[int(u)] -= 1
                        rand_in_degree[int(v)] -= 1
                    random_selected_rows.append(
                        {
                            "target_id": int(target_id),
                            "edge": [int(u), int(v)],
                            "relation": int(relation_id),
                            "edge_pos": int(edge_pos),
                            "selection_reason": "matched_local_random",
                        }
                    )
                    picked += 1
                    if picked >= desired:
                        break

        conf_edge_index, conf_edge_type = self._pruned_graph_from_removed_positions(
            edge_index_cpu,
            edge_type_cpu,
            conf_removed_positions,
        )
        rand_edge_index, rand_edge_type = self._pruned_graph_from_removed_positions(
            edge_index_cpu,
            edge_type_cpu,
            rand_removed_positions,
        )
        conf_zero_in, conf_zero_out = self._degree_arrays(conf_edge_index, num_nodes)
        rand_zero_in, rand_zero_out = self._degree_arrays(rand_edge_index, num_nodes)

        local_conf_context = self._rerun_frozen_g0_with_edge_override(
            stage_dir,
            "local_conf",
            conf_edge_index,
            conf_edge_type,
        )
        local_rand_context = self._rerun_frozen_g0_with_edge_override(
            stage_dir,
            "local_rand",
            rand_edge_index,
            rand_edge_type,
        )

        base_metrics = self._split_metrics_from_outputs(
            context["frozen_g0"]["outputs"],
            labels_np,
            self.val_mask,
            self.test_mask,
        )
        conf_metrics = self._split_metrics_from_outputs(
            local_conf_context["outputs"],
            labels_np,
            self.val_mask,
            self.test_mask,
        )
        rand_metrics = self._split_metrics_from_outputs(
            local_rand_context["outputs"],
            labels_np,
            self.val_mask,
            self.test_mask,
        )

        conf_selected_by_relation = {
            str(relation_id): int(sum(1 for edge_pos in conf_removed_positions if int(rel_np[edge_pos]) == relation_id))
            for relation_id in (0, 1)
        }
        rand_selected_by_relation = {
            str(relation_id): int(sum(1 for edge_pos in rand_removed_positions if int(rel_np[edge_pos]) == relation_id))
            for relation_id in (0, 1)
        }
        conf_selected_incident = np.zeros(num_nodes, dtype=np.int64)
        rand_selected_incident = np.zeros(num_nodes, dtype=np.int64)
        for edge_pos in conf_removed_positions:
            conf_selected_incident[int(src_np[edge_pos])] += 1
            if int(dst_np[edge_pos]) != int(src_np[edge_pos]):
                conf_selected_incident[int(dst_np[edge_pos])] += 1
        for edge_pos in rand_removed_positions:
            rand_selected_incident[int(src_np[edge_pos])] += 1
            if int(dst_np[edge_pos]) != int(src_np[edge_pos]):
                rand_selected_incident[int(dst_np[edge_pos])] += 1

        metrics_payload = {
            "contract": "local_conformal_prune_diag_metrics_v1",
            "q_source": "gnn_2hop_conformal_router_score",
            "routed_budget": routed_budget,
            "topk_per_relation": 1,
            "base": {
                "valid": base_metrics["valid"],
                "test": base_metrics["test"],
            },
            "local_conf": {
                "valid": conf_metrics["valid"],
                "test": conf_metrics["test"],
                "delta_vs_base": {
                    "valid_accuracy": float(conf_metrics["valid"]["accuracy"] - base_metrics["valid"]["accuracy"]),
                    "valid_macro_f1": float(conf_metrics["valid"]["macro_f1"] - base_metrics["valid"]["macro_f1"]),
                    "test_accuracy": float(conf_metrics["test"]["accuracy"] - base_metrics["test"]["accuracy"]),
                    "test_macro_f1": float(conf_metrics["test"]["macro_f1"] - base_metrics["test"]["macro_f1"]),
                },
            },
            "local_rand": {
                "valid": rand_metrics["valid"],
                "test": rand_metrics["test"],
                "delta_vs_base": {
                    "valid_accuracy": float(rand_metrics["valid"]["accuracy"] - base_metrics["valid"]["accuracy"]),
                    "valid_macro_f1": float(rand_metrics["valid"]["macro_f1"] - base_metrics["valid"]["macro_f1"]),
                    "test_accuracy": float(rand_metrics["test"]["accuracy"] - base_metrics["test"]["accuracy"]),
                    "test_macro_f1": float(rand_metrics["test"]["macro_f1"] - base_metrics["test"]["macro_f1"]),
                },
            },
            "routing": {
                "validation_threshold": threshold,
                "routed_counts": {
                    "train": int(routed_mask[self.train_mask].sum()),
                    "valid": int(routed_mask[self.val_mask].sum()),
                    "test": int(routed_mask[self.test_mask].sum()),
                    "all": int(routed_mask.sum()),
                },
                "tune_cal_split_metadata": router_bundle["split_metadata"],
            },
            "selection": {
                "candidate_edge_count": int(len(candidate_rows)),
                "harmful_candidate_count": int(sum(1 for row in candidate_rows if float(row["delta_q"]) < 0.0)),
                "negative_delta_ratio": float(sum(1 for row in candidate_rows if float(row["delta_q"]) < 0.0) / max(len(candidate_rows), 1)),
                "selected_harmful_edge_count": int(len(conf_removed_positions)),
                "selected_random_edge_count": int(len(rand_removed_positions)),
                "guard_blocked_count": int(sum(1 for row in candidate_rows if bool(row["guard_blocked"]))),
                "similarity_gate_threshold": similarity_gate_threshold,
                "similarity_gate_filtered_count": int(sum(1 for row in candidate_rows if not bool(row["similarity_gate_pass"]))),
                "duplicate_selected_attempts": int(duplicate_attempts),
                "selected_edges_per_relation": conf_selected_by_relation,
                "random_selected_edges_per_relation": rand_selected_by_relation,
                "local_conf_zero_in_nodes_after": int((conf_zero_in == 0).sum()),
                "local_conf_zero_out_nodes_after": int((conf_zero_out == 0).sum()),
                "local_rand_zero_in_nodes_after": int((rand_zero_in == 0).sum()),
                "local_rand_zero_out_nodes_after": int((rand_zero_out == 0).sum()),
            },
        }

        manifest = {
            "contract": "local_conformal_prune_diag_v1",
            "status": "completed",
            "stage": "local_conformal_prune_diag",
            "q_source": "gnn_2hop_conformal_router_score",
            "routed_budget": routed_budget,
            "topk_per_relation": 1,
            "candidate_scope": "1hop_incident_edges",
            "relation_split": [0, 1],
            "counterfactual_retrain": False,
            "propagation_retrain": "frozen_g0_after_edge_selection",
            "structural_guard": "no_zero_in_or_zero_out" if degree_guard_enabled else "disabled",
            "similarity_gate_threshold": similarity_gate_threshold,
            "test_labels_used_for_threshold": False,
            "oracle_labels_used": False,
            "diagnostic_only": True,
            "direction_mode": "incoming",
            "relation_mode": "agnostic",
            "frozen_g0_dir": str(context["frozen_g0"]["dir"]),
            "frozen_g0_manifest": context["frozen_g0"]["manifest"],
            "local_conf_frozen_g0_dir": str(local_conf_context["dir"]),
            "local_rand_frozen_g0_dir": str(local_rand_context["dir"]),
            "local_conf_graph_edge_index_path": str(stage_dir / "local_conf_edge_index.pt"),
            "local_conf_graph_edge_type_path": str(stage_dir / "local_conf_edge_type.pt"),
            "local_rand_graph_edge_index_path": str(stage_dir / "local_rand_edge_index.pt"),
            "local_rand_graph_edge_type_path": str(stage_dir / "local_rand_edge_type.pt"),
            "routing_threshold_valid": threshold,
            "tune_cal_split_metadata": router_bundle["split_metadata"],
            "risk_manifest_summary": {
                "thresholds": risk_manifest.get("thresholds"),
                "calibration_metadata": risk_manifest.get("calibration_metadata"),
            },
        }
        self._write_stage_manifest(
            stage_dir,
            manifest,
            artifact_namespace="stages/glance_counterfactual_router_internal",
            visibility="internal",
            resolved_task="glance_counterfactual_router_internal",
        )
        write_json(stage_dir / "metrics.json", metrics_payload)

        selected_rows_sorted = sorted(
            candidate_rows,
            key=lambda row: (
                int(row["target_id"]),
                int(row["relation"]),
                float(row["delta_q"]),
                int(row["edge_pos"]),
            ),
        )
        write_text(
            stage_dir / "selected_edges.jsonl",
            "\n".join(json.dumps({
                "target_id": int(row["target_id"]),
                "edge": [int(row["edge"][0]), int(row["edge"][1])],
                "relation": int(row["relation"]),
                "q_before": float(row["q_before"]),
                "q_after": float(row["q_after"]),
                "delta_q": float(row["delta_q"]),
                "selected": bool(row["selected"]),
                "selection_reason": str(row["selection_reason"]),
                "guard_blocked": bool(row["guard_blocked"]),
                "cosine": float(row["cosine"]),
                "similarity_gate_pass": bool(row["similarity_gate_pass"]),
            }, ensure_ascii=True, sort_keys=True) for row in selected_rows_sorted)
            + ("\n" if selected_rows_sorted else ""),
        )

        per_node_rows = []
        for node_idx in np.flatnonzero(self.test_mask):
            per_node_rows.append(
                {
                    "node_id": int(node_idx),
                    "label": int(labels_np[node_idx]),
                    "routed": bool(routed_mask[node_idx]),
                    "q_score": float(q_score[node_idx]),
                    "base_pred": int(base_metrics["pred"][node_idx]),
                    "local_conf_pred": int(conf_metrics["pred"][node_idx]),
                    "local_rand_pred": int(rand_metrics["pred"][node_idx]),
                    "base_correct": bool(base_metrics["pred"][node_idx] == labels_np[node_idx]),
                    "local_conf_correct": bool(conf_metrics["pred"][node_idx] == labels_np[node_idx]),
                    "local_rand_correct": bool(rand_metrics["pred"][node_idx] == labels_np[node_idx]),
                    "conf_selected_incident_edges": int(conf_selected_incident[node_idx]),
                    "rand_selected_incident_edges": int(rand_selected_incident[node_idx]),
                }
            )
        write_text(
            stage_dir / "per_node_test.jsonl",
            "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in per_node_rows) + "\n",
        )

        save_stage_artifacts(
            stage_dir,
            {
                "outputs.pt": {
                    "labels": labels_t,
                    "q_score": torch.tensor(q_score, dtype=torch.float32),
                    "routed_mask": torch.tensor(routed_mask, dtype=torch.bool),
                    "base_pred": torch.tensor(base_metrics["pred"], dtype=torch.long),
                    "local_conf_pred": torch.tensor(conf_metrics["pred"], dtype=torch.long),
                    "local_rand_pred": torch.tensor(rand_metrics["pred"], dtype=torch.long),
                    "local_conf_removed_edge_pos": torch.tensor(sorted(conf_removed_positions), dtype=torch.long),
                    "local_rand_removed_edge_pos": torch.tensor(sorted(rand_removed_positions), dtype=torch.long),
                },
                "local_conf_edge_index.pt": conf_edge_index,
                "local_conf_edge_type.pt": conf_edge_type,
                "local_rand_edge_index.pt": rand_edge_index,
                "local_rand_edge_type.pt": rand_edge_type,
                "risk_manifest": risk_manifest,
                "random_selection_summary": {
                    "selected_edges": random_selected_rows,
                },
                **base_bundle,
            },
        )
        write_text(
            stage_dir / "notes.md",
            "\n".join(
                [
                    "# Local conformal prune diagnostic",
                    "",
                    "This stage fits gnn_2hop_conformal on the original frozen_g0 posterior.",
                    "Routed nodes are selected by a validation-locked 10% threshold on router_score.",
                    "For each routed node and each relation bucket, incident edges are tested with single-edge counterfactual delta_q = q(G-e) - q(G).",
                    "Only top-1 harmful edges with delta_q < 0 are deleted for the local conformal graph; matched local random uses the same routed buckets and structural guard.",
                    f"Degree guard enabled: {degree_guard_enabled}.",
                    f"Similarity gate threshold: {similarity_gate_threshold}.",
                    "Propagation validation is performed by retraining frozen_g0 on the edited graph only.",
                ]
            )
            + "\n",
        )
        return {
            "stage": "local_conformal_prune_diag",
            "stage_dir": str(stage_dir),
            "metrics": metrics_payload,
        }

    def _run_local_conflict_prune_diag(self, stage_dir, base_bundle):
        context = self.ensure_backbone_context()
        gnn_outputs = context["gnn_outputs"]
        base_pred = gnn_outputs.get("pred")
        if base_pred is None:
            raise MissingFrozenArtifactError("local_conflict_prune_diag requires frozen_g0 outputs with pred.")

        edge_index, edge_type = self._active_graph_tensors()
        labels_t = torch.tensor(self.labels, dtype=torch.long)
        labels_np = labels_t.numpy()
        semantic_bundle = resolve_g0_feature_bundle(self.args, self.data)
        semantic_raw = semantic_bundle["raw_features"].detach().cpu().float()
        semantic_norm = F.normalize(semantic_raw, p=2, dim=1, eps=1e-12)
        degree_guard_enabled = not bool(getattr(self.args, "conflict_disable_degree_guard", False))
        topk_per_bucket = max(int(getattr(self.args, "conflict_topk_per_bucket", 1) or 1), 1)

        lm_head = self._build_or_load_lm_only_head()
        semantic_prob = lm_head["outputs"].get("prob_cal", lm_head["outputs"]["prob"]).detach().cpu().float()
        structural_bundle = self._fit_local_conflict_structural_view(edge_index, edge_type, labels_t)
        structural_outputs = structural_bundle["outputs"]
        structural_prob = structural_outputs["prob"].detach().cpu().float()
        structural_pred = structural_outputs["pred"].detach().cpu().long()

        router_bundle = self._fit_local_conflict_router(semantic_prob, structural_prob)
        conflict_score = router_bundle["conflict_score"]
        routed_mask = router_bundle["routed_mask"]
        routing_threshold = float(router_bundle["threshold"])
        routed_budget = float(router_bundle["budget"])

        edge_index_cpu = edge_index.detach().cpu().long()
        edge_type_cpu = edge_type.detach().cpu().long()
        src_np = edge_index_cpu[0].numpy().astype(np.int64)
        dst_np = edge_index_cpu[1].numpy().astype(np.int64)
        rel_np = edge_type_cpu.numpy().astype(np.int64)
        num_nodes = int(labels_t.numel())
        base_in_degree, base_out_degree = self._degree_arrays(edge_index_cpu, num_nodes)

        routed_node_ids = np.flatnonzero(routed_mask)
        routed_node_ids = np.asarray(
            sorted(routed_node_ids.tolist(), key=lambda node_id: (-float(conflict_score[int(node_id)]), int(node_id))),
            dtype=np.int64,
        )
        incident_edges = [[] for _ in range(num_nodes)]
        for edge_pos, (u, v) in enumerate(zip(src_np.tolist(), dst_np.tolist())):
            incident_edges[int(u)].append(int(edge_pos))
            if int(v) != int(u):
                incident_edges[int(v)].append(int(edge_pos))

        edge_conflict_after_cache = {}
        bucket_rows = {}
        candidate_rows = []
        for target_id in routed_node_ids.tolist():
            for edge_pos in incident_edges[int(target_id)]:
                u = int(src_np[edge_pos])
                v = int(dst_np[edge_pos])
                relation_id = int(rel_np[edge_pos])
                if int(target_id) == v:
                    target_role = "incoming"
                elif int(target_id) == u:
                    target_role = "outgoing"
                else:
                    continue
                bucket_key = (int(target_id), int(relation_id), str(target_role))
                rows = bucket_rows.setdefault(bucket_key, [])
                if edge_pos not in edge_conflict_after_cache:
                    cf_edge_index, cf_edge_type = self._remove_edge_at_position(edge_index_cpu, edge_type_cpu, edge_pos)
                    cf_outputs = self._run_structural_view_forward(structural_bundle["model"], cf_edge_index, cf_edge_type)
                    cf_prob = cf_outputs["prob"].detach().cpu().float()
                    cf_conflict = _js_divergence_from_probs(cf_prob, semantic_prob).detach().cpu().numpy().astype(np.float32)
                    edge_conflict_after_cache[int(edge_pos)] = cf_conflict
                conflict_before = float(conflict_score[int(target_id)])
                conflict_after = float(edge_conflict_after_cache[int(edge_pos)][int(target_id)])
                row = {
                    "target_id": int(target_id),
                    "edge": [u, v],
                    "relation": int(relation_id),
                    "target_role": str(target_role),
                    "edge_pos": int(edge_pos),
                    "semantic_cosine": float((semantic_norm[u] * semantic_norm[v]).sum().item()),
                    "conflict_before": conflict_before,
                    "conflict_after": conflict_after,
                    "delta_conflict": float(conflict_before - conflict_after),
                    "selected": False,
                    "selection_reason": None,
                    "guard_blocked": False,
                }
                rows.append(row)
                candidate_rows.append(row)

        conflict_removed_positions = set()
        conflict_bucket_budget = {}
        conflict_in_degree = base_in_degree.copy()
        conflict_out_degree = base_out_degree.copy()
        duplicate_attempts = 0
        for bucket_key in sorted(bucket_rows.keys(), key=lambda item: (item[0], item[1], item[2])):
            rows = bucket_rows.get(bucket_key, [])
            positive_rows = sorted(
                [row for row in rows if float(row["delta_conflict"]) > 0.0],
                key=lambda row: (-float(row["delta_conflict"]), int(row["edge_pos"])),
            )
            selected_count = 0
            for row in positive_rows:
                if selected_count >= topk_per_bucket:
                    break
                edge_pos = int(row["edge_pos"])
                u, v = row["edge"]
                if edge_pos in conflict_removed_positions:
                    row["selection_reason"] = "duplicate_already_selected"
                    duplicate_attempts += 1
                    continue
                if degree_guard_enabled and (conflict_out_degree[int(u)] <= 1 or conflict_in_degree[int(v)] <= 1):
                    row["guard_blocked"] = True
                    row["selection_reason"] = "degree_guard_blocked"
                    continue
                conflict_removed_positions.add(edge_pos)
                if degree_guard_enabled:
                    conflict_out_degree[int(u)] -= 1
                    conflict_in_degree[int(v)] -= 1
                row["selected"] = True
                row["selection_reason"] = "topk_positive_delta_conflict"
                selected_count += 1
            conflict_bucket_budget[bucket_key] = int(selected_count)

        for row in candidate_rows:
            if row["selection_reason"] is None:
                if float(row["delta_conflict"]) <= 0.0:
                    row["selection_reason"] = "non_positive_delta_conflict"
                else:
                    row["selection_reason"] = "not_topk_positive_delta_conflict"

        rng = np.random.default_rng(int(self.seed))
        rand_removed_positions = set()
        rand_in_degree = base_in_degree.copy()
        rand_out_degree = base_out_degree.copy()
        for bucket_key in sorted(bucket_rows.keys(), key=lambda item: (item[0], item[1], item[2])):
            desired = int(conflict_bucket_budget.get(bucket_key, 0))
            if desired <= 0:
                continue
            rows = list(bucket_rows.get(bucket_key, []))
            if not rows:
                continue
            order = rng.permutation(len(rows))
            picked = 0
            for row_idx in order.tolist():
                row = rows[int(row_idx)]
                edge_pos = int(row["edge_pos"])
                u, v = row["edge"]
                if edge_pos in rand_removed_positions:
                    continue
                if degree_guard_enabled and (rand_out_degree[int(u)] <= 1 or rand_in_degree[int(v)] <= 1):
                    continue
                rand_removed_positions.add(edge_pos)
                if degree_guard_enabled:
                    rand_out_degree[int(u)] -= 1
                    rand_in_degree[int(v)] -= 1
                picked += 1
                if picked >= desired:
                    break

        conflict_edge_index, conflict_edge_type = self._pruned_graph_from_removed_positions(
            edge_index_cpu,
            edge_type_cpu,
            conflict_removed_positions,
        )
        rand_edge_index, rand_edge_type = self._pruned_graph_from_removed_positions(
            edge_index_cpu,
            edge_type_cpu,
            rand_removed_positions,
        )
        conflict_zero_in, conflict_zero_out = self._degree_arrays(conflict_edge_index, num_nodes)
        rand_zero_in, rand_zero_out = self._degree_arrays(rand_edge_index, num_nodes)

        local_conflict_context = self._rerun_frozen_g0_with_edge_override(
            stage_dir,
            "local_conflict",
            conflict_edge_index,
            conflict_edge_type,
        )
        local_rand_context = self._rerun_frozen_g0_with_edge_override(
            stage_dir,
            "local_rand",
            rand_edge_index,
            rand_edge_type,
        )

        base_metrics = self._split_metrics_from_outputs(
            context["frozen_g0"]["outputs"],
            labels_np,
            self.val_mask,
            self.test_mask,
        )
        conflict_metrics = self._split_metrics_from_outputs(
            local_conflict_context["outputs"],
            labels_np,
            self.val_mask,
            self.test_mask,
        )
        rand_metrics = self._split_metrics_from_outputs(
            local_rand_context["outputs"],
            labels_np,
            self.val_mask,
            self.test_mask,
        )

        selected_by_relation_role = {}
        selected_by_bucket = {}
        for bucket_key, count in conflict_bucket_budget.items():
            relation_role_key = f"{bucket_key[1]}::{bucket_key[2]}"
            selected_by_relation_role[relation_role_key] = int(selected_by_relation_role.get(relation_role_key, 0) + count)
            selected_by_bucket[f"{bucket_key[0]}::{bucket_key[1]}::{bucket_key[2]}"] = int(count)

        routed_rows = []
        p_sem_np = semantic_prob.detach().cpu().numpy()
        p_struct_np = structural_prob.detach().cpu().numpy()
        base_pred_np = base_pred.detach().cpu().numpy().astype(np.int64)
        for split_name, idx_mask in (("train", self.train_mask), ("valid", self.val_mask), ("test", self.test_mask)):
            for node_id in np.flatnonzero(idx_mask).tolist():
                routed_rows.append(
                    {
                        "node_id": int(node_id),
                        "split": str(split_name),
                        "conflict_score": float(conflict_score[int(node_id)]),
                        "routing_threshold": float(routing_threshold),
                        "routed": bool(routed_mask[int(node_id)]),
                        "p_sem": [float(item) for item in p_sem_np[int(node_id)].tolist()],
                        "p_struct": [float(item) for item in p_struct_np[int(node_id)].tolist()],
                        "base_pred": int(base_pred_np[int(node_id)]),
                        "base_correct": bool(base_pred_np[int(node_id)] == int(labels_np[int(node_id)])),
                    }
                )

        selected_rows_sorted = sorted(
            candidate_rows,
            key=lambda row: (
                int(row["target_id"]),
                int(row["relation"]),
                str(row["target_role"]),
                -float(row["delta_conflict"]),
                int(row["edge_pos"]),
            ),
        )
        evidence_rows = [
            {
                "target_id": int(row["target_id"]),
                "edge": [int(row["edge"][0]), int(row["edge"][1])],
                "relation": int(row["relation"]),
                "target_role": str(row["target_role"]),
                "edge_pos": int(row["edge_pos"]),
                "semantic_cosine": float(row["semantic_cosine"]),
                "conflict_before": float(row["conflict_before"]),
                "conflict_after": float(row["conflict_after"]),
                "delta_conflict": float(row["delta_conflict"]),
                "removed_from_propagation": True,
                "evidence_role": "camouflage_conflict_evidence",
            }
            for row in selected_rows_sorted
            if bool(row["selected"])
        ]

        metrics_payload = {
            "contract": "local_conflict_prune_diag_metrics_v1",
            "routing_mode": "dignn_style_topology_semantic_conflict",
            "paper_faithful_dignn_style": True,
            "official_code_verified": False,
            "routed_budget": routed_budget,
            "topk_per_bucket": int(topk_per_bucket),
            "base": {
                "valid": base_metrics["valid"],
                "test": base_metrics["test"],
            },
            "local_conflict": {
                "valid": conflict_metrics["valid"],
                "test": conflict_metrics["test"],
                "delta_vs_base": {
                    "valid_accuracy": float(conflict_metrics["valid"]["accuracy"] - base_metrics["valid"]["accuracy"]),
                    "valid_macro_f1": float(conflict_metrics["valid"]["macro_f1"] - base_metrics["valid"]["macro_f1"]),
                    "test_accuracy": float(conflict_metrics["test"]["accuracy"] - base_metrics["test"]["accuracy"]),
                    "test_macro_f1": float(conflict_metrics["test"]["macro_f1"] - base_metrics["test"]["macro_f1"]),
                },
            },
            "local_rand": {
                "valid": rand_metrics["valid"],
                "test": rand_metrics["test"],
                "delta_vs_base": {
                    "valid_accuracy": float(rand_metrics["valid"]["accuracy"] - base_metrics["valid"]["accuracy"]),
                    "valid_macro_f1": float(rand_metrics["valid"]["macro_f1"] - base_metrics["valid"]["macro_f1"]),
                    "test_accuracy": float(rand_metrics["test"]["accuracy"] - base_metrics["test"]["accuracy"]),
                    "test_macro_f1": float(rand_metrics["test"]["macro_f1"] - base_metrics["test"]["macro_f1"]),
                },
            },
            "structural_view": {
                "train": structural_bundle["metrics"]["train"],
                "valid": structural_bundle["metrics"]["valid"],
                "test": structural_bundle["metrics"]["test"],
                "fit_summary": structural_bundle["fit_summary"],
            },
            "routing": {
                "routing_threshold_valid": float(routing_threshold),
                "routed_counts": {
                    "train": int(routed_mask[self.train_mask].sum()),
                    "valid": int(routed_mask[self.val_mask].sum()),
                    "test": int(routed_mask[self.test_mask].sum()),
                    "all": int(routed_mask.sum()),
                },
                "routed_count_valid_target": int(router_bundle["routed_count_valid_target"]),
                "test_conflict_summary": router_bundle["test_conflict_summary"],
            },
            "selection": {
                "candidate_edge_count": int(len(candidate_rows)),
                "positive_delta_conflict_count": int(sum(1 for row in candidate_rows if float(row["delta_conflict"]) > 0.0)),
                "positive_delta_conflict_ratio": float(sum(1 for row in candidate_rows if float(row["delta_conflict"]) > 0.0) / max(len(candidate_rows), 1)),
                "selected_conflict_edge_count": int(len(conflict_removed_positions)),
                "selected_random_edge_count": int(len(rand_removed_positions)),
                "guard_blocked_count": int(sum(1 for row in candidate_rows if bool(row["guard_blocked"]))),
                "duplicate_selected_attempts": int(duplicate_attempts),
                "selected_edges_per_relation_role": selected_by_relation_role,
                "selected_edges_per_bucket": selected_by_bucket,
                "local_conflict_zero_in_nodes_after": int((conflict_zero_in == 0).sum()),
                "local_conflict_zero_out_nodes_after": int((conflict_zero_out == 0).sum()),
                "local_rand_zero_in_nodes_after": int((rand_zero_in == 0).sum()),
                "local_rand_zero_out_nodes_after": int((rand_zero_out == 0).sum()),
                "conflict_before_mean": float(np.mean(conflict_score)) if conflict_score.size else 0.0,
                "conflict_after_mean_selected_targets": float(np.mean([row["conflict_after"] for row in selected_rows_sorted if bool(row["selected"])])) if evidence_rows else 0.0,
                "selected_semantic_cosine_mean": float(np.mean([row["semantic_cosine"] for row in selected_rows_sorted if bool(row["selected"])])) if evidence_rows else 0.0,
            },
        }

        manifest = {
            "contract": "local_conflict_prune_diag_v1",
            "status": "completed",
            "stage": "local_conflict_prune_diag",
            "routing_mode": "dignn_style_topology_semantic_conflict",
            "paper_faithful_dignn_style": True,
            "paper_faithful_dig_in_gnn_style_local_selection": True,
            "official_code_verified": False,
            "diagnostic_only": True,
            "candidate_scope": "1hop_incident_edges",
            "bucket_definition": "(relation, target_role)",
            "target_roles": ["incoming", "outgoing"],
            "topk_per_bucket": int(topk_per_bucket),
            "routed_budget": float(routed_budget),
            "routing_threshold_valid": float(routing_threshold),
            "structural_guard": "no_zero_in/no_zero_out" if degree_guard_enabled else "disabled",
            "structural_view": structural_bundle["model_config"],
            "semantic_view": {
                "source": lm_head.get("source", "lm_only_head"),
                "manifest": lm_head["manifest"],
            },
            "conflict_definition": "JS(p_struct(node;G), p_sem(node))",
            "counterfactual_definition": "delta_conflict = conflict_before - conflict_after_single_edge_delete",
            "test_labels_used_for_training": False,
            "test_labels_used_for_threshold": False,
            "oracle_labels_used": False,
            "frozen_g0_dir": str(context["frozen_g0"]["dir"]),
            "frozen_g0_manifest": context["frozen_g0"]["manifest"],
            "graph_source": context["graph_bundle"]["source"],
            "graph_source_root": context["graph_bundle"]["source_root"],
            "local_conflict_frozen_g0_dir": str(local_conflict_context["dir"]),
            "local_rand_frozen_g0_dir": str(local_rand_context["dir"]),
            "local_conflict_edge_index_path": str(stage_dir / "local_conflict_edge_index.pt"),
            "local_conflict_edge_type_path": str(stage_dir / "local_conflict_edge_type.pt"),
            "local_rand_edge_index_path": str(stage_dir / "local_rand_edge_index.pt"),
            "local_rand_edge_type_path": str(stage_dir / "local_rand_edge_type.pt"),
            "conflict_evidence_edges_path": str(stage_dir / "conflict_evidence_edges.jsonl"),
        }
        self._write_stage_manifest(
            stage_dir,
            manifest,
            artifact_namespace="stages/glance_oracle_refinement_internal",
            visibility="internal",
            resolved_task="glance_oracle_refinement_internal",
        )
        write_json(stage_dir / "metrics.json", metrics_payload)
        write_text(
            stage_dir / "selected_edges.jsonl",
            "\n".join(
                json.dumps(
                    {
                        "target_id": int(row["target_id"]),
                        "edge": [int(row["edge"][0]), int(row["edge"][1])],
                        "relation": int(row["relation"]),
                        "target_role": str(row["target_role"]),
                        "edge_pos": int(row["edge_pos"]),
                        "semantic_cosine": float(row["semantic_cosine"]),
                        "conflict_before": float(row["conflict_before"]),
                        "conflict_after": float(row["conflict_after"]),
                        "delta_conflict": float(row["delta_conflict"]),
                        "selected": bool(row["selected"]),
                        "selection_reason": str(row["selection_reason"]),
                        "guard_blocked": bool(row["guard_blocked"]),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                for row in selected_rows_sorted
            )
            + ("\n" if selected_rows_sorted else ""),
        )
        write_text(
            stage_dir / "routed_nodes.jsonl",
            "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in routed_rows)
            + ("\n" if routed_rows else ""),
        )
        write_text(
            stage_dir / "conflict_evidence_edges.jsonl",
            "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in evidence_rows)
            + ("\n" if evidence_rows else ""),
        )
        write_torch(stage_dir / "local_conflict_edge_index.pt", conflict_edge_index)
        write_torch(stage_dir / "local_conflict_edge_type.pt", conflict_edge_type)
        write_torch(stage_dir / "local_rand_edge_index.pt", rand_edge_index)
        write_torch(stage_dir / "local_rand_edge_type.pt", rand_edge_type)
        save_stage_artifacts(
            stage_dir,
            {
                "view_outputs.pt": {
                    "semantic_prob": semantic_prob,
                    "structural_prob": structural_prob,
                    "structural_pred": structural_pred,
                    "conflict_score": torch.tensor(conflict_score, dtype=torch.float32),
                    "routed_mask": torch.tensor(routed_mask, dtype=torch.bool),
                    "routing_threshold": torch.tensor([routing_threshold], dtype=torch.float32),
                    "structural_node_repr": structural_outputs["node_repr"],
                },
                **base_bundle,
            },
        )
        write_text(
            stage_dir / "notes.md",
            "\n".join(
                [
                    "# DIGNN-style local conflict propagation refiner",
                    "",
                    "This stage trains a structural-only RGCN view on the active graph and compares it against an LM-only semantic anchor.",
                    "Routed hard nodes are selected by high JS divergence between structural and semantic posteriors.",
                    "Candidate edges are target-centered 1-hop incident edges, bucketed by (relation, target_role).",
                    "Propagation validation is performed by retraining frozen_g0 on the edited graph only.",
                    "Removed conflict edges are preserved as evidence-ready provenance for later local evidence graph consumption.",
                ]
            )
            + "\n",
        )
        return {
            "stage": "local_conflict_prune_diag",
            "stage_dir": str(stage_dir),
            "metrics": metrics_payload,
        }

    @staticmethod
    def _node_cross_entropy_vector(prob, labels):
        prob_t = prob.detach().cpu().float() if torch.is_tensor(prob) else torch.tensor(prob, dtype=torch.float32)
        labels_t = labels.detach().cpu().long() if torch.is_tensor(labels) else torch.tensor(labels, dtype=torch.long)
        row = torch.arange(labels_t.numel(), dtype=torch.long)
        return (-torch.log(prob_t[row, labels_t].clamp_min(1e-8))).cpu()

    @staticmethod
    def _budget_key(budget):
        return int(round(float(budget) * 100))

    def _build_counterfactual_budget_rows(
        self,
        budgets,
        risk_score,
        pred_gnn,
        pred_refiner,
        labels_np,
        eval_idx,
        prob_gnn,
        prob_refiner,
    ):
        risk_score = np.asarray(risk_score, dtype=np.float64)
        pred_gnn_np = np.asarray(pred_gnn, dtype=np.int64)
        pred_ref_np = np.asarray(pred_refiner, dtype=np.int64)
        labels_np = np.asarray(labels_np, dtype=np.int64)
        eval_idx = np.asarray(eval_idx, dtype=np.int64)
        prob_gnn_t = prob_gnn.detach().cpu().float() if torch.is_tensor(prob_gnn) else torch.tensor(prob_gnn, dtype=torch.float32)
        prob_ref_t = prob_refiner.detach().cpu().float() if torch.is_tensor(prob_refiner) else torch.tensor(prob_refiner, dtype=torch.float32)

        order = eval_idx[np.argsort(-risk_score[eval_idx])] if eval_idx.size else np.asarray([], dtype=np.int64)
        base_eval = _score_all(labels_np[eval_idx], pred_gnn_np[eval_idx]) if eval_idx.size else {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "bot_f1": 0.0,
            "count": 0,
        }
        base_wrong_count = int((pred_gnn_np[eval_idx] != labels_np[eval_idx]).sum()) if eval_idx.size else 0
        base_correct_count = int(eval_idx.size - base_wrong_count)
        rows = []
        payloads = {}
        for budget in parse_budget_list(budgets):
            k = min(max(int(eval_idx.size * float(budget)), 1), eval_idx.size) if eval_idx.size else 0
            routed = order[:k]
            routed_mask = np.zeros(labels_np.shape[0], dtype=bool)
            routed_mask[routed] = True
            final_pred = pred_gnn_np.copy()
            final_pred[routed] = pred_ref_np[routed]
            test_mask = np.zeros(labels_np.shape[0], dtype=bool)
            test_mask[eval_idx] = True
            delta = _delta_table(pred_gnn_np, final_pred, labels_np, test_mask)
            routed_delta = _delta_table(pred_gnn_np, final_pred, labels_np, routed_mask)
            final_eval = _score_all(labels_np[eval_idx], final_pred[eval_idx]) if eval_idx.size else dict(base_eval)
            row = {
                "budget": float(budget),
                "routed_count": int(k),
                "fix": int(delta["fix"]),
                "break": int(delta["broke"]),
                "net": int(delta["net"]),
                "routed_fix": int(routed_delta["fix"]),
                "routed_break": int(routed_delta["broke"]),
                "routed_net": int(routed_delta["net"]),
                "wrong_node_fix_rate": float(delta["fix"] / base_wrong_count) if base_wrong_count else 0.0,
                "correct_node_break_rate": float(delta["broke"] / base_correct_count) if base_correct_count else 0.0,
                "accuracy": float(final_eval["accuracy"]),
                "macro_f1": float(final_eval["macro_f1"]),
                "bot_f1": float(final_eval["bot_f1"]),
                "delta_accuracy": float(final_eval["accuracy"] - base_eval["accuracy"]),
                "delta_macro_f1": float(final_eval["macro_f1"] - base_eval["macro_f1"]),
                "delta_bot_f1": float(final_eval["bot_f1"] - base_eval["bot_f1"]),
                "mean_base_confidence_routed": float(prob_gnn_t[routed].max(dim=1).values.mean().item()) if k else 0.0,
                "mean_refiner_confidence_routed": float(prob_ref_t[routed].max(dim=1).values.mean().item()) if k else 0.0,
            }
            rows.append(row)
            payloads[str(self._budget_key(budget))] = {
                "metrics": final_eval,
                "delta_vs_base": {
                    "accuracy": row["delta_accuracy"],
                    "macro_f1": row["delta_macro_f1"],
                    "bot_f1": row["delta_bot_f1"],
                },
                "routed_node_ids": [int(item) for item in routed.tolist()],
                "final_pred": torch.tensor(final_pred, dtype=torch.long),
                "routed_mask": torch.tensor(routed_mask, dtype=torch.bool),
            }
        return rows, payloads

    def _load_glance_counterfactual_router_artifact(self):
        manifest = self._require_stage_json("glance_counterfactual_router", "manifest")
        metrics = self._require_stage_json("glance_counterfactual_router", "metrics")
        comparison = self._read_stage_json("glance_counterfactual_router", "comparison")
        risk_manifest = self._require_stage_json("glance_counterfactual_router", "risk_manifest")
        outputs = self._require_stage_tensor("glance_counterfactual_router", "outputs")
        router_scores = outputs.get("risk_score")
        if router_scores is None:
            raise MissingFrozenArtifactError(
                "glance_counterfactual_router outputs.pt must contain risk_score for budgeted refine."
            )
        def _read_budget_curve_rows(filename):
            budget_curve_path = self.experiment_root / "stages" / "glance_counterfactual_router" / filename
            if not budget_curve_path.exists():
                return []
            import csv
            with open(budget_curve_path, "r", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
        return {
            "manifest": manifest,
            "metrics": metrics,
            "comparison": comparison,
            "risk_manifest": risk_manifest,
            "outputs": outputs,
            "router_scores": router_scores,
            "budget_curve_rows": _read_budget_curve_rows("budget_curve.csv"),
            "valid_budget_curve_rows": _read_budget_curve_rows("valid_budget_curve.csv"),
            "test_budget_curve_rows": _read_budget_curve_rows("test_budget_curve.csv"),
        }

    def _build_glance_joint_budget_rows(
        self,
        budgets,
        split_idx,
        labels_np,
        labels_t,
        base_logits,
        base_prob,
        base_pred,
        refiner_logits,
        refiner_prob,
        refiner_pred,
        router_score,
        router_prob,
        oracle_advantage,
    ):
        split_idx = np.asarray(split_idx, dtype=np.int64).reshape(-1)
        labels_np = np.asarray(labels_np, dtype=np.int64)
        base_pred_np = base_pred.detach().cpu().numpy() if torch.is_tensor(base_pred) else np.asarray(base_pred, dtype=np.int64)
        refiner_pred_np = refiner_pred.detach().cpu().numpy() if torch.is_tensor(refiner_pred) else np.asarray(refiner_pred, dtype=np.int64)
        router_score_np = router_score.detach().cpu().numpy() if torch.is_tensor(router_score) else np.asarray(router_score, dtype=np.float64)
        router_prob_t = router_prob.detach().cpu().float() if torch.is_tensor(router_prob) else torch.tensor(router_prob, dtype=torch.float32)
        oracle_advantage_t = oracle_advantage.detach().cpu().float() if torch.is_tensor(oracle_advantage) else torch.tensor(oracle_advantage, dtype=torch.float32)
        base_logits_t = base_logits.detach().cpu().float() if torch.is_tensor(base_logits) else torch.tensor(base_logits, dtype=torch.float32)
        base_prob_t = base_prob.detach().cpu().float() if torch.is_tensor(base_prob) else torch.tensor(base_prob, dtype=torch.float32)
        refiner_logits_t = refiner_logits.detach().cpu().float() if torch.is_tensor(refiner_logits) else torch.tensor(refiner_logits, dtype=torch.float32)
        refiner_prob_t = refiner_prob.detach().cpu().float() if torch.is_tensor(refiner_prob) else torch.tensor(refiner_prob, dtype=torch.float32)

        base_eval = _classification_metrics_from_logits(
            base_logits_t,
            labels_t,
            torch.tensor(split_idx, dtype=torch.long),
        ) if split_idx.size else {"loss": float("inf"), "accuracy": 0.0, "macro_f1": 0.0, "bot_f1": 0.0, "count": 0}
        base_wrong_count = int((base_pred_np[split_idx] != labels_np[split_idx]).sum()) if split_idx.size else 0
        base_correct_count = int(split_idx.size - base_wrong_count)
        order = split_idx[np.argsort(-router_score_np[split_idx])] if split_idx.size else np.asarray([], dtype=np.int64)

        rows = []
        payloads = {}
        for budget in parse_budget_list(budgets):
            k = min(max(int(round(split_idx.size * float(budget))), 1), split_idx.size) if split_idx.size else 0
            routed = order[:k]
            routed_mask = torch.zeros(base_pred_np.shape[0], dtype=torch.bool)
            if k:
                routed_mask[torch.tensor(routed, dtype=torch.long)] = True
            final_logits = base_logits_t.clone()
            final_prob = base_prob_t.clone()
            final_pred = base_pred.detach().cpu().clone() if torch.is_tensor(base_pred) else torch.tensor(base_pred_np, dtype=torch.long)
            if k:
                routed_t = torch.tensor(routed, dtype=torch.long)
                final_logits[routed_t] = refiner_logits_t[routed_t]
                final_prob[routed_t] = refiner_prob_t[routed_t]
                final_pred[routed_t] = refiner_pred_t = torch.tensor(refiner_pred_np[routed], dtype=torch.long)
            split_metrics = _classification_metrics_from_logits(
                final_logits,
                labels_t,
                torch.tensor(split_idx, dtype=torch.long),
            ) if split_idx.size else {"loss": float("inf"), "accuracy": 0.0, "macro_f1": 0.0, "bot_f1": 0.0, "count": 0}
            delta = _delta_table(base_pred_np, final_pred.numpy(), labels_np, _mask_from_idx(labels_np.shape[0], split_idx))

            routed_wrong_count = int((base_pred_np[routed] != labels_np[routed]).sum()) if k else 0
            routed_correct_count = int(k - routed_wrong_count)
            conditional_fix = int((refiner_pred_np[routed] == labels_np[routed]).sum()) if k else 0
            conditional_fix -= int((base_pred_np[routed] == labels_np[routed]).sum()) if k else 0
            row = {
                "budget": float(budget),
                "routed_count": int(k),
                "query_rate": float(k / max(int(split_idx.size), 1)),
                "fix": int(delta["fix"]),
                "break": int(delta["broke"]),
                "net": int(delta["net"]),
                "accuracy": float(split_metrics["accuracy"]),
                "macro_f1": float(split_metrics["macro_f1"]),
                "bot_f1": float(split_metrics["bot_f1"]),
                "loss": float(split_metrics["loss"]),
                "delta_accuracy": float(split_metrics["accuracy"] - base_eval["accuracy"]),
                "delta_macro_f1": float(split_metrics["macro_f1"] - base_eval["macro_f1"]),
                "delta_bot_f1": float(split_metrics["bot_f1"] - base_eval["bot_f1"]),
                "wrong_node_fix_rate": float(delta["fix"] / base_wrong_count) if base_wrong_count else 0.0,
                "correct_node_break_rate": float(delta["broke"] / base_correct_count) if base_correct_count else 0.0,
                "routed_wrong_count": int(routed_wrong_count),
                "routed_correct_count": int(routed_correct_count),
                "routed_wrong_coverage": float(routed_wrong_count / base_wrong_count) if base_wrong_count else 0.0,
                "routed_wrong_precision": float(routed_wrong_count / k) if k else 0.0,
                "conditional_fix_rate_on_selected_wrong": float(delta["fix"] / routed_wrong_count) if routed_wrong_count else 0.0,
                "conditional_break_rate_on_selected_correct": float(delta["broke"] / routed_correct_count) if routed_correct_count else 0.0,
                "mean_router_score_routed": float(router_score_np[routed].mean()) if k else 0.0,
                "mean_router_prob_routed": float(router_prob_t[routed].mean().item()) if k else 0.0,
                "mean_oracle_advantage_routed": float(oracle_advantage_t[routed].mean().item()) if k else 0.0,
            }
            rows.append(row)
            payloads[str(self._budget_key(budget))] = {
                "row": row,
                "routed_node_ids": [int(item) for item in routed.tolist()],
                "outputs": {
                    "logits": final_logits,
                    "prob": final_prob,
                    "pred": final_pred,
                    "routed_mask": routed_mask,
                    "router_prob": router_prob_t,
                    "router_score": torch.tensor(router_score_np, dtype=torch.float32),
                    "oracle_advantage": oracle_advantage_t,
                    "routed_count": int(k),
                },
            }
        return rows, payloads

    def _run_glance_counterfactual_router(self, stage_dir, base_bundle):
        context = self.ensure_backbone_context()
        semantic = self._load_semantic_finetune_artifact()
        refiner_artifact = self._load_glance_refiner_artifact()
        gnn_outputs = context["gnn_outputs"]
        p_gnn = gnn_outputs.get("prob")
        logits_gnn = gnn_outputs.get("logits")
        pred_gnn = gnn_outputs.get("pred")
        z_gnn = gnn_outputs.get("node_repr")
        if any(item is None for item in (p_gnn, logits_gnn, pred_gnn, z_gnn)):
            raise MissingFrozenArtifactError(
                "glance_counterfactual_router requires frozen_g0 outputs with prob/logits/pred/node_repr."
            )

        p_gnn = p_gnn.detach().cpu().float()
        logits_gnn = logits_gnn.detach().cpu().float()
        pred_gnn = pred_gnn.detach().cpu().long()
        z_gnn = z_gnn.detach().cpu().float()
        labels_t = torch.tensor(self.labels, dtype=torch.long)
        labels_np = labels_t.numpy()
        train_idx = _idx_numpy(self.data["train_idx"])
        valid_idx = _idx_numpy(self.data["valid_idx"])
        test_idx = _idx_numpy(self.data["test_idx"])
        semantic_views = self._build_k_hop_semantic_views(
            semantic["embeddings"],
            self.data.get("edge_index"),
        )
        if train_idx.size == 0:
            raise MissingFrozenArtifactError("glance_counterfactual_router requires a non-empty train_idx.")
        ref_outputs = refiner_artifact["outputs"]
        prob_ref = ref_outputs.get("prob")
        pred_ref = ref_outputs.get("pred")
        logits_ref = ref_outputs.get("logits")
        if prob_ref is None or pred_ref is None:
            raise MissingFrozenArtifactError(
                f"{refiner_artifact['stage_name']} outputs must include 'prob' and 'pred' for counterfactual routing."
            )
        prob_ref = prob_ref.detach().cpu().float() if torch.is_tensor(prob_ref) else torch.tensor(prob_ref, dtype=torch.float32)
        pred_ref = pred_ref.detach().cpu().long() if torch.is_tensor(pred_ref) else torch.tensor(pred_ref, dtype=torch.long)
        logits_ref = logits_ref.detach().cpu().float() if torch.is_tensor(logits_ref) else (
            torch.log(prob_ref.clamp_min(1e-8)) if logits_ref is None else torch.tensor(logits_ref, dtype=torch.float32)
        )

        gnn_loss = self._node_cross_entropy_vector(p_gnn, labels_t)
        refiner_loss = self._node_cross_entropy_vector(prob_ref, labels_t)
        gnn_correct = pred_gnn.eq(labels_t).float()
        refiner_correct = pred_ref.eq(labels_t).float()
        advantage = (gnn_loss - refiner_loss - float(getattr(self.args, "glance_llm_query_cost", 0.2))).numpy()

        budgets = self._router_budgets()
        router = GlanceForContextResidualRiskSelector(
            budgets=budgets,
            training_objective="glance_advantage",
            llm_query_cost=float(getattr(self.args, "glance_llm_query_cost", 0.2)),
        )
        router.fit(
            logits_gnn,
            p_gnn,
            labels_t,
            train_idx=train_idx,
            val_idx=valid_idx,
            edge_index=self.data.get("edge_index"),
            edge_type=self.data.get("edge_type"),
            node_repr=z_gnn,
            fit_idx=train_idx,
            gnn_loss=gnn_loss,
            llm_loss=refiner_loss,
            gnn_correct=gnn_correct,
            llm_correct=refiner_correct,
            training_objective="glance_advantage",
            llm_query_cost=float(getattr(self.args, "glance_llm_query_cost", 0.2)),
        )
        risk_manifest = router.build_manifest(
            logits_gnn,
            p_gnn,
            labels_t,
            val_idx=valid_idx,
            test_idx=test_idx,
            edge_index=self.data.get("edge_index"),
            edge_type=self.data.get("edge_type"),
            node_repr=z_gnn,
            q_probs=p_gnn,
            gnn_loss=gnn_loss,
            llm_loss=refiner_loss,
            gnn_correct=gnn_correct,
            llm_correct=refiner_correct,
            training_objective="glance_advantage",
            llm_query_cost=float(getattr(self.args, "glance_llm_query_cost", 0.2)),
        )
        risk_score = np.asarray(risk_manifest["risk_score"], dtype=np.float32)
        calibration_metadata = dict(risk_manifest.get("calibration_metadata", {}))
        split_audit = {
            "train_count": int(train_idx.size),
            "valid_count": int(valid_idx.size),
            "test_count": int(test_idx.size),
            "train_valid_overlap": int(np.intersect1d(train_idx, valid_idx).size),
            "train_test_overlap": int(np.intersect1d(train_idx, test_idx).size),
            "valid_test_overlap": int(np.intersect1d(valid_idx, test_idx).size),
        }
        calibration_metadata.update({
            "source": "glance_counterfactual_router",
            "paper_identity": "GLANCE-inspired counterfactual advantage router",
            "paper_faithful_glance_inspired": True,
            "official_code_verified": False,
            "counterfactual_outcome_used": True,
            "llm_counterfactual_outcome_used": True,
            "not_a_new_bot_classifier": True,
            "risk_score_semantics": "learned_router_score_proxy_for_counterfactual_advantage",
            "router_score_semantics": "deterministic_top_k_proxy_for_routing",
            "oracle_advantage_semantics": "counterfactual_reward_loss_gnn_minus_loss_refiner_minus_query_cost",
            "test_labels_used_for_training": False,
            "test_labels_used_for_threshold": False,
            "train_target": "1[loss_gnn - loss_refiner - cost > 0]",
            "refiner_input": "[z_gnn || z_sem_0 || z_sem_1 || z_sem_2]",
            "semantic_source": "semantic_finetune_roberta_finetuned",
            "split_audit": split_audit,
            "counterfactual_source_refiner": refiner_artifact["stage_name"],
        })
        risk_manifest["calibration_metadata"] = calibration_metadata

        valid_budget_rows, valid_budget_payloads = self._build_counterfactual_budget_rows(
            budgets=budgets,
            risk_score=risk_score,
            pred_gnn=pred_gnn.numpy(),
            pred_refiner=pred_ref.numpy(),
            labels_np=labels_np,
            eval_idx=valid_idx,
            prob_gnn=p_gnn,
            prob_refiner=prob_ref,
        )
        test_budget_rows, test_budget_payloads = self._build_counterfactual_budget_rows(
            budgets=budgets,
            risk_score=risk_score,
            pred_gnn=pred_gnn.numpy(),
            pred_refiner=pred_ref.numpy(),
            labels_np=labels_np,
            eval_idx=test_idx,
            prob_gnn=p_gnn,
            prob_refiner=prob_ref,
        )
        base_test = _score_all(labels_np[test_idx], pred_gnn.numpy()[test_idx])
        full_refiner_test = _score_all(labels_np[test_idx], pred_ref.numpy()[test_idx])
        full_delta = _delta_table(pred_gnn.numpy(), pred_ref.numpy(), labels_np, self.test_mask)
        base_wrong_count = int((pred_gnn.numpy()[test_idx] != labels_np[test_idx]).sum())
        base_correct_count = int(test_idx.size - base_wrong_count)
        selected_valid_row, best_key, valid_rows_by_key = self._select_best_budget_row(valid_budget_rows)
        test_rows_by_key = {str(self._budget_key(item["budget"])): item for item in test_budget_rows}
        selected_test_row = test_rows_by_key.get(best_key) if best_key is not None else None
        best_payload = test_budget_payloads.get(best_key, {}) if best_key is not None else {}
        best_final_pred = best_payload.get("final_pred", pred_gnn)
        best_routed_mask = best_payload.get("routed_mask", torch.zeros_like(pred_gnn, dtype=torch.bool))

        metrics_payload = {
            "contract": "glance_counterfactual_router_metrics_v1",
            "routing_mode": "glance_counterfactual_advantage",
            "paper_faithful_glance_inspired": True,
            "official_code_verified": False,
            "test_labels_used_for_training": False,
            "test_labels_used_for_threshold": False,
            "base_test": base_test,
            "full_refiner_test": full_refiner_test,
            "full_refiner_delta_vs_base": {
                "accuracy": float(full_refiner_test["accuracy"] - base_test["accuracy"]),
                "macro_f1": float(full_refiner_test["macro_f1"] - base_test["macro_f1"]),
                "bot_f1": float(full_refiner_test["bot_f1"] - base_test["bot_f1"]),
            },
            "full_refiner_fix_break": {
                "fix": int(full_delta["fix"]),
                "break": int(full_delta["broke"]),
                "net": int(full_delta["net"]),
                "wrong_node_fix_rate": float(full_delta["fix"] / base_wrong_count) if base_wrong_count else 0.0,
                "correct_node_break_rate": float(full_delta["broke"] / base_correct_count) if base_correct_count else 0.0,
            },
            "router_score_semantics": "learned_router_score_proxy_for_counterfactual_advantage",
            "oracle_advantage_semantics": "counterfactual_reward_loss_gnn_minus_loss_refiner_minus_query_cost",
            "selected_budget_source": "valid",
            "selected_budget": float(selected_valid_row["budget"]) if selected_valid_row else None,
            "selected_budget_valid": float(selected_valid_row["budget"]) if selected_valid_row else None,
            "selected_budget_key": best_key,
            "selected_valid_budget_metrics": selected_valid_row,
            "selected_test_budget_metrics_under_valid_choice": selected_test_row,
            "selected_budget_metrics": selected_test_row,
            "overall_delta_vs_base_gnn": selected_test_row and {
                "accuracy": float(selected_test_row["delta_accuracy"]),
                "macro_f1": float(selected_test_row["delta_macro_f1"]),
                "bot_f1": float(selected_test_row["delta_bot_f1"]),
            },
            "wrong_node_fix_rate": float(selected_test_row["wrong_node_fix_rate"]) if selected_test_row else 0.0,
            "correct_node_break_rate": float(selected_test_row["correct_node_break_rate"]) if selected_test_row else 0.0,
            "improved_count": int(selected_test_row["fix"]) if selected_test_row else 0,
            "degraded_count": int(selected_test_row["break"]) if selected_test_row else 0,
            "net_gain": int(selected_test_row["net"]) if selected_test_row else 0,
            "base_wrong_count": base_wrong_count,
            "base_correct_count": base_correct_count,
            "budget_curve": test_budget_rows,
            "budget_curve_legacy_scope": "test_sweep_diagnostic_only",
            "valid_budget_curve": valid_budget_rows,
            "test_budget_curve": test_budget_rows,
            "counterfactual_refiner_fit_summary": refiner_artifact["checkpoint"].get("fit_summary", {}),
            "router_training_summary": router.training_summary,
            "split_scope": {
                "train": int(train_idx.size),
                "valid": int(valid_idx.size),
                "test": int(test_idx.size),
            },
        }
        manifest = {
            "contract": "glance_counterfactual_router_v1",
            "status": "completed",
            "routing_mode": "glance_counterfactual_advantage",
            "paper_faithful_glance_inspired": True,
            "diagnostic_lane_only": True,
            "official_code_verified": False,
            "not_official_reproduction": True,
            "test_labels_used_for_training": False,
            "test_labels_used_for_threshold": False,
            "selected_budget_source": "valid",
            "counterfactual_outcome_used": True,
            "not_a_new_bot_classifier": True,
            "research_positioning": "paper_faithful_glance_inspired_counterfactual_router_for_refiner_utility",
            "no_llm_generation": True,
            "semantic_source": "semantic_finetune_roberta_finetuned",
            "semantic_stage_dir": str(semantic["dir"]),
            "semantic_manifest": semantic["manifest"],
            "frozen_g0_dir": str(context["frozen_g0"]["dir"]),
            "frozen_g0_manifest": context["frozen_g0"]["manifest"],
            "counterfactual_source_refiner": refiner_artifact["stage_name"],
            "counterfactual_source_refiner_manifest": refiner_artifact["manifest"],
            "router_training_target": "1[loss_gnn - loss_refiner - cost > 0]",
            "glance_llm_query_cost": float(getattr(self.args, "glance_llm_query_cost", 0.2)),
            "budgets": [float(item) for item in budgets],
            "selected_budget": float(selected_valid_row["budget"]) if selected_valid_row else None,
            "selected_budget_valid": float(selected_valid_row["budget"]) if selected_valid_row else None,
            "split_audit": split_audit,
            "refiner_architecture": {
                "type": str(refiner_artifact["manifest"].get("refiner_architecture", {}).get("type", "external_refiner")),
                "input": str(refiner_artifact["manifest"].get("refiner_architecture", {}).get("input", "external_refiner_input")),
                "hidden_dim": refiner_artifact["manifest"].get("refiner_architecture", {}).get("hidden_dim"),
                "activation": refiner_artifact["manifest"].get("refiner_architecture", {}).get("activation"),
                "dropout": refiner_artifact["manifest"].get("refiner_architecture", {}).get("dropout"),
                "output_dim": int(logits_gnn.shape[1]),
            },
        }
        self._write_stage_manifest(
            stage_dir,
            manifest,
            artifact_namespace="stages/glance_budgeted_refinement_internal",
            visibility="internal",
            resolved_task="glance_budgeted_refinement_internal",
        )
        write_json(stage_dir / "metrics.json", metrics_payload)
        write_json(stage_dir / "risk_manifest.json", risk_manifest)
        budget_payload_json = {}
        budget_payload_tensors = {}
        for budget_key, payload in test_budget_payloads.items():
            budget_payload_json[budget_key] = {
                "metrics": payload["metrics"],
                "delta_vs_base": payload["delta_vs_base"],
                "routed_node_ids": payload["routed_node_ids"],
            }
            budget_payload_tensors[f"budget_payload_{budget_key}.pt"] = {
                "final_pred": payload["final_pred"],
                "routed_mask": payload["routed_mask"],
                "routed_node_ids": torch.tensor(payload["routed_node_ids"], dtype=torch.long),
            }
        write_json(stage_dir / "budget_payloads.json", budget_payload_json)
        write_csv_rows(
            stage_dir / "budget_curve.csv",
            [
                "budget",
                "routed_count",
                "fix",
                "break",
                "net",
                "routed_fix",
                "routed_break",
                "routed_net",
                "wrong_node_fix_rate",
                "correct_node_break_rate",
                "accuracy",
                "macro_f1",
                "bot_f1",
                "delta_accuracy",
                "delta_macro_f1",
                "delta_bot_f1",
                "mean_base_confidence_routed",
                "mean_refiner_confidence_routed",
            ],
            test_budget_rows,
        )
        write_csv_rows(
            stage_dir / "valid_budget_curve.csv",
            [
                "budget",
                "routed_count",
                "fix",
                "break",
                "net",
                "routed_fix",
                "routed_break",
                "routed_net",
                "wrong_node_fix_rate",
                "correct_node_break_rate",
                "accuracy",
                "macro_f1",
                "bot_f1",
                "delta_accuracy",
                "delta_macro_f1",
                "delta_bot_f1",
                "mean_base_confidence_routed",
                "mean_refiner_confidence_routed",
            ],
            valid_budget_rows,
        )
        write_csv_rows(
            stage_dir / "test_budget_curve.csv",
            [
                "budget",
                "routed_count",
                "fix",
                "break",
                "net",
                "routed_fix",
                "routed_break",
                "routed_net",
                "wrong_node_fix_rate",
                "correct_node_break_rate",
                "accuracy",
                "macro_f1",
                "bot_f1",
                "delta_accuracy",
                "delta_macro_f1",
                "delta_bot_f1",
                "mean_base_confidence_routed",
                "mean_refiner_confidence_routed",
            ],
            test_budget_rows,
        )

        per_node_rows = []
        best_routed_np = best_routed_mask.detach().cpu().numpy().astype(bool)
        best_final_np = best_final_pred.detach().cpu().numpy().astype(np.int64)
        gnn_loss_np = gnn_loss.numpy()
        ref_loss_np = refiner_loss.numpy()
        for node_idx in test_idx.tolist():
            base_correct = bool(pred_gnn[node_idx].item() == int(labels_np[node_idx]))
            ref_correct = bool(pred_ref[node_idx].item() == int(labels_np[node_idx]))
            final_correct = bool(best_final_np[node_idx] == int(labels_np[node_idx]))
            per_node_rows.append({
                "node_id": int(node_idx),
                "base_pred": int(pred_gnn[node_idx].item()),
                "refiner_pred": int(pred_ref[node_idx].item()),
                "final_pred": int(best_final_np[node_idx]),
                "label": int(labels_np[node_idx]),
                "routed": bool(best_routed_np[node_idx]),
                "advantage_score": float(risk_score[node_idx]),
                "oracle_advantage": float(advantage[node_idx]),
                "base_loss": float(gnn_loss_np[node_idx]),
                "refiner_loss": float(ref_loss_np[node_idx]),
                "base_correct": base_correct,
                "refiner_correct": ref_correct,
                "final_correct": final_correct,
                "fix": bool((not base_correct) and final_correct),
                "break": bool(base_correct and (not final_correct)),
                "neighbor_count_1hop": int(semantic_views["count_1hop"][node_idx]),
                "neighbor_count_2hop": int(semantic_views["count_2hop"][node_idx]),
            })
        write_text(
            stage_dir / "per_node_test.jsonl",
            "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in per_node_rows) + "\n",
        )
        save_stage_artifacts(
            stage_dir,
            {
                "outputs.pt": {
                    "base_logits": logits_gnn,
                    "base_prob": p_gnn,
                    "base_pred": pred_gnn,
                    "refiner_logits": logits_ref,
                    "refiner_prob": prob_ref,
                    "refiner_pred": pred_ref,
                    "final_pred": best_final_pred,
                "labels": labels_t,
                "risk_score": torch.tensor(risk_score, dtype=torch.float32),
                "router_score": torch.tensor(risk_score, dtype=torch.float32),
                "selected_routed_mask": best_routed_mask,
                "gnn_loss": gnn_loss,
                "refiner_loss": refiner_loss,
                "oracle_advantage": torch.tensor(advantage, dtype=torch.float32),
                "neighbor_count_1hop": torch.tensor(semantic_views["count_1hop"], dtype=torch.long),
                    "neighbor_count_2hop": torch.tensor(semantic_views["count_2hop"], dtype=torch.long),
                },
                "checkpoint.pt": {
                    "counterfactual_source_refiner": refiner_artifact["stage_name"],
                    "counterfactual_source_refiner_fit_summary": refiner_artifact["checkpoint"].get("fit_summary", {}),
                    "router_state": router.state_dict_payload(),
                },
                "risk_manifest": risk_manifest,
                "valid_budget_curve": valid_budget_rows,
                "test_budget_curve": test_budget_rows,
                "budget_payloads": budget_payload_json,
                **budget_payload_tensors,
                **base_bundle,
            },
        )
        notes = [
            "# GLANCE counterfactual router",
            "",
            "This stage is a diagnostic utility-selector lane, not the main strict GLANCE baseline.",
            "It is a paper-faithful GLANCE-inspired adaptation for offline routed-set analysis, not an official reproduction.",
            "The learned router score is the proxy used for deterministic top-k routing.",
            "The oracle_advantage tensor is the post-hoc counterfactual reward trace: loss_gnn - loss_refiner - query_cost.",
            "The legacy risk_score field is retained as a compatibility alias of router_score.",
            "Budget sweeps are computed on both validation and test, but the final operating budget is selected on validation only and then locked for test reporting.",
            "No generative LLM is queried; semantic views come from cached/fine-tuned RoBERTa embeddings.",
        ]
        write_text(stage_dir / "notes.md", "\n".join(notes) + "\n")
        return {
            "stage": "glance_counterfactual_router",
            "stage_dir": str(stage_dir),
            "metrics": metrics_payload,
        }

    def _run_glance_refiner_analysis(self, stage_dir, base_bundle):
        oracle_metrics = self._require_stage_json("glance_oracle_refine", "metrics")
        oracle_manifest = self._require_stage_json("glance_oracle_refine", "manifest")
        oracle_rows = self._require_stage_jsonl("glance_oracle_refine", "per_node_test")
        full_metrics = self._require_stage_json("glance_full_graph_refine", "metrics")
        full_manifest = self._require_stage_json("glance_full_graph_refine", "manifest")
        full_rows = self._require_stage_jsonl("glance_full_graph_refine", "per_node_test")
        budgeted_metrics = self._read_stage_json("glance_budgeted_refine", "metrics")
        budgeted_manifest = self._read_stage_json("glance_budgeted_refine", "manifest")
        budgeted_rows = self._require_stage_jsonl("glance_budgeted_refine", "per_node_test") if budgeted_metrics and budgeted_manifest else []

        oracle_analysis = self._build_refiner_analysis_summary(oracle_rows, "oracle_wrong_nodes")
        full_analysis = self._build_refiner_analysis_summary(full_rows, "full_graph_supervised_refiner")
        budgeted_analysis = self._build_refiner_analysis_summary(budgeted_rows, "router_routed_fixed_budget_refiner") if budgeted_rows else None
        comparison = {
            "contract": "glance_refiner_analysis_v1",
            "oracle_upper_bound": {
                "manifest": oracle_manifest,
                "metrics": oracle_metrics,
                "analysis": oracle_analysis,
            },
            "full_graph_lower_bound": {
                "manifest": full_manifest,
                "metrics": full_metrics,
                "analysis": full_analysis,
            },
            "budgeted_router_refine": (
                {
                    "manifest": budgeted_manifest,
                    "metrics": budgeted_metrics,
                    "analysis": budgeted_analysis,
                }
                if budgeted_metrics and budgeted_manifest and budgeted_analysis
                else None
            ),
            "comparison_summary": {
                "oracle_wrong_fix_rate": float(oracle_analysis["wrong_node_fix_rate"]),
                "full_graph_wrong_fix_rate": float(full_analysis["wrong_node_fix_rate"]),
                "full_graph_correct_break_rate": float(full_analysis["correct_node_break_rate"]),
                "oracle_vs_full_graph_fix_gap": float(oracle_analysis["wrong_node_fix_rate"] - full_analysis["wrong_node_fix_rate"]),
                "full_graph_net_gain": int(full_metrics.get("net_gain", 0)),
                "budgeted_wrong_fix_rate": float(budgeted_analysis["wrong_node_fix_rate"]) if budgeted_analysis else None,
                "budgeted_correct_break_rate": float(budgeted_analysis["correct_node_break_rate"]) if budgeted_analysis else None,
                "budgeted_net_gain": int(budgeted_analysis["net_gain"]) if budgeted_analysis else None,
                "budgeted_routed_wrong_coverage": float(budgeted_analysis["routed_wrong_coverage"]) if budgeted_analysis else None,
                "budgeted_routed_wrong_precision": float(budgeted_analysis["routed_wrong_precision"]) if budgeted_analysis else None,
                "budgeted_conditional_fix_rate_on_selected_wrong": (
                    float(budgeted_analysis["conditional_fix_rate_on_selected_wrong"]) if budgeted_analysis else None
                ),
                "router_ablation_interpretation": (
                    "oracle upper bound isolates what the correction refiner can do on true hard nodes; "
                    "full-graph lower bound shows the cost of removing node selection; "
                    "budgeted refine under the router-selected validation budget is the deployable operating point."
                ),
            },
        }
        write_json(stage_dir / "analysis_summary.json", comparison)
        notes = [
            "# GLANCE refiner analysis",
            "",
            "This stage compares the error-only oracle upper bound, the no-router full-graph lower bound, and the router-selected budgeted refine operating point when present.",
            "Use the oracle stage to estimate refiner headroom on true hard nodes.",
            "Use the full-graph stage to diagnose where a missing router causes correct nodes to be harmed.",
            "When present, glance_budgeted_refine is interpreted under the router's validation-selected fixed budget, not a test-selected optimum.",
            "The gap between oracle wrong-node fix rate and full-graph wrong-node fix rate is the main signal for router necessity.",
        ]
        write_text(stage_dir / "notes.md", "\n".join(notes) + "\n")
        save_stage_artifacts(
            stage_dir,
            {
                "comparison": comparison,
                **base_bundle,
            },
        )
        return {
            "stage": "glance_refiner_analysis",
            "stage_dir": str(stage_dir),
            "metrics": comparison["comparison_summary"],
        }

    def _run_glance_budgeted_refine(self, stage_dir, base_bundle):
        context = self.ensure_backbone_context()
        semantic = self._load_semantic_finetune_artifact()
        router_artifact = self._load_glance_counterfactual_router_artifact()
        gnn_outputs = context["gnn_outputs"]
        p_gnn = gnn_outputs.get("prob")
        logits_gnn = gnn_outputs.get("logits")
        pred_gnn = gnn_outputs.get("pred")
        z_gnn = gnn_outputs.get("node_repr")
        if any(item is None for item in (p_gnn, logits_gnn, pred_gnn, z_gnn)):
            raise MissingFrozenArtifactError(
                "glance_budgeted_refine requires frozen_g0 outputs with prob/logits/pred/node_repr."
            )

        p_gnn = p_gnn.detach().cpu().float()
        logits_gnn = logits_gnn.detach().cpu().float()
        pred_gnn = pred_gnn.detach().cpu().long()
        z_gnn = z_gnn.detach().cpu().float()
        labels_t = torch.tensor(self.labels, dtype=torch.long)
        labels_np = labels_t.numpy()
        train_idx = _idx_numpy(self.data["train_idx"])
        valid_idx = _idx_numpy(self.data["valid_idx"])
        test_idx = _idx_numpy(self.data["test_idx"])

        semantic_views = self._build_k_hop_semantic_views(
            semantic["embeddings"],
            self.data.get("edge_index"),
        )
        refiner_features = torch.cat(
            [z_gnn, semantic_views["ego"], semantic_views["hop1"], semantic_views["hop2"]],
            dim=1,
        )

        risk_scores = router_artifact["router_scores"].detach().cpu().numpy().astype(np.float32)
        budgets = self._router_budgets()
        train_order = train_idx[np.argsort(-risk_scores[train_idx])] if train_idx.size else np.asarray([], dtype=np.int64)
        valid_order = valid_idx[np.argsort(-risk_scores[valid_idx])] if valid_idx.size else np.asarray([], dtype=np.int64)
        test_order = test_idx[np.argsort(-risk_scores[test_idx])] if test_idx.size else np.asarray([], dtype=np.int64)
        base_test = _score_all(labels_np[test_idx], pred_gnn.numpy()[test_idx]) if test_idx.size else {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "bot_f1": 0.0,
            "count": 0,
        }
        base_wrong_count = int((pred_gnn.numpy()[test_idx] != labels_np[test_idx]).sum()) if test_idx.size else 0
        base_correct_count = int(test_idx.size - base_wrong_count)

        budget_rows = []
        budget_payload_json = {}
        analysis_by_budget = {}
        candidate_outputs_by_key = {}
        router_selected_budget = router_artifact["metrics"].get(
            "selected_budget_valid",
            router_artifact["metrics"].get("selected_budget"),
        )
        if router_selected_budget is None:
            raise MissingFrozenArtifactError(
                "glance_budgeted_refine requires glance_counterfactual_router to persist a validation-selected budget."
            )
        router_selected_key = str(self._budget_key(router_selected_budget))

        for budget in budgets:
            train_k = min(max(int(train_idx.size * float(budget)), 1), train_idx.size) if train_idx.size else 0
            valid_k = min(max(int(valid_idx.size * float(budget)), 1), valid_idx.size) if valid_idx.size else 0
            test_k = min(max(int(test_idx.size * float(budget)), 1), test_idx.size) if test_idx.size else 0
            train_routed = train_order[:train_k]
            valid_routed = valid_order[:valid_k]
            test_routed = test_order[:test_k]
            if train_routed.size == 0:
                continue

            model = GlanceRefinerMLP(
                input_dim=int(refiner_features.shape[1]),
                hidden_dim=128,
                activation=str(getattr(self.args, "activation", "leakyrelu")).lower(),
                dropout=0.1,
            )
            model, fit_summary = self._train_glance_refiner(
                model,
                refiner_features[train_routed],
                labels_t[train_routed],
                refiner_features[valid_routed],
                labels_t[valid_routed],
            )
            model.eval()
            with torch.no_grad():
                train_logits_ref = model(refiner_features[train_routed]).cpu() if train_routed.size else torch.empty((0, 2), dtype=torch.float32)
                valid_logits_ref = model(refiner_features[valid_routed]).cpu() if valid_routed.size else torch.empty((0, 2), dtype=torch.float32)
                test_logits_ref = model(refiner_features[test_routed]).cpu() if test_routed.size else torch.empty((0, 2), dtype=torch.float32)

            final_logits = logits_gnn.clone()
            final_prob = p_gnn.clone()
            final_pred = pred_gnn.clone()
            if train_routed.size:
                final_logits[train_routed] = train_logits_ref
                final_prob[train_routed] = torch.softmax(train_logits_ref, dim=1)
                final_pred[train_routed] = train_logits_ref.argmax(dim=1)
            if valid_routed.size:
                final_logits[valid_routed] = valid_logits_ref
                final_prob[valid_routed] = torch.softmax(valid_logits_ref, dim=1)
                final_pred[valid_routed] = valid_logits_ref.argmax(dim=1)
            if test_routed.size:
                final_logits[test_routed] = test_logits_ref
                final_prob[test_routed] = torch.softmax(test_logits_ref, dim=1)
                final_pred[test_routed] = test_logits_ref.argmax(dim=1)

            test_mask = np.zeros(labels_np.shape[0], dtype=bool)
            test_mask[test_idx] = True
            routed_test_mask = np.zeros(labels_np.shape[0], dtype=bool)
            routed_test_mask[test_routed] = True
            delta = _delta_table(pred_gnn.numpy(), final_pred.numpy(), labels_np, test_mask)
            routed_delta = _delta_table(pred_gnn.numpy(), final_pred.numpy(), labels_np, routed_test_mask)
            overall_test = _score_all(labels_np[test_idx], final_pred.numpy()[test_idx]) if test_idx.size else dict(base_test)
            row = {
                "budget": float(budget),
                "train_routed_count": int(train_routed.size),
                "valid_routed_count": int(valid_routed.size),
                "test_routed_count": int(test_routed.size),
                "fix": int(delta["fix"]),
                "break": int(delta["broke"]),
                "net": int(delta["net"]),
                "routed_fix": int(routed_delta["fix"]),
                "routed_break": int(routed_delta["broke"]),
                "routed_net": int(routed_delta["net"]),
                "wrong_node_fix_rate": float(delta["fix"] / base_wrong_count) if base_wrong_count else 0.0,
                "correct_node_break_rate": float(delta["broke"] / base_correct_count) if base_correct_count else 0.0,
                "accuracy": float(overall_test["accuracy"]),
                "macro_f1": float(overall_test["macro_f1"]),
                "bot_f1": float(overall_test["bot_f1"]),
                "delta_accuracy": float(overall_test["accuracy"] - base_test["accuracy"]),
                "delta_macro_f1": float(overall_test["macro_f1"] - base_test["macro_f1"]),
                "delta_bot_f1": float(overall_test["bot_f1"] - base_test["bot_f1"]),
            }
            budget_rows.append(row)
            budget_key = str(self._budget_key(budget))
            budget_payload_json[budget_key] = {
                "metrics": overall_test,
                "delta_vs_base": {
                    "accuracy": row["delta_accuracy"],
                    "macro_f1": row["delta_macro_f1"],
                    "bot_f1": row["delta_bot_f1"],
                },
                "routed_node_ids_train": [int(item) for item in train_routed.tolist()],
                "routed_node_ids_valid": [int(item) for item in valid_routed.tolist()],
                "routed_node_ids_test": [int(item) for item in test_routed.tolist()],
            }
            per_node_rows = []
            test_routed_np = routed_test_mask
            for node_idx in test_idx.tolist():
                per_node_rows.append({
                    "node_id": int(node_idx),
                    "base_pred": int(pred_gnn[node_idx].item()),
                    "final_pred": int(final_pred[node_idx].item()),
                    "label": int(labels_np[node_idx]),
                    "routed": bool(test_routed_np[node_idx]),
                    "was_wrong_base": bool(pred_gnn[node_idx].item() != labels_np[node_idx]),
                    "is_wrong_final": bool(final_pred[node_idx].item() != labels_np[node_idx]),
                    "neighbor_count_1hop": int(semantic_views["count_1hop"][node_idx]),
                    "neighbor_count_2hop": int(semantic_views["count_2hop"][node_idx]),
                })
            analysis_by_budget[budget_key] = self._build_refiner_analysis_summary(per_node_rows, f"budget_{budget_key}")
            candidate_outputs_by_key[budget_key] = {
                "budget": float(budget),
                "logits": final_logits,
                "prob": final_prob,
                "pred": final_pred,
                "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "fit_summary": fit_summary,
                "per_node_rows": per_node_rows,
                "analysis_summary": analysis_by_budget[budget_key],
                "row": row,
                "routed_masks": {
                    "train": torch.zeros(labels_t.numel(), dtype=torch.bool).scatter_(0, torch.tensor(train_routed, dtype=torch.long), True) if train_routed.size else torch.zeros(labels_t.numel(), dtype=torch.bool),
                    "valid": torch.zeros(labels_t.numel(), dtype=torch.bool).scatter_(0, torch.tensor(valid_routed, dtype=torch.long), True) if valid_routed.size else torch.zeros(labels_t.numel(), dtype=torch.bool),
                    "test": torch.tensor(test_routed_np, dtype=torch.bool),
                },
            }

        if not budget_rows:
            raise MissingFrozenArtifactError("glance_budgeted_refine could not build any budgeted routed subset.")

        if router_selected_key not in candidate_outputs_by_key:
            raise MissingFrozenArtifactError(
                "glance_budgeted_refine could not match the router-selected validation budget "
                f"{router_selected_budget} to a local budgeted refiner run."
            )
        selected_budget = float(router_selected_budget)
        selected_key = router_selected_key
        selected_outputs = candidate_outputs_by_key[selected_key]
        metrics_payload = {
            "contract": "glance_budgeted_refine_metrics_v1",
            "routing_mode": "router_routed_fixed_budget_refiner",
            "paper_faithful_glance_inspired": True,
            "not_deployable": False,
            "counterfactual_source_refiner": router_artifact["manifest"].get("counterfactual_source_refiner"),
            "selected_budget_source": "valid",
            "budget_source_stage": "glance_counterfactual_router",
            "selected_budget": float(selected_budget),
            "selected_budget_valid": float(selected_budget),
            "selected_budget_key": selected_key,
            "selected_budget_metrics": selected_outputs["row"],
            "selected_test_budget_metrics_under_valid_choice": selected_outputs["row"],
            "base_test": base_test,
            "overall_delta_vs_base_gnn": {
                "accuracy": float(selected_outputs["row"]["delta_accuracy"]),
                "macro_f1": float(selected_outputs["row"]["delta_macro_f1"]),
                "bot_f1": float(selected_outputs["row"]["delta_bot_f1"]),
            },
            "wrong_node_fix_rate": float(selected_outputs["row"]["wrong_node_fix_rate"]),
            "correct_node_break_rate": float(selected_outputs["row"]["correct_node_break_rate"]),
            "improved_count": int(selected_outputs["row"]["fix"]),
            "degraded_count": int(selected_outputs["row"]["break"]),
            "net_gain": int(selected_outputs["row"]["net"]),
            "base_wrong_count": base_wrong_count,
            "base_correct_count": base_correct_count,
            "budget_curve": budget_rows,
            "budget_curve_scope": "test_sweep_diagnostic_only",
            "fit_summary": selected_outputs["fit_summary"],
            "split_scope": {
                "train": int(train_idx.size),
                "valid": int(valid_idx.size),
                "test": int(test_idx.size),
            },
        }
        manifest = {
            "contract": "glance_budgeted_refine_v1",
            "status": "completed",
            "routing_mode": "router_routed_fixed_budget_refiner",
            "paper_faithful_glance_inspired": True,
            "diagnostic_lane_only": True,
            "not_deployable": False,
            "research_positioning": "selector_gated_correction_refiner_under_fixed_budget",
            "counterfactual_source_refiner": router_artifact["manifest"].get("counterfactual_source_refiner"),
            "router_manifest": router_artifact["manifest"],
            "budget_source_stage": "glance_counterfactual_router",
            "selected_budget_source": "valid",
            "semantic_manifest": semantic["manifest"],
            "frozen_g0_manifest": context["frozen_g0"]["manifest"],
            "selected_budget": float(selected_budget),
            "selected_budget_valid": float(selected_budget),
            "budgets": [float(item) for item in budgets],
            "refiner_architecture": {
                "type": "glance_budgeted_refine_mlp",
                "input": "[z_gnn || z_sem_0 || z_sem_1 || z_sem_2]",
                "hidden_dim": 128,
                "activation": str(getattr(self.args, "activation", "leakyrelu")).lower(),
                "dropout": 0.1,
                "output_dim": int(logits_gnn.shape[1]),
            },
        }
        self._write_stage_manifest(
            stage_dir,
            manifest,
            artifact_namespace="stages/joint_router_refinement",
            visibility="public",
            resolved_task="joint_router_refinement",
        )
        write_json(stage_dir / "metrics.json", metrics_payload)
        write_json(stage_dir / "budget_payloads.json", budget_payload_json)
        write_json(stage_dir / "analysis_summary.json", selected_outputs["analysis_summary"])
        write_csv_rows(
            stage_dir / "budget_curve.csv",
            [
                "budget",
                "train_routed_count",
                "valid_routed_count",
                "test_routed_count",
                "fix",
                "break",
                "net",
                "routed_fix",
                "routed_break",
                "routed_net",
                "wrong_node_fix_rate",
                "correct_node_break_rate",
                "accuracy",
                "macro_f1",
                "bot_f1",
                "delta_accuracy",
                "delta_macro_f1",
                "delta_bot_f1",
            ],
            budget_rows,
        )
        write_text(
            stage_dir / "per_node_test.jsonl",
            "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in selected_outputs["per_node_rows"]) + "\n",
        )
        save_stage_artifacts(
            stage_dir,
            {
                "outputs.pt": {
                    "logits": selected_outputs["logits"],
                    "prob": selected_outputs["prob"],
                    "pred": selected_outputs["pred"],
                    "labels": labels_t,
                    "base_pred": pred_gnn,
                    "routed_masks": selected_outputs["routed_masks"],
                    "neighbor_count_1hop": torch.tensor(semantic_views["count_1hop"], dtype=torch.long),
                    "neighbor_count_2hop": torch.tensor(semantic_views["count_2hop"], dtype=torch.long),
                },
                "checkpoint.pt": {
                    "model": selected_outputs["model_state"],
                    "fit_summary": selected_outputs["fit_summary"],
                },
                "router_budget_payloads": budget_payload_json,
                **base_bundle,
            },
        )
        notes = [
            "# GLANCE budgeted refine",
            "",
            "This stage is a diagnostic routed-set correction lane, not the main strict GLANCE baseline.",
            "It consumes routed node selections from glance_counterfactual_router and trains a routed-only correction refiner.",
            "For each budget, the refiner is trained on train_routed(B), selected on valid_routed(B), and applied only to test_routed(B).",
            "The final artifact is locked to the router's validation-selected budget; the per-budget curve is diagnostic only.",
            "Non-routed nodes strictly preserve the base GNN prediction.",
        ]
        write_text(stage_dir / "notes.md", "\n".join(notes) + "\n")
        return {
            "stage": "glance_budgeted_refine",
            "stage_dir": str(stage_dir),
            "metrics": metrics_payload,
        }


    def _run_glance_joint_router_refine(self, stage_dir, base_bundle):
        context = self.ensure_backbone_context()
        semantic_bundle = self._load_glance_semantic_source_bundle()
        semantic = semantic_bundle["primary"]
        gnn_outputs = context["gnn_outputs"]
        p_gnn = gnn_outputs.get("prob")
        logits_gnn = gnn_outputs.get("logits")
        pred_gnn = gnn_outputs.get("pred")
        z_gnn = gnn_outputs.get("node_repr")
        if any(item is None for item in (p_gnn, logits_gnn, pred_gnn, z_gnn)):
            raise MissingFrozenArtifactError(
                "glance_joint_router_refine requires frozen_g0 outputs with prob/logits/pred/node_repr."
            )

        p_gnn = p_gnn.detach().cpu().float()
        logits_gnn = logits_gnn.detach().cpu().float()
        pred_gnn = pred_gnn.detach().cpu().long()
        z_gnn = z_gnn.detach().cpu().float()
        labels_t = torch.tensor(self.labels, dtype=torch.long)
        labels_np = labels_t.numpy()
        train_idx = _idx_numpy(self.data["train_idx"])
        valid_idx = _idx_numpy(self.data["valid_idx"])
        test_idx = _idx_numpy(self.data["test_idx"])
        if train_idx.size == 0 or valid_idx.size == 0 or test_idx.size == 0:
            raise MissingFrozenArtifactError(
                "glance_joint_router_refine requires non-empty train/valid/test splits."
            )

        configured_train_cap = int(getattr(self.args, "joint_train_node_cap", 3000))
        strict_train_idx = self._strict_glance_train_idx(train_idx, cap=configured_train_cap)
        train_cap_mode = "full_train_split" if configured_train_cap <= 0 else "capped_train_subset"
        semantic_view_mode = semantic.get("semantic_view_mode", semantic["manifest"].get("semantic_view_mode", "legacy_inbound_khop"))
        if semantic_view_mode == "prompt_expert_bundle_v1":
            refiner_features, semantic_views = self._build_prompt_expert_semantic_views(
                semantic.get("prompt_expert_bundle"),
                z_gnn,
            )
            prompt_expert_active_components = list(refiner_features.get("active_components", []))
        else:
            semantic_views = self._build_k_hop_semantic_views(
                semantic.get("precomputed_views") if semantic.get("precomputed_views") is not None else semantic["embeddings"],
                self.data.get("edge_index"),
            )
            refiner_features = torch.cat(
                [z_gnn, semantic_views["ego"], semantic_views["hop1"], semantic_views["hop2"]],
                dim=1,
            ).detach().cpu().float()
            prompt_expert_active_components = []
        refiner_explicit_gate = bool(getattr(self.args, "joint_refiner_explicit_gate", False))
        refiner_target_mode = str(getattr(self.args, "joint_refiner_target_mode", "predict")).lower()
        refiner_weight_mode = str(getattr(self.args, "joint_refiner_weight_mode", "off")).lower()
        refiner_base_wrong_weight = float(getattr(self.args, "joint_refiner_base_wrong_weight", 2.0))
        refiner_utility_weight = float(getattr(self.args, "joint_refiner_utility_weight", 3.0))
        refiner_gate_weight = float(getattr(self.args, "joint_refiner_gate_weight", 0.5))

        original_node_features_bundle = self._strict_glance_original_node_features()
        q_bundle = self._fit_strict_glance_q_probs(
            original_node_features_bundle["features"],
            strict_train_idx,
            valid_idx,
        )
        backbone_input_features = self._strict_glance_backbone_input_features()
        mc_bundle = self._strict_glance_mc_dropout_uncertainty(backbone_input_features)
        router_temperature_bundle = fit_reliability_temperature(
            logits_gnn[valid_idx],
            labels_np[valid_idx],
        )
        router_feature_bundle = self._build_strict_glance_router_features(
            logits_gnn=logits_gnn,
            p_gnn=p_gnn,
            z_gnn=z_gnn,
            original_node_features=original_node_features_bundle["features"],
            q_bundle=q_bundle,
            mc_bundle=mc_bundle,
            calibration_temperature=router_temperature_bundle["temperature"],
        )
        router_feature_bundle["full_feature_bundle"]["temperature_scaling"] = dict(router_temperature_bundle)
        router_features, scaler_state = self._standardize_glance_router_features(
            router_feature_bundle["features"],
            strict_train_idx,
        )

        activation = str(getattr(self.args, "activation", "leakyrelu")).lower()
        batch_size = 32
        max_epochs = 10
        patience = 2
        decay_factor = 0.5
        entropy_weight = 0.0
        router_weight = 0.0
        learning_rate = float(getattr(self.args, "lr_GNN", 5e-4))
        weight_decay = float(getattr(self.args, "weight_decay_GNN", 1e-5))
        eval_top_k = max(int(round(batch_size / 4.0)), 1)
        beta_candidates = (0.1, 0.2, 0.3)

        beta_runs = []
        selected_beta_run = None
        selected_beta_score = None
        for beta in beta_candidates:
            run = self._train_glance_joint_candidate(
                labels_t=labels_t,
                labels_np=labels_np,
                train_idx=strict_train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
                logits_gnn=logits_gnn,
                p_gnn=p_gnn,
                pred_gnn=pred_gnn,
                router_features=router_features,
                refiner_features=refiner_features,
                semantic_views=semantic_views,
                semantic_view_mode=semantic_view_mode,
                activation=activation,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                decay_factor=decay_factor,
                entropy_weight=entropy_weight,
                router_weight=router_weight,
                router_regression_weight=0.0,
                router_ranking_weight=1.0,
                router_calibration_weight=0.25,
                router_reliability_weight=0.0,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                beta=float(beta),
                eval_top_k=eval_top_k,
                refiner_explicit_gate=refiner_explicit_gate,
                refiner_target_mode=refiner_target_mode,
                refiner_weight_mode=refiner_weight_mode,
                refiner_base_wrong_weight=refiner_base_wrong_weight,
                refiner_utility_weight=refiner_utility_weight,
                refiner_gate_weight=refiner_gate_weight,
            )
            beta_runs.append(run)
            beta_score = (
                float(run["selected_budget_metrics"]["valid"]["macro_f1"]),
                -float(run["selected_budget_metrics"]["valid"]["loss"]),
            )
            if selected_beta_score is None or beta_score > selected_beta_score:
                selected_beta_score = beta_score
                selected_beta_run = run

        if selected_beta_run is None:
            raise MissingFrozenArtifactError("glance_joint_router_refine could not select a beta on validation.")

        final_run = selected_beta_run
        selected_beta = float(final_run["beta"])
        selected_budget = float(final_run["selected_budget"])
        selected_budget_key = str(final_run["selected_budget_key"])
        train_outputs = final_run["train_outputs"]
        valid_outputs = final_run["valid_outputs"]
        test_outputs = final_run["test_outputs"]
        final_logits = logits_gnn.clone()
        final_prob = p_gnn.clone()
        final_pred = pred_gnn.clone()
        routed_masks = {
            "train": train_outputs["routed_mask"],
            "valid": valid_outputs["routed_mask"],
            "test": test_outputs["routed_mask"],
        }
        router_prob = torch.zeros(pred_gnn.numel(), dtype=torch.float32)
        router_score = torch.zeros(pred_gnn.numel(), dtype=torch.float32)
        oracle_advantage = torch.zeros(pred_gnn.numel(), dtype=torch.float32)
        for split_outputs, split_idx in (
            (train_outputs, train_idx),
            (valid_outputs, valid_idx),
            (test_outputs, test_idx),
        ):
            split_mask = split_outputs["routed_mask"]
            split_idx_t = torch.tensor(split_idx, dtype=torch.long)
            final_logits[split_mask] = split_outputs["logits"][split_mask]
            final_prob[split_mask] = split_outputs["prob"][split_mask]
            final_pred[split_mask] = split_outputs["pred"][split_mask]
            router_prob[split_idx_t] = split_outputs["router_prob"][split_idx_t]
            router_score[split_idx_t] = split_outputs["router_score"][split_idx_t]
            oracle_advantage[split_idx_t] = split_outputs["oracle_advantage"][split_idx_t]

        base_test = _score_all(labels_np[test_idx], pred_gnn.numpy()[test_idx])
        overall_test = _score_all(labels_np[test_idx], final_pred.numpy()[test_idx])
        test_delta = _delta_table(pred_gnn.numpy(), final_pred.numpy(), labels_np, self.test_mask)
        per_node_rows = final_run["test_rows"]["rows"]
        analysis_summary = final_run["test_rows"]["analysis"]

        beta_sweep = []
        for run in beta_runs:
            beta_sweep.append({
                "beta": float(run["beta"]),
                "eval_top_k": int(run["eval_top_k"]),
                "best_epoch": int(run["best_epoch"]),
                "selected_budget": float(run["selected_budget"]),
                "selected_budget_key": str(run["selected_budget_key"]),
                "valid_accuracy": float(run["selected_budget_metrics"]["valid"]["accuracy"]),
                "valid_macro_f1": float(run["selected_budget_metrics"]["valid"]["macro_f1"]),
                "valid_bot_f1": float(run["selected_budget_metrics"]["valid"]["bot_f1"]),
                "valid_loss": float(run["selected_budget_metrics"]["valid"]["loss"]),
                "valid_routed_count": int(run["selected_budget_metrics"]["valid"]["routed_count"]),
                "valid_query_rate": float(run["selected_budget_metrics"]["valid"]["query_rate"]),
                "valid_net_gain": int(run["valid_rows"]["analysis"]["net_gain"]),
                "valid_routed_wrong_precision": float(run["valid_rows"]["analysis"]["routed_wrong_precision"]),
                "valid_routed_wrong_coverage": float(run["valid_rows"]["analysis"]["routed_wrong_coverage"]),
                "valid_conditional_fix_rate": float(run["valid_rows"]["analysis"]["conditional_fix_rate_on_selected_wrong"]),
                "test_accuracy": float(run["selected_budget_metrics"]["test"]["accuracy"]),
                "test_macro_f1": float(run["selected_budget_metrics"]["test"]["macro_f1"]),
                "test_bot_f1": float(run["selected_budget_metrics"]["test"]["bot_f1"]),
                "test_loss": float(run["selected_budget_metrics"]["test"]["loss"]),
                "test_routed_count": int(run["selected_budget_metrics"]["test"]["routed_count"]),
                "test_query_rate": float(run["selected_budget_metrics"]["test"]["query_rate"]),
                "test_net_gain": int(run["test_rows"]["analysis"]["net_gain"]),
                "test_routed_wrong_precision": float(run["test_rows"]["analysis"]["routed_wrong_precision"]),
                "test_routed_wrong_coverage": float(run["test_rows"]["analysis"]["routed_wrong_coverage"]),
                "test_conditional_fix_rate": float(run["test_rows"]["analysis"]["conditional_fix_rate_on_selected_wrong"]),
            })

        valid_budget_curve = final_run["valid_budget_curve"]
        test_budget_curve = final_run["test_budget_curve"]
        if getattr(self.args, "joint_refiner_embedding_path", None):
            dependency_alignment_note = (
                "joint_router_refinement keeps backbone provenance pinned to the current run's "
                "preparation/graph_detector artifact while allowing a refiner-only semantic override through "
                "--joint_refiner_embedding_path."
            )
        else:
            dependency_alignment_note = (
                "Public joint_router_refinement is bound to the current run's preparation/graph_detector artifact "
                "and reuses its recorded feature_manifest.path as the semantic tensor source."
            )

        metrics_payload = {
            "contract": "glance_joint_router_refine_metrics_v1",
            "routing_mode": "joint_topk_train_global_budget_eval_router_refiner",
            "paper_faithful_glance_inspired": False,
            "glance_style_joint_training": True,
            "strict_training_protocol_alignment": bool(configured_train_cap == 3000),
            "paper_text_router_alignment": False,
            "twibot20_adapted_training": bool(configured_train_cap <= 0),
            "strict_dependency_provenance_alignment": True,
            "semantic_alignment": False,
            "not_official_reproduction": True,
            "paper_text_evaluation_alignment": False,
            "evaluation_protocol": "validation_selected_global_budget_by_router_score",
            "evaluation_alignment_note": "Training keeps paper-text batch top-k routing; final public evaluation selects a global budget on validation and locks that budget on test for TwiBot20-suitable score-based routing.",
            "dependency_alignment_note": dependency_alignment_note,
            "semantic_source_mode": semantic_bundle["mode"],
            "semantic_sources": [item["manifest"].get("source_identity", item["manifest"].get("lm_model")) for item in semantic_bundle["sources"]],
            "primary_semantic_manifest": semantic_bundle["primary"]["manifest"],
            "joint_refiner_embedding_path": str(getattr(self.args, "joint_refiner_embedding_path", None) or ""),
            "semantic_view_mode": semantic_view_mode,
            "external_frozen_g0_root": str(getattr(self.args, "external_frozen_g0_root", None) or ""),
            "selected_beta_source": "validation",
            "selected_beta": selected_beta,
            "beta_candidates": [float(item) for item in beta_candidates],
            "selected_budget_source": "validation",
            "selected_budget": selected_budget,
            "selected_budget_key": selected_budget_key,
            "selected_valid_budget_metrics": final_run["selected_valid_budget_metrics"],
            "selected_test_budget_metrics_under_valid_choice": final_run["selected_test_budget_metrics_under_valid_choice"],
            "selected_beta_valid_metrics": final_run["selected_budget_metrics"]["valid"],
            "selected_beta_test_metrics": final_run["selected_budget_metrics"]["test"],
            "beta_sweep": beta_sweep,
            "valid_budget_curve": valid_budget_curve,
            "test_budget_curve": test_budget_curve,
            "router_training_objective": "base_wrong_reliability_bce_plus_pairwise_ranking",
            "oracle_advantage_semantics": "loss_gnn_minus_loss_refiner_minus_beta",
            "router_score_semantics": "learned_base_wrong_reliability_score_for_global_budget_routing",
            "router_reliability_semantics": "primary_base_wrong_probability_estimator",
            "router_feature_family": router_feature_bundle["full_feature_bundle"]["feature_family"],
            "router_feature_names": list(router_feature_bundle["feature_names"]),
            "router_temperature_scaling": dict(router_temperature_bundle),
            "refiner_target_mode": refiner_target_mode,
            "refiner_weight_mode": refiner_weight_mode,
            "refiner_explicit_gate": bool(refiner_explicit_gate),
            "refiner_base_wrong_weight": float(refiner_base_wrong_weight),
            "refiner_utility_weight": float(refiner_utility_weight),
            "refiner_gate_weight": float(refiner_gate_weight),
            "advantage_router_diagnostics": final_run["fit_summary"].get("advantage_router_diagnostics", {}),
            "base_test": base_test,
            "overall_test": overall_test,
            "overall_delta_vs_base_gnn": {
                "accuracy": float(overall_test["accuracy"] - base_test["accuracy"]),
                "macro_f1": float(overall_test["macro_f1"] - base_test["macro_f1"]),
                "bot_f1": float(overall_test["bot_f1"] - base_test["bot_f1"]),
            },
            "improved_count": int(test_delta["fix"]),
            "degraded_count": int(test_delta["broke"]),
            "net_gain": int(test_delta["net"]),
            "wrong_node_fix_rate": float(test_delta["fix"] / max(int((pred_gnn.numpy()[test_idx] != labels_np[test_idx]).sum()), 1)),
            "correct_node_break_rate": float(
                test_delta["broke"]
                / max(int(test_idx.size - int((pred_gnn.numpy()[test_idx] != labels_np[test_idx]).sum())), 1)
            ),
            "query_usage": {
                "train_routed_count": int(train_outputs["routed_count"]),
                "valid_routed_count": int(valid_outputs["routed_count"]),
                "test_routed_count": int(test_outputs["routed_count"]),
                "selected_budget": selected_budget,
                "eval_top_k": int(eval_top_k),
                "batch_size": int(batch_size),
            },
            "fit_summary": final_run["fit_summary"],
        }
        router_performance_summary = {
            "selected_beta": float(selected_beta),
            "selected_budget": float(selected_budget),
            "selected_budget_key": str(selected_budget_key),
            "router_temperature": float(router_temperature_bundle["temperature"]),
            "router_temperature_bundle": dict(router_temperature_bundle),
            "router_feature_family": str(router_feature_bundle["full_feature_bundle"]["feature_family"]),
            "router_input_dim": int(router_features.shape[1]),
            "router_diagnostics": final_run["fit_summary"].get("router_diagnostics", {}),
            "advantage_router_diagnostics": final_run["fit_summary"].get("advantage_router_diagnostics", {}),
            "topk_reference_router_diagnostics": final_run["fit_summary"].get("topk_reference", {}).get("router_diagnostics", {}),
            "topk_reference_advantage_router_diagnostics": final_run["fit_summary"].get("topk_reference", {}).get("advantage_router_diagnostics", {}),
            "selected_valid_budget_metrics": final_run["selected_budget_metrics"]["valid"],
            "selected_test_budget_metrics": final_run["selected_budget_metrics"]["test"],
            "selected_valid_router_analysis": final_run["valid_rows"]["analysis"],
            "selected_test_router_analysis": final_run["test_rows"]["analysis"],
            "overall_delta_vs_base_gnn": metrics_payload["overall_delta_vs_base_gnn"],
            "wrong_node_fix_rate": metrics_payload["wrong_node_fix_rate"],
            "correct_node_break_rate": metrics_payload["correct_node_break_rate"],
            "net_gain": metrics_payload["net_gain"],
            "best_epoch": int(final_run["best_epoch"]),
            "query_usage": metrics_payload["query_usage"],
        }
        router_performance_row = {
            "selected_beta": float(selected_beta),
            "selected_budget": float(selected_budget),
            "best_epoch": int(final_run["best_epoch"]),
            "router_temperature": float(router_temperature_bundle["temperature"]),
            "router_valid_positive_rate": final_run["fit_summary"]["router_diagnostics"]["valid"].get("positive_rate"),
            "router_valid_mean_score": final_run["fit_summary"]["router_diagnostics"]["valid"].get("mean_score"),
            "router_valid_auroc": final_run["fit_summary"]["router_diagnostics"]["valid"].get("auroc"),
            "router_valid_auprc": final_run["fit_summary"]["router_diagnostics"]["valid"].get("auprc"),
            "router_test_positive_rate": final_run["fit_summary"]["router_diagnostics"]["test"].get("positive_rate"),
            "router_test_mean_score": final_run["fit_summary"]["router_diagnostics"]["test"].get("mean_score"),
            "router_test_auroc": final_run["fit_summary"]["router_diagnostics"]["test"].get("auroc"),
            "router_test_auprc": final_run["fit_summary"]["router_diagnostics"]["test"].get("auprc"),
            "router_valid_adv_auroc": final_run["fit_summary"]["advantage_router_diagnostics"]["valid"].get("auroc"),
            "router_valid_adv_auprc": final_run["fit_summary"]["advantage_router_diagnostics"]["valid"].get("auprc"),
            "router_test_adv_auroc": final_run["fit_summary"]["advantage_router_diagnostics"]["test"].get("auroc"),
            "router_test_adv_auprc": final_run["fit_summary"]["advantage_router_diagnostics"]["test"].get("auprc"),
            "valid_query_rate": float(final_run["selected_budget_metrics"]["valid"]["query_rate"]),
            "valid_routed_count": int(final_run["selected_budget_metrics"]["valid"]["routed_count"]),
            "valid_routed_wrong_precision": float(final_run["valid_rows"]["analysis"]["routed_wrong_precision"]),
            "valid_routed_wrong_coverage": float(final_run["valid_rows"]["analysis"]["routed_wrong_coverage"]),
            "valid_conditional_fix_rate": float(final_run["valid_rows"]["analysis"]["conditional_fix_rate_on_selected_wrong"]),
            "test_query_rate": float(final_run["selected_budget_metrics"]["test"]["query_rate"]),
            "test_routed_count": int(final_run["selected_budget_metrics"]["test"]["routed_count"]),
            "test_routed_wrong_precision": float(final_run["test_rows"]["analysis"]["routed_wrong_precision"]),
            "test_routed_wrong_coverage": float(final_run["test_rows"]["analysis"]["routed_wrong_coverage"]),
            "test_conditional_fix_rate": float(final_run["test_rows"]["analysis"]["conditional_fix_rate_on_selected_wrong"]),
            "test_wrong_node_fix_rate": float(metrics_payload["wrong_node_fix_rate"]),
            "test_correct_node_break_rate": float(metrics_payload["correct_node_break_rate"]),
            "test_net_gain": int(metrics_payload["net_gain"]),
            "test_macro_f1": float(metrics_payload["overall_test"]["macro_f1"]),
            "test_delta_macro_f1": float(metrics_payload["overall_delta_vs_base_gnn"]["macro_f1"]),
        }
        manifest = {
            "contract": "glance_joint_router_refine_v1",
            "status": "completed",
            "routing_mode": "joint_topk_train_global_budget_eval_router_refiner",
            "paper_faithful_glance_inspired": False,
            "glance_style_joint_training": True,
            "strict_training_protocol_alignment": bool(configured_train_cap == 3000),
            "paper_text_router_alignment": False,
            "twibot20_adapted_training": bool(configured_train_cap <= 0),
            "strict_dependency_provenance_alignment": True,
            "semantic_alignment": False,
            "not_official_reproduction": True,
            "paper_text_evaluation_alignment": False,
            "evaluation_protocol": "validation_selected_global_budget_by_router_score",
            "evaluation_alignment_note": "Batch top-k is kept only inside strict GLANCE training. Public final evaluation uses a validation-selected global budget and test-locked router ranking.",
            "research_positioning": "task_adapted_reliability_router_under_glance_style_joint_training",
            "selected_beta_source": "validation",
            "selected_beta": selected_beta,
            "beta_candidates": [float(item) for item in beta_candidates],
            "selected_budget_source": "validation",
            "selected_budget": selected_budget,
            "selected_budget_key": selected_budget_key,
            "semantic_source_mode": semantic_bundle["mode"],
            "semantic_sources": [item["manifest"].get("source_identity", item["manifest"].get("lm_model")) for item in semantic_bundle["sources"]],
            "semantic_source": "semantic_single_source",
            "semantic_alignment_note": "Joint training keeps the current cached semantic embedding path; router objective is task-adapted toward reliability rather than the paper's advantage target.",
            "dependency_alignment_note": dependency_alignment_note,
            "semantic_manifest": semantic["manifest"],
            "primary_semantic_manifest": semantic_bundle["primary"]["manifest"],
            "joint_refiner_embedding_path": str(getattr(self.args, "joint_refiner_embedding_path", None) or ""),
            "external_frozen_g0_root": str(getattr(self.args, "external_frozen_g0_root", None) or ""),
            "frozen_g0_dir": str(context["frozen_g0"]["dir"]),
            "frozen_g0_manifest": context["frozen_g0"]["manifest"],
            "train_node_cap": int(configured_train_cap),
            "effective_train_node_count": int(strict_train_idx.size),
            "full_train_node_count": int(train_idx.size),
            "train_cap_mode": train_cap_mode,
            "uncertainty_source": mc_bundle["source"],
            "router_temperature_scaling": dict(router_temperature_bundle),
            "router_feature_family_used": list(router_feature_bundle["feature_names"]),
            "router_architecture": {
                "type": "reliability_first_router_mlp",
                "hidden_dim": 128,
                "bottleneck_dim": 64,
                "dropout": 0.1,
                "input_dim": int(router_features.shape[1]),
                "feature_count": int(router_features.shape[1]),
                "feature_family": list(router_feature_bundle["feature_names"]),
                "feature_bundle_metadata": router_feature_bundle["full_feature_bundle"],
                "scaler": scaler_state,
            },
            "refiner_architecture": {
                "type": (
                    "prompt_expert_bundle_refiner_mlp"
                    if semantic_view_mode == "prompt_expert_bundle_v1"
                    else ("gated_glance_joint_refiner_mlp" if refiner_explicit_gate else "glance_joint_refiner_mlp")
                ),
                "input": (
                    "[z_gnn || ego_proj || graph_following_proj || graph_follower_proj || graph_fused || tweet_proj || conflict_proj || log1p(count_following) || log1p(count_follower) || has_following || has_follower]"
                    if semantic_view_mode == "prompt_expert_bundle_v1"
                    else "[z_gnn || z_sem_0 || z_sem_1 || z_sem_2]"
                ),
                "hidden_dim": 128,
                "activation": activation,
                "dropout": 0.1,
                "output_dim": int(logits_gnn.shape[1]),
                "semantic_view_mode": semantic_view_mode,
                "feature_kind": "prompt_expert_bundle_v1" if semantic_view_mode == "prompt_expert_bundle_v1" else "flat_concat",
                "semantic_view_names": (
                    ["ego", "graph_following", "graph_follower", "tweet", "conflict"]
                    if semantic_view_mode == "prompt_expert_bundle_v1"
                    else ["ego", "hop1", "hop2"]
                ),
                "active_components": prompt_expert_active_components,
                "proj_dim": 256 if semantic_view_mode == "prompt_expert_bundle_v1" else None,
                "graph_gate_input": ["log1p(count_following)", "log1p(count_follower)", "has_following", "has_follower"]
                if semantic_view_mode == "prompt_expert_bundle_v1"
                else None,
                "target_mode": refiner_target_mode,
                "explicit_gate": bool(refiner_explicit_gate) if semantic_view_mode != "prompt_expert_bundle_v1" else False,
                "weight_mode": refiner_weight_mode,
            },
            "training_contract": {
                "frozen_gnn": True,
                "frozen_semantic_encoder": True,
                "train_router": True,
                "train_refiner": True,
                "batch_size": int(batch_size),
                "max_epochs": int(max_epochs),
                "patience": int(patience),
                "train_node_cap": int(configured_train_cap),
                "effective_train_node_count": int(strict_train_idx.size),
                "full_train_node_count": int(train_idx.size),
                "train_cap_mode": train_cap_mode,
                "training_budget_schedule": {
                    "k_start": int(batch_size),
                    "k_end": int(max(int(round(batch_size / 4.0)), 1)),
                    "decay_factor": float(decay_factor),
                },
                "router_weight": 0.0,
                "router_regression_weight": 0.0,
                "router_ranking_weight": float(1.0),
                "router_selection_weight": float(0.25),
                "router_calibration_weight": float(0.25),
                "router_reliability_weight": 0.0,
                "entropy_weight": 0.0,
                "beta_selection": "validation_only",
                "router_training_objective": "base_wrong_reliability_bce_plus_pairwise_ranking",
                "auxiliary_router_target": "none_oracle_advantage_retained_for_diagnostics_only",
                "refiner_target_mode": refiner_target_mode,
                "refiner_explicit_gate": bool(refiner_explicit_gate) if semantic_view_mode != "prompt_expert_bundle_v1" else False,
                "refiner_weight_mode": refiner_weight_mode,
                "refiner_base_wrong_weight": float(refiner_base_wrong_weight),
                "refiner_utility_weight": float(refiner_utility_weight),
                "refiner_gate_weight": float(refiner_gate_weight),
            },
            "q_estimator": {
                **q_bundle["classifier"],
                "train_count": int(q_bundle["train_count"]),
                "valid_count": int(q_bundle["valid_count"]),
                "temperature": float(q_bundle["temperature"]),
            },
            "mc_dropout": {
                "num_passes": int(mc_bundle["num_passes"]),
                "logits_shape": list(mc_bundle["logits_shape"]),
            },
            "original_node_features": {
                "path": str(original_node_features_bundle["path"]),
                "feature_manifest": original_node_features_bundle["feature_manifest"],
            },
        }

        self._write_stage_manifest(
            stage_dir,
            manifest,
            artifact_namespace="stages/joint_router_refinement",
            visibility="public",
            resolved_task="joint_router_refinement",
        )
        write_json(stage_dir / "metrics.json", metrics_payload)
        write_json(stage_dir / "analysis_summary.json", analysis_summary)
        write_json(stage_dir / "router_performance_summary.json", router_performance_summary)
        write_csv_rows(
            stage_dir / "router_performance_summary.csv",
            list(router_performance_row.keys()),
            [router_performance_row],
        )
        write_csv_rows(
            stage_dir / "router_epoch_curve.csv",
            [
                "epoch",
                "beta",
                "train_top_k",
                "eval_top_k",
                "router_train_auroc",
                "router_valid_auroc",
                "router_test_auroc",
                "router_valid_adv_auroc",
                "router_test_adv_auroc",
                "router_valid_auprc",
                "router_valid_adv_auprc",
                "refiner_valid_fix_rate",
                "refiner_valid_break_rate",
                "refiner_test_fix_rate",
                "refiner_test_break_rate",
            ],
            final_run["fit_summary"]["component_curve_summary"]["curve"],
        )
        write_csv_rows(
            stage_dir / "beta_sweep.csv",
            [
                "beta",
                "eval_top_k",
                "best_epoch",
                "selected_budget",
                "selected_budget_key",
                "valid_accuracy",
                "valid_macro_f1",
                "valid_bot_f1",
                "valid_loss",
                "valid_routed_count",
                "valid_query_rate",
                "valid_net_gain",
                "valid_routed_wrong_precision",
                "valid_routed_wrong_coverage",
                "valid_conditional_fix_rate",
                "test_accuracy",
                "test_macro_f1",
                "test_bot_f1",
                "test_loss",
                "test_routed_count",
                "test_query_rate",
                "test_net_gain",
                "test_routed_wrong_precision",
                "test_routed_wrong_coverage",
                "test_conditional_fix_rate",
            ],
            beta_sweep,
        )
        write_csv_rows(
            stage_dir / "valid_budget_curve.csv",
            [
                "budget",
                "routed_count",
                "query_rate",
                "fix",
                "break",
                "net",
                "wrong_node_fix_rate",
                "correct_node_break_rate",
                "routed_wrong_count",
                "routed_correct_count",
                "routed_wrong_coverage",
                "routed_wrong_precision",
                "conditional_fix_rate_on_selected_wrong",
                "conditional_break_rate_on_selected_correct",
                "accuracy",
                "macro_f1",
                "bot_f1",
                "loss",
                "delta_accuracy",
                "delta_macro_f1",
                "delta_bot_f1",
                "mean_router_score_routed",
                "mean_router_prob_routed",
                "mean_oracle_advantage_routed",
            ],
            valid_budget_curve,
        )
        write_csv_rows(
            stage_dir / "test_budget_curve.csv",
            [
                "budget",
                "routed_count",
                "query_rate",
                "fix",
                "break",
                "net",
                "wrong_node_fix_rate",
                "correct_node_break_rate",
                "routed_wrong_count",
                "routed_correct_count",
                "routed_wrong_coverage",
                "routed_wrong_precision",
                "conditional_fix_rate_on_selected_wrong",
                "conditional_break_rate_on_selected_correct",
                "accuracy",
                "macro_f1",
                "bot_f1",
                "loss",
                "delta_accuracy",
                "delta_macro_f1",
                "delta_bot_f1",
                "mean_router_score_routed",
                "mean_router_prob_routed",
                "mean_oracle_advantage_routed",
            ],
            test_budget_curve,
        )
        write_text(
            stage_dir / "per_node_test.jsonl",
            "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in per_node_rows) + "\n",
        )
        save_stage_artifacts(
            stage_dir,
            {
                "outputs.pt": {
                    "logits": final_logits,
                    "prob": final_prob,
                    "pred": final_pred,
                    "labels": labels_t,
                    "base_pred": pred_gnn,
                    "routed_masks": routed_masks,
                    "router_prob": router_prob,
                    "router_score": router_score,
                    "oracle_advantage": oracle_advantage,
                    "neighbor_count_1hop": torch.tensor(semantic_views.get("count_1hop", np.zeros(len(self.labels), dtype=np.int64)), dtype=torch.long),
                    "neighbor_count_2hop": torch.tensor(semantic_views.get("count_2hop", np.zeros(len(self.labels), dtype=np.int64)), dtype=torch.long),
                },
                "checkpoint.pt": {
                    "router_model": final_run["router_state"],
                    "refiner_model": final_run["refiner_state"],
                    "fit_summary": final_run["fit_summary"],
                    "selected_beta": selected_beta,
                    "selected_budget": selected_budget,
                    "selected_eval_top_k": int(eval_top_k),
                },
                "beta_sweep": beta_sweep,
                "valid_budget_curve": valid_budget_curve,
                "test_budget_curve": test_budget_curve,
                **base_bundle,
            },
        )
        notes = [
            "# GLANCE joint router-refiner baseline",
            "",
            "This stage implements a GLANCE-style joint router+refiner contract under the current cached semantic embedding path.",
            "The public router is task-adapted for TwiBot20 reliability routing: base confidence is temperature-scaled, social features are direction-aware, and router supervision targets base-wrong reliability rather than paper-text advantage.",
            "The semantic expert alignment is intentionally deferred, and the public evaluation protocol is TwiBot20-adapted global-budget routing.",
            "The base GNN and semantic expert remain frozen; only the router scorer MLP and the routed-node refiner MLP are trained.",
            "The router is trained with base-wrong reliability BCE plus pairwise ranking. Oracle advantage is still recorded as a post-hoc diagnostic trace.",
            f"Training uses deterministic batch top-k with K_start={int(batch_size)} and K_end={int(eval_top_k)}. Final evaluation ranks the whole split by router score, selects budget on validation, and locks the selected budget ({selected_budget:.3f}) on test. Beta is selected on validation only from {{0.1, 0.2, 0.3}}.",
            "Use router_performance_summary.json/csv, router_epoch_curve.csv, and the budget-curve CSVs to inspect router quality directly.",
        ]
        write_text(stage_dir / "notes.md", "\n".join(notes) + "\n")
        return {
            "stage": "glance_joint_router_refine",
            "stage_dir": str(stage_dir),
            "metrics": metrics_payload,
        }
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
        est = self._read_stage_json("estimator_ablation", "survivor_manifest") or {"best_p_proxy": default_p_mode}
        sem = self._read_stage_json("semantic_operator_ablation", "survivor_manifest") or {"best_semantic_operator": default_sem_mode}
        sem_source = self._read_stage_json("semantic_source_ablation", "survivor_manifest") or {"best_semantic_source": "S0_roberta_local_source"}
        rep = self._read_stage_json("repair_operator_ablation", "survivor_manifest") or {"best_repair_single_action": default_rep_mode}
        sel = self._read_stage_json("selector_ablation", "survivor_manifest") or {"best_selector": "true three-action"}
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

    def _run_estimator_matrix(self, stage_dir, base_bundle):
        if str(getattr(self, "graph_data_variant", "labeled")).lower() == "full_graph_support" and not self._graph_conformal_estimator_requested():
            raise MissingFrozenArtifactError(
                "graph_data_variant=full_graph_support currently supports only conformal-style estimator_ablation modes "
                "(`graph_conformal_set_estimator`, `posthoc_calibrated_ranker`, `calibrated_local_risk_router`, "
                "`conformal_knn_risk_router`, `gnn_2hop_conformal`)."
            )
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
            router_bundle, router_artifacts, router_budget_rows = self._build_glance_context_router_bundle(load_gats_outputs(self.experiment_root))
            router_mode_name = router_bundle["risk_manifest"].get("calibration_metadata", {}).get("source", "glance_for_context_residual_risk_selector")
            suite[str(router_mode_name)] = router_bundle

        requested_mode = str(getattr(self.args, "estimator_mode", "none")).lower()
        if requested_mode == "glance_for_context_residual_risk_selector":
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
        if best_mode == "glance_for_context_residual_risk_selector":
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
                        _stage_artifact_reference("minimal_pipeline", "risk_manifest.json"),
                        _stage_artifact_reference("minimal_pipeline", "subgroup_manifest_operational.json"),
                        _preparation_artifact_reference("graph_calibrator", "manifest.json"),
                        _preparation_artifact_reference("graph_calibrator", "outputs.pt"),
                        *(
                            [
                                "runtime/router_oof/manifest.json",
                                "runtime/router_oof/outputs.pt",
                                "runtime/lm_only_head/manifest.json",
                            ]
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
            if best_mode == "glance_for_context_residual_risk_selector":
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

    def _fail_outside_phase0_scope(self):
        require_stage_gates(self.experiment_root, self.execution_stage)
        dependency_map = {
            "semantic_operator_ablation": [("estimator_ablation", "survivor_manifest")],
            "repair_operator_ablation": [("estimator_ablation", "survivor_manifest")],
            "selector_ablation": [("repair_operator_ablation", "survivor_manifest")],
            "positioning_ablation": [("selector_ablation", "survivor_manifest")],
            "backbone_stress_test": [("selector_ablation", "survivor_manifest")],
            "appendix_ablation": [("estimator_ablation", "survivor_manifest")],
        }
        for stage_name, filename in dependency_map.get(self.execution_stage, []):
            self._require_stage_json(stage_name, filename)
        raise MissingFrozenArtifactError(
            f"Stage '{self.execution_stage}' is outside the Phase 0 validation scope. "
            "Strict frozen harness refuses to execute semantic, repair, selector, "
            "positioning, appendix, or backbone-stress recomputation paths in this pass."
        )

    def run(self):
        stage_dir = build_stage_dir(self.experiment_root, self.execution_stage)
        base_bundle = self._base_artifact_bundle()

        if self.execution_stage == "local_conformal_diagnostic":
            return self._run_local_conformal_prune_diag(stage_dir, base_bundle)

        if self.execution_stage == "local_conflict_prune_diag":
            return self._run_local_conflict_prune_diag(stage_dir, base_bundle)

        if self.execution_stage == "local_dignn_conflict_refine_diag":
            from trainer_dignn_conflict import run_local_dignn_conflict_refine_diag
            return run_local_dignn_conflict_refine_diag(self, stage_dir, base_bundle)

        if self.execution_stage == "joint_router_refinement":
            return self._run_glance_joint_router_refine(stage_dir, base_bundle)

        if self.execution_stage == "minimal_pipeline":
            return self._run_vertical_minimal(stage_dir, base_bundle)

        if self.execution_stage == "estimator_ablation":
            return self._run_estimator_matrix(stage_dir, base_bundle)

        if self.execution_stage in {
            "semantic_operator_ablation",
            "repair_operator_ablation",
            "selector_ablation",
            "positioning_ablation",
            "backbone_stress_test",
            "appendix_ablation",
        }:
            return self._fail_outside_phase0_scope()

        if self.execution_stage == "semantic_source_ablation":
            return self._run_semantic_source_matrix(stage_dir, base_bundle, None, None)

        raise ValueError(f"Unsupported stage: {self.execution_stage}")

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
                fused_x = out.get('fused_x', out['node_repr']).cpu()
                x_low = out.get('x_low')
                x_new = out.get('x_new')
                aux_features = out.get('aux_features', {})
            else:
                logits = self.model(x, edge_index, edge_type).cpu()
                prob = torch.softmax(logits, dim=1)
                node_repr = logits
                fused_x = logits
                x_low = None
                x_new = None
                aux_features = {}

        outputs = {
            'logits': logits,
            'prob': prob,
            'pred': prob.argmax(dim=1),
            'labels': self.hard_labels.cpu(),
            'node_repr': node_repr,
            'fused_x': fused_x,
            'aux_features': aux_features,
        }
        if torch.is_tensor(x_low):
            outputs['x_low'] = x_low.cpu()
        if torch.is_tensor(x_new):
            outputs['x_new'] = x_new.cpu()
        return outputs

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
