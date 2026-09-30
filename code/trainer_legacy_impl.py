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
from artifact_contracts import (
    MissingFrozenArtifactError as _contract_missing_frozen_artifact_error,
    PHASE_A_CONTRACT as _contract_phase_a_contract,
    PHASE_A_DISABLED_COMPONENTS as _contract_phase_a_disabled_components,
    canonical_stage_name as _contract_canonical_stage_name,
    canonical_stage_dir as _contract_canonical_stage_dir,
    legacy_stage_dir as _contract_legacy_stage_dir,
    stage_dir_read_candidates as _contract_stage_dir_read_candidates,
    stage_artifact_reference as _contract_stage_artifact_reference,
    preparation_artifact_reference as _contract_preparation_artifact_reference,
    split_provenance as _contract_split_provenance,
    frozen_g0_dir as _contract_frozen_g0_dir,
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
from trainer_distillation import (
    GNN_Trainer as _distill_gnn_trainer,
    LM_Trainer as _distill_lm_trainer,
    MLP_Trainer as _distill_mlp_trainer,
    _safe_pseudo_label_training_index as _distill_safe_pseudo_label_training_index,
    run_legacy_graph_seed as _distill_run_legacy_graph_seed,
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



_safe_pseudo_label_training_index = _distill_safe_pseudo_label_training_index




LM_Trainer = _distill_lm_trainer



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



run_legacy_graph_seed = _distill_run_legacy_graph_seed



# Compatibility rebinding: active preparation and semantic owners now live in
# dedicated modules. Keep legacy names stable for StageRunner while routing
# runtime behavior through the extracted implementations.
MissingFrozenArtifactError = _contract_missing_frozen_artifact_error
PHASE_A_CONTRACT = _contract_phase_a_contract
PHASE_A_DISABLED_COMPONENTS = _contract_phase_a_disabled_components
_canonical_stage_name = _contract_canonical_stage_name
_canonical_stage_dir = _contract_canonical_stage_dir
_legacy_stage_dir = _contract_legacy_stage_dir
_stage_dir_read_candidates = _contract_stage_dir_read_candidates
_stage_artifact_reference = _contract_stage_artifact_reference
_preparation_artifact_reference = _contract_preparation_artifact_reference
split_provenance = _contract_split_provenance
frozen_g0_dir = _contract_frozen_g0_dir
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

    @staticmethod
    def _node_cross_entropy_vector(prob, labels):
        prob_t = prob.detach().cpu().float() if torch.is_tensor(prob) else torch.tensor(prob, dtype=torch.float32)
        labels_t = labels.detach().cpu().long() if torch.is_tensor(labels) else torch.tensor(labels, dtype=torch.long)
        row = torch.arange(labels_t.numel(), dtype=torch.long)
        return (-torch.log(prob_t[row, labels_t].clamp_min(1e-8))).cpu()

    @staticmethod
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
            raise ValueError(
                "local_conflict_prune_diag is owned by trainer_graph.GraphStageMixin; "
                "use stage_runner.StageRunner instead of trainer_legacy_impl.StageRunner."
            )

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


GNN_Trainer = _distill_gnn_trainer



MLP_Trainer = _distill_mlp_trainer
