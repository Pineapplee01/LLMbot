import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from artifact_contracts import MissingFrozenArtifactError, frozen_g0_dir, split_provenance
from model_building import (
    PhaseAInputAdapter,
    _idx_tensor,
    _labels_to_index,
    _score_logits,
    build_GNN_model,
    phase_a_project_dim,
    phase_a_projector,
    resolve_g0_feature_bundle,
)
from runtime_env import _resolve_device
from stage_registry import resolve_stage_name
from subgroups import structural_features
from utils import (
    build_preparation_dir,
    ensure_dir,
    read_json,
    safe_torch_load,
    tensor_sha256,
    write_json,
    write_torch,
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
    graph_node_count = int(data.get("graph_node_count", int(features.shape[0])))
    labeled_node_count = int(data.get("labeled_node_count", int(labels.numel())))
    support_node_count = int(data.get("support_node_count", max(graph_node_count - labeled_node_count, 0)))
    graph_data_variant = str(data.get("graph_data_variant", "labeled"))
    if int(features.shape[0]) != graph_node_count:
        raise ValueError(f"G0 feature rows ({int(features.shape[0])}) must match graph_node_count ({graph_node_count}).")
    if int(raw_features.shape[0]) != graph_node_count:
        raise ValueError(f"G0 raw feature rows ({int(raw_features.shape[0])}) must match graph_node_count ({graph_node_count}).")
    if int(labels.numel()) != labeled_node_count:
        raise ValueError(f"Label rows ({int(labels.numel())}) must match labeled_node_count ({labeled_node_count}).")

    device = getattr(args, "device", torch.device("cpu"))
    if not isinstance(device, torch.device):
        device = _resolve_device(device)

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
                "num_nodes": int(graph_node_count),
                "graph_node_count": int(graph_node_count),
                "labeled_node_count": int(labeled_node_count),
                "support_node_count": int(support_node_count),
                "labels_sha256": tensor_sha256(data["labels"]),
            },
            "graph_data_variant": graph_data_variant,
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
    legacy_dir = experiment_root / "frozen" / "g0"
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
    if mode == "none":
        return {
            "mode": "none",
            "budget": 0.0,
            "degree_guard_enabled": False,
            "oracle_uses_labels": False,
            "diagnostic_only": False,
        }
    if mode not in {
        "directional_relation_aware_prune",
        "directional_relation_aware_random_prune",
        "directional_relation_aware_oracle_prune",
        "directional_relation_aware_oracle_prune_no_hetero_priority",
    }:
        raise ValueError(f"Unknown graph_refine_mode: {mode}")
    if not (0.0 < budget < 1.0):
        raise ValueError("graph_refine_budget must be in (0, 1) when --graph_refine_mode is not none.")
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
    }


def _directional_relation_aware_prune(edge_index, edge_type, raw_features, budget, mode, seed=None, labels=None):
    edge_index_cpu = (
        edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
    )
    edge_type_cpu = (
        edge_type.detach().cpu().long() if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long)
    )
    raw_features_cpu = (
        raw_features.detach().cpu().float()
        if torch.is_tensor(raw_features)
        else torch.tensor(raw_features, dtype=torch.float32)
    )

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
                        key=lambda idx: (float(priority[idx, 0].item()), float(priority[idx, 1].item())),
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
    if not math.isclose(
        float(refine_manifest.get("budget_requested", -1.0)),
        float(refine_request["budget"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return False
    if bool(refine_manifest.get("degree_guard_enabled", False)) != bool(refine_request["degree_guard_enabled"]):
        return False
    if bool(refine_manifest.get("oracle_uses_labels", False)) != bool(refine_request["oracle_uses_labels"]):
        return False
    if bool(refine_manifest.get("diagnostic_only", False)) != bool(refine_request["diagnostic_only"]):
        return False
    return True


def build_or_load_frozen_g0(args, seed, data, experiment_root):
    canonical_dir = Path(build_preparation_dir(experiment_root, "graph_detector"))
    legacy_dir = Path(experiment_root) / "frozen" / "g0"
    out_dir = canonical_dir if (canonical_dir / "manifest.json").exists() else legacy_dir
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
    graph_node_count = int(data.get("graph_node_count", int(labels.numel())))
    features = structural_features(data["edge_index"], data["edge_type"], graph_node_count)
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
    canonical_stage_name = resolve_stage_name(stage_name)
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
            "source_impl": "trainer_preparation.GraphGATSCalibrator",
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
