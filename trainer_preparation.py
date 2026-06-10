import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from artifact_contracts import MissingFrozenArtifactError, frozen_g0_dir, split_provenance
from GNNs import (
    mhlgc_llm_guided_contrastive_loss,
    mhlgc_mask_edges,
    mhlgc_mask_node_features,
)
from hypergnn import (
    build_relation_overlap_center_candidates as _shared_build_relation_overlap_center_candidates,
    feature_k_nearest_neighbor_query as _shared_feature_k_nearest_neighbor_query,
    hyperscan_knn_hypergraph_proxy_augment as _shared_hyperscan_knn_hypergraph_proxy_augment,
    relation_overlap_knn_proxy_augment as _shared_relation_overlap_knn_proxy_augment,
)
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


def _relation_cardinality_from_edge_type(edge_type, minimum=1):
    if edge_type is None:
        return int(max(int(minimum), 1))
    edge_type_t = edge_type.detach().cpu().long().view(-1) if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long).view(-1)
    if edge_type_t.numel() == 0:
        return int(max(int(minimum), 1))
    return int(max(int(minimum), int(edge_type_t.max().item()) + 1))


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
        "hyperscan_detector_style": getattr(args, "hyperscan_detector_style", "residual"),
        "graph_second_view_hypergraph_backend": getattr(args, "graph_second_view_hypergraph_backend", "pyg"),
        "graph_second_view_fusion": getattr(args, "graph_second_view_fusion", "residual"),
        "RGT_semantic_heads": getattr(args, "RGT_semantic_heads", 8),
        "device": device,
    }


def _load_mhlgc_semantic_embeddings(path, expected_rows):
    if not path:
        return None
    payload = safe_torch_load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("embeddings", "semantic_embeddings", "features", "x", "prompt_embeddings"):
            if key in payload:
                payload = payload[key]
                break
    if not torch.is_tensor(payload):
        payload = torch.as_tensor(payload)
    payload = payload.detach().cpu().float()
    if payload.dim() != 2:
        raise ValueError(f"--mhlgc_semantic_embedding_path must resolve to a 2-D tensor, got {tuple(payload.shape)}.")
    if int(payload.shape[0]) != int(expected_rows):
        raise ValueError(
            "--mhlgc_semantic_embedding_path row count must match graph nodes: "
            f"expected {int(expected_rows)}, got {int(payload.shape[0])}."
        )
    return payload.contiguous()


def _train_graph_backbone_once(
    config,
    x_projected,
    x_raw,
    y,
    edge_index,
    edge_type,
    train_idx,
    valid_idx,
    peft_enabled=False,
    peft_rank=8,
    peft_alpha=16.0,
    raw_feature_dim=None,
    learning_rate=5e-4,
    weight_decay=1e-5,
    max_epochs=1,
    training_loader_mode="full_batch",
    graph_batch_size=1024,
    neighbor_num_neighbors=64,
    max_update_steps=0,
    mhlgc_enabled=False,
    mhlgc_semantic_embeddings=None,
    mhlgc_loss_weight=0.0,
    mhlgc_beta=1.0,
    mhlgc_gamma=0.5,
    mhlgc_temperature=1.0,
    mhlgc_feature_mask_probability=0.15,
    mhlgc_edge_mask_probability=0.10,
    mhlgc_anchors_per_batch=1,
    mhlgc_positive_label=1,
):
    device = config["device"]
    model = build_GNN_model(config)
    input_adapter = None
    if peft_enabled:
        if raw_feature_dim is None:
            raise ValueError("raw_feature_dim is required when peft_enabled=True.")
        input_adapter = PhaseAInputAdapter(
            raw_dim=int(raw_feature_dim),
            projected_dim=int(x_projected.shape[1]),
            rank=int(peft_rank),
            alpha=float(peft_alpha),
        ).to(device)

    trainable_parameters = list(model.parameters())
    if input_adapter is not None:
        trainable_parameters += list(input_adapter.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    max_epochs = max(int(max_epochs), 1)
    max_update_steps = max(int(max_update_steps), 0)
    best_state = None
    best_metrics = None
    optimizer_steps = 0
    mhlgc_enabled = bool(mhlgc_enabled)
    mhlgc_loss_weight = float(mhlgc_loss_weight or 0.0)
    mhlgc_stats = {
        "enabled": bool(mhlgc_enabled),
        "loss_weight": float(mhlgc_loss_weight),
        "active_steps": 0,
        "total_steps": 0,
        "loss_sum": 0.0,
        "anchor_count_sum": 0,
        "negative_count_sum": 0,
    }

    def _record_mhlgc(loss_tensor, stats):
        mhlgc_stats["total_steps"] += 1
        if bool(stats.get("mhlgc_active", False)):
            mhlgc_stats["active_steps"] += 1
            mhlgc_stats["loss_sum"] += float(loss_tensor.detach().cpu().item())
            mhlgc_stats["anchor_count_sum"] += int(stats.get("mhlgc_anchor_count", 0))
            mhlgc_stats["negative_count_sum"] += int(stats.get("mhlgc_negative_count", 0))

    if str(training_loader_mode).lower() == "neighbor_subgraph":
        edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
        edge_type_cpu = edge_type.detach().cpu().long().view(-1) if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long).view(-1)
        x_projected_cpu = x_projected.detach().cpu().float() if torch.is_tensor(x_projected) else torch.tensor(x_projected, dtype=torch.float32)
        x_raw_cpu = x_raw.detach().cpu().float() if torch.is_tensor(x_raw) else torch.tensor(x_raw, dtype=torch.float32)
        y_cpu = y.detach().cpu().long() if torch.is_tensor(y) else torch.tensor(y, dtype=torch.long)
        train_idx_cpu = train_idx.detach().cpu().long().view(-1)
        valid_idx_cpu = valid_idx.detach().cpu().long().view(-1)
        n_layers = int(config.get("gnn_n_layers", config.get("n_layers", 2)))
        loader_neighbors = [int(neighbor_num_neighbors)] * max(int(n_layers), 1)
        loader_data = Data(
            x=x_projected_cpu,
            raw_x=x_raw_cpu,
            y=y_cpu,
            edge_index=edge_index_cpu,
            edge_type=edge_type_cpu,
            node_id=torch.arange(int(x_projected_cpu.shape[0]), dtype=torch.long),
        )
        loader_data.num_nodes = int(x_projected_cpu.shape[0])
        batch_size = max(int(graph_batch_size), 1)
        train_loader = NeighborLoader(
            data=loader_data,
            num_neighbors=loader_neighbors,
            input_nodes=train_idx_cpu,
            batch_size=batch_size,
            shuffle=True,
        )
        valid_loader = NeighborLoader(
            data=loader_data,
            num_neighbors=loader_neighbors,
            input_nodes=valid_idx_cpu,
            batch_size=batch_size,
            shuffle=False,
        )

        for epoch in range(max_epochs):
            model.train()
            if input_adapter is not None:
                input_adapter.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                x_batch = input_adapter(batch.x, batch.raw_x) if input_adapter is not None else batch.x
                batch_outputs = model.forward_outputs(x_batch, batch.edge_index, batch.edge_type.view(-1))
                logits = batch_outputs["logits"]
                seed_count = int(batch.batch_size)
                loss = F.cross_entropy(logits[:seed_count], batch.y[:seed_count])
                if mhlgc_enabled and mhlgc_loss_weight > 0.0:
                    x_aug_projected = mhlgc_mask_node_features(
                        batch.x,
                        mask_probability=float(mhlgc_feature_mask_probability),
                    )
                    raw_aug = (
                        mhlgc_mask_node_features(
                            batch.raw_x,
                            mask_probability=float(mhlgc_feature_mask_probability),
                        )
                        if input_adapter is not None
                        else batch.raw_x
                    )
                    x_aug = input_adapter(x_aug_projected, raw_aug) if input_adapter is not None else x_aug_projected
                    edge_index_aug, edge_type_aug = mhlgc_mask_edges(
                        batch.edge_index,
                        batch.edge_type.view(-1),
                        mask_probability=float(mhlgc_edge_mask_probability),
                    )
                    aug_outputs = model.forward_outputs(x_aug, edge_index_aug, edge_type_aug)
                    semantic_batch = None
                    if mhlgc_semantic_embeddings is not None:
                        semantic_batch = mhlgc_semantic_embeddings[
                            batch.node_id[:seed_count].detach().cpu().long()
                        ].to(device)
                    mhlgc_loss, mhlgc_step_stats = mhlgc_llm_guided_contrastive_loss(
                        origin_embeddings=batch_outputs["node_repr"][:seed_count],
                        augmented_embeddings=aug_outputs["node_repr"][:seed_count],
                        labels=batch.y[:seed_count],
                        fraud_scores=torch.softmax(logits[:seed_count], dim=-1)[:, int(mhlgc_positive_label)],
                        semantic_embeddings=semantic_batch,
                        positive_label=int(mhlgc_positive_label),
                        anchors_per_batch=int(mhlgc_anchors_per_batch),
                        beta=float(mhlgc_beta),
                        gamma=float(mhlgc_gamma if semantic_batch is not None else 0.0),
                        temperature=float(mhlgc_temperature),
                    )
                    _record_mhlgc(mhlgc_loss, mhlgc_step_stats)
                    loss = loss + mhlgc_loss_weight * mhlgc_loss
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                if max_update_steps > 0 and optimizer_steps >= max_update_steps:
                    break

            model.eval()
            if input_adapter is not None:
                input_adapter.eval()
            eval_logits = []
            eval_labels = []
            with torch.no_grad():
                for batch in valid_loader:
                    batch = batch.to(device)
                    x_eval = input_adapter(batch.x, batch.raw_x) if input_adapter is not None else batch.x
                    logits_eval = model(x_eval, batch.edge_index, batch.edge_type.view(-1))
                    seed_count = int(batch.batch_size)
                    eval_logits.append(logits_eval[:seed_count].detach().cpu())
                    eval_labels.append(batch.y[:seed_count].detach().cpu())
            stacked_logits = torch.cat(eval_logits, dim=0) if eval_logits else torch.empty((0, 2), dtype=torch.float32)
            stacked_labels = torch.cat(eval_labels, dim=0) if eval_labels else torch.empty((0,), dtype=torch.long)
            val_metrics = _score_logits(stacked_logits, stacked_labels, torch.arange(stacked_labels.numel(), dtype=torch.long))
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
            if max_update_steps > 0 and optimizer_steps >= max_update_steps:
                break
    else:
        for epoch in range(max_epochs):
            model.train()
            if input_adapter is not None:
                input_adapter.train()
            optimizer.zero_grad()
            x = input_adapter(x_projected, x_raw) if input_adapter is not None else x_projected
            outputs = model.forward_outputs(x, edge_index, edge_type)
            logits = outputs["logits"]
            loss = F.cross_entropy(logits[train_idx], y[train_idx])
            if mhlgc_enabled and mhlgc_loss_weight > 0.0:
                x_aug_projected = mhlgc_mask_node_features(
                    x_projected,
                    mask_probability=float(mhlgc_feature_mask_probability),
                )
                raw_aug = (
                    mhlgc_mask_node_features(x_raw, mask_probability=float(mhlgc_feature_mask_probability))
                    if input_adapter is not None
                    else x_raw
                )
                x_aug = input_adapter(x_aug_projected, raw_aug) if input_adapter is not None else x_aug_projected
                edge_index_aug, edge_type_aug = mhlgc_mask_edges(
                    edge_index,
                    edge_type,
                    mask_probability=float(mhlgc_edge_mask_probability),
                )
                aug_outputs = model.forward_outputs(x_aug, edge_index_aug, edge_type_aug)
                semantic_train = (
                    mhlgc_semantic_embeddings[train_idx.detach().cpu().long()].to(device)
                    if mhlgc_semantic_embeddings is not None
                    else None
                )
                mhlgc_loss, mhlgc_step_stats = mhlgc_llm_guided_contrastive_loss(
                    origin_embeddings=outputs["node_repr"][train_idx],
                    augmented_embeddings=aug_outputs["node_repr"][train_idx],
                    labels=y[train_idx],
                    fraud_scores=torch.softmax(logits[train_idx], dim=-1)[:, int(mhlgc_positive_label)],
                    semantic_embeddings=semantic_train,
                    positive_label=int(mhlgc_positive_label),
                    anchors_per_batch=int(mhlgc_anchors_per_batch),
                    beta=float(mhlgc_beta),
                    gamma=float(mhlgc_gamma if semantic_train is not None else 0.0),
                    temperature=float(mhlgc_temperature),
                )
                _record_mhlgc(mhlgc_loss, mhlgc_step_stats)
                loss = loss + mhlgc_loss_weight * mhlgc_loss
            loss.backward()
            optimizer.step()
            optimizer_steps += 1

            model.eval()
            if input_adapter is not None:
                input_adapter.eval()
            with torch.no_grad():
                x_eval = input_adapter(x_projected, x_raw) if input_adapter is not None else x_projected
                logits_eval = model(x_eval, edge_index, edge_type)
            val_metrics = _score_logits(logits_eval.detach().cpu(), y.detach().cpu(), valid_idx.detach().cpu())
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
            if max_update_steps > 0 and optimizer_steps >= max_update_steps:
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        if input_adapter is not None and best_state["input_adapter"] is not None:
            input_adapter.load_state_dict(best_state["input_adapter"])

    model.eval()
    if input_adapter is not None:
        input_adapter.eval()
    if str(training_loader_mode).lower() == "neighbor_subgraph":
        edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
        edge_type_cpu = edge_type.detach().cpu().long().view(-1) if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long).view(-1)
        x_projected_cpu = x_projected.detach().cpu().float() if torch.is_tensor(x_projected) else torch.tensor(x_projected, dtype=torch.float32)
        x_raw_cpu = x_raw.detach().cpu().float() if torch.is_tensor(x_raw) else torch.tensor(x_raw, dtype=torch.float32)
        y_cpu = y.detach().cpu().long() if torch.is_tensor(y) else torch.tensor(y, dtype=torch.long)
        n_layers = int(config.get("gnn_n_layers", config.get("n_layers", 2)))
        infer_data = Data(
            x=x_projected_cpu,
            raw_x=x_raw_cpu,
            y=y_cpu,
            edge_index=edge_index_cpu,
            edge_type=edge_type_cpu,
            node_id=torch.arange(int(x_projected_cpu.shape[0]), dtype=torch.long),
        )
        infer_data.num_nodes = int(x_projected_cpu.shape[0])
        infer_loader = NeighborLoader(
            data=infer_data,
            num_neighbors=[int(neighbor_num_neighbors)] * max(int(n_layers), 1),
            input_nodes=torch.arange(int(x_projected_cpu.shape[0]), dtype=torch.long),
            batch_size=max(int(graph_batch_size), 1),
            shuffle=False,
        )
        logits_full = None
        prob_full = None
        repr_full = None
        aux_accumulator = {}
        aux_weight = 0.0
        with torch.no_grad():
            for batch in infer_loader:
                batch = batch.to(device)
                x_final = input_adapter(batch.x, batch.raw_x) if input_adapter is not None else batch.x
                batch_outputs = model.forward_outputs(x_final, batch.edge_index, batch.edge_type.view(-1))
                seed_count = int(batch.batch_size)
                global_ids = batch.node_id[:seed_count].detach().cpu().long()
                batch_logits = batch_outputs["logits"][:seed_count].detach().cpu()
                batch_prob = batch_outputs["prob"][:seed_count].detach().cpu()
                batch_repr = batch_outputs["node_repr"][:seed_count].detach().cpu()
                if logits_full is None:
                    logits_full = torch.zeros((int(x_projected_cpu.shape[0]), int(batch_logits.shape[1])), dtype=batch_logits.dtype)
                    prob_full = torch.zeros((int(x_projected_cpu.shape[0]), int(batch_prob.shape[1])), dtype=batch_prob.dtype)
                    repr_full = torch.zeros((int(x_projected_cpu.shape[0]), int(batch_repr.shape[1])), dtype=batch_repr.dtype)
                logits_full[global_ids] = batch_logits
                prob_full[global_ids] = batch_prob
                repr_full[global_ids] = batch_repr
                branch_stats = batch_outputs.get("aux_features", {}).get("dynamic_similarity_branch", {})
                if isinstance(branch_stats, dict):
                    weight = float(seed_count)
                    aux_weight += weight
                    for key, value in branch_stats.items():
                        if isinstance(value, bool):
                            aux_accumulator.setdefault(key, 0.0)
                            aux_accumulator[key] += (1.0 if value else 0.0) * weight
                        elif isinstance(value, (int, float)):
                            aux_accumulator.setdefault(key, 0.0)
                            aux_accumulator[key] += float(value) * weight
                        else:
                            aux_accumulator.setdefault(key, value)
        aggregated_aux = {}
        for key, value in aux_accumulator.items():
            if isinstance(value, float) and aux_weight > 0.0 and key not in {"backend", "candidate_scope", "center_source", "feature_source"}:
                aggregated_aux[key] = value / aux_weight
            else:
                aggregated_aux[key] = value
        outputs = {
            "logits": logits_full if logits_full is not None else torch.empty((0, 2), dtype=torch.float32),
            "prob": prob_full if prob_full is not None else torch.empty((0, 2), dtype=torch.float32),
            "node_repr": repr_full if repr_full is not None else torch.empty((0, config["hidden_dim"]), dtype=torch.float32),
            "aux_features": {
                "dynamic_similarity_branch": aggregated_aux,
            },
        }
    else:
        with torch.no_grad():
            x_final = input_adapter(x_projected, x_raw) if input_adapter is not None else x_projected
            outputs = model.forward_outputs(x_final, edge_index, edge_type)
    return {
        "model": model,
        "input_adapter": input_adapter,
        "outputs": outputs,
        "best_metrics": best_metrics or {},
        "best_state": best_state,
        "optimizer_steps": int(optimizer_steps),
        "mhlgc_stats": {
            **mhlgc_stats,
            "loss_mean_active": (
                float(mhlgc_stats["loss_sum"] / float(mhlgc_stats["active_steps"]))
                if int(mhlgc_stats["active_steps"]) > 0
                else 0.0
            ),
        },
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
    hyperscan_backbone_names = {
        "rgcn_hyperscan",
        "rgcn_hyperscan_routed",
        "rgcn_hyperscan_dhg",
        "rgcn_hyperscan_nodeinput",
        "rgcn_hyperscan_dhg_nodeinput",
    }
    if refine_request["mode"] == "routed_dynamic_hyperscan_branch":
        requested_backbone = str(getattr(args, "GNN_model", getattr(args, "graph_backbone", "rgcn"))).lower()
        if requested_backbone not in hyperscan_backbone_names:
            raise ValueError(
                "routed_dynamic_hyperscan_branch requires --graph_backbone rgcn_hyperscan "
                "or one of the legacy rgcn_hyperscan_* aliases "
                "so the training-time similarity hypergraph branch is actually active."
            )
    if refine_request["mode"] == "hyperscan_neighborloader_batch_local_branch":
        requested_backbone = str(getattr(args, "GNN_model", getattr(args, "graph_backbone", "rgcn"))).lower()
        if requested_backbone not in hyperscan_backbone_names:
            raise ValueError(
                "hyperscan_neighborloader_batch_local_branch requires --graph_backbone "
                "rgcn_hyperscan or one of the legacy rgcn_hyperscan_* aliases."
            )
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
    graph_override = _resolve_optional_graph_override_paths(args)
    graph_override_manifest = None
    if graph_override is not None:
        edge_index = safe_torch_load(graph_override["edge_index_path"], map_location="cpu")
        edge_type = safe_torch_load(graph_override["edge_type_path"], map_location="cpu")
        edge_index = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
        edge_type = edge_type.detach().cpu().long().view(-1) if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long).view(-1)
        if edge_index.dim() != 2 or int(edge_index.shape[0]) != 2:
            raise ValueError(
                f"--external_graph_edge_index_path must contain a [2, num_edges] tensor, got {tuple(edge_index.shape)}."
            )
        if int(edge_type.numel()) != int(edge_index.shape[1]):
            raise ValueError(
                "--external_graph_edge_type_path must align with --external_graph_edge_index_path: "
                f"{int(edge_type.numel())} types for {int(edge_index.shape[1])} edges."
            )
        graph_override_manifest = {
            "mode": "external_graph_override",
            "edge_index_path": str(graph_override["edge_index_path"]),
            "edge_type_path": str(graph_override["edge_type_path"]),
            "edge_count": int(edge_index.shape[1]),
            "relation_cardinality": int(_relation_cardinality_from_edge_type(edge_type, minimum=0)),
            "edge_index_sha256": tensor_sha256(edge_index),
            "edge_type_sha256": tensor_sha256(edge_type),
        }
    train_idx_cpu = _idx_tensor(data["train_idx"])
    valid_idx_cpu = _idx_tensor(data["valid_idx"])

    graph_refine_stats = None
    dynamic_similarity_branch = None
    dynamic_similarity_branch_stats = None
    if refine_request["mode"] == "relation_overlap_knn_repr_prefit_augment":
        edge_index, edge_type, graph_refine_stats = _relation_overlap_knn_repr_prefit_augment(
            edge_index=edge_index,
            edge_type=edge_type,
            feature_tensor=features,
            raw_features=raw_features,
            labels=labels,
            args=args,
            train_idx=train_idx_cpu,
            valid_idx=valid_idx_cpu,
            k=int(refine_request.get("knn_k", 8)),
            refine_request=refine_request,
            labeled_node_count=labeled_node_count,
        )
        write_torch(out_dir / "pruned_edge_index.pt", edge_index)
        write_torch(out_dir / "pruned_edge_type.pt", edge_type)
    elif refine_request["mode"] == "routed_dynamic_hyperscan_branch":
        dynamic_similarity_branch = _relation_overlap_center_candidates(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=graph_node_count,
            refine_request=refine_request,
            labeled_node_count=labeled_node_count,
        )
        dynamic_similarity_branch_stats = {
            "mode": "routed_dynamic_hyperscan_branch",
            "budget_requested": 0.0,
            "num_nodes": int(graph_node_count),
            "num_edges_before": int(edge_index.shape[1]) if hasattr(edge_index, "shape") else int(np.asarray(edge_index).shape[1]),
            "num_edges_after": int(edge_index.shape[1]) if hasattr(edge_index, "shape") else int(np.asarray(edge_index).shape[1]),
            "num_edges_added": 0,
            "num_hyperedges": int(dynamic_similarity_branch["center_count"]),
            "center_count": int(dynamic_similarity_branch["center_count"]),
            "center_source": dynamic_similarity_branch["center_source"],
            "routed_nodes_path": str(refine_request.get("routed_nodes_path", "") or ""),
            "routed_nodes_split": str(refine_request.get("routed_nodes_split", "all") or "all"),
            "knn_k": int(refine_request.get("knn_k", 8)),
            "candidate_scope": str(refine_request.get("candidate_scope", "undirected_relation_1hop")),
            "proxy_expansion": "dynamic_hypergraph_branch",
            "feature_source": str(refine_request.get("feature_source", "relation_node_repr_plus_g0_input_dynamic_forward")),
            "backend": "dynamic_relation_overlap_hypergraph_branch",
            "similarity_metric": str(refine_request.get("similarity_metric", "cosine")),
            "relation_cardinality_before": int(_relation_cardinality_from_edge_type(edge_type, minimum=0)),
            "relation_cardinality_after": int(_relation_cardinality_from_edge_type(edge_type, minimum=0)),
            "mean_neighbor_distance": None,
            "mean_relation_candidates_per_center": float(dynamic_similarity_branch["mean_relation_candidates_per_center"]),
            "centers_with_relation_neighbors": int(dynamic_similarity_branch["centers_with_relation_neighbors"]),
            "training_time_dynamic": True,
            "hypergraph_branch_type": "routed_local_knn_overlap",
            "second_view_scope": str(refine_request.get("second_view_scope", "routed_nodes")),
            "hypergraph_backend": str(refine_request.get("hypergraph_backend", "pyg")),
            "fusion": str(refine_request.get("fusion", "residual")),
            "training_geometry": str(refine_request.get("training_geometry", getattr(args, "graph_training_loader_mode", "full_batch"))),
        }
    elif refine_request["mode"] == "hyperscan_neighborloader_batch_local_branch":
        if str(graph_data_variant).lower() != "labeled":
            raise ValueError(
                "hyperscan_neighborloader_batch_local_branch currently requires --graph_data_variant labeled "
                "to match the original HyperScan labeled-graph regime."
            )
        dynamic_similarity_branch = {
            "center_node_ids": [],
            "center_candidate_node_ids": [],
            "center_source": "neighborloader_seed_batch_local_knn",
            "center_count": int(labeled_node_count),
            "centers_with_relation_neighbors": 0,
            "mean_relation_candidates_per_center": 0.0,
        }
        dynamic_similarity_branch_stats = {
            "mode": "hyperscan_neighborloader_batch_local_branch",
            "budget_requested": 0.0,
            "num_nodes": int(graph_node_count),
            "num_edges_before": int(edge_index.shape[1]) if hasattr(edge_index, "shape") else int(np.asarray(edge_index).shape[1]),
            "num_edges_after": int(edge_index.shape[1]) if hasattr(edge_index, "shape") else int(np.asarray(edge_index).shape[1]),
            "num_edges_added": 0,
            "num_hyperedges": int(labeled_node_count),
            "center_count": int(labeled_node_count),
            "center_source": "neighborloader_seed_batch_local_knn",
            "routed_nodes_path": "",
            "routed_nodes_split": "all",
            "knn_k": int(refine_request.get("knn_k", 8)),
            "candidate_scope": "batch_local_subgraph_knn",
            "proxy_expansion": "dynamic_hypergraph_branch",
            "feature_source": str(refine_request.get("feature_source", "relation_node_repr_plus_g0_input_dynamic_forward")),
            "backend": "neighborloader_batch_local_hypergraph_branch",
            "similarity_metric": str(refine_request.get("similarity_metric", "cosine")),
            "relation_cardinality_before": int(_relation_cardinality_from_edge_type(edge_type, minimum=0)),
            "relation_cardinality_after": int(_relation_cardinality_from_edge_type(edge_type, minimum=0)),
            "mean_neighbor_distance": None,
            "mean_relation_candidates_per_center": 0.0,
            "centers_with_relation_neighbors": 0,
            "training_time_dynamic": True,
            "hypergraph_branch_type": "neighborloader_batch_local_knn",
            "training_loader_mode": "neighbor_subgraph",
            "neighbor_num_neighbors": int(refine_request.get("neighbor_num_neighbors", 64)),
            "second_view_scope": str(refine_request.get("second_view_scope", "neighborloader_batch")),
            "hypergraph_backend": str(refine_request.get("hypergraph_backend", "pyg")),
            "fusion": str(refine_request.get("fusion", "residual")),
            "training_geometry": str(refine_request.get("training_geometry", "neighbor_subgraph")),
        }
    elif refine_request["mode"] != "none":
        edge_index, edge_type, graph_refine_stats = _apply_graph_refinement(
            edge_index=edge_index,
            edge_type=edge_type,
            raw_features=raw_features,
            projected_features=features,
            budget=refine_request["budget"],
            mode=refine_request["mode"],
            refine_request=refine_request,
            seed=seed,
            labels=labels,
        )
        write_torch(out_dir / "pruned_edge_index.pt", edge_index)
        write_torch(out_dir / "pruned_edge_type.pt", edge_type)
    else:
        for stale_path in (
            out_dir / "pruned_edge_index.pt",
            out_dir / "pruned_edge_type.pt",
            out_dir / "graph_refine_stats.json",
            out_dir / "external_edge_index.pt",
            out_dir / "external_edge_type.pt",
        ):
            if stale_path.exists():
                stale_path.unlink()
    if graph_override_manifest is not None:
        write_torch(out_dir / "external_edge_index.pt", edge_index)
        write_torch(out_dir / "external_edge_type.pt", edge_type)
    active_relation_cardinality = _relation_cardinality_from_edge_type(
        edge_type,
        minimum=max(int(getattr(args, "n_relations", 2)), 1),
    )
    if graph_refine_stats is not None:
        graph_refine_stats["model_n_relations_after"] = int(active_relation_cardinality)
        write_json(out_dir / "graph_refine_stats.json", graph_refine_stats)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    train_idx = train_idx_cpu.to(device)
    valid_idx = valid_idx_cpu.to(device)

    config = _model_config(args, features, device)
    config["n_relations"] = int(active_relation_cardinality)
    if str(feature_manifest.get("node_input_family", "semantic_embedding")).lower() == "hyperscan_meta_tweet_proxy":
        config["tweet_dim"] = int(feature_manifest.get("tweet_dim", 0))
        config["num_prop_dim"] = int(feature_manifest.get("num_prop_dim", 0))
        config["cat_prop_dim"] = int(feature_manifest.get("cat_prop_dim", 0))
        config["node_input_family"] = "hyperscan_meta_tweet_proxy"
    if dynamic_similarity_branch is not None:
        config["dynamic_similarity_branch"] = {
            "enabled": True,
            "knn_k": int(refine_request.get("knn_k", 8)),
            "candidate_scope": str(refine_request.get("candidate_scope", "undirected_relation_1hop")),
            "similarity_metric": str(refine_request.get("similarity_metric", "cosine")),
            "feature_source": str(refine_request.get("feature_source", "relation_node_repr_plus_g0_input_dynamic_forward")),
            "center_source": str(dynamic_similarity_branch["center_source"]),
            "routed_nodes_path": str(refine_request.get("routed_nodes_path", "") or ""),
            "routed_nodes_split": str(refine_request.get("routed_nodes_split", "all") or "all"),
            "center_node_ids": list(dynamic_similarity_branch["center_node_ids"]),
            "center_candidate_node_ids": list(dynamic_similarity_branch["center_candidate_node_ids"]),
            "center_count": int(dynamic_similarity_branch["center_count"]),
            "centers_with_relation_neighbors": int(dynamic_similarity_branch["centers_with_relation_neighbors"]),
            "mean_relation_candidates_per_center": float(dynamic_similarity_branch["mean_relation_candidates_per_center"]),
            "batch_local_knn": bool(refine_request["mode"] == "hyperscan_neighborloader_batch_local_branch"),
            "second_view_scope": str(refine_request.get("second_view_scope", "")),
            "hypergraph_backend": str(refine_request.get("hypergraph_backend", "pyg")),
            "fusion": str(refine_request.get("fusion", "residual")),
            "training_geometry": str(refine_request.get("training_geometry", getattr(args, "graph_training_loader_mode", "full_batch"))),
        }
    max_epochs = int(getattr(args, "g0_epochs", 0) or getattr(args, "GNN_epochs_per_iter", 1))
    mhlgc_enabled = bool(getattr(args, "mhlgc_enable", False))
    mhlgc_semantic_embeddings = None
    if mhlgc_enabled:
        semantic_path = str(getattr(args, "mhlgc_semantic_embedding_path", "") or "")
        if not semantic_path:
            raise ValueError("--mhlgc_enable requires --mhlgc_semantic_embedding_path for LLM-guided contrastive learning.")
        mhlgc_semantic_embeddings = _load_mhlgc_semantic_embeddings(
            path=semantic_path,
            expected_rows=graph_node_count,
        )
    training_result = _train_graph_backbone_once(
        config=config,
        x_projected=x_projected,
        x_raw=x_raw,
        y=y,
        edge_index=edge_index,
        edge_type=edge_type,
        train_idx=train_idx,
        valid_idx=valid_idx,
        peft_enabled=bool(getattr(args, "peft", False)),
        peft_rank=int(getattr(args, "peft_rank", 8)),
        peft_alpha=float(getattr(args, "peft_alpha", 16.0)),
        raw_feature_dim=int(raw_features.shape[1]),
        learning_rate=float(getattr(args, "lr_GNN", 5e-4)),
        weight_decay=float(getattr(args, "weight_decay_GNN", 1e-5)),
        max_epochs=max_epochs,
        training_loader_mode=str(
            refine_request.get(
                "training_loader_mode",
                getattr(args, "graph_training_loader_mode", "full_batch"),
            )
        ),
        graph_batch_size=int(getattr(args, "batch_size_GNN", 1024)),
        neighbor_num_neighbors=int(refine_request.get("neighbor_num_neighbors", getattr(args, "graph_neighbor_num_neighbors", 64))),
        max_update_steps=int(getattr(args, "graph_training_max_steps", 0) or 0),
        mhlgc_enabled=mhlgc_enabled,
        mhlgc_semantic_embeddings=mhlgc_semantic_embeddings,
        mhlgc_loss_weight=float(getattr(args, "mhlgc_loss_weight", 0.0) or 0.0),
        mhlgc_beta=float(getattr(args, "mhlgc_beta", 1.0) or 1.0),
        mhlgc_gamma=float(getattr(args, "mhlgc_gamma", 0.5) or 0.5),
        mhlgc_temperature=float(getattr(args, "mhlgc_temperature", 1.0) or 1.0),
        mhlgc_feature_mask_probability=float(getattr(args, "mhlgc_feature_mask_probability", 0.15) or 0.0),
        mhlgc_edge_mask_probability=float(getattr(args, "mhlgc_edge_mask_probability", 0.10) or 0.0),
        mhlgc_anchors_per_batch=int(getattr(args, "mhlgc_anchors_per_batch", 1) or 1),
        mhlgc_positive_label=int(getattr(args, "mhlgc_positive_label", 1) or 1),
    )
    model = training_result["model"]
    input_adapter = training_result["input_adapter"]
    outputs = training_result["outputs"]
    best_metrics = training_result["best_metrics"]
    best_state = training_result["best_state"]
    if input_adapter is not None:
        feature_manifest["peft"]["trainable_parameter_count"] = count_trainable_parameters(input_adapter)
    outputs = {
        "logits": outputs["logits"].detach().cpu(),
        "prob": outputs["prob"].detach().cpu(),
        "pred": outputs["prob"].argmax(dim=1).detach().cpu(),
        "labels": data["labels"].detach().cpu() if torch.is_tensor(data["labels"]) else torch.tensor(data["labels"]),
        "node_repr": outputs["node_repr"].detach().cpu(),
        "aux_features": outputs.get("aux_features", {}),
    }
    if dynamic_similarity_branch_stats is not None:
        runtime_branch_stats = outputs["aux_features"].get("dynamic_similarity_branch", {})
        if isinstance(runtime_branch_stats, dict):
            graph_refine_stats = {**dynamic_similarity_branch_stats, **runtime_branch_stats}
        else:
            graph_refine_stats = dict(dynamic_similarity_branch_stats)
        graph_refine_stats["model_n_relations_after"] = int(active_relation_cardinality)
        write_json(out_dir / "graph_refine_stats.json", graph_refine_stats)

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
        "mhlgc": training_result.get("mhlgc_stats", {"enabled": False}),
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
            "detector": {
                "hyperscan_detector_style": str(getattr(args, "hyperscan_detector_style", "residual") or "residual"),
                "graph_second_view_fusion": str(getattr(args, "graph_second_view_fusion", "residual") or "residual"),
                "graph_second_view_hypergraph_backend": str(
                    getattr(args, "graph_second_view_hypergraph_backend", "pyg") or "pyg"
                ),
            },
            "second_view": {
                "scope": str(refine_request.get("second_view_scope", getattr(args, "graph_second_view_scope", "auto"))),
                "candidate_scope": str(
                    refine_request.get(
                        "candidate_scope",
                        getattr(args, "graph_refine_candidate_scope", "undirected_relation_1hop"),
                    )
                    or "undirected_relation_1hop"
                ),
                "training_geometry": str(
                    refine_request.get(
                        "training_geometry",
                        refine_request.get(
                            "training_loader_mode",
                            getattr(args, "graph_training_loader_mode", "full_batch"),
                        ),
                    )
                ),
                "hypergraph_backend": str(getattr(args, "graph_second_view_hypergraph_backend", "pyg") or "pyg"),
                "fusion": str(getattr(args, "graph_second_view_fusion", "residual") or "residual"),
            },
            "training_scope": "train_supervised_plus_mhlgc" if mhlgc_enabled else "train_only_supervised",
            "pseudo_label_policy": "disabled_by_default",
            "checkpoint_selection": {
                "primary": "validation_macro_f1",
                "tie_breaker": "validation_loss",
            },
            "training_loader": {
                "mode": str(
                    refine_request.get(
                        "training_loader_mode",
                        getattr(args, "graph_training_loader_mode", "full_batch"),
                    )
                ),
                "neighbor_num_neighbors": int(refine_request.get("neighbor_num_neighbors", getattr(args, "graph_neighbor_num_neighbors", 64))),
                "graph_batch_size": int(getattr(args, "batch_size_GNN", 1024)),
                "optimizer_steps": int(training_result.get("optimizer_steps", 0)),
                "max_update_steps": int(getattr(args, "graph_training_max_steps", 0) or 0),
            },
            "mhlgc": {
                **training_result.get("mhlgc_stats", {"enabled": False}),
                "semantic_embedding_path": str(getattr(args, "mhlgc_semantic_embedding_path", "") or ""),
                "beta": float(getattr(args, "mhlgc_beta", 1.0) or 1.0),
                "gamma": float(getattr(args, "mhlgc_gamma", 0.5) or 0.5),
                "temperature": float(getattr(args, "mhlgc_temperature", 1.0) or 1.0),
                "feature_mask_probability": float(getattr(args, "mhlgc_feature_mask_probability", 0.15) or 0.0),
                "edge_mask_probability": float(getattr(args, "mhlgc_edge_mask_probability", 0.10) or 0.0),
                "anchors_per_batch": int(getattr(args, "mhlgc_anchors_per_batch", 1) or 1),
                "positive_label": int(getattr(args, "mhlgc_positive_label", 1) or 1),
                "paper_alignment": (
                    "LLM-as-guide semantic hard-negative weighting over original and augmented graph views; "
                    "LLM is not used as predictor."
                ),
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
            "graph_override": graph_override_manifest if graph_override_manifest is not None else {"mode": "none"},
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


def _coerce_node_id_sequence(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(item) for item in value.detach().cpu().view(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_coerce_node_id_sequence(item))
        return out
    if isinstance(value, dict):
        for key in ("node_ids", "nodes", "indices", "target_node_ids", "routed_node_ids", "routed_nodes", "items", "rows"):
            if key in value:
                return _coerce_node_id_sequence(value[key])
        for key in ("node_id", "index", "id"):
            if key in value:
                return [int(value[key])]
        if len(value) == 1:
            return _coerce_node_id_sequence(next(iter(value.values())))
        raise ValueError(f"Could not infer node ids from mapping keys: {sorted(value.keys())}")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        return [int(stripped)]
    return [int(value)]


def _select_routed_node_split(raw, split_name="all"):
    split = str(split_name or "all").strip().lower()
    if isinstance(raw, dict):
        for wrapper_key in ("routed_nodes", "splits", "split_nodes", "selected_nodes"):
            nested = raw.get(wrapper_key)
            if isinstance(nested, dict):
                try:
                    return _select_routed_node_split(nested, split_name=split)
                except ValueError:
                    pass
    if split in {"", "all", "union", "target", "targets"}:
        if isinstance(raw, dict):
            for key in ("total", "all", "union", "target_node_ids", "routed_node_ids", "routed_nodes", "nodes"):
                if key in raw and not isinstance(raw.get(key), dict):
                    return raw[key]
            combined = []
            for key in ("train", "valid", "validation", "val", "test"):
                if key in raw:
                    combined.extend(_coerce_node_id_sequence(raw[key]))
            if combined:
                return combined
        return raw
    if not isinstance(raw, dict):
        raise ValueError("--routed_nodes_split requires a mapping-style routed node file.")
    aliases = {
        "train": ("train", "train_nodes", "train_node_ids", "routed_train", "train_routed_nodes"),
        "valid": ("valid", "validation", "val", "valid_nodes", "validation_nodes", "valid_node_ids", "routed_valid"),
        "val": ("valid", "validation", "val", "valid_nodes", "validation_nodes", "valid_node_ids", "routed_valid"),
        "test": ("test", "test_nodes", "test_node_ids", "routed_test", "test_routed_nodes"),
    }
    keys = aliases.get(split)
    if keys is None:
        raise ValueError(f"Unsupported --routed_nodes_split {split_name!r}; expected all/train/valid/test.")
    for key in keys:
        if key in raw:
            return raw[key]
    if "node_ids" in raw and isinstance(raw.get("split_counts"), dict):
        ordered_split_names = ("train", "valid", "test")
        counts = {str(key).lower(): int(value) for key, value in raw["split_counts"].items()}
        if split == "val":
            split = "valid"
        if split in counts:
            start = 0
            for name in ordered_split_names:
                count = int(counts.get(name, 0))
                end = start + count
                if name == split:
                    node_ids = _coerce_node_id_sequence(raw["node_ids"])
                    return node_ids[start:end]
                start = end
    raise ValueError(
        f"Could not find split {split_name!r} in routed node mapping. Available keys: {sorted(raw.keys())}"
    )


def _load_routed_center_node_ids(path: Path, split_name="all"):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = read_json(path, default=None)
        values = _coerce_node_id_sequence(_select_routed_node_split(raw, split_name=split_name))
    elif suffix == ".pt":
        raw = safe_torch_load(path, map_location="cpu")
        values = _coerce_node_id_sequence(_select_routed_node_split(raw, split_name=split_name))
    else:
        raise ValueError("Graph refine routed center selection currently supports only .json or .pt routed-node files.")
    seen = set()
    ordered = []
    for item in values:
        node_id = int(item)
        if node_id in seen:
            continue
        seen.add(node_id)
        ordered.append(node_id)
    return ordered


def _resolve_refine_center_node_ids(refine_request, num_nodes, labeled_node_count):
    routed_nodes_path = str(refine_request.get("routed_nodes_path", "") or "").strip()
    routed_nodes_split = str(refine_request.get("routed_nodes_split", "all") or "all").strip().lower()
    if routed_nodes_path:
        center_node_ids = _load_routed_center_node_ids(Path(routed_nodes_path), split_name=routed_nodes_split)
        if not center_node_ids:
            raise ValueError(
                f"--routed_nodes_path did not yield any center nodes for graph refinement split={routed_nodes_split}: "
                f"{routed_nodes_path}"
            )
        invalid = [int(item) for item in center_node_ids if int(item) < 0 or int(item) >= int(num_nodes)]
        if invalid:
            preview = invalid[:10]
            raise ValueError(
                f"Graph refine center nodes from --routed_nodes_path fall outside [0, {int(num_nodes) - 1}]: "
                f"{preview}{' ...' if len(invalid) > len(preview) else ''}"
            )
        return np.asarray(center_node_ids, dtype=np.int64), f"routed_nodes_path:{routed_nodes_split}"
    default_count = int(labeled_node_count) if int(labeled_node_count) > 0 else int(num_nodes)
    return np.arange(default_count, dtype=np.int64), "labeled_prefix"


def _relation_overlap_center_candidates(edge_index, edge_type, num_nodes, refine_request, labeled_node_count):
    center_node_ids, center_source = _resolve_refine_center_node_ids(
        refine_request=refine_request,
        num_nodes=int(num_nodes),
        labeled_node_count=int(labeled_node_count),
    )
    return _shared_build_relation_overlap_center_candidates(
        edge_index=edge_index,
        num_nodes=int(num_nodes),
        center_node_ids=center_node_ids.tolist(),
        center_source=center_source,
        candidate_scope=str(refine_request.get("candidate_scope", "undirected_relation_1hop")),
        labeled_node_count=int(labeled_node_count),
    )


def _graph_refine_request(args):
    mode = str(getattr(args, "graph_refine_mode", "none") or "none").lower()
    budget = float(getattr(args, "graph_refine_budget", 0.0) or 0.0)
    candidate_scope = str(getattr(args, "graph_refine_candidate_scope", "undirected_relation_1hop") or "undirected_relation_1hop").strip().lower()
    hypergraph_backend = str(getattr(args, "graph_second_view_hypergraph_backend", "pyg") or "pyg").strip().lower()
    fusion = str(getattr(args, "graph_second_view_fusion", "residual") or "residual").strip().lower()
    training_geometry = str(getattr(args, "graph_training_loader_mode", "full_batch") or "full_batch").strip().lower()
    if candidate_scope not in {"undirected_relation_1hop", "labeled_relation_1hop"}:
        raise ValueError(
            "--graph_refine_candidate_scope must be one of "
            "{undirected_relation_1hop, labeled_relation_1hop}."
        )
    if hypergraph_backend not in {"pyg", "dhg"}:
        raise ValueError("--graph_second_view_hypergraph_backend must be one of {pyg, dhg}.")
    if fusion not in {"residual", "multiattn"}:
        raise ValueError("--graph_second_view_fusion must be one of {residual, multiattn}.")
    if training_geometry not in {"full_batch", "neighbor_subgraph"}:
        raise ValueError("--graph_training_loader_mode must be one of {full_batch, neighbor_subgraph}.")
    if mode == "none":
        return {
            "mode": "none",
            "budget": 0.0,
            "degree_guard_enabled": False,
            "oracle_uses_labels": False,
            "diagnostic_only": False,
        }
    if mode == "hyperscan_knn_hypergraph_proxy_augment":
        knn_k = int(getattr(args, "graph_refine_knn_k", 8) or 8)
        if knn_k < 2:
            raise ValueError("--graph_refine_knn_k must be >= 2 for hyperscan_knn_hypergraph_proxy_augment.")
        return {
            "mode": mode,
            "budget": 0.0,
            "degree_guard_enabled": False,
            "oracle_uses_labels": False,
            "diagnostic_only": False,
            "knn_k": int(knn_k),
            "proxy_expansion": "bidirectional_center_star",
        }
    if mode == "relation_overlap_knn_proxy_augment":
        knn_k = int(getattr(args, "graph_refine_knn_k", 8) or 8)
        if knn_k < 1:
            raise ValueError("--graph_refine_knn_k must be >= 1 for relation_overlap_knn_proxy_augment.")
        return {
            "mode": mode,
            "budget": 0.0,
            "degree_guard_enabled": False,
            "oracle_uses_labels": False,
            "diagnostic_only": False,
            "knn_k": int(knn_k),
            "proxy_expansion": "bidirectional_center_star",
            "candidate_scope": candidate_scope,
            "similarity_metric": "cosine",
            "routed_nodes_path": str(getattr(args, "routed_nodes_path", "") or ""),
            "routed_nodes_split": str(getattr(args, "routed_nodes_split", "all") or "all"),
        }
    if mode == "relation_overlap_knn_repr_prefit_augment":
        knn_k = int(getattr(args, "graph_refine_knn_k", 8) or 8)
        if knn_k < 1:
            raise ValueError("--graph_refine_knn_k must be >= 1 for relation_overlap_knn_repr_prefit_augment.")
        return {
            "mode": mode,
            "budget": 0.0,
            "degree_guard_enabled": False,
            "oracle_uses_labels": False,
            "diagnostic_only": False,
            "knn_k": int(knn_k),
            "proxy_expansion": "bidirectional_center_star",
            "candidate_scope": candidate_scope,
            "similarity_metric": "cosine",
            "feature_source": "relation_prefit_node_repr_plus_g0_input",
            "prefit_max_epochs": int(getattr(args, "g0_epochs", 0) or getattr(args, "GNN_epochs_per_iter", 1) or 1),
            "routed_nodes_path": str(getattr(args, "routed_nodes_path", "") or ""),
            "routed_nodes_split": str(getattr(args, "routed_nodes_split", "all") or "all"),
        }
    if mode == "routed_dynamic_hyperscan_branch":
        knn_k = int(getattr(args, "graph_refine_knn_k", 8) or 8)
        if knn_k < 1:
            raise ValueError("--graph_refine_knn_k must be >= 1 for routed_dynamic_hyperscan_branch.")
        routed_nodes_path = str(getattr(args, "routed_nodes_path", "") or "")
        requested_second_view_scope = str(getattr(args, "graph_second_view_scope", "auto") or "auto").strip().lower()
        second_view_scope = (
            ("routed_nodes" if routed_nodes_path else "labeled_prefix")
            if requested_second_view_scope in {"", "auto"}
            else requested_second_view_scope
        )
        return {
            "mode": mode,
            "budget": 0.0,
            "degree_guard_enabled": False,
            "oracle_uses_labels": False,
            "diagnostic_only": False,
            "knn_k": int(knn_k),
            "proxy_expansion": "dynamic_hypergraph_branch",
            "candidate_scope": candidate_scope,
            "similarity_metric": "cosine",
            "feature_source": "relation_node_repr_plus_g0_input_dynamic_forward",
            "routed_nodes_path": routed_nodes_path,
            "routed_nodes_split": str(getattr(args, "routed_nodes_split", "all") or "all"),
            "second_view_scope": second_view_scope,
            "hypergraph_backend": hypergraph_backend,
            "fusion": fusion,
            "training_geometry": training_geometry,
        }
    if mode == "hyperscan_neighborloader_batch_local_branch":
        knn_k = int(getattr(args, "graph_refine_knn_k", 8) or 8)
        if knn_k < 1:
            raise ValueError("--graph_refine_knn_k must be >= 1 for hyperscan_neighborloader_batch_local_branch.")
        neighbor_num_neighbors = int(getattr(args, "graph_neighbor_num_neighbors", 64) or 64)
        if neighbor_num_neighbors < 1:
            raise ValueError("--graph_neighbor_num_neighbors must be >= 1 for hyperscan_neighborloader_batch_local_branch.")
        if str(getattr(args, "graph_second_view_training_geometry", "auto") or "auto").strip().lower() == "full_batch":
            raise ValueError(
                "--graph_second_view_scope neighborloader_batch requires "
                "--graph_second_view_training_geometry neighbor_subgraph or auto."
            )
        return {
            "mode": mode,
            "budget": 0.0,
            "degree_guard_enabled": False,
            "oracle_uses_labels": False,
            "diagnostic_only": False,
            "knn_k": int(knn_k),
            "proxy_expansion": "dynamic_hypergraph_branch",
            "candidate_scope": "batch_local_subgraph_knn",
            "similarity_metric": "cosine",
            "feature_source": "relation_node_repr_plus_g0_input_dynamic_forward",
            "training_loader_mode": "neighbor_subgraph",
            "neighbor_num_neighbors": int(neighbor_num_neighbors),
            "second_view_scope": "neighborloader_batch",
            "hypergraph_backend": hypergraph_backend,
            "fusion": fusion,
            "training_geometry": "neighbor_subgraph",
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


def _feature_k_nearest_neighbor_query(feature_tensor, k):
    return _shared_feature_k_nearest_neighbor_query(feature_tensor, k)


def _hyperscan_knn_hypergraph_proxy_augment(edge_index, edge_type, feature_tensor, k):
    return _shared_hyperscan_knn_hypergraph_proxy_augment(edge_index, edge_type, feature_tensor, k)


def _relation_overlap_knn_proxy_augment(edge_index, edge_type, feature_tensor, k, refine_request=None, labeled_node_count=None):
    refine_request = dict(refine_request or {})
    labeled_count = int(labeled_node_count) if labeled_node_count is not None else int(feature_tensor.shape[0])
    center_node_ids, center_source = _resolve_refine_center_node_ids(
        refine_request=refine_request,
        num_nodes=int(feature_tensor.shape[0]),
        labeled_node_count=labeled_count,
    )
    return _shared_relation_overlap_knn_proxy_augment(
        edge_index=edge_index,
        edge_type=edge_type,
        feature_tensor=feature_tensor,
        center_node_ids=center_node_ids.tolist(),
        center_source=center_source,
        k=int(k),
        refine_request={
            **refine_request,
            "labeled_node_count": int(labeled_count),
        },
    )


def _relation_overlap_knn_repr_prefit_augment(
    edge_index,
    edge_type,
    feature_tensor,
    raw_features,
    labels,
    args,
    train_idx,
    valid_idx,
    k,
    refine_request=None,
    labeled_node_count=None,
):
    refine_request = dict(refine_request or {})
    device = getattr(args, "device", torch.device("cpu"))
    if not isinstance(device, torch.device):
        device = _resolve_device(device)

    feature_cpu = (
        feature_tensor.detach().cpu().float()
        if torch.is_tensor(feature_tensor)
        else torch.tensor(feature_tensor, dtype=torch.float32)
    )
    raw_feature_cpu = (
        raw_features.detach().cpu().float()
        if torch.is_tensor(raw_features)
        else torch.tensor(raw_features, dtype=torch.float32)
    )
    labels_cpu = labels.detach().cpu().long() if torch.is_tensor(labels) else torch.tensor(labels, dtype=torch.long)
    edge_index_cpu = (
        edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
    )
    edge_type_cpu = (
        edge_type.detach().cpu().long() if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long)
    )
    train_idx_cpu = _as_long_cpu_tensor(train_idx)
    valid_idx_cpu = _as_long_cpu_tensor(valid_idx)

    config = _model_config(args, feature_cpu, device)
    config["n_relations"] = int(_relation_cardinality_from_edge_type(edge_type_cpu, minimum=max(int(getattr(args, "n_relations", 2)), 1)))

    prefit_result = _train_graph_backbone_once(
        config=config,
        x_projected=feature_cpu.to(device),
        x_raw=raw_feature_cpu.to(device),
        y=labels_cpu.to(device),
        edge_index=edge_index_cpu.to(device),
        edge_type=edge_type_cpu.to(device),
        train_idx=train_idx_cpu.to(device),
        valid_idx=valid_idx_cpu.to(device),
        peft_enabled=bool(getattr(args, "peft", False)),
        peft_rank=int(getattr(args, "peft_rank", 8)),
        peft_alpha=float(getattr(args, "peft_alpha", 16.0)),
        raw_feature_dim=int(raw_feature_cpu.shape[1]),
        learning_rate=float(getattr(args, "lr_GNN", 5e-4)),
        weight_decay=float(getattr(args, "weight_decay_GNN", 1e-5)),
        max_epochs=int(refine_request.get("prefit_max_epochs", 1)),
    )
    prefit_repr = prefit_result["outputs"]["node_repr"].detach().cpu().float()
    hybrid_feature = torch.cat([prefit_repr, feature_cpu], dim=1)

    augmented_edge_index, augmented_edge_type, stats = _relation_overlap_knn_proxy_augment(
        edge_index=edge_index_cpu,
        edge_type=edge_type_cpu,
        feature_tensor=hybrid_feature,
        k=k,
        refine_request={
            **refine_request,
            "feature_source": "relation_prefit_node_repr_plus_g0_input",
            "similarity_metric": str(refine_request.get("similarity_metric", "cosine")),
        },
        labeled_node_count=labeled_node_count,
    )
    stats["backend"] = "local_relation_overlap_topk_relation_prefit"
    stats["prefit_feature_dim"] = int(prefit_repr.shape[1])
    stats["prefit_concat_dim"] = int(hybrid_feature.shape[1])
    stats["prefit_val_macro_f1"] = float(prefit_result["best_metrics"].get("macro_f1", 0.0))
    stats["prefit_val_accuracy"] = float(prefit_result["best_metrics"].get("accuracy", 0.0))
    stats["prefit_max_epochs"] = int(refine_request.get("prefit_max_epochs", 1))
    stats["feature_source"] = "relation_prefit_node_repr_plus_g0_input"
    return augmented_edge_index, augmented_edge_type, stats


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


def _apply_graph_refinement(edge_index, edge_type, raw_features, projected_features, budget, mode, refine_request=None, seed=None, labels=None):
    refine_request = dict(refine_request or {})
    if mode == "hyperscan_knn_hypergraph_proxy_augment":
        knn_k = int(refine_request.get("knn_k", 8))
        return _hyperscan_knn_hypergraph_proxy_augment(
            edge_index=edge_index,
            edge_type=edge_type,
            feature_tensor=projected_features,
            k=knn_k,
        )
    if mode == "relation_overlap_knn_proxy_augment":
        knn_k = int(refine_request.get("knn_k", 8))
        labeled_node_count = int(labels.numel()) if labels is not None else int(projected_features.shape[0])
        return _relation_overlap_knn_proxy_augment(
            edge_index=edge_index,
            edge_type=edge_type,
            feature_tensor=projected_features,
            k=knn_k,
            refine_request=refine_request,
            labeled_node_count=labeled_node_count,
        )
    return _directional_relation_aware_prune(
        edge_index=edge_index,
        edge_type=edge_type,
        raw_features=raw_features,
        budget=budget,
        mode=mode,
        seed=seed,
        labels=labels,
    )


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


def _graph_override_manifest_request(args):
    graph_override = _resolve_optional_graph_override_paths(args)
    if graph_override is None:
        return None
    return {
        "edge_index_path": str(graph_override["edge_index_path"]),
        "edge_type_path": str(graph_override["edge_type_path"]),
    }


def _mhlgc_manifest_request(args):
    enabled = bool(getattr(args, "mhlgc_enable", False))
    return {
        "enabled": enabled,
        "semantic_embedding_path": str(getattr(args, "mhlgc_semantic_embedding_path", "") or "") if enabled else "",
        "loss_weight": float(getattr(args, "mhlgc_loss_weight", 0.0) or 0.0),
        "beta": float(getattr(args, "mhlgc_beta", 1.0) or 1.0),
        "gamma": float(getattr(args, "mhlgc_gamma", 0.5) or 0.5),
        "temperature": float(getattr(args, "mhlgc_temperature", 1.0) or 1.0),
        "feature_mask_probability": float(getattr(args, "mhlgc_feature_mask_probability", 0.15) or 0.0),
        "edge_mask_probability": float(getattr(args, "mhlgc_edge_mask_probability", 0.10) or 0.0),
        "anchors_per_batch": int(getattr(args, "mhlgc_anchors_per_batch", 1) or 1),
        "positive_label": int(getattr(args, "mhlgc_positive_label", 1) or 1),
    }


def _mhlgc_manifest_matches_request(args, manifest):
    requested = _mhlgc_manifest_request(args)
    existing = manifest.get("mhlgc", {})
    if not isinstance(existing, dict):
        existing = {}
    if bool(existing.get("enabled", False)) != bool(requested["enabled"]):
        return False
    if not requested["enabled"]:
        return True
    if str(existing.get("semantic_embedding_path", "") or "") != requested["semantic_embedding_path"]:
        return False
    for key in ("loss_weight", "beta", "gamma", "temperature", "feature_mask_probability", "edge_mask_probability"):
        if not math.isclose(
            float(existing.get(key, -1.0)),
            float(requested[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False
    for key in ("anchors_per_batch", "positive_label"):
        if int(existing.get(key, -1)) != int(requested[key]):
            return False
    return True


def _detector_manifest_matches_request(args, manifest):
    requested_detector_style = str(getattr(args, "hyperscan_detector_style", "residual") or "residual").lower()
    existing_detector = manifest.get("detector", {})
    if not isinstance(existing_detector, dict):
        existing_detector = {}
    existing_detector_style = str(existing_detector.get("hyperscan_detector_style", "residual") or "residual").lower()
    if existing_detector_style != requested_detector_style:
        return False
    requested_backend = str(getattr(args, "graph_second_view_hypergraph_backend", "pyg") or "pyg").lower()
    existing_second_view = manifest.get("second_view", {})
    if not isinstance(existing_second_view, dict):
        existing_second_view = {}
    existing_backend = str(
        existing_detector.get("graph_second_view_hypergraph_backend", "")
        or existing_second_view.get("hypergraph_backend", "")
        or "pyg"
    ).lower()
    if existing_backend != requested_backend:
        return False
    requested_fusion = str(getattr(args, "graph_second_view_fusion", "residual") or "residual").lower()
    existing_fusion = str(
        existing_detector.get("graph_second_view_fusion", "")
        or existing_second_view.get("fusion", "")
        or ("multiattn" if existing_detector_style == "original_cross_attention" else "residual")
    ).lower()
    return existing_fusion == requested_fusion


def _existing_g0_matches_request(args, manifest):
    if manifest.get("contract") != FROZEN_G0_CONTRACT:
        return False
    if manifest.get("backbone", "").lower() != str(getattr(args, "GNN_model", "rgcn")).lower():
        return False
    if not _mhlgc_manifest_matches_request(args, manifest):
        return False

    requested_graph_override = _graph_override_manifest_request(args)
    existing_graph_override = manifest.get("graph_override")
    if requested_graph_override is None:
        if existing_graph_override not in (None, {}, {"mode": "none"}):
            return False
    else:
        if not isinstance(existing_graph_override, dict):
            return False
        if str(existing_graph_override.get("edge_index_path", "")) != requested_graph_override["edge_index_path"]:
            return False
        if str(existing_graph_override.get("edge_type_path", "")) != requested_graph_override["edge_type_path"]:
            return False

    requested_path = _requested_feature_path(args)
    feature_manifest = manifest.get("feature_manifest", {})
    if requested_path is not None and feature_manifest.get("path") != requested_path:
        return False
    projected_dim = feature_manifest.get("projected_dim")
    if projected_dim is not None and int(projected_dim) != phase_a_project_dim(args):
        return False
    projector_name = str(feature_manifest.get("projector", "") or "").lower()
    if projector_name and projector_name != phase_a_projector(args):
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
        return refine_manifest in (None, {}, {"mode": "none"}) and _detector_manifest_matches_request(args, manifest)
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
    if refine_request["mode"] in {
        "hyperscan_knn_hypergraph_proxy_augment",
        "relation_overlap_knn_proxy_augment",
        "relation_overlap_knn_repr_prefit_augment",
        "routed_dynamic_hyperscan_branch",
        "hyperscan_neighborloader_batch_local_branch",
    }:
        if int(refine_manifest.get("knn_k", -1)) != int(refine_request.get("knn_k", -1)):
            return False
        if str(refine_manifest.get("proxy_expansion", "")).lower() != str(refine_request.get("proxy_expansion", "")).lower():
            return False
    if refine_request["mode"] in {
        "relation_overlap_knn_proxy_augment",
        "relation_overlap_knn_repr_prefit_augment",
        "routed_dynamic_hyperscan_branch",
    }:
        if str(refine_manifest.get("candidate_scope", "")).lower() != str(refine_request.get("candidate_scope", "")).lower():
            return False
        if str(refine_manifest.get("similarity_metric", "")).lower() != str(refine_request.get("similarity_metric", "")).lower():
            return False
        if str(refine_manifest.get("routed_nodes_path", "") or "") != str(refine_request.get("routed_nodes_path", "") or ""):
            return False
        if str(refine_manifest.get("routed_nodes_split", "")).lower() != str(refine_request.get("routed_nodes_split", "all")).lower():
            return False
    if refine_request["mode"] == "relation_overlap_knn_repr_prefit_augment":
        if str(refine_manifest.get("feature_source", "")).lower() != str(refine_request.get("feature_source", "")).lower():
            return False
        if int(refine_manifest.get("prefit_max_epochs", -1)) != int(refine_request.get("prefit_max_epochs", -1)):
            return False
    if refine_request["mode"] == "routed_dynamic_hyperscan_branch":
        if str(refine_manifest.get("feature_source", "")).lower() != str(refine_request.get("feature_source", "")).lower():
            return False
        if bool(refine_manifest.get("training_time_dynamic", False)) is not True:
            return False
    if refine_request["mode"] == "hyperscan_neighborloader_batch_local_branch":
        if str(refine_manifest.get("feature_source", "")).lower() != str(refine_request.get("feature_source", "")).lower():
            return False
        if str(refine_manifest.get("training_loader_mode", "")).lower() != str(refine_request.get("training_loader_mode", "")).lower():
            return False
        if int(refine_manifest.get("neighbor_num_neighbors", -1)) != int(refine_request.get("neighbor_num_neighbors", -1)):
            return False
    if not _detector_manifest_matches_request(args, manifest):
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
