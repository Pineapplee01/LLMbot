import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from artifact_contracts import MissingFrozenArtifactError, frozen_g0_dir, split_provenance
from GNNs import (
    routed_bot_edge_mask_human_contrastive_loss,
    routed_frozen_alignment_loss,
    routed_same_node_contrastive_loss,
    routed_supervised_contrastive_loss,
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
    _build_hyperscan_meta_tweet_proxy_bundle,
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


def _load_dualspace_construct_bundle(args, data):
    requested_backbone = str(getattr(args, "GNN_model", getattr(args, "graph_backbone", "rgcn"))).lower()
    if "rgcn_h2fag_dualspace" not in requested_backbone:
        return None
    if "hyperscan" not in requested_backbone:
        return None

    graph_node_input_family = str(getattr(args, "graph_node_input_family", "semantic_embedding") or "semantic_embedding").lower()
    construct_path = str(getattr(args, "graph_construct_embedding_path", "") or "").strip()

    if "nodeinput" in requested_backbone:
        if graph_node_input_family != "hyperscan_meta_tweet_proxy":
            raise ValueError(
                "dualspace *_nodeinput backbones require --graph_node_input_family hyperscan_meta_tweet_proxy."
            )
        if not construct_path:
            raise ValueError(
                "dualspace *_nodeinput hyperscan backbones require --graph_construct_embedding_path "
                "pointing to the clean construct-side tweet or tweet|num|cat tensor."
            )
        lowered_construct_path = construct_path.lower()
        if any(token in lowered_construct_path for token in ["iter_-1", "iter_minus1", "node_repr"]):
            raise ValueError(
                "dualspace *_nodeinput hyperscan construct space must not use iter_-1 decision embeddings "
                "or final detector node_repr artifacts."
            )
        construct_path_obj = Path(construct_path)
        sidecar_manifest_path = construct_path_obj.with_suffix(".manifest.json")
        sidecar_manifest = read_json(sidecar_manifest_path, default=None) if sidecar_manifest_path.exists() else None
        sidecar_hint_text = " ".join(
            str(item or "")
            for item in (
                (sidecar_manifest or {}).get("contract"),
                (sidecar_manifest or {}).get("artifact"),
                (sidecar_manifest or {}).get("source_path"),
                (sidecar_manifest or {}).get("output_path"),
                construct_path_obj.name,
            )
        ).lower()
        precomputed_nodeinput = (
            "tweet_num_cat" in sidecar_hint_text
            or "node_input" in sidecar_hint_text
            or str((sidecar_manifest or {}).get("contract", "")).strip().lower()
            == "hyperscan_meta_tweet_proxy_full_graph_node_input"
        )

        construct_tensor = safe_torch_load(construct_path_obj, map_location="cpu")
        if isinstance(construct_tensor, dict):
            for key in ("embeddings", "features", "x"):
                if key in construct_tensor:
                    construct_tensor = construct_tensor[key]
                    break
        if not torch.is_tensor(construct_tensor):
            construct_tensor = torch.as_tensor(construct_tensor, dtype=torch.float32)
        construct_tensor = construct_tensor.detach().cpu().float()
        graph_variant = str(data.get("graph_data_variant", getattr(args, "graph_data_variant", "labeled"))).lower()
        if int(construct_tensor.dim()) != 2:
            raise ValueError(
                "dualspace *_nodeinput hyperscan construct tensor must be 2-D [num_nodes, dim]."
            )
        if graph_variant == "labeled":
            expected_rows = int(data.get("labeled_node_count", int(data.get("graph_node_count", int(construct_tensor.shape[0])))))
        else:
            expected_rows = int(data.get("graph_node_count", int(construct_tensor.shape[0])))
        if int(construct_tensor.shape[0]) != expected_rows:
            raise ValueError(
                "dualspace *_nodeinput hyperscan construct tensor row count must match the active graph view: "
                f"expected {expected_rows}, got {int(construct_tensor.shape[0])}."
            )
        if precomputed_nodeinput:
            if int(construct_tensor.shape[1]) <= 8:
                raise ValueError(
                    "dualspace *_nodeinput hyperscan precomputed construct tensors must be ordered as tweet|num|cat "
                    "with feature dim > 8."
                )
            tweet_dim = int(construct_tensor.shape[1]) - 8
            manifest_source = "hyperscan_meta_tweet_proxy_precomputed_construct"
            construct_features = construct_tensor.contiguous()
            manifest = {
                "source": manifest_source,
                "path": str(construct_path_obj),
                "tweet_embedding_path": str(construct_path_obj),
                "graph_data_variant": graph_variant,
                "node_input_family": "hyperscan_meta_tweet_proxy",
                "projection_applied": False,
                "projector": "node_input_proxy",
                "fit_scope": "all_nodes_unlabeled" if graph_variant == "full_graph_support" else "labeled_nodes_only",
                "raw_dim": int(construct_features.shape[1]),
                "projected_dim": int(construct_features.shape[1]),
                "raw_sha256": tensor_sha256(construct_features),
                "projected_sha256": tensor_sha256(construct_features),
                "sha256": tensor_sha256(construct_features),
                "projection_time_seconds": 0.0,
                "tweet_dim": int(tweet_dim),
                "num_prop_dim": 5,
                "cat_prop_dim": 3,
            }
            if isinstance(sidecar_manifest, dict):
                manifest["source_manifest_path"] = str(sidecar_manifest_path)
                manifest["sidecar_contract"] = str(sidecar_manifest.get("contract", "") or "")
        else:
            bundle = _build_hyperscan_meta_tweet_proxy_bundle(args, data, construct_path_obj)
            construct_features = bundle["raw_features"].detach().cpu().float()
            manifest = dict(bundle["feature_manifest"])
        manifest["dualspace_role"] = "hyperscan_clean_representation"
        manifest["construct_source"] = "graph_construct_embedding_path"
        manifest["path"] = str(construct_path_obj)
        manifest["tweet_embedding_path"] = str(construct_path_obj)
        return {"features": construct_features, "feature_manifest": manifest}

    if not construct_path:
        raise ValueError(
            "dualspace hyperscan backbones require --graph_construct_embedding_path pointing to the clean Hyperscan construct tensor."
        )
    construct_tensor = safe_torch_load(construct_path, map_location="cpu")
    if not torch.is_tensor(construct_tensor):
        construct_tensor = torch.as_tensor(construct_tensor, dtype=torch.float32)
    construct_tensor = construct_tensor.detach().cpu().float()
    graph_node_count = int(data.get("graph_node_count", int(construct_tensor.shape[0])))
    if int(construct_tensor.shape[0]) != graph_node_count:
        raise ValueError(
            f"dualspace construct tensor rows ({int(construct_tensor.shape[0])}) must match graph_node_count ({graph_node_count})."
        )
    manifest = {
        "source": "tensor_file",
        "path": str(construct_path),
        "dualspace_role": "hyperscan_clean_representation",
        "construct_source": "graph_construct_embedding_path",
        "graph_data_variant": str(data.get("graph_data_variant", getattr(args, "graph_data_variant", "labeled"))).lower(),
        "raw_dim": int(construct_tensor.shape[1]),
        "projected_dim": int(construct_tensor.shape[1]),
        "projection_applied": False,
        "projector": "construct_identity",
        "fit_scope": "all_nodes_unlabeled",
        "sha256": tensor_sha256(construct_tensor),
        "raw_sha256": tensor_sha256(construct_tensor),
        "projected_sha256": tensor_sha256(construct_tensor),
    }
    return {"features": construct_tensor, "feature_manifest": manifest}


def _is_better(candidate, incumbent):
    if incumbent is None:
        return True
    if candidate["macro_f1"] > incumbent["macro_f1"]:
        return True
    if candidate["macro_f1"] == incumbent["macro_f1"] and candidate["loss"] < incumbent["loss"]:
        return True
    return False


def _is_better_with_policy(candidate, incumbent, primary="validation_macro_f1", tie_breaker="validation_loss"):
    if incumbent is None:
        return True
    primary = str(primary or "validation_macro_f1").strip().lower()
    tie_breaker = str(tie_breaker or "validation_loss").strip().lower()
    if primary == "validation_accuracy":
        candidate_primary = float(candidate.get("accuracy", float("-inf")))
        incumbent_primary = float(incumbent.get("accuracy", float("-inf")))
    else:
        candidate_primary = float(candidate.get("macro_f1", float("-inf")))
        incumbent_primary = float(incumbent.get("macro_f1", float("-inf")))
    if candidate_primary > incumbent_primary:
        return True
    if candidate_primary < incumbent_primary:
        return False
    if tie_breaker == "validation_loss":
        return float(candidate.get("loss", float("inf"))) < float(incumbent.get("loss", float("inf")))
    return False


def _score_repeated_neighborloader_logits(logits_cpu, labels_cpu):
    logits_cpu = logits_cpu.detach().cpu()
    labels_cpu = labels_cpu.detach().cpu().long().view(-1)
    if int(logits_cpu.size(0)) != int(labels_cpu.numel()):
        raise ValueError("Repeated NeighborLoader logits and labels must have the same row count.")
    prob = torch.softmax(logits_cpu, dim=1)
    pred = prob.argmax(dim=1)
    accuracy = float((pred == labels_cpu).float().mean().item()) if int(labels_cpu.numel()) > 0 else 0.0
    macro_f1 = float(
        f1_score(
            labels_cpu.numpy(),
            pred.numpy(),
            average="macro",
            zero_division=0,
        )
    ) if int(labels_cpu.numel()) > 0 else 0.0
    loss = float(F.cross_entropy(logits_cpu, labels_cpu).item()) if int(labels_cpu.numel()) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "loss": loss,
        "count": int(labels_cpu.numel()),
    }


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
        "construct_input_dim": int(getattr(args, "_construct_input_dim", 0) or 0),
        "graph_construct_embedding_path": str(getattr(args, "graph_construct_embedding_path", "") or ""),
        "SimpleHGN_att_res": getattr(args, "SimpleHGN_att_res", 0.2),
        "att_heads": getattr(args, "att_heads", 8),
        "hyperscan_detector_style": getattr(args, "hyperscan_detector_style", "residual"),
        "graph_second_view_hypergraph_backend": getattr(args, "graph_second_view_hypergraph_backend", "pyg"),
        "graph_second_view_fusion": getattr(args, "graph_second_view_fusion", "residual"),
        "routed_multiview_refiner": {
            "enabled": bool(getattr(args, "mhlgc_enable", False) and getattr(args, "mhlgc_anchor_source", "semantic_nonzero_positive") == "routed_target_mask"),
            "num_heads": 4,
            "num_layers": 2,
            "ff_hidden_dim": int(getattr(args, "hidden_dim", 128)) * 2,
            "dropout": float(getattr(args, "GNN_dropout", 0.4)),
            "relation_token_count": 4,
            "semantic_token_count": 2,
            "bidirectional_multiattn": True,
            "semantic_input_dim": int(getattr(args, "mhlgc_semantic_input_dim", 0) or 0),
        },
        "routed_highpass": {
            "mode": str(getattr(args, "routed_highpass_mode", "off") or "off"),
            "target": str(getattr(args, "routed_highpass_target", "logits") or "logits"),
        },
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


def _resolve_mhlgc_target_mask_from_payload(path, expected_rows):
    if not path:
        return None
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return None
    mask = payload.get("target_node_mask")
    if mask is None and "target_node_ids" in payload:
        target_ids = payload["target_node_ids"]
        if torch.is_tensor(target_ids):
            target_ids = target_ids.detach().cpu().long().view(-1)
        else:
            target_ids = torch.tensor(target_ids, dtype=torch.long).view(-1)
        mask = torch.zeros((int(expected_rows),), dtype=torch.bool)
        valid = target_ids[(target_ids >= 0) & (target_ids < int(expected_rows))]
        if int(valid.numel()) > 0:
            mask[valid] = True
        return mask.contiguous()
    if mask is None:
        return None
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    mask = mask.detach().cpu().bool().view(-1)
    if int(mask.numel()) != int(expected_rows):
        raise ValueError(
            "--mhlgc_semantic_embedding_path target_node_mask row count must match graph nodes: "
            f"expected {int(expected_rows)}, got {int(mask.numel())}."
        )
    return mask.contiguous()


def _load_mhlgc_routed_multiview_rows(path, expected_rows):
    if not path:
        return None
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return None
    rows = payload.get("routed_multiview_rows")
    if rows is None:
        return None
    if not isinstance(rows, list):
        raise ValueError("--mhlgc_semantic_embedding_path routed_multiview_rows must be stored as a list.")
    if int(len(rows)) != int(expected_rows):
        raise ValueError(
            "--mhlgc_semantic_embedding_path routed_multiview_rows row count must match graph nodes: "
            f"expected {int(expected_rows)}, got {int(len(rows))}."
        )
    return rows


def _select_mhlgc_contrast_embeddings(outputs, input_semantic_embeddings, index_tensor, contrast_space):
    contrast_space = str(contrast_space or "node_repr").strip().lower()
    if contrast_space == "semantic":
        if input_semantic_embeddings is None:
            raise ValueError("--mhlgc_contrast_space semantic requires input LM embeddings for the selected nodes.")
        return input_semantic_embeddings[index_tensor]
    if contrast_space == "x_new":
        x_new = outputs.get("x_new")
        if not torch.is_tensor(x_new):
            raise ValueError(
                "--mhlgc_contrast_space x_new requires a HyperScan-style backbone that exposes outputs['x_new']."
            )
        return x_new[index_tensor]
    if contrast_space in {"low_high_concat", "low_high"}:
        x_low = outputs.get("x_low")
        x_high = outputs.get("x_high")
        if not torch.is_tensor(x_low) or not torch.is_tensor(x_high):
            raise ValueError(
                "--mhlgc_contrast_space low_high_concat requires a HyperScan-style backbone that exposes "
                "outputs['x_low'] and outputs['x_high']."
            )
        return torch.cat([x_low, x_high], dim=1)[index_tensor]
    if contrast_space in {"fused_x", "node_repr"}:
        fused_x = outputs.get("fused_x")
        if not torch.is_tensor(fused_x):
            fused_x = outputs.get("node_repr")
        if not torch.is_tensor(fused_x):
            raise ValueError("MH-LGC requires outputs['fused_x'] or legacy outputs['node_repr'] from the graph backbone.")
        return fused_x[index_tensor]
    raise ValueError(
        f"Unsupported --mhlgc_contrast_space {contrast_space!r}; "
        "expected one of fused_x/node_repr/x_new/low_high_concat/semantic."
    )


def _project_mhlgc_semantic(semantic_embeddings, target_dim, semantic_projector, optimizer, device, projector_mode):
    if semantic_embeddings is None:
        return None, semantic_projector
    semantic_embeddings = semantic_embeddings.to(device)
    target_dim = int(target_dim)
    if int(semantic_embeddings.shape[1]) == target_dim:
        return semantic_embeddings, semantic_projector
    if projector_mode == "none":
        raise ValueError(
            "repair-aware MH-LGC requires semantic and target dimensions to match when "
            "--mhlgc_semantic_projector none is set."
        )
    if (
        semantic_projector is None
        or int(semantic_projector.in_features) != int(semantic_embeddings.shape[1])
        or int(semantic_projector.out_features) != target_dim
    ):
        semantic_projector = nn.Linear(
            int(semantic_embeddings.shape[1]),
            target_dim,
            bias=False,
        ).to(device)
        optimizer.add_param_group({"params": semantic_projector.parameters()})
    return semantic_projector(semantic_embeddings), semantic_projector


def _load_routed_contrast_frozen_embeddings(path, expected_rows):
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
        raise ValueError(f"--routed_contrast_frozen_path must resolve to a 2-D tensor, got {tuple(payload.shape)}.")
    if int(payload.shape[0]) != int(expected_rows):
        raise ValueError(
            "--routed_contrast_frozen_path row count must match graph nodes: "
            f"expected {int(expected_rows)}, got {int(payload.shape[0])}."
        )
    return payload.contiguous()


def _flatten_routed_view_members(row):
    if not isinstance(row, dict):
        return []
    members = []
    for key in ("following_view", "follower_view", "mutual_view", "semantic_knn_view"):
        view = row.get(key)
        if not isinstance(view, dict):
            continue
        view_members = view.get("members", [])
        if isinstance(view_members, list):
            members.extend([item for item in view_members if isinstance(item, dict)])
    return members


def _compute_reliability_masks(rows, node_ids, edge_index=None):
    undirected_neighbors = {}
    if edge_index is not None:
        edge_index_cpu = (
            edge_index.detach().cpu().long()
            if torch.is_tensor(edge_index)
            else torch.tensor(edge_index, dtype=torch.long)
        )
        if edge_index_cpu.dim() == 2 and int(edge_index_cpu.size(0)) == 2:
            src = edge_index_cpu[0].tolist()
            dst = edge_index_cpu[1].tolist()
            for s, d in zip(src, dst):
                s = int(s)
                d = int(d)
                undirected_neighbors.setdefault(s, set()).add(d)
                undirected_neighbors.setdefault(d, set()).add(s)
    node_ids_list = [int(item) for item in node_ids]
    stats = []
    similarity_values = []
    for node_id in node_ids_list:
        row = rows[int(node_id)] if rows is not None and 0 <= int(node_id) < len(rows) else None
        members = _flatten_routed_view_members(row)
        support_count = int(len(members))
        reciprocal_or_common = False
        node_neighbors = undirected_neighbors.get(int(node_id), set())
        for item in members:
            relation_tags = set(str(tag) for tag in list(item.get("relation_to_target", [])))
            candidate_id = int(item.get("node_id", -1))
            candidate_neighbors = undirected_neighbors.get(candidate_id, set())
            common_neighbors = len(node_neighbors.intersection(candidate_neighbors))
            if (
                "mutual" in relation_tags
                or ("target_follows_candidate" in relation_tags and "candidate_follows_target" in relation_tags)
                or common_neighbors > 0
            ):
                reciprocal_or_common = True
                break
        sims = [float(item.get("semantic_similarity", 0.0)) for item in members if "semantic_similarity" in item]
        mean_similarity = float(np.mean(sims)) if sims else 0.0
        similarity_values.append(mean_similarity)
        stats.append(
            {
                "support_count": support_count,
                "has_reciprocal_or_common": bool(reciprocal_or_common),
                "mean_similarity": mean_similarity,
            }
        )
    median_similarity = float(np.median(similarity_values)) if similarity_values else 0.0
    mask_values = []
    for item in stats:
        reliable = (
            int(item["support_count"]) >= 2
            and bool(item["has_reciprocal_or_common"])
            and float(item["mean_similarity"]) >= median_similarity
        )
        mask_values.append(bool(reliable))
    reliable_mask = torch.tensor(mask_values, dtype=torch.bool)
    stats_payload = {
        "eligible_count": int(len(node_ids_list)),
        "reliable_count": int(reliable_mask.sum().item()),
        "reliable_ratio": float(reliable_mask.float().mean().item()) if int(reliable_mask.numel()) > 0 else 0.0,
        "median_mean_similarity": float(median_similarity),
        "mean_support_count": float(np.mean([float(item["support_count"]) for item in stats])) if stats else 0.0,
    }
    return reliable_mask, stats_payload


def _load_routed_highpass_risk_scores(path, expected_rows):
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"--routed_highpass_risk_path does not exist: {path}")
    payload = safe_torch_load(path, map_location="cpu") if path.suffix.lower() == ".pt" else read_json(path, default=None)
    if isinstance(payload, dict):
        for key in ("risk_score", "router_score", "abstain_risk", "scores"):
            if key in payload:
                payload = payload[key]
                break
    if payload is None:
        raise ValueError("--routed_highpass_risk_path must contain risk_score/router_score/abstain_risk or a raw vector.")
    risk = torch.as_tensor(payload, dtype=torch.float32).detach().cpu().view(-1)
    if int(risk.numel()) != int(expected_rows):
        raise ValueError(
            "--routed_highpass_risk_path row count must match graph nodes: "
            f"expected {int(expected_rows)}, got {int(risk.numel())}."
        )
    return torch.nan_to_num(risk, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0).contiguous()


def _load_optional_fullgraph_vector(path, expected_rows, allowed_keys, dtype=torch.float32, value_name="vector"):
    path = str(path or "").strip()
    if not path:
        return None
    payload_path = Path(path)
    if not payload_path.exists():
        raise FileNotFoundError(f"{value_name} path does not exist: {payload_path}")
    payload = safe_torch_load(payload_path, map_location="cpu") if payload_path.suffix.lower() == ".pt" else read_json(payload_path, default=None)
    if isinstance(payload, dict):
        for key in allowed_keys:
            if key in payload:
                payload = payload[key]
                break
    if payload is None:
        raise ValueError(f"{value_name} path must contain one of {tuple(allowed_keys)} or a raw vector.")
    vector = torch.as_tensor(payload, dtype=dtype).detach().cpu().view(-1)
    if int(vector.numel()) != int(expected_rows):
        raise ValueError(
            f"{value_name} row count must match graph nodes: expected {int(expected_rows)}, got {int(vector.numel())}."
        )
    return vector.contiguous()


def _load_llm_edge_retain_cache(path, center_ids, candidate_rows):
    path = str(path or "").strip()
    if not path:
        return None, {
            "llm_edge_retain_path": "",
            "llm_edge_retain_source": "unavailable",
            "llm_edge_retain_candidate_count": 0,
            "llm_edge_retain_keep_count": 0,
            "llm_edge_retain_keep_ratio": 0.0,
        }
    payload_path = Path(path)
    if not payload_path.exists():
        raise FileNotFoundError(f"--graph_second_view_llm_edge_retain_path does not exist: {payload_path}")
    payload = safe_torch_load(payload_path, map_location="cpu") if payload_path.suffix.lower() == ".pt" else read_json(payload_path, default=None)
    if not isinstance(payload, dict):
        raise ValueError("--graph_second_view_llm_edge_retain_path must contain a dict payload.")
    cache_centers = [int(item) for item in payload.get("center_node_ids", [])]
    cache_rows = [[int(v) for v in row] for row in payload.get("center_candidate_node_ids", [])]
    cache_keep = payload.get("center_candidate_keep_mask", None)
    if cache_keep is None:
        cache_keep = payload.get("keep_mask", None)
    if cache_keep is None:
        raise ValueError("--graph_second_view_llm_edge_retain_path must contain center_candidate_keep_mask.")
    keep_rows = []
    for row in cache_keep:
        if torch.is_tensor(row):
            keep_rows.append([bool(v) for v in row.detach().cpu().bool().view(-1).tolist()])
        else:
            keep_rows.append([bool(v) for v in list(row)])
    if len(cache_centers) != len(cache_rows) or len(cache_centers) != len(keep_rows):
        raise ValueError("LLM edge-retention cache center/candidate/keep rows must have matching lengths.")

    requested_centers = [int(item) for item in center_ids]
    requested_rows = [[int(v) for v in row] for row in candidate_rows]
    cache_by_center = {}
    for center, candidates, keeps in zip(cache_centers, cache_rows, keep_rows):
        if len(candidates) != len(keeps):
            raise ValueError(f"LLM edge-retention cache row for center {center} has mismatched candidates/keep mask.")
        cache_by_center[int(center)] = {
            "candidate_ids": [int(v) for v in candidates],
            "keep_mask": [bool(v) for v in keeps],
        }

    aligned_candidate_rows = []
    aligned_keep_rows = []
    keep_count = 0
    cached_candidate_count = 0
    missing_centers = []
    for center, requested in zip(requested_centers, requested_rows):
        cached = cache_by_center.get(int(center))
        if cached is None:
            missing_centers.append(int(center))
            aligned_candidate_rows.append([])
            aligned_keep_rows.append([])
            continue
        cached_candidate_count += int(len(cached["candidate_ids"]))
        allowed = {
            int(candidate): bool(keep)
            for candidate, keep in zip(cached["candidate_ids"], cached["keep_mask"])
            if bool(keep)
        }
        filtered = [int(candidate) for candidate in requested if int(candidate) in allowed]
        aligned_candidate_rows.append(filtered)
        aligned_keep_rows.append([True for _ in filtered])
        keep_count += int(len(filtered))
    if missing_centers:
        preview = missing_centers[:10]
        raise ValueError(
            "LLM edge-retention cache is missing requested second-view centers: "
            f"{preview}{' ...' if len(missing_centers) > len(preview) else ''}."
        )
    stats = {
        "llm_edge_retain_path": str(payload_path),
        "llm_edge_retain_source": str(payload.get("method_family", "llm_knn_edge_retain_v1")),
        "llm_edge_retain_candidate_count": int(cached_candidate_count),
        "llm_edge_retain_keep_count": int(keep_count),
        "llm_edge_retain_keep_ratio": float(keep_count / float(cached_candidate_count)) if cached_candidate_count else 0.0,
        "llm_edge_retain_prompt_model": str(payload.get("explain_model_path", "")),
        "llm_edge_retain_threshold": float(payload.get("retain_threshold", 0.5)),
    }
    return {
        "center_candidate_node_ids": aligned_candidate_rows,
        "center_candidate_keep_mask": aligned_keep_rows,
    }, stats


def _load_router_support_cache(path, center_ids):
    path = str(path or "").strip()
    if not path:
        return None, {
            "router_support_path": "",
            "router_support_source": "unavailable",
            "router_support_center_count": 0,
            "router_support_candidate_count": 0,
        }
    payload_path = Path(path)
    if not payload_path.exists():
        raise FileNotFoundError(f"--graph_second_view_router_support_path does not exist: {payload_path}")
    payload = safe_torch_load(payload_path, map_location="cpu") if payload_path.suffix.lower() == ".pt" else read_json(payload_path, default=None)
    if not isinstance(payload, dict):
        raise ValueError("--graph_second_view_router_support_path must contain a dict payload.")
    support_payload = payload.get("support_group_payload", payload)
    if not isinstance(support_payload, dict):
        raise ValueError("--graph_second_view_router_support_path must contain support_group_payload or a direct payload dict.")
    cache_centers = [int(item) for item in support_payload.get("center_node_ids", [])]
    cache_rows = [[int(v) for v in row] for row in support_payload.get("center_candidate_node_ids", [])]
    if len(cache_centers) != len(cache_rows):
        raise ValueError("Router support cache center/candidate rows must have matching lengths.")
    cache_by_center = {
        int(center): [int(v) for v in row]
        for center, row in zip(cache_centers, cache_rows)
    }
    requested_centers = [int(item) for item in center_ids]
    aligned_candidate_rows = []
    missing_centers = []
    total_candidates = 0
    for center in requested_centers:
        row = cache_by_center.get(int(center))
        if row is None:
            missing_centers.append(int(center))
            aligned_candidate_rows.append([])
            continue
        aligned_candidate_rows.append([int(v) for v in row])
        total_candidates += int(len(row))
    if missing_centers:
        preview = missing_centers[:10]
        raise ValueError(
            "Router support cache is missing requested second-view centers: "
            f"{preview}{' ...' if len(missing_centers) > len(preview) else ''}."
        )
    stats = {
        "router_support_path": str(payload_path),
        "router_support_source": str(support_payload.get("source", "conformal_knn_risk_router")),
        "router_support_contract": str(support_payload.get("contract", "unknown")),
        "router_support_center_count": int(len(requested_centers)),
        "router_support_candidate_count": int(total_candidates),
        "router_support_mean_candidates_per_center": (
            float(total_candidates / float(len(requested_centers))) if requested_centers else 0.0
        ),
        "router_support_backend": str(support_payload.get("backend", "")),
        "router_support_neighbor_mode": str(support_payload.get("neighbor_mode", "")),
    }
    return {
        "center_candidate_node_ids": aligned_candidate_rows,
    }, stats


def _base_confidence_risk_from_current_features(features):
    feature_cpu = (
        features.detach().cpu().float()
        if torch.is_tensor(features)
        else torch.tensor(features, dtype=torch.float32)
    )
    if feature_cpu.dim() != 2:
        raise ValueError("features must be a 2D tensor for fallback second-view candidate confidence.")
    if int(feature_cpu.size(1)) == 2:
        prob = torch.softmax(feature_cpu[:, :2], dim=1)
        return (1.0 - prob.max(dim=1).values).clamp(0.0, 1.0).contiguous()
    raise ValueError(
        "Quota-based second-view candidate policies require --graph_second_view_candidate_risk_path "
        "unless the current input features are exactly two-dimensional logits/probabilities."
    )


def _build_second_view_candidate_role_bundle(
    candidate_bundle,
    refine_request,
    num_nodes,
    labeled_node_count,
    features,
):
    policy = str(refine_request.get("candidate_policy", "default") or "default").strip().lower()
    center_ids = [int(item) for item in candidate_bundle.get("center_node_ids", [])]
    candidate_rows = [[int(v) for v in row] for row in candidate_bundle.get("center_candidate_node_ids", [])]
    routed_ids = set()
    candidate_routed_nodes_path = str(
        refine_request.get("candidate_routed_nodes_path", "")
        or refine_request.get("routed_nodes_path", "")
        or ""
    ).strip()
    if candidate_routed_nodes_path:
        routed_ids = set(
            int(item)
            for item in _load_routed_center_node_ids(Path(candidate_routed_nodes_path), split_name="all")
            if 0 <= int(item) < int(num_nodes)
        )

    role_rows = []
    role_counts = {"hard": 0, "stable": 0, "counterfactual": 0, "generic": 0}
    role_contract = "none"
    stats = {
        "candidate_policy": policy,
        "candidate_role_contract": role_contract,
        "candidate_policy_uses_labels": False,
        "candidate_policy_scope": "training_time_second_view_member_selection",
        "candidate_policy_routed_count": int(len(routed_ids)),
        "candidate_routed_nodes_path": candidate_routed_nodes_path,
        "llm_edge_retain_path": str(refine_request.get("llm_edge_retain_path", "") or ""),
        "router_support_path": str(refine_request.get("router_support_path", "") or ""),
    }

    if policy == "default":
        candidate_bundle["center_candidate_role_ids"] = []
        stats["candidate_policy_quotas"] = {}
        stats["candidate_role_counts"] = role_counts
        candidate_bundle["candidate_policy_stats"] = stats
        return candidate_bundle
    if policy not in {"exclude_routed", "post_topk_exclude_routed", "stable_quota", "mixed_quota", "llm_retain", "router_support_transfer"}:
        raise ValueError(
            "--graph_second_view_candidate_policy must be one of "
            "{default, exclude_routed, post_topk_exclude_routed, stable_quota, mixed_quota, llm_retain, router_support_transfer}."
        )

    if policy == "llm_retain":
        retain_bundle, retain_stats = _load_llm_edge_retain_cache(
            refine_request.get("llm_edge_retain_path", ""),
            center_ids=center_ids,
            candidate_rows=candidate_rows,
        )
        if retain_bundle is None:
            raise ValueError(
                "--graph_second_view_candidate_policy llm_retain requires "
                "--graph_second_view_llm_edge_retain_path."
            )
        candidate_bundle["center_candidate_node_ids"] = retain_bundle["center_candidate_node_ids"]
        candidate_bundle["center_candidate_keep_mask"] = retain_bundle["center_candidate_keep_mask"]
        candidate_bundle["center_candidate_role_ids"] = [
            [2 for _ in row] for row in retain_bundle["center_candidate_node_ids"]
        ]
        stats.update(
            {
                **retain_stats,
                "candidate_role_contract": "1=drop_by_llm_or_absent,2=llm_retained_nonrouted_support",
                "candidate_policy_quotas": {},
                "candidate_role_counts": {
                    "hard": 0,
                    "stable": int(retain_stats["llm_edge_retain_keep_count"]),
                    "counterfactual": 0,
                    "generic": 0,
                },
            }
        )
        candidate_bundle["candidate_policy_stats"] = stats
        return candidate_bundle

    if policy == "router_support_transfer":
        support_bundle, support_stats = _load_router_support_cache(
            refine_request.get("router_support_path", ""),
            center_ids=center_ids,
        )
        if support_bundle is None:
            raise ValueError(
                "--graph_second_view_candidate_policy router_support_transfer requires "
                "--graph_second_view_router_support_path."
            )
        candidate_bundle["center_candidate_node_ids"] = support_bundle["center_candidate_node_ids"]
        candidate_bundle["center_candidate_role_ids"] = [
            [0 for _ in row] for row in support_bundle["center_candidate_node_ids"]
        ]
        stats.update(
            {
                **support_stats,
                "candidate_role_contract": "0=router_selected_support_member",
                "candidate_policy_quotas": {},
                "candidate_role_counts": {
                    "hard": 0,
                    "stable": 0,
                    "counterfactual": 0,
                    "generic": int(support_stats["router_support_candidate_count"]),
                },
                "candidate_policy_scope": "router_support_neighborhood_transfer",
                "candidate_policy_uses_labels": False,
            }
        )
        candidate_bundle["candidate_policy_stats"] = stats
        return candidate_bundle

    candidate_risk_path = str(refine_request.get("candidate_risk_path", "") or "").strip()
    risk = _load_optional_fullgraph_vector(
        candidate_risk_path,
        expected_rows=int(num_nodes),
        allowed_keys=("risk_score", "router_score", "abstain_risk", "scores"),
        dtype=torch.float32,
        value_name="--graph_second_view_candidate_risk_path",
    )
    risk_source = "candidate_risk_path" if candidate_risk_path else "unavailable"
    if risk is None and policy in {"stable_quota", "mixed_quota"}:
        risk = _base_confidence_risk_from_current_features(features)
        risk_source = "fallback_base_confidence_from_current_input_features"
    pred = _load_optional_fullgraph_vector(
        refine_request.get("candidate_pred_path", ""),
        expected_rows=int(num_nodes),
        allowed_keys=("pred", "prediction", "preds", "base_pred"),
        dtype=torch.long,
        value_name="--graph_second_view_candidate_pred_path",
    )
    pred_source = "candidate_pred_path" if pred is not None else "unavailable"
    if policy == "mixed_quota" and pred is None:
        raise ValueError(
            "--graph_second_view_candidate_policy mixed_quota requires "
            "--graph_second_view_candidate_pred_path so counterfactual support is explicitly defined."
        )

    stable_mask = np.zeros(int(num_nodes), dtype=bool)
    stable_cutoff = None
    if risk is not None:
        risk_np = torch.nan_to_num(risk.float(), nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0).numpy()
        nonrouted_limit = int(num_nodes) if str(refine_request.get("candidate_scope", "")).lower() == "hyperscan_full" else int(labeled_node_count)
        nonrouted_ids = np.asarray([idx for idx in range(nonrouted_limit) if idx not in routed_ids], dtype=np.int64)
        if nonrouted_ids.size > 0:
            quantile = float(refine_request.get("stable_quantile", 0.50))
            quantile = min(max(quantile, 0.0), 1.0)
            stable_cutoff = float(np.quantile(risk_np[nonrouted_ids], quantile))
            stable_mask[:nonrouted_limit] = risk_np[:nonrouted_limit] <= stable_cutoff
            if routed_ids:
                routed_arr = np.asarray([idx for idx in routed_ids if 0 <= idx < int(num_nodes)], dtype=np.int64)
                if routed_arr.size > 0:
                    stable_mask[routed_arr] = False

    pred_np = pred.numpy() if pred is not None else None
    for center, row in zip(center_ids, candidate_rows):
        center_pred = int(pred_np[int(center)]) if pred_np is not None and 0 <= int(center) < int(pred_np.shape[0]) else None
        current_roles = []
        for cand in row:
            role = 0
            if int(cand) in routed_ids:
                role = 1
                role_counts["hard"] += 1
            elif (bool(stable_mask[int(cand)]) if 0 <= int(cand) < int(stable_mask.shape[0]) else False):
                if pred_np is not None and center_pred is not None and int(pred_np[int(cand)]) != int(center_pred):
                    role = 3
                    role_counts["counterfactual"] += 1
                else:
                    role = 2
                    role_counts["stable"] += 1
            else:
                role_counts["generic"] += 1
            current_roles.append(int(role))
        role_rows.append(current_roles)

    stable_quota = int(refine_request.get("stable_quota", 0) or 0)
    if stable_quota <= 0:
        stable_quota = max(1, int(refine_request.get("knn_k", 8)) // 2)
    role_contract = "0=generic,1=hard_routed,2=stable_nonrouted,3=stable_counterfactual"
    stats.update(
        {
            "candidate_role_contract": role_contract,
            "candidate_policy_quotas": {
                "stable": int(stable_quota),
                "hard": int(refine_request.get("mixed_hard_quota", 2) or 2),
                "stable_mixed": int(refine_request.get("mixed_stable_quota", 4) or 4),
                "counterfactual": int(refine_request.get("mixed_counterfactual_quota", 2) or 2),
            },
            "candidate_role_counts": role_counts,
            "candidate_risk_source": risk_source,
            "candidate_pred_source": pred_source,
            "stable_quantile": float(refine_request.get("stable_quantile", 0.50)),
            "stable_risk_cutoff": stable_cutoff,
            "stable_candidate_count": int(stable_mask.sum()),
        }
    )
    candidate_bundle["center_candidate_role_ids"] = role_rows
    candidate_bundle["candidate_policy_stats"] = stats
    return candidate_bundle


def _build_relation_1hop_candidate_lists(edge_index, num_nodes, max_neighbors, candidate_scope, labeled_node_count):
    edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
    if edge_index_cpu.dim() != 2 or int(edge_index_cpu.size(0)) != 2:
        raise ValueError("routed high-pass correction requires edge_index shaped [2, num_edges].")
    num_nodes = int(num_nodes)
    scope = str(candidate_scope or "relation_1hop").strip().lower()
    if scope not in {"relation_1hop", "relation_1hop_plus_xnew_knn"}:
        raise ValueError("--routed_highpass_candidate_scope must be one of {relation_1hop, relation_1hop_plus_xnew_knn}.")
    labeled_limit = int(labeled_node_count) if labeled_node_count is not None else num_nodes
    labeled_limit = max(0, min(num_nodes, labeled_limit))
    adjacency = [set() for _ in range(num_nodes)]
    src = edge_index_cpu[0].tolist()
    dst = edge_index_cpu[1].tolist()
    for s, d in zip(src, dst):
        s = int(s)
        d = int(d)
        if 0 <= s < num_nodes and 0 <= d < num_nodes and s != d:
            adjacency[s].add(d)
            adjacency[d].add(s)
    rows = []
    for node_id, values in enumerate(adjacency):
        candidates = sorted(int(item) for item in values)
        if scope == "relation_1hop_plus_xnew_knn":
            candidates = [item for item in candidates if item < labeled_limit]
        rows.append(torch.tensor(candidates, dtype=torch.long))
    stats = {
        "candidate_scope": scope,
        "max_neighbors": int(max_neighbors),
        "mean_candidates": float(np.mean([int(item.numel()) for item in rows])) if rows else 0.0,
        "nodes_with_candidates": int(sum(1 for item in rows if int(item.numel()) > 0)),
    }
    return rows, stats


def _build_routed_highpass_bundle(
    outputs_dict,
    labels_tensor,
    train_mask,
    routed_mask,
    candidate_rows,
    risk_scores,
    mode,
    candidate_scope,
    max_neighbors,
    device,
    edge_role_head=None,
    target="logits",
):
    target = str(target or "logits").strip().lower()
    if target not in {"logits", "x_high"}:
        raise ValueError("--routed_highpass_target must be one of {logits, x_high}.")
    if target == "x_high":
        fused = outputs_dict.get("x_high")
        if not torch.is_tensor(fused):
            raise ValueError("routed high-pass target=x_high requires outputs['x_high'].")
    else:
        fused = outputs_dict.get("fused_x")
        if not torch.is_tensor(fused):
            fused = outputs_dict.get("node_repr")
    if not torch.is_tensor(fused):
        raise ValueError("routed high-pass correction requires outputs['fused_x'] or outputs['node_repr'].")
    base_logits = outputs_dict["logits"].detach()
    base_prob = torch.softmax(base_logits, dim=-1)
    pred = base_prob.argmax(dim=1)
    entropy = -(base_prob * torch.log(base_prob.clamp_min(1e-12))).sum(dim=1)
    top2 = torch.topk(base_prob, k=min(2, int(base_prob.size(1))), dim=1).values
    margin = top2[:, 0] - (top2[:, 1] if int(top2.size(1)) > 1 else 0.0)
    num_nodes = int(fused.size(0))
    hidden_dim = int(fused.size(1))
    fused_source = fused
    fused_detached = fused.detach()
    x_new = outputs_dict.get("x_new")
    x_new_detached = x_new.detach() if torch.is_tensor(x_new) else None
    candidate_scope = str(candidate_scope or "relation_1hop").strip().lower()
    max_neighbors = max(int(max_neighbors), 1)
    low_agg = torch.zeros((num_nodes, hidden_dim), dtype=fused.dtype, device=device)
    high_agg = torch.zeros((num_nodes, hidden_dim), dtype=fused.dtype, device=device)
    scalar_features = torch.zeros((num_nodes, 6), dtype=fused.dtype, device=device)
    degree_values = torch.zeros((num_nodes,), dtype=fused.dtype, device=device)
    same_pred_ratio = torch.zeros((num_nodes,), dtype=fused.dtype, device=device)
    diff_pred_ratio = torch.zeros((num_nodes,), dtype=fused.dtype, device=device)
    labels_flat = labels_tensor.view(-1)
    train_mask = train_mask.to(device=device).bool().view(-1)
    routed_mask = routed_mask.to(device=device).bool().view(-1)
    if int(routed_mask.numel()) != num_nodes:
        raise ValueError("routed_highpass routed mask must align with graph rows.")
    risk = risk_scores
    if risk is None:
        risk = routed_mask.to(dtype=fused.dtype)
    else:
        risk = risk.to(device=device, dtype=fused.dtype).view(-1)
    if int(risk.numel()) != num_nodes:
        raise ValueError("routed_highpass risk scores must align with graph rows.")
    edge_role_logits = []
    edge_role_targets = []
    active_ids = torch.where(routed_mask)[0].detach().cpu().long().tolist()
    for node_id in active_ids:
        if node_id < 0 or node_id >= len(candidate_rows):
            continue
        cand_cpu = candidate_rows[int(node_id)]
        if not torch.is_tensor(cand_cpu) or int(cand_cpu.numel()) == 0:
            continue
        cand = cand_cpu.to(device=device).long()
        cand = cand[(cand >= 0) & (cand < num_nodes) & (cand != int(node_id))]
        if candidate_scope == "relation_1hop_plus_xnew_knn" and x_new_detached is not None:
            pool_limit = max(0, min(int(labels_flat.numel()), num_nodes))
            if pool_limit > 1 and int(node_id) < pool_limit:
                pool = torch.arange(pool_limit, device=device, dtype=torch.long)
                pool = pool[pool != int(node_id)]
                if int(pool.numel()) > 0:
                    pool_scores = F.cosine_similarity(
                        x_new_detached[int(node_id)].unsqueeze(0),
                        x_new_detached[pool],
                        dim=1,
                    )
                    knn_count = min(max_neighbors, int(pool.numel()))
                    knn = pool[torch.topk(pool_scores, k=knn_count, largest=True).indices]
                    cand = torch.unique(torch.cat([cand, knn], dim=0))
        if int(cand.numel()) == 0:
            continue
        center = fused_source[int(node_id)]
        neighbor = fused_source[cand]
        center_detached = fused_detached[int(node_id)]
        neighbor_detached = fused_detached[cand]
        if candidate_scope == "relation_1hop_plus_xnew_knn" and x_new_detached is not None:
            ranking_scores = F.cosine_similarity(
                x_new_detached[int(node_id)].unsqueeze(0),
                x_new_detached[cand],
                dim=1,
            )
        else:
            ranking_scores = F.cosine_similarity(center_detached.unsqueeze(0), neighbor_detached, dim=1)
        if int(cand.numel()) > max_neighbors:
            topk = torch.topk(ranking_scores, k=max_neighbors, largest=True).indices
            cand = cand[topk]
            neighbor = fused_source[cand]
            neighbor_detached = fused_detached[cand]
            ranking_scores = ranking_scores[topk]
        sim = ranking_scores.clamp_min(0.0)
        pred_agree = pred[cand] == pred[int(node_id)]
        base_hetero_score = (~pred_agree).to(dtype=fused.dtype)
        role_context = torch.cat(
            [
                center.unsqueeze(0).expand(int(cand.numel()), -1),
                neighbor,
                center.unsqueeze(0).expand(int(cand.numel()), -1) - neighbor,
            ],
            dim=1,
        )
        if edge_role_head is not None:
            learned_hetero_score = torch.sigmoid(edge_role_head(role_context).view(-1))
            hetero_score = 0.5 * base_hetero_score + 0.5 * learned_hetero_score
        else:
            hetero_score = base_hetero_score
        low_weight = sim * (1.0 - hetero_score).clamp(0.0, 1.0)
        high_weight = sim * hetero_score.clamp(0.0, 1.0)
        if float(low_weight.sum().detach().cpu().item()) <= 1e-12:
            low_weight = sim
        if float(high_weight.sum().detach().cpu().item()) <= 1e-12:
            high_weight = sim
        low_agg[int(node_id)] = (neighbor * (low_weight / low_weight.sum().clamp_min(1e-12)).unsqueeze(1)).sum(dim=0)
        high_residual = center.unsqueeze(0) - neighbor
        high_agg[int(node_id)] = (
            high_residual * (high_weight / high_weight.sum().clamp_min(1e-12)).unsqueeze(1)
        ).sum(dim=0)
        degree_values[int(node_id)] = float(int(cand.numel()))
        same_pred_ratio[int(node_id)] = pred_agree.to(dtype=fused.dtype).mean()
        diff_pred_ratio[int(node_id)] = 1.0 - same_pred_ratio[int(node_id)]
        train_cand = cand[train_mask[cand]]
        if bool(train_mask[int(node_id)].item()) and int(train_cand.numel()) > 0:
            center_label = labels_flat[int(node_id)]
            valid_label_mask = labels_flat[train_cand] >= 0
            if bool((center_label >= 0).item()) and bool(valid_label_mask.any().item()):
                train_cand = train_cand[valid_label_mask]
                same_label = labels_flat[train_cand] == center_label
                supervised_role_context = torch.cat(
                    [
                        center.unsqueeze(0).expand(int(train_cand.numel()), -1),
                        fused_source[train_cand],
                        center.unsqueeze(0).expand(int(train_cand.numel()), -1) - fused_source[train_cand],
                    ],
                    dim=1,
                )
                if edge_role_head is not None:
                    edge_role_logits.append(edge_role_head(supervised_role_context).view(-1))
                edge_role_targets.append((~same_label).to(dtype=fused.dtype))
    denom = max(float(degree_values.max().detach().cpu().item()), 1.0)
    scalar_features[:, 0] = risk
    scalar_features[:, 1] = entropy.detach()
    scalar_features[:, 2] = margin.detach()
    scalar_features[:, 3] = (degree_values / denom).clamp(0.0, 1.0)
    scalar_features[:, 4] = same_pred_ratio
    scalar_features[:, 5] = diff_pred_ratio
    bundle = {
        "low_agg": low_agg,
        "high_agg": high_agg,
        "scalar_features": scalar_features,
        "routed_mask": routed_mask,
        "risk_scores": risk,
        "mode": str(mode),
        "target": target,
    }
    stats = {
        "target": target,
        "routed_node_count": int(routed_mask.sum().detach().cpu().item()),
        "active_with_candidates": int((degree_values[routed_mask] > 0).sum().detach().cpu().item()) if bool(routed_mask.any()) else 0,
        "mean_degree_active": float(degree_values[routed_mask].mean().detach().cpu().item()) if bool(routed_mask.any()) else 0.0,
        "mean_same_pred_ratio_active": float(same_pred_ratio[routed_mask].mean().detach().cpu().item()) if bool(routed_mask.any()) else 0.0,
        "mean_diff_pred_ratio_active": float(diff_pred_ratio[routed_mask].mean().detach().cpu().item()) if bool(routed_mask.any()) else 0.0,
    }
    if edge_role_logits:
        bundle["edge_role_logits"] = torch.cat(edge_role_logits, dim=0)
        bundle["edge_role_targets"] = torch.cat(edge_role_targets, dim=0)
        stats["edge_role_pair_count"] = int(bundle["edge_role_targets"].numel())
    else:
        bundle["edge_role_logits"] = None
        bundle["edge_role_targets"] = None
        stats["edge_role_pair_count"] = 0
    return bundle, stats


def _project_routed_frozen_embeddings(frozen_embeddings, target_dim, frozen_projector, optimizer, device):
    if frozen_embeddings is None:
        return None, frozen_projector
    frozen_embeddings = frozen_embeddings.to(device)
    target_dim = int(target_dim)
    if int(frozen_embeddings.shape[1]) == target_dim:
        return frozen_embeddings, frozen_projector
    if (
        frozen_projector is None
        or int(frozen_projector.in_features) != int(frozen_embeddings.shape[1])
        or int(frozen_projector.out_features) != target_dim
    ):
        frozen_projector = nn.Linear(
            int(frozen_embeddings.shape[1]),
            target_dim,
            bias=False,
        ).to(device)
        optimizer.add_param_group({"params": frozen_projector.parameters()})
    return frozen_projector(frozen_embeddings), frozen_projector


def _float_cli_arg(value, default):
    """Preserve explicit 0.0 values instead of falling back through Python truthiness."""
    return float(default if value is None else value)


def _expand_labeled_targets_to_graph(y_cpu, graph_num_nodes, ignore_index=-100):
    y_cpu = y_cpu.detach().cpu().long().view(-1) if torch.is_tensor(y_cpu) else torch.tensor(y_cpu, dtype=torch.long).view(-1)
    graph_num_nodes = int(graph_num_nodes)
    if int(y_cpu.numel()) > graph_num_nodes:
        raise ValueError(
            f"Labeled target rows ({int(y_cpu.numel())}) exceed graph nodes ({graph_num_nodes})."
        )
    if int(y_cpu.numel()) == graph_num_nodes:
        return y_cpu.contiguous()
    y_full = torch.full((graph_num_nodes,), int(ignore_index), dtype=torch.long)
    y_full[: int(y_cpu.numel())] = y_cpu
    return y_full.contiguous()


def _train_graph_backbone_once(
    config,
    x_projected,
    x_raw,
    x_construct,
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
    graph_neighborloader_contract="seed_only",
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
    mhlgc_hyperedge_mask_probability=0.0,
    mhlgc_anchors_per_batch=1,
    mhlgc_negative_count=0,
    mhlgc_positive_label=1,
    mhlgc_target_mask=None,
    mhlgc_routed_multiview_rows=None,
    mhlgc_contrast_space="fused_x",
    mhlgc_anchor_source="semantic_nonzero_positive",
    mhlgc_pair_mode="augmentation",
    mhlgc_semantic_projector="auto",
    routed_contrast_enabled=False,
    routed_contrast_family="none",
    routed_contrast_gate="heuristic_reliable",
    routed_contrast_weight=0.0,
    routed_contrast_temperature=0.2,
    routed_contrast_frozen_embeddings=None,
    routed_contrast_train_node_ids=None,
    routed_contrast_multiview_rows=None,
    routed_highpass_mode="off",
    routed_highpass_target="logits",
    routed_highpass_candidate_scope="relation_1hop",
    routed_highpass_max_neighbors=32,
    routed_highpass_loss_weight=1.0,
    routed_highpass_preserve_weight=0.2,
    routed_highpass_risk_gate_weight=0.5,
    routed_highpass_edge_role_weight=0.1,
    routed_highpass_train_node_ids=None,
    routed_highpass_node_ids=None,
    routed_highpass_risk_scores=None,
    routed_highpass_candidate_rows=None,
):
    device = config["device"]
    config = dict(config)
    routed_refiner_cfg = dict(config.get("routed_multiview_refiner", {}) or {})
    if mhlgc_semantic_embeddings is not None and routed_refiner_cfg:
        routed_refiner_cfg["semantic_input_dim"] = int(mhlgc_semantic_embeddings.shape[1])
        config["routed_multiview_refiner"] = routed_refiner_cfg
    model = build_GNN_model(config).to(device)
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
    best_checkpoint_primary = "validation_macro_f1"
    best_checkpoint_tie_breaker = "validation_loss"
    optimizer_steps = 0
    neighborloader_contract = str(graph_neighborloader_contract or "seed_only").strip().lower()
    if neighborloader_contract not in {"seed_only", "hyperscan_sampled_subgraph"}:
        raise ValueError(
            "--graph_neighborloader_contract must be one of {seed_only, hyperscan_sampled_subgraph}."
        )
    neighborloader_contract_metrics = {
        "contract": neighborloader_contract,
        "supervision_scope": "seed_only_first_batch_rows",
        "validation_scope": "canonical_valid_seed_nodes",
        "test_scope": "full_graph_deduplicated_export_only",
        "duplicate_counting": "none",
        "checkpoint_selection_primary": "validation_macro_f1",
        "checkpoint_selection_tie_breaker": "validation_loss",
        "valid_best_checkpoint": {},
        "test_best_checkpoint": {},
    }
    mhlgc_enabled = bool(mhlgc_enabled)
    mhlgc_loss_weight = float(mhlgc_loss_weight or 0.0)
    mhlgc_contrast_space = str(mhlgc_contrast_space or "fused_x").strip().lower()
    mhlgc_anchor_source = str(mhlgc_anchor_source or "semantic_nonzero_positive").strip().lower()
    mhlgc_pair_mode = str(mhlgc_pair_mode or "augmentation").strip().lower()
    mhlgc_semantic_projector = str(mhlgc_semantic_projector or "auto").strip().lower()
    if mhlgc_pair_mode not in {"augmentation", "repair_aware"}:
        raise ValueError("--mhlgc_pair_mode must be one of {augmentation, repair_aware}.")
    if mhlgc_anchor_source not in {"semantic_nonzero_positive", "routed_target_mask", "positive_label"}:
        raise ValueError(
            "--mhlgc_anchor_source must be one of {semantic_nonzero_positive, routed_target_mask, positive_label}."
        )
    if mhlgc_semantic_projector not in {"auto", "none"}:
        raise ValueError("--mhlgc_semantic_projector must be one of {auto, none}.")
    routed_contrast_enabled = bool(routed_contrast_enabled)
    routed_contrast_family = str(routed_contrast_family or "none").strip().lower()
    routed_contrast_gate = str(routed_contrast_gate or "heuristic_reliable").strip().lower()
    routed_contrast_weight = float(routed_contrast_weight or 0.0)
    routed_contrast_temperature = float(routed_contrast_temperature or 0.2)
    if routed_contrast_family not in {"none", "low_high", "three_view_control", "supcon_class", "hybrid", "bot_edge_mask_human"}:
        raise ValueError(
            "--routed_contrast_family must be one of "
            "{none, low_high, three_view_control, supcon_class, hybrid, bot_edge_mask_human}."
        )
    if routed_contrast_gate not in {"none", "heuristic_reliable"}:
        raise ValueError("--routed_contrast_gate must be one of {none, heuristic_reliable}.")
    if routed_contrast_enabled and mhlgc_enabled:
        raise ValueError("--mhlgc_enable and --routed_contrast_family cannot be enabled together.")
    routed_contrast_enabled = bool(routed_contrast_enabled and routed_contrast_family != "none" and routed_contrast_weight > 0.0)
    if routed_contrast_enabled and str(training_loader_mode).lower() != "full_batch":
        raise ValueError("routed contrast ablations are only supported with --graph_training_loader_mode full_batch.")
    routed_highpass_mode = str(routed_highpass_mode or "off").strip().lower()
    if routed_highpass_mode not in {"off", "low_only", "high_only", "adaptive"}:
        raise ValueError("--routed_highpass_mode must be one of {off, low_only, high_only, adaptive}.")
    routed_highpass_target = str(routed_highpass_target or "logits").strip().lower()
    if routed_highpass_target not in {"logits", "x_high"}:
        raise ValueError("--routed_highpass_target must be one of {logits, x_high}.")
    routed_highpass_enabled = routed_highpass_mode != "off"
    if routed_highpass_enabled and str(training_loader_mode).lower() != "full_batch":
        raise ValueError("routed high-pass correction is only supported with --graph_training_loader_mode full_batch.")
    if routed_highpass_enabled and mhlgc_enabled:
        raise ValueError("--routed_highpass_mode and --mhlgc_enable cannot be enabled together.")
    if routed_highpass_enabled and routed_contrast_enabled:
        raise ValueError("--routed_highpass_mode and --routed_contrast_family cannot be enabled together.")
    if routed_highpass_enabled and routed_highpass_train_node_ids is None:
        raise ValueError("routed high-pass correction requires routed train node ids from --routed_nodes_path.")
    if routed_highpass_enabled and routed_highpass_candidate_rows is None:
        raise ValueError("routed high-pass correction requires prebuilt relation-1hop candidate rows.")
    target_mask_cpu = None
    if mhlgc_target_mask is not None:
        if not torch.is_tensor(mhlgc_target_mask):
            mhlgc_target_mask = torch.as_tensor(mhlgc_target_mask)
        target_mask_cpu = mhlgc_target_mask.detach().cpu().bool().view(-1).contiguous()
    routed_multiview_rows = None
    if mhlgc_routed_multiview_rows is not None:
        if not isinstance(mhlgc_routed_multiview_rows, list):
            raise ValueError("mhlgc_routed_multiview_rows must be a full-graph aligned list when provided.")
        routed_multiview_rows = list(mhlgc_routed_multiview_rows)
    semantic_projector = None
    frozen_projector = None
    routed_contrast_rows = None
    if routed_contrast_multiview_rows is not None:
        if not isinstance(routed_contrast_multiview_rows, list):
            raise ValueError("routed_contrast_multiview_rows must be a full-graph aligned list when provided.")
        routed_contrast_rows = list(routed_contrast_multiview_rows)
    elif routed_multiview_rows is not None:
        routed_contrast_rows = list(routed_multiview_rows)
    routed_train_node_ids_cpu = None
    routed_train_mask_cpu = None
    routed_gate_mask_cpu = None
    routed_gate_stats = {
        "enabled": bool(routed_contrast_enabled),
        "gate": routed_contrast_gate,
        "train_routed_count": 0,
        "eligible_count": 0,
        "reliable_count": 0,
        "reliable_ratio": 0.0,
        "median_mean_similarity": 0.0,
        "mean_support_count": 0.0,
    }
    if routed_contrast_enabled:
        if routed_contrast_train_node_ids is None:
            raise ValueError("routed contrast requires routed_contrast_train_node_ids from --routed_nodes_path split=train.")
        routed_train_node_ids_cpu = _as_long_cpu_tensor(routed_contrast_train_node_ids)
        if int(routed_train_node_ids_cpu.numel()) == 0:
            raise ValueError("routed contrast requires at least one routed train node.")
        routed_train_mask_cpu = torch.zeros((int(x_projected.shape[0]),), dtype=torch.bool)
        routed_train_mask_cpu[routed_train_node_ids_cpu] = True
        routed_gate_stats["train_routed_count"] = int(routed_train_node_ids_cpu.numel())
        if routed_contrast_gate == "heuristic_reliable":
            if routed_contrast_rows is None:
                raise ValueError(
                    "--routed_contrast_gate heuristic_reliable requires routed multiview rows, "
                    "currently loaded from --mhlgc_semantic_embedding_path payloads."
                )
            routed_gate_mask_cpu, computed_gate_stats = _compute_reliability_masks(
                rows=routed_contrast_rows,
                node_ids=routed_train_node_ids_cpu.tolist(),
                edge_index=edge_index,
            )
            routed_gate_stats.update(computed_gate_stats)
        else:
            routed_gate_mask_cpu = torch.ones((int(routed_train_node_ids_cpu.numel()),), dtype=torch.bool)
            routed_gate_stats["eligible_count"] = int(routed_train_node_ids_cpu.numel())
            routed_gate_stats["reliable_count"] = int(routed_train_node_ids_cpu.numel())
            routed_gate_stats["reliable_ratio"] = 1.0
    routed_highpass_train_ids_cpu = _as_long_cpu_tensor(routed_highpass_train_node_ids) if routed_highpass_enabled else torch.empty(0, dtype=torch.long)
    routed_highpass_ids_cpu = _as_long_cpu_tensor(routed_highpass_node_ids) if routed_highpass_enabled else torch.empty(0, dtype=torch.long)
    if routed_highpass_enabled and int(routed_highpass_ids_cpu.numel()) == 0:
        routed_highpass_ids_cpu = routed_highpass_train_ids_cpu
    routed_highpass_train_mask_cpu = torch.zeros((int(x_projected.shape[0]),), dtype=torch.bool)
    routed_highpass_mask_cpu = torch.zeros((int(x_projected.shape[0]),), dtype=torch.bool)
    if routed_highpass_enabled:
        routed_highpass_train_mask_cpu[routed_highpass_train_ids_cpu] = True
        routed_highpass_mask_cpu[routed_highpass_ids_cpu] = True
    train_supervision_mask_cpu = torch.zeros((int(x_projected.shape[0]),), dtype=torch.bool)
    train_supervision_mask_cpu[_as_long_cpu_tensor(train_idx)] = True
    routed_highpass_risk_cpu = routed_highpass_risk_scores.detach().cpu().float() if torch.is_tensor(routed_highpass_risk_scores) else None
    routed_highpass_stats = {
        "enabled": bool(routed_highpass_enabled),
        "mode": routed_highpass_mode,
        "target": routed_highpass_target,
        "candidate_scope": str(routed_highpass_candidate_scope),
        "max_neighbors": int(routed_highpass_max_neighbors),
        "loss_weight": float(routed_highpass_loss_weight),
        "preserve_weight": float(routed_highpass_preserve_weight),
        "risk_gate_weight": float(routed_highpass_risk_gate_weight),
        "edge_role_weight": float(routed_highpass_edge_role_weight),
        "train_routed_count": int(routed_highpass_train_ids_cpu.numel()),
        "routed_count": int(routed_highpass_ids_cpu.numel()),
        "training_contract": (
            "stage0_base_ce_then_frozen_routed_correction"
            if routed_highpass_enabled
            else "disabled"
        ),
        "base_val_accuracy": 0.0,
        "base_val_macro_f1": 0.0,
        "stage2_optimizer_steps": 0,
        "active_steps": 0,
        "total_steps": 0,
        "loss_sum": 0.0,
        "ce_loss_sum": 0.0,
        "preserve_loss_sum": 0.0,
        "gate_loss_sum": 0.0,
        "edge_role_loss_sum": 0.0,
        "gate_self_sum": 0.0,
        "gate_low_sum": 0.0,
        "gate_high_sum": 0.0,
    }
    mhlgc_stats = {
        "enabled": bool(mhlgc_enabled),
        "loss_weight": float(mhlgc_loss_weight),
        "contrast_space": mhlgc_contrast_space,
        "anchor_source": mhlgc_anchor_source,
        "pair_mode": mhlgc_pair_mode,
        "active_steps": 0,
        "total_steps": 0,
        "loss_sum": 0.0,
        "anchor_count_sum": 0,
        "negative_count_sum": 0,
        "negative_candidate_count_sum": 0,
    }
    routed_contrast_stats = {
        "enabled": bool(routed_contrast_enabled),
        "family": routed_contrast_family,
        "gate": routed_contrast_gate,
        "weight": float(routed_contrast_weight),
        "temperature": float(routed_contrast_temperature),
        "active_steps": 0,
        "total_steps": 0,
        "loss_sum": 0.0,
        "view_loss_sum": 0.0,
        "supcon_loss_sum": 0.0,
        "frozen_loss_sum": 0.0,
        "eligible_count_sum": 0,
        "anchor_count_sum": 0,
        "positive_count_sum": 0,
        "negative_count_sum": 0,
    }

    def _record_mhlgc(loss_tensor, stats):
        mhlgc_stats["total_steps"] += 1
        if bool(stats.get("mhlgc_active", False)):
            mhlgc_stats["active_steps"] += 1
            mhlgc_stats["loss_sum"] += float(loss_tensor.detach().cpu().item())
            mhlgc_stats["anchor_count_sum"] += int(stats.get("mhlgc_anchor_count", 0))
            mhlgc_stats["negative_count_sum"] += int(stats.get("mhlgc_negative_count", 0))
            mhlgc_stats["negative_candidate_count_sum"] += int(stats.get("mhlgc_negative_candidate_count", 0))

    def _make_routed_multiview_bundle(node_ids, semantic_rows=None):
        if routed_multiview_rows is None:
            return None
        node_ids_list = [int(item) for item in node_ids]
        rows = [routed_multiview_rows[int(node_id)] for node_id in node_ids_list]
        return {
            "rows": rows,
            "target_node_ids": node_ids_list,
            "semantic_embeddings": semantic_rows,
        }

    def _record_routed_contrast(total_loss_tensor, stats):
        routed_contrast_stats["total_steps"] += 1
        routed_contrast_stats["eligible_count_sum"] += int(stats.get("eligible_count", 0))
        routed_contrast_stats["anchor_count_sum"] += int(stats.get("anchor_count", 0))
        routed_contrast_stats["positive_count_sum"] += int(stats.get("positive_count", 0))
        routed_contrast_stats["negative_count_sum"] += int(stats.get("negative_count", 0))
        if bool(stats.get("active", False)):
            routed_contrast_stats["active_steps"] += 1
            routed_contrast_stats["loss_sum"] += float(total_loss_tensor.detach().cpu().item())
            routed_contrast_stats["view_loss_sum"] += float(stats.get("view_loss", 0.0))
            routed_contrast_stats["supcon_loss_sum"] += float(stats.get("supcon_loss", 0.0))
            routed_contrast_stats["frozen_loss_sum"] += float(stats.get("frozen_loss", 0.0))

    def _record_routed_highpass(loss_tensor, stats):
        routed_highpass_stats["total_steps"] += 1
        if bool(stats.get("active", False)):
            routed_highpass_stats["active_steps"] += 1
            routed_highpass_stats["loss_sum"] += float(loss_tensor.detach().cpu().item())
            routed_highpass_stats["ce_loss_sum"] += float(stats.get("ce_loss", 0.0))
            routed_highpass_stats["preserve_loss_sum"] += float(stats.get("preserve_loss", 0.0))
            routed_highpass_stats["gate_loss_sum"] += float(stats.get("gate_loss", 0.0))
            routed_highpass_stats["edge_role_loss_sum"] += float(stats.get("edge_role_loss", 0.0))
            routed_highpass_stats["gate_self_sum"] += float(stats.get("gate_self_mean", 0.0))
            routed_highpass_stats["gate_low_sum"] += float(stats.get("gate_low_mean", 0.0))
            routed_highpass_stats["gate_high_sum"] += float(stats.get("gate_high_mean", 0.0))

    def _make_routed_highpass_bundle(outputs_dict):
        if not routed_highpass_enabled:
            return None, {"active": False}
        edge_role_head = getattr(getattr(model, "routed_highpass_correction", None), "edge_role_head", None)
        return _build_routed_highpass_bundle(
            outputs_dict=outputs_dict,
            labels_tensor=y,
            train_mask=train_supervision_mask_cpu.to(device),
            routed_mask=routed_highpass_mask_cpu.to(device),
            candidate_rows=routed_highpass_candidate_rows,
            risk_scores=(routed_highpass_risk_cpu.to(device) if routed_highpass_risk_cpu is not None else None),
            mode=routed_highpass_mode,
            candidate_scope=routed_highpass_candidate_scope,
            max_neighbors=int(routed_highpass_max_neighbors),
            device=device,
            edge_role_head=edge_role_head,
            target=routed_highpass_target,
        )

    def _compute_routed_highpass_loss(base_outputs, corrected_outputs):
        zero = corrected_outputs["logits"].sum() * 0.0
        if not routed_highpass_enabled:
            return zero, {"active": False}
        aux = corrected_outputs.get("routed_highpass_aux")
        if not isinstance(aux, dict) or not torch.is_tensor(aux.get("gate")):
            return zero, {"active": False}
        routed_ids = routed_highpass_train_ids_cpu.to(device)
        if int(routed_ids.numel()) == 0:
            return zero, {"active": False}
        logits = corrected_outputs["logits"]
        base_logits = base_outputs["logits"].detach()
        ce_loss = F.cross_entropy(logits[routed_ids], y[routed_ids])
        non_routed_train_mask = train_supervision_mask_cpu.to(device).bool()
        non_routed_train_mask = non_routed_train_mask & (~routed_highpass_train_mask_cpu.to(device).bool())
        if bool(non_routed_train_mask.any().item()):
            preserve_loss = F.kl_div(
                F.log_softmax(logits[non_routed_train_mask], dim=-1),
                F.softmax(base_logits[non_routed_train_mask], dim=-1),
                reduction="batchmean",
            )
        else:
            preserve_loss = zero
        gate = aux["gate"]
        gate_train_mask = train_supervision_mask_cpu.to(device).bool()
        target_high = routed_highpass_train_mask_cpu.to(device).to(dtype=gate.dtype)
        if routed_highpass_risk_cpu is not None:
            target_high = torch.maximum(target_high, routed_highpass_risk_cpu.to(device=device, dtype=gate.dtype))
        if bool(gate_train_mask.any().item()):
            gate_loss = F.binary_cross_entropy(
                gate[gate_train_mask, 2].clamp(1e-6, 1.0 - 1e-6),
                target_high[gate_train_mask],
            )
        else:
            gate_loss = zero
        edge_role_loss = zero
        bundle = corrected_outputs.get("_routed_highpass_bundle")
        if isinstance(bundle, dict) and torch.is_tensor(bundle.get("edge_role_logits")):
            edge_logits = bundle["edge_role_logits"].to(device)
            edge_targets = bundle["edge_role_targets"].to(device)
            if int(edge_targets.numel()) > 0:
                edge_role_loss = F.binary_cross_entropy_with_logits(
                    edge_logits.view(-1),
                    edge_targets.view(-1).to(dtype=edge_logits.dtype),
                )
        total = (
            float(routed_highpass_loss_weight) * ce_loss
            + float(routed_highpass_preserve_weight) * preserve_loss
            + float(routed_highpass_risk_gate_weight) * gate_loss
            + float(routed_highpass_edge_role_weight) * edge_role_loss
        )
        active_gate = gate[routed_ids]
        stats = {
            "active": True,
            "ce_loss": float(ce_loss.detach().cpu().item()),
            "preserve_loss": float(preserve_loss.detach().cpu().item()),
            "gate_loss": float(gate_loss.detach().cpu().item()),
            "edge_role_loss": float(edge_role_loss.detach().cpu().item()),
            "gate_self_mean": float(active_gate[:, 0].detach().mean().cpu().item()) if int(active_gate.numel()) else 0.0,
            "gate_low_mean": float(active_gate[:, 1].detach().mean().cpu().item()) if int(active_gate.numel()) else 0.0,
            "gate_high_mean": float(active_gate[:, 2].detach().mean().cpu().item()) if int(active_gate.numel()) else 0.0,
        }
        return total, stats

    def _compute_routed_contrast(outputs_dict, labels_tensor, node_feature_frozen):
        nonlocal frozen_projector
        zero = outputs_dict["logits"].sum() * 0.0
        if not routed_contrast_enabled:
            return zero, {"active": False, "eligible_count": 0, "anchor_count": 0, "positive_count": 0, "negative_count": 0}
        x_low_all = outputs_dict.get("x_low")
        x_high_all = outputs_dict.get("x_high")
        fused_all = outputs_dict.get("fused_x")
        if not torch.is_tensor(fused_all):
            fused_all = outputs_dict.get("node_repr")
        if not torch.is_tensor(fused_all):
            raise ValueError("routed contrast requires outputs['fused_x'] or outputs['node_repr'].")
        if routed_contrast_family in {"low_high", "three_view_control", "hybrid"}:
            if not torch.is_tensor(x_low_all) or not torch.is_tensor(x_high_all):
                raise ValueError(
                    "routed contrast families {low_high, three_view_control, hybrid} require a HyperScan-style "
                    "backbone that exposes outputs['x_low'] and outputs['x_high']."
                )
        eligible_ids_cpu = routed_train_node_ids_cpu
        if routed_contrast_gate == "heuristic_reliable":
            eligible_ids_cpu = routed_train_node_ids_cpu[routed_gate_mask_cpu]
        eligible_count = int(eligible_ids_cpu.numel())
        base_stats = {
            "active": False,
            "eligible_count": int(eligible_count),
            "anchor_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "view_loss": 0.0,
            "supcon_loss": 0.0,
            "frozen_loss": 0.0,
        }
        if eligible_count < 2:
            return zero, base_stats
        eligible_ids = eligible_ids_cpu.to(device)
        total_loss = zero
        if routed_contrast_family in {"low_high", "three_view_control", "hybrid"}:
            view_loss, view_stats = routed_same_node_contrastive_loss(
                x_low_all[eligible_ids],
                x_high_all[eligible_ids],
                temperature=float(routed_contrast_temperature),
            )
            total_loss = total_loss + routed_contrast_weight * view_loss
            base_stats["active"] = bool(view_stats.get("active", False))
            base_stats["anchor_count"] = int(view_stats.get("anchor_count", 0))
            base_stats["positive_count"] = int(view_stats.get("positive_count", 0))
            base_stats["negative_count"] = int(view_stats.get("negative_count", 0))
            base_stats["view_loss"] = float(view_loss.detach().cpu().item())
            if routed_contrast_family == "three_view_control":
                frozen_selected = node_feature_frozen[eligible_ids_cpu]
                frozen_projected, frozen_projector = _project_routed_frozen_embeddings(
                    frozen_selected,
                    target_dim=int(x_low_all.shape[1]),
                    frozen_projector=frozen_projector,
                    optimizer=optimizer,
                    device=device,
                )
                frozen_loss, frozen_stats = routed_frozen_alignment_loss(
                    frozen_projected,
                    x_low_all[eligible_ids],
                )
                total_loss = total_loss + (0.5 * routed_contrast_weight) * frozen_loss
                base_stats["active"] = bool(base_stats["active"] or frozen_stats.get("active", False))
                base_stats["frozen_loss"] = float(frozen_loss.detach().cpu().item())
        if routed_contrast_family in {"supcon_class", "hybrid"}:
            supcon_loss, supcon_stats = routed_supervised_contrastive_loss(
                fused_all[eligible_ids],
                labels_tensor[eligible_ids],
                temperature=float(routed_contrast_temperature),
            )
            total_loss = total_loss + routed_contrast_weight * supcon_loss
            base_stats["active"] = bool(base_stats["active"] or supcon_stats.get("active", False))
            base_stats["anchor_count"] = max(int(base_stats["anchor_count"]), int(supcon_stats.get("anchor_count", 0)))
            base_stats["positive_count"] = max(int(base_stats["positive_count"]), int(supcon_stats.get("positive_pair_count", 0)))
            base_stats["negative_count"] = max(int(base_stats["negative_count"]), int(supcon_stats.get("negative_pair_count", 0)))
            base_stats["supcon_loss"] = float(supcon_loss.detach().cpu().item())
        if routed_contrast_family == "bot_edge_mask_human":
            bot_mask = labels_tensor[eligible_ids] == 1
            human_mask = labels_tensor[eligible_ids] == 0
            bot_ids = eligible_ids[bot_mask]
            human_ids = eligible_ids[human_mask]
            if int(bot_ids.numel()) < 1 or int(human_ids.numel()) < 1:
                return zero, base_stats
            edge_index_aug_local, edge_type_aug_local = mhlgc_mask_edges(
                edge_index,
                edge_type,
                mask_probability=float(mhlgc_edge_mask_probability),
            )
            positive_outputs = model.forward_outputs(
                x,
                edge_index_aug_local,
                edge_type_aug_local,
                construct_x=x_construct,
                routed_multiview_bundle=routed_bundle_full,
            )
            positive_fused = positive_outputs.get("fused_x")
            if not torch.is_tensor(positive_fused):
                positive_fused = positive_outputs.get("node_repr")
            bot_hard_loss, bot_hard_stats = routed_bot_edge_mask_human_contrastive_loss(
                anchor_embeddings=fused_all[bot_ids],
                positive_embeddings=positive_fused[bot_ids],
                negative_embeddings=fused_all[human_ids],
                temperature=float(routed_contrast_temperature),
            )
            total_loss = total_loss + routed_contrast_weight * bot_hard_loss
            base_stats["active"] = bool(base_stats["active"] or bot_hard_stats.get("active", False))
            base_stats["anchor_count"] = max(int(base_stats["anchor_count"]), int(bot_hard_stats.get("anchor_count", 0)))
            base_stats["positive_count"] = max(int(base_stats["positive_count"]), int(bot_hard_stats.get("positive_count", 0)))
            base_stats["negative_count"] = max(int(base_stats["negative_count"]), int(bot_hard_stats.get("negative_count", 0)))
            base_stats["supcon_loss"] = float(bot_hard_loss.detach().cpu().item())
        return total_loss, base_stats

    if str(training_loader_mode).lower() == "neighbor_subgraph":
        if neighborloader_contract == "hyperscan_sampled_subgraph":
            best_checkpoint_primary = "validation_accuracy"
            neighborloader_contract_metrics.update(
                {
                    "supervision_scope": "sampled_subgraph_all_rows",
                    "validation_scope": "sampled_subgraph_all_rows",
                    "test_scope": "sampled_subgraph_all_rows",
                    "duplicate_counting": "repeated_batch_rows",
                    "checkpoint_selection_primary": "validation_accuracy",
                }
            )
        edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
        edge_type_cpu = edge_type.detach().cpu().long().view(-1) if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long).view(-1)
        x_projected_cpu = x_projected.detach().cpu().float() if torch.is_tensor(x_projected) else torch.tensor(x_projected, dtype=torch.float32)
        x_raw_cpu = x_raw.detach().cpu().float() if torch.is_tensor(x_raw) else torch.tensor(x_raw, dtype=torch.float32)
        y_cpu = y.detach().cpu().long() if torch.is_tensor(y) else torch.tensor(y, dtype=torch.long)
        graph_num_nodes = int(x_projected_cpu.shape[0])
        labeled_node_count = int(y_cpu.numel())
        y_full_cpu = _expand_labeled_targets_to_graph(y_cpu, graph_num_nodes)
        train_idx_cpu = train_idx.detach().cpu().long().view(-1)
        valid_idx_cpu = valid_idx.detach().cpu().long().view(-1)
        n_layers = int(config.get("gnn_n_layers", config.get("n_layers", 2)))
        loader_neighbors = [int(neighbor_num_neighbors)] * max(int(n_layers), 1)
        loader_data = Data(
            x=x_projected_cpu,
            raw_x=x_raw_cpu,
            construct_x=(
                x_construct.detach().cpu().float()
                if torch.is_tensor(x_construct)
                else torch.tensor(x_construct, dtype=torch.float32)
            ),
            y=y_full_cpu,
            edge_index=edge_index_cpu,
            edge_type=edge_type_cpu,
            node_id=torch.arange(graph_num_nodes, dtype=torch.long),
        )
        loader_data.num_nodes = graph_num_nodes
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
                construct_batch = batch.construct_x if hasattr(batch, "construct_x") else None
                semantic_batch = None
                semantic_batch_all = None
                if mhlgc_semantic_embeddings is not None:
                    semantic_batch = mhlgc_semantic_embeddings[
                        batch.node_id[: int(batch.batch_size)].detach().cpu().long()
                    ].to(device)
                    semantic_batch_all = mhlgc_semantic_embeddings[
                        batch.node_id.detach().cpu().long()
                    ].to(device)
                routed_bundle_batch = _make_routed_multiview_bundle(
                    batch.node_id.detach().cpu().long().tolist(),
                    semantic_rows=semantic_batch_all,
                )
                batch_outputs = model.forward_outputs(
                    x_batch,
                    batch.edge_index,
                    batch.edge_type.view(-1),
                    construct_x=construct_batch,
                    routed_multiview_bundle=routed_bundle_batch,
                    batch_node_ids=batch.node_id,
                )
                logits = batch_outputs["logits"]
                seed_count = int(batch.batch_size)
                if neighborloader_contract == "hyperscan_sampled_subgraph":
                    loss = F.cross_entropy(logits, batch.y)
                else:
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
                    aug_outputs = model.forward_outputs(
                        x_aug,
                        edge_index_aug,
                        edge_type_aug,
                        construct_x=construct_batch,
                        hyperedge_mask_probability=float(mhlgc_hyperedge_mask_probability),
                        routed_multiview_bundle=routed_bundle_batch,
                        batch_node_ids=batch.node_id,
                    )
                    anchor_mask_batch = None
                    repair_mask_batch_all = None
                    if mhlgc_anchor_source == "routed_target_mask" and target_mask_cpu is not None:
                        anchor_mask_batch = target_mask_cpu[
                            batch.node_id[:seed_count].detach().cpu().long()
                        ].to(device)
                        repair_mask_batch_all = target_mask_cpu[
                            batch.node_id.detach().cpu().long()
                        ].to(device)
                    elif mhlgc_anchor_source == "positive_label":
                        anchor_mask_batch = torch.ones((seed_count,), dtype=torch.bool, device=device)
                        repair_mask_batch_all = torch.ones((int(batch.node_id.numel()),), dtype=torch.bool, device=device)
                    origin_embeddings = _select_mhlgc_contrast_embeddings(
                        batch_outputs,
                        x_batch[:seed_count],
                        torch.arange(seed_count, device=device),
                        mhlgc_contrast_space,
                    )
                    if mhlgc_pair_mode == "repair_aware":
                        if semantic_batch is None:
                            raise ValueError("--mhlgc_pair_mode repair_aware requires --mhlgc_semantic_embedding_path.")
                        if mhlgc_contrast_space == "semantic":
                            semantic_positive, semantic_projector = _project_mhlgc_semantic(
                                semantic_batch,
                                target_dim=int(origin_embeddings.shape[1]),
                                semantic_projector=semantic_projector,
                                optimizer=optimizer,
                                device=device,
                                projector_mode=mhlgc_semantic_projector,
                            )
                            guided_gate = (
                                anchor_mask_batch.float().unsqueeze(1)
                                if anchor_mask_batch is not None
                                else (semantic_batch.norm(dim=1, keepdim=True) > 0).float()
                            )
                            guided_gate = guided_gate.to(origin_embeddings.dtype)
                            augmented_embeddings = origin_embeddings + guided_gate * semantic_positive
                        else:
                            x_new_batch = batch_outputs.get("x_new")
                            if not torch.is_tensor(x_new_batch):
                                raise ValueError(
                                    "--mhlgc_pair_mode repair_aware with contrast spaces fused_x/node_repr/x_new "
                                    "requires a HyperScan-style backbone that exposes outputs['x_new'] for "
                                    "second-view repair."
                                )
                            semantic_repair_all, semantic_projector = _project_mhlgc_semantic(
                                semantic_batch_all,
                                target_dim=int(x_new_batch.shape[1]),
                                semantic_projector=semantic_projector,
                                optimizer=optimizer,
                                device=device,
                                projector_mode=mhlgc_semantic_projector,
                            )
                            if repair_mask_batch_all is None:
                                repair_mask_batch_all = semantic_batch_all.norm(dim=1) > 0.0
                            repaired_outputs = model.forward_outputs(
                                x_batch,
                                batch.edge_index,
                                batch.edge_type.view(-1),
                                construct_x=construct_batch,
                                second_view_repair_delta=semantic_repair_all,
                                second_view_repair_mask=repair_mask_batch_all,
                                routed_multiview_bundle=routed_bundle_batch,
                                batch_node_ids=batch.node_id,
                            )
                            augmented_embeddings = _select_mhlgc_contrast_embeddings(
                                repaired_outputs,
                                x_batch[:seed_count],
                                torch.arange(seed_count, device=device),
                                mhlgc_contrast_space,
                            )
                    else:
                        augmented_embeddings = _select_mhlgc_contrast_embeddings(
                            aug_outputs,
                            x_aug[:seed_count],
                            torch.arange(seed_count, device=device),
                            mhlgc_contrast_space,
                        )
                    mhlgc_loss, mhlgc_step_stats = mhlgc_llm_guided_contrastive_loss(
                        origin_embeddings=origin_embeddings,
                        augmented_embeddings=augmented_embeddings,
                        labels=batch.y[:seed_count],
                        fraud_scores=torch.softmax(logits[:seed_count], dim=-1)[:, int(mhlgc_positive_label)],
                        semantic_embeddings=semantic_batch,
                        positive_label=int(mhlgc_positive_label),
                        anchors_per_batch=int(mhlgc_anchors_per_batch),
                        beta=float(mhlgc_beta),
                        gamma=float(mhlgc_gamma if semantic_batch is not None else 0.0),
                        temperature=float(mhlgc_temperature),
                        negative_count=int(mhlgc_negative_count),
                        anchor_mask=anchor_mask_batch,
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
            eval_logits_by_node = None
            eval_seen = torch.zeros((labeled_node_count,), dtype=torch.bool)
            repeated_eval_logits = []
            repeated_eval_labels = []
            with torch.no_grad():
                for batch in valid_loader:
                    batch = batch.to(device)
                    x_eval = input_adapter(batch.x, batch.raw_x) if input_adapter is not None else batch.x
                    semantic_eval = None
                    if mhlgc_semantic_embeddings is not None:
                        semantic_eval = mhlgc_semantic_embeddings[
                            batch.node_id.detach().cpu().long()
                        ].to(device)
                    routed_bundle_eval = _make_routed_multiview_bundle(
                        batch.node_id.detach().cpu().long().tolist(),
                        semantic_rows=semantic_eval,
                    )
                    logits_eval = model(
                        x_eval,
                        batch.edge_index,
                        batch.edge_type.view(-1),
                        construct_x=(batch.construct_x if hasattr(batch, "construct_x") else None),
                        routed_multiview_bundle=routed_bundle_eval,
                        batch_node_ids=batch.node_id,
                    )
                    if neighborloader_contract == "hyperscan_sampled_subgraph":
                        repeated_eval_logits.append(logits_eval.detach().cpu())
                        repeated_eval_labels.append(batch.y.detach().cpu().long())
                    else:
                        seed_count = int(batch.batch_size)
                        global_ids = batch.node_id[:seed_count].detach().cpu().long()
                        batch_logits = logits_eval[:seed_count].detach().cpu()
                        valid_seed_mask = (global_ids >= 0) & (global_ids < labeled_node_count)
                        if not bool(valid_seed_mask.any()):
                            continue
                        if eval_logits_by_node is None:
                            eval_logits_by_node = torch.zeros(
                                (labeled_node_count, int(batch_logits.shape[1])),
                                dtype=batch_logits.dtype,
                            )
                        eval_ids = global_ids[valid_seed_mask]
                        eval_logits_by_node[eval_ids] = batch_logits[valid_seed_mask]
                        eval_seen[eval_ids] = True
            if neighborloader_contract == "hyperscan_sampled_subgraph":
                if not repeated_eval_logits:
                    raise RuntimeError("NeighborLoader validation did not produce any sampled-subgraph logits.")
                val_metrics = _score_repeated_neighborloader_logits(
                    torch.cat(repeated_eval_logits, dim=0),
                    torch.cat(repeated_eval_labels, dim=0),
                )
            else:
                if eval_logits_by_node is None or not bool(torch.all(eval_seen[valid_idx_cpu]).item()):
                    missing = valid_idx_cpu[~eval_seen[valid_idx_cpu]]
                    raise RuntimeError(
                        "NeighborLoader validation did not produce aligned logits for every valid node. "
                        f"Missing {int(missing.numel())} ids."
                    )
                val_metrics = _score_logits(eval_logits_by_node, y_cpu, valid_idx_cpu)
            val_metrics["epoch"] = int(epoch)
            if _is_better_with_policy(
                val_metrics,
                best_metrics,
                primary=best_checkpoint_primary,
                tie_breaker=best_checkpoint_tie_breaker,
            ):
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
            semantic_all = mhlgc_semantic_embeddings.to(device) if mhlgc_semantic_embeddings is not None else None
            routed_bundle_full = _make_routed_multiview_bundle(
                list(range(int(x.shape[0]))),
                semantic_rows=semantic_all,
            )
            base_outputs = model.forward_outputs(
                x,
                edge_index,
                edge_type,
                construct_x=x_construct,
                routed_multiview_bundle=routed_bundle_full,
            )
            outputs = base_outputs
            logits = outputs["logits"]
            loss = F.cross_entropy(logits[train_idx], y[train_idx])
            if routed_contrast_enabled:
                routed_frozen_train = (
                    routed_contrast_frozen_embeddings
                    if routed_contrast_frozen_embeddings is not None
                    else x_projected.detach().cpu()
                )
                routed_loss, routed_step_stats = _compute_routed_contrast(
                    outputs_dict=outputs,
                    labels_tensor=y,
                    node_feature_frozen=routed_frozen_train,
                )
                _record_routed_contrast(routed_loss, routed_step_stats)
                loss = loss + routed_loss
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
                aug_outputs = model.forward_outputs(
                    x_aug,
                    edge_index_aug,
                    edge_type_aug,
                    construct_x=x_construct,
                    hyperedge_mask_probability=float(mhlgc_hyperedge_mask_probability),
                    routed_multiview_bundle=routed_bundle_full,
                )
                semantic_train = (
                    mhlgc_semantic_embeddings[train_idx.detach().cpu().long()].to(device)
                    if mhlgc_semantic_embeddings is not None
                    else None
                )
                anchor_mask_train = None
                repair_mask_all = None
                if mhlgc_anchor_source == "routed_target_mask" and target_mask_cpu is not None:
                    anchor_mask_train = target_mask_cpu[train_idx.detach().cpu().long()].to(device)
                    repair_mask_all = target_mask_cpu.to(device)
                elif mhlgc_anchor_source == "positive_label":
                    anchor_mask_train = torch.ones((int(train_idx.numel()),), dtype=torch.bool, device=device)
                    repair_mask_all = torch.ones((int(x.shape[0]),), dtype=torch.bool, device=device)
                origin_embeddings = _select_mhlgc_contrast_embeddings(
                    outputs,
                    x,
                    train_idx,
                    mhlgc_contrast_space,
                )
                if mhlgc_pair_mode == "repair_aware":
                    if semantic_train is None:
                        raise ValueError("--mhlgc_pair_mode repair_aware requires --mhlgc_semantic_embedding_path.")
                    if mhlgc_contrast_space == "semantic":
                        semantic_positive, semantic_projector = _project_mhlgc_semantic(
                            semantic_train,
                            target_dim=int(origin_embeddings.shape[1]),
                            semantic_projector=semantic_projector,
                            optimizer=optimizer,
                            device=device,
                            projector_mode=mhlgc_semantic_projector,
                        )
                        guided_gate = (
                            anchor_mask_train.float().unsqueeze(1)
                            if anchor_mask_train is not None
                            else (semantic_train.norm(dim=1, keepdim=True) > 0).float()
                        )
                        guided_gate = guided_gate.to(origin_embeddings.dtype)
                        augmented_embeddings = origin_embeddings + guided_gate * semantic_positive
                    else:
                        x_new_full = outputs.get("x_new")
                        if not torch.is_tensor(x_new_full):
                            raise ValueError(
                                "--mhlgc_pair_mode repair_aware with contrast spaces fused_x/node_repr/x_new "
                                "requires a HyperScan-style backbone that exposes outputs['x_new'] for "
                                "second-view repair."
                            )
                        semantic_repair_all, semantic_projector = _project_mhlgc_semantic(
                            semantic_all,
                            target_dim=int(x_new_full.shape[1]),
                            semantic_projector=semantic_projector,
                            optimizer=optimizer,
                            device=device,
                            projector_mode=mhlgc_semantic_projector,
                        )
                        if repair_mask_all is None:
                            repair_mask_all = semantic_all.norm(dim=1) > 0.0
                        repaired_outputs = model.forward_outputs(
                            x,
                            edge_index,
                            edge_type,
                            construct_x=x_construct,
                            second_view_repair_delta=semantic_repair_all,
                            second_view_repair_mask=repair_mask_all,
                            routed_multiview_bundle=routed_bundle_full,
                        )
                        augmented_embeddings = _select_mhlgc_contrast_embeddings(
                            repaired_outputs,
                            x,
                            train_idx,
                            mhlgc_contrast_space,
                        )
                else:
                    augmented_embeddings = _select_mhlgc_contrast_embeddings(
                        aug_outputs,
                        x_aug,
                        train_idx,
                        mhlgc_contrast_space,
                    )
                mhlgc_loss, mhlgc_step_stats = mhlgc_llm_guided_contrastive_loss(
                    origin_embeddings=origin_embeddings,
                    augmented_embeddings=augmented_embeddings,
                    labels=y[train_idx],
                    fraud_scores=torch.softmax(logits[train_idx], dim=-1)[:, int(mhlgc_positive_label)],
                    semantic_embeddings=semantic_train,
                    positive_label=int(mhlgc_positive_label),
                    anchors_per_batch=int(mhlgc_anchors_per_batch),
                    beta=float(mhlgc_beta),
                    gamma=float(mhlgc_gamma if semantic_train is not None else 0.0),
                    temperature=float(mhlgc_temperature),
                    negative_count=int(mhlgc_negative_count),
                    anchor_mask=anchor_mask_train,
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
                semantic_eval_full = mhlgc_semantic_embeddings.to(device) if mhlgc_semantic_embeddings is not None else None
                routed_bundle_eval = _make_routed_multiview_bundle(
                    list(range(int(x_eval.shape[0]))),
                    semantic_rows=semantic_eval_full,
                )
                eval_outputs = model.forward_outputs(
                    x_eval,
                    edge_index,
                    edge_type,
                    construct_x=x_construct,
                    routed_multiview_bundle=routed_bundle_eval,
                )
                logits_eval = eval_outputs["logits"]
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

    if routed_highpass_enabled:
        routed_highpass_stats["base_val_accuracy"] = float((best_metrics or {}).get("accuracy", 0.0))
        routed_highpass_stats["base_val_macro_f1"] = float((best_metrics or {}).get("macro_f1", 0.0))
        correction_module = getattr(model, "routed_highpass_correction", None)
        if correction_module is None:
            raise ValueError("--routed_highpass_mode requires a graph backbone with routed high-pass correction support.")
        for parameter in model.parameters():
            parameter.requires_grad = False
        if input_adapter is not None:
            for parameter in input_adapter.parameters():
                parameter.requires_grad = False
        for parameter in correction_module.parameters():
            parameter.requires_grad = True
        correction_optimizer = torch.optim.AdamW(
            [parameter for parameter in correction_module.parameters() if parameter.requires_grad],
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        correction_best_metrics = None
        correction_best_state = None
        correction_best_input_adapter_state = (
            {key: value.detach().cpu().clone() for key, value in input_adapter.state_dict().items()}
            if input_adapter is not None
            else None
        )
        stage2_steps = 0
        semantic_stage2 = mhlgc_semantic_embeddings.to(device) if mhlgc_semantic_embeddings is not None else None

        def _stage2_base_outputs():
            x_stage2 = input_adapter(x_projected, x_raw) if input_adapter is not None else x_projected
            routed_bundle_stage2 = _make_routed_multiview_bundle(
                list(range(int(x_stage2.shape[0]))),
                semantic_rows=semantic_stage2,
            )
            return x_stage2, routed_bundle_stage2, model.forward_outputs(
                x_stage2,
                edge_index,
                edge_type,
                construct_x=x_construct,
                routed_multiview_bundle=routed_bundle_stage2,
            )

        for epoch in range(max_epochs):
            model.eval()
            correction_module.train()
            correction_optimizer.zero_grad()
            with torch.no_grad():
                x_stage2, routed_bundle_stage2, base_outputs = _stage2_base_outputs()
            highpass_bundle, highpass_build_stats = _make_routed_highpass_bundle(base_outputs)
            outputs = model.forward_outputs(
                x_stage2,
                edge_index,
                edge_type,
                construct_x=x_construct,
                routed_multiview_bundle=routed_bundle_stage2,
                routed_highpass_bundle=highpass_bundle,
            )
            outputs["_routed_highpass_bundle"] = highpass_bundle
            highpass_loss, highpass_step_stats = _compute_routed_highpass_loss(
                base_outputs=base_outputs,
                corrected_outputs=outputs,
            )
            highpass_step_stats.update(highpass_build_stats)
            _record_routed_highpass(highpass_loss, highpass_step_stats)
            highpass_loss.backward()
            correction_optimizer.step()
            optimizer_steps += 1
            stage2_steps += 1

            model.eval()
            correction_module.eval()
            with torch.no_grad():
                x_eval, routed_bundle_eval, eval_base_outputs = _stage2_base_outputs()
                eval_highpass_bundle, _ = _make_routed_highpass_bundle(eval_base_outputs)
                eval_outputs = model.forward_outputs(
                    x_eval,
                    edge_index,
                    edge_type,
                    construct_x=x_construct,
                    routed_multiview_bundle=routed_bundle_eval,
                    routed_highpass_bundle=eval_highpass_bundle,
                )
                logits_eval = eval_outputs["logits"]
            val_metrics = _score_logits(logits_eval.detach().cpu(), y.detach().cpu(), valid_idx.detach().cpu())
            val_metrics["epoch"] = int(epoch)
            val_metrics["stage"] = "routed_highpass_correction"
            if _is_better(val_metrics, correction_best_metrics):
                correction_best_metrics = val_metrics
                correction_best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if max_update_steps > 0 and optimizer_steps >= max_update_steps:
                break

        routed_highpass_stats["stage2_optimizer_steps"] = int(stage2_steps)
        if correction_best_state is not None:
            model.load_state_dict(correction_best_state)
            best_state = {
                "model": correction_best_state,
                "input_adapter": correction_best_input_adapter_state,
            }
            best_metrics = correction_best_metrics or best_metrics

    model.eval()
    if input_adapter is not None:
        input_adapter.eval()
    if str(training_loader_mode).lower() == "neighbor_subgraph":
        edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
        edge_type_cpu = edge_type.detach().cpu().long().view(-1) if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long).view(-1)
        x_projected_cpu = x_projected.detach().cpu().float() if torch.is_tensor(x_projected) else torch.tensor(x_projected, dtype=torch.float32)
        x_raw_cpu = x_raw.detach().cpu().float() if torch.is_tensor(x_raw) else torch.tensor(x_raw, dtype=torch.float32)
        y_cpu = y.detach().cpu().long() if torch.is_tensor(y) else torch.tensor(y, dtype=torch.long)
        y_full_cpu = _expand_labeled_targets_to_graph(y_cpu, int(x_projected_cpu.shape[0]))
        n_layers = int(config.get("gnn_n_layers", config.get("n_layers", 2)))
        infer_data = Data(
            x=x_projected_cpu,
            raw_x=x_raw_cpu,
            construct_x=(
                x_construct.detach().cpu().float()
                if torch.is_tensor(x_construct)
                else torch.tensor(x_construct, dtype=torch.float32)
            ),
            y=y_full_cpu,
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
        fused_x_full = None
        x_low_full = None
        x_new_full = None
        extra_full_tensors = {}
        aux_accumulator = {}
        aux_weight = 0.0
        extra_tensor_keys = [
            "x_ego_dec",
            "x_rel_low1_dec",
            "x_rel_high1_dec",
            "x_rel_state1_dec",
            "x_rel_low2_dec",
            "x_rel_high2_dec",
            "x_rel_state2_dec",
            "x_clean_base",
            "x_new_clean",
            "x_high_low1_clean",
            "x_high_high1_clean",
            "x_high_state1_clean",
            "x_high_low2_clean",
            "x_high_high2_clean",
            "x_high_clean",
            "x_high",
        ]
        repeated_test_logits = []
        repeated_test_labels = []
        with torch.no_grad():
            for batch in infer_loader:
                batch = batch.to(device)
                x_final = input_adapter(batch.x, batch.raw_x) if input_adapter is not None else batch.x
                semantic_infer = None
                if mhlgc_semantic_embeddings is not None:
                    semantic_infer = mhlgc_semantic_embeddings[
                        batch.node_id.detach().cpu().long()
                    ].to(device)
                routed_bundle_infer = _make_routed_multiview_bundle(
                    batch.node_id.detach().cpu().long().tolist(),
                    semantic_rows=semantic_infer,
                )
                batch_outputs = model.forward_outputs(
                    x_final,
                    batch.edge_index,
                    batch.edge_type.view(-1),
                    construct_x=(batch.construct_x if hasattr(batch, "construct_x") else None),
                    routed_multiview_bundle=routed_bundle_infer,
                    batch_node_ids=batch.node_id,
                )
                if neighborloader_contract == "hyperscan_sampled_subgraph":
                    repeated_test_logits.append(batch_outputs["logits"].detach().cpu())
                    repeated_test_labels.append(batch.y.detach().cpu().long())
                seed_count = int(batch.batch_size)
                global_ids = batch.node_id[:seed_count].detach().cpu().long()
                batch_logits = batch_outputs["logits"][:seed_count].detach().cpu()
                batch_prob = batch_outputs["prob"][:seed_count].detach().cpu()
                batch_repr = batch_outputs["node_repr"][:seed_count].detach().cpu()
                batch_fused_x = batch_outputs.get("fused_x")
                if torch.is_tensor(batch_fused_x):
                    batch_fused_x = batch_fused_x[:seed_count].detach().cpu()
                else:
                    batch_fused_x = batch_repr
                batch_x_low = batch_outputs.get("x_low")
                if torch.is_tensor(batch_x_low):
                    batch_x_low = batch_x_low[:seed_count].detach().cpu()
                else:
                    batch_x_low = None
                batch_x_new = batch_outputs.get("x_new")
                if torch.is_tensor(batch_x_new):
                    batch_x_new = batch_x_new[:seed_count].detach().cpu()
                else:
                    batch_x_new = None
                batch_extra_tensors = {}
                for key in extra_tensor_keys:
                    value = batch_outputs.get(key)
                    if torch.is_tensor(value):
                        batch_extra_tensors[key] = value[:seed_count].detach().cpu()
                if logits_full is None:
                    logits_full = torch.zeros((int(x_projected_cpu.shape[0]), int(batch_logits.shape[1])), dtype=batch_logits.dtype)
                    prob_full = torch.zeros((int(x_projected_cpu.shape[0]), int(batch_prob.shape[1])), dtype=batch_prob.dtype)
                    repr_full = torch.zeros((int(x_projected_cpu.shape[0]), int(batch_repr.shape[1])), dtype=batch_repr.dtype)
                    fused_x_full = torch.zeros((int(x_projected_cpu.shape[0]), int(batch_fused_x.shape[1])), dtype=batch_fused_x.dtype)
                if batch_x_low is not None and x_low_full is None:
                    x_low_full = torch.zeros((int(x_projected_cpu.shape[0]), int(batch_x_low.shape[1])), dtype=batch_x_low.dtype)
                if batch_x_new is not None and x_new_full is None:
                    x_new_full = torch.zeros((int(x_projected_cpu.shape[0]), int(batch_x_new.shape[1])), dtype=batch_x_new.dtype)
                logits_full[global_ids] = batch_logits
                prob_full[global_ids] = batch_prob
                repr_full[global_ids] = batch_repr
                fused_x_full[global_ids] = batch_fused_x
                if batch_x_low is not None and x_low_full is not None:
                    x_low_full[global_ids] = batch_x_low
                if batch_x_new is not None and x_new_full is not None:
                    x_new_full[global_ids] = batch_x_new
                for key, value in batch_extra_tensors.items():
                    if key not in extra_full_tensors:
                        extra_full_tensors[key] = torch.zeros(
                            (int(x_projected_cpu.shape[0]), int(value.shape[1])),
                            dtype=value.dtype,
                        )
                    extra_full_tensors[key][global_ids] = value
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
        if neighborloader_contract == "hyperscan_sampled_subgraph":
            if not repeated_test_logits:
                raise RuntimeError("NeighborLoader test/infer pass did not produce any sampled-subgraph logits.")
            neighborloader_contract_metrics["test_best_checkpoint"] = _score_repeated_neighborloader_logits(
                torch.cat(repeated_test_logits, dim=0),
                torch.cat(repeated_test_labels, dim=0),
            )
        outputs = {
            "logits": logits_full if logits_full is not None else torch.empty((0, 2), dtype=torch.float32),
            "prob": prob_full if prob_full is not None else torch.empty((0, 2), dtype=torch.float32),
            "node_repr": repr_full if repr_full is not None else torch.empty((0, config["hidden_dim"]), dtype=torch.float32),
            "fused_x": fused_x_full if fused_x_full is not None else torch.empty((0, config["hidden_dim"]), dtype=torch.float32),
            "aux_features": {
                "dynamic_similarity_branch": aggregated_aux,
            },
        }
        if x_low_full is not None:
            outputs["x_low"] = x_low_full
        if x_new_full is not None:
            outputs["x_new"] = x_new_full
        for key, value in extra_full_tensors.items():
            outputs[key] = value
    else:
        with torch.no_grad():
            x_final = input_adapter(x_projected, x_raw) if input_adapter is not None else x_projected
            semantic_final = mhlgc_semantic_embeddings.to(device) if mhlgc_semantic_embeddings is not None else None
            routed_bundle_final = _make_routed_multiview_bundle(
                list(range(int(x_final.shape[0]))),
                semantic_rows=semantic_final,
            )
            base_final_outputs = model.forward_outputs(
                x_final,
                edge_index,
                edge_type,
                construct_x=x_construct,
                routed_multiview_bundle=routed_bundle_final,
            )
            outputs = base_final_outputs
            if routed_highpass_enabled:
                final_highpass_bundle, _ = _make_routed_highpass_bundle(base_final_outputs)
                outputs = model.forward_outputs(
                    x_final,
                    edge_index,
                    edge_type,
                    construct_x=x_construct,
                    routed_multiview_bundle=routed_bundle_final,
                    routed_highpass_bundle=final_highpass_bundle,
                )
    return {
        "model": model,
        "input_adapter": input_adapter,
        "outputs": outputs,
        "best_metrics": best_metrics or {},
        "best_state": best_state,
        "optimizer_steps": int(optimizer_steps),
        "neighborloader_contract_metrics": {
            **neighborloader_contract_metrics,
            "checkpoint_selection_primary": best_checkpoint_primary,
            "checkpoint_selection_tie_breaker": best_checkpoint_tie_breaker,
            "valid_best_checkpoint": dict(best_metrics or {}),
        },
        "mhlgc_stats": {
            **mhlgc_stats,
            "loss_mean_active": (
                float(mhlgc_stats["loss_sum"] / float(mhlgc_stats["active_steps"]))
                if int(mhlgc_stats["active_steps"]) > 0
                else 0.0
            ),
        },
        "routed_contrast_stats": {
            **routed_contrast_stats,
            "loss_mean_active": (
                float(routed_contrast_stats["loss_sum"] / float(routed_contrast_stats["active_steps"]))
                if int(routed_contrast_stats["active_steps"]) > 0
                else 0.0
            ),
            "view_loss_mean_active": (
                float(routed_contrast_stats["view_loss_sum"] / float(routed_contrast_stats["active_steps"]))
                if int(routed_contrast_stats["active_steps"]) > 0
                else 0.0
            ),
            "supcon_loss_mean_active": (
                float(routed_contrast_stats["supcon_loss_sum"] / float(routed_contrast_stats["active_steps"]))
                if int(routed_contrast_stats["active_steps"]) > 0
                else 0.0
            ),
            "frozen_loss_mean_active": (
                float(routed_contrast_stats["frozen_loss_sum"] / float(routed_contrast_stats["active_steps"]))
                if int(routed_contrast_stats["active_steps"]) > 0
                else 0.0
            ),
            "reliability_gate": dict(routed_gate_stats),
        },
        "routed_highpass_stats": {
            **routed_highpass_stats,
            "loss_mean_active": (
                float(routed_highpass_stats["loss_sum"] / float(routed_highpass_stats["active_steps"]))
                if int(routed_highpass_stats["active_steps"]) > 0
                else 0.0
            ),
            "ce_loss_mean_active": (
                float(routed_highpass_stats["ce_loss_sum"] / float(routed_highpass_stats["active_steps"]))
                if int(routed_highpass_stats["active_steps"]) > 0
                else 0.0
            ),
            "preserve_loss_mean_active": (
                float(routed_highpass_stats["preserve_loss_sum"] / float(routed_highpass_stats["active_steps"]))
                if int(routed_highpass_stats["active_steps"]) > 0
                else 0.0
            ),
            "gate_loss_mean_active": (
                float(routed_highpass_stats["gate_loss_sum"] / float(routed_highpass_stats["active_steps"]))
                if int(routed_highpass_stats["active_steps"]) > 0
                else 0.0
            ),
            "edge_role_loss_mean_active": (
                float(routed_highpass_stats["edge_role_loss_sum"] / float(routed_highpass_stats["active_steps"]))
                if int(routed_highpass_stats["active_steps"]) > 0
                else 0.0
            ),
            "gate_self_mean_active": (
                float(routed_highpass_stats["gate_self_sum"] / float(routed_highpass_stats["active_steps"]))
                if int(routed_highpass_stats["active_steps"]) > 0
                else 0.0
            ),
            "gate_low_mean_active": (
                float(routed_highpass_stats["gate_low_sum"] / float(routed_highpass_stats["active_steps"]))
                if int(routed_highpass_stats["active_steps"]) > 0
                else 0.0
            ),
            "gate_high_mean_active": (
                float(routed_highpass_stats["gate_high_sum"] / float(routed_highpass_stats["active_steps"]))
                if int(routed_highpass_stats["active_steps"]) > 0
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
    construct_bundle = _load_dualspace_construct_bundle(args, data)
    construct_features = (
        construct_bundle["features"]
        if construct_bundle is not None
        else raw_features
    )
    construct_feature_manifest = (
        construct_bundle["feature_manifest"]
        if construct_bundle is not None
        else {
            "source": "raw_features_fallback",
            "dualspace_role": "shared_with_raw_features",
            "path": feature_manifest.get("path", ""),
            "raw_dim": int(raw_features.shape[1]),
            "projected_dim": int(raw_features.shape[1]),
        }
    )
    refine_request = _graph_refine_request(args)
    hyperscan_backbone_names = {
        "rgcn_hyperscan",
        "rgcn_hyperscan_routed",
        "rgcn_hyperscan_dhg",
        "rgcn_hyperscan_nodeinput",
        "rgcn_hyperscan_dhg_nodeinput",
        "rgcn_h2fag_dualspace_hyperscan",
        "rgcn_h2fag_dualspace_hyperscan_nodeinput",
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
    if int(construct_features.shape[0]) != graph_node_count:
        raise ValueError(f"construct feature rows ({int(construct_features.shape[0])}) must match graph_node_count ({graph_node_count}).")
    if int(labels.numel()) != labeled_node_count:
        raise ValueError(f"Label rows ({int(labels.numel())}) must match labeled_node_count ({labeled_node_count}).")

    device = getattr(args, "device", torch.device("cpu"))
    if not isinstance(device, torch.device):
        device = _resolve_device(device)

    x_projected = features.to(device)
    x_raw = raw_features.to(device)
    x_construct = construct_features.to(device)
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
            features=features,
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
            "candidate_routed_nodes_path": str(refine_request.get("candidate_routed_nodes_path", "") or ""),
            "knn_k": int(refine_request.get("knn_k", 8)),
            "candidate_scope": str(refine_request.get("candidate_scope", "undirected_relation_1hop")),
            "proxy_expansion": "dynamic_hypergraph_branch",
            "feature_source": str(refine_request.get("feature_source", "x_low_plus_x_in_dynamic_forward")),
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
            **dict(dynamic_similarity_branch.get("candidate_policy_stats", {}) or {}),
        }
    elif refine_request["mode"] == "hyperscan_neighborloader_batch_local_branch":
        if str(graph_data_variant).lower() != "labeled":
            raise ValueError(
                "hyperscan_neighborloader_batch_local_branch currently requires --graph_data_variant labeled "
                "to match the original HyperScan labeled-graph regime."
            )
        candidate_policy = str(refine_request.get("candidate_policy", "default") or "default").strip().lower()
        candidate_routed_nodes_path = str(refine_request.get("candidate_routed_nodes_path", "") or "").strip()
        batch_local_routed_node_ids = []
        if candidate_policy in {"exclude_routed", "post_topk_exclude_routed"}:
            batch_local_routed_node_ids = [
                int(item)
                for item in _load_routed_center_node_ids(Path(candidate_routed_nodes_path), split_name="all")
                if 0 <= int(item) < int(graph_node_count)
            ]
        dynamic_similarity_branch = {
            "center_node_ids": [],
            "center_candidate_node_ids": [],
            "center_source": "neighborloader_seed_batch_local_knn",
            "center_count": int(labeled_node_count),
            "centers_with_relation_neighbors": 0,
            "mean_relation_candidates_per_center": 0.0,
            "batch_local_routed_node_ids": batch_local_routed_node_ids,
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
            "routed_nodes_path": str(refine_request.get("routed_nodes_path", "") or ""),
            "routed_nodes_split": str(refine_request.get("routed_nodes_split", "all") or "all"),
            "candidate_policy": candidate_policy,
            "candidate_role_contract": (
                "batch_local_global_routed_mask"
                if candidate_policy in {"exclude_routed", "post_topk_exclude_routed"}
                else "none"
            ),
            "candidate_routed_nodes_path": candidate_routed_nodes_path,
            "candidate_policy_routed_count": int(len(batch_local_routed_node_ids)),
            "knn_k": int(refine_request.get("knn_k", 8)),
            "candidate_scope": "batch_local_subgraph_knn",
            "proxy_expansion": "dynamic_hypergraph_branch",
            "feature_source": str(refine_request.get("feature_source", "x_low_plus_x_in_dynamic_forward")),
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
    config["construct_input_dim"] = int(construct_features.shape[1])
    config["construct_feature_manifest"] = construct_feature_manifest
    config["n_relations"] = int(active_relation_cardinality)
    requested_backbone_name = str(
        getattr(args, "GNN_model", getattr(args, "graph_backbone", "rgcn")) or "rgcn"
    ).lower()
    uses_dualspace_nodeinput = (
        "rgcn_h2fag_dualspace" in requested_backbone_name
        and "nodeinput" in requested_backbone_name
    )
    node_input_manifest = construct_feature_manifest if uses_dualspace_nodeinput else feature_manifest
    if str(node_input_manifest.get("node_input_family", "semantic_embedding")).lower() == "hyperscan_meta_tweet_proxy":
        config["tweet_dim"] = int(node_input_manifest.get("tweet_dim", 0))
        config["num_prop_dim"] = int(node_input_manifest.get("num_prop_dim", 0))
        config["cat_prop_dim"] = int(node_input_manifest.get("cat_prop_dim", 0))
        config["node_input_family"] = "hyperscan_meta_tweet_proxy"
    if dynamic_similarity_branch is not None:
        config["dynamic_similarity_branch"] = {
            "enabled": True,
            "knn_k": int(refine_request.get("knn_k", 8)),
            "candidate_scope": str(refine_request.get("candidate_scope", "undirected_relation_1hop")),
            "similarity_metric": str(refine_request.get("similarity_metric", "cosine")),
            "feature_source": str(refine_request.get("feature_source", "x_low_plus_x_in_dynamic_forward")),
            "center_source": str(dynamic_similarity_branch["center_source"]),
            "routed_nodes_path": str(refine_request.get("routed_nodes_path", "") or ""),
            "routed_nodes_split": str(refine_request.get("routed_nodes_split", "all") or "all"),
            "candidate_routed_nodes_path": str(refine_request.get("candidate_routed_nodes_path", "") or ""),
            "center_node_ids": list(dynamic_similarity_branch["center_node_ids"]),
            "center_candidate_node_ids": list(dynamic_similarity_branch["center_candidate_node_ids"]),
            "center_candidate_role_ids": list(dynamic_similarity_branch.get("center_candidate_role_ids", [])),
            "center_candidate_keep_mask": list(dynamic_similarity_branch.get("center_candidate_keep_mask", [])),
            "batch_local_routed_node_ids": list(dynamic_similarity_branch.get("batch_local_routed_node_ids", [])),
            "center_count": int(dynamic_similarity_branch["center_count"]),
            "centers_with_relation_neighbors": int(dynamic_similarity_branch["centers_with_relation_neighbors"]),
            "mean_relation_candidates_per_center": float(dynamic_similarity_branch["mean_relation_candidates_per_center"]),
            "batch_local_knn": bool(refine_request["mode"] == "hyperscan_neighborloader_batch_local_branch"),
            "second_view_scope": str(refine_request.get("second_view_scope", "")),
            "hypergraph_backend": str(refine_request.get("hypergraph_backend", "pyg")),
            "fusion": str(refine_request.get("fusion", "residual")),
            "training_geometry": str(refine_request.get("training_geometry", getattr(args, "graph_training_loader_mode", "full_batch"))),
            "candidate_policy": str(refine_request.get("candidate_policy", "default") or "default"),
            "candidate_routed_nodes_path": str(refine_request.get("candidate_routed_nodes_path", "") or ""),
            "candidate_policy_quotas": {
                "stable": int(
                    refine_request.get("stable_quota", 0)
                    or max(1, int(refine_request.get("knn_k", 8)) // 2)
                ),
                "hard": int(refine_request.get("mixed_hard_quota", 2) or 2),
                "stable_mixed": int(refine_request.get("mixed_stable_quota", 4) or 4),
                "counterfactual": int(refine_request.get("mixed_counterfactual_quota", 2) or 2),
            },
            "candidate_role_contract": str(
                (dynamic_similarity_branch.get("candidate_policy_stats", {}) or {}).get(
                    "candidate_role_contract",
                    "none",
                )
            ),
        }
    max_epochs = int(getattr(args, "g0_epochs", 0) or getattr(args, "GNN_epochs_per_iter", 1))
    mhlgc_enabled = bool(getattr(args, "mhlgc_enable", False))
    routed_contrast_family = str(getattr(args, "routed_contrast_family", "none") or "none").strip().lower()
    routed_contrast_enabled = routed_contrast_family != "none"
    requested_backbone = str(getattr(args, "GNN_model", getattr(args, "graph_backbone", "rgcn"))).lower()
    dualspace_hyperscan_active = "rgcn_h2fag_dualspace_hyperscan" in requested_backbone
    if dualspace_hyperscan_active and mhlgc_enabled:
        raise ValueError("dualspace hyperscan backbones do not support --mhlgc_enable in v1.")
    if dualspace_hyperscan_active and routed_contrast_enabled:
        raise ValueError("dualspace hyperscan backbones do not support --routed_contrast_family in v1.")
    mhlgc_semantic_embeddings = None
    mhlgc_target_mask = None
    mhlgc_routed_multiview_rows = None
    if mhlgc_enabled:
        semantic_path = str(getattr(args, "mhlgc_semantic_embedding_path", "") or "")
        if not semantic_path:
            raise ValueError("--mhlgc_enable requires --mhlgc_semantic_embedding_path for LLM-guided contrastive learning.")
        mhlgc_semantic_embeddings = _load_mhlgc_semantic_embeddings(
            path=semantic_path,
            expected_rows=graph_node_count,
        )
        mhlgc_target_mask = _resolve_mhlgc_target_mask_from_payload(
            path=semantic_path,
            expected_rows=graph_node_count,
        )
        mhlgc_routed_multiview_rows = _load_mhlgc_routed_multiview_rows(
            path=semantic_path,
            expected_rows=graph_node_count,
        )
    routed_contrast_frozen_embeddings = None
    routed_contrast_train_node_ids = None
    routed_highpass_mode = str(getattr(args, "routed_highpass_mode", "off") or "off").strip().lower()
    routed_highpass_enabled = routed_highpass_mode != "off"
    if dualspace_hyperscan_active and routed_highpass_enabled:
        raise ValueError("dualspace hyperscan backbones do not support --routed_highpass_mode in v1.")
    routed_highpass_node_ids = None
    routed_highpass_train_node_ids = None
    routed_highpass_risk_scores = None
    routed_highpass_candidate_rows = None
    if routed_contrast_enabled:
        routed_contrast_frozen_path = str(
            getattr(args, "routed_contrast_frozen_path", "") or getattr(args, "embedding_path", "") or ""
        )
        if not routed_contrast_frozen_path:
            raise ValueError("--routed_contrast_family requires --routed_contrast_frozen_path or a resolved --embedding_path.")
        routed_contrast_frozen_embeddings = _load_routed_contrast_frozen_embeddings(
            path=routed_contrast_frozen_path,
            expected_rows=graph_node_count,
        )
        routed_nodes_path = str(getattr(args, "routed_nodes_path", "") or "").strip()
        if not routed_nodes_path:
            raise ValueError("--routed_contrast_family requires --routed_nodes_path so routed train anchors are defined.")
        routed_contrast_train_node_ids = _load_routed_center_node_ids(Path(routed_nodes_path), split_name="train")
        if not routed_contrast_train_node_ids:
            raise ValueError("--routed_contrast_family requires non-empty routed train node ids from --routed_nodes_path.")
        if (
            str(getattr(args, "routed_contrast_gate", "heuristic_reliable") or "heuristic_reliable").strip().lower()
            == "heuristic_reliable"
            and mhlgc_routed_multiview_rows is None
        ):
            semantic_path = str(getattr(args, "mhlgc_semantic_embedding_path", "") or "")
            raise ValueError(
                "--routed_contrast_gate heuristic_reliable currently requires routed multiview rows from "
                "--mhlgc_semantic_embedding_path (precompute mhlgc_llm_guide payload). "
                f"Current path: {semantic_path or '<empty>'}."
            )
    if routed_highpass_enabled:
        routed_nodes_path = str(getattr(args, "routed_nodes_path", "") or "").strip()
        if not routed_nodes_path:
            raise ValueError("--routed_highpass_mode requires --routed_nodes_path.")
        routed_highpass_node_ids = _load_routed_center_node_ids(Path(routed_nodes_path), split_name="all")
        routed_highpass_train_node_ids = _load_routed_center_node_ids(Path(routed_nodes_path), split_name="train")
        if not routed_highpass_train_node_ids:
            raise ValueError("--routed_highpass_mode requires non-empty train nodes from --routed_nodes_path.")
        routed_highpass_risk_scores = _load_routed_highpass_risk_scores(
            str(getattr(args, "routed_highpass_risk_path", "") or ""),
            expected_rows=graph_node_count,
        )
        routed_highpass_candidate_rows, routed_highpass_candidate_stats = _build_relation_1hop_candidate_lists(
            edge_index=edge_index.detach().cpu(),
            num_nodes=graph_node_count,
            max_neighbors=int(getattr(args, "routed_highpass_max_neighbors", 32) or 32),
            candidate_scope=str(getattr(args, "routed_highpass_candidate_scope", "relation_1hop") or "relation_1hop"),
            labeled_node_count=labeled_node_count,
        )
    else:
        routed_highpass_candidate_stats = {}
    training_result = _train_graph_backbone_once(
        config=config,
        x_projected=x_projected,
        x_raw=x_raw,
        x_construct=x_construct,
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
        graph_neighborloader_contract=str(
            getattr(args, "graph_neighborloader_contract", "seed_only") or "seed_only"
        ),
        graph_batch_size=int(getattr(args, "batch_size_GNN", 1024)),
        neighbor_num_neighbors=int(refine_request.get("neighbor_num_neighbors", getattr(args, "graph_neighbor_num_neighbors", 64))),
        max_update_steps=int(getattr(args, "graph_training_max_steps", 0) or 0),
        mhlgc_enabled=mhlgc_enabled,
        mhlgc_semantic_embeddings=mhlgc_semantic_embeddings,
        mhlgc_loss_weight=_float_cli_arg(getattr(args, "mhlgc_loss_weight", None), 0.0),
        mhlgc_beta=_float_cli_arg(getattr(args, "mhlgc_beta", None), 1.0),
        mhlgc_gamma=_float_cli_arg(getattr(args, "mhlgc_gamma", None), 0.5),
        mhlgc_temperature=_float_cli_arg(getattr(args, "mhlgc_temperature", None), 1.0),
        mhlgc_feature_mask_probability=_float_cli_arg(
            getattr(args, "mhlgc_feature_mask_probability", None), 0.15
        ),
        mhlgc_edge_mask_probability=_float_cli_arg(
            getattr(args, "mhlgc_edge_mask_probability", None), 0.10
        ),
        mhlgc_hyperedge_mask_probability=_float_cli_arg(
            getattr(args, "mhlgc_hyperedge_mask_probability", None), 0.0
        ),
        mhlgc_anchors_per_batch=int(getattr(args, "mhlgc_anchors_per_batch", 1) or 1),
        mhlgc_negative_count=int(getattr(args, "mhlgc_negative_count", 0) or 0),
        mhlgc_positive_label=int(getattr(args, "mhlgc_positive_label", 1) or 1),
        mhlgc_target_mask=mhlgc_target_mask,
        mhlgc_routed_multiview_rows=mhlgc_routed_multiview_rows,
        mhlgc_contrast_space=str(getattr(args, "mhlgc_contrast_space", "fused_x") or "fused_x"),
        mhlgc_anchor_source=str(
            getattr(args, "mhlgc_anchor_source", "semantic_nonzero_positive") or "semantic_nonzero_positive"
        ),
        mhlgc_pair_mode=str(getattr(args, "mhlgc_pair_mode", "augmentation") or "augmentation"),
        mhlgc_semantic_projector=str(getattr(args, "mhlgc_semantic_projector", "auto") or "auto"),
        routed_contrast_enabled=routed_contrast_enabled,
        routed_contrast_family=routed_contrast_family,
        routed_contrast_gate=str(getattr(args, "routed_contrast_gate", "heuristic_reliable") or "heuristic_reliable"),
        routed_contrast_weight=_float_cli_arg(getattr(args, "routed_contrast_weight", None), 0.0),
        routed_contrast_temperature=_float_cli_arg(getattr(args, "routed_contrast_temperature", None), 0.2),
        routed_contrast_frozen_embeddings=routed_contrast_frozen_embeddings,
        routed_contrast_train_node_ids=routed_contrast_train_node_ids,
        routed_contrast_multiview_rows=mhlgc_routed_multiview_rows,
        routed_highpass_mode=routed_highpass_mode,
        routed_highpass_target=str(getattr(args, "routed_highpass_target", "logits") or "logits"),
        routed_highpass_candidate_scope=str(getattr(args, "routed_highpass_candidate_scope", "relation_1hop") or "relation_1hop"),
        routed_highpass_max_neighbors=int(getattr(args, "routed_highpass_max_neighbors", 32) or 32),
        routed_highpass_loss_weight=_float_cli_arg(getattr(args, "routed_highpass_loss_weight", None), 1.0),
        routed_highpass_preserve_weight=_float_cli_arg(getattr(args, "routed_highpass_preserve_weight", None), 0.2),
        routed_highpass_risk_gate_weight=_float_cli_arg(getattr(args, "routed_highpass_risk_gate_weight", None), 0.5),
        routed_highpass_edge_role_weight=_float_cli_arg(getattr(args, "routed_highpass_edge_role_weight", None), 0.1),
        routed_highpass_train_node_ids=routed_highpass_train_node_ids,
        routed_highpass_node_ids=routed_highpass_node_ids,
        routed_highpass_risk_scores=routed_highpass_risk_scores,
        routed_highpass_candidate_rows=routed_highpass_candidate_rows,
    )
    model = training_result["model"]
    input_adapter = training_result["input_adapter"]
    outputs = training_result["outputs"]
    raw_outputs = outputs
    best_metrics = training_result["best_metrics"]
    best_state = training_result["best_state"]
    neighborloader_contract_metrics = dict(training_result.get("neighborloader_contract_metrics", {}) or {})
    if input_adapter is not None:
        feature_manifest["peft"]["trainable_parameter_count"] = count_trainable_parameters(input_adapter)
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
    if torch.is_tensor(raw_outputs.get("x_high")):
        outputs["x_high"] = raw_outputs["x_high"].detach().cpu()
    if torch.is_tensor(raw_outputs.get("x_high_base")):
        outputs["x_high_base"] = raw_outputs["x_high_base"].detach().cpu()
    if torch.is_tensor(raw_outputs.get("x_new")):
        outputs["x_new"] = raw_outputs["x_new"].detach().cpu()
    extra_tensor_keys = [
        "x_ego_dec",
        "x_rel_low1_dec",
        "x_rel_high1_dec",
        "x_rel_state1_dec",
        "x_rel_low2_dec",
        "x_rel_high2_dec",
        "x_rel_state2_dec",
        "x_clean_base",
        "x_new_clean",
        "x_high_low1_clean",
        "x_high_high1_clean",
        "x_high_state1_clean",
        "x_high_low2_clean",
        "x_high_high2_clean",
        "x_high_clean",
    ]
    for key in extra_tensor_keys:
        if torch.is_tensor(raw_outputs.get(key)):
            outputs[key] = raw_outputs[key].detach().cpu()
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
        "neighborloader_contract_metrics": neighborloader_contract_metrics,
        "mhlgc": training_result.get("mhlgc_stats", {"enabled": False}),
        "routed_highpass": training_result.get("routed_highpass_stats", {"enabled": False}),
    }
    write_torch(out_dir / "checkpoint.pt", checkpoint)
    write_torch(out_dir / "outputs.pt", outputs)
    write_json(out_dir / "selection_metrics.json", best_metrics or {})
    if neighborloader_contract_metrics.get("contract") == "hyperscan_sampled_subgraph":
        write_json(out_dir / "neighborloader_contract_metrics.json", neighborloader_contract_metrics)
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
                "positioning": (
                    "strong_graph_consumer"
                    if str(getattr(args, "graph_second_view_fusion", "residual") or "residual").strip().lower()
                    in {"multiattn", "multiattn_adaptive"}
                    else "lightweight_residual_consumer"
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
                "positioning": str(refine_request.get("positioning", "none") or "none"),
                "control_only": bool(refine_request.get("control_only", False)),
            },
            "training_scope": (
                "train_supervised_plus_mhlgc"
                if mhlgc_enabled
                else (
                    "train_supervised_with_routed_graph_correction"
                    if routed_highpass_enabled
                    else "train_only_supervised"
                )
            ),
            "routed_contrast_scope": "routed_train_only" if routed_contrast_enabled else "disabled",
            "routed_highpass_scope": (
                "stage0_base_ce_then_frozen_routed_correction"
                if routed_highpass_enabled
                else "disabled"
            ),
            "representation_roles": {
            "z_sem": "semantic_input_space",
            "z_construct": (
                "hyperscan_clean_representation"
                if construct_bundle is not None
                else "x_new_high_order_construction_space"
            ),
            "z_decision": "iter_minus1",
            "z_pred": "fused_x_final_detector_space",
        },
            "pseudo_label_policy": "disabled_by_default",
            "checkpoint_selection": {
                "primary": str(
                    neighborloader_contract_metrics.get("checkpoint_selection_primary", "validation_macro_f1")
                    or "validation_macro_f1"
                ),
                "tie_breaker": str(
                    neighborloader_contract_metrics.get("checkpoint_selection_tie_breaker", "validation_loss")
                    or "validation_loss"
                ),
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
                "neighborloader_contract": str(
                    neighborloader_contract_metrics.get("contract", getattr(args, "graph_neighborloader_contract", "seed_only"))
                    or "seed_only"
                ),
                "supervision_scope": str(
                    neighborloader_contract_metrics.get("supervision_scope", "seed_only_first_batch_rows")
                    or "seed_only_first_batch_rows"
                ),
                "validation_metric_scope": str(
                    neighborloader_contract_metrics.get("validation_scope", "canonical_valid_seed_nodes")
                    or "canonical_valid_seed_nodes"
                ),
                "test_metric_scope": str(
                    neighborloader_contract_metrics.get("test_scope", "full_graph_deduplicated_export_only")
                    or "full_graph_deduplicated_export_only"
                ),
                "duplicate_counting": str(
                    neighborloader_contract_metrics.get("duplicate_counting", "none") or "none"
                ),
                "optimizer_steps": int(training_result.get("optimizer_steps", 0)),
                "max_update_steps": int(getattr(args, "graph_training_max_steps", 0) or 0),
            },
            "mhlgc": {
                **training_result.get("mhlgc_stats", {"enabled": False}),
                "semantic_embedding_path": str(getattr(args, "mhlgc_semantic_embedding_path", "") or ""),
                "beta": _float_cli_arg(getattr(args, "mhlgc_beta", None), 1.0),
                "gamma": _float_cli_arg(getattr(args, "mhlgc_gamma", None), 0.5),
                "temperature": _float_cli_arg(getattr(args, "mhlgc_temperature", None), 1.0),
                "feature_mask_probability": _float_cli_arg(
                    getattr(args, "mhlgc_feature_mask_probability", None), 0.15
                ),
                "edge_mask_probability": _float_cli_arg(
                    getattr(args, "mhlgc_edge_mask_probability", None), 0.10
                ),
                "hyperedge_mask_probability": _float_cli_arg(
                    getattr(args, "mhlgc_hyperedge_mask_probability", None), 0.0
                ),
                "anchors_per_batch": int(getattr(args, "mhlgc_anchors_per_batch", 1) or 1),
                "negative_count_per_anchor": int(getattr(args, "mhlgc_negative_count", 0) or 0),
                "positive_label": int(getattr(args, "mhlgc_positive_label", 1) or 1),
                "contrast_space": str(getattr(args, "mhlgc_contrast_space", "fused_x") or "fused_x"),
                "anchor_source": str(
                    getattr(args, "mhlgc_anchor_source", "semantic_nonzero_positive") or "semantic_nonzero_positive"
                ),
                "pair_mode": str(getattr(args, "mhlgc_pair_mode", "augmentation") or "augmentation"),
                "semantic_projector": str(getattr(args, "mhlgc_semantic_projector", "auto") or "auto"),
                "paper_alignment": (
                    "LLM-as-guide social-bot adaptation over selected graph representation spaces; "
                    "default path keeps masked-view hard-negative weighting, while repair-aware mode "
                    "treats the guide as a routed-node correction signal rather than a predictor."
                ),
            },
            "routed_contrast": {
                **training_result.get("routed_contrast_stats", {"enabled": False}),
                "family": routed_contrast_family,
                "gate": str(getattr(args, "routed_contrast_gate", "heuristic_reliable") or "heuristic_reliable"),
                "weight": _float_cli_arg(getattr(args, "routed_contrast_weight", None), 0.0),
                "temperature": _float_cli_arg(getattr(args, "routed_contrast_temperature", None), 0.2),
                "frozen_path": str(getattr(args, "routed_contrast_frozen_path", "") or getattr(args, "embedding_path", "") or ""),
                "paper_alignment": (
                    "Routed contrast is a BotSCL-style / low-high selective ablation over routed train nodes. "
                    "It is not MH-LGC faithful reproduction and not a default mainline detector objective."
                ),
            },
            "routed_highpass": {
                **training_result.get("routed_highpass_stats", {"enabled": False}),
                "mode": routed_highpass_mode,
                "target": str(getattr(args, "routed_highpass_target", "logits") or "logits"),
                "candidate_scope": str(getattr(args, "routed_highpass_candidate_scope", "relation_1hop") or "relation_1hop"),
                "candidate_stats": routed_highpass_candidate_stats,
                "max_neighbors": int(getattr(args, "routed_highpass_max_neighbors", 32) or 32),
                "loss_weight": _float_cli_arg(getattr(args, "routed_highpass_loss_weight", None), 1.0),
                "preserve_weight": _float_cli_arg(getattr(args, "routed_highpass_preserve_weight", None), 0.2),
                "risk_gate_weight": _float_cli_arg(getattr(args, "routed_highpass_risk_gate_weight", None), 0.5),
                "edge_role_weight": _float_cli_arg(getattr(args, "routed_highpass_edge_role_weight", None), 0.1),
                "risk_path": str(getattr(args, "routed_highpass_risk_path", "") or ""),
                "routed_nodes_path": str(getattr(args, "routed_nodes_path", "") or ""),
                "paper_alignment": (
                    "Conformal-risk supervised routed-only high-pass correction. "
                    "target=logits applies legacy detector-space delta-logit correction; "
                    "target=x_high corrects the HGNN high-order branch before cross-attention. "
                    "Risk supervises correction utility/gating, not bot probability; non-routed nodes are preserved."
                ),
            },
            "feature_manifest": feature_manifest,
            "construct_feature_manifest": construct_feature_manifest,
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
            "output_tensors": [
                name
                for name in [
                    "logits",
                    "prob",
                    "pred",
                    "labels",
                    "fused_x",
                    "node_repr",
                    "x_low",
                    "x_high",
                    "x_high_base",
                    "x_new",
                    "aux_features",
                ]
                if name in outputs
            ],
            "graph_override": graph_override_manifest if graph_override_manifest is not None else {"mode": "none"},
            **(
                {
                    "graph_refine": {
                        **graph_refine_stats,
                        "embedding_source": feature_manifest.get("path"),
                        "positioning": str(refine_request.get("positioning", "none") or "none"),
                        "control_only": bool(refine_request.get("control_only", False)),
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
    neighborloader_contract_metrics_path = out_dir / "neighborloader_contract_metrics.json"
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
        "neighborloader_contract_metrics": read_json(neighborloader_contract_metrics_path, default={}),
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


def _relation_overlap_center_candidates(edge_index, edge_type, num_nodes, refine_request, labeled_node_count, features=None):
    center_node_ids, center_source = _resolve_refine_center_node_ids(
        refine_request=refine_request,
        num_nodes=int(num_nodes),
        labeled_node_count=int(labeled_node_count),
    )
    candidate_scope = str(refine_request.get("candidate_scope", "undirected_relation_1hop")).strip().lower()
    if candidate_scope in {"labeled_full", "hyperscan_full"}:
        candidate_limit = int(labeled_node_count) if candidate_scope == "labeled_full" else int(num_nodes)
        candidate_limit = max(0, min(int(num_nodes), int(candidate_limit)))
        candidate_pool = np.arange(candidate_limit, dtype=np.int64)
        candidate_rows = []
        total_candidate_count = 0
        centers_with_candidates = 0
        for center in center_node_ids.tolist():
            row = candidate_pool[candidate_pool != int(center)]
            row_list = [int(item) for item in row.tolist()]
            if row_list:
                total_candidate_count += int(len(row_list))
                centers_with_candidates += 1
            candidate_rows.append(row_list)
        bundle = {
            "center_node_ids": [int(item) for item in center_node_ids.tolist()],
            "center_candidate_node_ids": candidate_rows,
            "center_source": center_source,
            "center_count": int(len(center_node_ids)),
            "centers_with_relation_neighbors": int(centers_with_candidates),
            "mean_relation_candidates_per_center": (
                float(total_candidate_count / float(centers_with_candidates)) if centers_with_candidates else 0.0
            ),
            "candidate_pool_scope": candidate_scope,
        }
    else:
        bundle = _shared_build_relation_overlap_center_candidates(
            edge_index=edge_index,
            num_nodes=int(num_nodes),
            center_node_ids=center_node_ids.tolist(),
            center_source=center_source,
            candidate_scope=candidate_scope,
            labeled_node_count=int(labeled_node_count),
        )
        bundle["candidate_pool_scope"] = candidate_scope
    return _build_second_view_candidate_role_bundle(
        candidate_bundle=bundle,
        refine_request=refine_request,
        num_nodes=int(num_nodes),
        labeled_node_count=int(labeled_node_count),
        features=features,
    )


def _graph_refine_request(args):
    mode = str(getattr(args, "graph_refine_mode", "none") or "none").lower()
    budget = float(getattr(args, "graph_refine_budget", 0.0) or 0.0)
    candidate_scope = str(getattr(args, "graph_refine_candidate_scope", "undirected_relation_1hop") or "undirected_relation_1hop").strip().lower()
    hypergraph_backend = str(getattr(args, "graph_second_view_hypergraph_backend", "pyg") or "pyg").strip().lower()
    fusion = str(getattr(args, "graph_second_view_fusion", "residual") or "residual").strip().lower()
    training_geometry = str(getattr(args, "graph_training_loader_mode", "full_batch") or "full_batch").strip().lower()
    graph_refine_positioning = str(getattr(args, "graph_refine_positioning", "none") or "none").strip().lower()
    graph_refine_control_only = bool(getattr(args, "graph_refine_control_only", False))
    if candidate_scope not in {"undirected_relation_1hop", "labeled_relation_1hop", "labeled_full", "hyperscan_full"}:
        raise ValueError(
            "--graph_refine_candidate_scope must be one of "
            "{undirected_relation_1hop, labeled_relation_1hop, labeled_full, hyperscan_full}."
        )
    if hypergraph_backend not in {"pyg", "dhg"}:
        raise ValueError("--graph_second_view_hypergraph_backend must be one of {pyg, dhg}.")
    if fusion not in {"residual", "multiattn", "multiattn_adaptive"}:
        raise ValueError("--graph_second_view_fusion must be one of {residual, multiattn, multiattn_adaptive}.")
    if training_geometry not in {"full_batch", "neighbor_subgraph"}:
        raise ValueError("--graph_training_loader_mode must be one of {full_batch, neighbor_subgraph}.")
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
    if mode == "hyperscan_knn_hypergraph_proxy_augment":
        knn_k = int(getattr(args, "graph_refine_knn_k", 8) or 8)
        if knn_k < 2:
            raise ValueError("--graph_refine_knn_k must be >= 2 for hyperscan_knn_hypergraph_proxy_augment.")
        return {
            "mode": mode,
            "budget": 0.0,
            "degree_guard_enabled": False,
            "oracle_uses_labels": False,
            "diagnostic_only": True,
            "knn_k": int(knn_k),
            "proxy_expansion": "bidirectional_center_star",
            "positioning": graph_refine_positioning or "diagnostic_control",
            "control_only": True,
            "mainline_recommendation": "Use HyperScan-style second-view modes for mainline graph consumption.",
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
            "diagnostic_only": True,
            "knn_k": int(knn_k),
            "proxy_expansion": "bidirectional_center_star",
            "candidate_scope": candidate_scope,
            "similarity_metric": "cosine",
            "routed_nodes_path": str(getattr(args, "routed_nodes_path", "") or ""),
            "routed_nodes_split": str(getattr(args, "routed_nodes_split", "all") or "all"),
            "positioning": graph_refine_positioning or "diagnostic_control",
            "control_only": True,
            "mainline_recommendation": "Use HyperScan-style second-view modes for mainline graph consumption.",
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
            "diagnostic_only": True,
            "knn_k": int(knn_k),
            "proxy_expansion": "bidirectional_center_star",
            "candidate_scope": candidate_scope,
            "similarity_metric": "cosine",
            "feature_source": "relation_prefit_node_repr_plus_g0_input",
            "prefit_max_epochs": int(getattr(args, "g0_epochs", 0) or getattr(args, "GNN_epochs_per_iter", 1) or 1),
            "routed_nodes_path": str(getattr(args, "routed_nodes_path", "") or ""),
            "routed_nodes_split": str(getattr(args, "routed_nodes_split", "all") or "all"),
            "positioning": graph_refine_positioning or "diagnostic_control",
            "control_only": True,
            "mainline_recommendation": "Use HyperScan-style second-view modes for mainline graph consumption.",
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
            "positioning": graph_refine_positioning or "second_view_mainline",
            "control_only": graph_refine_control_only,
            "candidate_scope": candidate_scope,
            "similarity_metric": "cosine",
            "feature_source": "x_low_plus_x_in_dynamic_forward",
            "routed_nodes_path": routed_nodes_path,
            "routed_nodes_split": str(getattr(args, "routed_nodes_split", "all") or "all"),
            "second_view_scope": second_view_scope,
            "hypergraph_backend": hypergraph_backend,
            "fusion": fusion,
            "training_geometry": training_geometry,
            "candidate_policy": str(getattr(args, "graph_second_view_candidate_policy", "default") or "default").strip().lower(),
            "candidate_routed_nodes_path": str(
                getattr(args, "graph_second_view_candidate_routed_nodes_path", "") or ""
            ),
            "candidate_risk_path": str(getattr(args, "graph_second_view_candidate_risk_path", "") or ""),
            "candidate_pred_path": str(getattr(args, "graph_second_view_candidate_pred_path", "") or ""),
            "llm_edge_retain_path": str(getattr(args, "graph_second_view_llm_edge_retain_path", "") or ""),
            "router_support_path": str(getattr(args, "graph_second_view_router_support_path", "") or ""),
            "stable_quantile": float(getattr(args, "graph_second_view_stable_quantile", 0.50) or 0.50),
            "stable_quota": int(getattr(args, "graph_second_view_stable_quota", 0) or 0),
            "mixed_hard_quota": int(getattr(args, "graph_second_view_mixed_hard_quota", 2) or 2),
            "mixed_stable_quota": int(getattr(args, "graph_second_view_mixed_stable_quota", 4) or 4),
            "mixed_counterfactual_quota": int(
                getattr(args, "graph_second_view_mixed_counterfactual_quota", 2) or 2
            ),
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
        candidate_policy = str(
            getattr(args, "graph_second_view_candidate_policy", "default") or "default"
        ).strip().lower()
        if candidate_policy in {"stable_quota", "mixed_quota", "llm_retain", "router_support_transfer"}:
            raise ValueError(
                "neighborloader_batch currently supports only default, exclude_routed, "
                "or post_topk_exclude_routed candidate policies. Use routed_dynamic_hyperscan_branch "
                "with full-batch geometry for llm_retain or router_support_transfer."
            )
        candidate_routed_nodes_path = str(
            getattr(args, "graph_second_view_candidate_routed_nodes_path", "") or ""
        ).strip()
        if candidate_policy in {"exclude_routed", "post_topk_exclude_routed"} and not candidate_routed_nodes_path:
            raise ValueError(
                "--graph_second_view_candidate_policy exclude_routed/post_topk_exclude_routed "
                "requires --graph_second_view_candidate_routed_nodes_path for neighborloader_batch."
            )
        return {
            "mode": mode,
            "budget": 0.0,
            "degree_guard_enabled": False,
            "oracle_uses_labels": False,
            "diagnostic_only": False,
            "knn_k": int(knn_k),
            "proxy_expansion": "dynamic_hypergraph_branch",
            "positioning": graph_refine_positioning or "second_view_mainline",
            "control_only": graph_refine_control_only,
            "candidate_scope": "batch_local_subgraph_knn",
            "similarity_metric": "cosine",
            "feature_source": "x_low_plus_x_in_dynamic_forward",
            "training_loader_mode": "neighbor_subgraph",
            "neighbor_num_neighbors": int(neighbor_num_neighbors),
            "second_view_scope": "neighborloader_batch",
            "hypergraph_backend": hypergraph_backend,
            "fusion": fusion,
            "training_geometry": "neighbor_subgraph",
            "candidate_policy": candidate_policy,
            "candidate_routed_nodes_path": candidate_routed_nodes_path,
            "routed_nodes_path": str(getattr(args, "routed_nodes_path", "") or ""),
            "routed_nodes_split": str(getattr(args, "routed_nodes_split", "all") or "all"),
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
        x_construct=raw_feature_cpu.to(device),
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
        "loss_weight": _float_cli_arg(getattr(args, "mhlgc_loss_weight", None), 0.0),
        "beta": _float_cli_arg(getattr(args, "mhlgc_beta", None), 1.0),
        "gamma": _float_cli_arg(getattr(args, "mhlgc_gamma", None), 0.5),
        "temperature": _float_cli_arg(getattr(args, "mhlgc_temperature", None), 1.0),
        "feature_mask_probability": _float_cli_arg(
            getattr(args, "mhlgc_feature_mask_probability", None), 0.15
        ),
        "edge_mask_probability": _float_cli_arg(
            getattr(args, "mhlgc_edge_mask_probability", None), 0.10
        ),
        "hyperedge_mask_probability": _float_cli_arg(
            getattr(args, "mhlgc_hyperedge_mask_probability", None), 0.0
        ),
        "anchors_per_batch": int(getattr(args, "mhlgc_anchors_per_batch", 1) or 1),
        "negative_count_per_anchor": int(getattr(args, "mhlgc_negative_count", 0) or 0),
        "positive_label": int(getattr(args, "mhlgc_positive_label", 1) or 1),
        "contrast_space": str(getattr(args, "mhlgc_contrast_space", "fused_x") or "fused_x"),
        "anchor_source": str(
            getattr(args, "mhlgc_anchor_source", "semantic_nonzero_positive") or "semantic_nonzero_positive"
        ),
        "pair_mode": str(getattr(args, "mhlgc_pair_mode", "augmentation") or "augmentation"),
        "semantic_projector": str(getattr(args, "mhlgc_semantic_projector", "auto") or "auto"),
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
    for key in (
        "loss_weight",
        "beta",
        "gamma",
        "temperature",
        "feature_mask_probability",
        "edge_mask_probability",
        "hyperedge_mask_probability",
    ):
        if not math.isclose(
            float(existing.get(key, -1.0)),
            float(requested[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False
    for key in (
        "anchors_per_batch",
        "negative_count_per_anchor",
        "positive_label",
    ):
        if int(existing.get(key, -1)) != int(requested[key]):
            return False
    for key in ("contrast_space", "anchor_source", "pair_mode", "semantic_projector"):
        if str(existing.get(key, "") or "").lower() != str(requested[key]).lower():
            return False
    return True


def _routed_contrast_manifest_request(args):
    family = str(getattr(args, "routed_contrast_family", "none") or "none").strip().lower()
    enabled = family != "none"
    return {
        "enabled": enabled,
        "family": family,
        "gate": str(getattr(args, "routed_contrast_gate", "heuristic_reliable") or "heuristic_reliable"),
        "weight": _float_cli_arg(getattr(args, "routed_contrast_weight", None), 0.0),
        "temperature": _float_cli_arg(getattr(args, "routed_contrast_temperature", None), 0.2),
        "frozen_path": str(getattr(args, "routed_contrast_frozen_path", "") or getattr(args, "embedding_path", "") or "") if enabled else "",
        "routed_nodes_path": str(getattr(args, "routed_nodes_path", "") or "") if enabled else "",
    }


def _routed_contrast_manifest_matches_request(args, manifest):
    requested = _routed_contrast_manifest_request(args)
    existing = manifest.get("routed_contrast", {})
    if not isinstance(existing, dict):
        existing = {}
    if bool(existing.get("enabled", False)) != bool(requested["enabled"]):
        return False
    if not requested["enabled"]:
        return True
    for key in ("family", "gate", "frozen_path", "routed_nodes_path"):
        if str(existing.get(key, "") or "").lower() != str(requested[key]).lower():
            return False
    for key in ("weight", "temperature"):
        if not math.isclose(
            float(existing.get(key, -1.0)),
            float(requested[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
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


def _neighborloader_contract_manifest_matches_request(args, manifest):
    requested_loader_mode = str(getattr(args, "graph_training_loader_mode", "full_batch") or "full_batch").strip().lower()
    if requested_loader_mode != "neighbor_subgraph":
        return True
    requested_contract = str(getattr(args, "graph_neighborloader_contract", "seed_only") or "seed_only").strip().lower()
    training_loader = manifest.get("training_loader", {})
    if not isinstance(training_loader, dict):
        training_loader = {}
    existing_contract = str(training_loader.get("neighborloader_contract", "seed_only") or "seed_only").strip().lower()
    return existing_contract == requested_contract


def _existing_g0_matches_request(args, manifest):
    if manifest.get("contract") != FROZEN_G0_CONTRACT:
        return False
    if manifest.get("backbone", "").lower() != str(getattr(args, "GNN_model", "rgcn")).lower():
        return False
    if not _mhlgc_manifest_matches_request(args, manifest):
        return False
    if not _routed_contrast_manifest_matches_request(args, manifest):
        return False
    if not _neighborloader_contract_manifest_matches_request(args, manifest):
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
    requested_construct_path = str(getattr(args, "graph_construct_embedding_path", "") or "").strip()
    existing_construct_manifest = manifest.get("construct_feature_manifest", {})
    existing_construct_path = str(existing_construct_manifest.get("path", "") or "").strip()
    if requested_construct_path:
        if not existing_construct_path:
            return False
        if str(Path(existing_construct_path)) != str(Path(requested_construct_path)):
            return False
    elif existing_construct_path and str(existing_construct_manifest.get("dualspace_role", "") or "").strip().lower() == "hyperscan_clean_representation":
        requested_backbone = str(getattr(args, "GNN_model", getattr(args, "graph_backbone", "rgcn")) or "rgcn").lower()
        if "rgcn_h2fag_dualspace_hyperscan" in requested_backbone:
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
