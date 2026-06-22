import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from artifact_contracts import MissingFrozenArtifactError
from hypergnn import (
    feature_k_nearest_neighbor_query as _shared_feature_k_nearest_neighbor_query,
    hyperscan_knn_hypergraph_proxy_augment as _shared_hyperscan_knn_hypergraph_proxy_augment,
    relation_overlap_knn_proxy_augment as _shared_relation_overlap_knn_proxy_augment,
)
from utils import read_json, safe_torch_load


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
