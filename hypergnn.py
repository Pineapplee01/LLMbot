"""Shared HyperScan-style graph helpers for routed-node local KNN grouping.

This module centralizes the graph-side logic that was previously scattered
across prompt-cache precompute and graph-detector preparation code:

- selection-feature resolution for relation-aware KNN grouping
- routed-node local relation-overlap candidate construction
- HyperScan-style proxy graph augmentation helpers
- dynamic routed hypergraph incidence construction
- prompt-side support/contrast directional KNN selection
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _load_feature_tensor(path: Path):
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("embeddings", "features", "x"):
            if key in payload:
                payload = payload[key]
                break
    if not torch.is_tensor(payload):
        payload = torch.as_tensor(payload)
    payload = payload.detach().cpu().float()
    if payload.dim() != 2:
        raise ValueError(f"Expected a 2-D embedding tensor at {path}, got shape {tuple(payload.shape)}.")
    return payload.contiguous()


def _resolve_default_selection_embedding_path(dataset_path: Path, seed: int):
    candidates = [
        dataset_path / f"embeddings_iter_-1_seed_{int(seed)}.pt",
        dataset_path / "embeddings_roberta.pt",
        dataset_path / "finetuned_roberta_embeddings_iter_2_seed1.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_selection_feature_bundle(args, dataset_path: Path, graph_variant: str, graph_node_count: int, labeled_node_count: int):
    explicit_labeled = getattr(args, "selection_embedding_path", None)
    labeled_path = Path(explicit_labeled) if explicit_labeled else _resolve_default_selection_embedding_path(dataset_path, int(args.seed))
    if labeled_path is None or not labeled_path.exists():
        raise FileNotFoundError(
            "center_induced_relation_aware requires --selection_embedding_path or a default labeled embedding tensor "
            f"(for example embeddings_iter_-1_seed_{int(args.seed)}.pt) under {dataset_path}."
        )
    labeled_features = _load_feature_tensor(labeled_path)

    if graph_variant == "full_graph_support":
        if int(labeled_features.shape[0]) == int(graph_node_count):
            return {
                "features": F.normalize(labeled_features, p=2, dim=1, eps=1e-12),
                "mode": "single_full_graph_tensor",
                "labeled_path": str(labeled_path),
                "support_path": "",
            }
        if int(labeled_features.shape[0]) != int(labeled_node_count):
            raise ValueError(
                "full_graph_support center-induced selection expects labeled selection embeddings to have either "
                f"{labeled_node_count} rows or {graph_node_count} rows, got {int(labeled_features.shape[0])}."
            )
        explicit_support = getattr(args, "support_selection_embedding_path", None)
        support_path = Path(explicit_support) if explicit_support else dataset_path / "support_roberta_embeddings_new.pt"
        if not support_path.exists():
            raise FileNotFoundError(
                "full_graph_support center-induced selection requires --support_selection_embedding_path or the "
                f"default support embedding tensor at {support_path}."
            )
        support_features = _load_feature_tensor(support_path)
        expected_support = int(graph_node_count) - int(labeled_node_count)
        if int(support_features.shape[0]) != expected_support:
            raise ValueError(
                "Support selection embeddings do not match the full-graph support suffix size: "
                f"expected {expected_support}, got {int(support_features.shape[0])}."
            )
        if int(support_features.shape[1]) != int(labeled_features.shape[1]):
            raise ValueError(
                "Labeled and support selection embeddings must share the same feature dimension for "
                "center_induced_relation_aware."
            )
        full_features = torch.cat([labeled_features, support_features], dim=0)
        return {
            "features": F.normalize(full_features, p=2, dim=1, eps=1e-12),
            "mode": "runtime_labeled_plus_support_concat",
            "labeled_path": str(labeled_path),
            "support_path": str(support_path),
        }

    if int(labeled_features.shape[0]) != int(graph_node_count):
        raise ValueError(
            f"labeled graph variant expects selection embeddings with {graph_node_count} rows, "
            f"got {int(labeled_features.shape[0])}."
        )
    return {
        "features": F.normalize(labeled_features, p=2, dim=1, eps=1e-12),
        "mode": "single_labeled_tensor",
        "labeled_path": str(labeled_path),
        "support_path": "",
    }


def _relation_cardinality_from_edge_type(edge_type, minimum=1):
    if edge_type is None:
        return int(max(int(minimum), 1))
    edge_type_t = edge_type.detach().cpu().long().view(-1) if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long).view(-1)
    if edge_type_t.numel() == 0:
        return int(max(int(minimum), 1))
    return int(max(int(minimum), int(edge_type_t.max().item()) + 1))


def _filter_relation_overlap_candidates(candidate_ids, candidate_scope, labeled_node_count, num_nodes):
    scope = str(candidate_scope or "undirected_relation_1hop").strip().lower()
    if scope == "undirected_relation_1hop":
        return [int(item) for item in candidate_ids]
    if scope == "labeled_relation_1hop":
        labeled_limit = int(labeled_node_count) if labeled_node_count is not None else int(num_nodes)
        labeled_limit = max(0, min(int(num_nodes), int(labeled_limit)))
        return [int(item) for item in candidate_ids if int(item) < labeled_limit]
    raise ValueError(f"Unsupported relation-overlap candidate_scope: {candidate_scope}")


def build_relation_overlap_center_candidates(
    edge_index,
    num_nodes,
    center_node_ids,
    center_source="explicit_centers",
    candidate_scope="undirected_relation_1hop",
    labeled_node_count=None,
):
    edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
    if edge_index_cpu.dim() != 2 or edge_index_cpu.size(0) != 2:
        raise ValueError("edge_index must be shaped [2, num_edges] for routed dynamic HyperScan preparation.")

    adjacency = {}
    src = edge_index_cpu[0].tolist()
    dst = edge_index_cpu[1].tolist()
    for u_raw, v_raw in zip(src, dst):
        u = int(u_raw)
        v = int(v_raw)
        if not (0 <= u < int(num_nodes) and 0 <= v < int(num_nodes)):
            continue
        if u == v:
            continue
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)

    candidate_lists = []
    total_candidate_count = 0
    centers_with_relation_neighbors = 0
    normalized_centers = [int(item) for item in center_node_ids]
    for center in normalized_centers:
        candidate_ids = adjacency.get(int(center))
        if not candidate_ids:
            candidate_lists.append([])
            continue
        candidate_arr = sorted(
            _filter_relation_overlap_candidates(
                (int(item) for item in candidate_ids if int(item) != int(center)),
                candidate_scope=candidate_scope,
                labeled_node_count=labeled_node_count,
                num_nodes=num_nodes,
            )
        )
        if candidate_arr:
            total_candidate_count += int(len(candidate_arr))
            centers_with_relation_neighbors += 1
        candidate_lists.append(candidate_arr)

    return {
        "center_node_ids": list(normalized_centers),
        "center_candidate_node_ids": [[int(v) for v in row] for row in candidate_lists],
        "center_source": str(center_source),
        "center_count": int(len(normalized_centers)),
        "centers_with_relation_neighbors": int(centers_with_relation_neighbors),
        "mean_relation_candidates_per_center": (
            float(total_candidate_count / float(centers_with_relation_neighbors)) if centers_with_relation_neighbors else 0.0
        ),
    }


def feature_k_nearest_neighbor_query(feature_tensor, k):
    feature_cpu = (
        feature_tensor.detach().cpu().float()
        if torch.is_tensor(feature_tensor)
        else torch.tensor(feature_tensor, dtype=torch.float32)
    )
    if feature_cpu.dim() != 2:
        raise ValueError("feature_tensor must be a 2D tensor for KNN hypergraph augmentation.")
    feature_cpu = torch.nan_to_num(feature_cpu, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    num_nodes = int(feature_cpu.size(0))
    if num_nodes == 0:
        return np.empty((0, 0), dtype=np.int64), np.empty((0, 0), dtype=np.float32), "empty"

    k_effective = min(int(k), num_nodes)
    if k_effective <= 0:
        raise ValueError("k must be positive for KNN hypergraph augmentation.")

    try:
        from scipy.spatial import cKDTree

        feature_np = feature_cpu.numpy()
        tree = cKDTree(feature_np)
        distances, neighbors = tree.query(feature_np, k=k_effective)
        backend = "scipy_ckdtree"
    except Exception as exc:
        if num_nodes > 4096:
            raise RuntimeError(
                "KNN hypergraph augmentation requires SciPy cKDTree for graphs larger than 4096 nodes. "
                "Install/repair SciPy or lower the graph size before retrying."
            ) from exc
        pairwise_distance = torch.cdist(feature_cpu, feature_cpu, p=2)
        topk = torch.topk(pairwise_distance, k=k_effective, largest=False, dim=1)
        distances = topk.values.detach().cpu().numpy()
        neighbors = topk.indices.detach().cpu().numpy()
        backend = "torch_cdist_topk"

    neighbors = np.asarray(neighbors, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float32)
    if neighbors.ndim == 1:
        neighbors = neighbors.reshape(-1, 1)
        distances = distances.reshape(-1, 1)

    node_ids = np.arange(num_nodes, dtype=np.int64)
    has_self = (neighbors == node_ids[:, None]).any(axis=1)
    if neighbors.shape[1] > 0 and not np.all(has_self):
        neighbors[~has_self, -1] = node_ids[~has_self]
        distances[~has_self, -1] = 0.0
    return neighbors, distances, backend


def hyperscan_knn_hypergraph_proxy_augment(edge_index, edge_type, feature_tensor, k):
    edge_index_cpu = (
        edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
    )
    edge_type_cpu = (
        edge_type.detach().cpu().long() if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long)
    )
    if edge_index_cpu.dim() != 2 or edge_index_cpu.size(0) != 2:
        raise ValueError("edge_index must be shaped [2, num_edges] for hyperscan_knn_hypergraph_proxy_augment.")
    if edge_type_cpu.numel() != edge_index_cpu.size(1):
        raise ValueError("edge_type must align with edge_index for hyperscan_knn_hypergraph_proxy_augment.")

    feature_cpu = (
        feature_tensor.detach().cpu().float()
        if torch.is_tensor(feature_tensor)
        else torch.tensor(feature_tensor, dtype=torch.float32)
    )
    if feature_cpu.dim() != 2:
        raise ValueError("feature_tensor must be 2D for hyperscan_knn_hypergraph_proxy_augment.")

    num_nodes = int(feature_cpu.size(0))
    num_edges_before = int(edge_index_cpu.size(1))
    relation_cardinality_before = _relation_cardinality_from_edge_type(edge_type_cpu, minimum=0)

    if num_nodes <= 1:
        stats = {
            "mode": "hyperscan_knn_hypergraph_proxy_augment",
            "budget_requested": 0.0,
            "num_nodes": int(num_nodes),
            "num_edges_before": int(num_edges_before),
            "num_edges_after": int(num_edges_before),
            "num_edges_added": 0,
            "num_hyperedges": int(num_nodes),
            "knn_k": int(min(int(k), max(num_nodes, 1))),
            "proxy_expansion": "bidirectional_center_star",
            "feature_source": "g0_projected_input_features",
            "backend": "degenerate_singleton",
            "relation_id_assigned": int(relation_cardinality_before),
            "relation_cardinality_before": int(relation_cardinality_before),
            "relation_cardinality_after": int(relation_cardinality_before),
            "mean_neighbor_distance": None,
            "mean_neighbors_per_hyperedge": 0.0,
            "duplicate_proxy_edges_removed": 0,
        }
        return edge_index_cpu.contiguous(), edge_type_cpu.contiguous(), stats

    neighbors, distances, backend = feature_k_nearest_neighbor_query(feature_cpu, k)
    node_ids = np.arange(num_nodes, dtype=np.int64)[:, None]
    membership_mask = neighbors != node_ids
    center_ids = np.broadcast_to(node_ids, neighbors.shape)
    center_members = center_ids[membership_mask]
    neighbor_members = neighbors[membership_mask]

    if center_members.size == 0:
        stats = {
            "mode": "hyperscan_knn_hypergraph_proxy_augment",
            "budget_requested": 0.0,
            "num_nodes": int(num_nodes),
            "num_edges_before": int(num_edges_before),
            "num_edges_after": int(num_edges_before),
            "num_edges_added": 0,
            "num_hyperedges": int(num_nodes),
            "knn_k": int(neighbors.shape[1]),
            "proxy_expansion": "bidirectional_center_star",
            "feature_source": "g0_projected_input_features",
            "backend": backend,
            "relation_id_assigned": int(relation_cardinality_before),
            "relation_cardinality_before": int(relation_cardinality_before),
            "relation_cardinality_after": int(relation_cardinality_before),
            "mean_neighbor_distance": None,
            "mean_neighbors_per_hyperedge": 0.0,
            "duplicate_proxy_edges_removed": 0,
        }
        return edge_index_cpu.contiguous(), edge_type_cpu.contiguous(), stats

    proxy_pairs = np.concatenate(
        [
            np.stack([center_members, neighbor_members], axis=1),
            np.stack([neighbor_members, center_members], axis=1),
        ],
        axis=0,
    )
    proxy_pairs_unique = np.unique(proxy_pairs, axis=0)
    new_edge_index = torch.from_numpy(proxy_pairs_unique.T.copy()).long()
    relation_id_assigned = int(relation_cardinality_before)
    new_edge_type = torch.full((int(new_edge_index.size(1)),), relation_id_assigned, dtype=torch.long)

    augmented_edge_index = torch.cat([edge_index_cpu, new_edge_index], dim=1).contiguous()
    augmented_edge_type = torch.cat([edge_type_cpu, new_edge_type], dim=0).contiguous()
    num_edges_added = int(new_edge_index.size(1))
    num_edges_after = int(augmented_edge_index.size(1))
    duplicate_proxy_edges_removed = int(proxy_pairs.shape[0] - proxy_pairs_unique.shape[0])
    mean_neighbor_distance = float(distances[membership_mask].mean().item()) if membership_mask.any() else None
    mean_neighbors_per_hyperedge = float(membership_mask.sum() / float(num_nodes)) if num_nodes else 0.0
    stats = {
        "mode": "hyperscan_knn_hypergraph_proxy_augment",
        "budget_requested": 0.0,
        "num_nodes": int(num_nodes),
        "num_edges_before": int(num_edges_before),
        "num_edges_after": int(num_edges_after),
        "num_edges_added": int(num_edges_added),
        "num_hyperedges": int(num_nodes),
        "knn_k": int(neighbors.shape[1]),
        "proxy_expansion": "bidirectional_center_star",
        "feature_source": "g0_projected_input_features",
        "backend": backend,
        "relation_id_assigned": int(relation_id_assigned),
        "relation_cardinality_before": int(relation_cardinality_before),
        "relation_cardinality_after": int(_relation_cardinality_from_edge_type(augmented_edge_type, minimum=0)),
        "mean_neighbor_distance": mean_neighbor_distance,
        "mean_neighbors_per_hyperedge": mean_neighbors_per_hyperedge,
        "duplicate_proxy_edges_removed": int(duplicate_proxy_edges_removed),
    }
    return augmented_edge_index, augmented_edge_type, stats


def relation_overlap_knn_proxy_augment(edge_index, edge_type, feature_tensor, center_node_ids, center_source="labeled_prefix", k=8, refine_request=None):
    refine_request = dict(refine_request or {})
    edge_index_cpu = (
        edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
    )
    edge_type_cpu = (
        edge_type.detach().cpu().long() if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long)
    )
    if edge_index_cpu.dim() != 2 or edge_index_cpu.size(0) != 2:
        raise ValueError("edge_index must be shaped [2, num_edges] for relation_overlap_knn_proxy_augment.")
    if edge_type_cpu.numel() != edge_index_cpu.size(1):
        raise ValueError("edge_type must align with edge_index for relation_overlap_knn_proxy_augment.")

    feature_cpu = (
        feature_tensor.detach().cpu().float()
        if torch.is_tensor(feature_tensor)
        else torch.tensor(feature_tensor, dtype=torch.float32)
    )
    if feature_cpu.dim() != 2:
        raise ValueError("feature_tensor must be 2D for relation_overlap_knn_proxy_augment.")

    num_nodes = int(feature_cpu.size(0))
    num_edges_before = int(edge_index_cpu.size(1))
    relation_cardinality_before = _relation_cardinality_from_edge_type(edge_type_cpu, minimum=0)
    center_ids = np.asarray([int(item) for item in center_node_ids], dtype=np.int64)

    candidate_scope = str(refine_request.get("candidate_scope", "undirected_relation_1hop") or "undirected_relation_1hop")
    labeled_node_count = refine_request.get("labeled_node_count", None)
    candidate_bundle = build_relation_overlap_center_candidates(
        edge_index=edge_index_cpu,
        num_nodes=int(num_nodes),
        center_node_ids=center_ids.tolist(),
        center_source=center_source,
        candidate_scope=candidate_scope,
        labeled_node_count=labeled_node_count,
    )

    feature_norm = F.normalize(
        torch.nan_to_num(feature_cpu, nan=0.0, posinf=0.0, neginf=0.0),
        p=2,
        dim=1,
        eps=1e-12,
    ).cpu().numpy()

    selected_pairs = []
    total_candidate_count = 0
    total_selected_count = 0
    centers_with_relation_neighbors = 0
    centers_with_selected_overlap = 0

    for center, candidate_row in zip(center_ids.tolist(), list(candidate_bundle["center_candidate_node_ids"])):
        if not candidate_row:
            continue
        candidate_arr = np.asarray([int(item) for item in candidate_row], dtype=np.int64)
        if candidate_arr.size == 0:
            continue
        centers_with_relation_neighbors += 1
        total_candidate_count += int(candidate_arr.size)
        center_vec = feature_norm[int(center)]
        candidate_vecs = feature_norm[candidate_arr]
        scores = candidate_vecs @ center_vec
        topk = min(int(k), int(candidate_arr.size))
        if topk <= 0:
            continue
        if topk == int(candidate_arr.size):
            top_positions = np.argsort(-scores, kind="stable")
        else:
            partial = np.argpartition(-scores, topk - 1)[:topk]
            top_positions = partial[np.argsort(-scores[partial], kind="stable")]
        selected = candidate_arr[top_positions]
        if selected.size == 0:
            continue
        centers_with_selected_overlap += 1
        total_selected_count += int(selected.size)
        selected_pairs.append(np.stack([np.full(selected.shape, int(center), dtype=np.int64), selected], axis=1))
        selected_pairs.append(np.stack([selected, np.full(selected.shape, int(center), dtype=np.int64)], axis=1))

    if not selected_pairs:
        stats = {
            "mode": "relation_overlap_knn_proxy_augment",
            "budget_requested": 0.0,
            "num_nodes": int(num_nodes),
            "num_edges_before": int(num_edges_before),
            "num_edges_after": int(num_edges_before),
            "num_edges_added": 0,
            "num_hyperedges": int(center_ids.size),
            "center_count": int(center_ids.size),
            "center_source": str(center_source),
            "routed_nodes_path": str(refine_request.get("routed_nodes_path", "") or ""),
            "routed_nodes_split": str(refine_request.get("routed_nodes_split", "all") or "all"),
            "knn_k": int(k),
            "candidate_scope": str(refine_request.get("candidate_scope", "undirected_relation_1hop")),
            "proxy_expansion": str(refine_request.get("proxy_expansion", "bidirectional_center_star")),
            "feature_source": str(refine_request.get("feature_source", "g0_projected_input_features")),
            "backend": "local_relation_overlap_topk",
            "similarity_metric": str(refine_request.get("similarity_metric", "cosine")),
            "relation_id_assigned": int(relation_cardinality_before),
            "relation_cardinality_before": int(relation_cardinality_before),
            "relation_cardinality_after": int(relation_cardinality_before),
            "mean_neighbor_distance": None,
            "mean_relation_candidates_per_center": 0.0,
            "mean_selected_overlap_per_center": 0.0,
            "centers_with_relation_neighbors": int(centers_with_relation_neighbors),
            "centers_with_selected_overlap": int(centers_with_selected_overlap),
            "duplicate_proxy_edges_removed": 0,
        }
        return edge_index_cpu.contiguous(), edge_type_cpu.contiguous(), stats

    proxy_pairs = np.concatenate(selected_pairs, axis=0)
    proxy_pairs_unique = np.unique(proxy_pairs, axis=0)
    new_edge_index = torch.from_numpy(proxy_pairs_unique.T.copy()).long()
    relation_id_assigned = int(relation_cardinality_before)
    new_edge_type = torch.full((int(new_edge_index.size(1)),), relation_id_assigned, dtype=torch.long)

    augmented_edge_index = torch.cat([edge_index_cpu, new_edge_index], dim=1).contiguous()
    augmented_edge_type = torch.cat([edge_type_cpu, new_edge_type], dim=0).contiguous()
    num_edges_added = int(new_edge_index.size(1))
    num_edges_after = int(augmented_edge_index.size(1))
    duplicate_proxy_edges_removed = int(proxy_pairs.shape[0] - proxy_pairs_unique.shape[0])
    stats = {
        "mode": "relation_overlap_knn_proxy_augment",
        "budget_requested": 0.0,
        "num_nodes": int(num_nodes),
        "num_edges_before": int(num_edges_before),
        "num_edges_after": int(num_edges_after),
        "num_edges_added": int(num_edges_added),
        "num_hyperedges": int(center_ids.size),
        "center_count": int(center_ids.size),
        "center_source": str(center_source),
        "routed_nodes_path": str(refine_request.get("routed_nodes_path", "") or ""),
        "routed_nodes_split": str(refine_request.get("routed_nodes_split", "all") or "all"),
        "knn_k": int(k),
        "candidate_scope": str(refine_request.get("candidate_scope", "undirected_relation_1hop")),
        "proxy_expansion": str(refine_request.get("proxy_expansion", "bidirectional_center_star")),
        "feature_source": str(refine_request.get("feature_source", "g0_projected_input_features")),
        "backend": "local_relation_overlap_topk",
        "similarity_metric": str(refine_request.get("similarity_metric", "cosine")),
        "relation_id_assigned": int(relation_id_assigned),
        "relation_cardinality_before": int(relation_cardinality_before),
        "relation_cardinality_after": int(_relation_cardinality_from_edge_type(augmented_edge_type, minimum=0)),
        "mean_neighbor_distance": None,
        "mean_relation_candidates_per_center": (
            float(total_candidate_count / float(centers_with_relation_neighbors)) if centers_with_relation_neighbors else 0.0
        ),
        "mean_selected_overlap_per_center": (
            float(total_selected_count / float(centers_with_selected_overlap)) if centers_with_selected_overlap else 0.0
        ),
        "centers_with_relation_neighbors": int(centers_with_relation_neighbors),
        "centers_with_selected_overlap": int(centers_with_selected_overlap),
        "duplicate_proxy_edges_removed": int(duplicate_proxy_edges_removed),
    }
    return augmented_edge_index, augmented_edge_type, stats


def _center_induced_relation_stats(node_id, candidate, texts, context, selection_features):
    following = context["following"]
    follower = context["follower"]
    undirected = context["undirected"]
    text_len = len(texts[candidate]) if isinstance(texts[candidate], str) else 0
    has_text = 1 if text_len > 0 else 0
    ego_follows_candidate = (candidate in following[node_id]) or (node_id in follower[candidate])
    candidate_follows_ego = (candidate in follower[node_id]) or (node_id in following[candidate])
    mutual = 1 if (ego_follows_candidate and candidate_follows_ego) else 0
    common_neighbors = len(undirected[node_id].intersection(undirected[candidate]))
    candidate_degree = len(undirected[candidate])
    center_vec = selection_features[int(node_id)]
    candidate_vec = selection_features[int(candidate)]
    similarity = float(torch.sum(center_vec * candidate_vec).item())
    nontrivial_structure = 1 if (candidate_degree > 1 or common_neighbors > 0 or mutual > 0) else 0
    return {
        "candidate": int(candidate),
        "similarity": similarity,
        "mutual": int(mutual),
        "common_neighbors": int(common_neighbors),
        "candidate_degree": int(candidate_degree),
        "text_length": int(text_len),
        "has_text": int(has_text),
        "nontrivial_structure": int(nontrivial_structure),
        "reciprocal": int(mutual),
    }


def _center_support_sort_key(stats):
    return (
        -float(stats["similarity"]),
        -int(stats["mutual"]),
        -int(stats["common_neighbors"]),
        -int(stats["candidate_degree"]),
        -int(stats["text_length"]),
        int(stats["candidate"]),
    )


def _center_contrast_sort_key(stats):
    return (
        float(stats["similarity"]),
        -int(stats["has_text"]),
        -int(stats["nontrivial_structure"]),
        -int(stats["candidate_degree"]),
        -int(stats["common_neighbors"]),
        -int(stats["text_length"]),
        int(stats["candidate"]),
    )


def _partition_center_induced_candidates(candidate_stats, quota):
    quota = max(int(quota), 0)
    support_quota = int(np.ceil(float(quota) / 2.0))
    contrast_quota = int(np.floor(float(quota) / 2.0))
    support_ranked = sorted(candidate_stats, key=_center_support_sort_key)
    selected_support = support_ranked[:support_quota]
    used = {int(item["candidate"]) for item in selected_support}
    contrast_ranked = sorted(
        [item for item in candidate_stats if int(item["candidate"]) not in used],
        key=_center_contrast_sort_key,
    )
    selected_contrast = contrast_ranked[:contrast_quota]
    used.update(int(item["candidate"]) for item in selected_contrast)
    fallback_used = False
    if len(selected_support) + len(selected_contrast) < quota:
        fallback_used = True
        support_overflow = [item for item in support_ranked if int(item["candidate"]) not in used]
        contrast_overflow = [item for item in contrast_ranked if int(item["candidate"]) not in used]
        overflow = support_overflow + contrast_overflow
        for item in overflow:
            if len(selected_support) + len(selected_contrast) >= quota:
                break
            if int(item["candidate"]) in used:
                continue
            if len(selected_support) < support_quota:
                selected_support.append(item)
            else:
                selected_contrast.append(item)
            used.add(int(item["candidate"]))
    return selected_support, selected_contrast, fallback_used


def _selected_similarity_mean(items):
    if not items:
        return 0.0
    return float(np.mean([float(item["similarity"]) for item in items]))


def _selected_reciprocal_ratio(items):
    if not items:
        return 0.0
    return float(np.mean([float(item["reciprocal"]) for item in items]))


def select_center_induced_directional_neighbors(
    node_id,
    texts,
    context,
    selection_features,
    following_quota,
    follower_quota,
):
    following_candidates = [
        _center_induced_relation_stats(node_id, cand, texts, context, selection_features)
        for cand in sorted(set(int(n) for n in context["following"][node_id] if int(n) >= 0 and int(n) != int(node_id)))
    ]
    follower_candidates = [
        _center_induced_relation_stats(node_id, cand, texts, context, selection_features)
        for cand in sorted(set(int(n) for n in context["follower"][node_id] if int(n) >= 0 and int(n) != int(node_id)))
    ]

    following_support, following_contrast, following_fallback = _partition_center_induced_candidates(
        following_candidates,
        quota=int(following_quota),
    )
    follower_support, follower_contrast, follower_fallback = _partition_center_induced_candidates(
        follower_candidates,
        quota=int(follower_quota),
    )

    summary = {
        "candidate_count_following": int(len(following_candidates)),
        "candidate_count_follower": int(len(follower_candidates)),
        "selected_count_following": int(len(following_support) + len(following_contrast)),
        "selected_count_follower": int(len(follower_support) + len(follower_contrast)),
        "mean_sim_following_support": _selected_similarity_mean(following_support),
        "mean_sim_following_contrast": _selected_similarity_mean(following_contrast),
        "mean_sim_follower_support": _selected_similarity_mean(follower_support),
        "mean_sim_follower_contrast": _selected_similarity_mean(follower_contrast),
        "reciprocal_ratio_following_selected": _selected_reciprocal_ratio(following_support + following_contrast),
        "reciprocal_ratio_follower_selected": _selected_reciprocal_ratio(follower_support + follower_contrast),
        "fallback_used": bool(following_fallback or follower_fallback),
    }
    return {
        "following_support": following_support,
        "following_contrast": following_contrast,
        "follower_support": follower_support,
        "follower_contrast": follower_contrast,
        "summary": summary,
    }


def _select_topk_positions(scores, k):
    topk = min(int(k), int(scores.numel()))
    if topk <= 0:
        return scores.new_empty((0,), dtype=torch.long)
    return torch.topk(scores, k=topk, largest=True).indices


def _append_bucket_positions(selected, used, scores, bucket_mask, quota):
    if int(quota) <= 0:
        return 0
    positions = torch.nonzero(bucket_mask, as_tuple=False).view(-1)
    if positions.numel() == 0:
        return 0
    ordered_local = _select_topk_positions(scores[positions], int(quota))
    added = 0
    for pos in positions[ordered_local].tolist():
        pos = int(pos)
        if pos in used:
            continue
        selected.append(pos)
        used.add(pos)
        added += 1
        if added >= int(quota):
            break
    return added


def _select_dynamic_neighbors(candidate_ids, scores, roles, knn_k, candidate_policy, candidate_policy_quotas):
    policy = str(candidate_policy or "default").strip().lower()
    topk = min(int(knn_k), int(candidate_ids.numel()))
    if topk <= 0:
        return candidate_ids.new_empty((0,), dtype=torch.long), roles.new_empty((0,), dtype=torch.long)
    if policy in {"default", ""} or roles is None:
        selected_positions = _select_topk_positions(scores, topk)
        return candidate_ids[selected_positions], roles[selected_positions] if roles is not None else None

    quotas = dict(candidate_policy_quotas or {})
    selected_positions = []
    used_positions = set()
    refill = True

    if policy == "stable_quota":
        stable_quota = int(quotas.get("stable", max(1, topk // 2)))
        _append_bucket_positions(
            selected_positions,
            used_positions,
            scores,
            (roles == 2) | (roles == 3),
            min(stable_quota, topk),
        )
    elif policy == "mixed_quota":
        for role_name, role_id in (("hard", 1), ("stable", 2), ("counterfactual", 3)):
            remaining = topk - len(selected_positions)
            if remaining <= 0:
                break
            quota_key = "stable_mixed" if role_name == "stable" and "stable_mixed" in quotas else role_name
            quota = int(quotas.get(quota_key, 0))
            _append_bucket_positions(
                selected_positions,
                used_positions,
                scores,
                roles == int(role_id),
                min(quota, remaining),
            )
    elif policy == "post_topk_exclude_routed":
        top_positions = _select_topk_positions(scores, topk)
        keep_mask = roles[top_positions] != 1
        selected_positions = [int(pos) for pos in top_positions[keep_mask].tolist()]
        used_positions = set(selected_positions)
        refill = False
    elif policy in {"exclude_routed", "llm_retain"}:
        keep_mask = roles != 1
        kept_positions = torch.nonzero(keep_mask, as_tuple=False).view(-1)
        if kept_positions.numel() == 0:
            return candidate_ids.new_empty((0,), dtype=torch.long), roles.new_empty((0,), dtype=torch.long)
        ordered_local = _select_topk_positions(scores[kept_positions], topk)
        selected_positions = [int(pos) for pos in kept_positions[ordered_local].tolist()]
        used_positions = set(selected_positions)
    else:
        selected_positions = []
        used_positions = set()

    remaining = (topk - len(selected_positions)) if refill else 0
    if remaining > 0:
        all_positions = _select_topk_positions(scores, int(scores.numel()))
        for pos in all_positions.tolist():
            pos = int(pos)
            if pos in used_positions:
                continue
            if policy in {"exclude_routed", "post_topk_exclude_routed", "llm_retain"} and int(roles[pos].item()) == 1:
                continue
            selected_positions.append(pos)
            used_positions.add(pos)
            remaining -= 1
            if remaining <= 0:
                break

    if not selected_positions:
        return candidate_ids.new_empty((0,), dtype=torch.long), roles.new_empty((0,), dtype=torch.long)
    selected_position_tensor = torch.tensor(selected_positions, dtype=torch.long, device=candidate_ids.device)
    return candidate_ids[selected_position_tensor], roles[selected_position_tensor]


def build_dynamic_hypergraph(
    feature_tensor,
    center_node_ids,
    center_candidate_node_ids,
    branch_enabled=True,
    knn_k=8,
    static_stats=None,
    center_candidate_role_ids=None,
    center_candidate_keep_mask=None,
    candidate_policy="default",
    candidate_policy_quotas=None,
):
    stats = {
        **dict(static_stats or {}),
        "hypergraph_branch_active": bool(branch_enabled),
        "num_hyperedges_built": 0,
        "incidence_count": 0,
        "center_count_with_members": 0,
        "mean_selected_neighbors_per_center": 0.0,
        "incident_node_count": 0,
        "selected_role_hard": 0,
        "selected_role_stable": 0,
        "selected_role_counterfactual": 0,
        "selected_role_generic": 0,
        "llm_retain_candidate_count": 0,
        "llm_retain_dropped_count": 0,
    }
    center_ids_tensor = center_node_ids.detach().cpu().long().view(-1) if torch.is_tensor(center_node_ids) else torch.tensor(center_node_ids, dtype=torch.long).view(-1)
    if not branch_enabled or center_ids_tensor.numel() == 0:
        return None, None, stats

    feature_view = F.normalize(
        torch.nan_to_num(feature_tensor.detach(), nan=0.0, posinf=0.0, neginf=0.0),
        p=2,
        dim=1,
        eps=1e-12,
    )
    device = feature_tensor.device
    node_members = []
    hyperedge_members = []
    incident_mask = torch.zeros(feature_tensor.size(0), dtype=torch.bool, device=device)
    total_selected_neighbors = 0
    valid_hyperedges = 0
    role_rows = list(center_candidate_role_ids) if center_candidate_role_ids is not None else None
    keep_rows = list(center_candidate_keep_mask) if center_candidate_keep_mask is not None else None

    hyperedge_id = 0
    for row_idx, (center_id, candidate_ids_raw) in enumerate(zip(center_ids_tensor.tolist(), list(center_candidate_node_ids))):
        center = int(center_id)
        if center < 0 or center >= int(feature_view.size(0)):
            continue
        candidate_ids = candidate_ids_raw.to(device=device) if torch.is_tensor(candidate_ids_raw) else torch.tensor(candidate_ids_raw, dtype=torch.long, device=device)
        valid_mask = (candidate_ids >= 0) & (candidate_ids < int(feature_view.size(0)))
        if role_rows is not None:
            roles_raw = role_rows[int(row_idx)]
            roles = roles_raw.to(device=device) if torch.is_tensor(roles_raw) else torch.tensor(roles_raw, dtype=torch.long, device=device)
            if int(roles.numel()) != int(candidate_ids.numel()):
                raise ValueError("dynamic_similarity_branch candidate role ids must align with candidate ids.")
            roles = roles[valid_mask].long()
        else:
            roles = None
        candidate_ids = candidate_ids[valid_mask]
        if keep_rows is not None:
            keep_raw = keep_rows[int(row_idx)]
            keep_mask = keep_raw.to(device=device) if torch.is_tensor(keep_raw) else torch.tensor(keep_raw, dtype=torch.bool, device=device)
            if int(keep_mask.numel()) != int(valid_mask.numel()):
                raise ValueError("dynamic_similarity_branch candidate keep mask must align with candidate ids.")
            keep_mask = keep_mask[valid_mask].bool()
            stats["llm_retain_candidate_count"] += int(keep_mask.numel())
            stats["llm_retain_dropped_count"] += int((~keep_mask).sum().item())
            candidate_ids = candidate_ids[keep_mask]
            if roles is not None:
                roles = roles[keep_mask]
        if candidate_ids.numel() == 0:
            continue
        scores = torch.mv(feature_view[candidate_ids], feature_view[center])
        selected, selected_roles = _select_dynamic_neighbors(
            candidate_ids=candidate_ids,
            scores=scores,
            roles=roles,
            knn_k=knn_k,
            candidate_policy=candidate_policy,
            candidate_policy_quotas=candidate_policy_quotas,
        )
        if selected.numel() == 0:
            continue
        if selected_roles is not None and selected_roles.numel() > 0:
            stats["selected_role_hard"] += int((selected_roles == 1).sum().item())
            stats["selected_role_stable"] += int((selected_roles == 2).sum().item())
            stats["selected_role_counterfactual"] += int((selected_roles == 3).sum().item())
            stats["selected_role_generic"] += int((selected_roles == 0).sum().item())
        members = torch.cat(
            [
                torch.tensor([center], dtype=torch.long, device=device),
                selected,
            ],
            dim=0,
        ).unique(sorted=False)
        if members.numel() == 0:
            continue
        incident_mask[members] = True
        node_members.append(members)
        hyperedge_members.append(torch.full((members.numel(),), hyperedge_id, dtype=torch.long, device=device))
        total_selected_neighbors += int(max(int(members.numel()) - 1, 0))
        valid_hyperedges += 1
        hyperedge_id += 1

    if not node_members:
        stats["hypergraph_branch_active"] = False
        return None, incident_mask, stats

    hyperedge_index = torch.stack([torch.cat(node_members, dim=0), torch.cat(hyperedge_members, dim=0)], dim=0)
    stats.update(
        {
            "num_hyperedges_built": int(valid_hyperedges),
            "incidence_count": int(hyperedge_index.size(1)),
            "center_count_with_members": int(valid_hyperedges),
            "mean_selected_neighbors_per_center": (
                float(total_selected_neighbors / float(valid_hyperedges)) if valid_hyperedges else 0.0
            ),
            "incident_node_count": int(incident_mask.sum().item()),
        }
    )
    return hyperedge_index, incident_mask, stats


def _batch_local_routed_roles(node_ids, routed_node_ids, num_nodes, device):
    if node_ids is None or routed_node_ids is None:
        return None
    node_ids_t = node_ids.to(device=device).long().view(-1) if torch.is_tensor(node_ids) else torch.tensor(node_ids, dtype=torch.long, device=device).view(-1)
    if int(node_ids_t.numel()) != int(num_nodes):
        raise ValueError(
            "batch-local routed member filtering requires node_ids to align with feature_tensor rows: "
            f"expected {int(num_nodes)}, got {int(node_ids_t.numel())}."
        )
    routed_ids_t = (
        routed_node_ids.to(device=device).long().view(-1)
        if torch.is_tensor(routed_node_ids)
        else torch.tensor(routed_node_ids, dtype=torch.long, device=device).view(-1)
    )
    if int(routed_ids_t.numel()) == 0:
        return torch.zeros((int(num_nodes),), dtype=torch.bool, device=device)
    return torch.isin(node_ids_t, routed_ids_t)


def _batch_local_feature_k_nearest_neighbor_query(feature_tensor, k):
    feature = torch.nan_to_num(
        feature_tensor.detach().float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).contiguous()
    num_nodes = int(feature.size(0))
    if num_nodes == 0:
        return torch.empty((0, 0), dtype=torch.long, device=feature_tensor.device), "empty"
    k_effective = min(int(k), num_nodes)
    if k_effective <= 0:
        raise ValueError("k must be positive for batch-local KNN hypergraph construction.")
    with torch.no_grad():
        distances = torch.cdist(feature, feature, p=2)
        neighbors = torch.topk(distances, k=k_effective, largest=False, dim=1).indices.long()
    return neighbors, f"torch_cdist_topk_{feature.device.type}"


def build_batch_local_knn_hypergraph(
    feature_tensor,
    branch_enabled=True,
    knn_k=8,
    static_stats=None,
    node_ids=None,
    routed_node_ids=None,
    candidate_policy="default",
    candidate_policy_quotas=None,
):
    policy = str(candidate_policy or "default").strip().lower()
    if policy in {"stable_quota", "mixed_quota"}:
        raise ValueError(
            "batch-local KNN currently supports only default, exclude_routed, "
            "or post_topk_exclude_routed candidate policies."
        )
    stats = {
        **dict(static_stats or {}),
        "hypergraph_branch_active": bool(branch_enabled),
        "num_hyperedges_built": 0,
        "incidence_count": 0,
        "center_count_with_members": 0,
        "mean_selected_neighbors_per_center": 0.0,
        "incident_node_count": 0,
        "candidate_policy": policy,
        "candidate_role_contract": "batch_local_global_routed_mask" if policy != "default" else "none",
        "selected_role_hard": 0,
        "selected_role_stable": 0,
        "selected_role_counterfactual": 0,
        "selected_role_generic": 0,
        "removed_role_hard": 0,
        "batch_local_routed_candidate_count": 0,
    }
    if not branch_enabled or feature_tensor.size(0) == 0:
        return None, None, stats

    device = feature_tensor.device
    routed_roles = _batch_local_routed_roles(
        node_ids=node_ids,
        routed_node_ids=routed_node_ids,
        num_nodes=int(feature_tensor.size(0)),
        device=device,
    )
    if routed_roles is not None:
        stats["batch_local_routed_candidate_count"] = int(routed_roles.sum().item())
    query_k = int(knn_k)
    if policy == "exclude_routed" and routed_roles is not None:
        query_k = min(int(feature_tensor.size(0)), int(knn_k) + int(routed_roles.sum().item()))
    neighbors, backend = _batch_local_feature_k_nearest_neighbor_query(feature_tensor, query_k)
    if int(neighbors.numel()) == 0:
        stats["backend"] = backend
        return None, torch.zeros(feature_tensor.size(0), dtype=torch.bool, device=feature_tensor.device), stats

    node_members = []
    hyperedge_members = []
    incident_mask = torch.zeros(feature_tensor.size(0), dtype=torch.bool, device=device)
    total_selected_neighbors = 0

    for hyperedge_id, member_row in enumerate(neighbors.detach().cpu().tolist()):
        selected = []
        removed_hard = 0
        for raw_item in member_row:
            item = int(raw_item)
            if item < 0 or item >= int(feature_tensor.size(0)):
                continue
            is_center = item == int(hyperedge_id)
            is_routed_member = bool(routed_roles[item].item()) if routed_roles is not None else False
            if policy == "post_topk_exclude_routed" and is_routed_member and not is_center:
                removed_hard += 1
                continue
            if policy == "exclude_routed" and is_routed_member and not is_center:
                removed_hard += 1
                continue
            selected.append(item)
            if policy == "exclude_routed" and len(selected) >= int(knn_k):
                break
        members = torch.tensor(sorted(set(selected)), dtype=torch.long, device=device)
        if members.numel() == 0:
            continue
        non_center_members = members[members != int(hyperedge_id)]
        if routed_roles is not None and int(non_center_members.numel()) > 0:
            selected_hard = int(routed_roles[non_center_members].sum().item())
            stats["selected_role_hard"] += selected_hard
            stats["selected_role_generic"] += int(non_center_members.numel()) - selected_hard
        else:
            stats["selected_role_generic"] += int(non_center_members.numel())
        stats["removed_role_hard"] += int(removed_hard)
        incident_mask[members] = True
        node_members.append(members)
        hyperedge_members.append(torch.full((members.numel(),), int(hyperedge_id), dtype=torch.long, device=device))
        total_selected_neighbors += int(max(int(members.numel()) - 1, 0))

    if not node_members:
        stats["backend"] = backend
        stats["hypergraph_branch_active"] = False
        return None, incident_mask, stats

    hyperedge_index = torch.stack([torch.cat(node_members, dim=0), torch.cat(hyperedge_members, dim=0)], dim=0)
    valid_hyperedges = int(len(node_members))
    stats.update(
        {
            "backend": backend,
            "num_hyperedges_built": valid_hyperedges,
            "incidence_count": int(hyperedge_index.size(1)),
            "center_count_with_members": valid_hyperedges,
            "mean_selected_neighbors_per_center": (
                float(total_selected_neighbors / float(valid_hyperedges)) if valid_hyperedges else 0.0
            ),
            "incident_node_count": int(incident_mask.sum().item()),
        }
    )
    return hyperedge_index, incident_mask, stats
