from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from conflict_refiner import (
    DignnStructuralConfig,
    build_directed_topology_features,
    fit_dignn_structural_encoder,
    harmful_edge_score,
    js_divergence_from_probs,
)
from model_building import _idx_tensor, _score_logits, resolve_g0_feature_bundle
from utils import save_stage_artifacts, write_json, write_text, write_torch


def _score_all(labels, pred):
    from sklearn.metrics import accuracy_score, f1_score

    labels_np = np.asarray(labels)
    pred_np = np.asarray(pred)
    return {
        "accuracy": float(accuracy_score(labels_np, pred_np)),
        "macro_f1": float(f1_score(labels_np, pred_np, average="macro", zero_division=0)),
        "bot_f1": float(f1_score(labels_np, pred_np, average="binary", zero_division=0)),
        "count": int(labels_np.shape[0]),
    }


def _split_metrics_from_outputs(outputs, labels_np, valid_mask, test_mask):
    pred = outputs.get("pred")
    pred_np = pred.detach().cpu().numpy().astype(np.int64) if torch.is_tensor(pred) else np.asarray(pred, dtype=np.int64)
    return {
        "valid": _score_all(labels_np[valid_mask], pred_np[valid_mask]) if valid_mask.any() else {"accuracy": 0.0, "macro_f1": 0.0, "bot_f1": 0.0, "count": 0},
        "test": _score_all(labels_np[test_mask], pred_np[test_mask]) if test_mask.any() else {"accuracy": 0.0, "macro_f1": 0.0, "bot_f1": 0.0, "count": 0},
        "pred": pred_np,
    }


def _degree_arrays(edge_index, num_nodes):
    edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
    src = edge_index_cpu[0]
    dst = edge_index_cpu[1]
    out_degree = torch.bincount(src, minlength=int(num_nodes)).cpu().numpy().astype(np.int64)
    in_degree = torch.bincount(dst, minlength=int(num_nodes)).cpu().numpy().astype(np.int64)
    return in_degree, out_degree


def _remove_edge_at_position(edge_index, edge_type, edge_pos):
    edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
    edge_type_cpu = edge_type.detach().cpu().long() if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long)
    keep_mask = torch.ones(edge_index_cpu.size(1), dtype=torch.bool)
    keep_mask[int(edge_pos)] = False
    return edge_index_cpu[:, keep_mask].contiguous(), edge_type_cpu[keep_mask].contiguous()


def _pruned_graph_from_removed_positions(edge_index, edge_type, removed_positions):
    edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
    edge_type_cpu = edge_type.detach().cpu().long() if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long)
    if not removed_positions:
        return edge_index_cpu.contiguous(), edge_type_cpu.contiguous()
    keep_mask = torch.ones(edge_index_cpu.size(1), dtype=torch.bool)
    keep_mask[torch.tensor(sorted(int(pos) for pos in removed_positions), dtype=torch.long)] = False
    return edge_index_cpu[:, keep_mask].contiguous(), edge_type_cpu[keep_mask].contiguous()


def _fit_router(stage_runner, semantic_prob, structural_prob):
    valid_idx = np.asarray(stage_runner.data["valid_idx"], dtype=np.int64)
    test_idx = np.asarray(stage_runner.data["test_idx"], dtype=np.int64)
    budget = float(
        getattr(stage_runner.args, "local_dignn_conflict_router_budget", None)
        or getattr(stage_runner.args, "conflict_router_budget", 0.10)
        or 0.10
    )
    if not (0.0 < budget < 1.0):
        raise ValueError("--local_dignn_conflict_router_budget must be in (0, 1) for local_dignn_conflict_refine_diag.")
    conflict_score = js_divergence_from_probs(structural_prob, semantic_prob).detach().cpu().numpy().astype(np.float32)
    valid_scores = conflict_score[valid_idx]
    if valid_scores.size == 0:
        raise RuntimeError("local_dignn_conflict_refine_diag requires non-empty valid conflict scores.")
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


def _run_structural_forward(model, device, topology_features, attribute_features):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.forward_outputs(topology_features.to(device), attribute_features.to(device))
    payload = {key: value.detach().cpu() for key, value in outputs.items()}
    return payload


def run_local_dignn_conflict_refine_diag(stage_runner, stage_dir: Path, base_bundle: Dict[str, object]) -> Dict[str, object]:
    context = stage_runner.ensure_backbone_context()
    gnn_outputs = context["gnn_outputs"]
    base_pred = gnn_outputs.get("pred")
    if base_pred is None:
        raise RuntimeError("local_dignn_conflict_refine_diag requires frozen_g0 outputs with pred.")

    edge_index, edge_type = stage_runner._active_graph_tensors()
    labels_t = torch.tensor(stage_runner.labels, dtype=torch.long)
    labels_np = labels_t.numpy()
    semantic_bundle = resolve_g0_feature_bundle(stage_runner.args, stage_runner.data)
    semantic_raw = semantic_bundle["raw_features"].detach().cpu().float()
    semantic_norm = F.normalize(semantic_raw, p=2, dim=1, eps=1e-12)
    degree_guard_enabled = not bool(
        getattr(stage_runner.args, "local_dignn_conflict_disable_degree_guard", False)
        or getattr(stage_runner.args, "conflict_disable_degree_guard", False)
    )
    topk_per_bucket = max(
        int(
            getattr(stage_runner.args, "local_dignn_conflict_topk_per_bucket", None)
            or getattr(stage_runner.args, "conflict_topk_per_bucket", 1)
            or 1
        ),
        1,
    )

    lm_head = stage_runner._build_or_load_lm_only_head()
    semantic_prob = lm_head["outputs"].get("prob_cal", lm_head["outputs"]["prob"]).detach().cpu().float()
    topology_features = build_directed_topology_features(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=int(labels_t.numel()),
        n_relations=int(getattr(stage_runner.args, "n_relations", 2)),
    ).float()
    config = DignnStructuralConfig()
    structural_bundle = fit_dignn_structural_encoder(
        topology_features=topology_features,
        attribute_features=semantic_raw,
        labels=labels_t,
        train_idx=_idx_tensor(stage_runner.data["train_idx"]),
        valid_idx=_idx_tensor(stage_runner.data["valid_idx"]),
        test_idx=_idx_tensor(stage_runner.data["test_idx"]),
        semantic_prob=semantic_prob,
        device=stage_runner.device,
        config=config,
        score_logits_fn=_score_logits,
    )
    structural_outputs = structural_bundle["outputs"]
    structural_prob = structural_outputs["fused_prob"].detach().cpu().float()
    structural_pred = structural_outputs["pred"].detach().cpu().long()
    node_support_cpu = structural_bundle["node_support"].detach().cpu().float()

    router_bundle = _fit_router(stage_runner, semantic_prob, structural_prob)
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
    base_in_degree, base_out_degree = _degree_arrays(edge_index_cpu, num_nodes)

    routed_node_ids = np.flatnonzero(routed_mask)
    routed_node_ids = np.asarray(sorted(routed_node_ids.tolist(), key=lambda node_id: (-float(conflict_score[int(node_id)]), int(node_id))), dtype=np.int64)
    incident_edges: List[List[int]] = [[] for _ in range(num_nodes)]
    for edge_pos, (u, v) in enumerate(zip(src_np.tolist(), dst_np.tolist())):
        incident_edges[int(u)].append(int(edge_pos))
        if int(v) != int(u):
            incident_edges[int(v)].append(int(edge_pos))

    edge_after_cache = {}
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
            if edge_pos not in edge_after_cache:
                cf_edge_index, cf_edge_type = _remove_edge_at_position(edge_index_cpu, edge_type_cpu, edge_pos)
                cf_topology = build_directed_topology_features(
                    edge_index=cf_edge_index,
                    edge_type=cf_edge_type,
                    num_nodes=num_nodes,
                    n_relations=int(getattr(stage_runner.args, "n_relations", 2)),
                ).float()
                cf_outputs = _run_structural_forward(structural_bundle["model"], stage_runner.device, cf_topology, semantic_raw)
                cf_prob = cf_outputs["fused_prob"].detach().cpu().float()
                cf_conflict = js_divergence_from_probs(cf_prob, semantic_prob).detach().cpu().numpy().astype(np.float32)
                edge_after_cache[int(edge_pos)] = {
                    "conflict": cf_conflict,
                    "node_support_mean": float(cf_outputs["node_support"].mean().item()) if "node_support" in cf_outputs else 0.0,
                }
            conflict_before = float(conflict_score[int(target_id)])
            conflict_after = float(edge_after_cache[int(edge_pos)]["conflict"][int(target_id)])
            reconstruction_support = float(node_support_cpu[int(target_id)].item()) if node_support_cpu.numel() else 0.0
            delta_conflict = float(conflict_before - conflict_after)
            score = harmful_edge_score(delta_conflict, reconstruction_support, float(config.harmful_support_weight))
            row = {
                "target_id": int(target_id),
                "edge": [u, v],
                "relation": int(relation_id),
                "target_role": str(target_role),
                "edge_pos": int(edge_pos),
                "semantic_cosine": float((semantic_norm[u] * semantic_norm[v]).sum().item()),
                "conflict_before": conflict_before,
                "conflict_after": conflict_after,
                "delta_conflict": delta_conflict,
                "reconstruction_support": reconstruction_support,
                "harmful_score": float(score),
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
            key=lambda row: (-float(row["harmful_score"]), -float(row["delta_conflict"]), int(row["edge_pos"])),
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
            row["selection_reason"] = "topk_positive_delta_conflict_with_reconstruction_support"
            selected_count += 1
        conflict_bucket_budget[bucket_key] = int(selected_count)

    for row in candidate_rows:
        if row["selection_reason"] is None:
            if float(row["delta_conflict"]) <= 0.0:
                row["selection_reason"] = "non_positive_delta_conflict"
            else:
                row["selection_reason"] = "not_topk_positive_delta_conflict"

    rng = np.random.default_rng(int(stage_runner.seed))
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

    conflict_edge_index, conflict_edge_type = _pruned_graph_from_removed_positions(edge_index_cpu, edge_type_cpu, conflict_removed_positions)
    rand_edge_index, rand_edge_type = _pruned_graph_from_removed_positions(edge_index_cpu, edge_type_cpu, rand_removed_positions)
    conflict_zero_in, conflict_zero_out = _degree_arrays(conflict_edge_index, num_nodes)
    rand_zero_in, rand_zero_out = _degree_arrays(rand_edge_index, num_nodes)

    local_conflict_context = stage_runner._rerun_frozen_g0_with_edge_override(stage_dir, "local_dignn_conflict", conflict_edge_index, conflict_edge_type)
    local_rand_context = stage_runner._rerun_frozen_g0_with_edge_override(stage_dir, "local_dignn_rand", rand_edge_index, rand_edge_type)

    base_metrics = _split_metrics_from_outputs(context["frozen_g0"]["outputs"], labels_np, stage_runner.val_mask, stage_runner.test_mask)
    conflict_metrics = _split_metrics_from_outputs(local_conflict_context["outputs"], labels_np, stage_runner.val_mask, stage_runner.test_mask)
    rand_metrics = _split_metrics_from_outputs(local_rand_context["outputs"], labels_np, stage_runner.val_mask, stage_runner.test_mask)

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
    for split_name, idx_mask in (("train", stage_runner.train_mask), ("valid", stage_runner.val_mask), ("test", stage_runner.test_mask)):
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
            -float(row["harmful_score"]),
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
            "reconstruction_support": float(row["reconstruction_support"]),
            "harmful_score": float(row["harmful_score"]),
            "removed_from_propagation": True,
            "evidence_role": "camouflage_conflict_evidence",
        }
        for row in selected_rows_sorted
        if bool(row["selected"])
    ]

    metrics_payload = {
        "contract": "local_dignn_conflict_refine_diag_metrics_v1",
        "routing_mode": "dignn_style_dual_view_conflict_plus_reconstruction",
        "paper_faithful_dignn_style": True,
        "official_code_verified": False,
        "routed_budget": routed_budget,
        "topk_per_bucket": int(topk_per_bucket),
        "base": {"valid": base_metrics["valid"], "test": base_metrics["test"]},
        "local_dignn_conflict": {
            "valid": conflict_metrics["valid"],
            "test": conflict_metrics["test"],
            "delta_vs_base": {
                "valid_accuracy": float(conflict_metrics["valid"]["accuracy"] - base_metrics["valid"]["accuracy"]),
                "valid_macro_f1": float(conflict_metrics["valid"]["macro_f1"] - base_metrics["valid"]["macro_f1"]),
                "test_accuracy": float(conflict_metrics["test"]["accuracy"] - base_metrics["test"]["accuracy"]),
                "test_macro_f1": float(conflict_metrics["test"]["macro_f1"] - base_metrics["test"]["macro_f1"]),
            },
        },
        "local_dignn_rand": {
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
            "node_support_mean": float(node_support_cpu.mean().item()) if node_support_cpu.numel() else 0.0,
            "model_config": structural_bundle["model_config"],
        },
        "routing": {
            "routing_threshold_valid": float(routing_threshold),
            "routed_counts": {
                "train": int(routed_mask[stage_runner.train_mask].sum()),
                "valid": int(routed_mask[stage_runner.val_mask].sum()),
                "test": int(routed_mask[stage_runner.test_mask].sum()),
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
            "selected_reconstruction_support_mean": float(np.mean([row["reconstruction_support"] for row in selected_rows_sorted if bool(row["selected"])])) if evidence_rows else 0.0,
        },
    }

    manifest = {
        "contract": "local_dignn_conflict_refine_diag_v1",
        "status": "completed",
        "stage": "local_dignn_conflict_refine_diag",
        "routing_mode": "dignn_style_dual_view_conflict_plus_reconstruction",
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
        "semantic_view": {"source": lm_head.get("source", "lm_only_head"), "manifest": lm_head["manifest"]},
        "dual_view_proxy": {
            "topology_features": "directed_degree_relation_statistics",
            "attribute_features": "raw_semantic_embedding",
            "fusion": "attention_weighted_sum",
            "mi_objective": "topology_attribute_mutual_exclusive_proxy",
        },
        "conflict_definition": "JS(p_struct(node;G), p_sem(node))",
        "counterfactual_definition": "delta_conflict = conflict_before - conflict_after_single_edge_delete",
        "harmful_edge_scoring": "delta_conflict + support_weight*(1-reconstruction_support)",
        "test_labels_used_for_training": False,
        "test_labels_used_for_threshold": False,
        "oracle_labels_used": False,
        "frozen_g0_dir": str(context["frozen_g0"]["dir"]),
        "frozen_g0_manifest": context["frozen_g0"]["manifest"],
        "graph_source": context["graph_bundle"]["source"],
        "graph_source_root": context["graph_bundle"]["source_root"],
        "local_dignn_conflict_frozen_g0_dir": str(local_conflict_context["dir"]),
        "local_dignn_rand_frozen_g0_dir": str(local_rand_context["dir"]),
        "local_dignn_conflict_edge_index_path": str(stage_dir / "local_dignn_conflict_edge_index.pt"),
        "local_dignn_conflict_edge_type_path": str(stage_dir / "local_dignn_conflict_edge_type.pt"),
        "local_dignn_rand_edge_index_path": str(stage_dir / "local_dignn_rand_edge_index.pt"),
        "local_dignn_rand_edge_type_path": str(stage_dir / "local_dignn_rand_edge_type.pt"),
        "conflict_evidence_edges_path": str(stage_dir / "conflict_evidence_edges.jsonl"),
    }
    stage_runner._write_stage_manifest(stage_dir, manifest)
    write_json(stage_dir / "metrics.json", metrics_payload)
    write_text(
        stage_dir / "selected_edges.jsonl",
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in selected_rows_sorted) + ("\n" if selected_rows_sorted else ""),
    )
    write_text(
        stage_dir / "routed_nodes.jsonl",
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in routed_rows) + ("\n" if routed_rows else ""),
    )
    write_text(
        stage_dir / "conflict_evidence_edges.jsonl",
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in evidence_rows) + ("\n" if evidence_rows else ""),
    )
    write_torch(stage_dir / "local_dignn_conflict_edge_index.pt", conflict_edge_index)
    write_torch(stage_dir / "local_dignn_conflict_edge_type.pt", conflict_edge_type)
    write_torch(stage_dir / "local_dignn_rand_edge_index.pt", rand_edge_index)
    write_torch(stage_dir / "local_dignn_rand_edge_type.pt", rand_edge_type)
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
                "topology_features": topology_features,
                "attribute_features": semantic_raw,
                "topology_repr": structural_outputs["topology_repr"],
                "attribute_repr": structural_outputs["attribute_repr"],
                "fused_repr": structural_outputs["fused_repr"],
                "attention_weights": structural_outputs["attention_weights"],
                "node_support": node_support_cpu,
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
                "This stage trains a DIGNN-style structural encoder with disentangled shared/private views and edge reconstruction support.",
                "Routed hard nodes are selected by high JS divergence between structural and semantic posteriors.",
                "Candidate edges are target-centered 1-hop incident edges, bucketed by (relation, target_role).",
                "Propagation validation is performed by retraining frozen_g0 on the edited graph only.",
                "Removed conflict edges are preserved as evidence-ready provenance for later local evidence graph consumption.",
            ]
        )
        + "\n",
    )
    return {
        "stage": "local_dignn_conflict_refine_diag",
        "stage_dir": str(stage_dir),
        "metrics": metrics_payload,
    }
