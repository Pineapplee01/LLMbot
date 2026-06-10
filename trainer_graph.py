import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from artifact_contracts import MissingFrozenArtifactError
from estimators import fit_temperature_scaling
from model_building import _idx_tensor, _score_logits, resolve_g0_feature_bundle
from trainer_preparation import _resolve_optional_graph_override_paths, build_or_load_frozen_g0
from utils import ensure_dir, safe_torch_load, save_stage_artifacts, tensor_sha256, write_json, write_text, write_torch


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


def _classification_consistency_proxy(probabilities, reference_labels):
    prob_np = probabilities.detach().cpu().numpy() if torch.is_tensor(probabilities) else np.asarray(probabilities)
    label_np = np.asarray(reference_labels, dtype=np.int64)
    if prob_np.ndim != 2 or prob_np.shape[1] < 2:
        return np.zeros(prob_np.shape[0] if prob_np.ndim >= 1 else 0, dtype=np.float32)
    if label_np.ndim == 0:
        label_np = np.full(prob_np.shape[0], int(label_np), dtype=np.int64)
    if label_np.shape[0] < prob_np.shape[0]:
        if label_np.shape[0] == 0:
            return np.zeros(prob_np.shape[0], dtype=np.float32)
        pad_value = int(label_np[-1])
        label_np = np.pad(label_np, (0, prob_np.shape[0] - label_np.shape[0]), constant_values=pad_value)
    label_np = np.clip(label_np[: prob_np.shape[0]], a_min=0, a_max=prob_np.shape[1] - 1)
    row_idx = np.arange(prob_np.shape[0], dtype=np.int64)
    clipped = np.clip(prob_np[row_idx, label_np], a_min=1e-8, a_max=1.0)
    return clipped.astype(np.float32)


class GraphStageMixin:
    """Shared graph-stage runtime owner for the active mainline.

    This mixin owns graph bundle resolution and graph-override reruns while
    heavier local graph diagnostic executors still live in the legacy body
    during the current migration window.
    """

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
        num_nodes = int(self.graph_node_count)
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
                "dataset_node_count_match": bool(num_nodes == int(self.graph_node_count)),
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

    def _build_or_load_lm_only_head(self):
        graph_variant = str(getattr(self, "graph_data_variant", "labeled")).lower()
        if graph_variant != "full_graph_support":
            return super()._build_or_load_lm_only_head()

        existing = self._load_lm_only_head()
        if existing is not None:
            existing_outputs = existing.get("outputs", {})
            existing_prob = existing_outputs.get("prob_cal", existing_outputs.get("prob"))
            existing_rows = int(existing_prob.shape[0]) if torch.is_tensor(existing_prob) else 0
            manifest = existing.get("manifest", {})
            manifest_variant = str(
                manifest.get("graph_data_variant", manifest.get("feature_manifest", {}).get("graph_data_variant", "labeled"))
            ).lower()
            if existing_rows == int(self.graph_node_count) and manifest_variant == "full_graph_support":
                return existing

        if bool(getattr(self.args, "claim_grade", False)):
            raise MissingFrozenArtifactError(
                "claim_grade full-graph local conflict diagnostics require a full-graph lm_only_head artifact; "
                "implicit recompute is disallowed."
            )

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        context = self.ensure_backbone_context()
        feature_manifest = context["frozen_g0"]["manifest"].get("feature_manifest", {})
        feature_path = feature_manifest.get("labeled_embedding_path") or feature_manifest.get("path")
        if not feature_path or not Path(feature_path).exists():
            raise MissingFrozenArtifactError(
                "Full-graph local conflict diagnostics require a readable labeled semantic embedding tensor "
                "from the frozen G0 feature manifest."
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
        train_idx = self.data["train_idx"].detach().cpu().long().numpy() if torch.is_tensor(self.data["train_idx"]) else np.asarray(self.data["train_idx"], dtype=np.int64)
        valid_idx = self.data["valid_idx"].detach().cpu().long().numpy() if torch.is_tensor(self.data["valid_idx"]) else np.asarray(self.data["valid_idx"], dtype=np.int64)
        test_idx = self.data["test_idx"].detach().cpu().long().numpy() if torch.is_tensor(self.data["test_idx"]) else np.asarray(self.data["test_idx"], dtype=np.int64)
        if features.shape[0] < labels.shape[0]:
            raise MissingFrozenArtifactError(
                f"Full-graph lm_only_head features ({int(features.shape[0])} rows) must cover labeled nodes ({int(labels.shape[0])})."
            )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train_idx])
        x_all = scaler.transform(features)
        classes = np.unique(labels[train_idx])
        output_rows = int(features.shape[0])
        if classes.size < 2:
            prob = np.zeros((output_rows, 2), dtype=np.float32)
            prob[:, int(classes[0])] = 1.0
            classifier_payload = {"type": "constant", "class": int(classes[0])}
        else:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(x_train, labels[train_idx])
            prob = clf.predict_proba(x_all).astype(np.float32)
            if prob.shape[1] == 1:
                fixed = np.zeros((output_rows, 2), dtype=np.float32)
                fixed[:, int(clf.classes_[0])] = prob[:, 0]
                prob = fixed
            elif list(clf.classes_) != [0, 1]:
                fixed = np.zeros((output_rows, 2), dtype=np.float32)
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
            "graph_data_variant": "full_graph_support",
            "training_scope": "train_idx_supervised_lightweight_head_labeled_prefix",
            "calibration_scope": "valid_idx_temperature_scaling_labeled_prefix",
            "input_source": "cached_semantic_embedding_from_frozen_g0_manifest",
            "feature_manifest": feature_bundle["feature_manifest"],
            "temperature": float(temperature),
            "output_rows": int(output_rows),
            "graph_node_count": int(self.graph_node_count),
            "labeled_node_count": int(self.labeled_node_count),
            "support_node_count": int(self.support_node_count),
            "split_counts": {
                "train": int(train_idx.size),
                "valid": int(valid_idx.size),
                "test": int(test_idx.size),
            },
        }
        write_json(out_dir / "manifest.json", manifest)
        return {"manifest": manifest, "outputs": outputs, "dir": out_dir, "source": "derived_from_frozen_g0_feature_manifest"}

    def _fit_local_conflict_structural_view(self, edge_index, edge_type, labels_t):
        graph_variant = str(getattr(self, "graph_data_variant", "labeled")).lower()
        if graph_variant != "full_graph_support":
            return super()._fit_local_conflict_structural_view(edge_index, edge_type, labels_t)

        import trainer_legacy_impl as _legacy_impl

        train_idx = _idx_tensor(self.data["train_idx"])
        valid_idx = _idx_tensor(self.data["valid_idx"])
        test_idx = _idx_tensor(self.data["test_idx"])
        if train_idx.numel() == 0 or valid_idx.numel() == 0:
            raise MissingFrozenArtifactError(
                "local_conflict_prune_diag requires non-empty train_idx and valid_idx for structural-view training."
            )

        labeled_count = int(labels_t.numel())
        num_nodes = int(self.graph_node_count)
        relation_cardinality = int(edge_type.max().item()) + 1 if edge_type.numel() else int(getattr(self.args, "n_relations", 2))
        model = _legacy_impl.StructuralConflictRGCN(
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
            labeled_logits = eval_outputs["logits"][:labeled_count].detach().cpu()
            valid_metrics = _score_logits(labeled_logits, labels_t.detach().cpu(), valid_idx.cpu())
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
            if _legacy_impl._is_better(valid_metrics, best_metrics):
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
        labeled_logits = final_outputs["logits"][:labeled_count].detach().cpu()
        return {
            "model": model,
            "outputs": {key: value.detach().cpu() for key, value in final_outputs.items()},
            "checkpoint": {key: value.detach().cpu().clone() for key, value in best_state.items()},
            "fit_summary": {
                "max_epochs": int(max_epochs),
                "patience": int(patience),
                "best_epoch": int(best_metrics.get("epoch", -1) if best_metrics else -1),
                "history": history,
                "supervision_scope": "labeled_prefix_only",
                "graph_node_count": int(self.graph_node_count),
                "labeled_node_count": int(self.labeled_node_count),
            },
            "metrics": {
                "train": _score_logits(labeled_logits, labels_t.detach().cpu(), train_idx.cpu()),
                "valid": _score_logits(labeled_logits, labels_t.detach().cpu(), valid_idx.cpu()),
                "test": _score_logits(labeled_logits, labels_t.detach().cpu(), test_idx.cpu()),
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
                "training_scope": "labeled_prefix_only_on_full_graph_structure",
                "graph_node_count": int(self.graph_node_count),
                "labeled_node_count": int(self.labeled_node_count),
                "checkpoint_selection": {
                    "primary": "validation_macro_f1",
                    "tie_breaker": "validation_loss",
                },
            },
        }

    @staticmethod
    def _split_metrics_from_outputs(outputs, labels_np, valid_mask, test_mask):
        pred = outputs.get("pred")
        if pred is None:
            raise MissingFrozenArtifactError("Expected outputs.pt to include 'pred' for graph diagnostics.")
        pred_np = pred.detach().cpu().numpy().astype(np.int64) if torch.is_tensor(pred) else np.asarray(pred, dtype=np.int64)
        labels_np = np.asarray(labels_np, dtype=np.int64)
        labeled_count = int(labels_np.shape[0])
        if pred_np.shape[0] < labeled_count:
            raise MissingFrozenArtifactError(
                f"Graph diagnostic predictions ({int(pred_np.shape[0])}) must cover labeled nodes ({labeled_count})."
            )
        pred_eval_np = pred_np[:labeled_count]
        return {
            "valid": _score_all(labels_np[valid_mask], pred_eval_np[valid_mask]) if valid_mask.any() else {"accuracy": 0.0, "macro_f1": 0.0, "bot_f1": 0.0, "count": 0},
            "test": _score_all(labels_np[test_mask], pred_eval_np[test_mask]) if test_mask.any() else {"accuracy": 0.0, "macro_f1": 0.0, "bot_f1": 0.0, "count": 0},
            "pred": pred_eval_np,
            "pred_full": pred_np,
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

    def _full_graph_labeled_prefix_edge_tensors(self):
        edge_index, edge_type = self._active_graph_tensors()
        graph_variant = str(getattr(self, "graph_data_variant", "labeled")).lower()
        if graph_variant != "full_graph_support":
            return edge_index.detach().cpu().long(), edge_type.detach().cpu().long()
        labeled_limit = int(self.labeled_node_count)
        edge_index_cpu = edge_index.detach().cpu().long()
        edge_type_cpu = edge_type.detach().cpu().long()
        keep_mask = (edge_index_cpu[0] < labeled_limit) & (edge_index_cpu[1] < labeled_limit)
        return edge_index_cpu[:, keep_mask].contiguous(), edge_type_cpu[keep_mask].contiguous()

    @staticmethod
    def _relation_direction_name(relation_id, target_role):
        relation_name = "following" if int(relation_id) == 1 else "follower"
        return f"{relation_name}_{str(target_role)}"

    @staticmethod
    def _reciprocity_per_node(edge_index, num_nodes):
        edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
        adjacency = [set() for _ in range(int(num_nodes))]
        src = edge_index_cpu[0].tolist()
        dst = edge_index_cpu[1].tolist()
        for u, v in zip(src, dst):
            u_i = int(u)
            v_i = int(v)
            if 0 <= u_i < int(num_nodes) and 0 <= v_i < int(num_nodes) and u_i != v_i:
                adjacency[u_i].add(v_i)
        reciprocity = np.zeros(int(num_nodes), dtype=np.float32)
        for node_id in range(int(num_nodes)):
            neighbors = adjacency[node_id]
            if not neighbors:
                continue
            reciprocal = sum(1 for neigh in neighbors if node_id in adjacency[neigh])
            reciprocity[node_id] = float(reciprocal / max(len(neighbors), 1))
        return reciprocity

    @staticmethod
    def _support_exposure(edge_index, edge_type, labeled_node_count):
        edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
        edge_type_cpu = edge_type.detach().cpu().long() if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long)
        labeled_count = int(labeled_node_count)
        src = edge_index_cpu[0].tolist()
        dst = edge_index_cpu[1].tolist()
        rel = edge_type_cpu.tolist()
        exposure = {}
        for node_id in range(labeled_count):
            support_neighbors = set()
            support_only_1hop = set()
            support_only_2hop = set()
            support_direction_counts = {
                "following_incoming": 0,
                "following_outgoing": 0,
                "follower_incoming": 0,
                "follower_outgoing": 0,
            }
            for edge_pos, (u, v, r) in enumerate(zip(src, dst, rel)):
                u_i = int(u)
                v_i = int(v)
                relation_id = int(r)
                if u_i == node_id and v_i >= labeled_count:
                    support_neighbors.add(v_i)
                    support_only_1hop.add(v_i)
                    support_direction_counts[GraphStageMixin._relation_direction_name(relation_id, "outgoing")] += 1
                elif v_i == node_id and u_i >= labeled_count:
                    support_neighbors.add(u_i)
                    support_only_1hop.add(u_i)
                    support_direction_counts[GraphStageMixin._relation_direction_name(relation_id, "incoming")] += 1
            if support_neighbors:
                first_hop = set(support_neighbors)
                for edge_pos, (u, v) in enumerate(zip(src, dst)):
                    u_i = int(u)
                    v_i = int(v)
                    if u_i in first_hop and v_i >= labeled_count and v_i != node_id:
                        support_only_2hop.add(v_i)
                    elif v_i in first_hop and u_i >= labeled_count and u_i != node_id:
                        support_only_2hop.add(u_i)
                support_only_2hop -= support_only_1hop
            exposure[node_id] = {
                "support_neighbor_ids": sorted(int(item) for item in support_neighbors),
                "support_neighbor_count_delta": int(len(support_neighbors)),
                "support_neighbors_by_relation_direction": support_direction_counts,
                "support_only_1hop_count": int(len(support_only_1hop)),
                "support_only_2hop_count": int(len(support_only_2hop)),
            }
        return exposure

    def _support_consistency(self, semantic_raw, semantic_prob, base_pred_np):
        graph_variant = str(getattr(self, "graph_data_variant", "labeled")).lower()
        labeled_count = int(self.labeled_node_count)
        if graph_variant != "full_graph_support" or semantic_raw.shape[0] <= labeled_count:
            return {
                node_id: {
                    "support_semantic_similarity_mean": 0.0,
                    "support_prediction_consistency_proxy_mean": 0.0,
                }
                for node_id in range(labeled_count)
            }
        semantic_norm = F.normalize(semantic_raw.detach().cpu().float(), p=2, dim=1, eps=1e-12)
        semantic_prob_np = semantic_prob.detach().cpu().numpy() if torch.is_tensor(semantic_prob) else np.asarray(semantic_prob)
        base_pred_full = np.asarray(base_pred_np, dtype=np.int64)
        if base_pred_full.shape[0] < semantic_prob_np.shape[0]:
            fallback_pred = semantic_prob_np.argmax(axis=1).astype(np.int64)
            if base_pred_full.shape[0] > 0:
                fallback_pred[: base_pred_full.shape[0]] = base_pred_full
            base_pred_full = fallback_pred
        exposure = self._support_exposure(*self._active_graph_tensors(), labeled_count)
        summary = {}
        for node_id in range(labeled_count):
            support_ids = exposure[node_id]["support_neighbor_ids"]
            if not support_ids:
                summary[node_id] = {
                    "support_semantic_similarity_mean": 0.0,
                    "support_prediction_consistency_proxy_mean": 0.0,
                }
                continue
            sims = []
            proxies = []
            node_vec = semantic_norm[node_id]
            target_pred = int(base_pred_full[node_id]) if base_pred_full.shape[0] > node_id else int(semantic_prob_np[node_id].argmax())
            for support_id in support_ids:
                if support_id >= semantic_norm.shape[0]:
                    continue
                sims.append(float((node_vec * semantic_norm[int(support_id)]).sum().item()))
                proxies.append(float(_classification_consistency_proxy(semantic_prob_np[[int(support_id)]], np.asarray([target_pred], dtype=np.int64))[0]))
            summary[node_id] = {
                "support_semantic_similarity_mean": float(np.mean(sims)) if sims else 0.0,
                "support_prediction_consistency_proxy_mean": float(np.mean(proxies)) if proxies else 0.0,
            }
        return summary

    def _structural_shock(self, full_edge_index, labeled_edge_index):
        labeled_count = int(self.labeled_node_count)
        full_in, full_out = self._degree_arrays(full_edge_index, int(self.graph_node_count))
        labeled_in, labeled_out = self._degree_arrays(labeled_edge_index, labeled_count)
        full_recip = self._reciprocity_per_node(full_edge_index, int(self.graph_node_count))[:labeled_count]
        labeled_recip = self._reciprocity_per_node(labeled_edge_index, labeled_count)
        shock = {}
        for node_id in range(labeled_count):
            full_total = int(full_in[node_id] + full_out[node_id])
            labeled_total = int(labeled_in[node_id] + labeled_out[node_id])
            full_disagreement = abs(int(full_in[node_id]) - int(full_out[node_id]))
            labeled_disagreement = abs(int(labeled_in[node_id]) - int(labeled_out[node_id]))
            shock[node_id] = {
                "full_in_degree": int(full_in[node_id]),
                "full_out_degree": int(full_out[node_id]),
                "labeled_in_degree": int(labeled_in[node_id]),
                "labeled_out_degree": int(labeled_out[node_id]),
                "delta_in_degree": int(full_in[node_id] - labeled_in[node_id]),
                "delta_out_degree": int(full_out[node_id] - labeled_out[node_id]),
                "delta_total_degree": int(full_total - labeled_total),
                "full_reciprocity": float(full_recip[node_id]),
                "labeled_reciprocity": float(labeled_recip[node_id]),
                "delta_reciprocity": float(full_recip[node_id] - labeled_recip[node_id]),
                "full_disagreement": int(full_disagreement),
                "labeled_disagreement": int(labeled_disagreement),
                "delta_disagreement": int(full_disagreement - labeled_disagreement),
            }
        return shock

    @staticmethod
    def _density_bucket(total_degree):
        degree = int(total_degree)
        if degree <= 1:
            return "sparse"
        if degree <= 5:
            return "low_density"
        return "high_density"

    @staticmethod
    def _failure_mechanism(row):
        if bool(row.get("support_induced_wrong", False)):
            return "support_induced_fragile"
        if int(row.get("has_following", 0)) == 0 and int(row.get("has_follower", 0)) == 0:
            return "sparse_isolated"
        if bool(row.get("was_wrong_base", False)) and int(row.get("has_follower", 0)) == 1 and int(row.get("has_following", 0)) == 1:
            return "dense_directional_recoverable"
        return "conflict_dominant"

    @staticmethod
    def _regime_label(row, prefix):
        directional = "both_present" if int(row.get(f"{prefix}_has_following", row.get("has_following", 0))) == 1 and int(row.get(f"{prefix}_has_follower", row.get("has_follower", 0))) == 1 else "no_directional"
        density = GraphStageMixin._density_bucket(row.get(f"{prefix}_total_degree", 0))
        correctness = "wrong" if bool(row.get(f"{prefix}_is_wrong", False)) else "correct"
        if prefix == "full" and bool(row.get("support_induced_wrong", False)):
            return "support-induced wrong"
        return f"{directional} -> {density} -> {correctness}"

    def _error_regime_shift(self, rows):
        regime_counts = {}
        total = max(len(rows), 1)
        for row in rows:
            labeled_regime = self._regime_label(row, "labeled")
            full_regime = self._regime_label(row, "full")
            key = f"{labeled_regime} => {full_regime}"
            regime_counts[key] = int(regime_counts.get(key, 0) + 1)
        return {
            key: {
                "count": int(value),
                "share": float(value / total),
            }
            for key, value in sorted(regime_counts.items(), key=lambda item: (-item[1], item[0]))
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
        degree_guard_enabled = not bool(getattr(self.args, "conflict_disable_degree_guard", False))
        topk_per_bucket = max(int(getattr(self.args, "conflict_topk_per_bucket", 1) or 1), 1)
        labeled_count = int(self.labeled_node_count)
        graph_variant = str(getattr(self, "graph_data_variant", "labeled")).lower()
        if semantic_raw.shape[0] < labeled_count:
            raise MissingFrozenArtifactError(
                f"local_conflict_prune_diag semantic rows ({int(semantic_raw.shape[0])}) must cover labeled nodes ({labeled_count})."
            )
        semantic_norm = F.normalize(semantic_raw, p=2, dim=1, eps=1e-12)

        lm_head = self._build_or_load_lm_only_head()
        semantic_prob = lm_head["outputs"].get("prob_cal", lm_head["outputs"]["prob"]).detach().cpu().float()
        if int(semantic_prob.shape[0]) < labeled_count:
            raise MissingFrozenArtifactError(
                f"local_conflict_prune_diag semantic probabilities ({int(semantic_prob.shape[0])}) must cover labeled nodes ({labeled_count})."
            )
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
        num_nodes = int(self.graph_node_count)
        base_in_degree, base_out_degree = self._degree_arrays(edge_index_cpu, num_nodes)
        labeled_edge_index_cpu, labeled_edge_type_cpu = self._full_graph_labeled_prefix_edge_tensors()
        labeled_in_degree, labeled_out_degree = self._degree_arrays(labeled_edge_index_cpu, labeled_count)
        support_exposure = self._support_exposure(edge_index_cpu, edge_type_cpu, labeled_count)
        support_consistency = self._support_consistency(semantic_raw, semantic_prob, base_pred.detach().cpu().numpy().astype(np.int64))
        structural_shock = self._structural_shock(edge_index_cpu, labeled_edge_index_cpu)

        routed_node_ids = np.flatnonzero(routed_mask[:labeled_count])
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

        base_pred_np = base_pred.detach().cpu().numpy().astype(np.int64)
        full_base_wrong = base_pred_np[:labeled_count] != labels_np[:labeled_count]
        local_conf_pred_np = conflict_metrics["pred"]
        local_rand_pred_np = rand_metrics["pred"]
        full_is_wrong = local_conf_pred_np[:labeled_count] != labels_np[:labeled_count]
        full_rand_is_wrong = local_rand_pred_np[:labeled_count] != labels_np[:labeled_count]
        support_induced_wrong_mask = (~full_base_wrong) & full_is_wrong
        labeled_directional = {}
        for node_id in range(labeled_count):
            labeled_directional[node_id] = {
                "has_following": int(labeled_out_degree[node_id] > 0),
                "has_follower": int(labeled_in_degree[node_id] > 0),
                "total_degree": int(labeled_in_degree[node_id] + labeled_out_degree[node_id]),
            }
        full_directional = {}
        for node_id in range(labeled_count):
            support_dirs = support_exposure[node_id]["support_neighbors_by_relation_direction"]
            full_directional[node_id] = {
                "has_following": int((base_out_degree[node_id] > 0) or support_dirs["following_outgoing"] > 0 or support_dirs["following_incoming"] > 0),
                "has_follower": int((base_in_degree[node_id] > 0) or support_dirs["follower_outgoing"] > 0 or support_dirs["follower_incoming"] > 0),
                "total_degree": int(base_in_degree[node_id] + base_out_degree[node_id]),
            }

        routed_rows = []
        p_sem_np = semantic_prob.detach().cpu().numpy()
        p_struct_np = structural_prob.detach().cpu().numpy()
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

        error_migration = {
            "labeled_wrong_to_full_graph_correct": int((full_base_wrong & (~full_is_wrong)).sum()),
            "labeled_correct_to_full_graph_wrong": int(((~full_base_wrong) & full_is_wrong).sum()),
            "labeled_wrong_stays_wrong": int((full_base_wrong & full_is_wrong).sum()),
            "labeled_correct_stays_correct": int(((~full_base_wrong) & (~full_is_wrong)).sum()),
            "labeled_correct_to_local_rand_wrong": int(((~full_base_wrong) & full_rand_is_wrong).sum()),
        }

        per_node_rows = []
        for node_id in np.flatnonzero(self.test_mask).tolist():
            support_stats = support_exposure[int(node_id)]
            support_consistency_row = support_consistency[int(node_id)]
            shock_row = structural_shock[int(node_id)]
            row = {
                "node_id": int(node_id),
                "label": int(labels_np[int(node_id)]),
                "split": "test",
                "routed": bool(routed_mask[int(node_id)]),
                "base_pred": int(base_pred_np[int(node_id)]),
                "local_conf_pred": int(local_conf_pred_np[int(node_id)]),
                "local_rand_pred": int(local_rand_pred_np[int(node_id)]),
                "was_wrong_base": bool(full_base_wrong[int(node_id)]),
                "is_wrong_full_graph": bool(full_is_wrong[int(node_id)]),
                "is_wrong_local_rand": bool(full_rand_is_wrong[int(node_id)]),
                "labeled_is_wrong": bool(full_base_wrong[int(node_id)]),
                "full_is_wrong": bool(full_is_wrong[int(node_id)]),
                "support_induced_wrong": bool(support_induced_wrong_mask[int(node_id)]),
                "labeled_has_following": int(labeled_directional[int(node_id)]["has_following"]),
                "labeled_has_follower": int(labeled_directional[int(node_id)]["has_follower"]),
                "full_has_following": int(full_directional[int(node_id)]["has_following"]),
                "full_has_follower": int(full_directional[int(node_id)]["has_follower"]),
                "has_following": int(full_directional[int(node_id)]["has_following"]),
                "has_follower": int(full_directional[int(node_id)]["has_follower"]),
                "labeled_total_degree": int(labeled_directional[int(node_id)]["total_degree"]),
                "full_total_degree": int(full_directional[int(node_id)]["total_degree"]),
                "support_neighbor_count_delta": int(support_stats["support_neighbor_count_delta"]),
                "support_neighbors_by_relation_direction": dict(support_stats["support_neighbors_by_relation_direction"]),
                "support_only_1hop_count": int(support_stats["support_only_1hop_count"]),
                "support_only_2hop_count": int(support_stats["support_only_2hop_count"]),
                "support_semantic_similarity_mean": float(support_consistency_row["support_semantic_similarity_mean"]),
                "support_prediction_consistency_proxy_mean": float(support_consistency_row["support_prediction_consistency_proxy_mean"]),
                "delta_in_degree": int(shock_row["delta_in_degree"]),
                "delta_out_degree": int(shock_row["delta_out_degree"]),
                "delta_total_degree": int(shock_row["delta_total_degree"]),
                "delta_reciprocity": float(shock_row["delta_reciprocity"]),
                "delta_disagreement": int(shock_row["delta_disagreement"]),
                "conflict_score": float(conflict_score[int(node_id)]),
            }
            row["failure_mechanism"] = self._failure_mechanism(row)
            row["labeled_regime"] = self._regime_label(row, "labeled")
            row["full_regime"] = self._regime_label(row, "full")
            per_node_rows.append(row)

        failure_mechanism_summary = {}
        for mechanism in ("sparse_isolated", "dense_directional_recoverable", "conflict_dominant", "support_induced_fragile"):
            subset = [row for row in per_node_rows if row["failure_mechanism"] == mechanism]
            if not subset:
                continue
            failure_mechanism_summary[mechanism] = {
                "count": int(len(subset)),
                "share": float(len(subset) / max(len(per_node_rows), 1)),
                "base_wrong_rate": float(sum(1 for row in subset if row["was_wrong_base"]) / len(subset)),
                "full_graph_wrong_rate": float(sum(1 for row in subset if row["is_wrong_full_graph"]) / len(subset)),
                "mean_support_neighbor_count_delta": float(np.mean([row["support_neighbor_count_delta"] for row in subset])),
                "mean_delta_total_degree": float(np.mean([row["delta_total_degree"] for row in subset])),
            }

        support_exposure_summary = {
            "mean_support_neighbor_count_delta": float(np.mean([row["support_neighbor_count_delta"] for row in per_node_rows])) if per_node_rows else 0.0,
            "mean_support_only_1hop_count": float(np.mean([row["support_only_1hop_count"] for row in per_node_rows])) if per_node_rows else 0.0,
            "mean_support_only_2hop_count": float(np.mean([row["support_only_2hop_count"] for row in per_node_rows])) if per_node_rows else 0.0,
        }
        structural_shock_summary = {
            "mean_delta_in_degree": float(np.mean([row["delta_in_degree"] for row in per_node_rows])) if per_node_rows else 0.0,
            "mean_delta_out_degree": float(np.mean([row["delta_out_degree"] for row in per_node_rows])) if per_node_rows else 0.0,
            "mean_delta_total_degree": float(np.mean([row["delta_total_degree"] for row in per_node_rows])) if per_node_rows else 0.0,
            "mean_delta_reciprocity": float(np.mean([row["delta_reciprocity"] for row in per_node_rows])) if per_node_rows else 0.0,
            "mean_delta_disagreement": float(np.mean([row["delta_disagreement"] for row in per_node_rows])) if per_node_rows else 0.0,
        }

        metrics_payload = {
            "contract": "local_conflict_prune_diag_metrics_v1",
            "routing_mode": "dignn_style_topology_semantic_conflict",
            "paper_faithful_dignn_style": True,
            "official_code_verified": False,
            "graph_data_variant": graph_variant,
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
            "error_migration": error_migration,
            "support_exposure": support_exposure_summary,
            "support_consistency": {
                "mean_support_semantic_similarity": float(np.mean([row["support_semantic_similarity_mean"] for row in per_node_rows])) if per_node_rows else 0.0,
                "mean_support_prediction_consistency_proxy": float(np.mean([row["support_prediction_consistency_proxy_mean"] for row in per_node_rows])) if per_node_rows else 0.0,
            },
            "structural_shock": structural_shock_summary,
            "error_regime_shift": self._error_regime_shift(per_node_rows),
            "failure_mechanisms": failure_mechanism_summary,
        }

        manifest = {
            "contract": "local_conflict_prune_diag_v1",
            "status": "completed",
            "stage": "local_conflict_prune_diag",
            "graph_data_variant": graph_variant,
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
            "supports_full_graph_diagnostic_comparison": bool(graph_variant == "full_graph_support"),
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
            "per_node_test_path": str(stage_dir / "per_node_test.jsonl"),
        }
        self._write_stage_manifest(
            stage_dir,
            manifest,
            artifact_namespace="stages/local_conflict_prune_diag",
            visibility="public",
            resolved_task="local_conflict_prune_diag",
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
        write_text(
            stage_dir / "per_node_test.jsonl",
            "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in per_node_rows)
            + ("\n" if per_node_rows else ""),
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
                    "base_pred": torch.tensor(base_pred_np, dtype=torch.long),
                    "local_conf_pred": torch.tensor(local_conf_pred_np, dtype=torch.long),
                    "local_rand_pred": torch.tensor(local_rand_pred_np, dtype=torch.long),
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
                    "When graph_data_variant=full_graph_support, the stage also compares labeled-prefix failure regimes against the support-augmented full graph and writes support-exposure, structural-shock, and error-migration diagnostics.",
                ]
            )
            + "\n",
        )
        return {
            "stage": "local_conflict_prune_diag",
            "stage_dir": str(stage_dir),
            "metrics": metrics_payload,
        }
