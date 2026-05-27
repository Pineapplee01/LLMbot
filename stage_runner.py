import json
from pathlib import Path

import trainer_legacy_impl as _legacy_impl

from artifact_contracts import MissingFrozenArtifactError, split_provenance, stage_dir_read_candidates
from model_building import _labels_to_index
from runtime_env import _resolve_device
from stage_registry import legacy_name_for, resolve_stage_name
from trainer_glance import GlanceStageMixin
from trainer_graph import GraphStageMixin
from trainer_preparation import load_frozen_g0
from utils import (
    build_experiment_root,
    build_stage_dir,
    capture_code_metadata,
    safe_torch_load,
    tensor_sha256,
    write_json,
)


def _mask_from_idx(num_nodes, idx):
    mask = [False] * int(num_nodes)
    values = idx.detach().cpu().long().tolist() if hasattr(idx, "detach") else list(idx)
    for item in values:
        mask[int(item)] = True
    import numpy as np

    return np.asarray(mask, dtype=bool)


class StageRunner(GlanceStageMixin, GraphStageMixin, _legacy_impl.StageRunner):
    """Shared StageRunner skeleton for the active mainline.

    The heavy graph-aware and GLANCE execution branches still inherit from the
    legacy implementation during the current migration window. This owner class
    now controls shared runtime state, provenance wiring, dependency loading,
    and top-level dispatch.
    """

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
        self.legacy_execution_stage = legacy_name_for(self.execution_stage) or self.execution_stage
        self.requested_task = getattr(
            args,
            "requested_experiment_task",
            getattr(args, "experiment_task", self.requested_stage),
        )

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
        return stage_dir_read_candidates(self.experiment_root, stage_name)

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
            "artifact_namespace": artifact_namespace or f"stages/{self.execution_stage}",
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
