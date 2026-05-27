from pathlib import Path

from model_building import _idx_tensor
from stage_registry import legacy_name_for, resolve_stage_name
from utils import build_preparation_dir, ensure_dir, tensor_sha256


class MissingFrozenArtifactError(RuntimeError):
    pass


PHASE_A_CONTRACT = "phase_a_semantic_source_x_gnn_backbone"
PHASE_A_DISABLED_COMPONENTS = {
    "estimator_mode": "none",
    "semantic_mode": "off",
    "repair_mode": "noop",
    "selector_mode": "none",
    "appendix_mode": "none",
}


def canonical_stage_name(stage_name):
    return resolve_stage_name(stage_name)


def canonical_stage_dir(root, stage_name):
    return Path(root) / "stages" / canonical_stage_name(stage_name)


def legacy_stage_dir(root, stage_name):
    legacy_stage = legacy_name_for(stage_name) or canonical_stage_name(stage_name)
    return Path(root) / "stages" / legacy_stage


def stage_dir_read_candidates(root, stage_name):
    canonical_dir = canonical_stage_dir(root, stage_name)
    legacy_dir = legacy_stage_dir(root, stage_name)
    candidates = [canonical_dir]
    if legacy_dir != canonical_dir:
        candidates.append(legacy_dir)
    return candidates


def stage_artifact_reference(stage_name, filename):
    stage_name = canonical_stage_name(stage_name)
    suffix = filename if "." in str(filename) else f"{filename}.json"
    return f"stages/{stage_name}/{suffix}"


def preparation_artifact_reference(namespace, filename):
    suffix = filename if "." in str(filename) else f"{filename}.json"
    return f"preparation/{namespace}/{suffix}"


def split_provenance(data):
    return {
        "train": {
            "size": int(_idx_tensor(data["train_idx"]).numel()),
            "sha256": tensor_sha256(_idx_tensor(data["train_idx"])),
        },
        "valid": {
            "size": int(_idx_tensor(data["valid_idx"]).numel()),
            "sha256": tensor_sha256(_idx_tensor(data["valid_idx"])),
        },
        "test": {
            "size": int(_idx_tensor(data["test_idx"]).numel()),
            "sha256": tensor_sha256(_idx_tensor(data["test_idx"])),
        },
    }


def frozen_g0_dir(experiment_root):
    return ensure_dir(build_preparation_dir(experiment_root, "graph_detector"))
