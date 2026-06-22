from pathlib import Path

from artifact_contracts import MissingFrozenArtifactError
from stage_registry import resolve_stage_name
from utils import build_preparation_dir, ensure_dir, read_json, safe_torch_load


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


def gate_dir(experiment_root, gate_name):
    if gate_name == GATS_GATE_NAME:
        return ensure_dir(build_preparation_dir(experiment_root, "graph_calibrator"))
    return ensure_dir(Path(experiment_root) / "frozen" / "gates" / gate_name)


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
