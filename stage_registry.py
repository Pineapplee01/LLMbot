from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class StageSpec:
    canonical_name: str
    legacy_names: Tuple[str, ...]
    visibility: str
    family: str
    runner_kind: str
    claim_grade_allowed: bool
    requires_canonical_split: bool
    forces_use_gnn: bool
    graph_data_mode: str
    artifact_namespace: str


def _spec(
    canonical_name,
    *,
    legacy_names=(),
    visibility,
    family,
    runner_kind,
    claim_grade_allowed,
    requires_canonical_split,
    forces_use_gnn,
    graph_data_mode,
    artifact_namespace,
):
    return StageSpec(
        canonical_name=canonical_name,
        legacy_names=tuple(legacy_names),
        visibility=visibility,
        family=family,
        runner_kind=runner_kind,
        claim_grade_allowed=bool(claim_grade_allowed),
        requires_canonical_split=bool(requires_canonical_split),
        forces_use_gnn=bool(forces_use_gnn),
        graph_data_mode=graph_data_mode,
        artifact_namespace=artifact_namespace,
    )


STAGE_REGISTRY: Dict[str, StageSpec] = {
    "distillation_pipeline": _spec(
        "distillation_pipeline",
        legacy_names=("legacy_distill",),
        visibility="public",
        family="pipeline",
        runner_kind="legacy_distill",
        claim_grade_allowed=False,
        requires_canonical_split=False,
        forces_use_gnn=False,
        graph_data_mode="optional",
        artifact_namespace="stages/distillation_pipeline",
    ),
    "semantic_encoder_finetune": _spec(
        "semantic_encoder_finetune",
        legacy_names=("semantic_finetune",),
        visibility="public",
        family="preparation",
        runner_kind="semantic_encoder_finetune",
        claim_grade_allowed=False,
        requires_canonical_split=True,
        forces_use_gnn=False,
        graph_data_mode="none",
        artifact_namespace="preparation/semantic_encoder",
    ),
    "semantic_embedding_classifier": _spec(
        "semantic_embedding_classifier",
        legacy_names=("lm_embedding_classifier",),
        visibility="public",
        family="preparation",
        runner_kind="semantic_embedding_classifier",
        claim_grade_allowed=False,
        requires_canonical_split=True,
        forces_use_gnn=False,
        graph_data_mode="none",
        artifact_namespace="preparation/semantic_embedding_classifier",
    ),
    "semantic_correction_gate": _spec(
        "semantic_correction_gate",
        legacy_names=(),
        visibility="public",
        family="preparation",
        runner_kind="semantic_correction_gate",
        claim_grade_allowed=False,
        requires_canonical_split=True,
        forces_use_gnn=False,
        graph_data_mode="none",
        artifact_namespace="preparation/semantic_correction_gate",
    ),
    "graph_detector_prepare": _spec(
        "graph_detector_prepare",
        legacy_names=("frozen_g0",),
        visibility="public",
        family="preparation",
        runner_kind="graph_detector_prepare",
        claim_grade_allowed=True,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="preparation/graph_detector",
    ),
    "graph_calibration_prepare": _spec(
        "graph_calibration_prepare",
        legacy_names=("frozen_gats",),
        visibility="public",
        family="preparation",
        runner_kind="graph_calibration_prepare",
        claim_grade_allowed=True,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="preparation/graph_calibrator",
    ),
    "local_conformal_diagnostic": _spec(
        "local_conformal_diagnostic",
        legacy_names=("local_conformal_prune_diag",),
        visibility="public",
        family="diagnostic",
        runner_kind="stage_runner",
        claim_grade_allowed=True,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/local_conformal_diagnostic",
    ),
    "local_conflict_prune_diag": _spec(
        "local_conflict_prune_diag",
        legacy_names=(),
        visibility="public",
        family="diagnostic",
        runner_kind="stage_runner",
        claim_grade_allowed=True,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/local_conflict_prune_diag",
    ),
    "local_dignn_conflict_refine_diag": _spec(
        "local_dignn_conflict_refine_diag",
        legacy_names=(),
        visibility="public",
        family="diagnostic",
        runner_kind="stage_runner",
        claim_grade_allowed=True,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/local_dignn_conflict_refine_diag",
    ),
    "joint_router_refinement": _spec(
        "joint_router_refinement",
        legacy_names=("glance_joint_router_refine",),
        visibility="public",
        family="refinement",
        runner_kind="stage_runner",
        claim_grade_allowed=True,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/joint_router_refinement",
    ),
    "router_only_ablation": _spec(
        "router_only_ablation",
        legacy_names=(),
        visibility="public",
        family="ablation",
        runner_kind="stage_runner",
        claim_grade_allowed=True,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/router_only_ablation",
    ),
    "prompt_expert_quality_audit": _spec(
        "prompt_expert_quality_audit",
        legacy_names=(),
        visibility="public",
        family="diagnostic",
        runner_kind="stage_runner",
        claim_grade_allowed=False,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/prompt_expert_quality_audit",
    ),
    "minimal_pipeline": _spec(
        "minimal_pipeline",
        legacy_names=("vertical_minimal",),
        visibility="public",
        family="pipeline",
        runner_kind="stage_runner",
        claim_grade_allowed=True,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/minimal_pipeline",
    ),
    "estimator_ablation": _spec(
        "estimator_ablation",
        legacy_names=("estimator_matrix",),
        visibility="public",
        family="ablation",
        runner_kind="stage_runner",
        claim_grade_allowed=True,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/estimator_ablation",
    ),
    "semantic_operator_ablation": _spec(
        "semantic_operator_ablation",
        legacy_names=("semantic_matrix",),
        visibility="public",
        family="ablation",
        runner_kind="stage_runner",
        claim_grade_allowed=True,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/semantic_operator_ablation",
    ),
    "semantic_source_ablation": _spec(
        "semantic_source_ablation",
        legacy_names=("semantic_source_matrix",),
        visibility="public",
        family="ablation",
        runner_kind="stage_runner",
        claim_grade_allowed=True,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/semantic_source_ablation",
    ),
    "repair_operator_ablation": _spec(
        "repair_operator_ablation",
        legacy_names=("repair_matrix",),
        visibility="public",
        family="ablation",
        runner_kind="stage_runner",
        claim_grade_allowed=True,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/repair_operator_ablation",
    ),
    "selector_ablation": _spec(
        "selector_ablation",
        legacy_names=("selector_matrix",),
        visibility="public",
        family="ablation",
        runner_kind="stage_runner",
        claim_grade_allowed=True,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/selector_ablation",
    ),
    "positioning_ablation": _spec(
        "positioning_ablation",
        legacy_names=("positioning_matrix",),
        visibility="public",
        family="ablation",
        runner_kind="stage_runner",
        claim_grade_allowed=True,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/positioning_ablation",
    ),
    "backbone_stress_test": _spec(
        "backbone_stress_test",
        legacy_names=("backbone_stress",),
        visibility="public",
        family="ablation",
        runner_kind="stage_runner",
        claim_grade_allowed=True,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/backbone_stress_test",
    ),
    "appendix_ablation": _spec(
        "appendix_ablation",
        legacy_names=("appendix",),
        visibility="public",
        family="ablation",
        runner_kind="stage_runner",
        claim_grade_allowed=True,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/appendix_ablation",
    ),
    "glance_oracle_refinement_internal": _spec(
        "glance_oracle_refinement_internal",
        legacy_names=("glance_oracle_refine",),
        visibility="internal",
        family="internal_refinement",
        runner_kind="stage_runner",
        claim_grade_allowed=False,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/glance_oracle_refinement_internal",
    ),
    "glance_full_graph_refinement_internal": _spec(
        "glance_full_graph_refinement_internal",
        legacy_names=("glance_full_graph_refine",),
        visibility="internal",
        family="internal_refinement",
        runner_kind="stage_runner",
        claim_grade_allowed=False,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/glance_full_graph_refinement_internal",
    ),
    "glance_counterfactual_router_internal": _spec(
        "glance_counterfactual_router_internal",
        legacy_names=("glance_counterfactual_router",),
        visibility="internal",
        family="internal_refinement",
        runner_kind="stage_runner",
        claim_grade_allowed=False,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/glance_counterfactual_router_internal",
    ),
    "glance_budgeted_refinement_internal": _spec(
        "glance_budgeted_refinement_internal",
        legacy_names=("glance_budgeted_refine",),
        visibility="internal",
        family="internal_refinement",
        runner_kind="stage_runner",
        claim_grade_allowed=False,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/glance_budgeted_refinement_internal",
    ),
    "glance_refiner_analysis_internal": _spec(
        "glance_refiner_analysis_internal",
        legacy_names=("glance_refiner_analysis",),
        visibility="internal",
        family="internal_analysis",
        runner_kind="stage_runner",
        claim_grade_allowed=False,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/glance_refiner_analysis_internal",
    ),
    "phase_a_single_cell_internal": _spec(
        "phase_a_single_cell_internal",
        legacy_names=("phase_a_single_cell_internal",),
        visibility="internal",
        family="internal_phase_a",
        runner_kind="phase_a_single_cell",
        claim_grade_allowed=False,
        requires_canonical_split=True,
        forces_use_gnn=True,
        graph_data_mode="required",
        artifact_namespace="stages/phase_a_single_cell_internal",
    ),
}

DEPRECATED_STAGE_VALUES = {"eqc_v8_matrix"}

_STAGE_LOOKUP: Dict[str, StageSpec] = {}
for _spec_item in STAGE_REGISTRY.values():
    _STAGE_LOOKUP[_spec_item.canonical_name] = _spec_item
    for _legacy_name in _spec_item.legacy_names:
        _STAGE_LOOKUP[_legacy_name] = _spec_item
for _deprecated_name in DEPRECATED_STAGE_VALUES:
    _STAGE_LOOKUP[_deprecated_name] = _spec(
        _deprecated_name,
        visibility="internal",
        family="deprecated",
        runner_kind="deprecated",
        claim_grade_allowed=False,
        requires_canonical_split=False,
        forces_use_gnn=False,
        graph_data_mode="optional",
        artifact_namespace=f"stages/{_deprecated_name}",
    )


def get_stage_spec(canonical_name: str) -> StageSpec:
    return STAGE_REGISTRY[canonical_name]


def resolve_stage_spec(name: str) -> StageSpec:
    key = str(name).strip()
    if key not in _STAGE_LOOKUP:
        raise KeyError(f"Unknown stage/task: {name}")
    return _STAGE_LOOKUP[key]


def resolve_stage_name(name: str) -> str:
    return resolve_stage_spec(name).canonical_name


def public_stage_specs() -> Tuple[StageSpec, ...]:
    return tuple(spec for spec in STAGE_REGISTRY.values() if spec.visibility == "public")


def public_canonical_stage_names() -> Tuple[str, ...]:
    return tuple(spec.canonical_name for spec in public_stage_specs())


def internal_stage_specs() -> Tuple[StageSpec, ...]:
    return tuple(spec for spec in STAGE_REGISTRY.values() if spec.visibility == "internal")


def accepted_stage_values(*, include_internal: bool, include_deprecated: bool) -> Tuple[str, ...]:
    accepted = []
    for spec in STAGE_REGISTRY.values():
        if spec.visibility == "internal" and not include_internal:
            continue
        accepted.append(spec.canonical_name)
        accepted.extend(spec.legacy_names)
    if include_deprecated:
        accepted.extend(sorted(DEPRECATED_STAGE_VALUES))
    seen = set()
    ordered = []
    for item in accepted:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def legacy_name_for(stage_name: str) -> Optional[str]:
    spec = resolve_stage_spec(stage_name)
    return spec.legacy_names[0] if spec.legacy_names else None


def legacy_stage_map() -> Dict[str, str]:
    mapping = {}
    for spec in STAGE_REGISTRY.values():
        for legacy_name in spec.legacy_names:
            mapping[legacy_name] = spec.canonical_name
    return mapping


def is_public_stage(stage_name: str) -> bool:
    return resolve_stage_spec(stage_name).visibility == "public"


def is_internal_stage(stage_name: str) -> bool:
    return resolve_stage_spec(stage_name).visibility == "internal"


def iter_stage_specs() -> Iterable[StageSpec]:
    return STAGE_REGISTRY.values()
