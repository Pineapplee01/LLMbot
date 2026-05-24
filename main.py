import sys

# Keep parser/help validation independent from the training runtime.
# `python main.py --help` should work even when torch and GPU deps are absent.
if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    from parser_args import parser_args as _parser_args

    _parser_args(["--help"])

import shutil
from pathlib import Path

import torch
from sklearn.metrics import f1_score

from model_building import _labels_to_index, _score_logits
from parser_args import parser_args
from stage_registry import get_stage_spec, resolve_stage_spec
from stage_helpers import MissingFrozenArtifactError, PHASE_A_CONTRACT, PHASE_A_DISABLED_COMPONENTS, _resolve_device
from stage_runner import StageRunner
from trainer import run_legacy_graph_seed
from trainer_preparation import build_or_load_faithful_gats, build_or_load_frozen_g0, load_frozen_g0
from trainer_semantic import run_semantic_finetune_seed
from utils import (
    build_experiment_root,
    build_stage_dir,
    load_raw_data,
    read_json,
    reset_split,
    safe_torch_load,
    seed_setting,
    setup_wandb,
    write_json,
    write_text,
    write_torch,
)


PHASE_A_INTERNAL_STAGE = "phase_a_single_cell_internal"
PHASE_A_CLOSED_CONTROLS = {
    "estimator_mode": "none",
    "semantic_mode": "off",
    "repair_mode": "noop",
    "selector_mode": "none",
    "appendix_mode": "none",
}


def _normalized_reset_split(reset_split):
    return str(reset_split).strip()


def _is_phase_a_module_controls(args):
    if str(getattr(args, "experiment_task", "distillation_pipeline")) != "distillation_pipeline":
        return False
    has_embedding_source = (
        getattr(args, "embedding_path", None)
        or str(getattr(args, "semantic_encoder", "auto")).lower() != "auto"
    )
    if not has_embedding_source:
        return False
    if not bool(getattr(args, "use_GNN", False)):
        return False
    for name, expected in PHASE_A_CLOSED_CONTROLS.items():
        if str(getattr(args, name, expected)).lower() != expected:
            return False
    return True


def _phase_a_semantic_source_name(args):
    semantic_encoder = str(getattr(args, "semantic_encoder", "auto")).lower()
    if semantic_encoder != "auto":
        return semantic_encoder
    return str(getattr(args, "text_encoder", "roberta")).lower()


def _phase_a_single_cell_id(args):
    return f"{_phase_a_semantic_source_name(args)}__{str(args.graph_backbone).lower()}"


def _find_latest_semantic_finetune_embedding(seed, semantic_encoder):
    candidates = []
    pattern = f"*/seed_{int(seed)}/stages/semantic_encoder_finetune/manifest.json"
    legacy_pattern = f"*/seed_{int(seed)}/stages/semantic_finetune/manifest.json"
    for current_pattern in (pattern, legacy_pattern):
        for manifest_path in Path(".").glob(current_pattern):
            manifest = read_json(manifest_path, default={}) or {}
            if manifest.get("status") != "completed":
                continue
            if str(manifest.get("semantic_backbone", manifest.get("semantic_encoder", ""))).lower() != semantic_encoder:
                continue
            embedding_path = Path(manifest.get("embeddings_path", manifest_path.parent / "embeddings.pt"))
            if embedding_path.exists():
                candidates.append(embedding_path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_semantic_backbone_embedding_path(args, data, experiment_root=None, seed=None):
    semantic_encoder = str(getattr(args, "semantic_encoder", "auto")).lower()
    dataset_path = Path(data.get("dataset_path", "."))
    if semantic_encoder == "qwen3_peft":
        if experiment_root is not None:
            for relative in ("stages/semantic_encoder_finetune/embeddings.pt", "stages/semantic_finetune/embeddings.pt"):
                candidate = Path(experiment_root) / relative
                if candidate.exists():
                    return candidate, "semantic_encoder_finetune_current_experiment"
        if seed is not None:
            candidate = _find_latest_semantic_finetune_embedding(seed, semantic_encoder)
            if candidate is not None:
                return candidate, "semantic_encoder_finetune_latest_matching_seed"
        raise MissingFrozenArtifactError(
            "Phase A with --semantic_encoder qwen3_peft requires embeddings from "
            "--experiment_task semantic_encoder_finetune. Run semantic_encoder_finetune first or pass --embedding_path."
        )

    if semantic_encoder == "qwen3_frozen":
        candidates = [dataset_path / "qwen3_emb_last.pt"]
        candidates.extend(sorted(dataset_path.glob("**/qwen3_emb_last.pt")))
        candidates.extend(sorted(dataset_path.glob("**/qwen3_emb_L-1.pt")))
    elif semantic_encoder in {"roberta", "roberta_finetuned"}:
        candidates = [dataset_path / "embeddings_roberta.pt"]
    else:
        candidates = []
    for candidate in candidates:
        if candidate.exists():
            return candidate, f"semantic_encoder_{semantic_encoder}"
    raise MissingFrozenArtifactError(
        f"Could not resolve embeddings for --semantic_encoder {semantic_encoder}. "
        "Pass --embedding_path explicitly."
    )


def _resolve_phase_a_embedding_path(args, data, experiment_root=None, seed=None):
    embedding_path = getattr(args, "embedding_path", None)
    if embedding_path:
        path = Path(embedding_path)
        resolver = "embedding_path"
    else:
        semantic_encoder = str(getattr(args, "semantic_encoder", "auto")).lower()
        if semantic_encoder == "auto":
            raise MissingFrozenArtifactError(
                "Phase A module-controls entry requires --embedding_path pointing to a cached "
                "semantic embedding tensor, or an explicit --semantic_encoder. "
                "Structural fallback and implicit LM cache lookup are disabled for Phase A."
            )
        path, resolver = _resolve_semantic_backbone_embedding_path(
            args,
            data,
            experiment_root=experiment_root,
            seed=seed,
        )

    if not path.exists():
        raise MissingFrozenArtifactError(
            "Phase A module-controls entry requires a cached semantic embedding tensor. "
            f"Resolved path does not exist: {path}. "
            "Pass --embedding_path /path/to/embedding.pt."
        )

    args.embedding_path = str(path)
    args.emb_path = str(path)
    args.g0_feature_path = str(path)
    return {
        "resolver": resolver,
        "LM_model": getattr(args, "text_encoder", "roberta"),
        "semantic_backbone": str(getattr(args, "semantic_encoder", "auto")).lower(),
        "semantic_source": _phase_a_semantic_source_name(args),
        "path": str(path),
        "formal_comparison_eligible": resolver == "embedding_path",
    }


def _maybe_force_phase_a_retrain(args, experiment_root, embedding_path):
    manifest = read_json(Path(experiment_root) / "frozen" / "g0" / "manifest.json", default=None)
    if not manifest:
        return
    feature_manifest = manifest.get("feature_manifest", {})
    existing_path = feature_manifest.get("path")
    existing_backbone = str(manifest.get("backbone", "")).lower()
    requested_backbone = str(getattr(args, "graph_backbone", "")).lower()
    if existing_path != str(embedding_path) or existing_backbone != requested_backbone:
        args.force_retrain_backbone = True


def _phase_a_idx_tensor(idx):
    return idx.detach().cpu().long() if torch.is_tensor(idx) else torch.tensor(idx, dtype=torch.long)


def _phase_a_primary_metrics(outputs, labels, idx):
    metrics = _score_logits(outputs["logits"], labels, idx)
    idx = _phase_a_idx_tensor(idx)
    if idx.numel() == 0:
        metrics["bot_f1"] = 0.0
        return metrics
    pred = outputs["logits"][idx].argmax(dim=1).detach().cpu().numpy()
    y = labels[idx].detach().cpu().numpy()
    metrics["bot_f1"] = float(f1_score(y, pred, average="binary", pos_label=1, zero_division=0))
    return metrics


def _phase_a_supporting_metrics(outputs, labels, idx):
    idx = _phase_a_idx_tensor(idx)
    if idx.numel() == 0 or "prob" not in outputs:
        return {"ECE": 0.0, "Brier": 0.0, "HCW_rate": 0.0}

    prob = outputs["prob"][idx].detach().cpu().float()
    y = labels[idx].detach().cpu().long()
    pred = prob.argmax(dim=1)
    confidence = prob.max(dim=1).values
    correct = pred.eq(y).float()
    wrong = 1.0 - correct

    ece = 0.0
    bins = torch.linspace(0.0, 1.0, 11)
    for bin_idx in range(10):
        left = bins[bin_idx]
        right = bins[bin_idx + 1]
        if bin_idx == 9:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if mask.any():
            ece += float(mask.float().mean().item() * abs(confidence[mask].mean().item() - correct[mask].mean().item()))

    one_hot = torch.nn.functional.one_hot(y, num_classes=prob.shape[1]).float()
    brier = torch.mean(torch.sum((prob - one_hot) ** 2, dim=1)).item()
    hcw_rate = (((confidence >= 0.9).float() * wrong).sum() / max(int(idx.numel()), 1)).item()
    return {"ECE": float(ece), "Brier": float(brier), "HCW_rate": float(hcw_rate)}


def _count_state_dict_parameters(state_dict):
    if not isinstance(state_dict, dict):
        return 0
    return int(sum(value.numel() for value in state_dict.values() if torch.is_tensor(value)))


def _phase_a_trainable_parameter_count(g0_context):
    checkpoint_path = Path(g0_context.get("checkpoint_path", ""))
    if not checkpoint_path.exists():
        return None
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    total = _count_state_dict_parameters(checkpoint.get("model", {}))
    adapter = checkpoint.get("input_adapter", {})
    total += _count_state_dict_parameters(adapter.get("state_dict"))
    return total


def _phase_a_cost_report(embedding_manifest, g0_context):
    embedding_path = Path(embedding_manifest["path"])
    embedding_cache_size = embedding_path.stat().st_size if embedding_path.exists() else None
    feature_manifest = g0_context.get("manifest", {}).get("feature_manifest", {})
    peft_manifest = feature_manifest.get("peft", {})
    return {
        "peak_gpu_memory": None,
        "wall_clock_training_time": None,
        "inference_latency_per_node": None,
        "trainable_parameter_count": _phase_a_trainable_parameter_count(g0_context),
        "peft_trainable_parameter_count": peft_manifest.get("trainable_parameter_count", 0),
        "embedding_cache_size": embedding_cache_size,
        "precompute_time": feature_manifest.get("projection_time_seconds"),
    }


def _write_phase_a_cell_artifacts(stage_dir, cell_id, outputs, checkpoint_path, metrics, cost_report, subgroup_report, manifest):
    cell_dir = Path(stage_dir) / "cells" / cell_id
    write_json(cell_dir / "metrics.json", metrics)
    write_json(cell_dir / "cost_report.json", cost_report)
    write_json(cell_dir / "subgroup_report.json", subgroup_report)
    write_json(cell_dir / "manifest.json", manifest)
    write_torch(cell_dir / "outputs.pt", outputs)
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        cell_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint_path, cell_dir / "checkpoint.pt")
    else:
        write_torch(
            cell_dir / "checkpoint.pt",
            {"source_checkpoint_path": str(checkpoint_path), "status": "missing_at_phase_a_write_time"},
        )
    return cell_dir


def _write_phase_a_single_cell_stage(args, seed, data, experiment_root, g0_context, embedding_manifest):
    stage_dir = build_stage_dir(experiment_root, PHASE_A_INTERNAL_STAGE)
    outputs = g0_context["outputs"]
    labels = _labels_to_index(data["labels"])
    validation_metrics = _phase_a_primary_metrics(outputs, labels, data["valid_idx"])
    test_metrics = _phase_a_primary_metrics(outputs, labels, data["test_idx"])
    cell_id = _phase_a_single_cell_id(args)
    g0_manifest = g0_context["manifest"]
    user_entry = getattr(args, "_phase_a_user_entry", "module_controls_phase_a")
    cost_report = _phase_a_cost_report(embedding_manifest, g0_context)
    subgroup_report = {
        "scope": "analysis_only",
        "status": "schema_reserved",
        "subgroups": {},
        "supporting_metrics": _phase_a_supporting_metrics(outputs, labels, data["test_idx"]),
    }

    contract = {
        "contract": PHASE_A_CONTRACT,
        "stage_label": "Phase A: semantic encoder -> node embedding -> GNN detector",
        "stage": PHASE_A_INTERNAL_STAGE,
        "stage_role": "internal_artifact_namespace",
        "user_entry": user_entry,
        "mode": "module_controls_phase_a",
        "data_flow": "raw_account_text -> cached_semantic_encoder_embedding -> node_embedding -> gnn_detector",
        "disabled_components": PHASE_A_DISABLED_COMPONENTS,
        "control_variables": {
            "split": "canonical dataset split; --reset_split -1 required by main.py",
            "semantic_source": "--semantic_encoder plus resolved semantic embedding tensor",
            "input_projection": "all semantic sources projected to --phase_a_project_dim before GNN input",
            "gnn_backbone": "--graph_backbone",
            "module_controls": PHASE_A_CLOSED_CONTROLS,
            "selection_scope": "validation macro-F1, validation loss tie-breaker",
        },
        "core_matrix": {
            "semantic_sources": ["roberta_finetuned", "qwen3_frozen", "qwen3_peft"],
            "gnn_backbones": ["rgcn", "rgt"],
            "project_dim": int(getattr(args, "phase_a_project_dim", 768)),
            "projector": getattr(args, "phase_a_projector", "pca"),
            "cell_execution": "one text_encoder x graph_backbone cell per --embedding_path command",
            "gated_cell": "qwen3_peft__rgt",
            "gate_rule": (
                "Run qwen3_peft x rgt only if qwen3_peft x rgcn improves validation macro_f1 "
                "over qwen3_frozen x rgcn by at least 0.5pp, or closes at least 25% of the "
                "validation macro_f1 gap to roberta_finetuned x rgcn."
            ),
        },
    }
    metrics = {
        "cell_id": cell_id,
        "validation": validation_metrics,
        "test": test_metrics,
        "selection": g0_context.get("selection_metrics", {}),
        "primary_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "bot_f1": test_metrics["bot_f1"],
        },
        "supporting_metrics": subgroup_report["supporting_metrics"],
    }
    cell_manifest = {
        "contract": PHASE_A_CONTRACT,
        "cell_id": cell_id,
        "seed": int(seed),
        "semantic_source": _phase_a_semantic_source_name(args),
        "gnn_backbone": getattr(args, "graph_backbone", "rgcn"),
        "embedding_manifest": embedding_manifest,
        "feature_manifest": g0_manifest.get("feature_manifest", {}),
        "disabled_components": PHASE_A_DISABLED_COMPONENTS,
        "user_entry": user_entry,
        "frozen_g0_dir": str(g0_context["dir"]),
        "frozen_g0_checkpoint": g0_context["checkpoint_path"],
    }
    cell_dir = _write_phase_a_cell_artifacts(
        stage_dir,
        cell_id,
        outputs,
        g0_context["checkpoint_path"],
        metrics,
        cost_report,
        subgroup_report,
        cell_manifest,
    )
    summary = {
        **contract,
        "seed": int(seed),
        "cell_id": cell_id,
        "LM_model": getattr(args, "text_encoder", "roberta"),
        "semantic_backbone": getattr(args, "semantic_encoder", "auto"),
        "GNN_model": getattr(args, "graph_backbone", "rgcn"),
        "embedding_manifest": embedding_manifest,
        "frozen_g0_dir": str(g0_context["dir"]),
        "frozen_g0_checkpoint": g0_context["checkpoint_path"],
        "frozen_g0_manifest": g0_manifest,
        "cell_dir": str(cell_dir),
        "cost_report": cost_report,
        "subgroup_report": subgroup_report,
        "metrics": metrics,
    }
    survivor = {
        "contract": PHASE_A_CONTRACT,
        "best_cell": cell_id,
        "best_semantic_source": _phase_a_semantic_source_name(args),
        "best_gnn_backbone": getattr(args, "graph_backbone", "rgcn"),
        "selection_metric": "validation_macro_f1",
        "tie_breaker": "validation_loss",
        "mode": "module_controls_phase_a",
        "user_entry": user_entry,
    }

    write_json(stage_dir / "phase_a_contract.json", contract)
    write_json(stage_dir / "summary.json", summary)
    write_json(stage_dir / "metrics.json", metrics)
    write_json(stage_dir / "cost_report.json", cost_report)
    write_json(stage_dir / "subgroup_report.json", subgroup_report)
    write_json(stage_dir / "survivor_manifest.json", survivor)
    write_text(
        stage_dir / "notes.md",
        "\n".join(
            [
                "# Phase A single-cell smoke",
                "",
                "Entry: main.py module controls with --embedding_path and graph execution enabled.",
                "Semantic embedding is resolved from --embedding_path or explicit --semantic_encoder.",
                "GNN detector is selected by --graph_backbone.",
                "Post-hoc estimator, local semantic enhancement, repair, selector, calibration, same-head refinement, and appendix lanes are disabled.",
            ]
        )
        + "\n",
    )
    return {"stage": PHASE_A_INTERNAL_STAGE, "cell_id": cell_id, "stage_dir": str(stage_dir)}


def _parse_seed_list(seeds):
    return [int(seed) for seed in str(seeds).strip().split(",")]


def _stage_result(stage, seed, artifact_dir, **extra):
    result = {
        "stage": stage,
        "seed": int(seed),
        "status": "completed",
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
    }
    result.update(extra)
    return result


def _format_stage_result(result):
    if isinstance(result, dict):
        stage = result.get("stage", "unknown")
        seed = result.get("seed", "unknown")
        status = result.get("status", "completed")
        return f"Completed stage {stage} for seed {seed}: {status}"
    return str(result)


def _requires_canonical_split(stage):
    return bool(resolve_stage_spec(stage).requires_canonical_split)


def _validate_claim_grade_run(args, execution_stage, reset_split_value):
    if not getattr(args, "claim_grade", False):
        return
    if reset_split_value != "-1":
        raise ValueError("Claim-grade runs require canonical splits; pass --reset_split -1.")
    if execution_stage == "distillation_pipeline":
        raise ValueError(
            "distillation_pipeline is compatibility-only for claim-grade runs; use a formal staged entrypoint."
        )


def _stage_uses_graph_data(args, execution_stage):
    if execution_stage == "distillation_pipeline":
        return args.use_GNN
    return resolve_stage_spec(execution_stage).graph_data_mode != "none"


def _load_seed_data(args, execution_stage):
    data = load_raw_data(args.dataset, use_GNN=_stage_uses_graph_data(args, execution_stage))
    if args.reset_split != "-1":
        train_idx, valid_idx, test_idx = reset_split(len(data["user_text"]), args.reset_split)
        data["train_idx"], data["valid_idx"], data["test_idx"] = train_idx, valid_idx, test_idx
    return data


def _run_seed_stage(args, seed, data, run, execution_stage, experiment_root):
    if execution_stage == "semantic_encoder_finetune":
        result = run_semantic_finetune_seed(args, seed, data, experiment_root, run)
        return _stage_result(
            "semantic_encoder_finetune",
            seed,
            result.get("artifact_dir") or result.get("stage_dir"),
            result=result,
        )

    if execution_stage == "graph_detector_prepare":
        result = build_or_load_frozen_g0(args, seed, data, experiment_root)
        return _stage_result("graph_detector_prepare", seed, result.get("artifact_dir"), result=result)

    if execution_stage == "graph_calibration_prepare":
        g0_context = load_frozen_g0(experiment_root)
        result = build_or_load_faithful_gats(args, seed, data, experiment_root, g0_context)
        return _stage_result("graph_calibration_prepare", seed, result.get("artifact_dir"), result=result)

    if execution_stage == PHASE_A_INTERNAL_STAGE:
        embedding_manifest = _resolve_phase_a_embedding_path(args, data, experiment_root=experiment_root, seed=seed)
        _maybe_force_phase_a_retrain(args, experiment_root, embedding_manifest["path"])
        g0_context = build_or_load_frozen_g0(args, seed, data, experiment_root)
        result = _write_phase_a_single_cell_stage(args, seed, data, experiment_root, g0_context, embedding_manifest)
        return _stage_result(execution_stage, seed, result.get("stage_dir"), result=result)

    if execution_stage == "distillation_pipeline":
        run_legacy_graph_seed(args, seed, data, run)
        return _stage_result("distillation_pipeline", seed, experiment_root)

    runner = StageRunner(args=args, seed=seed, data=data, run=run)
    result = runner.run()
    return _stage_result(
        execution_stage,
        seed,
        result.get("artifact_dir") or result.get("stage_dir") if isinstance(result, dict) else None,
        result=result,
    )


def main(args):
    args.device = _resolve_device(args.device)
    requested_stage = args.experiment_task
    requested_spec = get_stage_spec(requested_stage)
    phase_a_by_controls = _is_phase_a_module_controls(args)
    execution_stage = PHASE_A_INTERNAL_STAGE if phase_a_by_controls else requested_spec.canonical_name
    args._phase_a_user_entry = "module_controls_phase_a" if phase_a_by_controls else "stage_compatibility"
    reset_split_value = _normalized_reset_split(args.reset_split)

    _validate_claim_grade_run(args, execution_stage, reset_split_value)

    if _requires_canonical_split(execution_stage) and reset_split_value != "-1":
        raise ValueError(
            f"Stage '{execution_stage}' requires canonical splits; pass --reset_split -1."
        )

    args.requested_stage = requested_stage
    args.execution_task = execution_stage
    args.stage = execution_stage
    args.reset_split = reset_split_value
    if execution_stage not in {"distillation_pipeline", "semantic_encoder_finetune"}:
        args.use_GNN = True

    for seed in _parse_seed_list(args.seeds):
        seed_setting(seed)
        data = _load_seed_data(args, execution_stage)
        run = setup_wandb(args, seed)
        experiment_root = build_experiment_root(args, seed)
        print(_format_stage_result(_run_seed_stage(args, seed, data, run, execution_stage, experiment_root)))
        run.finish()


if __name__ == "__main__":
    main(parser_args())
