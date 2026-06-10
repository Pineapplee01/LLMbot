import argparse
import sys

from stage_registry import (
    DEPRECATED_STAGE_VALUES,
    accepted_stage_values,
    legacy_name_for,
    public_canonical_stage_names,
    resolve_stage_spec,
)


CANONICAL_STAGE_CHOICES = public_canonical_stage_names()
LEGACY_STAGE_CHOICES = accepted_stage_values(include_internal=False, include_deprecated=True)

RENAMED_FLAGS = {
    "--stage": "--experiment_task",
    "--GNN_model": "--graph_backbone",
    "--LM_model": "--text_encoder",
    "--semantic_backbone": "--semantic_encoder",
    "--batch_size_LM": "--lm_batch_size",
    "--batch_size_GNN": "--gnn_batch_size",
    "--optimizer_LM": "--lm_optimizer",
    "--optimizer_GNN": "--gnn_optimizer",
    "--LM_dropout": "--lm_dropout",
    "--LM_att_dropout": "--lm_attention_dropout",
    "--GNN_dropout": "--gnn_dropout",
    "--lr_LM": "--lm_learning_rate",
    "--lr_GNN": "--gnn_learning_rate",
    "--weight_decay_LM": "--lm_weight_decay",
    "--weight_decay_GNN": "--gnn_weight_decay",
    "--emb_path": "--embedding_path",
    "--g0_feature_path": "--embedding_path",
    "--g0_epochs": "--graph_detector_epochs",
    "--gats_max_iter": "--gate_calibrator_max_iter",
}

DEPRECATED_UNWIRED_FLAGS = {
    "--batch_size_MLP",
    "--MLP_n_layers",
    "--MLP_hidden_dim",
    "--optimizer_MLP",
    "--MLP_dropout",
    "--MLP_KD_epochs",
    "--MLP_epochs_per_iter",
    "--pl_ratio_MLP",
    "--lr_MLP",
    "--weight_decay_MLP",
    "--is_processed",
    "--gamma",
    "--selector_budget",
    "--gain_clip_value",
    "--phase_a_semantic_sources",
    "--phase_a_feature_paths",
    "--phase_a_gnn_family",
    "--phase_a_fixed_gnn",
    "--phase_a_fixed_semantic_source",
    "--phase_a_latency_repeats",
    "--phase_a_include_structural_smoke",
    "--semantic_mode",
    "--repair_mode",
    "--selector_mode",
    "--appendix_mode",
}


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _extract_cli_flags(raw_args):
    if raw_args is None:
        return []
    flags = []
    for token in raw_args:
        if token == "--":
            break
        if token.startswith("--"):
            flags.append(token.split("=", 1)[0])
    return flags


def _dedupe_preserve_order(values):
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _deprecated_cli_flags(args, raw_args):
    deprecated = []
    for flag in _extract_cli_flags(raw_args):
        if flag in RENAMED_FLAGS or flag in DEPRECATED_UNWIRED_FLAGS:
            deprecated.append(flag)
    if getattr(args, "legacy_task_name_used", None):
        deprecated.append(f"stage:{args.legacy_task_name_used}")
    if getattr(args, "requested_experiment_task", None) in DEPRECATED_STAGE_VALUES:
        deprecated.append(f"stage:{args.requested_experiment_task}")
    return _dedupe_preserve_order(deprecated)


def normalize_args(args, raw_args=None):
    raw_flags = set(_extract_cli_flags(raw_args or []))
    args.explicit_cli_flags = sorted(raw_flags)
    requested_task = getattr(args, "experiment_task", None)
    if requested_task is None:
        requested_task = getattr(args, "stage", None)
    requested_task = str(requested_task).strip()
    resolved_spec = resolve_stage_spec(requested_task)

    args.requested_experiment_task = requested_task
    args.experiment_task = resolved_spec.canonical_name
    args.stage = resolved_spec.canonical_name
    args.requested_stage = requested_task
    args.stage_spec = resolved_spec
    args.stage_visibility = resolved_spec.visibility
    args.legacy_stage = legacy_name_for(args.experiment_task)
    args.legacy_task_name_used = requested_task if requested_task in resolved_spec.legacy_names else None

    args.graph_backbone = getattr(args, "graph_backbone", None)
    args.GNN_model = args.graph_backbone

    second_view_scope = str(getattr(args, "graph_second_view_scope", "auto") or "auto").strip().lower()
    scope_to_refine_mode = {
        "none": "none",
        "labeled_prefix": "routed_dynamic_hyperscan_branch",
        "routed_nodes": "routed_dynamic_hyperscan_branch",
        "neighborloader_batch": "hyperscan_neighborloader_batch_local_branch",
    }
    if second_view_scope != "auto":
        if second_view_scope == "routed_nodes" and not str(getattr(args, "routed_nodes_path", "") or "").strip():
            raise ValueError("--graph_second_view_scope routed_nodes requires --routed_nodes_path.")
        if second_view_scope == "labeled_prefix" and str(getattr(args, "routed_nodes_path", "") or "").strip():
            raise ValueError(
                "--graph_second_view_scope labeled_prefix conflicts with --routed_nodes_path; "
                "use --graph_second_view_scope routed_nodes for routed centers."
            )
        target_mode = scope_to_refine_mode[second_view_scope]
        existing_mode = str(getattr(args, "graph_refine_mode", "none") or "none").strip().lower()
        if "--graph_refine_mode" in raw_flags and existing_mode != target_mode:
            raise ValueError(
                "--graph_second_view_scope conflicts with explicit --graph_refine_mode: "
                f"{second_view_scope} maps to {target_mode}, got {existing_mode}."
            )
        args.graph_refine_mode = target_mode

    second_view_candidate_scope = str(
        getattr(args, "graph_second_view_candidate_scope", "auto") or "auto"
    ).strip().lower()
    if second_view_candidate_scope != "auto":
        existing_candidate_scope = str(
            getattr(args, "graph_refine_candidate_scope", "undirected_relation_1hop")
            or "undirected_relation_1hop"
        ).strip().lower()
        if "--graph_refine_candidate_scope" in raw_flags and existing_candidate_scope != second_view_candidate_scope:
            raise ValueError(
                "--graph_second_view_candidate_scope conflicts with explicit --graph_refine_candidate_scope: "
                f"{second_view_candidate_scope} vs {existing_candidate_scope}."
            )
        args.graph_refine_candidate_scope = second_view_candidate_scope

    second_view_geometry = str(
        getattr(args, "graph_second_view_training_geometry", "auto") or "auto"
    ).strip().lower()
    if second_view_geometry != "auto":
        existing_geometry = str(getattr(args, "graph_training_loader_mode", "full_batch") or "full_batch").strip().lower()
        if "--graph_training_loader_mode" in raw_flags and existing_geometry != second_view_geometry:
            raise ValueError(
                "--graph_second_view_training_geometry conflicts with explicit --graph_training_loader_mode: "
                f"{second_view_geometry} vs {existing_geometry}."
            )
        args.graph_training_loader_mode = second_view_geometry
    elif second_view_scope == "neighborloader_batch":
        existing_geometry = str(getattr(args, "graph_training_loader_mode", "full_batch") or "full_batch").strip().lower()
        if "--graph_training_loader_mode" in raw_flags and existing_geometry != "neighbor_subgraph":
            raise ValueError(
                "--graph_second_view_scope neighborloader_batch requires "
                "--graph_training_loader_mode neighbor_subgraph when the legacy geometry flag is explicit."
            )
        args.graph_training_loader_mode = "neighbor_subgraph"

    legacy_dhg_backbones = {"rgcn_hyperscan_dhg", "rgcn_hyperscan_dhg_nodeinput"}
    legacy_pyg_backbones = {"rgcn_hyperscan_routed", "rgcn_hyperscan_nodeinput"}
    if "--graph_second_view_hypergraph_backend" not in raw_flags:
        if str(args.graph_backbone).lower() in legacy_dhg_backbones:
            args.graph_second_view_hypergraph_backend = "dhg"
        elif str(args.graph_backbone).lower() in legacy_pyg_backbones:
            args.graph_second_view_hypergraph_backend = "pyg"

    second_view_fusion = str(getattr(args, "graph_second_view_fusion", "residual") or "residual").strip().lower()
    if "--hyperscan_detector_style" in raw_flags:
        legacy_style = str(getattr(args, "hyperscan_detector_style", "residual") or "residual").strip().lower()
        legacy_fusion = "multiattn" if legacy_style in {"original_cross_attention", "multiattn"} else legacy_style
        if "--graph_second_view_fusion" in raw_flags and legacy_fusion != second_view_fusion:
            raise ValueError(
                "--graph_second_view_fusion conflicts with explicit --hyperscan_detector_style: "
                f"{second_view_fusion} vs {legacy_style}."
            )
        second_view_fusion = legacy_fusion
    args.graph_second_view_fusion = second_view_fusion
    args.hyperscan_detector_style = "original_cross_attention" if second_view_fusion == "multiattn" else "residual"

    args.conformal_knn_config_source = "explicit_cli_or_router_defaults"
    args.conformal_knn_inherited_from_second_view = False
    if str(getattr(args, "estimator_mode", "none") or "none").strip().lower() == "conformal_knn_risk_router":
        inherited_fields = []
        graph_mode = str(getattr(args, "graph_refine_mode", "none") or "none").strip().lower()
        hyper_knn_modes = {
            "hyperscan_knn_hypergraph_proxy_augment",
            "routed_dynamic_hyperscan_branch",
            "hyperscan_neighborloader_batch_local_branch",
        }
        if "--conformal_knn_k" not in raw_flags and graph_mode in hyper_knn_modes:
            args.conformal_knn_k = int(getattr(args, "graph_refine_knn_k", getattr(args, "conformal_knn_k", 8)) or 8)
            inherited_fields.append("k")
        if "--conformal_knn_repr_source" not in raw_flags and graph_mode in {
            "routed_dynamic_hyperscan_branch",
            "hyperscan_neighborloader_batch_local_branch",
        }:
            args.conformal_knn_repr_source = "x_new"
            inherited_fields.append("repr_source")
        if "--conformal_knn_candidate_scope" not in raw_flags:
            graph_candidate = str(
                getattr(args, "graph_refine_candidate_scope", "undirected_relation_1hop")
                or "undirected_relation_1hop"
            ).strip().lower()
            if graph_mode == "routed_dynamic_hyperscan_branch":
                args.conformal_knn_candidate_scope = graph_candidate
                inherited_fields.append("candidate_scope")
            elif graph_mode == "hyperscan_knn_hypergraph_proxy_augment":
                args.conformal_knn_candidate_scope = "hyperscan_full"
                inherited_fields.append("candidate_scope")
            elif graph_mode == "hyperscan_neighborloader_batch_local_branch":
                args.conformal_knn_candidate_scope = "labeled_full"
                inherited_fields.append("candidate_scope")
        if inherited_fields:
            args.conformal_knn_inherited_from_second_view = True
            args.conformal_knn_config_source = "parser_graph_second_view_defaults:" + ",".join(inherited_fields)

    args.text_encoder = getattr(args, "text_encoder", None)
    args.LM_model = args.text_encoder
    args.semantic_encoder = getattr(args, "semantic_encoder", None)
    args.semantic_backbone = args.semantic_encoder
    args.embedding_path = getattr(args, "embedding_path", None)
    args.emb_path = args.embedding_path
    args.g0_feature_path = args.embedding_path
    args.graph_data_variant = getattr(args, "graph_data_variant", "labeled")
    args.support_embedding_path = getattr(args, "support_embedding_path", "support_roberta_embeddings_new.pt")

    if getattr(args, "risk_budgets", None) is None and getattr(args, "router_budgets", None) is not None:
        args.risk_budgets = args.router_budgets
    if getattr(args, "router_budgets", None) is None and getattr(args, "risk_budgets", None) is not None:
        args.router_budgets = args.risk_budgets

    args.reset_split = str(args.reset_split).strip()
    args.deprecated_cli_flags = _deprecated_cli_flags(args, raw_args)
    return args


def parser_args(argv=None):
    raw_args = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_task",
        dest="experiment_task",
        type=str,
        default="distillation_pipeline",
        choices=CANONICAL_STAGE_CHOICES,
        help="Canonical experiment task. Use legacy --stage only for historical commands.",
    )
    parser.add_argument(
        "--stage",
        dest="experiment_task",
        type=str,
        default=argparse.SUPPRESS,
        choices=LEGACY_STAGE_CHOICES,
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--project_name", type=str, default="llmbot-mainline")
    parser.add_argument("--experiment_name", type=str, default="llmbot_run")
    parser.add_argument("--artifact_root", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--reuse_existing_artifacts", action="store_true")
    parser.add_argument("--force_retrain_backbone", action="store_true")
    parser.add_argument(
        "--claim_grade",
        action="store_true",
        help="Require claim-grade comparability gates before launching a run.",
    )

    parser.add_argument("--dataset", type=str, default="TwiBot-20", help="Dataset name")
    parser.add_argument(
        "--graph_data_variant",
        type=str,
        default="labeled",
        choices=["labeled", "full_graph_support"],
        help="Graph-data contract: original labeled-only graph or full graph with support-node neighbors.",
    )
    parser.add_argument(
        "--support_embedding_path",
        type=str,
        default="support_roberta_embeddings_new.pt",
        help="Support-node semantic embedding tensor used by full_graph_support runtime concatenation.",
    )
    parser.add_argument("--lm_batch_size", dest="batch_size_LM", type=int, default=32)
    parser.add_argument("--batch_size_LM", dest="batch_size_LM", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--gnn_batch_size", dest="batch_size_GNN", type=int, default=300000)
    parser.add_argument("--batch_size_GNN", dest="batch_size_GNN", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--batch_size_MLP", type=int, default=300000, help=argparse.SUPPRESS)
    parser.add_argument("--raw_data_filepath", type=str, default="./data/raw/")
    parser.add_argument("--is_processed", type=_parse_bool, default=True, help=argparse.SUPPRESS)
    parser.add_argument("--reset_split", type=str, default="-1")
    parser.add_argument(
        "--routed_nodes_path",
        type=str,
        default=None,
        help=(
            "Optional routed-nodes json used by semantic stages and routed-node experiments. "
            "When set for semantic_encoder_finetune or semantic_embedding_classifier, train/validation/test "
            "indices are replaced by the routed train/valid/test subsets stored in that file."
        ),
    )
    parser.add_argument(
        "--routed_nodes_split",
        type=str,
        default="all",
        choices=["all", "train", "valid", "val", "test"],
        help=(
            "Optional split selector for --routed_nodes_path. "
            "Graph-refine and routed-node stages default to the routed union via all."
        ),
    )
    parser.add_argument(
        "--semantic_text_source_path",
        type=str,
        default=None,
        help=(
            "Optional prompt/text sidecar for semantic_encoder_finetune and semantic_embedding_classifier. "
            "When set, rows must contain node_id and a text field such as prompt/user/text; those texts replace "
            "norm_user_text for the listed node ids while preserving full-graph row alignment."
        ),
    )
    parser.add_argument(
        "--semantic_text_field",
        type=str,
        default="prompt",
        help="Field to read from --semantic_text_source_path rows; falls back to prompt/user/text when missing.",
    )
    parser.add_argument(
        "--semantic_gate_base_outputs_path",
        type=str,
        default=None,
        help="Base detector outputs.pt for semantic_correction_gate; expected keys include base_pred/base_prob or pred/prob.",
    )
    parser.add_argument(
        "--semantic_gate_candidate_output_paths",
        type=str,
        default=None,
        help="Comma-separated candidate outputs.pt paths for semantic_correction_gate, e.g. PEFT predictor and embedding-MLP outputs.",
    )
    parser.add_argument(
        "--semantic_gate_candidate_names",
        type=str,
        default=None,
        help="Comma-separated candidate names matching --semantic_gate_candidate_output_paths.",
    )
    parser.add_argument(
        "--semantic_gate_epochs",
        type=int,
        default=200,
        help="Training epochs for the lightweight semantic_correction_gate MLP.",
    )
    parser.add_argument(
        "--semantic_gate_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for the semantic_correction_gate MLP.",
    )
    parser.add_argument(
        "--semantic_gate_learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for semantic_correction_gate.",
    )
    parser.add_argument(
        "--semantic_gate_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for semantic_correction_gate.",
    )
    parser.add_argument(
        "--semantic_gate_break_weight",
        type=float,
        default=2.0,
        help="Sample weight for base-correct/candidate-wrong negative utility examples in semantic_correction_gate.",
    )
    parser.add_argument(
        "--semantic_gate_threshold_policy",
        type=str,
        default="global",
        choices=["global"],
        help="Validation-locked threshold policy for semantic_correction_gate.",
    )
    parser.add_argument(
        "--semantic_gate_selection_policy",
        type=str,
        default="threshold",
        choices=["threshold", "defer_softmax"],
        help=(
            "Selection policy for semantic_correction_gate. 'threshold' preserves the legacy independent "
            "candidate BCE gates; 'defer_softmax' trains one action-level keep-base-vs-candidate softmax gate."
        ),
    )
    parser.add_argument(
        "--semantic_gate_safety_policy",
        type=str,
        default="none",
        choices=["none", "break_first"],
        help=(
            "Optional safety policy for --semantic_gate_selection_policy defer_softmax. "
            "'break_first' trains a candidate break-risk head and applies a validation-locked break threshold."
        ),
    )
    parser.add_argument(
        "--semantic_gate_break_budget",
        type=float,
        default=-1.0,
        help=(
            "Optional maximum validation correct-node break rate for defer_softmax policy selection. "
            "Negative disables the budget and picks the best validation net."
        ),
    )
    parser.add_argument(
        "--semantic_gate_feature_family",
        type=str,
        default="probability",
        choices=["probability", "node_attribute", "local_competence"],
        help=(
            "Feature family for semantic_correction_gate. "
            "'probability' preserves the original base/candidate probability meta-features; "
            "'node_attribute' appends target-account metadata, tweet-behavior, and local graph attribute features; "
            "'local_competence' additionally appends train-routed nearest-neighbor action competence estimates."
        ),
    )
    parser.add_argument(
        "--semantic_gate_local_k",
        type=int,
        default=25,
        help=(
            "Top-k routed train neighbors used by --semantic_gate_feature_family local_competence "
            "to estimate per-candidate local fix/break competence."
        ),
    )

    parser.add_argument(
        "--graph_backbone",
        dest="graph_backbone",
        type=str,
        default="botrgcn",
        choices=[
            "botrgcn",
            "rgcn",
            "rgcn_hyperscan",
            "rgcn_hyperscan_routed",
            "rgcn_hyperscan_dhg",
            "rgcn_hyperscan_nodeinput",
            "rgcn_hyperscan_dhg_nodeinput",
            "rgt",
            "hgt",
            "simplehgn",
            "gatv2",
        ],
        help="Canonical graph detector backbone.",
    )
    parser.add_argument(
        "--GNN_model",
        dest="graph_backbone",
        type=str,
        default=argparse.SUPPRESS,
        choices=[
            "botrgcn",
            "rgcn",
            "rgcn_hyperscan",
            "rgcn_hyperscan_routed",
            "rgcn_hyperscan_dhg",
            "rgcn_hyperscan_nodeinput",
            "rgcn_hyperscan_dhg_nodeinput",
            "rgt",
            "hgt",
            "simplehgn",
            "gatv2",
        ],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--estimator_mode",
        type=str,
        default="none",
        choices=[
            "none",
            "msp_ts",
            "posthoc_calibrated_ranker",
            "calibrated_local_risk_router",
            "conformal_knn_risk_router",
            "login_uncertainty_router",
            "graph_conformal_set_estimator",
            "gnn_2hop_conformal",
            "glance_for_context_residual_risk_selector",
        ],
    )
    parser.add_argument(
        "--semantic_mode",
        type=str,
        default="off",
        choices=["off", "ridge_local", "lagnn_local"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--repair_mode",
        type=str,
        default="noop",
        choices=["noop", "prune", "disagreement_local", "ego_refinement"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--selector_mode",
        type=str,
        default="none",
        choices=["none", "rule", "risk_only", "risk_regime", "gain_margin", "binary", "three_action", "swap"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--appendix_mode",
        type=str,
        default="none",
        choices=["none", "qwen_frozen", "qwen_peft", "ib_edl", "glance_boundary", "cs"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--joint_router_family",
        type=str,
        default="reliability_mlp",
        choices=["reliability_mlp", "selectivenet"],
        help="Router family for joint_router_refinement: current reliability MLP or a SelectiveNet-style selective predictor.",
    )
    parser.add_argument(
        "--joint_selectivenet_coverages",
        type=str,
        default="0.10,0.20,0.30,0.40",
        help="Comma-separated coverage targets for SelectiveNet-style router experiments.",
    )
    parser.add_argument(
        "--joint_selectivenet_alpha",
        type=float,
        default=0.5,
        help="SelectiveNet-style loss mixture weight on the selective branch; (1-alpha) weights the auxiliary prediction head.",
    )
    parser.add_argument(
        "--joint_selectivenet_lambda",
        type=float,
        default=32.0,
        help="SelectiveNet-style coverage penalty weight.",
    )

    parser.add_argument("--text_encoder", dest="text_encoder", type=str, default="roberta")
    parser.add_argument("--LM_model", dest="text_encoder", type=str, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--semantic_encoder",
        dest="semantic_encoder",
        type=str,
        default="auto",
        choices=["auto", "roberta", "roberta_finetuned", "qwen3_frozen", "qwen3_peft"],
        help="Semantic encoder source for semantic finetune and Phase A embedding resolution.",
    )
    parser.add_argument(
        "--semantic_backbone",
        dest="semantic_encoder",
        type=str,
        default=argparse.SUPPRESS,
        choices=["auto", "roberta", "roberta_finetuned", "qwen3_frozen", "qwen3_peft"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--qwen_model_path",
        type=str,
        default="/root/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        help=(
            "Local path or HF id used by --semantic_encoder qwen3_peft. "
            "The plumbing name stays qwen3_peft, but the actual model can be a local Qwen2.5-Instruct snapshot."
        ),
    )
    parser.add_argument(
        "--finetuned_roberta_checkpoint_path",
        type=str,
        default=None,
        help=(
            "Optional SimTeG LM checkpoint (.pkl) used to initialize --semantic_encoder roberta_finetuned. "
            "If omitted, semantic stages search for TwiBot-20_seed_<seed>/checkpoints/LM_pretrain/best.pkl."
        ),
    )
    parser.add_argument(
        "--qwen_trust_remote_code",
        action="store_true",
        help="Explicitly allow HuggingFace remote code when loading Qwen models/tokenizers.",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--semantic_supervision_mode",
        type=str,
        default="classifier",
        choices=["classifier", "answer_token"],
        help=(
            "Supervision mode for semantic_encoder_finetune. "
            "'classifier' keeps the legacy hidden-state classifier-head CE path; "
            "'answer_token' performs causal-LM answer-token finetuning over the prompt text."
        ),
    )
    parser.add_argument("--lm_optimizer", dest="optimizer_LM", type=str, default="adamw")
    parser.add_argument("--optimizer_LM", dest="optimizer_LM", type=str, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--LM_classifier_n_layers", type=int, default=2)
    parser.add_argument("--LM_classifier_hidden_dim", type=int, default=128)
    parser.add_argument("--lm_dropout", dest="LM_dropout", type=float, default=0.1)
    parser.add_argument("--LM_dropout", dest="LM_dropout", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--lm_attention_dropout", dest="LM_att_dropout", type=float, default=0.1)
    parser.add_argument("--LM_att_dropout", dest="LM_att_dropout", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--label_smoothing_factor", type=float, default=0.0)
    parser.add_argument("--warmup", type=float, default=0.6)

    parser.add_argument("--use_GNN", action="store_true")
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_relations", type=int, default=2)
    parser.add_argument("--activation", type=str, default="leakyrelu")
    parser.add_argument("--gnn_optimizer", dest="optimizer_GNN", type=str, default="adamw")
    parser.add_argument("--optimizer_GNN", dest="optimizer_GNN", type=str, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--gnn_dropout", dest="GNN_dropout", type=float, default=0.4)
    parser.add_argument("--GNN_dropout", dest="GNN_dropout", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--att_heads", type=int, default=8)
    parser.add_argument(
        "--graph_second_view_scope",
        type=str,
        default="auto",
        choices=["auto", "none", "labeled_prefix", "routed_nodes", "neighborloader_batch"],
        help=(
            "Canonical scope switch for the HyperScan-style second view. "
            "`auto` preserves --graph_refine_mode; `none` disables the second view; "
            "`labeled_prefix` maps to a labeled-prefix dynamic branch; "
            "`routed_nodes` maps to routed_dynamic_hyperscan_branch and requires --routed_nodes_path; "
            "`neighborloader_batch` maps to hyperscan_neighborloader_batch_local_branch."
        ),
    )
    parser.add_argument(
        "--graph_second_view_candidate_scope",
        type=str,
        default="auto",
        choices=["auto", "undirected_relation_1hop", "labeled_relation_1hop"],
        help=(
            "Canonical KNN candidate-scope switch for relation-local second-view grouping. "
            "`auto` preserves --graph_refine_candidate_scope."
        ),
    )
    parser.add_argument(
        "--graph_second_view_training_geometry",
        type=str,
        default="auto",
        choices=["auto", "full_batch", "neighbor_subgraph"],
        help=(
            "Canonical graph-detector training-geometry switch for second-view comparisons. "
            "`auto` preserves --graph_training_loader_mode."
        ),
    )
    parser.add_argument(
        "--graph_second_view_hypergraph_backend",
        type=str,
        default="pyg",
        choices=["pyg", "dhg"],
        help=(
            "Second-view hypergraph realization backend for rgcn_hyperscan* backbones: "
            "`pyg` uses torch_geometric.nn.HypergraphConv; `dhg` uses DHG HGNNConv over "
            "the same dynamic incidence specification."
        ),
    )
    parser.add_argument(
        "--graph_second_view_fusion",
        type=str,
        default="residual",
        choices=["residual", "multiattn"],
        help=(
            "Second-view fusion realization for rgcn_hyperscan* backbones. "
            "`residual` uses x_low plus an incident-gated delta; `multiattn` uses the "
            "HyperScan-style bidirectional MultiAttn concat detector."
        ),
    )
    parser.add_argument(
        "--hyperscan_detector_style",
        type=str,
        default="residual",
        choices=["residual", "original_cross_attention", "multiattn"],
        help=(
            "Legacy detector/fusion head for HyperScan-style graph backbones. "
            "Prefer --graph_second_view_fusion for new experiments. "
            "`residual` keeps the current lightweight residual fusion head; "
            "`original_cross_attention`/`multiattn` swaps in the paper-style bidirectional "
            "cross-attention, concat, LeakyReLU, and linear detector while leaving "
            "the relation and hypergraph branches unchanged."
        ),
    )
    parser.add_argument("--SimpleHGN_att_res", type=float, default=0.2)
    parser.add_argument("--RGT_semantic_heads", type=int, default=8)
    parser.add_argument(
        "--graph_refine_mode",
        type=str,
        default="none",
        choices=[
            "none",
            "directional_relation_aware_prune",
            "directional_relation_aware_random_prune",
            "directional_relation_aware_oracle_prune",
            "directional_relation_aware_oracle_prune_no_hetero_priority",
            "hyperscan_knn_hypergraph_proxy_augment",
            "relation_overlap_knn_proxy_augment",
            "relation_overlap_knn_repr_prefit_augment",
            "routed_dynamic_hyperscan_branch",
            "hyperscan_neighborloader_batch_local_branch",
        ],
        help="Optional graph refinement policy applied before or during graph_detector_prepare GNN training.",
    )
    parser.add_argument(
        "--graph_refine_budget",
        type=float,
        default=0.0,
        help="Requested per-relation prune fraction for graph_refine_mode when graph-only refinement is enabled.",
    )
    parser.add_argument(
        "--graph_refine_knn_k",
        type=int,
        default=8,
        help=(
            "Neighborhood size for hyperscan_knn_hypergraph_proxy_augment and "
            "relation_overlap_knn_proxy_augment and "
            "relation_overlap_knn_repr_prefit_augment and "
            "routed_dynamic_hyperscan_branch."
        ),
    )
    parser.add_argument(
        "--graph_refine_candidate_scope",
        type=str,
        default="undirected_relation_1hop",
        choices=[
            "undirected_relation_1hop",
            "labeled_relation_1hop",
        ],
        help=(
            "Candidate pool for local KNN grouping inside relation_overlap_* and "
            "routed_dynamic_hyperscan_branch. "
            "`undirected_relation_1hop` uses all relation-1hop neighbors; "
            "`labeled_relation_1hop` filters those neighbors to the labeled prefix only."
        ),
    )
    parser.add_argument(
        "--graph_neighbor_num_neighbors",
        type=int,
        default=64,
        help=(
            "NeighborLoader fanout per GNN layer when graph-detector training uses "
            "neighbor-subgraph batches."
        ),
    )
    parser.add_argument(
        "--graph_training_loader_mode",
        type=str,
        default="full_batch",
        choices=[
            "full_batch",
            "neighbor_subgraph",
        ],
        help=(
            "Graph-detector optimization regime. "
            "`full_batch` keeps the current all-node update. "
            "`neighbor_subgraph` trains on NeighborLoader sampled subgraph batches."
        ),
    )
    parser.add_argument(
        "--graph_training_max_steps",
        type=int,
        default=0,
        help=(
            "Optional total optimizer-step budget for graph-detector training. "
            "Use this to match NeighborLoader runs to the same update-count budget "
            "as a full-batch baseline. 0 disables the cap."
        ),
    )
    parser.add_argument(
        "--mhlgc_enable",
        action="store_true",
        help=(
            "Enable MH-LGC-style LLM-guided contrastive learning for graph_detector_prepare. "
            "Requires --mhlgc_semantic_embedding_path and adds an auxiliary hard-negative InfoNCE loss."
        ),
    )
    parser.add_argument(
        "--mhlgc_semantic_embedding_path",
        type=str,
        default="",
        help=(
            "Graph-node aligned LLM semantic embedding tensor used by --mhlgc_enable. "
            "The tensor may be stored directly or under an embeddings/semantic_embeddings/features key."
        ),
    )
    parser.add_argument(
        "--mhlgc_loss_weight",
        type=float,
        default=0.0,
        help="Weight lambda for the MH-LGC auxiliary contrastive loss. Default 0 keeps existing CE-only behavior.",
    )
    parser.add_argument(
        "--mhlgc_beta",
        type=float,
        default=1.0,
        help="Hard-negative impact beta in the MH-LGC negative weighting distribution.",
    )
    parser.add_argument(
        "--mhlgc_gamma",
        type=float,
        default=0.5,
        help="Balance gamma between GNN embedding similarity and LLM semantic similarity in MH-LGC.",
    )
    parser.add_argument(
        "--mhlgc_temperature",
        type=float,
        default=1.0,
        help="InfoNCE temperature for MH-LGC.",
    )
    parser.add_argument(
        "--mhlgc_feature_mask_probability",
        type=float,
        default=0.15,
        help="Feature masking probability used to create the positive augmented view in MH-LGC.",
    )
    parser.add_argument(
        "--mhlgc_edge_mask_probability",
        type=float,
        default=0.10,
        help="Edge masking probability used to create the positive augmented relation view in MH-LGC.",
    )
    parser.add_argument(
        "--mhlgc_anchors_per_batch",
        type=int,
        default=1,
        help="Number of lowest-positive-score positive-label nodes used as borderline anchors per batch.",
    )
    parser.add_argument(
        "--mhlgc_positive_label",
        type=int,
        default=1,
        help="Positive/fraud/bot class index for selecting MH-LGC borderline anchors. TwiBot labels use 1 for bot.",
    )
    parser.add_argument(
        "--local_conf_disable_degree_guard",
        action="store_true",
        help="Disable the no-zero-in/no-zero-out structural guard inside local_conformal_diagnostic.",
    )
    parser.add_argument(
        "--local_conf_similarity_gate_threshold",
        type=float,
        default=None,
        help=(
            "Optional max cosine threshold for local_conformal_diagnostic delete candidates. "
            "When set, only harmful edges with semantic cosine <= threshold remain eligible."
        ),
    )
    parser.add_argument(
        "--conflict_router_budget",
        type=float,
        default=0.10,
        help="Target routed-node fraction for local_conflict_diagnostic, tuned on validation and applied to test.",
    )
    parser.add_argument(
        "--conflict_topk_per_bucket",
        type=int,
        default=1,
        help="Maximum number of positive-delta conflict edges removed per (relation, target_role) bucket.",
    )
    parser.add_argument(
        "--conflict_disable_degree_guard",
        action="store_true",
        help="Disable the no-zero-in/no-zero-out structural guard inside local_conflict_diagnostic.",
    )
    parser.add_argument(
        "--local_dignn_conflict_router_budget",
        type=float,
        default=0.10,
        help="Target routed-node fraction for local_dignn_conflict_refine_diag.",
    )
    parser.add_argument(
        "--local_dignn_conflict_topk_per_bucket",
        type=int,
        default=1,
        help="Maximum number of selected edges per (relation, target_role) bucket for local_dignn_conflict_refine_diag.",
    )
    parser.add_argument(
        "--local_dignn_conflict_disable_degree_guard",
        action="store_true",
        help="Disable the no-zero-in/no-zero-out structural guard inside local_dignn_conflict_refine_diag.",
    )
    parser.add_argument("--external_graph_edge_index_path", type=str, default=None)
    parser.add_argument("--external_graph_edge_type_path", type=str, default=None)

    parser.add_argument("--MLP_n_layers", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--MLP_hidden_dim", type=int, default=128, help=argparse.SUPPRESS)
    parser.add_argument("--optimizer_MLP", type=str, default="adamw", help=argparse.SUPPRESS)
    parser.add_argument("--MLP_dropout", type=float, default=0.4, help=argparse.SUPPRESS)

    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--LM_pretrain_epochs", type=float, default=5)
    parser.add_argument(
        "--semantic_max_steps",
        type=int,
        default=0,
        help="Optional step cap for --experiment_task semantic_encoder_finetune; 0 uses one pass over the selected train subset.",
    )
    parser.add_argument(
        "--semantic_train_limit",
        type=int,
        default=0,
        help="Optional train_idx cap for --experiment_task semantic_encoder_finetune smoke runs; 0 uses the full train split.",
    )
    parser.add_argument("--MLP_KD_epochs", type=int, default=300, help=argparse.SUPPRESS)
    parser.add_argument("--LM_eval_patience", type=int, default=20)
    parser.add_argument("--LM_accumulation", type=int, default=1)
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument("--GNN_epochs_per_iter", type=int, default=200)
    parser.add_argument("--LM_epochs_per_iter", type=int, default=3)
    parser.add_argument("--MLP_epochs_per_iter", type=int, default=300, help=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=3)
    parser.add_argument("--pl_ratio_LM", type=float, default=0.5)
    parser.add_argument("--pl_ratio_GNN", type=float, default=0.0)
    parser.add_argument("--pl_ratio_MLP", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.7, help=argparse.SUPPRESS)

    parser.add_argument("--lm_learning_rate", dest="lr_LM", type=float, default=1e-5)
    parser.add_argument("--lr_LM", dest="lr_LM", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--lm_weight_decay", dest="weight_decay_LM", type=float, default=0.01)
    parser.add_argument("--weight_decay_LM", dest="weight_decay_LM", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--gnn_learning_rate", dest="lr_GNN", type=float, default=5e-4)
    parser.add_argument("--lr_GNN", dest="lr_GNN", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--gnn_weight_decay", dest="weight_decay_GNN", type=float, default=1e-5)
    parser.add_argument("--weight_decay_GNN", dest="weight_decay_GNN", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--lr_MLP", type=float, default=5e-4, help=argparse.SUPPRESS)
    parser.add_argument("--weight_decay_MLP", type=float, default=1e-5, help=argparse.SUPPRESS)

    parser.add_argument("--sparse_degree_quantile", type=float, default=0.25)
    parser.add_argument("--propagation_quantile", type=float, default=0.85)
    parser.add_argument("--bootstrap_samples", type=int, default=512)
    parser.add_argument(
        "--external_frozen_g0_root",
        type=str,
        default=None,
        help=(
            "Optional experiment root whose graph-detector preparation artifact is reused read-only by "
            "diagnostic graph-aware stages. Public joint_router_refinement requires the current run's own "
            "preparation/graph_detector artifact unless --joint_refiner_embedding_path is used for a "
            "refiner-only semantic override ablation."
        ),
    )
    parser.add_argument(
        "--joint_train_node_cap",
        type=int,
        default=3000,
        help=(
            "Training-node cap for public joint_router_refinement. "
            "Use 3000 for the paper-style cap, or 0 to train router/refiner on the full train split."
        ),
    )
    parser.add_argument(
        "--joint_refiner_explicit_gate",
        action="store_true",
        help=(
            "Enable an explicit keep/change gate inside public joint_router_refinement. "
            "The gate interpolates between the frozen GNN path and the routed refiner path."
        ),
    )
    parser.add_argument(
        "--joint_refiner_target_mode",
        type=str,
        default="predict",
        choices=["predict", "keep_change"],
        help=(
            "Training target for the strict joint refiner. "
            "'predict' keeps direct label prediction, while 'keep_change' learns whether the routed node should flip the base detector decision."
        ),
    )
    parser.add_argument(
        "--joint_refiner_gate_target",
        type=str,
        default="base_wrong",
        choices=["base_wrong", "utility_positive"],
        help=(
            "Gate supervision target for explicit-gate joint refiners. "
            "'base_wrong' preserves the old keep/change target, while 'utility_positive' trains the gate to change only when raw refiner utility is positive."
        ),
    )
    parser.add_argument(
        "--joint_refiner_weight_mode",
        type=str,
        default="off",
        choices=["off", "base_wrong", "utility_positive", "base_wrong_plus_utility"],
        help=(
            "Optional routed-node loss reweighting for public joint_router_refinement. "
            "Weights can emphasize base-wrong nodes, oracle-utility-positive nodes, or both."
        ),
    )
    parser.add_argument(
        "--joint_refiner_base_wrong_weight",
        type=float,
        default=2.0,
        help="Multiplicative routed-node loss weight applied to base-wrong nodes when joint_refiner_weight_mode includes base_wrong.",
    )
    parser.add_argument(
        "--joint_refiner_utility_weight",
        type=float,
        default=3.0,
        help="Multiplicative routed-node loss weight applied to oracle-utility-positive nodes when joint_refiner_weight_mode includes utility_positive.",
    )
    parser.add_argument(
        "--joint_refiner_gate_weight",
        type=float,
        default=0.5,
        help="Loss weight for explicit keep/change gate supervision when joint_refiner_explicit_gate is enabled.",
    )
    parser.add_argument(
        "--joint_utility_advantage_experiment",
        type=str,
        default="off",
        choices=[
            "off",
            "utility_gate_only",
            "botmoe_selector_only",
            "utility_gate_botmoe_selector",
        ],
        help=(
            "Public GLANCE utility-advantage experiment preset for joint_router_refinement. "
            "'off' preserves the existing default behavior; utility-gate presets use the explicit keep/change gate, "
            "and BotMoE selector presets keep expert selection separate from the keep/change decision."
        ),
    )
    parser.add_argument(
        "--joint_refiner_gate_policy",
        type=str,
        default="soft_mix",
        choices=["soft_mix", "hard_keep_change"],
        help=(
            "Policy for applying the explicit keep/change gate in utility-advantage joint refiners. "
            "'soft_mix' preserves the existing interpolation behavior; 'hard_keep_change' applies thresholded keep/change decisions at inference/evaluation time."
        ),
    )
    parser.add_argument(
        "--joint_refiner_gate_threshold",
        type=float,
        default=0.5,
        help="Threshold for hard_keep_change explicit-gate application; default 0.5 preserves standard binary gate semantics.",
    )
    parser.add_argument(
        "--joint_prompt_expert_fusion",
        type=str,
        default="projector_concat",
        choices=[
            "projector_concat",
            "mpe_gated",
            "gaugllm_selector",
            "gaugllm_mope",
            "botmoe_selector",
            "utility_correction_moe",
            "metades_selector",
            "conflict_aware_correction_moe",
            "raw_concat_ego_following",
            "raw_concat_ego_follower",
            "raw_concat_ego_following_follower",
            "raw_concat_single_graph_following",
            "raw_concat_single_graph_follower",
            "raw_concat_single_tweet",
            "raw_concat_single_conflict",
            "raw_concat_follower_tweet",
            "raw_concat_following_triplet",
            "raw_concat_follower_triplet",
            "ultratag_propagated_follower_triplet",
            "raw_concat_metadata_anchor",
            "raw_concat_metadata_only",
        ],
        help=(
            "Fusion head for prompt-expert joint refiners. "
            "'projector_concat' preserves the existing projected concat path; "
            "'mpe_gated' uses the existing node-conditioned softmax over graph_following, graph_follower, tweet, conflict, and metadata_structured experts; "
            "'gaugllm_selector' reuses routed explanation sidecars at runtime, encodes selector-context texts with the finetuned SimTeG RoBERTa line, and applies a strict 4-expert context-aware selector over graph_following, graph_follower, tweet, and conflict; "
            "'gaugllm_mope' reuses the same runtime sidecars but applies the official GAugLLM SimilarityAttentionMLP-style content-plus-context similarity MoPE head; "
            "'botmoe_selector' applies sparse BotMoE-style noisy top-k selection over graph_following, graph_follower, tweet, and conflict only; "
            "'utility_correction_moe' trains expert-specific correction heads over graph_follower, tweet, conflict, and metadata_structured, then uses per-expert utility probabilities as the abstain-aware selector; "
            "'metades_selector' is a META-DES-style competence selector over graph_follower, tweet, conflict, and a runtime follower_triplet action, using base/expert confidence and disagreement meta-features for per-action utility; "
            "'conflict_aware_correction_moe' selects only first-order prompt experts graph_following, graph_follower, and tweet, while using conflict plus runtime explanation-context embeddings as safety/context features for correction utility; "
            "'raw_concat_ego_following' concatenates z_gnn + ego + graph_following + structural side features without projectors; "
            "'raw_concat_ego_follower' concatenates z_gnn + ego + graph_follower + structural side features without projectors; "
            "'raw_concat_ego_following_follower' concatenates z_gnn + ego + graph_following + graph_follower + structural side features without projectors; "
            "'raw_concat_single_*' concatenates z_gnn + one raw expert + structural side features without projectors; "
            "'raw_concat_follower_tweet' concatenates z_gnn + graph_follower + tweet + structural side features without projectors; "
            "'raw_concat_following_triplet' concatenates z_gnn + graph_following + tweet + conflict + structural side features without projectors; "
            "'raw_concat_follower_triplet' concatenates z_gnn + graph_follower + tweet + conflict + structural side features without projectors; "
            "'ultratag_propagated_follower_triplet' is a legacy mean-propagation routed-node ablation, not an UltraTAG-S reproduction: it mean-propagates existing prompt-expert embeddings over one graph hop at runtime, then concatenates z_gnn + propagated graph_follower/tweet/conflict + structural side features; "
            "'raw_concat_metadata_anchor' adds the structured metadata expert to the follower_triplet anchor; "
            "'raw_concat_metadata_only' isolates z_gnn + metadata_structured + structural side features."
        ),
    )
    parser.add_argument(
        "--joint_prompt_expert_mope_attention",
        type=str,
        default="similarity",
        choices=["similarity"],
        help=(
            "Attention family for --joint_prompt_expert_fusion gaugllm_mope. "
            "Only 'similarity' is currently implemented, matching the official GAugLLM default SimilarityAttentionMLP."
        ),
    )
    parser.add_argument(
        "--joint_prompt_expert_mope_temperature",
        type=float,
        default=0.2,
        help=(
            "Softmax temperature for --joint_prompt_expert_fusion gaugllm_mope. "
            "The default 0.2 preserves the official GAugLLM SimilarityAttentionMLP setting."
        ),
    )
    parser.add_argument(
        "--joint_prompt_expert_mope_logit_norm",
        type=str,
        default="none",
        choices=["none", "branch_zscore", "combined_zscore"],
        help=(
            "Optional diagnostic calibration for gaugllm_mope selector logits. "
            "'none' preserves the official formula; 'branch_zscore' normalizes content and similarity logits separately "
            "across experts per node before summing; 'combined_zscore' normalizes the summed logits before temperature."
        ),
    )
    parser.add_argument(
        "--joint_prompt_expert_mope_entropy_weight",
        type=float,
        default=0.0,
        help=(
            "Lightweight selector-entropy regularization weight for gaugllm_mope. "
            "Adds weight * (log(E) - mean_entropy(selector_weights)) to the routed-node refiner loss "
            "so diagnostics can test whether a softer selector reduces single-expert collapse."
        ),
    )
    parser.add_argument(
        "--joint_prompt_expert_mope_load_balance_weight",
        type=float,
        default=0.0,
        help=(
            "Lightweight load-balance regularization weight for gaugllm_mope. "
            "Adds weight * ||mean(selector_weights) - availability_prior||^2 over routed training nodes, "
            "where availability_prior is derived from expert_presence_mask in the current batch."
        ),
    )
    parser.add_argument(
        "--joint_prompt_expert_botmoe_top_k",
        type=int,
        default=1,
        help=(
            "Sparse top-k expert count for --joint_prompt_expert_fusion botmoe_selector. "
            "Defaults to 1, matching the sparse single-expert BotMoE setting used in the local BotMoE reference."
        ),
    )
    parser.add_argument(
        "--joint_prompt_expert_botmoe_noisy_gating",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable train-time noisy top-k gating for botmoe_selector. "
            "Use --no-joint_prompt_expert_botmoe_noisy_gating for deterministic sparse gating diagnostics."
        ),
    )
    parser.add_argument(
        "--joint_prompt_expert_botmoe_aux_weight",
        type=float,
        default=1e-2,
        help=(
            "Auxiliary BotMoE load-balancing coefficient for botmoe_selector. "
            "Scales cv_squared(importance) + cv_squared(load) on routed training batches."
        ),
    )
    parser.add_argument(
        "--joint_correction_moe_expert_weight",
        type=float,
        default=0.5,
        help=(
            "Auxiliary expert-head classification loss weight for correction-selector fusion modes "
            "(utility_correction_moe and metades_selector). "
            "The loss is applied only on routed nodes and available correction experts."
        ),
    )
    parser.add_argument(
        "--joint_correction_moe_utility_weight",
        type=float,
        default=1.0,
        help=(
            "Auxiliary per-expert utility BCE loss weight for correction-selector fusion modes "
            "(utility_correction_moe and metades_selector). "
            "Target semantics are controlled by --joint_correction_moe_utility_target."
        ),
    )
    parser.add_argument(
        "--joint_correction_moe_utility_target",
        type=str,
        default="loss_advantage",
        choices=["loss_advantage", "decision_gain", "hybrid", "net_gain"],
        help=(
            "Per-expert utility target for correction-selector fusion modes "
            "(utility_correction_moe and metades_selector). "
            "'loss_advantage' preserves base_loss - expert_loss - beta > 0; "
            "'decision_gain' uses base_wrong and expert_pred_correct; "
            "'hybrid' treats either condition as utility-positive; "
            "'net_gain' uses the decision-gain positive target while treating break actions "
            "as explicit negative-utility examples through --joint_correction_moe_break_weight."
        ),
    )
    parser.add_argument(
        "--joint_correction_moe_break_weight",
        type=float,
        default=1.0,
        help=(
            "Break-action penalty for correction-selector fusion modes. "
            "A base-correct/expert-wrong action receives utility reward -break_weight "
            "for pairwise ranking and receives this multiplier in the utility BCE negative weight. "
            "Default 1.0 preserves old negative weighting unless the flag is raised."
        ),
    )
    parser.add_argument(
        "--joint_correction_moe_ranking_weight",
        type=float,
        default=0.0,
        help=(
            "Within-node per-action utility ranking loss weight for correction-selector fusion modes. "
            "When positive, utility logits are trained to rank fix actions above no-gain actions "
            "and no-gain actions above break actions using the correction utility reward matrix."
        ),
    )
    parser.add_argument(
        "--joint_correction_moe_ranking_margin",
        type=float,
        default=0.0,
        help="Softplus margin for --joint_correction_moe_ranking_weight.",
    )
    parser.add_argument(
        "--joint_correction_moe_gate_calibration",
        type=str,
        default="off",
        choices=["off", "global_threshold", "per_action_threshold"],
        help=(
            "Validation-locked gate calibration for correction-selector fusion modes. "
            "'global_threshold' selects one utility gate threshold on valid and locks it to test; "
            "'per_action_threshold' selects one threshold per selected correction action on valid. "
            "The default 'off' keeps the existing --joint_refiner_gate_threshold path."
        ),
    )
    parser.add_argument(
        "--prompt_expert_quality_cache_path",
        type=str,
        default=None,
        help=(
            "Prompt-expert quality audit input cache. Expected to be a prompt_expert_bundle_v2 "
            ".pt payload with graph_following, graph_follower, tweet, conflict, and adjacent "
            "manifest/explanation sidecars."
        ),
    )
    parser.add_argument(
        "--prompt_expert_quality_reference_stage",
        type=str,
        default=None,
        help=(
            "Reference joint_router_refinement stage directory used by prompt_expert_quality_audit "
            "to load routed masks, labels, base predictions, and correction utility tensors. "
            "If omitted, --joint_router_reuse_root is used."
        ),
    )
    parser.add_argument(
        "--prompt_expert_quality_components",
        type=str,
        default="graph_following,graph_follower,tweet,conflict",
        help=(
            "Comma-separated prompt expert components audited by prompt_expert_quality_audit. "
            "Defaults to the four v2 explanation-first experts."
        ),
    )
    parser.add_argument(
        "--prompt_expert_quality_probe_C",
        type=float,
        default=0.1,
        help="Logistic-regression C used by prompt_expert_quality_audit separability probes.",
    )
    parser.add_argument(
        "--prompt_expert_quality_threshold_grid",
        type=int,
        default=501,
        help="Number of validation threshold candidates for prompt_expert_quality_audit utility probes.",
    )
    parser.add_argument(
        "--embedding_path",
        dest="embedding_path",
        type=str,
        default=None,
        help=(
            "Cached semantic embedding tensor for Phase A and graph_detector_prepare. "
            "joint_router_refinement now inherits its semantic source from the current run's "
            "graph_detector_prepare feature_manifest and only accepts an explicit path when it matches that artifact."
        ),
    )
    parser.add_argument("--emb_path", dest="embedding_path", type=str, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--g0_feature_path", dest="embedding_path", type=str, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--joint_refiner_embedding_path",
        type=str,
        default=None,
        help=(
            "Optional refiner-only semantic override for joint_router_refinement. "
            "Keeps graph_detector_prepare backbone features unchanged and only replaces the joint refiner semantic source. "
            "Accepts either a plain [num_nodes, d] tensor, a prompt-cache payload with ego/hop1/hop2 tensors, "
            "or a prompt-expert bundle payload with ego/graph_following/graph_follower/tweet/conflict/metadata_structured components."
        ),
    )
    parser.add_argument(
        "--joint_routing_protocol",
        type=str,
        default="joint_train",
        choices=["joint_train", "frozen_router_reuse"],
        help=(
            "Routing protocol for joint_router_refinement. "
            "'joint_train' keeps the current jointly trained router+refiner path; "
            "'frozen_router_reuse' freezes the router and reuses a prior joint_router_refinement routing artifact."
        ),
    )
    parser.add_argument(
        "--joint_router_reuse_root",
        type=str,
        default=None,
        help=(
            "Root or stage directory containing the router artifact reused when "
            "--joint_routing_protocol frozen_router_reuse."
        ),
    )
    parser.add_argument(
        "--phase_a_project_dim",
        type=int,
        default=768,
        help="Fixed semantic embedding width before Phase A GNN input.",
    )
    parser.add_argument(
        "--phase_a_projector",
        type=str,
        default="pca",
        choices=["pca"],
        help="Unlabeled projection method used to align Phase A semantic embedding widths.",
    )
    parser.add_argument(
        "--graph_node_input_family",
        type=str,
        default="semantic_embedding",
        choices=["semantic_embedding", "hyperscan_meta_tweet_proxy"],
        help=(
            "Node-input family for Phase-A graph detectors. "
            "`semantic_embedding` keeps the current single-tensor semantic input. "
            "`hyperscan_meta_tweet_proxy` rebuilds a HyperScan-style "
            "tweet+numeric+categorical node input from the labeled graph."
        ),
    )
    parser.add_argument(
        "--peft",
        action="store_true",
        help="Enable a LoRA-style trainable input adapter on top of the frozen Phase A projection.",
    )
    parser.add_argument("--peft_rank", type=int, default=8)
    parser.add_argument("--peft_alpha", type=float, default=16.0)
    parser.add_argument("--graph_detector_epochs", dest="g0_epochs", type=int, default=0)
    parser.add_argument("--g0_epochs", dest="g0_epochs", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--gate_calibrator_max_iter", dest="gats_max_iter", type=int, default=300)
    parser.add_argument("--gats_max_iter", dest="gats_max_iter", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--risk_budgets", type=str, default=None, help="Comma-separated residual-risk routing budgets in (0, 1].")
    parser.add_argument("--router_budgets", dest="router_budgets", type=str, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--conformal_knn_k",
        type=int,
        default=8,
        help="KNN neighborhood size for --estimator_mode conformal_knn_risk_router.",
    )
    parser.add_argument(
        "--conformal_knn_candidate_scope",
        type=str,
        default="labeled_full",
        choices=["labeled_full", "hyperscan_full", "labeled_relation_1hop", "undirected_relation_1hop"],
        help=(
            "Candidate pool for --estimator_mode conformal_knn_risk_router. "
            "`hyperscan_full` aligns with HyperScan's full feature-pool kNN hyperedge construction; "
            "`labeled_full` searches the labeled graph; `labeled_relation_1hop` searches only labeled 1-hop relation neighbors."
        ),
    )
    parser.add_argument(
        "--conformal_knn_target_top_n",
        type=int,
        default=200,
        help=(
            "Number of top-ranked target-node diagnostics to store for "
            "--estimator_mode conformal_knn_risk_router. KNN neighbors are support evidence, "
            "not routed outputs."
        ),
    )
    parser.add_argument(
        "--conformal_knn_anchor_top_n",
        dest="conformal_knn_target_top_n",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--conformal_knn_shrinkage_tau",
        type=float,
        default=3.0,
        help="Shrinkage strength when blending KNN local risk back toward the base conformal risk.",
    )
    parser.add_argument(
        "--conformal_knn_ncp_lambda",
        type=float,
        default=1.0,
        help=(
            "Neighborhood Conformal Prediction localization temperature lambda_L for "
            "exp(-distance/lambda_L) KNN support weighting in conformal_knn_risk_router."
        ),
    )
    parser.add_argument(
        "--conformal_knn_neighbor_mode",
        type=str,
        default="standard",
        choices=["standard", "mutual", "threshold", "adaptive", "mutual_adaptive"],
        help=(
            "Non-model KNN support filter for conformal_knn_risk_router. `standard` preserves "
            "the previous top-k behavior; `mutual` keeps only reciprocal feature neighbors; "
            "`threshold` keeps neighbors above --conformal_knn_similarity_threshold; "
            "`adaptive` may use --conformal_knn_adaptive_max_k before thresholding; "
            "`mutual_adaptive` combines reciprocal and threshold/adaptive filtering."
        ),
    )
    parser.add_argument(
        "--conformal_knn_similarity_threshold",
        type=float,
        default=-1.0,
        help=(
            "Cosine-similarity floor used by threshold/adaptive conformal KNN neighbor modes. "
            "Values <= -1 disable threshold filtering."
        ),
    )
    parser.add_argument(
        "--conformal_knn_min_support",
        type=int,
        default=1,
        help=(
            "Minimum filtered KNN support size required before using local KNN evidence. "
            "Centers below this count fall back to target-only/global local-calibration features."
        ),
    )
    parser.add_argument(
        "--conformal_knn_adaptive_max_k",
        type=int,
        default=0,
        help=(
            "Optional maximum candidate count fetched before adaptive/threshold KNN filtering. "
            "A value of 0 keeps --conformal_knn_k as the fetch size."
        ),
    )
    parser.add_argument(
        "--conformal_knn_hubness_correction",
        type=str,
        default="none",
        choices=["none", "degree"],
        help=(
            "Optional non-model hubness correction for NCP KNN weights. `degree` downweights "
            "neighbors that appear in many centers' raw top-k support lists."
        ),
    )
    parser.add_argument(
        "--conformal_knn_repr_source",
        type=str,
        default="node_repr",
        choices=["node_repr", "x_low", "x_new"],
        help=(
            "Representation space used by --estimator_mode conformal_knn_risk_router. "
            "`node_repr` keeps the frozen GNN output; `x_low` treats that relation-view output as "
            "the HyperScan low view; `x_new` concatenates x_low with the frozen G0 input features."
        ),
    )
    parser.add_argument(
        "--conformal_knn_learning_mode",
        type=str,
        default="fixed",
        choices=["fixed", "ncp_local"],
        help=(
            "Risk-fusion mode for conformal_knn_risk_router. `fixed` preserves the "
            "validation-selected fixed score families; `ncp_local` uses KNN neighborhood "
            "calibration samples with exp(-distance/lambda_L) weights to compute local "
            "conformal features, then selects among nonparametric NCP-local score families "
            "on the tune split by AUPRC-error first, without fitting a logistic head."
        ),
    )
    parser.add_argument(
        "--conformal_knn_score_family_override",
        type=str,
        default="auto",
        help=(
            "Optional score-family override for conformal_knn_risk_router. "
            "`auto` selects on the validation tune split; `base_only` is the strict target-only "
            "risk baseline for target->KNN support-group ablations."
        ),
    )
    parser.add_argument("--router_oof_mode", type=str, default="artifact", choices=["artifact"])
    parser.add_argument("--phase_a_semantic_sources", type=str, default="auto", help=argparse.SUPPRESS)
    parser.add_argument("--phase_a_feature_paths", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--phase_a_gnn_family", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--phase_a_fixed_gnn", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--phase_a_fixed_semantic_source", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--phase_a_latency_repeats", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--phase_a_include_structural_smoke", action="store_true", help=argparse.SUPPRESS)

    return normalize_args(parser.parse_args(raw_args), raw_args=raw_args)
