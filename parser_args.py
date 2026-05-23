import argparse
import sys


CANONICAL_STAGE_CHOICES = (
    "frozen_g0",
    "frozen_gats",
    "semantic_finetune",
    "local_conformal_prune_diag",
    "glance_joint_router_refine",
    "legacy_distill",
    "vertical_minimal",
    "estimator_matrix",
    "semantic_matrix",
    "semantic_source_matrix",
    "repair_matrix",
    "selector_matrix",
    "positioning_matrix",
    "backbone_stress",
    "appendix",
)
LEGACY_STAGE_CHOICES = CANONICAL_STAGE_CHOICES
DEPRECATED_STAGE_VALUES = set()

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
    "--embedding_path": "--emb_path",
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

    if getattr(args, "stage", None) in DEPRECATED_STAGE_VALUES:
        deprecated.append(f"stage:{args.stage}")

    return _dedupe_preserve_order(deprecated)


def normalize_args(args, raw_args=None):
    """Normalize shared compatibility aliases without reviving retired GLANCE flags."""
    if getattr(args, "emb_path", None):
        args.g0_feature_path = args.emb_path
    elif getattr(args, "g0_feature_path", None):
        args.emb_path = args.g0_feature_path

    args.reset_split = str(args.reset_split).strip()
    args.deprecated_cli_flags = _deprecated_cli_flags(args, raw_args)
    return args


def parser_args(argv=None):
    raw_args = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser()

    # Core routing
    parser.add_argument(
        "--experiment_task",
        dest="stage",
        type=str,
        default="legacy_distill",
        choices=CANONICAL_STAGE_CHOICES,
        help="Canonical experiment task. Use legacy --stage only for historical commands.",
    )
    parser.add_argument(
        "--stage",
        dest="stage",
        type=str,
        default=argparse.SUPPRESS,
        choices=LEGACY_STAGE_CHOICES,
        help=argparse.SUPPRESS,
    )

    # Experiment metadata
    parser.add_argument("--project_name", type=str, default="baseline-core")
    parser.add_argument("--experiment_name", type=str, default="baseline_experiment")
    parser.add_argument("--artifact_root", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--reuse_existing_artifacts", action="store_true")
    parser.add_argument("--force_retrain_backbone", action="store_true")
    parser.add_argument(
        "--claim_grade",
        action="store_true",
        help="Require claim-grade comparability gates before launching a run.",
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="TwiBot-20", help="Dataset name")
    parser.add_argument("--lm_batch_size", dest="batch_size_LM", type=int, default=32)
    parser.add_argument("--batch_size_LM", dest="batch_size_LM", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--gnn_batch_size", dest="batch_size_GNN", type=int, default=300000)
    parser.add_argument("--batch_size_GNN", dest="batch_size_GNN", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--batch_size_MLP", type=int, default=300000, help=argparse.SUPPRESS)
    parser.add_argument("--raw_data_filepath", type=str, default="./data/raw/")
    parser.add_argument("--is_processed", type=_parse_bool, default=True, help=argparse.SUPPRESS)
    parser.add_argument("--reset_split", type=str, default="-1")

    # Stage routing
    parser.add_argument(
        "--graph_backbone",
        dest="GNN_model",
        type=str,
        default="botrgcn",
        choices=["botrgcn", "rgcn", "rgt", "hgt", "simplehgn", "gatv2"],
        help="Canonical graph detector backbone.",
    )
    parser.add_argument(
        "--GNN_model",
        dest="GNN_model",
        type=str,
        default=argparse.SUPPRESS,
        choices=["botrgcn", "rgcn", "rgt", "hgt", "simplehgn", "gatv2"],
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

    # LM configuration
    parser.add_argument("--text_encoder", dest="LM_model", type=str, default="roberta")
    parser.add_argument("--LM_model", dest="LM_model", type=str, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--semantic_encoder",
        dest="semantic_backbone",
        type=str,
        default="auto",
        choices=["auto", "roberta", "roberta_finetuned", "qwen3_frozen", "qwen3_peft"],
        help="Semantic encoder source for semantic finetune and Phase A embedding resolution.",
    )
    parser.add_argument(
        "--semantic_backbone",
        dest="semantic_backbone",
        type=str,
        default=argparse.SUPPRESS,
        choices=["auto", "roberta", "roberta_finetuned", "qwen3_frozen", "qwen3_peft"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--qwen_model_path",
        type=str,
        default="/root/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        help="Local path or HF id for Qwen3-Embedding-8B when --semantic_backbone qwen3_peft.",
    )
    parser.add_argument(
        "--qwen_trust_remote_code",
        action="store_true",
        help="Explicitly allow HuggingFace remote code when loading Qwen models/tokenizers.",
    )
    parser.add_argument("--max_length", type=int, default=512)
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

    # GNN configuration
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
        ],
        help="Optional graph-only refinement applied before frozen_g0 GNN training.",
    )
    parser.add_argument(
        "--graph_refine_budget",
        type=float,
        default=0.0,
        help="Requested per-relation prune fraction for graph_refine_mode when graph-only refinement is enabled.",
    )
    parser.add_argument(
        "--local_conf_disable_degree_guard",
        action="store_true",
        help="Disable the no-zero-in/no-zero-out structural guard inside local_conformal_prune_diag.",
    )
    parser.add_argument(
        "--local_conf_similarity_gate_threshold",
        type=float,
        default=None,
        help=(
            "Optional max cosine threshold for local_conformal_prune_diag delete candidates. "
            "When set, only harmful edges with semantic cosine <= threshold remain eligible."
        ),
    )
    parser.add_argument(
        "--external_graph_edge_index_path",
        type=str,
        default=None,
        help="Optional refined edge_index tensor consumed by graph-aware refiner stages.",
    )
    parser.add_argument(
        "--external_graph_edge_type_path",
        type=str,
        default=None,
        help="Optional refined edge_type tensor consumed by graph-aware refiner stages.",
    )

    # MLP configuration
    parser.add_argument("--MLP_n_layers", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--MLP_hidden_dim", type=int, default=128, help=argparse.SUPPRESS)
    parser.add_argument("--optimizer_MLP", type=str, default="adamw", help=argparse.SUPPRESS)
    parser.add_argument("--MLP_dropout", type=float, default=0.4, help=argparse.SUPPRESS)

    # Training and evaluation
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--LM_pretrain_epochs", type=float, default=5)
    parser.add_argument(
        "--semantic_max_steps",
        type=int,
        default=0,
        help="Optional step cap for --experiment_task semantic_finetune; 0 uses one pass over the selected train subset.",
    )
    parser.add_argument(
        "--semantic_train_limit",
        type=int,
        default=0,
        help="Optional train_idx cap for --experiment_task semantic_finetune smoke runs; 0 uses the full train split.",
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

    # Module-specific controls
    parser.add_argument("--sparse_degree_quantile", type=float, default=0.25)
    parser.add_argument("--propagation_quantile", type=float, default=0.85)
    parser.add_argument("--bootstrap_samples", type=int, default=512)
    parser.add_argument(
        "--external_frozen_g0_root",
        type=str,
        default=None,
        help="Optional experiment root whose frozen/g0 artifact is reused read-only by strict GLANCE stages.",
    )
    parser.add_argument(
        "--emb_path",
        dest="emb_path",
        type=str,
        default=None,
        help="Cached semantic embedding tensor for Phase A, frozen_g0, and strict GLANCE.",
    )
    parser.add_argument(
        "--embedding_path",
        dest="emb_path",
        type=str,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--g0_feature_path",
        dest="g0_feature_path",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
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
    parser.add_argument(
        "--phase_a_semantic_sources",
        type=str,
        default="auto",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase_a_feature_paths",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase_a_gnn_family",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase_a_fixed_gnn",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase_a_fixed_semantic_source",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--phase_a_latency_repeats", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument(
        "--phase_a_include_structural_smoke",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    return normalize_args(parser.parse_args(raw_args), raw_args=raw_args)
