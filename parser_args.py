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
        "--graph_backbone",
        dest="graph_backbone",
        type=str,
        default="botrgcn",
        choices=["botrgcn", "rgcn", "rgt", "hgt", "simplehgn", "gatv2"],
        help="Canonical graph detector backbone.",
    )
    parser.add_argument(
        "--GNN_model",
        dest="graph_backbone",
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
            "calibrated_local_risk_router",
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
        help="Local path or HF id for Qwen3-Embedding-8B when --semantic_encoder qwen3_peft.",
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
        help="Optional graph-only refinement applied before graph_detector_prepare GNN training.",
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
    parser.add_argument("--router_oof_mode", type=str, default="artifact", choices=["artifact"])
    parser.add_argument("--phase_a_semantic_sources", type=str, default="auto", help=argparse.SUPPRESS)
    parser.add_argument("--phase_a_feature_paths", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--phase_a_gnn_family", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--phase_a_fixed_gnn", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--phase_a_fixed_semantic_source", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--phase_a_latency_repeats", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--phase_a_include_structural_smoke", action="store_true", help=argparse.SUPPRESS)

    return normalize_args(parser.parse_args(raw_args), raw_args=raw_args)
