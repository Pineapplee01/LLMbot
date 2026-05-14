import argparse


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def normalize_args(args):
    """Normalize deprecated aliases without removing compatibility flags."""
    default_risk_budgets = "0.10,0.20,0.30,0.40"

    if getattr(args, "emb_path", None):
        args.g0_feature_path = args.emb_path
    elif getattr(args, "g0_feature_path", None):
        args.emb_path = args.g0_feature_path

    if getattr(args, "risk_variant", None) is None:
        args.risk_variant = getattr(args, "router_variant", None)

    if getattr(args, "risk_budgets", None):
        args.router_budgets = args.risk_budgets
    elif getattr(args, "router_budgets", None):
        args.risk_budgets = args.router_budgets
    else:
        args.risk_budgets = default_risk_budgets
        args.router_budgets = default_risk_budgets

    args.reset_split = str(args.reset_split).strip()
    return args


def parser_args():
    parser = argparse.ArgumentParser()

    # Core routing
    parser.add_argument(
        "--stage",
        type=str,
        default="legacy_distill",
        choices=[
            "frozen_g0",
            "frozen_gats",
            "semantic_finetune",
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
            "eqc_v8_matrix",
        ],
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
    parser.add_argument("--batch_size_LM", type=int, default=32)
    parser.add_argument("--batch_size_GNN", type=int, default=300000)
    parser.add_argument("--batch_size_MLP", type=int, default=300000)
    parser.add_argument("--raw_data_filepath", type=str, default="./data/raw/")
    parser.add_argument("--is_processed", type=_parse_bool, default=True)
    parser.add_argument("--reset_split", type=str, default="-1")

    # Stage routing
    parser.add_argument(
        "--GNN_model",
        type=str,
        default="botrgcn",
        choices=["botrgcn", "rgcn", "rgt", "hgt", "simplehgn", "gatv2"],
    )
    parser.add_argument(
        "--estimator_mode",
        type=str,
        default="none",
        choices=[
            "none",
            "msp_ts",
            "final_output_ranker",
            "dual_posterior_ranker",
            "login_uncertainty_router",
            "graph_conformal_set_estimator",
            "gnn_2hop_conformal",
            "eqc_v8_stage1",
            "cagcn_proxy",
            "gats_proxy",
            "calibrated_residual_risk_selector",
            "glance_for_context_residual_risk_selector",
            "calibrated_multiview_router",
        ],
    )
    parser.add_argument(
        "--semantic_mode",
        type=str,
        default="off",
        choices=["off", "ridge_local", "lagnn_local"],
    )
    parser.add_argument(
        "--repair_mode",
        type=str,
        default="noop",
        choices=["noop", "prune", "disagreement_local", "ego_refinement", "eqc_v8_stage4"],
    )
    parser.add_argument(
        "--selector_mode",
        type=str,
        default="none",
        choices=["none", "rule", "risk_only", "risk_regime", "gain_margin", "binary", "three_action", "swap"],
    )
    parser.add_argument(
        "--appendix_mode",
        type=str,
        default="none",
        choices=["none", "qwen_frozen", "qwen_peft", "ib_edl", "glance_boundary", "cs"],
    )

    # LM configuration
    parser.add_argument("--LM_model", type=str, default="roberta")
    parser.add_argument(
        "--semantic_backbone",
        type=str,
        default="auto",
        choices=["auto", "roberta", "roberta_finetuned", "qwen3_frozen", "qwen3_peft"],
        help="Semantic encoder source for semantic finetune and Phase A embedding resolution.",
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
    parser.add_argument("--optimizer_LM", type=str, default="adamw")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--LM_classifier_n_layers", type=int, default=2)
    parser.add_argument("--LM_classifier_hidden_dim", type=int, default=128)
    parser.add_argument("--LM_dropout", type=float, default=0.1)
    parser.add_argument("--LM_att_dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing_factor", type=float, default=0.0)
    parser.add_argument("--warmup", type=float, default=0.6)

    # GNN configuration
    parser.add_argument("--use_GNN", action="store_true")
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_relations", type=int, default=2)
    parser.add_argument("--activation", type=str, default="leakyrelu")
    parser.add_argument("--optimizer_GNN", type=str, default="adamw")
    parser.add_argument("--GNN_dropout", type=float, default=0.4)
    parser.add_argument("--att_heads", type=int, default=8)
    parser.add_argument("--SimpleHGN_att_res", type=float, default=0.2)
    parser.add_argument("--RGT_semantic_heads", type=int, default=8)

    # MLP configuration
    parser.add_argument("--MLP_n_layers", type=int, default=3)
    parser.add_argument("--MLP_hidden_dim", type=int, default=128)
    parser.add_argument("--optimizer_MLP", type=str, default="adamw")
    parser.add_argument("--MLP_dropout", type=float, default=0.4)

    # Training and evaluation
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--LM_pretrain_epochs", type=float, default=5)
    parser.add_argument(
        "--semantic_max_steps",
        type=int,
        default=0,
        help="Optional step cap for --stage semantic_finetune; 0 uses one pass over the selected train subset.",
    )
    parser.add_argument(
        "--semantic_train_limit",
        type=int,
        default=0,
        help="Optional train_idx cap for --stage semantic_finetune smoke runs; 0 uses the full train split.",
    )
    parser.add_argument("--MLP_KD_epochs", type=int, default=300)
    parser.add_argument("--LM_eval_patience", type=int, default=20)
    parser.add_argument("--LM_accumulation", type=int, default=1)
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument("--GNN_epochs_per_iter", type=int, default=200)
    parser.add_argument("--LM_epochs_per_iter", type=int, default=3)
    parser.add_argument("--MLP_epochs_per_iter", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=3)
    parser.add_argument("--pl_ratio_LM", type=float, default=0.5)
    parser.add_argument("--pl_ratio_GNN", type=float, default=0.0)
    parser.add_argument("--pl_ratio_MLP", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.7)

    parser.add_argument("--lr_LM", type=float, default=1e-5)
    parser.add_argument("--weight_decay_LM", type=float, default=0.01)
    parser.add_argument("--lr_GNN", type=float, default=5e-4)
    parser.add_argument("--weight_decay_GNN", type=float, default=1e-5)
    parser.add_argument("--lr_MLP", type=float, default=5e-4)
    parser.add_argument("--weight_decay_MLP", type=float, default=1e-5)

    # Module-specific controls
    parser.add_argument("--selector_budget", type=float, default=0.15)
    parser.add_argument("--sparse_degree_quantile", type=float, default=0.25)
    parser.add_argument("--propagation_quantile", type=float, default=0.85)
    parser.add_argument("--gain_clip_value", type=float, default=2.0)
    parser.add_argument("--bootstrap_samples", type=int, default=512)
    parser.add_argument(
        "--router_mode",
        type=str,
        default="off",
        choices=["off", "multiview"],
        help="Stage 2 residual-risk router mode; multiview enables the calibrated multi-view router.",
    )
    parser.add_argument(
        "--router_variant",
        type=str,
        default="router_4_view_ensemble",
        choices=["router_1_pred", "router_2_disagreement", "router_3_graph_context", "router_4_view_ensemble"],
        help="Deprecated alias for --risk_variant; preserved for legacy router ablations.",
    )
    parser.add_argument(
        "--risk_variant",
        type=str,
        default=None,
        choices=[
            "prediction_only",
            "dual_posterior_disagreement",
            "gets_lite_graph_context",
            "gets_posthoc",
            "glance_for_context",
            "view_ensemble",
            "router_1_pred",
            "router_2_disagreement",
            "router_3_graph_context",
            "router_4_view_ensemble",
        ],
        help="Stage 2 residual-risk selector feature family. Defaults to legacy --router_variant when omitted.",
    )
    parser.add_argument("--router_oof_folds", type=int, default=5)
    parser.add_argument(
        "--router_oof_mode",
        type=str,
        default="artifact",
        choices=["artifact"],
        help="OOF protocol source for router training; artifact mode requires frozen OOF outputs.",
    )
    parser.add_argument("--router_lambda_rank", type=float, default=0.2)
    parser.add_argument("--router_rank_margin", type=float, default=0.1)
    parser.add_argument(
        "--glance_training_objective",
        type=str,
        default="residual_error",
        choices=["residual_error", "glance_advantage"],
        help="GLANCE-inspired selector objective. glance_advantage requires explicit GNN-vs-LLM counterfactual supervision.",
    )
    parser.add_argument("--glance_llm_query_cost", type=float, default=0.2)
    parser.add_argument("--risk_budgets", type=str, default=None)
    parser.add_argument("--router_budgets", type=str, default=None, help="Deprecated alias for --risk_budgets.")
    parser.add_argument(
        "--emb_path",
        type=str,
        default=None,
        help="Cached semantic embedding tensor for Phase A / frozen G0 graph detection.",
    )
    parser.add_argument(
        "--g0_feature_path",
        type=str,
        default=None,
        help="Compatibility alias for --emb_path; prefer --emb_path for new Phase A runs.",
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
    parser.add_argument("--g0_epochs", type=int, default=0)
    parser.add_argument("--gats_max_iter", type=int, default=300)
    parser.add_argument(
        "--phase_a_semantic_sources",
        type=str,
        default="auto",
        help=(
            "Comma-separated Phase A semantic sources. Use auto, roberta_finetuned, "
            "qwen3_frozen, custom_g0_feature, or structural_smoke."
        ),
    )
    parser.add_argument(
        "--phase_a_feature_paths",
        type=str,
        default=None,
        help=(
            "Comma-separated name=path mapping for Phase A embeddings. "
            "Paths may include {seed}, e.g. roberta_finetuned=/runs/seed_{seed}/emb.pt."
        ),
    )
    parser.add_argument(
        "--phase_a_gnn_family",
        type=str,
        default=None,
        help="Comma-separated GNN family for Phase A; defaults to --GNN_model.",
    )
    parser.add_argument(
        "--phase_a_fixed_gnn",
        type=str,
        default=None,
        help="Fixed GNN used to summarize the semantic-encoder family block.",
    )
    parser.add_argument(
        "--phase_a_fixed_semantic_source",
        type=str,
        default=None,
        help="Fixed semantic source used to summarize the GNN-family block.",
    )
    parser.add_argument("--phase_a_latency_repeats", type=int, default=3)
    parser.add_argument(
        "--phase_a_include_structural_smoke",
        action="store_true",
        help="Allow non-semantic structural features for engineering smoke only.",
    )

    # v8 EQC pipeline (see refine-logs/FINAL_PROPOSAL.md for the 7-stage skeleton).
    parser.add_argument(
        "--eqc_v8_block",
        type=str,
        default="sanity",
        choices=["sanity", "block_e", "block_p", "block_r", "block_l", "block_ct"],
        help=(
            "Selects the v8 EQC experimental block when --stage eqc_v8_matrix. "
            "'sanity' runs a one-seed smoke through the full Stage-1..Stage-6 plumbing; "
            "'block_e' runs the seven-variant Stage-6 encoder falsification grid; "
            "'block_p' runs the leave-one-stage-out necessity audit; "
            "'block_r' runs the retrieval iteration budget-matched comparison; "
            "'block_l' compares the Stage-6 linear probe against a two-layer MLP; "
            "'block_ct' runs the cross-dataset calibration transfer experiment."
        ),
    )
    parser.add_argument(
        "--eqc_v8_llm_backbone",
        type=str,
        default="qwen",
        choices=["qwen", "mistral", "none"],
        help="Stage-6 LLM backbone for eqc_v8 runs ('none' forces the E-null variant).",
    )
    parser.add_argument(
        "--eqc_v8_schema_mode",
        type=str,
        default="canonical",
        choices=["canonical", "shuffle_role", "shuffle_order", "collapse", "minimal"],
        help="Stage-5 evidence-graph scramble mode; canonical is the v1.0 schema.",
    )
    parser.add_argument(
        "--eqc_v8_token_budget",
        type=int,
        default=512,
        help="Pre-registered Stage-5 evidence-graph token budget (default: 512).",
    )
    parser.add_argument(
        "--eqc_v8_t_ret",
        type=int,
        default=2,
        help="Stage-3 retrieval iteration count T_ret (default: 2 as per FINAL_PROPOSAL).",
    )
    parser.add_argument(
        "--eqc_v8_k_struct",
        type=int,
        default=8,
        help="Stage-3 structural proposals per iteration before text confirmation.",
    )
    parser.add_argument(
        "--eqc_v8_epsilon_stop",
        type=float,
        default=1e-3,
        help="Stage-3 conformal stop tolerance epsilon on q(v) improvement.",
    )
    parser.add_argument(
        "--eqc_v8_single_pass_budget_matched",
        action="store_true",
        help="Stage-3 ablation: run a single-pass PPR with 2*k_struct candidates (Block-R control).",
    )
    parser.add_argument(
        "--eqc_v8_lambda_reweight",
        type=float,
        default=0.2,
        help="Stage-4 soft-reweight magnitude lambda (default: 0.2).",
    )
    parser.add_argument(
        "--eqc_v8_rho_stability_cap",
        type=float,
        default=0.2,
        help="Stage-4 accept-rule L-infinity posterior drift cap rho (default: 0.2).",
    )
    parser.add_argument(
        "--eqc_v8_calibrator_folds",
        type=int,
        default=5,
        help="CalibratedExpectedErrorProxy cross-fitting folds (default: 5).",
    )
    parser.add_argument(
        "--eqc_v8_output_dir",
        type=str,
        default=None,
        help="Per-seed artifact directory for v8 runs; falls back to --artifact_root when omitted.",
    )
    parser.add_argument(
        "--eqc_v8_hard_node_budget",
        type=float,
        default=0.15,
        help="Fraction of nodes routed to Stage-3..Stage-6 by the Stage-2 lex rule.",
    )
    parser.add_argument(
        "--eqc_v8_alpha",
        type=float,
        default=0.15,
        help="Stage-1 conformal quantile alpha (1-alpha coverage target).",
    )

    return normalize_args(parser.parse_args())
