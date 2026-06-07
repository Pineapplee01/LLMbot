import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from LM import LM_Model
from GNNs import BotRGCN, GATv2Bot, HGT, RGCN, RGT, SimpleHGN
from estimators import build_estimator as _build_estimator
from operators import build_repair_operator as _build_repair_operator
from operators import build_semantic_operator as _build_semantic_operator
from subgroups import structural_features
from utils import safe_torch_load, tensor_sha256

_TOKENIZER_SOURCES = {
    "deberta": "microsoft/deberta-v3-base",
    "roberta-f": "yzxjb/roberta-finetuned-20",
    "roberta_finetuned": "yzxjb/roberta-finetuned-20",
    "roberta": "roberta-base",
    "bert": "bert-base-uncased",
    "twhin-bert": "Twitter/twhin-bert-base",
    "xlm-roberta": "xlm-roberta-base",
}

_QWEN_MODEL_NAMES = {"qwen_peft", "qwen3_peft"}
_DEFAULT_QWEN_MODEL_PATH = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-8B/"
    "snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
)
_EXTRA_SPECIAL_TOKENS = ["DESCRIPTION:", "METADATA:", "TWEET:"]
_EXTRA_TOKENS = ["@USER", "#HASHTAG", "HTTPURL", "EMOJI", "RT", "None"]

_GNN_BUILDERS = {
    "botrgcn": BotRGCN,
    "rgcn": RGCN,
    "rgt": RGT,
    "simplehgn": SimpleHGN,
    "hgt": HGT,
    "gatv2": GATv2Bot,
}


def _hf_cache_root():
    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get("TRANSFORMERS_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".cache" / "huggingface" / "hub"


def _looks_like_pretrained_dir(path):
    path = Path(path)
    if not path.is_dir():
        return False
    marker_files = (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
        "pytorch_model.bin",
    )
    return any((path / marker).exists() for marker in marker_files)


def _ordered_snapshot_dirs(cache_dir):
    snapshots_dir = Path(cache_dir) / "snapshots"
    if not snapshots_dir.is_dir():
        return []
    ordered = []
    main_ref = Path(cache_dir) / "refs" / "main"
    if main_ref.is_file():
        ref_name = main_ref.read_text(encoding="utf-8").strip()
        if ref_name:
            ordered.append(snapshots_dir / ref_name)
    ordered.extend(sorted(snapshots_dir.iterdir(), key=lambda item: item.name, reverse=True))
    result = []
    seen = set()
    for snapshot_dir in ordered:
        snapshot_dir = Path(snapshot_dir)
        key = str(snapshot_dir)
        if key in seen:
            continue
        seen.add(key)
        if _looks_like_pretrained_dir(snapshot_dir):
            result.append(snapshot_dir)
    return result


def _resolve_local_pretrained_source(model_source):
    text = str(model_source or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.exists():
        if _looks_like_pretrained_dir(candidate):
            return str(candidate)
        for snapshot_dir in _ordered_snapshot_dirs(candidate):
            return str(snapshot_dir)
        return None
    normalized = text.replace("\\", "/").strip("/")
    if "/" not in normalized:
        cache_dir = _hf_cache_root() / f"models--{normalized}"
    else:
        cache_dir = _hf_cache_root() / f"models--{normalized.replace('/', '--')}"
    for snapshot_dir in _ordered_snapshot_dirs(cache_dir):
        return str(snapshot_dir)
    return None


class PhaseAInputAdapter(nn.Module):
    """LoRA-style trainable residual adapter on top of frozen Phase A features."""

    def __init__(self, raw_dim, projected_dim, rank=8, alpha=16.0):
        super().__init__()
        rank = int(rank)
        if rank <= 0:
            raise ValueError("--peft_rank must be positive when --peft is enabled.")
        self.raw_dim = int(raw_dim)
        self.projected_dim = int(projected_dim)
        self.rank = rank
        self.alpha = float(alpha)
        self.lora_A = nn.Parameter(torch.zeros((self.rank, self.raw_dim)))
        self.lora_B = nn.Parameter(torch.zeros((self.projected_dim, self.rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def scale(self):
        return self.alpha / float(self.rank)

    def forward(self, projected_features, raw_features):
        residual = raw_features @ self.lora_A.T @ self.lora_B.T
        return projected_features + residual * self.scale


_MODE_METADATA = {
    "none": {
        "claim_role": "disabled_control",
        "scientific_gate": "phase_a_disabled_component",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    },
    "msp_ts": {
        "claim_role": "anchor_control",
        "scientific_gate": "phase0_required",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    },
    "graph_conformal_set_estimator": {
        "claim_role": "stage3_ego_quality_estimator",
        "stage2_role": "hard_node_quality_gate",
        "canonical_stage2": False,
        "scientific_gate": "calibration_split_labels_only",
        "paper_identity": "Graph Conformal Prediction-Set Ego Quality Estimator",
        "paper_identity_risk": "low_if_prediction_set_quality_gate_only",
        "promotion_rule": "stage3_candidate_after_ablation",
        "prediction_set_estimator": True,
        "diagnosis_or_action": False,
    },
    "gnn_2hop_conformal": {
        "claim_role": "stage3_ego_quality_estimator",
        "stage2_role": "gnn_side_2hop_hard_node_quality_gate",
        "canonical_stage2": False,
        "scientific_gate": "valid_tune_then_valid_cal_labels_only",
        "paper_identity": "GNN-side 2-hop Conformal Ego Quality Estimator",
        "paper_identity_risk": "low_if_prediction_set_quality_gate_only",
        "promotion_rule": "main_gnn_side_estimator_after_ablation",
        "prediction_set_estimator": True,
        "diagnosis_or_action": False,
        "lm_gnn_disagreement_used": False,
        "input_boundary": "gnn_posterior_optional_gnn_or_simteg_embedding_original_graph_only",
        "snaps_reference": "row-normalized k-hop aggregation; no LM-GNN disagreement in main path",
    },
    "login_uncertainty_router": {
        "claim_role": "stage_b_login_official_uncertainty_router",
        "stage2_role": "hard_node_selector",
        "canonical_stage2": False,
        "scientific_gate": "phase_a_frozen_gnn_uncertainty_only",
        "paper_identity": "LOGIN_init Node-Selection Uncertainty Router",
        "paper_identity_risk": "paper_aligned_uncertainty_module_only",
        "promotion_rule": "ablation_only_never_canonical",
        "screening_only": True,
        "diagnosis_or_action": False,
        "llm_call": False,
        "login_scope": "node_selection_uncertainty_only",
        "official_code_verified": False,
        "paper_aligned": True,
        "repo_locally_verified": False,
        "verified_scope": "node_selection_uncertainty_only",
        "not_full_LOGIN_reproduction": True,
    },
    "gats_faithful": {
        "claim_role": "faithful_calibration_anchor",
        "scientific_gate": "phase0_required",
        "paper_identity_risk": "low",
        "promotion_rule": "gate_only",
    },
    "off": {
        "claim_role": "anchor_control",
        "scientific_gate": "phase0_required",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    },
    "ridge_local": {
        "claim_role": "candidate_mainline",
        "scientific_gate": "sparse_only",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    },
    "lagnn_local": {
        "claim_role": "candidate_mainline",
        "scientific_gate": "sparse_only",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    },
    "noop": {
        "claim_role": "anchor_control",
        "scientific_gate": "phase0_required",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    },
    "prune": {
        "claim_role": "candidate_mainline",
        "scientific_gate": "propcorr_only",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    },
    "disagreement_local": {
        "claim_role": "candidate_mainline",
        "scientific_gate": "propcorr_only",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    },
    "ego_refinement": {
        "claim_role": "stage3_candidate_mainline",
        "scientific_gate": "conformal_quality_guided_local_ego_refinement",
        "paper_identity_risk": "medium_combination_innovation",
        "promotion_rule": "stage3_candidate_after_ablation",
    },
    "rule": {
        "claim_role": "anchor_control",
        "scientific_gate": "best_single_action_frozen",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    },
    "risk_only": {
        "claim_role": "supporting_baseline",
        "scientific_gate": "best_single_action_frozen",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    },
    "risk_regime": {
        "claim_role": "supporting_baseline",
        "scientific_gate": "best_single_action_frozen",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    },
    "gain_margin": {
        "claim_role": "candidate_mainline",
        "scientific_gate": "best_single_action_frozen",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    },
    "binary": {
        "claim_role": "supporting_baseline",
        "scientific_gate": "best_single_action_frozen",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    },
    "three_action": {
        "claim_role": "candidate_mainline",
        "scientific_gate": "best_single_action_frozen",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    },
    "swap": {
        "claim_role": "boundary_baseline",
        "scientific_gate": "best_single_action_frozen",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    },
    "glance_boundary": {
        "claim_role": "main_paper_boundary_baseline",
        "scientific_gate": "supporting_mainline",
        "paper_identity_risk": "occupied_claim_space",
        "promotion_rule": "never",
    },
    "ib_edl": {
        "claim_role": "supporting_mainline_calibration",
        "scientific_gate": "supporting_mainline",
        "paper_identity_risk": "calibration_only",
        "promotion_rule": "supporting_table_only",
    },
    "cs": {
        "claim_role": "conditional_supporting_baseline",
        "scientific_gate": "supporting_mainline",
        "paper_identity_risk": "low",
        "promotion_rule": "if_main_gain_modest",
    },
    "qwen_frozen": {
        "claim_role": "supporting_mainline",
        "scientific_gate": "supporting_mainline",
        "paper_identity_risk": "low",
        "promotion_rule": "semantic_lane_only",
    },
    "qwen_peft": {
        "claim_role": "supporting_mainline",
        "scientific_gate": "supporting_mainline",
        "paper_identity_risk": "low",
        "promotion_rule": "conditional_on_qwen_signal",
    },
}


def mode_metadata(mode):
    return dict(_MODE_METADATA.get(mode, {}))


def _print_model_info(label, model, include_repr=False):
    print(f"Information about {label} model:")
    if include_repr:
        print(model)
    print("total params:", sum(param.numel() for param in model.parameters()))


def _build_tokenizer(model_name, model_config):
    if model_name in _QWEN_MODEL_NAMES:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.get("qwen_model_path", _DEFAULT_QWEN_MODEL_PATH),
            trust_remote_code=bool(model_config.get("qwen_trust_remote_code", False)),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    tokenizer_source = _TOKENIZER_SOURCES.get(model_name)
    if tokenizer_source is None:
        raise ValueError(f"Unsupported LM model: {model_config['lm_model']}")
    resolved_source = model_config.get("pretrained_model_source") or tokenizer_source
    kwargs = {"local_files_only": True} if model_config.get("pretrained_model_source") else {}
    return AutoTokenizer.from_pretrained(resolved_source, **kwargs)


def _resize_tokenizer_if_needed(model_name, tokenizer, lm_model):
    if model_name == "roberta-f" or model_name in _QWEN_MODEL_NAMES:
        return
    tokenizer.add_special_tokens({"additional_special_tokens": _EXTRA_SPECIAL_TOKENS})
    tokenizer.add_tokens(_EXTRA_TOKENS)
    lm_model.LM.resize_token_embeddings(len(tokenizer))


def _with_metadata(component, mode):
    component.metadata = mode_metadata(mode)
    return component


def _labels_to_index(labels):
    labels = labels.detach().cpu() if torch.is_tensor(labels) else torch.tensor(labels)
    if labels.dim() > 1:
        return labels.argmax(dim=1).long()
    return labels.long()


def _idx_tensor(idx):
    return idx.detach().cpu().long() if torch.is_tensor(idx) else torch.tensor(idx, dtype=torch.long)


def _score_logits(logits, labels, idx):
    idx = _idx_tensor(idx)
    if idx.numel() == 0:
        return {"loss": float("inf"), "accuracy": 0.0, "macro_f1": 0.0}
    logits_idx = logits[idx]
    labels_idx = labels[idx]
    loss = torch.nn.functional.cross_entropy(logits_idx, labels_idx).item()
    pred = logits_idx.argmax(dim=1).detach().cpu().numpy()
    y_true = labels_idx.detach().cpu().numpy()
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


def _fallback_structural_node_features(data):
    labels = _labels_to_index(data["labels"])
    graph_node_count = int(data.get("graph_node_count", int(labels.numel())))
    features = structural_features(data["edge_index"], data["edge_type"], graph_node_count)
    table = np.column_stack(
        [
            features["in_degree"],
            features["out_degree"],
            features["total_degree"],
            features["relation_skew"],
        ]
    ).astype(np.float32)
    return torch.from_numpy(table), {"source": "structural_fallback", "path": None}


def _load_tensor_features(feature_path):
    loaded = safe_torch_load(feature_path, map_location="cpu")
    if isinstance(loaded, dict):
        for key in ("embeddings", "features", "x"):
            if key in loaded:
                loaded = loaded[key]
                break
    if not torch.is_tensor(loaded):
        loaded = torch.tensor(loaded)
    features = loaded.float().cpu()
    if features.dim() != 2:
        raise ValueError(f"G0 feature tensor must be 2D [num_nodes, dim], got shape {tuple(features.shape)}.")
    return features


def _resolve_support_embedding_path(args, data):
    support_path = getattr(args, "support_embedding_path", None) or "support_roberta_embeddings_new.pt"
    support_path = Path(support_path)
    if not support_path.is_absolute():
        support_path = Path(data.get("dataset_path", ".")) / support_path
    return support_path


def resolve_seed_aware_roberta_embedding_path(data, seed=None):
    dataset_path = Path(data.get("dataset_path", "."))
    candidates = []
    if seed is not None:
        candidates.append(dataset_path / f"embeddings_iter_-1_seed_{int(seed)}.pt")
    candidates.append(dataset_path / "embeddings_roberta.pt")
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _runtime_concat_full_graph_support_embeddings(args, data, labeled_feature_path):
    labeled_feature_path = Path(labeled_feature_path)
    labeled_features = _load_tensor_features(labeled_feature_path)
    labeled_node_count = int(data.get("labeled_node_count", 0))
    graph_node_count = int(data.get("graph_node_count", labeled_node_count))
    support_idx = data.get("support_idx")
    if labeled_node_count <= 0 or graph_node_count <= labeled_node_count or support_idx is None:
        raise ValueError(
            "full_graph_support requires labeled_node_count, graph_node_count, and support_idx from load_raw_data()."
        )
    if int(labeled_features.shape[0]) != labeled_node_count:
        raise ValueError(
            "full_graph_support expects --embedding_path to point to a labeled-node embedding tensor "
            f"with {labeled_node_count} rows, got {int(labeled_features.shape[0])}."
        )
    support_feature_path = _resolve_support_embedding_path(args, data)
    if not support_feature_path.exists():
        raise FileNotFoundError(
            "full_graph_support could not find the support embedding tensor at "
            f"{support_feature_path}."
        )
    support_features = _load_tensor_features(support_feature_path)
    support_count = int(len(support_idx))
    if int(support_features.shape[0]) != support_count:
        raise ValueError(
            "full_graph_support support embedding rows must match len(support_idx): "
            f"{int(support_features.shape[0])} vs {support_count}."
        )
    if int(support_features.shape[1]) != int(labeled_features.shape[1]):
        raise ValueError(
            "full_graph_support requires labeled and support embeddings to have the same feature dimension: "
            f"{int(labeled_features.shape[1])} vs {int(support_features.shape[1])}."
        )
    full_features = torch.cat([labeled_features, support_features], dim=0).contiguous()
    if int(full_features.shape[0]) != graph_node_count:
        raise ValueError(
            "full_graph_support runtime concatenation produced the wrong graph node count: "
            f"{int(full_features.shape[0])} vs expected {graph_node_count}."
        )
    return {
        "features": full_features,
        "labeled_features": labeled_features,
        "support_features": support_features,
        "labeled_feature_path": labeled_feature_path,
        "support_feature_path": support_feature_path,
    }


def phase_a_project_dim(args):
    return int(getattr(args, "phase_a_project_dim", 768))


def phase_a_projector(args):
    return str(getattr(args, "phase_a_projector", "pca")).lower()


def _project_semantic_features(features, args):
    target_dim = phase_a_project_dim(args)
    projector = phase_a_projector(args)
    if target_dim <= 0:
        raise ValueError("--phase_a_project_dim must be positive.")
    if projector != "pca":
        raise ValueError(f"Unsupported --phase_a_projector '{projector}'.")

    raw_dim = int(features.shape[1])
    manifest = {
        "raw_dim": raw_dim,
        "projected_dim": target_dim,
        "projector": projector,
        "fit_scope": "all_nodes_unlabeled",
        "raw_sha256": tensor_sha256(features),
        "projection_time_seconds": 0.0,
    }
    projector_state = {
        "projector": "identity",
        "target_dim": target_dim,
    }

    if raw_dim < target_dim:
        raise ValueError(
            f"Semantic embedding dim ({raw_dim}) is smaller than --phase_a_project_dim ({target_dim}). "
            "Phase A does not implicitly pad embeddings because that would change the comparison boundary."
        )

    if raw_dim == target_dim:
        projected = features.contiguous()
        manifest.update(
            {
                "projection_applied": False,
                "projected_sha256": tensor_sha256(projected),
            }
        )
        return projected, manifest, projector_state

    if int(features.shape[0]) < target_dim:
        raise ValueError(
            f"PCA projection to {target_dim}d requires at least {target_dim} nodes; "
            f"got {int(features.shape[0])}."
        )

    start = perf_counter()
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    _, _, components = torch.pca_lowrank(centered, q=target_dim, center=False)
    components = components[:, :target_dim].contiguous()
    projected = torch.matmul(centered, components).contiguous()
    manifest.update(
        {
            "projection_applied": True,
            "projection_time_seconds": float(perf_counter() - start),
            "projected_sha256": tensor_sha256(projected),
        }
    )
    projector_state = {
        "projector": projector,
        "target_dim": target_dim,
        "fit_scope": "all_nodes_unlabeled",
        "mean": mean.squeeze(0).contiguous(),
        "components": components,
    }
    return projected, manifest, projector_state


def resolve_g0_feature_bundle(args, data):
    feature_path = getattr(args, "emb_path", None) or getattr(args, "g0_feature_path", None)
    graph_data_variant = str(data.get("graph_data_variant", getattr(args, "graph_data_variant", "labeled"))).lower()
    if feature_path:
        feature_path = Path(feature_path)
    else:
        dataset_path = Path(data.get("dataset_path", "."))
        semantic_encoder = str(
            getattr(args, "semantic_encoder", getattr(args, "semantic_backbone", "auto"))
        ).lower()
        active_seed = getattr(args, "active_seed", None)
        if semantic_encoder in {"auto", "roberta", "roberta_finetuned"}:
            feature_path = resolve_seed_aware_roberta_embedding_path(data, seed=active_seed)
        else:
            candidates = [dataset_path / "qwen3_emb_last.pt", dataset_path / "embeddings_roberta.pt"]
            feature_path = next((candidate for candidate in candidates if candidate.exists()), None)

    if graph_data_variant == "full_graph_support" and feature_path is None:
        raise ValueError(
            "full_graph_support requires --embedding_path to point to the labeled RoBERTa embedding tensor."
        )

    if feature_path is None:
        features, manifest = _fallback_structural_node_features(data)
        manifest["raw_dim"] = int(features.shape[1])
        manifest["projected_dim"] = int(features.shape[1])
        manifest["projector"] = "structural_fallback"
        manifest["projection_applied"] = False
        manifest["raw_sha256"] = tensor_sha256(features)
        manifest["projected_sha256"] = tensor_sha256(features)
        manifest["projection_time_seconds"] = 0.0
        manifest["peft"] = {
            "enabled": bool(getattr(args, "peft", False)),
            "rank": int(getattr(args, "peft_rank", 8)),
            "alpha": float(getattr(args, "peft_alpha", 16.0)),
            "trainable_parameter_count": 0,
        }
        return {
            "features": features,
            "raw_features": features,
            "feature_manifest": manifest,
            "projector_state": {"projector": "structural_fallback"},
        }

    if graph_data_variant == "full_graph_support":
        concat_bundle = _runtime_concat_full_graph_support_embeddings(args, data, feature_path)
        raw_features = concat_bundle["features"]
    else:
        raw_features = _load_tensor_features(feature_path)
    projected, projection_manifest, projector_state = _project_semantic_features(raw_features, args)
    feature_manifest = {
        "source": "tensor_file",
        "path": str(feature_path),
        "sha256": projection_manifest["projected_sha256"],
        **projection_manifest,
        "graph_data_variant": graph_data_variant,
        "peft": {
            "enabled": bool(getattr(args, "peft", False)),
            "rank": int(getattr(args, "peft_rank", 8)),
            "alpha": float(getattr(args, "peft_alpha", 16.0)),
            "trainable_parameter_count": 0,
        },
    }
    if graph_data_variant == "full_graph_support":
        feature_manifest.update(
            {
                "runtime_concat": True,
                "labeled_embedding_path": str(concat_bundle["labeled_feature_path"]),
                "support_embedding_path": str(concat_bundle["support_feature_path"]),
                "graph_node_count": int(raw_features.shape[0]),
                "labeled_node_count": int(concat_bundle["labeled_features"].shape[0]),
                "support_node_count": int(concat_bundle["support_features"].shape[0]),
            }
        )
    return {
        "features": projected,
        "raw_features": raw_features,
        "feature_manifest": feature_manifest,
        "projector_state": projector_state,
    }


def resolve_g0_node_features(args, data):
    bundle = resolve_g0_feature_bundle(args, data)
    return bundle["features"], bundle["feature_manifest"]


def build_LM_model(model_config):
    model_name = model_config["lm_model"].lower()
    if model_name not in _QWEN_MODEL_NAMES:
        tokenizer_source = _TOKENIZER_SOURCES.get(model_name)
        if tokenizer_source:
            resolved_source = _resolve_local_pretrained_source(tokenizer_source)
            if resolved_source:
                model_config = dict(model_config)
                model_config["pretrained_model_source"] = resolved_source
    lm_model = LM_Model(model_config).to(model_config["device"])
    tokenizer = _build_tokenizer(model_name, model_config)
    _resize_tokenizer_if_needed(model_name, tokenizer, lm_model)
    _print_model_info("LM", lm_model)
    return lm_model, tokenizer


def build_GNN_model(model_config):
    model_name = model_config["GNN_model"].lower()
    builder = _GNN_BUILDERS.get(model_name)
    if builder is None:
        raise ValueError(f"Unknown GNN model '{model_name}'")

    gnn_model = builder(model_config).to(model_config["device"])
    _print_model_info("GNN", gnn_model, include_repr=True)
    return gnn_model


def build_estimator(mode):
    return _with_metadata(_build_estimator(mode), mode)


def build_semantic_operator(mode):
    return _with_metadata(_build_semantic_operator(mode), mode)


def build_repair_operator(mode):
    return _with_metadata(_build_repair_operator(mode), mode)


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _risk_budget_mask(risk_score, budget):
    n_items = len(risk_score)
    k_items = max(int(n_items * budget), 1)
    mask = np.zeros(n_items, dtype=bool)
    top_idx = np.argsort(risk_score)[-k_items:]
    mask[top_idx] = True
    return mask


class BaseSelector:
    def __init__(self, budget=0.15):
        self.budget = budget

    def fit(self, *args, **kwargs):
        return self

    def select(self, **kwargs):
        raise NotImplementedError


class RuleSelector(BaseSelector):
    def select(self, risk_score, sparse_mask=None, prop_mask=None, semantic_action=1, repair_action=2, **kwargs):
        risk_score = _to_numpy(risk_score)
        sparse_mask = _to_numpy(sparse_mask).astype(bool) if sparse_mask is not None else np.zeros_like(risk_score, dtype=bool)
        prop_mask = _to_numpy(prop_mask).astype(bool) if prop_mask is not None else np.zeros_like(risk_score, dtype=bool)
        intervene = _risk_budget_mask(risk_score, self.budget)
        actions = np.zeros_like(risk_score, dtype=np.int32)
        actions[intervene & sparse_mask] = semantic_action
        actions[intervene & ~sparse_mask & prop_mask] = repair_action
        return actions


class RiskOnlySelector(BaseSelector):
    def select(self, risk_score, best_single_action=1, **kwargs):
        risk_score = _to_numpy(risk_score)
        intervene = _risk_budget_mask(risk_score, self.budget)
        actions = np.zeros_like(risk_score, dtype=np.int32)
        actions[intervene] = best_single_action
        return actions


class RiskRegimeSelector(BaseSelector):
    def select(self, risk_score, sparse_mask=None, prop_mask=None, semantic_action=1, repair_action=2, **kwargs):
        return RuleSelector(budget=self.budget).select(
            risk_score=risk_score,
            sparse_mask=sparse_mask,
            prop_mask=prop_mask,
            semantic_action=semantic_action,
            repair_action=repair_action,
        )


class BinarySelector(BaseSelector):
    def select(self, risk_score, best_single_action=1, **kwargs):
        return RiskOnlySelector(budget=self.budget).select(
            risk_score=risk_score,
            best_single_action=best_single_action,
        )


class ThreeActionSelector(BaseSelector):
    def select(self, risk_score, sparse_mask=None, prop_mask=None, semantic_action=1, repair_action=2, **kwargs):
        return RuleSelector(budget=self.budget).select(
            risk_score=risk_score,
            sparse_mask=sparse_mask,
            prop_mask=prop_mask,
            semantic_action=semantic_action,
            repair_action=repair_action,
        )


class SwapSelector(BaseSelector):
    def select(self, base_actions, **kwargs):
        actions = _to_numpy(base_actions).astype(np.int32).copy()
        swapped = actions.copy()
        swapped[actions == 1] = 2
        swapped[actions == 2] = 1
        return swapped


class GainMarginSelector(BaseSelector):
    def __init__(self, budget=0.15, clip_value=2.0):
        super().__init__(budget=budget)
        self.clip_value = clip_value
        self.model = None

    def fit(self, train_features, train_gain, train_fix_broke=None):
        train_features = _to_numpy(train_features)
        train_gain = np.clip(_to_numpy(train_gain), -self.clip_value, self.clip_value)
        labels = (train_gain > 0).astype(np.int32)
        if labels.min() == labels.max():
            self.model = None
            return self
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.model.fit(train_features, labels)
        return self

    def select(self, train_features=None, test_features=None, fallback_risk=None, best_single_action=1, **kwargs):
        if self.model is None or test_features is None:
            return RiskOnlySelector(budget=self.budget).select(
                risk_score=fallback_risk,
                best_single_action=best_single_action,
            )
        probs = self.model.predict_proba(_to_numpy(test_features))[:, 1]
        return RiskOnlySelector(budget=self.budget).select(
            risk_score=probs,
            best_single_action=best_single_action,
        )


def build_selector(mode, budget=0.15, gain_clip_value=2.0):
    if mode == "rule":
        return _with_metadata(RuleSelector(budget=budget), mode)
    if mode == "risk_only":
        return _with_metadata(RiskOnlySelector(budget=budget), mode)
    if mode == "risk_regime":
        return _with_metadata(RiskRegimeSelector(budget=budget), mode)
    if mode == "gain_margin":
        return _with_metadata(GainMarginSelector(budget=budget, clip_value=gain_clip_value), mode)
    if mode == "binary":
        return _with_metadata(BinarySelector(budget=budget), mode)
    if mode == "three_action":
        return _with_metadata(ThreeActionSelector(budget=budget), mode)
    if mode == "swap":
        return _with_metadata(SwapSelector(budget=budget), mode)
    raise ValueError(f"Unknown selector mode: {mode}")


_DEFAULT_GNN_CFG = {
    "gnn_n_layers": 2,
    "n_layers": 2,
    "n_relations": 2,
    "activation": "leakyrelu",
    "dropout": 0.4,
    "gnn_hidden_dim": 128,
    "hidden_dim": 128,
    "lm_input_dim": 768,
    "att_heads": 8,
    "RGT_semantic_heads": 8,
    "GNN_model": "rgcn",
}


def load_gnn_checkpoint(ckpt_path, device, default_cfg=None, claim_grade=False):
    ckpt = safe_torch_load(ckpt_path, map_location=device)
    if "model_config" in ckpt:
        cfg = dict(ckpt["model_config"])
        cfg["device"] = device
    else:
        if claim_grade:
            raise ValueError(
                f"Claim-grade GNN checkpoint requires model_config; missing in {ckpt_path}."
            )
        cfg = dict(default_cfg if default_cfg is not None else _DEFAULT_GNN_CFG)
        cfg["device"] = device
    model = build_GNN_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg
