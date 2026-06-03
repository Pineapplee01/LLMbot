import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from artifact_contracts import MissingFrozenArtifactError
from estimators import GlanceForContextResidualRiskSelector
from estimators import parse_budget_list
from model_building import _runtime_concat_full_graph_support_embeddings, resolve_seed_aware_roberta_embedding_path
from router import GlanceReliabilityRouterMLP, fit_reliability_temperature
from trainer_semantic import _classification_metrics_from_logits
from utils import read_json, safe_torch_load, save_stage_artifacts, tensor_sha256, write_csv_rows, write_json, write_text


def _infer_embedding_source_identity(path):
    path = Path(path)
    stem = path.stem.lower()
    name = path.name.lower()
    token = f"{stem} {name}"
    if "qwen" in token:
        return {
            "semantic_backbone": "qwen_cached",
            "lm_model": "qwen_cached",
            "source_identity": "qwen_cached_embedding",
        }
    if "roberta" in token:
        return {
            "semantic_backbone": "roberta_finetuned",
            "lm_model": "roberta-f",
            "source_identity": "roberta_finetuned_embedding",
        }
    return {
        "semantic_backbone": "external_embedding_unknown",
        "lm_model": "external_embedding_unknown",
        "source_identity": "external_embedding_unknown",
    }


def _direct_embedding_manifest(path):
    identity = _infer_embedding_source_identity(path)
    return {
        "contract": "direct_embedding_fallback_v2",
        "status": "completed_external_embedding",
        "embeddings_path": str(Path(path)),
        "source": "direct_embedding_path",
        **identity,
    }


def _idx_numpy(idx):
    if torch.is_tensor(idx):
        return idx.detach().cpu().long().numpy()
    return np.asarray(idx, dtype=np.int64)


def _score_all(labels, pred):
    labels_np = np.asarray(labels)
    pred_np = np.asarray(pred)
    return {
        "accuracy": float(accuracy_score(labels_np, pred_np)),
        "macro_f1": float(f1_score(labels_np, pred_np, average="macro", zero_division=0)),
        "bot_f1": float(f1_score(labels_np, pred_np, average="binary", zero_division=0)),
        "count": int(labels_np.shape[0]),
    }


def _delta_table(base_pred, new_pred, labels, idx_mask):
    labels_np = np.asarray(labels)
    base_pred = np.asarray(base_pred)
    new_pred = np.asarray(new_pred)
    affected = idx_mask.astype(bool)
    base_correct = base_pred == labels_np
    new_correct = new_pred == labels_np
    fix = int((~base_correct & new_correct & affected).sum())
    broke = int((base_correct & ~new_correct & affected).sum())
    return {
        "fix": fix,
        "broke": broke,
        "net": fix - broke,
        "touched": int(affected.sum()),
    }


def _labeled_prefix_pred(pred, labeled_count):
    pred_np = np.asarray(pred)
    return pred_np[: int(labeled_count)]


def _node_cross_entropy_vector(prob, labels):
    prob_t = prob.detach().cpu().float() if torch.is_tensor(prob) else torch.tensor(prob, dtype=torch.float32)
    labels_t = labels.detach().cpu().long() if torch.is_tensor(labels) else torch.tensor(labels, dtype=torch.long)
    row = torch.arange(labels_t.numel(), dtype=torch.long)
    return (-torch.log(prob_t[row, labels_t].clamp_min(1e-8))).cpu()


_PROMPT_EXPERT_COMPONENT_ORDER = ("ego", "graph_following", "graph_follower", "tweet", "conflict", "metadata_structured")
_GAUGLLM_SELECTOR_COMPONENT_ORDER = ("graph_following", "graph_follower", "tweet", "conflict")
_PROMPT_EXPERT_SEMANTIC_VIEW_MODES = {
    "prompt_expert_bundle_v1",
    "prompt_expert_bundle_center_induced_v1",
    "prompt_expert_bundle_v2",
}


def _is_prompt_expert_semantic_view_mode(value):
    return str(value or "").strip().lower() in _PROMPT_EXPERT_SEMANTIC_VIEW_MODES


def _is_prompt_expert_feature_kind(value):
    token = str(value or "").strip().lower()
    return token in {"prompt_expert_bundle", "prompt_expert_bundle_v1"} or token in _PROMPT_EXPERT_SEMANTIC_VIEW_MODES


def _selector_component_order_for_fusion(fusion_mode):
    token = str(fusion_mode or "projector_concat").strip().lower()
    if token == "gaugllm_selector":
        return _GAUGLLM_SELECTOR_COMPONENT_ORDER
    if token == "mpe_gated":
        return PromptExpertBundleRefinerMLP.MPE_COMPONENT_ORDER
    return _GAUGLLM_SELECTOR_COMPONENT_ORDER


def _adjacent_prompt_expert_manifest_path(cache_path):
    cache_path = Path(cache_path)
    return cache_path.with_name(f"{cache_path.stem}_manifest.json")


def _load_jsonl_last_row_by_node(sidecar_path, expected_component_name):
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.exists():
        raise MissingFrozenArtifactError(f"Missing explanation sidecar for {expected_component_name}: {sidecar_path}")
    rows_by_node = {}
    with sidecar_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise MissingFrozenArtifactError(
                    f"Invalid JSON in explanation sidecar {sidecar_path} line {line_no}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise MissingFrozenArtifactError(
                    f"Explanation sidecar {sidecar_path} line {line_no} must be a JSON object."
                )
            node_id = row.get("node_id")
            if node_id is None:
                raise MissingFrozenArtifactError(
                    f"Explanation sidecar {sidecar_path} line {line_no} is missing node_id."
                )
            component_name = str(row.get("component_name", expected_component_name) or expected_component_name).strip()
            if component_name != expected_component_name:
                raise MissingFrozenArtifactError(
                    f"Explanation sidecar {sidecar_path} line {line_no} declares component_name={component_name!r}, "
                    f"expected {expected_component_name!r}."
                )
            explanation = ""
            for key in ("explanation", "explanation_text", "generated_text", "response"):
                value = row.get(key)
                if value is not None and str(value).strip():
                    explanation = str(value).strip()
                    break
            if not explanation:
                raise MissingFrozenArtifactError(
                    f"Explanation sidecar {sidecar_path} line {line_no} does not contain usable explanation text."
                )
            rows_by_node[int(node_id)] = {
                "node_id": int(node_id),
                "component_name": component_name,
                "prompt_hash": str(row.get("prompt_hash", "") or ""),
                "prompt_role": str(row.get("prompt_role", "") or ""),
                "generation_mode": str(row.get("generation_mode", "") or ""),
                "explanation": explanation,
            }
    return rows_by_node


def _load_prompt_expert_selector_sidecars(cache_manifest_path, required_components, target_node_ids=None):
    cache_manifest_path = Path(cache_manifest_path)
    if not cache_manifest_path.exists():
        raise MissingFrozenArtifactError(
            f"gaugllm_selector requires adjacent cache manifest {cache_manifest_path}, but it does not exist."
        )
    cache_manifest = read_json(cache_manifest_path, default={}) or {}
    sidecar_paths = cache_manifest.get("component_explanation_sidecar_paths")
    if not isinstance(sidecar_paths, dict):
        raise MissingFrozenArtifactError(
            f"Adjacent cache manifest {cache_manifest_path} does not contain component_explanation_sidecar_paths."
        )
    target_node_set = None
    if target_node_ids is not None:
        target_node_ids = _coerce_long_tensor_1d(target_node_ids)
        if target_node_ids is not None:
            target_node_set = {int(item) for item in target_node_ids.tolist()}
    rows_by_component = {}
    resolved_paths = {}
    for component_name in required_components:
        sidecar_path = sidecar_paths.get(component_name)
        if not sidecar_path:
            raise MissingFrozenArtifactError(
                f"Adjacent cache manifest {cache_manifest_path} is missing the sidecar path for {component_name}."
            )
        resolved_path = Path(sidecar_path)
        rows = _load_jsonl_last_row_by_node(resolved_path, component_name)
        if target_node_set is not None:
            missing = sorted(target_node_set.difference(rows.keys()))
            if missing:
                preview = ", ".join(str(item) for item in missing[:10])
                raise MissingFrozenArtifactError(
                    f"gaugllm_selector sidecar {resolved_path} is missing {len(missing)} routed nodes for {component_name}. "
                    f"First missing ids: {preview}."
                )
        rows_by_component[component_name] = rows
        resolved_paths[component_name] = str(resolved_path)
    return cache_manifest, resolved_paths, rows_by_component


def _load_simteg_lm_checkpoint_into_encoder(model, checkpoint_path):
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise MissingFrozenArtifactError(
            f"SimTeG LM checkpoint at {checkpoint_path} does not contain a model state_dict."
        )
    lm_state = {
        key[len("LM.") :]: value
        for key, value in state_dict.items()
        if isinstance(key, str) and key.startswith("LM.")
    }
    if not lm_state:
        raise MissingFrozenArtifactError(
            f"SimTeG LM checkpoint at {checkpoint_path} does not contain LM.* encoder weights."
        )
    load_result = model.load_state_dict(lm_state, strict=False)
    return {
        "loaded_key_count": int(len(lm_state)),
        "missing_key_count": int(len(getattr(load_result, "missing_keys", []))),
        "unexpected_key_count": int(len(getattr(load_result, "unexpected_keys", []))),
        "checkpoint_path": str(checkpoint_path),
    }


def _load_runtime_selector_encoder(cache_manifest, device):
    from transformers import AutoModel, AutoTokenizer

    resolved_model_path = cache_manifest.get("resolved_model_path")
    checkpoint_path = cache_manifest.get("finetuned_roberta_checkpoint_path")
    if not resolved_model_path:
        raise MissingFrozenArtifactError(
            "gaugllm_selector requires resolved_model_path in the routed prompt-expert cache manifest."
        )
    if not checkpoint_path:
        raise MissingFrozenArtifactError(
            "gaugllm_selector requires finetuned_roberta_checkpoint_path in the routed prompt-expert cache manifest."
        )
    model_path = Path(str(resolved_model_path)).expanduser()
    checkpoint_path = Path(str(checkpoint_path)).expanduser()
    if not model_path.exists():
        raise MissingFrozenArtifactError(
            f"gaugllm_selector could not resolve the finetuned RoBERTa source directory: {model_path}"
        )
    if not checkpoint_path.exists():
        raise MissingFrozenArtifactError(
            f"gaugllm_selector could not resolve the SimTeG LM checkpoint: {checkpoint_path}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    checkpoint_summary = _load_simteg_lm_checkpoint_into_encoder(model, checkpoint_path)
    model = model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer, model, checkpoint_summary


def _encode_simteg_texts(model, tokenizer, texts, device, batch_size, max_length, normalize=False):
    if not texts:
        hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
        return torch.empty((0, hidden_size), dtype=torch.float32)
    tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
    if tokenizer_max_length is None or int(tokenizer_max_length) <= 0 or int(tokenizer_max_length) > 100000:
        tokenizer_max_length = int(max_length)
    effective_max_length = max(1, min(int(max_length), int(tokenizer_max_length), 512))
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), max(int(batch_size), 1)):
            batch_texts = texts[start : start + max(int(batch_size), 1)]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=effective_max_length,
                add_special_tokens=False,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch, output_hidden_states=True)
            pooled = outputs.hidden_states[-1].mean(dim=1)
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.detach().cpu().float())
    return torch.cat(chunks, dim=0)


def _coerce_long_tensor_1d(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.detach().cpu().long().view(-1)
    else:
        try:
            tensor = torch.as_tensor(value, dtype=torch.long).view(-1)
        except Exception:
            return None
    return tensor


def _coerce_bool_tensor_1d(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.detach().cpu().bool().view(-1)
    else:
        try:
            tensor = torch.as_tensor(value).detach().cpu().bool().view(-1)
        except Exception:
            return None
    return tensor


def _infer_zero_fill_target_mask(component_tensors, scalar_tensors, expected_nodes, expected_count):
    if expected_count <= 0 or expected_count >= expected_nodes:
        return None
    evidence_columns = []
    for value in component_tensors.values():
        if torch.is_tensor(value) and value.dim() == 2 and int(value.shape[0]) == int(expected_nodes):
            evidence_columns.append(value.detach().cpu().float().abs().sum(dim=1) > 1e-8)
    for value in scalar_tensors.values():
        if torch.is_tensor(value) and value.dim() == 1 and int(value.shape[0]) == int(expected_nodes):
            evidence_columns.append(value.detach().cpu().float().abs() > 1e-8)
    if not evidence_columns:
        return None
    inferred_mask = torch.stack([column.bool() for column in evidence_columns], dim=0).any(dim=0)
    if int(inferred_mask.numel()) != int(expected_nodes):
        return None
    inferred_count = int(inferred_mask.sum().item())
    if inferred_count == 0 or inferred_count != int(expected_count):
        return None
    return inferred_mask


def _resolve_prompt_target_membership(bundle, expected_nodes):
    expected_nodes = int(expected_nodes)
    target_scope = str(bundle.get("target_node_scope", "all_nodes")).strip().lower()
    requested_count = int(bundle.get("target_node_count", 0) or 0)
    payload_row_layout = str(bundle.get("payload_row_layout", "") or "").strip().lower()
    explicit_mask = _coerce_bool_tensor_1d(bundle.get("target_node_mask"))
    explicit_ids = _coerce_long_tensor_1d(bundle.get("target_node_ids"))
    resolved_mask = None
    resolved_ids = None
    mask_source = "all_nodes_default"

    if explicit_mask is not None:
        if int(explicit_mask.numel()) != expected_nodes:
            raise MissingFrozenArtifactError(
                f"Prompt-expert target_node_mask has {int(explicit_mask.numel())} rows, expected {expected_nodes}."
            )
        resolved_mask = explicit_mask.bool()
        resolved_ids = torch.nonzero(resolved_mask, as_tuple=False).view(-1).long()
        mask_source = "payload_target_node_mask"
    elif explicit_ids is not None:
        if explicit_ids.numel() > 0:
            if int(explicit_ids.min().item()) < 0 or int(explicit_ids.max().item()) >= expected_nodes:
                raise MissingFrozenArtifactError(
                    "Prompt-expert target_node_ids contain indices outside the current graph range."
                )
            resolved_mask = torch.zeros(expected_nodes, dtype=torch.bool)
            resolved_mask[explicit_ids.long()] = True
            resolved_ids = explicit_ids.long()
            mask_source = "payload_target_node_ids"
    if resolved_mask is None and requested_count > 0 and requested_count < expected_nodes:
        inferred_mask = _infer_zero_fill_target_mask(
            bundle.get("component_tensors", {}),
            bundle.get("scalar_tensors", {}),
            expected_nodes,
            requested_count,
        )
        if inferred_mask is not None:
            resolved_mask = inferred_mask.bool()
            resolved_ids = torch.nonzero(resolved_mask, as_tuple=False).view(-1).long()
            mask_source = "zero_fill_content_fallback"
    if resolved_mask is None and target_scope in {"labeled", "labeled_prefix"} and requested_count > 0 and requested_count <= expected_nodes:
        resolved_mask = torch.zeros(expected_nodes, dtype=torch.bool)
        resolved_mask[:requested_count] = True
        resolved_ids = torch.arange(requested_count, dtype=torch.long)
        mask_source = "labeled_prefix_fallback"
    if resolved_mask is None:
        resolved_mask = torch.ones(expected_nodes, dtype=torch.bool)
        resolved_ids = torch.arange(expected_nodes, dtype=torch.long)
        mask_source = "all_nodes_default"

    resolved_count = int(resolved_mask.sum().item())
    if requested_count > 0 and requested_count < expected_nodes and resolved_count != requested_count:
        if mask_source.startswith("payload_"):
            raise MissingFrozenArtifactError(
                "Prompt-expert target membership metadata is inconsistent with target_node_count."
            )
        if "zero_fill" in payload_row_layout:
            raise MissingFrozenArtifactError(
                "Prompt-expert zero-fill payload could not recover the exact routed-node mask; "
                "rerun precompute.py so the cache persists explicit target membership metadata."
            )
    return resolved_mask.bool(), resolved_ids.long(), mask_source


def _full_graph_eligible_refiner_mask(refiner_features, total_nodes):
    total_nodes = int(total_nodes)
    if isinstance(refiner_features, dict):
        mask = _coerce_bool_tensor_1d(refiner_features.get("eligible_refiner_mask"))
        if mask is not None:
            if int(mask.numel()) != total_nodes:
                raise ValueError(
                    f"eligible_refiner_mask has {int(mask.numel())} rows, expected {total_nodes}."
                )
            return mask.bool()
    return torch.ones(total_nodes, dtype=torch.bool)


def _slice_prompt_expert_bundle_to_labeled_prefix(prompt_expert_bundle, labeled_rows):
    labeled_rows = int(labeled_rows)
    sliced = dict(prompt_expert_bundle)
    original_full_graph_node_count = int(sliced.get("full_graph_node_count", 0) or 0)
    original_target_node_count = int(sliced.get("target_node_count", 0) or 0)
    component_tensors = {}
    for name, value in dict(sliced.get("component_tensors", {})).items():
        if torch.is_tensor(value) and value.dim() >= 1 and int(value.shape[0]) > labeled_rows:
            component_tensors[name] = value[:labeled_rows].detach().cpu()
        else:
            component_tensors[name] = value
    scalar_tensors = {}
    for name, value in dict(sliced.get("scalar_tensors", {})).items():
        if torch.is_tensor(value) and value.dim() >= 1 and int(value.shape[0]) > labeled_rows:
            scalar_tensors[name] = value[:labeled_rows].detach().cpu()
        else:
            scalar_tensors[name] = value
    target_node_mask = _coerce_bool_tensor_1d(sliced.get("target_node_mask"))
    if target_node_mask is not None and int(target_node_mask.numel()) > labeled_rows:
        target_node_mask = target_node_mask[:labeled_rows].bool()
        sliced["target_node_mask"] = target_node_mask
        sliced["target_node_ids"] = torch.nonzero(target_node_mask, as_tuple=False).view(-1).long()
        sliced["target_node_count"] = int(target_node_mask.sum().item())
    else:
        target_node_ids = _coerce_long_tensor_1d(sliced.get("target_node_ids"))
        if target_node_ids is not None:
            target_node_ids = target_node_ids[target_node_ids < labeled_rows].long()
            target_node_mask = torch.zeros(labeled_rows, dtype=torch.bool)
            if target_node_ids.numel():
                target_node_mask[target_node_ids] = True
            sliced["target_node_ids"] = target_node_ids
            sliced["target_node_mask"] = target_node_mask
            sliced["target_node_count"] = int(target_node_mask.sum().item())
    sliced["component_tensors"] = component_tensors
    sliced["scalar_tensors"] = scalar_tensors
    sliced["original_full_graph_node_count"] = original_full_graph_node_count
    sliced["original_target_node_count"] = original_target_node_count
    sliced["full_graph_node_count"] = labeled_rows
    sliced["labeled_node_count"] = labeled_rows
    sliced["payload_row_layout"] = "labeled_prefix_sliced_from_full_graph"
    sliced["semantic_row_scope"] = "labeled_prefix_sliced_from_full_graph_prompt_payload"
    return sliced


def _mask_from_idx(num_nodes, idx):
    mask = np.zeros(num_nodes, dtype=bool)
    mask[_idx_numpy(idx)] = True
    return mask


def _normalize_glance_advantage_target(advantage):
    if advantage.numel() == 0:
        return advantage
    centered = advantage - advantage.mean()
    scale = centered.std(unbiased=False).clamp_min(1e-6)
    return centered / scale


def _scale_glance_advantage_target(advantage):
    if advantage.numel() == 0:
        return advantage
    scale = advantage.std(unbiased=False).clamp_min(1e-6)
    return advantage / scale


def _glance_oracle_topk_membership(advantage, k):
    if advantage.numel() == 0:
        return torch.zeros_like(advantage, dtype=torch.float32)
    k = min(max(int(k), 1), int(advantage.numel()))
    target = torch.zeros_like(advantage, dtype=torch.float32)
    top_idx = torch.topk(advantage.detach(), k=k, largest=True, sorted=False).indices
    target[top_idx] = 1.0
    return target


def _glance_pairwise_ranking_loss(scores, targets):
    if scores.numel() < 2:
        return scores.sum() * 0.0
    score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
    pair_mask = torch.triu(torch.ones_like(target_diff, dtype=torch.bool), diagonal=1)
    pair_mask = pair_mask & (target_diff.abs() > 1e-6)
    if not bool(pair_mask.any().item()):
        return scores.sum() * 0.0
    signed_margin = torch.sign(target_diff[pair_mask]) * score_diff[pair_mask]
    pair_weight = target_diff[pair_mask].abs()
    loss = F.softplus(-signed_margin)
    return (loss * pair_weight).sum() / pair_weight.sum().clamp_min(1e-6)


def _safe_score_corr(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return None
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _safe_binary_score_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size or labels.size == 0:
        return {
            "positive_rate": 0.0,
            "mean_score": 0.0,
            "auroc": None,
            "auprc": None,
        }
    metrics = {
        "positive_rate": float(labels.mean()),
        "mean_score": float(scores.mean()),
        "auroc": None,
        "auprc": None,
    }
    if np.unique(labels).size < 2:
        return metrics
    try:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
    except Exception:
        metrics["auroc"] = None
    try:
        metrics["auprc"] = float(average_precision_score(labels, scores))
    except Exception:
        metrics["auprc"] = None
    return metrics


def _safe_advantage_router_metrics(advantage, scores, routed_mask=None):
    advantage = np.asarray(advantage, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if advantage.size != scores.size or advantage.size == 0:
        return {
            "positive_rate": 0.0,
            "mean_advantage": 0.0,
            "score_adv_corr": None,
            "auroc": None,
            "auprc": None,
        }
    labels = (advantage > 0.0).astype(np.int64)
    metrics = {
        "positive_rate": float(labels.mean()),
        "mean_advantage": float(advantage.mean()),
        "score_adv_corr": _safe_score_corr(scores, advantage),
        "auroc": None,
        "auprc": None,
    }
    if np.unique(labels).size >= 2:
        try:
            metrics["auroc"] = float(roc_auc_score(labels, scores))
        except Exception:
            metrics["auroc"] = None
        try:
            metrics["auprc"] = float(average_precision_score(labels, scores))
        except Exception:
            metrics["auprc"] = None
    if routed_mask is not None:
        routed_mask = np.asarray(routed_mask, dtype=bool).reshape(-1)
        if routed_mask.size == labels.size and routed_mask.any():
            routed_labels = labels[routed_mask]
            metrics["routed_positive_precision"] = float(routed_labels.mean())
            metrics["routed_mean_advantage"] = float(advantage[routed_mask].mean())
            positive_count = int(labels.sum())
            metrics["routed_positive_coverage"] = float(routed_labels.sum() / positive_count) if positive_count > 0 else 0.0
            budget = int(routed_mask.sum())
            oracle_idx = np.argsort(-advantage)[:budget]
            routed_idx = np.flatnonzero(routed_mask)
            metrics["routed_oracle_topk_overlap"] = float(
                len(set(routed_idx.tolist()).intersection(set(oracle_idx.tolist()))) / max(budget, 1)
            )
        else:
            metrics["routed_positive_precision"] = 0.0
            metrics["routed_mean_advantage"] = 0.0
            metrics["routed_positive_coverage"] = 0.0
            metrics["routed_oracle_topk_overlap"] = 0.0
    return metrics


def _apply_glance_router_scaler(feature_matrix, scaler_state):
    feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
    if feature_matrix.ndim != 2:
        raise ValueError("GLANCE router feature matrix must be 2-D.")
    mean = np.asarray((scaler_state or {}).get("mean", []), dtype=np.float32).reshape(-1)
    scale = np.asarray((scaler_state or {}).get("scale", []), dtype=np.float32).reshape(-1)
    if mean.size != feature_matrix.shape[1] or scale.size != feature_matrix.shape[1]:
        raise ValueError(
            "Frozen-router scaler state does not align with the current router feature dimension: "
            f"expected {feature_matrix.shape[1]}, got mean={mean.size}, scale={scale.size}."
        )
    scaled = (feature_matrix - mean) / np.clip(scale, 1e-6, None)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return torch.tensor(scaled, dtype=torch.float32)


class GlanceRefinerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, activation="leakyrelu", dropout=0.1):
        super().__init__()
        activation = str(activation).lower()
        if activation == "leakyrelu":
            act = nn.LeakyReLU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == "elu":
            act = nn.ELU()
        else:
            raise ValueError(f"Unsupported refiner activation: {activation}")
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            act,
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, x):
        return self.net(x)


class GatedGlanceRefinerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, activation="leakyrelu", dropout=0.1):
        super().__init__()
        activation = str(activation).lower()
        if activation == "leakyrelu":
            act_factory = nn.LeakyReLU
        elif activation == "relu":
            act_factory = nn.ReLU
        elif activation == "elu":
            act_factory = nn.ELU
        else:
            raise ValueError(f"Unsupported refiner activation: {activation}")
        self.shared = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            act_factory(),
            nn.Dropout(float(dropout)),
        )
        self.classifier = nn.Linear(int(hidden_dim), 2)
        self.gate_head = nn.Linear(int(hidden_dim), 1)

    def forward(self, x):
        hidden = self.shared(x)
        logits = self.classifier(hidden)
        gate_logits = self.gate_head(hidden).squeeze(-1)
        gate_prob = torch.sigmoid(gate_logits)
        return logits, gate_logits, gate_prob


class PromptExpertBundleRefinerMLP(nn.Module):
    COMPONENT_ORDER = _PROMPT_EXPERT_COMPONENT_ORDER
    MPE_COMPONENT_ORDER = ("graph_following", "graph_follower", "tweet", "conflict", "metadata_structured")
    GAUGLLM_COMPONENT_ORDER = _GAUGLLM_SELECTOR_COMPONENT_ORDER
    DEFAULT_MPE_COMPONENT = "tweet"
    RAW_CONCAT_COMPONENTS = {
        "raw_concat_single_graph_following": ("graph_following",),
        "raw_concat_single_graph_follower": ("graph_follower",),
        "raw_concat_single_tweet": ("tweet",),
        "raw_concat_single_conflict": ("conflict",),
        "raw_concat_following_triplet": ("graph_following", "tweet", "conflict"),
        "raw_concat_follower_triplet": ("graph_follower", "tweet", "conflict"),
        "raw_concat_metadata_anchor": ("graph_follower", "tweet", "conflict", "metadata_structured"),
        "raw_concat_metadata_only": ("metadata_structured",),
    }

    def __init__(
        self,
        z_gnn_dim,
        component_dims,
        proj_dim=256,
        hidden_dim=128,
        structural_dim=4,
        graph_gate_dim=None,
        activation="leakyrelu",
        dropout=0.1,
        explicit_gate=False,
        fusion_mode="projector_concat",
        mpe_gate_dim=None,
    ):
        super().__init__()
        activation = str(activation).lower()
        if activation == "leakyrelu":
            act_factory = nn.LeakyReLU
        elif activation == "relu":
            act_factory = nn.ReLU
        elif activation == "elu":
            act_factory = nn.ELU
        else:
            raise ValueError(f"Unsupported refiner activation: {activation}")
        self.component_dims = {name: int(component_dims[name]) for name in self.COMPONENT_ORDER}
        self.z_gnn_dim = int(z_gnn_dim)
        self.proj_dim = int(proj_dim)
        self.structural_dim = int(structural_dim)
        self.graph_gate_dim = int(graph_gate_dim if graph_gate_dim is not None else structural_dim)
        self.explicit_gate = bool(explicit_gate)
        self.fusion_mode = str(fusion_mode or "projector_concat").lower()
        supported_fusion_modes = {"projector_concat", "mpe_gated", "gaugllm_selector", *self.RAW_CONCAT_COMPONENTS.keys()}
        if self.fusion_mode not in supported_fusion_modes:
            raise ValueError(f"Unsupported prompt-expert fusion mode: {fusion_mode}")
        self.raw_concat_components = tuple(self.RAW_CONCAT_COMPONENTS.get(self.fusion_mode, ()))
        self.uses_projectors = self.fusion_mode in {"projector_concat", "mpe_gated", "gaugllm_selector"}
        self.projectors = (
            nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.Linear(self.component_dims[name], self.proj_dim),
                        act_factory(),
                        nn.Dropout(float(dropout)),
                    )
                    for name in self.COMPONENT_ORDER
                }
            )
            if self.uses_projectors
            else None
        )
        self.projector_concat_components = (
            "ego",
            "graph_following",
            "graph_follower",
            "tweet",
            "conflict",
            "metadata_structured",
        )
        self.graph_gate = nn.Linear(self.graph_gate_dim, 2) if self.fusion_mode == "projector_concat" else None
        self.mpe_gate_dim = int(mpe_gate_dim if mpe_gate_dim is not None else self.z_gnn_dim + self.structural_dim)
        self.mpe_gate = (
            nn.Sequential(
                nn.Linear(self.mpe_gate_dim, int(hidden_dim)),
                act_factory(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), len(self.MPE_COMPONENT_ORDER)),
            )
            if self.fusion_mode == "mpe_gated"
            else None
        )
        self.selector_context_projectors = (
            nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.Linear(self.component_dims[name], self.proj_dim),
                        act_factory(),
                        nn.Dropout(float(dropout)),
                    )
                    for name in self.GAUGLLM_COMPONENT_ORDER
                }
            )
            if self.fusion_mode == "gaugllm_selector"
            else None
        )
        self.gaug_expert_query = (
            nn.Sequential(
                nn.Linear(self.z_gnn_dim + self.structural_dim + self.proj_dim, self.proj_dim),
                act_factory(),
                nn.Dropout(float(dropout)),
            )
            if self.fusion_mode == "gaugllm_selector"
            else None
        )
        self.gaug_context_query = (
            nn.Sequential(
                nn.Linear(self.structural_dim + self.proj_dim, self.proj_dim),
                act_factory(),
                nn.Dropout(float(dropout)),
            )
            if self.fusion_mode == "gaugllm_selector"
            else None
        )
        if self.fusion_mode == "mpe_gated":
            input_dim = self.z_gnn_dim + self.proj_dim + self.structural_dim
        elif self.fusion_mode == "gaugllm_selector":
            input_dim = self.z_gnn_dim + self.proj_dim + self.proj_dim + self.structural_dim
        elif self.fusion_mode == "projector_concat":
            input_dim = self.z_gnn_dim + (len(self.projector_concat_components) + 1) * self.proj_dim + self.structural_dim
        else:
            raw_dim = sum(self.component_dims[name] for name in self.raw_concat_components)
            input_dim = self.z_gnn_dim + raw_dim + self.structural_dim
        self.shared = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            act_factory(),
            nn.Dropout(float(dropout)),
        )
        self.classifier = nn.Linear(int(hidden_dim), 2)
        self.gate_head = nn.Linear(int(hidden_dim), 1) if self.explicit_gate else None

    def forward(
        self,
        z_gnn,
        semantic_views,
        structural_features,
        graph_gate_features=None,
        mpe_gate_features=None,
        expert_presence_mask=None,
        selector_context_views=None,
    ):
        mpe_weights = None
        mpe_selected_expert = None
        if self.fusion_mode == "mpe_gated":
            projected = {
                name: self.projectors[name](semantic_views[name])
                for name in self.COMPONENT_ORDER
            }
            if mpe_gate_features is None:
                mpe_gate_features = torch.cat([z_gnn, structural_features], dim=1)
            expert_stack = torch.stack(
                [projected[name] for name in self.MPE_COMPONENT_ORDER],
                dim=1,
            )
            selector_logits = self.mpe_gate(mpe_gate_features)
            if expert_presence_mask is not None:
                expert_presence_mask = expert_presence_mask.bool()
                if expert_presence_mask.dim() != 2 or int(expert_presence_mask.shape[1]) != len(self.MPE_COMPONENT_ORDER):
                    raise ValueError(
                        "PromptExpertBundleRefinerMLP expert_presence_mask must be shaped "
                        f"[batch, {len(self.MPE_COMPONENT_ORDER)}]."
                    )
                row_has_expert = expert_presence_mask.any(dim=1, keepdim=True)
                if not bool(row_has_expert.all().item()):
                    fallback_mask = torch.zeros_like(expert_presence_mask)
                    default_idx = self.MPE_COMPONENT_ORDER.index(self.DEFAULT_MPE_COMPONENT)
                    fallback_mask[:, default_idx] = True
                    expert_presence_mask = torch.where(row_has_expert, expert_presence_mask, fallback_mask)
                selector_logits = selector_logits.masked_fill(~expert_presence_mask, -1e4)
            mpe_weights = torch.softmax(selector_logits, dim=1)
            mpe_selected_expert = torch.argmax(mpe_weights, dim=1)
            expert_fused = torch.sum(mpe_weights.unsqueeze(-1) * expert_stack, dim=1)
            x = torch.cat([z_gnn, expert_fused, structural_features], dim=1)
        elif self.fusion_mode == "gaugllm_selector":
            if selector_context_views is None:
                raise ValueError("gaugllm_selector requires selector_context_views.")
            projected = {
                name: self.projectors[name](semantic_views[name])
                for name in self.COMPONENT_ORDER
            }
            context_projected = {
                name: self.selector_context_projectors[name](selector_context_views[name])
                for name in self.GAUGLLM_COMPONENT_ORDER
            }
            if expert_presence_mask is not None:
                expert_presence_mask = expert_presence_mask.bool()
                if expert_presence_mask.dim() != 2 or int(expert_presence_mask.shape[1]) != len(self.GAUGLLM_COMPONENT_ORDER):
                    raise ValueError(
                        "PromptExpertBundleRefinerMLP expert_presence_mask must be shaped "
                        f"[batch, {len(self.GAUGLLM_COMPONENT_ORDER)}] for gaugllm_selector."
                    )
                row_has_expert = expert_presence_mask.any(dim=1, keepdim=True)
                if not bool(row_has_expert.all().item()):
                    fallback_mask = torch.zeros_like(expert_presence_mask)
                    default_idx = self.GAUGLLM_COMPONENT_ORDER.index(self.DEFAULT_MPE_COMPONENT)
                    fallback_mask[:, default_idx] = True
                    expert_presence_mask = torch.where(row_has_expert, expert_presence_mask, fallback_mask)
            metadata_proj = projected["metadata_structured"]
            expert_stack = torch.stack(
                [projected[name] for name in self.GAUGLLM_COMPONENT_ORDER],
                dim=1,
            )
            context_stack = torch.stack(
                [context_projected[name] for name in self.GAUGLLM_COMPONENT_ORDER],
                dim=1,
            )
            expert_query = self.gaug_expert_query(torch.cat([z_gnn, structural_features, metadata_proj], dim=1))
            context_query = self.gaug_context_query(torch.cat([metadata_proj, structural_features], dim=1))
            scale = math.sqrt(float(self.proj_dim))
            expert_logits = torch.sum(expert_stack * expert_query.unsqueeze(1), dim=2) / scale
            context_logits = torch.sum(context_stack * context_query.unsqueeze(1), dim=2) / scale
            selector_logits = expert_logits + context_logits
            if expert_presence_mask is not None:
                selector_logits = selector_logits.masked_fill(~expert_presence_mask, -1e4)
            mpe_weights = torch.softmax(selector_logits, dim=1)
            mpe_selected_expert = torch.argmax(mpe_weights, dim=1)
            expert_fused = torch.sum(mpe_weights.unsqueeze(-1) * expert_stack, dim=1)
            x = torch.cat([z_gnn, expert_fused, metadata_proj, structural_features], dim=1)
        elif self.fusion_mode == "projector_concat":
            projected = {
                name: self.projectors[name](semantic_views[name])
                for name in self.COMPONENT_ORDER
            }
            if graph_gate_features is None:
                graph_gate_features = structural_features[:, : self.graph_gate_dim]
            gate_weights = torch.softmax(self.graph_gate(graph_gate_features), dim=1)
            graph_fused = (
                gate_weights[:, 0:1] * projected["graph_following"]
                + gate_weights[:, 1:2] * projected["graph_follower"]
            )
            x = torch.cat(
                [
                    z_gnn,
                    projected["ego"],
                    projected["graph_following"],
                    projected["graph_follower"],
                    graph_fused,
                    projected["tweet"],
                    projected["conflict"],
                    projected["metadata_structured"],
                    structural_features,
                ],
                dim=1,
            )
        else:
            raw_components = [semantic_views[name] for name in self.raw_concat_components]
            x = torch.cat(
                [z_gnn, *raw_components, structural_features],
                dim=1,
            )
        hidden = self.shared(x)
        logits = self.classifier(hidden)
        if self.gate_head is None:
            return (logits, mpe_weights, mpe_selected_expert) if mpe_weights is not None else logits
        gate_logits = self.gate_head(hidden).squeeze(-1)
        gate_prob = torch.sigmoid(gate_logits)
        return logits, gate_logits, gate_prob, mpe_weights, mpe_selected_expert


def _normalize_semantic_payload(payload):
    payload_keys = []
    precomputed_views = None
    prompt_expert_bundle = None
    embeddings = payload
    if isinstance(payload, dict):
        payload_keys = sorted(str(key) for key in payload.keys())
        direct_views = {}
        for view_name in ("ego", "hop1", "hop2"):
            if view_name not in payload:
                continue
            view_value = payload[view_name]
            if not torch.is_tensor(view_value):
                try:
                    view_value = torch.as_tensor(view_value)
                except Exception:
                    continue
            if view_value.dim() != 2:
                continue
            direct_views[view_name] = view_value.detach().cpu().float()
        if len(direct_views) == 3:
            precomputed_views = direct_views
        expert_component_tensors = {}
        for component_name in _PROMPT_EXPERT_COMPONENT_ORDER:
            if component_name not in payload:
                continue
            component_value = payload[component_name]
            if not torch.is_tensor(component_value):
                try:
                    component_value = torch.as_tensor(component_value)
                except Exception:
                    continue
            if component_value.dim() != 2:
                continue
            expert_component_tensors[component_name] = component_value.detach().cpu().float()
        expert_scalar_tensors = {}
        for scalar_key in (
            "count_following",
            "count_follower",
            "has_following",
            "has_follower",
            "candidate_count_following",
            "candidate_count_follower",
            "selected_count_following",
            "selected_count_follower",
            "mean_sim_following_support",
            "mean_sim_following_contrast",
            "mean_sim_follower_support",
            "mean_sim_follower_contrast",
            "reciprocal_ratio_following_selected",
            "reciprocal_ratio_follower_selected",
            "rt_ratio",
            "url_ratio",
            "hashtag_ratio",
        ):
            if scalar_key not in payload:
                continue
            scalar_value = payload[scalar_key]
            if not torch.is_tensor(scalar_value):
                try:
                    scalar_value = torch.as_tensor(scalar_value)
                except Exception:
                    continue
            if scalar_value.dim() == 2 and int(scalar_value.shape[1]) == 1:
                scalar_value = scalar_value.view(-1)
            if scalar_value.dim() != 1:
                continue
            expert_scalar_tensors[scalar_key] = scalar_value.detach().cpu().float().view(-1)
        payload_semantic_view_mode = str(payload.get("semantic_view_mode", "") or "").strip().lower()
        if expert_component_tensors or expert_scalar_tensors or _is_prompt_expert_semantic_view_mode(payload_semantic_view_mode):
            active_components = payload.get("active_components")
            if active_components is None:
                active_components = list(expert_component_tensors.keys())
            elif isinstance(active_components, str):
                active_components = [active_components]
            else:
                active_components = [str(item) for item in active_components]
            prompt_expert_bundle = {
                "component_tensors": expert_component_tensors,
                "scalar_tensors": expert_scalar_tensors,
                "active_components": list(active_components),
                "semantic_view_mode": payload_semantic_view_mode if _is_prompt_expert_semantic_view_mode(payload_semantic_view_mode) else "prompt_expert_bundle_v1",
                "graph_data_variant": str(payload.get("graph_data_variant", "labeled")),
                "target_node_count": int(payload.get("target_node_count", 0) or 0),
                "full_graph_node_count": int(payload.get("full_graph_node_count", 0) or 0),
                "labeled_node_count": int(payload.get("labeled_node_count", 0) or 0),
                "target_node_scope": str(payload.get("target_node_scope", "all_nodes")),
                "payload_row_layout": str(payload.get("payload_row_layout", "")),
                "target_node_ids": payload.get("target_node_ids"),
                "target_node_mask": payload.get("target_node_mask"),
            }
        for candidate_key in ("embeddings", "features", "x"):
            if candidate_key in payload:
                embeddings = payload[candidate_key]
                break
        else:
            if precomputed_views is not None:
                embeddings = precomputed_views["ego"]
    if not torch.is_tensor(embeddings):
        embeddings = torch.as_tensor(embeddings)
    embeddings = embeddings.detach().cpu().float()
    if precomputed_views is not None:
        expected_nodes = int(embeddings.shape[0]) if embeddings.dim() == 2 else int(precomputed_views["ego"].shape[0])
        for view_name, view_tensor in precomputed_views.items():
            if view_tensor.dim() != 2:
                raise MissingFrozenArtifactError(
                    f"Semantic payload view {view_name} must be 2-D, got {tuple(view_tensor.shape)}."
                )
            if int(view_tensor.shape[0]) != expected_nodes:
                raise MissingFrozenArtifactError(
                    f"Semantic payload view {view_name} has {int(view_tensor.shape[0])} nodes, expected {expected_nodes}."
                )
    if prompt_expert_bundle is not None:
        expected_nodes = int(embeddings.shape[0]) if embeddings.dim() == 2 else None
        if expected_nodes is None:
            expert_components = prompt_expert_bundle["component_tensors"]
            if expert_components:
                expected_nodes = int(next(iter(expert_components.values())).shape[0])
        if expected_nodes is None:
            raise MissingFrozenArtifactError(
                "Prompt-expert semantic payload must provide either a 2-D embeddings tensor or at least one expert component tensor."
            )
        target_node_mask, target_node_ids, target_mask_source = _resolve_prompt_target_membership(
            prompt_expert_bundle,
            expected_nodes,
        )
        prompt_expert_bundle["target_node_mask"] = target_node_mask
        prompt_expert_bundle["target_node_ids"] = target_node_ids
        prompt_expert_bundle["target_mask_source"] = str(target_mask_source)
        prompt_expert_bundle["target_node_count"] = int(target_node_mask.sum().item())
        for component_name, component_tensor in prompt_expert_bundle["component_tensors"].items():
            if component_tensor.dim() != 2:
                raise MissingFrozenArtifactError(
                    f"Prompt-expert component {component_name} must be 2-D, got {tuple(component_tensor.shape)}."
                )
            component_rows = int(component_tensor.shape[0])
            if component_rows not in {expected_nodes, int(prompt_expert_bundle["target_node_count"])}:
                raise MissingFrozenArtifactError(
                    f"Prompt-expert component {component_name} has {component_rows} rows, expected either "
                    f"{expected_nodes} full-graph rows or {int(prompt_expert_bundle['target_node_count'])} target rows."
                )
        for scalar_key, scalar_tensor in prompt_expert_bundle["scalar_tensors"].items():
            if scalar_tensor.dim() != 1:
                raise MissingFrozenArtifactError(
                    f"Prompt-expert scalar {scalar_key} must be 1-D, got {tuple(scalar_tensor.shape)}."
                )
            scalar_rows = int(scalar_tensor.shape[0])
            if scalar_rows not in {expected_nodes, int(prompt_expert_bundle["target_node_count"])}:
                raise MissingFrozenArtifactError(
                    f"Prompt-expert scalar {scalar_key} has {scalar_rows} rows, expected either "
                    f"{expected_nodes} full-graph rows or {int(prompt_expert_bundle['target_node_count'])} target rows."
                )
    return embeddings, precomputed_views, prompt_expert_bundle, payload_keys


def _refiner_feature_lookup(refiner_features, batch_idx, semantic_view_mode, device):
    batch_idx_cpu = batch_idx.detach().cpu()
    if isinstance(refiner_features, dict) and _is_prompt_expert_feature_kind(refiner_features.get("feature_kind")):
        selector_context_views = refiner_features.get("selector_context_views")
        return {
            "feature_kind": str(refiner_features.get("feature_kind", "prompt_expert_bundle")),
            "semantic_view_mode": str(refiner_features.get("semantic_view_mode", semantic_view_mode)),
            "z_gnn": refiner_features["z_gnn"][batch_idx_cpu].to(device),
            "semantic_views": {
                key: value[batch_idx_cpu].to(device)
                for key, value in refiner_features["semantic_views"].items()
                if torch.is_tensor(value) and value.dim() == 2
            },
            "structural_features": refiner_features["structural_features"][batch_idx_cpu].to(device),
            "graph_gate_features": refiner_features["graph_gate_features"][batch_idx_cpu].to(device),
            "mpe_gate_features": refiner_features["mpe_gate_features"][batch_idx_cpu].to(device)
            if "mpe_gate_features" in refiner_features
            else None,
            "expert_presence_mask": refiner_features["expert_presence_mask"][batch_idx_cpu].to(device)
            if "expert_presence_mask" in refiner_features
            else None,
            "selector_context_views": {
                key: value[batch_idx_cpu].to(device)
                for key, value in selector_context_views.items()
                if torch.is_tensor(value) and value.dim() == 2
            }
            if isinstance(selector_context_views, dict)
            else None,
            "selector_component_order": list(refiner_features.get("selector_component_order", [])),
        }
    return refiner_features[batch_idx_cpu].to(device)


class GlanceStageMixin:
    """Shared GLANCE input/runtime owner for the active mainline."""

    def _glance_refiner_source_stage(self):
        source = str(getattr(self.args, "glance_refiner_source", "oracle")).lower()
        if source == "oracle":
            return "glance_oracle_refine"
        if source == "full_graph":
            return "glance_full_graph_refine"
        raise ValueError(f"Unsupported --glance_refiner_source: {source}")

    def _load_glance_refiner_artifact(self):
        stage_name = self._glance_refiner_source_stage()
        manifest = self._require_stage_json(stage_name, "manifest")
        metrics = self._require_stage_json(stage_name, "metrics")
        outputs = self._require_stage_tensor(stage_name, "outputs")
        checkpoint = self._require_stage_tensor(stage_name, "checkpoint")
        per_node_rows = self._require_stage_jsonl(stage_name, "per_node_test")
        analysis_summary = self._read_stage_json(stage_name, "analysis_summary")
        return {
            "stage_name": stage_name,
            "manifest": manifest,
            "metrics": metrics,
            "outputs": outputs,
            "checkpoint": checkpoint,
            "per_node_rows": per_node_rows,
            "analysis_summary": analysis_summary,
        }

    def _oracle_wrong_node_masks(self, base_pred):
        base_pred = np.asarray(base_pred, dtype=np.int64).reshape(-1)
        wrong = base_pred != self.labels
        return {
            "train": wrong & self.train_mask,
            "valid": wrong & self.val_mask,
            "test": wrong & self.test_mask,
            "all": wrong,
        }

    def _build_k_hop_semantic_views(self, semantic_embeddings, edge_index):
        direct_views = None
        if isinstance(semantic_embeddings, dict):
            maybe_direct = {}
            for view_name in ("ego", "hop1", "hop2"):
                if view_name not in semantic_embeddings:
                    maybe_direct = None
                    break
                view_tensor = semantic_embeddings[view_name]
                if not torch.is_tensor(view_tensor):
                    view_tensor = torch.as_tensor(view_tensor)
                maybe_direct[view_name] = view_tensor.detach().cpu().float()
            if maybe_direct is not None:
                direct_views = maybe_direct
                x_sem = direct_views["ego"]
            else:
                base_tensor = None
                for key in ("embeddings", "features", "x"):
                    if key in semantic_embeddings:
                        base_tensor = semantic_embeddings[key]
                        break
                if base_tensor is None:
                    raise ValueError("semantic_embeddings dict must provide ego/hop1/hop2 or embeddings/features/x.")
                if not torch.is_tensor(base_tensor):
                    base_tensor = torch.as_tensor(base_tensor)
                x_sem = base_tensor.detach().cpu().float()
        else:
            x_sem = semantic_embeddings.detach().cpu().float()
        num_nodes = int(x_sem.shape[0])
        if edge_index is None:
            zero = torch.zeros_like(x_sem)
            counts = np.zeros(num_nodes, dtype=np.int64)
            return {
                "ego": x_sem,
                "hop1": direct_views["hop1"].clone() if direct_views is not None else zero.clone(),
                "hop2": direct_views["hop2"].clone() if direct_views is not None else zero.clone(),
                "count_1hop": counts.copy(),
                "count_2hop": counts.copy(),
            }

        edge_index_t = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
        if edge_index_t.dim() != 2 or edge_index_t.size(0) != 2:
            raise ValueError("edge_index must be shaped [2, num_edges] for glance_oracle_refine.")
        if edge_index_t.numel() == 0:
            zero = torch.zeros_like(x_sem)
            counts = np.zeros(num_nodes, dtype=np.int64)
            return {
                "ego": x_sem,
                "hop1": direct_views["hop1"].clone() if direct_views is not None else zero.clone(),
                "hop2": direct_views["hop2"].clone() if direct_views is not None else zero.clone(),
                "count_1hop": counts.copy(),
                "count_2hop": counts.copy(),
            }

        src = edge_index_t[0].numpy()
        dst = edge_index_t[1].numpy()
        neighbors_in = [[] for _ in range(num_nodes)]
        for s, d in zip(src.tolist(), dst.tolist()):
            if 0 <= s < num_nodes and 0 <= d < num_nodes:
                neighbors_in[d].append(s)

        hop1 = direct_views["hop1"].clone() if direct_views is not None else torch.zeros_like(x_sem)
        hop2 = direct_views["hop2"].clone() if direct_views is not None else torch.zeros_like(x_sem)
        count_1hop = np.zeros(num_nodes, dtype=np.int64)
        count_2hop = np.zeros(num_nodes, dtype=np.int64)

        for node_idx in range(num_nodes):
            n1 = sorted(set(int(n) for n in neighbors_in[node_idx] if int(n) != node_idx))
            count_1hop[node_idx] = len(n1)
            if direct_views is None and n1:
                hop1[node_idx] = x_sem[torch.tensor(n1, dtype=torch.long)].mean(dim=0)

            n1_set = set(n1)
            n2 = set()
            for neigh in n1:
                for cand in neighbors_in[neigh]:
                    cand = int(cand)
                    if cand == node_idx or cand in n1_set:
                        continue
                    n2.add(cand)
            n2 = sorted(n2)
            count_2hop[node_idx] = len(n2)
            if direct_views is None and n2:
                hop2[node_idx] = x_sem[torch.tensor(n2, dtype=torch.long)].mean(dim=0)

        return {
            "ego": x_sem,
            "hop1": hop1,
            "hop2": hop2,
            "count_1hop": count_1hop,
            "count_2hop": count_2hop,
        }

    def _build_prompt_expert_semantic_views(self, expert_bundle, z_gnn, base_prob=None):
        if not expert_bundle:
            raise MissingFrozenArtifactError(
                "Prompt-expert semantic views require expert_bundle payload metadata from the semantic cache."
            )
        semantic_view_mode = str(expert_bundle.get("semantic_view_mode", "prompt_expert_bundle_v1"))
        num_nodes = int(z_gnn.shape[0])
        component_tensors = dict(expert_bundle.get("component_tensors", {}))
        base_component = next(iter(component_tensors.values()), None)
        if base_component is None:
            raise MissingFrozenArtifactError(
                "Prompt-expert semantic payload requires at least one named expert component in the prompt cache payload."
            )
        semantic_dim = int(base_component.shape[1])
        zero_component = torch.zeros((num_nodes, semantic_dim), dtype=torch.float32)
        component_default_dims = {
            name: int(value.shape[1])
            for name, value in component_tensors.items()
            if torch.is_tensor(value) and value.dim() == 2
        }
        component_default_dims.setdefault("metadata_structured", 1)
        target_scope = str(expert_bundle.get("target_node_scope", "all_nodes")).lower()
        target_node_count = int(expert_bundle.get("target_node_count", 0) or 0)
        full_graph_node_count = int(expert_bundle.get("full_graph_node_count", 0) or 0)
        target_node_ids = _coerce_long_tensor_1d(expert_bundle.get("target_node_ids"))
        eligible_refiner_mask = _coerce_bool_tensor_1d(expert_bundle.get("target_node_mask"))
        if eligible_refiner_mask is None:
            eligible_refiner_mask = torch.ones(num_nodes, dtype=torch.bool)
        elif int(eligible_refiner_mask.numel()) != num_nodes:
            raise MissingFrozenArtifactError(
                f"Prompt-expert target_node_mask has {int(eligible_refiner_mask.numel())} rows, expected {num_nodes}."
            )
        if target_node_ids is None:
            target_node_ids = torch.nonzero(eligible_refiner_mask, as_tuple=False).view(-1).long()
        active_component_set = {str(item) for item in expert_bundle.get("active_components", [])}
        if full_graph_node_count and full_graph_node_count != num_nodes:
            raise MissingFrozenArtifactError(
                f"Prompt-expert payload expects full_graph_node_count={full_graph_node_count}, but routed backbone exposes {num_nodes} nodes."
            )

        def _expand_component_to_graph(value, name):
            rows = int(value.shape[0])
            if rows == num_nodes:
                return value.detach().cpu().float()
            if target_node_ids is not None and rows == int(target_node_ids.numel()) and rows <= num_nodes:
                out = torch.zeros((num_nodes, int(value.shape[1])), dtype=torch.float32)
                out[target_node_ids.long()] = value.detach().cpu().float()
                return out
            if target_scope in {"labeled_prefix", "labeled"} and target_node_count > 0 and rows == target_node_count and rows <= num_nodes:
                out = torch.zeros((num_nodes, int(value.shape[1])), dtype=torch.float32)
                out[:rows] = value.detach().cpu().float()
                return out
            raise MissingFrozenArtifactError(
                f"Prompt-expert component {name} has {rows} rows, expected either {num_nodes} full-graph rows or {target_node_count} labeled-prefix rows."
            )

        def _expand_scalar_to_graph(value, name):
            rows = int(value.shape[0])
            if rows == num_nodes:
                return value.detach().cpu().float().view(-1)
            if target_node_ids is not None and rows == int(target_node_ids.numel()) and rows <= num_nodes:
                out = torch.zeros((num_nodes,), dtype=torch.float32)
                out[target_node_ids.long()] = value.detach().cpu().float().view(-1)
                return out
            if target_scope in {"labeled_prefix", "labeled"} and target_node_count > 0 and rows == target_node_count and rows <= num_nodes:
                out = torch.zeros((num_nodes,), dtype=torch.float32)
                out[:rows] = value.detach().cpu().float().view(-1)
                return out
            raise MissingFrozenArtifactError(
                f"Prompt-expert scalar {name} has {rows} rows, expected either {num_nodes} full-graph rows or {target_node_count} labeled-prefix rows."
            )

        semantic_views = {}
        for component_name in _PROMPT_EXPERT_COMPONENT_ORDER:
            component_value = component_tensors.get(component_name)
            if component_value is None:
                fallback_dim = int(component_default_dims.get(component_name, semantic_dim))
                semantic_views[component_name] = torch.zeros((num_nodes, fallback_dim), dtype=torch.float32)
                continue
            semantic_views[component_name] = _expand_component_to_graph(component_value, component_name)

        scalar_tensors = dict(expert_bundle.get("scalar_tensors", {}))

        def _scalar(name):
            value = scalar_tensors.get(name)
            if value is None:
                return torch.zeros(num_nodes, dtype=torch.float32)
            return _expand_scalar_to_graph(value, name)

        scalar_feature_order = [
            "count_following",
            "count_follower",
            "has_following",
            "has_follower",
            "candidate_count_following",
            "candidate_count_follower",
            "selected_count_following",
            "selected_count_follower",
            "mean_sim_following_support",
            "mean_sim_following_contrast",
            "mean_sim_follower_support",
            "mean_sim_follower_contrast",
            "reciprocal_ratio_following_selected",
            "reciprocal_ratio_follower_selected",
            "rt_ratio",
            "url_ratio",
            "hashtag_ratio",
        ]
        scalar_feature_tensors = {
            name: _scalar(name)
            for name in scalar_feature_order
            if name in scalar_tensors or name in {"count_following", "count_follower", "has_following", "has_follower"}
        }
        count_following = scalar_feature_tensors["count_following"]
        count_follower = scalar_feature_tensors["count_follower"]
        has_following = scalar_feature_tensors["has_following"]
        has_follower = scalar_feature_tensors["has_follower"]
        active_structural_feature_names = [name for name in scalar_feature_order if name in scalar_feature_tensors]
        structural_columns = []
        for name in active_structural_feature_names:
            value = scalar_feature_tensors.get(name)
            if value is None:
                continue
            if name.startswith("count_") or name.startswith("candidate_count_") or name.startswith("selected_count_"):
                structural_columns.append(torch.log1p(value))
            else:
                structural_columns.append(value)
        structural_features = torch.stack(structural_columns, dim=1).float()
        if base_prob is None:
            base_prob = torch.softmax(z_gnn.float(), dim=1)
        else:
            base_prob = base_prob.detach().cpu().float()
        base_entropy = -(base_prob * base_prob.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        prompt_expert_fusion = str(getattr(self.args, "joint_prompt_expert_fusion", "projector_concat")).lower()
        selector_component_order = _selector_component_order_for_fusion(prompt_expert_fusion)
        if semantic_view_mode == "prompt_expert_bundle_v2":
            graph_gate_feature_names = [
                "log1p(count_following)",
                "log1p(count_follower)",
                "has_following",
                "has_follower",
            ]
            graph_gate_features = torch.stack(
                [
                    torch.log1p(count_following),
                    torch.log1p(count_follower),
                    has_following,
                    has_follower,
                ],
                dim=1,
            ).float()
        else:
            graph_gate_feature_names = [
                f"log1p({name})"
                if name.startswith("count_") or name.startswith("candidate_count_") or name.startswith("selected_count_")
                else name
                for name in active_structural_feature_names
            ]
            graph_gate_features = structural_features

        component_norm_by_name = {
            name: semantic_views[name].norm(dim=1).float()
            for name in _PROMPT_EXPERT_COMPONENT_ORDER
        }
        graph_following_available = (
            scalar_feature_tensors.get("selected_count_following", count_following).gt(0)
            if "graph_following" in active_component_set
            else torch.zeros(num_nodes, dtype=torch.bool)
        )
        graph_follower_available = (
            scalar_feature_tensors.get("selected_count_follower", count_follower).gt(0)
            if "graph_follower" in active_component_set
            else torch.zeros(num_nodes, dtype=torch.bool)
        )
        tweet_available = (
            component_norm_by_name["tweet"].gt(1e-8)
            if "tweet" in active_component_set
            else torch.zeros(num_nodes, dtype=torch.bool)
        )
        conflict_available = (
            component_norm_by_name["conflict"].gt(1e-8)
            if "conflict" in active_component_set
            else torch.zeros(num_nodes, dtype=torch.bool)
        )
        metadata_structured_available = (
            component_norm_by_name["metadata_structured"].gt(1e-8)
            if "metadata_structured" in active_component_set
            else torch.zeros(num_nodes, dtype=torch.bool)
        )
        expert_available_by_name = {
            "graph_following": graph_following_available,
            "graph_follower": graph_follower_available,
            "tweet": tweet_available,
            "conflict": conflict_available,
            "metadata_structured": metadata_structured_available,
        }
        expert_presence_mask = torch.stack(
            [expert_available_by_name[name] for name in selector_component_order],
            dim=1,
        ).bool()
        expert_presence_mask = expert_presence_mask & eligible_refiner_mask.unsqueeze(1)
        rows_missing_expert = eligible_refiner_mask & ~expert_presence_mask.any(dim=1)
        if bool(rows_missing_expert.any().item()):
            fallback_candidates = [
                name
                for name in selector_component_order
                if name in active_component_set
            ]
            fallback_name = (
                PromptExpertBundleRefinerMLP.DEFAULT_MPE_COMPONENT
                if PromptExpertBundleRefinerMLP.DEFAULT_MPE_COMPONENT in fallback_candidates
                else (fallback_candidates[0] if fallback_candidates else PromptExpertBundleRefinerMLP.DEFAULT_MPE_COMPONENT)
            )
            fallback_idx = selector_component_order.index(fallback_name)
            expert_presence_mask[rows_missing_expert, fallback_idx] = True
        selector_component_norms = torch.stack(
            [component_norm_by_name[name] for name in selector_component_order],
            dim=1,
        ).float()
        mpe_gate_features = torch.cat(
            [
                z_gnn.detach().cpu().float(),
                base_prob,
                base_entropy,
                structural_features,
                selector_component_norms,
                expert_presence_mask.float(),
            ],
            dim=1,
        )
        selector_feature_names = (
            [f"z_gnn[{idx}]" for idx in range(int(z_gnn.shape[1]))]
            + [f"base_prob[{idx}]" for idx in range(int(base_prob.shape[1]))]
            + ["base_entropy"]
            + active_structural_feature_names
            + [f"{name}_norm" for name in selector_component_order]
            + [f"{name}_available" for name in selector_component_order]
        )
        selector_context_views = None
        if prompt_expert_fusion == "gaugllm_selector":
            selector_context_views = expert_bundle.get("_runtime_selector_context_views")
            if selector_context_views is None:
                selector_rows = expert_bundle.get("selector_runtime_explanation_rows")
                cache_manifest = expert_bundle.get("selector_runtime_cache_manifest")
                if not isinstance(selector_rows, dict) or not isinstance(cache_manifest, dict):
                    raise MissingFrozenArtifactError(
                        "gaugllm_selector requires selector_runtime_explanation_rows and selector_runtime_cache_manifest in the prompt cache bundle."
                    )
                role_prefixes = {
                    "graph_following": "This expert explains who the account chooses to follow and what that implies about social role.",
                    "graph_follower": "This expert explains who follows the account and what that implies about audience or amplification.",
                    "tweet": "This expert explains the account's posting behavior and automation cues.",
                    "conflict": "This expert explains cross-view consistency and contradiction.",
                }
                encode_device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
                tokenizer, model, checkpoint_summary = _load_runtime_selector_encoder(cache_manifest, encode_device)
                normalize_selector_context = bool(cache_manifest.get("normalize", False))
                max_length = int(cache_manifest.get("max_length_hop", 512) or 512)
                batch_size = max(1, min(int(getattr(self.args, "batch_size", 32) or 32), 64))
                selector_context_views = {}
                target_node_list = [int(item) for item in target_node_ids.tolist()]
                try:
                    for component_name in _GAUGLLM_SELECTOR_COMPONENT_ORDER:
                        component_rows = selector_rows.get(component_name)
                        if not isinstance(component_rows, dict):
                            raise MissingFrozenArtifactError(
                                f"gaugllm_selector is missing runtime explanation rows for {component_name}."
                            )
                        texts = []
                        for node_id in target_node_list:
                            row = component_rows.get(node_id)
                            if row is None:
                                raise MissingFrozenArtifactError(
                                    f"gaugllm_selector is missing explanation text for node {node_id} in {component_name}."
                                )
                            explanation = str(row.get("explanation", "") or "").strip()
                            if not explanation:
                                raise MissingFrozenArtifactError(
                                    f"gaugllm_selector explanation text is empty for node {node_id} in {component_name}."
                                )
                            metadata_present = bool(component_norm_by_name["metadata_structured"][node_id].item() > 1e-8)
                            stats_suffix = (
                                "Node stats: "
                                f"count_following={int(count_following[node_id].item())}; "
                                f"count_follower={int(count_follower[node_id].item())}; "
                                f"has_following={int(has_following[node_id].item())}; "
                                f"has_follower={int(has_follower[node_id].item())}; "
                                f"rt_ratio={float(scalar_feature_tensors.get('rt_ratio', torch.zeros(num_nodes))[node_id].item()):.4f}; "
                                f"url_ratio={float(scalar_feature_tensors.get('url_ratio', torch.zeros(num_nodes))[node_id].item()):.4f}; "
                                f"hashtag_ratio={float(scalar_feature_tensors.get('hashtag_ratio', torch.zeros(num_nodes))[node_id].item()):.4f}; "
                                f"metadata_structured_present={'yes' if metadata_present else 'no'}."
                            )
                            texts.append(
                                role_prefixes[component_name]
                                + "\n\nExplanation: "
                                + explanation
                                + "\n\n"
                                + stats_suffix
                            )
                        encoded = _encode_simteg_texts(
                            model,
                            tokenizer,
                            texts,
                            encode_device,
                            batch_size=batch_size,
                            max_length=max_length,
                            normalize=normalize_selector_context,
                        )
                        full_graph_encoded = torch.zeros((num_nodes, int(encoded.shape[1])), dtype=torch.float32)
                        full_graph_encoded[target_node_ids.long()] = encoded.detach().cpu().float()
                        selector_context_views[component_name] = full_graph_encoded
                finally:
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                expert_bundle["_runtime_selector_context_views"] = selector_context_views
                expert_bundle["_runtime_selector_context_checkpoint"] = checkpoint_summary
        analysis_views = dict(semantic_views)
        analysis_views.update(
            {
                "count_following": count_following.numpy().astype(np.int64),
                "count_follower": count_follower.numpy().astype(np.int64),
                "has_following": has_following.numpy().astype(np.int64),
                "has_follower": has_follower.numpy().astype(np.int64),
                "count_1hop": (count_following + count_follower).numpy().astype(np.int64),
                "count_2hop": np.zeros(num_nodes, dtype=np.int64),
                "structural_feature_names": active_structural_feature_names,
                "eligible_refiner_mask": eligible_refiner_mask.numpy().astype(np.int64),
                "prompt_target_node_ids": target_node_ids.numpy().astype(np.int64),
            }
        )
        for name, value in scalar_feature_tensors.items():
            array = value.numpy()
            if name.startswith("count_") or name.startswith("candidate_count_") or name.startswith("selected_count_"):
                analysis_views[name] = array.astype(np.int64)
            else:
                analysis_views[name] = array.astype(np.float32)
        analysis_views["expert_presence_mask"] = expert_presence_mask.numpy().astype(np.int64)
        refiner_features = {
            "feature_kind": "prompt_expert_bundle",
            "semantic_view_mode": semantic_view_mode,
            "z_gnn": z_gnn.detach().cpu().float(),
            "semantic_views": semantic_views,
            "structural_features": structural_features,
            "graph_gate_features": graph_gate_features,
            "mpe_gate_features": mpe_gate_features,
            "expert_presence_mask": expert_presence_mask,
            "eligible_refiner_mask": eligible_refiner_mask.bool(),
            "target_node_ids": target_node_ids.long(),
            "active_components": list(expert_bundle.get("active_components", [])),
            "structural_feature_names": active_structural_feature_names,
            "graph_gate_feature_names": graph_gate_feature_names,
            "mpe_gate_feature_names": selector_feature_names,
            "selector_component_order": list(selector_component_order),
            "selector_context_views": selector_context_views,
            "selector_context_source": str(expert_bundle.get("selector_runtime_cache_manifest_path", "") or ""),
            "target_mask_source": str(expert_bundle.get("target_mask_source", "unknown")),
        }
        return refiner_features, analysis_views

    def _build_relation_aware_1hop_semantic_views(self, semantic_embeddings, edge_index, edge_type, include_2hop=False):
        x_sem = semantic_embeddings.detach().cpu().float()
        num_nodes = int(x_sem.shape[0])
        zero = torch.zeros_like(x_sem)
        zero_counts = np.zeros(num_nodes, dtype=np.int64)
        if edge_index is None or edge_type is None:
            return {
                "ego": x_sem,
                "in_rel0": zero.clone(),
                "in_rel1": zero.clone(),
                "out_rel0": zero.clone(),
                "out_rel1": zero.clone(),
                "in_rel0_2hop": zero.clone(),
                "in_rel1_2hop": zero.clone(),
                "out_rel0_2hop": zero.clone(),
                "out_rel1_2hop": zero.clone(),
                "count_in_rel0": zero_counts.copy(),
                "count_in_rel1": zero_counts.copy(),
                "count_out_rel0": zero_counts.copy(),
                "count_out_rel1": zero_counts.copy(),
                "count_in_rel0_2hop": zero_counts.copy(),
                "count_in_rel1_2hop": zero_counts.copy(),
                "count_out_rel0_2hop": zero_counts.copy(),
                "count_out_rel1_2hop": zero_counts.copy(),
                "has_in_rel0": zero_counts.copy(),
                "has_in_rel1": zero_counts.copy(),
                "has_out_rel0": zero_counts.copy(),
                "has_out_rel1": zero_counts.copy(),
                "has_in_rel0_2hop": zero_counts.copy(),
                "has_in_rel1_2hop": zero_counts.copy(),
                "has_out_rel0_2hop": zero_counts.copy(),
                "has_out_rel1_2hop": zero_counts.copy(),
                "count_1hop": zero_counts.copy(),
                "count_2hop": zero_counts.copy(),
            }

        edge_index_t = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
        edge_type_t = edge_type.detach().cpu().long() if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long)
        if edge_index_t.dim() != 2 or edge_index_t.size(0) != 2:
            raise ValueError("edge_index must be shaped [2, num_edges] for relation-aware semantic views.")
        if edge_type_t.numel() != edge_index_t.size(1):
            raise ValueError("edge_type must align with edge_index for relation-aware semantic views.")

        src = edge_index_t[0].numpy()
        dst = edge_index_t[1].numpy()
        rel = edge_type_t.numpy()
        buckets = {
            "in_rel0": [[] for _ in range(num_nodes)],
            "in_rel1": [[] for _ in range(num_nodes)],
            "out_rel0": [[] for _ in range(num_nodes)],
            "out_rel1": [[] for _ in range(num_nodes)],
        }
        for s, d, r in zip(src.tolist(), dst.tolist(), rel.tolist()):
            if not (0 <= s < num_nodes and 0 <= d < num_nodes):
                continue
            if int(r) == 0:
                buckets["in_rel0"][d].append(s)
                buckets["out_rel0"][s].append(d)
            elif int(r) == 1:
                buckets["in_rel1"][d].append(s)
                buckets["out_rel1"][s].append(d)

        outputs = {"ego": x_sem}
        union_1hop = [set() for _ in range(num_nodes)]
        for key in ("in_rel0", "in_rel1", "out_rel0", "out_rel1"):
            view_tensor = torch.zeros_like(x_sem)
            count_arr = np.zeros(num_nodes, dtype=np.int64)
            has_arr = np.zeros(num_nodes, dtype=np.int64)
            for node_idx in range(num_nodes):
                neighbors = sorted(set(int(n) for n in buckets[key][node_idx] if int(n) != node_idx))
                union_1hop[node_idx].update(neighbors)
                count_arr[node_idx] = len(neighbors)
                has_arr[node_idx] = 1 if neighbors else 0
                if neighbors:
                    view_tensor[node_idx] = x_sem[torch.tensor(neighbors, dtype=torch.long)].mean(dim=0)
            outputs[key] = view_tensor
            outputs[f"count_{key}"] = count_arr
            outputs[f"has_{key}"] = has_arr
        outputs["count_1hop"] = np.asarray([len(item) for item in union_1hop], dtype=np.int64)
        outputs["count_2hop"] = zero_counts.copy()
        if include_2hop:
            key_map = {
                "in_rel0_2hop": "in_rel0",
                "in_rel1_2hop": "in_rel1",
                "out_rel0_2hop": "out_rel0",
                "out_rel1_2hop": "out_rel1",
            }
            union_2hop = [set() for _ in range(num_nodes)]
            for key, parent_key in key_map.items():
                view_tensor = torch.zeros_like(x_sem)
                count_arr = np.zeros(num_nodes, dtype=np.int64)
                has_arr = np.zeros(num_nodes, dtype=np.int64)
                parent_bucket = buckets[parent_key]
                for node_idx in range(num_nodes):
                    n1 = sorted(set(int(n) for n in parent_bucket[node_idx] if int(n) != node_idx))
                    n1_set = set(n1)
                    n2 = set()
                    for neigh in n1:
                        for cand in parent_bucket[neigh]:
                            cand = int(cand)
                            if cand == node_idx or cand in n1_set:
                                continue
                            n2.add(cand)
                    neighbors = sorted(n2)
                    union_2hop[node_idx].update(neighbors)
                    count_arr[node_idx] = len(neighbors)
                    has_arr[node_idx] = 1 if neighbors else 0
                    if neighbors:
                        view_tensor[node_idx] = x_sem[torch.tensor(neighbors, dtype=torch.long)].mean(dim=0)
                outputs[key] = view_tensor
                outputs[f"count_{key}"] = count_arr
                outputs[f"has_{key}"] = has_arr
            outputs["count_2hop"] = np.asarray([len(item) for item in union_2hop], dtype=np.int64)
        return outputs

    def _train_glance_refiner(self, model, train_features, train_labels, valid_features, valid_labels):
        train_features = train_features.detach().cpu().float()
        valid_features = valid_features.detach().cpu().float()
        train_labels = train_labels.detach().cpu().long()
        valid_labels = valid_labels.detach().cpu().long()

        if train_features.size(0) == 0:
            raise ValueError(f"{self.args.stage} requires at least one train node.")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(getattr(self.args, "lr_LM", 1e-5)),
            weight_decay=float(getattr(self.args, "weight_decay_LM", 0.01)),
        )
        batch_size = max(min(int(getattr(self.args, "batch_size_LM", 32)), int(train_features.size(0))), 1)
        loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)
        max_epochs = max(int(getattr(self.args, "LM_pretrain_epochs", 5)), 1)
        patience = max(int(getattr(self.args, "LM_eval_patience", 20)), 1)

        best_state = None
        best_metrics = None
        best_score = (-1.0, float("inf"))
        wait = 0
        train_losses = []

        model.to(self.device)
        for epoch in range(max_epochs):
            model.train()
            epoch_losses = []
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = F.cross_entropy(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            train_losses.append(train_loss)

            model.eval()
            with torch.no_grad():
                train_logits = model(train_features.to(self.device)).cpu()
                if valid_features.size(0) > 0:
                    valid_logits = model(valid_features.to(self.device)).cpu()
                else:
                    valid_logits = torch.empty((0, 2), dtype=torch.float32)
            train_metrics = _classification_metrics_from_logits(train_logits, train_labels.cpu(), torch.arange(train_labels.numel()))
            if valid_features.size(0) > 0:
                valid_metrics = _classification_metrics_from_logits(valid_logits, valid_labels.cpu(), torch.arange(valid_labels.numel()))
                current_score = (float(valid_metrics["macro_f1"]), -float(valid_metrics["loss"]))
            else:
                valid_metrics = {"loss": float("inf"), "accuracy": 0.0, "macro_f1": 0.0, "bot_f1": 0.0, "count": 0}
                current_score = (float(train_metrics["macro_f1"]), -float(train_metrics["loss"]))

            if current_score > best_score:
                best_score = current_score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_metrics = {
                    "epoch": epoch + 1,
                    "train": train_metrics,
                    "valid": valid_metrics,
                    "train_loss": train_loss,
                }
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is None:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = {
                "epoch": 0,
                "train": _classification_metrics_from_logits(
                    model(train_features.to(self.device)).detach().cpu(),
                    train_labels.cpu(),
                    torch.arange(train_labels.numel()),
                ),
                "valid": {"loss": float("inf"), "accuracy": 0.0, "macro_f1": 0.0, "bot_f1": 0.0, "count": int(valid_labels.numel())},
                "train_loss": 0.0,
            }

        model.load_state_dict(best_state)
        model.to("cpu")
        return model, {
            "best_epoch": int(best_metrics["epoch"]),
            "train_losses": train_losses,
            "best_train": best_metrics["train"],
            "best_valid": best_metrics["valid"],
            "hidden_dim": 128,
            "dropout": 0.1,
            "activation": str(getattr(self.args, "activation", "leakyrelu")).lower(),
        }

    @staticmethod
    def _refiner_bucket_summary(rows, key):
        buckets = {
            "0": lambda x: x == 0,
            "1": lambda x: x == 1,
            "2-5": lambda x: 2 <= x <= 5,
            "6-10": lambda x: 6 <= x <= 10,
            "11+": lambda x: x >= 11,
        }
        result = {}
        for name, fn in buckets.items():
            subset = [row for row in rows if fn(int(row[key]))]
            if not subset:
                continue
            base_wrong = sum(1 for row in subset if row["was_wrong_base"])
            fixed = sum(1 for row in subset if row["was_wrong_base"] and not row["is_wrong_final"])
            base_correct = sum(1 for row in subset if not row["was_wrong_base"])
            broken = sum(1 for row in subset if (not row["was_wrong_base"]) and row["is_wrong_final"])
            result[name] = {
                "count": int(len(subset)),
                "base_wrong": int(base_wrong),
                "fixed": int(fixed),
                "fix_rate": float(fixed / base_wrong) if base_wrong else None,
                "base_correct": int(base_correct),
                "broken": int(broken),
                "break_rate": float(broken / base_correct) if base_correct else None,
            }
        return result

    def _build_refiner_analysis_summary(self, per_node_rows, mode_name):
        rows = list(per_node_rows)
        base_wrong = [row for row in rows if row["was_wrong_base"]]
        base_correct = [row for row in rows if not row["was_wrong_base"]]
        fixed = [row for row in base_wrong if not row["is_wrong_final"]]
        still_wrong = [row for row in base_wrong if row["is_wrong_final"]]
        broken = [row for row in base_correct if row["is_wrong_final"]]
        preserved = [row for row in base_correct if not row["is_wrong_final"]]
        changed = [row for row in rows if row["base_pred"] != row["final_pred"]]
        routed = [row for row in rows if bool(row.get("routed", True))]
        routed_wrong = [row for row in routed if row["was_wrong_base"]]
        routed_correct = [row for row in routed if not row["was_wrong_base"]]
        routed_fixed = [row for row in routed_wrong if not row["is_wrong_final"]]
        routed_broken = [row for row in routed_correct if row["is_wrong_final"]]
        routed_gate_rows = [row for row in routed if "gate_prob" in row]
        gate_positive = [row for row in routed_gate_rows if bool(row.get("gate_decision", False))]
        utility_positive = [row for row in routed_gate_rows if float(row.get("oracle_advantage", 0.0)) > 0.0]
        gate_true_positive = [
            row
            for row in routed_gate_rows
            if bool(row.get("gate_decision", False)) and float(row.get("oracle_advantage", 0.0)) > 0.0
        ]

        def _avg(items, key):
            if not items:
                return 0.0
            return float(sum(float(row[key]) for row in items) / len(items))

        directional_bucket_analysis = None
        if rows and any(
            any(key in row for key in ("has_following", "has_follower", "neighbor_count_following", "neighbor_count_follower"))
            for row in rows
        ):
            directional_bucket_analysis = {}
            directional_buckets = {
                "no_directional_neighbors": lambda row: (int(row.get("has_following", 0)) == 0) and (int(row.get("has_follower", 0)) == 0),
                "following_only": lambda row: (int(row.get("has_following", 0)) == 1) and (int(row.get("has_follower", 0)) == 0),
                "follower_only": lambda row: (int(row.get("has_following", 0)) == 0) and (int(row.get("has_follower", 0)) == 1),
                "both_present": lambda row: (int(row.get("has_following", 0)) == 1) and (int(row.get("has_follower", 0)) == 1),
            }
            for bucket_name, fn in directional_buckets.items():
                subset = [row for row in rows if fn(row)]
                if not subset:
                    continue
                subset_base_wrong = [row for row in subset if row["was_wrong_base"]]
                subset_base_correct = [row for row in subset if not row["was_wrong_base"]]
                subset_fixed = [row for row in subset_base_wrong if not row["is_wrong_final"]]
                subset_broken = [row for row in subset_base_correct if row["is_wrong_final"]]
                directional_bucket_analysis[bucket_name] = {
                    "count": int(len(subset)),
                    "wrong_node_fix_rate": float(len(subset_fixed) / len(subset_base_wrong)) if subset_base_wrong else None,
                    "correct_node_break_rate": float(len(subset_broken) / len(subset_base_correct)) if subset_base_correct else None,
                }

        summary = {
            "mode": str(mode_name),
            "count": int(len(rows)),
            "base_wrong_count": int(len(base_wrong)),
            "base_correct_count": int(len(base_correct)),
            "fixed_wrong_count": int(len(fixed)),
            "still_wrong_count": int(len(still_wrong)),
            "broken_correct_count": int(len(broken)),
            "preserved_correct_count": int(len(preserved)),
            "changed_prediction_count": int(len(changed)),
            "wrong_node_fix_rate": float(len(fixed) / len(base_wrong)) if base_wrong else 0.0,
            "correct_node_break_rate": float(len(broken) / len(base_correct)) if base_correct else 0.0,
            "net_gain": int(len(fixed) - len(broken)),
            "routed_count": int(len(routed)),
            "routed_wrong_count": int(len(routed_wrong)),
            "routed_correct_count": int(len(routed_correct)),
            "routed_wrong_precision": float(len(routed_wrong) / len(routed)) if routed else 0.0,
            "routed_wrong_coverage": float(len(routed_wrong) / len(base_wrong)) if base_wrong else 0.0,
            "conditional_fix_rate_on_selected_wrong": float(len(routed_fixed) / len(routed_wrong)) if routed_wrong else 0.0,
            "conditional_break_rate_on_selected_correct": float(len(routed_broken) / len(routed_correct)) if routed_correct else 0.0,
            "gate_positive_rate": float(len(gate_positive) / len(routed_gate_rows)) if routed_gate_rows else 0.0,
            "mean_gate_prob_routed": _avg(routed_gate_rows, "gate_prob") if routed_gate_rows else 0.0,
            "gate_precision": float(len(gate_true_positive) / len(gate_positive)) if gate_positive else 0.0,
            "gate_recall": float(len(gate_true_positive) / len(utility_positive)) if utility_positive else 0.0,
            "mean_1hop_neighbors_fixed": _avg(fixed, "neighbor_count_1hop"),
            "mean_1hop_neighbors_broken": _avg(broken, "neighbor_count_1hop"),
            "mean_2hop_neighbors_fixed": _avg(fixed, "neighbor_count_2hop"),
            "mean_2hop_neighbors_broken": _avg(broken, "neighbor_count_2hop"),
            "bucket_1hop": self._refiner_bucket_summary(rows, "neighbor_count_1hop"),
            "bucket_2hop": self._refiner_bucket_summary(rows, "neighbor_count_2hop"),
            "directional_bucket_analysis": directional_bucket_analysis,
        }
        return summary

    def _load_semantic_finetune_artifact(self):
        stage_dirs = [
            self.experiment_root / "preparation" / "semantic_encoder",
            self.experiment_root / "stages" / "semantic_encoder_finetune",
            self.experiment_root / "stages" / "semantic_finetune",
        ]
        stage_dir = None
        manifest_path = None
        embeddings_path = None
        outputs_path = None
        for candidate_dir in stage_dirs:
            candidate_manifest = candidate_dir / "manifest.json"
            candidate_embeddings = candidate_dir / "embeddings.pt"
            candidate_outputs = candidate_dir / "outputs.pt"
            if candidate_manifest.exists() and candidate_embeddings.exists() and candidate_outputs.exists():
                stage_dir = candidate_dir
                manifest_path = candidate_manifest
                embeddings_path = candidate_embeddings
                outputs_path = candidate_outputs
                break
        if stage_dir is None:
            fallback_path = (
                getattr(self.args, "embedding_path", None)
                or getattr(self.args, "emb_path", None)
                or getattr(self.args, "g0_feature_path", None)
            )
            if not fallback_path:
                candidate = resolve_seed_aware_roberta_embedding_path(self.data, seed=self.seed)
                if candidate is not None:
                    fallback_path = candidate
            if not fallback_path:
                raise MissingFrozenArtifactError(
                    "Strict GLANCE stages require same-seed semantic_encoder_finetune artifacts or an explicit "
                    "--embedding_path tensor. Implicit seed_1 fallback is disabled."
                )
            fallback_path = Path(fallback_path)
            if not fallback_path.exists():
                raise MissingFrozenArtifactError(
                    "Strict GLANCE stages require semantic_encoder_finetune artifacts or an explicit "
                    f"--embedding_path tensor. Resolved path does not exist: {fallback_path}."
                )
            embeddings = safe_torch_load(fallback_path, map_location="cpu")
            if not torch.is_tensor(embeddings) or embeddings.dim() != 2:
                raise MissingFrozenArtifactError(
                    f"Fallback semantic embedding at {fallback_path} must be a 2D tensor."
                )
            if int(embeddings.shape[0]) != int(len(self.labels)):
                raise MissingFrozenArtifactError(
                    "Fallback semantic embedding does not align with the current dataset node count."
                )
            return {
                "dir": fallback_path.parent,
                "manifest": _direct_embedding_manifest(fallback_path),
                "embeddings": embeddings.detach().cpu().float(),
                "outputs": None,
            }
        manifest = read_json(manifest_path, default={}) or {}
        semantic_backbone = str(manifest.get("semantic_backbone", "")).lower()
        if semantic_backbone not in {"roberta_finetuned", "roberta-f"}:
            raise MissingFrozenArtifactError(
                "glance_oracle_refine requires semantic_finetune artifacts produced by "
                "--semantic_backbone roberta_finetuned."
            )
        embeddings = safe_torch_load(embeddings_path, map_location="cpu")
        outputs = safe_torch_load(outputs_path, map_location="cpu")
        if not torch.is_tensor(embeddings) or embeddings.dim() != 2:
            raise MissingFrozenArtifactError(
                f"semantic_finetune embeddings at {embeddings_path} must be a 2D tensor."
            )
        if int(embeddings.shape[0]) != int(len(self.labels)):
            raise MissingFrozenArtifactError(
                "semantic_finetune embeddings do not align with the current dataset node count."
            )
        return {
            "dir": stage_dir,
            "manifest": manifest,
            "embeddings": embeddings.detach().cpu().float(),
            "outputs": outputs,
        }

    def _load_glance_semantic_source_bundle(self):
        feature_manifest, feature_path = self._strict_joint_feature_manifest()
        override_path = getattr(self.args, "joint_refiner_embedding_path", None)
        requested_path = (
            getattr(self.args, "embedding_path", None)
            or getattr(self.args, "emb_path", None)
            or getattr(self.args, "g0_feature_path", None)
        )
        full_graph_support = str(getattr(self, "graph_data_variant", "labeled")).lower() == "full_graph_support"
        if override_path:
            emb_path = Path(override_path)
            if requested_path:
                requested_path = Path(requested_path)
                if requested_path != feature_path:
                    raise MissingFrozenArtifactError(
                        "joint_router_refinement keeps backbone provenance pinned to the current run's "
                        "graph_detector_prepare artifact even when --joint_refiner_embedding_path is used. "
                        f"Requested --embedding_path {requested_path} does not match frozen_g0 feature_manifest.path {feature_path}."
                    )
            provenance_binding = "joint_refiner_override_same_backbone"
            manifest_source = "joint_refiner_embedding_override"
            semantic_override_role = "refiner_only"
        else:
            if requested_path:
                requested_path = Path(requested_path)
                if requested_path != feature_path:
                    raise MissingFrozenArtifactError(
                        "joint_router_refinement requires semantic input provenance to match the current run's "
                        "graph_detector_prepare artifact. "
                        f"Requested --embedding_path {requested_path} does not match frozen_g0 feature_manifest.path {feature_path}."
                    )
            emb_path = feature_path
            provenance_binding = "strict_same_root_graph_detector_prepare"
            manifest_source = "graph_detector_prepare_feature_manifest"
            semantic_override_role = "shared_backbone_and_refiner"
        if not emb_path.exists():
            raise MissingFrozenArtifactError(
                "joint_router_refinement could not find the requested semantic embedding tensor at "
                f"{emb_path}."
            )
        if full_graph_support and not override_path:
            concat_bundle = _runtime_concat_full_graph_support_embeddings(self.args, self.data, emb_path)
            embeddings = concat_bundle["features"].detach().cpu().float()
            precomputed_views = None
            prompt_expert_bundle = None
            payload_keys = ["runtime_full_graph_concat"]
            if int(embeddings.shape[0]) != int(getattr(self, "graph_node_count", embeddings.shape[0])):
                raise MissingFrozenArtifactError(
                    "joint_router_refinement full_graph_support semantic embeddings do not align with graph_node_count."
                )
            semantic_view_mode = "legacy_inbound_khop"
            extra_manifest = {
                "runtime_concat": True,
                "graph_data_variant": "full_graph_support",
                "labeled_embedding_path": str(concat_bundle["labeled_feature_path"]),
                "support_embedding_path": str(concat_bundle["support_feature_path"]),
                "graph_node_count": int(embeddings.shape[0]),
                "labeled_node_count": int(concat_bundle["labeled_features"].shape[0]),
                "support_node_count": int(concat_bundle["support_features"].shape[0]),
            }
        else:
            payload = safe_torch_load(emb_path, map_location="cpu")
            embeddings, precomputed_views, prompt_expert_bundle, payload_keys = _normalize_semantic_payload(payload)
            if embeddings.dim() != 2:
                raise MissingFrozenArtifactError(
                    f"joint_router_refinement expects --embedding_path to contain a 2-D tensor, got {tuple(embeddings.shape)}."
                )
            expected_rows = int(getattr(self, "graph_node_count", len(self.labels))) if full_graph_support else int(len(self.labels))
            labeled_rows = int(getattr(self, "labeled_node_count", len(self.labels)))
            if full_graph_support:
                valid_rows = {expected_rows, labeled_rows}
                if int(embeddings.shape[0]) not in valid_rows:
                    raise MissingFrozenArtifactError(
                        "joint_router_refinement full_graph_support semantic override must align with either "
                        f"graph_node_count={expected_rows} or labeled_node_count={labeled_rows}, got {int(embeddings.shape[0])}."
                    )
            elif prompt_expert_bundle is None and int(embeddings.shape[0]) != int(len(self.labels)):
                raise MissingFrozenArtifactError(
                    "joint_router_refinement semantic embeddings do not align with the current dataset node count."
                )
            if precomputed_views is not None:
                for view_name, view_tensor in precomputed_views.items():
                    if int(view_tensor.shape[0]) != int(embeddings.shape[0]):
                        raise MissingFrozenArtifactError(
                            f"joint_router_refinement semantic view {view_name} does not align with the semantic override row count."
                        )
            if prompt_expert_bundle is not None:
                payload_full_graph_rows = int(prompt_expert_bundle.get("full_graph_node_count", 0) or 0)
                payload_labeled_rows = int(prompt_expert_bundle.get("labeled_node_count", 0) or 0)
                current_labeled_rows = int(len(self.labels))
                if (
                    not full_graph_support
                    and int(embeddings.shape[0]) != current_labeled_rows
                    and payload_full_graph_rows == int(embeddings.shape[0])
                    and payload_labeled_rows == current_labeled_rows
                ):
                    prompt_expert_bundle = _slice_prompt_expert_bundle_to_labeled_prefix(
                        prompt_expert_bundle,
                        current_labeled_rows,
                    )
                    embeddings = embeddings[:current_labeled_rows].detach().cpu().float()
                semantic_view_mode = str(prompt_expert_bundle.get("semantic_view_mode", "prompt_expert_bundle_v1"))
            elif precomputed_views is not None:
                semantic_view_mode = "precomputed_prompt_views"
            else:
                semantic_view_mode = "legacy_inbound_khop"
            prompt_expert_fusion = str(getattr(self.args, "joint_prompt_expert_fusion", "projector_concat")).lower()
            if prompt_expert_bundle is not None and prompt_expert_fusion == "gaugllm_selector":
                target_node_ids = _coerce_long_tensor_1d(prompt_expert_bundle.get("target_node_ids"))
                if target_node_ids is None:
                    target_node_mask = _coerce_bool_tensor_1d(prompt_expert_bundle.get("target_node_mask"))
                    if target_node_mask is not None:
                        target_node_ids = torch.nonzero(target_node_mask, as_tuple=False).view(-1).long()
                cache_manifest_path = _adjacent_prompt_expert_manifest_path(emb_path)
                cache_manifest, sidecar_paths, explanation_rows = _load_prompt_expert_selector_sidecars(
                    cache_manifest_path,
                    _GAUGLLM_SELECTOR_COMPONENT_ORDER,
                    target_node_ids=target_node_ids,
                )
                prompt_expert_bundle = dict(prompt_expert_bundle)
                prompt_expert_bundle["selector_runtime_cache_manifest_path"] = str(cache_manifest_path)
                prompt_expert_bundle["selector_runtime_cache_manifest"] = cache_manifest
                prompt_expert_bundle["selector_runtime_sidecar_paths"] = sidecar_paths
                prompt_expert_bundle["selector_runtime_explanation_rows"] = explanation_rows
            if not full_graph_support and int(embeddings.shape[0]) != int(len(self.labels)):
                raise MissingFrozenArtifactError(
                    "joint_router_refinement semantic embeddings do not align with the current dataset node count."
                )
            extra_manifest = {
                "graph_data_variant": str(getattr(self, "graph_data_variant", "labeled")),
                "semantic_row_count": int(embeddings.shape[0]),
                "semantic_row_scope": (
                    "labeled_prefix_sliced_from_full_graph_prompt_payload"
                    if isinstance(prompt_expert_bundle, dict)
                    and str(prompt_expert_bundle.get("semantic_row_scope", "")) == "labeled_prefix_sliced_from_full_graph_prompt_payload"
                    else
                    "graph_wide"
                    if int(embeddings.shape[0]) == int(getattr(self, "graph_node_count", len(self.labels)))
                    else "labeled_prefix"
                ),
            }
            if prompt_expert_bundle is not None and prompt_expert_bundle.get("selector_runtime_cache_manifest_path"):
                extra_manifest["adjacent_prompt_expert_manifest_path"] = str(
                    prompt_expert_bundle.get("selector_runtime_cache_manifest_path")
                )
                extra_manifest["selector_runtime_sidecar_paths"] = dict(
                    prompt_expert_bundle.get("selector_runtime_sidecar_paths", {})
                )
                extra_manifest["selector_runtime_component_order"] = list(_GAUGLLM_SELECTOR_COMPONENT_ORDER)
        primary = {
            "dir": emb_path.parent,
            "manifest": {
                **_direct_embedding_manifest(emb_path),
                "source": manifest_source,
                "feature_manifest": feature_manifest,
                "provenance_binding": provenance_binding,
                "semantic_override_role": semantic_override_role,
                "payload_keys": payload_keys,
                "semantic_view_mode": semantic_view_mode,
                "active_components": list(prompt_expert_bundle.get("active_components", [])) if prompt_expert_bundle else [],
                **extra_manifest,
            },
            "embeddings": embeddings,
            "precomputed_views": precomputed_views,
            "prompt_expert_bundle": prompt_expert_bundle,
            "semantic_view_mode": semantic_view_mode,
            "outputs": None,
        }
        bundle = {
            "mode": "single_source",
            "primary": primary,
            "secondary": None,
            "sources": [primary],
        }
        return bundle

    def _strict_glance_train_idx(self, train_idx, cap=3000):
        train_idx = np.asarray(train_idx, dtype=np.int64).reshape(-1)
        cap = int(cap)
        if cap <= 0 or train_idx.size <= cap:
            return train_idx
        rng = np.random.default_rng(int(self.seed))
        selected = np.sort(rng.choice(train_idx, size=cap, replace=False))
        return selected.astype(np.int64)

    def _resolve_joint_router_reuse_stage_dir(self):
        reuse_root = getattr(self.args, "joint_router_reuse_root", None)
        if not reuse_root:
            raise MissingFrozenArtifactError(
                "--joint_routing_protocol frozen_router_reuse requires --joint_router_reuse_root."
            )
        root = Path(reuse_root)
        candidates = [
            root,
            root / "stages" / "joint_router_refinement",
            root / f"seed_{int(self.seed)}" / "stages" / "joint_router_refinement",
        ]
        for candidate in candidates:
            if (candidate / "checkpoint.pt").exists() and (candidate / "manifest.json").exists():
                return candidate
        raise MissingFrozenArtifactError(
            "Could not resolve a joint_router_refinement stage directory from --joint_router_reuse_root. "
            f"Tried: {', '.join(str(item) for item in candidates)}."
        )

    def _load_joint_router_reuse_bundle(self, router_feature_matrix):
        stage_dir = self._resolve_joint_router_reuse_stage_dir()
        manifest = read_json(stage_dir / "manifest.json", default={}) or {}
        metrics = read_json(stage_dir / "metrics.json", default={}) or {}
        checkpoint = safe_torch_load(stage_dir / "checkpoint.pt", map_location="cpu")
        router_state = checkpoint.get("router_model") if isinstance(checkpoint, dict) else None
        if router_state is None:
            raise MissingFrozenArtifactError(
                f"Frozen-router reuse checkpoint at {stage_dir / 'checkpoint.pt'} does not contain router_model."
            )
        router_arch = manifest.get("router_architecture", {}) or {}
        input_dim = int(router_arch.get("input_dim", router_feature_matrix.shape[1]))
        if input_dim != int(router_feature_matrix.shape[1]):
            raise MissingFrozenArtifactError(
                "Frozen-router reuse artifact input_dim does not match the current router feature dimension: "
                f"{input_dim} vs {int(router_feature_matrix.shape[1])}."
            )
        scaler_state = router_arch.get("scaler", {}) or {}
        router_temperature = metrics.get("router_temperature_scaling", {}) or manifest.get("router_temperature_scaling", {}) or {}
        router_model = GlanceReliabilityRouterMLP(
            input_dim=input_dim,
            hidden_dim=int(router_arch.get("hidden_dim", 128)),
            bottleneck_dim=int(router_arch.get("bottleneck_dim", 64)),
            dropout=float(router_arch.get("dropout", 0.1)),
        )
        router_model.load_state_dict(router_state)
        router_model.to("cpu")
        selected_budget = metrics.get("selected_budget")
        if selected_budget is None:
            selected_budget = manifest.get("selected_budget")
        selected_beta = checkpoint.get("selected_beta") if isinstance(checkpoint, dict) else None
        if selected_beta is None:
            selected_beta = metrics.get("selected_beta", manifest.get("selected_beta"))
        if selected_budget is None or selected_beta is None:
            raise MissingFrozenArtifactError(
                "Frozen-router reuse artifact must record selected_budget and selected_beta."
            )
        scaled_router_features = _apply_glance_router_scaler(router_feature_matrix, scaler_state)
        return {
            "stage_dir": stage_dir,
            "manifest": manifest,
            "metrics": metrics,
            "checkpoint": checkpoint,
            "router_model": router_model,
            "router_state": router_state,
            "scaled_router_features": scaled_router_features,
            "scaler_state": scaler_state,
            "selected_budget": float(selected_budget),
            "selected_beta": float(selected_beta),
            "selected_eval_top_k": int(checkpoint.get("selected_eval_top_k", max(int(round(32 / 4.0)), 1))),
            "router_temperature_bundle": router_temperature,
        }

    def _read_joint_router_reuse_metadata(self):
        stage_dir = self._resolve_joint_router_reuse_stage_dir()
        manifest = read_json(stage_dir / "manifest.json", default={}) or {}
        metrics = read_json(stage_dir / "metrics.json", default={}) or {}
        checkpoint = safe_torch_load(stage_dir / "checkpoint.pt", map_location="cpu")
        selected_budget = metrics.get("selected_budget", manifest.get("selected_budget"))
        selected_beta = checkpoint.get("selected_beta") if isinstance(checkpoint, dict) else None
        if selected_beta is None:
            selected_beta = metrics.get("selected_beta", manifest.get("selected_beta"))
        if selected_budget is None or selected_beta is None:
            raise MissingFrozenArtifactError(
                "Frozen-router reuse artifact must record selected_budget and selected_beta."
            )
        return {
            "stage_dir": stage_dir,
            "manifest": manifest,
            "metrics": metrics,
            "checkpoint": checkpoint,
            "router_temperature_bundle": metrics.get("router_temperature_scaling", {}) or manifest.get("router_temperature_scaling", {}) or {},
            "selected_budget": float(selected_budget),
            "selected_beta": float(selected_beta),
        }

    def _run_glance_joint_router_refine(self, stage_dir, base_bundle):
        context = self.ensure_backbone_context()
        semantic_bundle = self._load_glance_semantic_source_bundle()
        semantic = semantic_bundle["primary"]
        gnn_outputs = context["gnn_outputs"]
        p_gnn = gnn_outputs.get("prob")
        logits_gnn = gnn_outputs.get("logits")
        pred_gnn = gnn_outputs.get("pred")
        z_gnn = gnn_outputs.get("node_repr")
        if any(item is None for item in (p_gnn, logits_gnn, pred_gnn, z_gnn)):
            raise MissingFrozenArtifactError(
                "glance_joint_router_refine requires frozen_g0 outputs with prob/logits/pred/node_repr."
            )

        p_gnn = p_gnn.detach().cpu().float()
        logits_gnn = logits_gnn.detach().cpu().float()
        pred_gnn = pred_gnn.detach().cpu().long()
        z_gnn = z_gnn.detach().cpu().float()
        labels_t = torch.tensor(self.labels, dtype=torch.long)
        labels_np = labels_t.numpy()
        train_idx = _idx_numpy(self.data["train_idx"])
        valid_idx = _idx_numpy(self.data["valid_idx"])
        test_idx = _idx_numpy(self.data["test_idx"])
        if train_idx.size == 0 or valid_idx.size == 0 or test_idx.size == 0:
            raise MissingFrozenArtifactError(
                "glance_joint_router_refine requires non-empty train/valid/test splits."
            )

        configured_train_cap = int(getattr(self.args, "joint_train_node_cap", 3000))
        strict_train_idx = self._strict_glance_train_idx(train_idx, cap=configured_train_cap)
        train_cap_mode = "full_train_split" if configured_train_cap <= 0 else "capped_train_subset"
        semantic_view_mode = semantic.get("semantic_view_mode", semantic["manifest"].get("semantic_view_mode", "legacy_inbound_khop"))
        if _is_prompt_expert_semantic_view_mode(semantic_view_mode):
            refiner_features, semantic_views = self._build_prompt_expert_semantic_views(
                semantic.get("prompt_expert_bundle"),
                z_gnn,
                base_prob=p_gnn,
            )
            prompt_expert_active_components = list(refiner_features.get("active_components", []))
        else:
            semantic_views = self._build_k_hop_semantic_views(
                semantic.get("precomputed_views") if semantic.get("precomputed_views") is not None else semantic["embeddings"],
                self.data.get("edge_index"),
            )
            refiner_features = torch.cat(
                [z_gnn, semantic_views["ego"], semantic_views["hop1"], semantic_views["hop2"]],
                dim=1,
            ).detach().cpu().float()
            prompt_expert_active_components = []
        refiner_explicit_gate = bool(getattr(self.args, "joint_refiner_explicit_gate", False))
        refiner_target_mode = str(getattr(self.args, "joint_refiner_target_mode", "predict")).lower()
        refiner_gate_target = str(getattr(self.args, "joint_refiner_gate_target", "base_wrong")).lower()
        refiner_weight_mode = str(getattr(self.args, "joint_refiner_weight_mode", "off")).lower()
        refiner_base_wrong_weight = float(getattr(self.args, "joint_refiner_base_wrong_weight", 2.0))
        refiner_utility_weight = float(getattr(self.args, "joint_refiner_utility_weight", 3.0))
        refiner_gate_weight = float(getattr(self.args, "joint_refiner_gate_weight", 0.5))
        prompt_expert_fusion = str(getattr(self.args, "joint_prompt_expert_fusion", "projector_concat")).lower()
        selector_component_order = (
            list(refiner_features.get("selector_component_order", _selector_component_order_for_fusion(prompt_expert_fusion)))
            if isinstance(refiner_features, dict)
            else list(_selector_component_order_for_fusion(prompt_expert_fusion))
        )
        if prompt_expert_fusion != "projector_concat" and not _is_prompt_expert_semantic_view_mode(semantic_view_mode):
            raise ValueError("--joint_prompt_expert_fusion only applies to prompt-expert semantic payloads.")
        joint_routing_protocol = str(getattr(self.args, "joint_routing_protocol", "joint_train")).lower()
        frozen_router_metadata = self._read_joint_router_reuse_metadata() if joint_routing_protocol == "frozen_router_reuse" else None

        original_node_features_bundle = self._strict_glance_original_node_features()
        q_bundle = self._fit_strict_glance_q_probs(
            original_node_features_bundle["features"],
            strict_train_idx,
            valid_idx,
        )
        backbone_input_features = self._strict_glance_backbone_input_features()
        mc_bundle = self._strict_glance_mc_dropout_uncertainty(backbone_input_features)
        if frozen_router_metadata is not None and frozen_router_metadata.get("router_temperature_bundle", {}).get("temperature") is not None:
            router_temperature_bundle = dict(frozen_router_metadata["router_temperature_bundle"])
        else:
            router_temperature_bundle = fit_reliability_temperature(
                logits_gnn[valid_idx],
                labels_np[valid_idx],
            )
        router_feature_bundle = self._build_strict_glance_router_features(
            logits_gnn=logits_gnn,
            p_gnn=p_gnn,
            z_gnn=z_gnn,
            original_node_features=original_node_features_bundle["features"],
            q_bundle=q_bundle,
            mc_bundle=mc_bundle,
            calibration_temperature=router_temperature_bundle["temperature"],
        )
        router_feature_bundle["full_feature_bundle"]["temperature_scaling"] = dict(router_temperature_bundle)
        if frozen_router_metadata is not None:
            frozen_router_bundle = self._load_joint_router_reuse_bundle(router_feature_bundle["features"])
            router_features = frozen_router_bundle["scaled_router_features"]
            scaler_state = frozen_router_bundle["scaler_state"]
        else:
            frozen_router_bundle = None
            router_features, scaler_state = self._standardize_glance_router_features(
                router_feature_bundle["features"],
                strict_train_idx,
            )

        activation = str(getattr(self.args, "activation", "leakyrelu")).lower()
        batch_size = 32
        max_epochs = 10
        patience = 2
        decay_factor = 0.5
        entropy_weight = 0.0
        router_weight = 0.0
        learning_rate = float(getattr(self.args, "lr_GNN", 5e-4))
        weight_decay = float(getattr(self.args, "weight_decay_GNN", 1e-5))
        eval_top_k = (
            int(frozen_router_bundle["selected_eval_top_k"])
            if frozen_router_bundle is not None
            else max(int(round(batch_size / 4.0)), 1)
        )
        beta_candidates = (
            (float(frozen_router_bundle["selected_beta"]),)
            if frozen_router_bundle is not None
            else (0.1, 0.2, 0.3)
        )

        beta_runs = []
        selected_beta_run = None
        selected_beta_score = None
        for beta in beta_candidates:
            run = self._train_glance_joint_candidate(
                labels_t=labels_t,
                labels_np=labels_np,
                train_idx=strict_train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
                logits_gnn=logits_gnn,
                p_gnn=p_gnn,
                pred_gnn=pred_gnn,
                router_features=router_features,
                refiner_features=refiner_features,
                semantic_views=semantic_views,
                semantic_view_mode=semantic_view_mode,
                activation=activation,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                decay_factor=decay_factor,
                entropy_weight=entropy_weight,
                router_weight=router_weight,
                router_regression_weight=0.0,
                router_ranking_weight=1.0,
                router_calibration_weight=0.25,
                router_reliability_weight=0.0,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                beta=float(beta),
                eval_top_k=eval_top_k,
                refiner_explicit_gate=refiner_explicit_gate,
                refiner_target_mode=refiner_target_mode,
                refiner_gate_target=refiner_gate_target,
                refiner_weight_mode=refiner_weight_mode,
                refiner_base_wrong_weight=refiner_base_wrong_weight,
                refiner_utility_weight=refiner_utility_weight,
                refiner_gate_weight=refiner_gate_weight,
                prompt_expert_fusion=prompt_expert_fusion,
                routing_protocol=joint_routing_protocol,
                frozen_router_bundle=frozen_router_bundle,
                selected_budget_override=(float(frozen_router_bundle["selected_budget"]) if frozen_router_bundle is not None else None),
            )
            beta_runs.append(run)
            beta_score = (
                float(run["selected_budget_metrics"]["valid"]["macro_f1"]),
                -float(run["selected_budget_metrics"]["valid"]["loss"]),
            )
            if selected_beta_score is None or beta_score > selected_beta_score:
                selected_beta_score = beta_score
                selected_beta_run = run

        if selected_beta_run is None:
            raise MissingFrozenArtifactError("glance_joint_router_refine could not select a beta on validation.")

        final_run = selected_beta_run
        selected_beta = float(final_run["beta"])
        selected_budget = float(final_run["selected_budget"])
        selected_budget_key = str(final_run["selected_budget_key"])
        train_outputs = final_run["train_outputs"]
        valid_outputs = final_run["valid_outputs"]
        test_outputs = final_run["test_outputs"]
        final_logits = logits_gnn.clone()
        final_prob = p_gnn.clone()
        final_pred = pred_gnn.clone()
        routed_masks = {
            "train": train_outputs["routed_mask"],
            "valid": valid_outputs["routed_mask"],
            "test": test_outputs["routed_mask"],
        }
        eligible_refiner_masks = {
            "train": train_outputs.get("eligible_refiner_mask", torch.ones(pred_gnn.numel(), dtype=torch.bool)),
            "valid": valid_outputs.get("eligible_refiner_mask", torch.ones(pred_gnn.numel(), dtype=torch.bool)),
            "test": test_outputs.get("eligible_refiner_mask", torch.ones(pred_gnn.numel(), dtype=torch.bool)),
        }
        router_prob = torch.zeros(pred_gnn.numel(), dtype=torch.float32)
        router_score = torch.zeros(pred_gnn.numel(), dtype=torch.float32)
        oracle_advantage = torch.zeros(pred_gnn.numel(), dtype=torch.float32)
        gate_prob = torch.zeros(pred_gnn.numel(), dtype=torch.float32)
        selector_component_order = (
            list(train_outputs.get("selector_component_order", []))
            or list(valid_outputs.get("selector_component_order", []))
            or list(test_outputs.get("selector_component_order", []))
            or list(PromptExpertBundleRefinerMLP.MPE_COMPONENT_ORDER)
        )
        mpe_gate_weights = torch.zeros((pred_gnn.numel(), len(selector_component_order)), dtype=torch.float32)
        mpe_selected_expert = torch.full((pred_gnn.numel(),), -1, dtype=torch.long)
        for split_outputs, split_idx in (
            (train_outputs, train_idx),
            (valid_outputs, valid_idx),
            (test_outputs, test_idx),
        ):
            split_mask = split_outputs["routed_mask"]
            split_idx_t = torch.tensor(split_idx, dtype=torch.long)
            final_logits[split_mask] = split_outputs["logits"][split_mask]
            final_prob[split_mask] = split_outputs["prob"][split_mask]
            final_pred[split_mask] = split_outputs["pred"][split_mask]
            router_prob[split_idx_t] = split_outputs["router_prob"][split_idx_t]
            router_score[split_idx_t] = split_outputs["router_score"][split_idx_t]
            oracle_advantage[split_idx_t] = split_outputs["oracle_advantage"][split_idx_t]
            if split_outputs.get("gate_prob") is not None:
                gate_prob[split_idx_t] = split_outputs["gate_prob"][split_idx_t]
            if split_outputs.get("mpe_gate_weights") is not None:
                mpe_gate_weights[split_idx_t] = split_outputs["mpe_gate_weights"][split_idx_t]
            if split_outputs.get("mpe_selected_expert") is not None:
                mpe_selected_expert[split_idx_t] = split_outputs["mpe_selected_expert"][split_idx_t].long()

        pred_gnn_labeled = _labeled_prefix_pred(pred_gnn.numpy(), labels_np.shape[0])
        final_pred_labeled = _labeled_prefix_pred(final_pred.numpy(), labels_np.shape[0])
        base_test = _score_all(labels_np[test_idx], pred_gnn_labeled[test_idx])
        overall_test = _score_all(labels_np[test_idx], final_pred_labeled[test_idx])
        test_delta = _delta_table(pred_gnn_labeled, final_pred_labeled, labels_np, self.test_mask)
        per_node_rows = final_run["test_rows"]["rows"]
        analysis_summary = final_run["test_rows"]["analysis"]
        test_idx_t = torch.tensor(test_idx, dtype=torch.long)
        selected_expert_test = mpe_selected_expert[test_idx_t].clone()
        selector_weights_test = mpe_gate_weights[test_idx_t].clone()
        selected_expert_name_test = [
            (
                selector_component_order[int(expert_idx)]
                if 0 <= int(expert_idx) < len(selector_component_order)
                else ""
            )
            for expert_idx in selected_expert_test.tolist()
        ]

        beta_sweep = []
        for run in beta_runs:
            beta_sweep.append({
                "beta": float(run["beta"]),
                "eval_top_k": int(run["eval_top_k"]),
                "best_epoch": int(run["best_epoch"]),
                "selected_budget": float(run["selected_budget"]),
                "selected_budget_key": str(run["selected_budget_key"]),
                "valid_accuracy": float(run["selected_budget_metrics"]["valid"]["accuracy"]),
                "valid_macro_f1": float(run["selected_budget_metrics"]["valid"]["macro_f1"]),
                "valid_bot_f1": float(run["selected_budget_metrics"]["valid"]["bot_f1"]),
                "valid_loss": float(run["selected_budget_metrics"]["valid"]["loss"]),
                "valid_routed_count": int(run["selected_budget_metrics"]["valid"]["routed_count"]),
                "valid_query_rate": float(run["selected_budget_metrics"]["valid"]["query_rate"]),
                "valid_net_gain": int(run["valid_rows"]["analysis"]["net_gain"]),
                "valid_routed_wrong_precision": float(run["valid_rows"]["analysis"]["routed_wrong_precision"]),
                "valid_routed_wrong_coverage": float(run["valid_rows"]["analysis"]["routed_wrong_coverage"]),
                "valid_conditional_fix_rate": float(run["valid_rows"]["analysis"]["conditional_fix_rate_on_selected_wrong"]),
                "valid_gate_positive_rate": float(run["valid_rows"]["analysis"].get("gate_positive_rate", 0.0)),
                "valid_mean_gate_prob_routed": float(run["valid_rows"]["analysis"].get("mean_gate_prob_routed", 0.0)),
                "valid_gate_precision": float(run["valid_rows"]["analysis"].get("gate_precision", 0.0)),
                "valid_gate_recall": float(run["valid_rows"]["analysis"].get("gate_recall", 0.0)),
                "test_accuracy": float(run["selected_budget_metrics"]["test"]["accuracy"]),
                "test_macro_f1": float(run["selected_budget_metrics"]["test"]["macro_f1"]),
                "test_bot_f1": float(run["selected_budget_metrics"]["test"]["bot_f1"]),
                "test_loss": float(run["selected_budget_metrics"]["test"]["loss"]),
                "test_routed_count": int(run["selected_budget_metrics"]["test"]["routed_count"]),
                "test_query_rate": float(run["selected_budget_metrics"]["test"]["query_rate"]),
                "test_net_gain": int(run["test_rows"]["analysis"]["net_gain"]),
                "test_routed_wrong_precision": float(run["test_rows"]["analysis"]["routed_wrong_precision"]),
                "test_routed_wrong_coverage": float(run["test_rows"]["analysis"]["routed_wrong_coverage"]),
                "test_conditional_fix_rate": float(run["test_rows"]["analysis"]["conditional_fix_rate_on_selected_wrong"]),
                "test_gate_positive_rate": float(run["test_rows"]["analysis"].get("gate_positive_rate", 0.0)),
                "test_mean_gate_prob_routed": float(run["test_rows"]["analysis"].get("mean_gate_prob_routed", 0.0)),
                "test_gate_precision": float(run["test_rows"]["analysis"].get("gate_precision", 0.0)),
                "test_gate_recall": float(run["test_rows"]["analysis"].get("gate_recall", 0.0)),
            })

        valid_budget_curve = final_run["valid_budget_curve"]
        test_budget_curve = final_run["test_budget_curve"]
        if getattr(self.args, "joint_refiner_embedding_path", None):
            dependency_alignment_note = (
                "joint_router_refinement keeps backbone provenance pinned to the current run's "
                "preparation/graph_detector artifact while allowing a refiner-only semantic override through "
                "--joint_refiner_embedding_path."
            )
        else:
            dependency_alignment_note = (
                "Public joint_router_refinement is bound to the current run's preparation/graph_detector artifact "
                "and reuses its recorded feature_manifest.path as the semantic tensor source."
            )

        metrics_payload = {
            "contract": "glance_joint_router_refine_metrics_v1",
            "routing_protocol": joint_routing_protocol,
            "graph_data_variant": str(getattr(self, "graph_data_variant", "labeled")),
            "graph_node_count": int(getattr(self, "graph_node_count", len(self.labels))),
            "labeled_node_count": int(getattr(self, "labeled_node_count", len(self.labels))),
            "support_node_count": int(getattr(self, "support_node_count", 0)),
            "routing_mode": "joint_topk_train_global_budget_eval_router_refiner",
            "paper_faithful_glance_inspired": False,
            "glance_style_joint_training": True,
            "strict_training_protocol_alignment": bool(configured_train_cap == 3000),
            "paper_text_router_alignment": False,
            "twibot20_adapted_training": bool(configured_train_cap <= 0),
            "strict_dependency_provenance_alignment": True,
            "semantic_alignment": False,
            "not_official_reproduction": True,
            "paper_text_evaluation_alignment": False,
            "evaluation_protocol": "validation_selected_global_budget_by_router_score",
            "evaluation_alignment_note": "Training keeps paper-text batch top-k routing; final public evaluation selects a global budget on validation and locks that budget on test for TwiBot20-suitable score-based routing.",
            "dependency_alignment_note": dependency_alignment_note,
            "semantic_source_mode": semantic_bundle["mode"],
            "semantic_sources": [item["manifest"].get("source_identity", item["manifest"].get("lm_model")) for item in semantic_bundle["sources"]],
            "primary_semantic_manifest": semantic_bundle["primary"]["manifest"],
            "joint_refiner_embedding_path": str(getattr(self.args, "joint_refiner_embedding_path", None) or ""),
            "semantic_view_mode": semantic_view_mode,
            "external_frozen_g0_root": str(getattr(self.args, "external_frozen_g0_root", None) or ""),
            "selected_beta_source": "validation",
            "selected_beta": selected_beta,
            "beta_candidates": [float(item) for item in beta_candidates],
            "selected_budget_source": "validation",
            "selected_budget": selected_budget,
            "selected_budget_key": selected_budget_key,
            "router_reuse_root": str(getattr(self.args, "joint_router_reuse_root", None) or ""),
            "router_reuse_selected_budget": float(frozen_router_bundle["selected_budget"]) if frozen_router_bundle is not None else None,
            "router_reuse_selected_beta": float(frozen_router_bundle["selected_beta"]) if frozen_router_bundle is not None else None,
            "fixed_router_comparable": bool(joint_routing_protocol == "frozen_router_reuse"),
            "selected_valid_budget_metrics": final_run["selected_valid_budget_metrics"],
            "selected_test_budget_metrics_under_valid_choice": final_run["selected_test_budget_metrics_under_valid_choice"],
            "selected_beta_valid_metrics": final_run["selected_budget_metrics"]["valid"],
            "selected_beta_test_metrics": final_run["selected_budget_metrics"]["test"],
            "beta_sweep": beta_sweep,
            "valid_budget_curve": valid_budget_curve,
            "test_budget_curve": test_budget_curve,
            "router_training_objective": "base_wrong_reliability_bce_plus_pairwise_ranking",
            "oracle_advantage_semantics": "loss_gnn_minus_loss_refiner_minus_beta",
            "router_score_semantics": "learned_base_wrong_reliability_score_for_global_budget_routing",
            "router_reliability_semantics": "primary_base_wrong_probability_estimator",
            "router_feature_family": router_feature_bundle["full_feature_bundle"]["feature_family"],
            "router_feature_names": list(router_feature_bundle["feature_names"]),
            "router_temperature_scaling": dict(router_temperature_bundle),
            "refiner_target_mode": refiner_target_mode,
            "joint_refiner_gate_target": refiner_gate_target,
            "refiner_weight_mode": refiner_weight_mode,
            "refiner_explicit_gate": bool(refiner_explicit_gate),
            "refiner_base_wrong_weight": float(refiner_base_wrong_weight),
            "refiner_utility_weight": float(refiner_utility_weight),
            "refiner_gate_weight": float(refiner_gate_weight),
            "joint_prompt_expert_fusion": prompt_expert_fusion,
            "mpe_gate_component_order": list(selector_component_order),
            "prompt_expert_selector_component_order": list(selector_component_order),
            "prompt_expert_target_mask_source": (
                str(refiner_features.get("target_mask_source", "all_nodes_default"))
                if isinstance(refiner_features, dict)
                else None
            ),
            "prompt_expert_selector_context_source": (
                str(refiner_features.get("selector_context_source", ""))
                if isinstance(refiner_features, dict)
                else ""
            ),
            "advantage_router_diagnostics": final_run["fit_summary"].get("advantage_router_diagnostics", {}),
            "base_test": base_test,
            "overall_test": overall_test,
            "overall_delta_vs_base_gnn": {
                "accuracy": float(overall_test["accuracy"] - base_test["accuracy"]),
                "macro_f1": float(overall_test["macro_f1"] - base_test["macro_f1"]),
                "bot_f1": float(overall_test["bot_f1"] - base_test["bot_f1"]),
            },
            "improved_count": int(test_delta["fix"]),
            "degraded_count": int(test_delta["broke"]),
            "net_gain": int(test_delta["net"]),
            "wrong_node_fix_rate": float(test_delta["fix"] / max(int((pred_gnn_labeled[test_idx] != labels_np[test_idx]).sum()), 1)),
            "correct_node_break_rate": float(
                test_delta["broke"]
                / max(int(test_idx.size - int((pred_gnn_labeled[test_idx] != labels_np[test_idx]).sum())), 1)
            ),
            "gate_positive_rate": float(analysis_summary.get("gate_positive_rate", 0.0)),
            "mean_gate_prob_routed": float(analysis_summary.get("mean_gate_prob_routed", 0.0)),
            "gate_precision": float(analysis_summary.get("gate_precision", 0.0)),
            "gate_recall": float(analysis_summary.get("gate_recall", 0.0)),
            "query_usage": {
                "train_routed_count": int(train_outputs["routed_count"]),
                "train_eligible_count": int(train_outputs.get("eligible_count", 0)),
                "valid_routed_count": int(valid_outputs["routed_count"]),
                "valid_eligible_count": int(valid_outputs.get("eligible_count", 0)),
                "test_routed_count": int(test_outputs["routed_count"]),
                "test_eligible_count": int(test_outputs.get("eligible_count", 0)),
                "selected_budget": selected_budget,
                "eval_top_k": int(eval_top_k),
                "batch_size": int(batch_size),
            },
            "fit_summary": final_run["fit_summary"],
        }
        router_performance_summary = {
            "selected_beta": float(selected_beta),
            "selected_budget": float(selected_budget),
            "selected_budget_key": str(selected_budget_key),
            "routing_protocol": joint_routing_protocol,
            "router_temperature": float(router_temperature_bundle["temperature"]),
            "router_temperature_bundle": dict(router_temperature_bundle),
            "router_feature_family": str(router_feature_bundle["full_feature_bundle"]["feature_family"]),
            "router_input_dim": int(router_features.shape[1]),
            "router_diagnostics": final_run["fit_summary"].get("router_diagnostics", {}),
            "advantage_router_diagnostics": final_run["fit_summary"].get("advantage_router_diagnostics", {}),
            "topk_reference_router_diagnostics": final_run["fit_summary"].get("topk_reference", {}).get("router_diagnostics", {}),
            "topk_reference_advantage_router_diagnostics": final_run["fit_summary"].get("topk_reference", {}).get("advantage_router_diagnostics", {}),
            "selected_valid_budget_metrics": final_run["selected_budget_metrics"]["valid"],
            "selected_test_budget_metrics": final_run["selected_budget_metrics"]["test"],
            "selected_valid_router_analysis": final_run["valid_rows"]["analysis"],
            "selected_test_router_analysis": final_run["test_rows"]["analysis"],
            "overall_delta_vs_base_gnn": metrics_payload["overall_delta_vs_base_gnn"],
            "wrong_node_fix_rate": metrics_payload["wrong_node_fix_rate"],
            "correct_node_break_rate": metrics_payload["correct_node_break_rate"],
            "gate_positive_rate": metrics_payload["gate_positive_rate"],
            "mean_gate_prob_routed": metrics_payload["mean_gate_prob_routed"],
            "gate_precision": metrics_payload["gate_precision"],
            "gate_recall": metrics_payload["gate_recall"],
            "net_gain": metrics_payload["net_gain"],
            "best_epoch": int(final_run["best_epoch"]),
            "query_usage": metrics_payload["query_usage"],
        }
        router_performance_row = {
            "selected_beta": float(selected_beta),
            "selected_budget": float(selected_budget),
            "best_epoch": int(final_run["best_epoch"]),
            "router_temperature": float(router_temperature_bundle["temperature"]),
            "router_valid_positive_rate": final_run["fit_summary"]["router_diagnostics"]["valid"].get("positive_rate"),
            "router_valid_mean_score": final_run["fit_summary"]["router_diagnostics"]["valid"].get("mean_score"),
            "router_valid_auroc": final_run["fit_summary"]["router_diagnostics"]["valid"].get("auroc"),
            "router_valid_auprc": final_run["fit_summary"]["router_diagnostics"]["valid"].get("auprc"),
            "router_test_positive_rate": final_run["fit_summary"]["router_diagnostics"]["test"].get("positive_rate"),
            "router_test_mean_score": final_run["fit_summary"]["router_diagnostics"]["test"].get("mean_score"),
            "router_test_auroc": final_run["fit_summary"]["router_diagnostics"]["test"].get("auroc"),
            "router_test_auprc": final_run["fit_summary"]["router_diagnostics"]["test"].get("auprc"),
            "router_valid_adv_auroc": final_run["fit_summary"]["advantage_router_diagnostics"]["valid"].get("auroc"),
            "router_valid_adv_auprc": final_run["fit_summary"]["advantage_router_diagnostics"]["valid"].get("auprc"),
            "router_test_adv_auroc": final_run["fit_summary"]["advantage_router_diagnostics"]["test"].get("auroc"),
            "router_test_adv_auprc": final_run["fit_summary"]["advantage_router_diagnostics"]["test"].get("auprc"),
            "valid_query_rate": float(final_run["selected_budget_metrics"]["valid"]["query_rate"]),
            "valid_routed_count": int(final_run["selected_budget_metrics"]["valid"]["routed_count"]),
            "valid_routed_wrong_precision": float(final_run["valid_rows"]["analysis"]["routed_wrong_precision"]),
            "valid_routed_wrong_coverage": float(final_run["valid_rows"]["analysis"]["routed_wrong_coverage"]),
            "valid_conditional_fix_rate": float(final_run["valid_rows"]["analysis"]["conditional_fix_rate_on_selected_wrong"]),
            "test_query_rate": float(final_run["selected_budget_metrics"]["test"]["query_rate"]),
            "test_routed_count": int(final_run["selected_budget_metrics"]["test"]["routed_count"]),
            "test_routed_wrong_precision": float(final_run["test_rows"]["analysis"]["routed_wrong_precision"]),
            "test_routed_wrong_coverage": float(final_run["test_rows"]["analysis"]["routed_wrong_coverage"]),
            "test_conditional_fix_rate": float(final_run["test_rows"]["analysis"]["conditional_fix_rate_on_selected_wrong"]),
            "test_wrong_node_fix_rate": float(metrics_payload["wrong_node_fix_rate"]),
            "test_correct_node_break_rate": float(metrics_payload["correct_node_break_rate"]),
            "test_gate_positive_rate": float(metrics_payload["gate_positive_rate"]),
            "test_mean_gate_prob_routed": float(metrics_payload["mean_gate_prob_routed"]),
            "test_gate_precision": float(metrics_payload["gate_precision"]),
            "test_gate_recall": float(metrics_payload["gate_recall"]),
            "test_net_gain": int(metrics_payload["net_gain"]),
            "test_macro_f1": float(metrics_payload["overall_test"]["macro_f1"]),
            "test_delta_macro_f1": float(metrics_payload["overall_delta_vs_base_gnn"]["macro_f1"]),
        }
        manifest = {
            "contract": "glance_joint_router_refine_v1",
            "status": "completed",
            "graph_data_variant": str(getattr(self, "graph_data_variant", "labeled")),
            "node_id_manifest": {
                "num_nodes": int(getattr(self, "graph_node_count", len(self.labels))),
                "graph_node_count": int(getattr(self, "graph_node_count", len(self.labels))),
                "labeled_node_count": int(getattr(self, "labeled_node_count", len(self.labels))),
                "support_node_count": int(getattr(self, "support_node_count", 0)),
                "labels_sha256": tensor_sha256(self.data["labels"]),
            },
            "routing_mode": "joint_topk_train_global_budget_eval_router_refiner",
            "paper_faithful_glance_inspired": False,
            "glance_style_joint_training": True,
            "strict_training_protocol_alignment": bool(configured_train_cap == 3000),
            "paper_text_router_alignment": False,
            "twibot20_adapted_training": bool(configured_train_cap <= 0),
            "strict_dependency_provenance_alignment": True,
            "semantic_alignment": False,
            "not_official_reproduction": True,
            "paper_text_evaluation_alignment": False,
            "evaluation_protocol": "validation_selected_global_budget_by_router_score",
            "evaluation_alignment_note": "Batch top-k is kept only inside strict GLANCE training. Public final evaluation uses a validation-selected global budget and test-locked router ranking.",
            "research_positioning": "task_adapted_reliability_router_under_glance_style_joint_training",
            "routing_protocol": joint_routing_protocol,
            "selected_beta_source": "validation",
            "selected_beta": selected_beta,
            "beta_candidates": [float(item) for item in beta_candidates],
            "selected_budget_source": "validation",
            "selected_budget": selected_budget,
            "selected_budget_key": selected_budget_key,
            "router_reuse_root": str(getattr(self.args, "joint_router_reuse_root", None) or ""),
            "router_reuse_selected_budget": float(frozen_router_bundle["selected_budget"]) if frozen_router_bundle is not None else None,
            "router_reuse_selected_beta": float(frozen_router_bundle["selected_beta"]) if frozen_router_bundle is not None else None,
            "fixed_router_comparable": bool(joint_routing_protocol == "frozen_router_reuse"),
            "semantic_source_mode": semantic_bundle["mode"],
            "semantic_sources": [item["manifest"].get("source_identity", item["manifest"].get("lm_model")) for item in semantic_bundle["sources"]],
            "semantic_source": "semantic_single_source",
            "semantic_alignment_note": "Joint training keeps the current cached semantic embedding path; router objective is task-adapted toward reliability rather than the paper's advantage target.",
            "dependency_alignment_note": dependency_alignment_note,
            "semantic_manifest": semantic["manifest"],
            "primary_semantic_manifest": semantic_bundle["primary"]["manifest"],
            "joint_refiner_embedding_path": str(getattr(self.args, "joint_refiner_embedding_path", None) or ""),
            "external_frozen_g0_root": str(getattr(self.args, "external_frozen_g0_root", None) or ""),
            "frozen_g0_dir": str(context["frozen_g0"]["dir"]),
            "frozen_g0_manifest": context["frozen_g0"]["manifest"],
            "train_node_cap": int(configured_train_cap),
            "effective_train_node_count": int(strict_train_idx.size),
            "full_train_node_count": int(train_idx.size),
            "train_cap_mode": train_cap_mode,
            "uncertainty_source": mc_bundle["source"],
            "router_temperature_scaling": dict(router_temperature_bundle),
            "router_feature_family_used": list(router_feature_bundle["feature_names"]),
            "router_architecture": {
                "type": "reliability_first_router_mlp",
                "hidden_dim": 128,
                "bottleneck_dim": 64,
                "dropout": 0.1,
                "input_dim": int(router_features.shape[1]),
                "feature_count": int(router_features.shape[1]),
                "feature_family": list(router_feature_bundle["feature_names"]),
                "feature_bundle_metadata": router_feature_bundle["full_feature_bundle"],
                "scaler": scaler_state,
            },
            "refiner_architecture": {
                "type": (
                    "prompt_expert_bundle_refiner_mlp"
                    if _is_prompt_expert_semantic_view_mode(semantic_view_mode)
                    else ("gated_glance_joint_refiner_mlp" if refiner_explicit_gate else "glance_joint_refiner_mlp")
                ),
                "input": (
                    "[z_gnn || mpe_gated_sum(graph_following, graph_follower, tweet, conflict) || structural_side_channel]"
                    if _is_prompt_expert_semantic_view_mode(semantic_view_mode) and prompt_expert_fusion == "mpe_gated"
                    else "[z_gnn || gaugllm_selector_fused(graph_following, graph_follower, tweet, conflict) || metadata_structured_proj || structural_side_channel]"
                    if _is_prompt_expert_semantic_view_mode(semantic_view_mode) and prompt_expert_fusion == "gaugllm_selector"
                    else (
                        "[z_gnn || "
                        + " || ".join(PromptExpertBundleRefinerMLP.RAW_CONCAT_COMPONENTS[prompt_expert_fusion])
                        + " || structural_side_channel]"
                    )
                    if _is_prompt_expert_semantic_view_mode(semantic_view_mode)
                    and prompt_expert_fusion in PromptExpertBundleRefinerMLP.RAW_CONCAT_COMPONENTS
                    else "[z_gnn || ego_proj || graph_following_proj || graph_follower_proj || graph_fused || tweet_proj || conflict_proj || metadata_structured_proj || structural_side_channel]"
                    if _is_prompt_expert_semantic_view_mode(semantic_view_mode)
                    else "[z_gnn || z_sem_0 || z_sem_1 || z_sem_2]"
                ),
                "hidden_dim": 128,
                "activation": activation,
                "dropout": 0.1,
                "output_dim": int(logits_gnn.shape[1]),
                "semantic_view_mode": semantic_view_mode,
                "feature_kind": (
                    refiner_features.get("feature_kind", "prompt_expert_bundle")
                    if _is_prompt_expert_semantic_view_mode(semantic_view_mode) and isinstance(refiner_features, dict)
                    else "flat_concat"
                ),
                "semantic_view_names": (
                    list(_PROMPT_EXPERT_COMPONENT_ORDER)
                    if _is_prompt_expert_semantic_view_mode(semantic_view_mode)
                    else ["ego", "hop1", "hop2"]
                ),
                "active_components": prompt_expert_active_components,
                "proj_dim": 256
                if _is_prompt_expert_semantic_view_mode(semantic_view_mode)
                and prompt_expert_fusion in {"projector_concat", "mpe_gated", "gaugllm_selector"}
                else None,
                "fusion_mode": prompt_expert_fusion,
                "raw_concat_components": list(PromptExpertBundleRefinerMLP.RAW_CONCAT_COMPONENTS.get(prompt_expert_fusion, ()))
                if _is_prompt_expert_semantic_view_mode(semantic_view_mode)
                else None,
                "mpe_gate_component_order": list(selector_component_order)
                if _is_prompt_expert_semantic_view_mode(semantic_view_mode)
                else None,
                "mpe_gate_input": (
                    list(refiner_features.get("mpe_gate_feature_names", []))
                    if _is_prompt_expert_semantic_view_mode(semantic_view_mode) and isinstance(refiner_features, dict)
                    else None
                ),
                "selector_context_reuse": (
                    prompt_expert_fusion == "gaugllm_selector"
                    and _is_prompt_expert_semantic_view_mode(semantic_view_mode)
                ),
                "selector_context_source": (
                    str(refiner_features.get("selector_context_source", ""))
                    if _is_prompt_expert_semantic_view_mode(semantic_view_mode) and isinstance(refiner_features, dict)
                    else ""
                ),
                "graph_gate_input": (
                    list(refiner_features.get("graph_gate_feature_names", []))
                    if _is_prompt_expert_semantic_view_mode(semantic_view_mode) and isinstance(refiner_features, dict)
                    else None
                ),
                "structural_side_channel": (
                    list(refiner_features.get("structural_feature_names", []))
                    if _is_prompt_expert_semantic_view_mode(semantic_view_mode) and isinstance(refiner_features, dict)
                    else None
                ),
                "target_mode": refiner_target_mode,
                "explicit_gate": bool(refiner_explicit_gate),
                "gate_target": refiner_gate_target,
                "weight_mode": refiner_weight_mode,
            },
            "training_contract": {
                "frozen_gnn": True,
                "frozen_semantic_encoder": True,
                "train_router": True,
                "train_refiner": True,
                "batch_size": int(batch_size),
                "max_epochs": int(max_epochs),
                "patience": int(patience),
                "train_node_cap": int(configured_train_cap),
                "effective_train_node_count": int(strict_train_idx.size),
                "full_train_node_count": int(train_idx.size),
                "train_cap_mode": train_cap_mode,
                "training_budget_schedule": {
                    "k_start": int(batch_size),
                    "k_end": int(max(int(round(batch_size / 4.0)), 1)),
                    "decay_factor": float(decay_factor),
                },
                "router_weight": 0.0,
                "router_regression_weight": 0.0,
                "router_ranking_weight": float(1.0),
                "router_selection_weight": float(0.25),
                "router_calibration_weight": float(0.25),
                "router_reliability_weight": 0.0,
                "entropy_weight": 0.0,
                "beta_selection": "validation_only",
                "router_training_objective": "base_wrong_reliability_bce_plus_pairwise_ranking",
                "auxiliary_router_target": "none_oracle_advantage_retained_for_diagnostics_only",
                "refiner_target_mode": refiner_target_mode,
                "joint_refiner_gate_target": refiner_gate_target,
                "refiner_explicit_gate": bool(refiner_explicit_gate),
                "refiner_weight_mode": refiner_weight_mode,
                "refiner_base_wrong_weight": float(refiner_base_wrong_weight),
                "refiner_utility_weight": float(refiner_utility_weight),
                "refiner_gate_weight": float(refiner_gate_weight),
            },
            "q_estimator": {
                **q_bundle["classifier"],
                "train_count": int(q_bundle["train_count"]),
                "valid_count": int(q_bundle["valid_count"]),
                "temperature": float(q_bundle["temperature"]),
            },
            "mc_dropout": {
                "num_passes": int(mc_bundle["num_passes"]),
                "logits_shape": list(mc_bundle["logits_shape"]),
            },
            "original_node_features": {
                "path": str(original_node_features_bundle["path"]),
                "feature_manifest": original_node_features_bundle["feature_manifest"],
            },
        }

        self._write_stage_manifest(
            stage_dir,
            manifest,
            artifact_namespace="stages/joint_router_refinement",
            visibility="public",
            resolved_task="joint_router_refinement",
        )
        write_json(stage_dir / "metrics.json", metrics_payload)
        write_json(stage_dir / "analysis_summary.json", analysis_summary)
        write_json(stage_dir / "router_performance_summary.json", router_performance_summary)
        write_csv_rows(
            stage_dir / "router_performance_summary.csv",
            list(router_performance_row.keys()),
            [router_performance_row],
        )
        write_csv_rows(
            stage_dir / "router_epoch_curve.csv",
            [
                "epoch",
                "beta",
                "train_top_k",
                "eval_top_k",
                "router_train_auroc",
                "router_valid_auroc",
                "router_test_auroc",
                "router_valid_adv_auroc",
                "router_test_adv_auroc",
                "router_valid_auprc",
                "router_valid_adv_auprc",
                "refiner_valid_fix_rate",
                "refiner_valid_break_rate",
                "refiner_test_fix_rate",
                "refiner_test_break_rate",
            ],
            final_run["fit_summary"]["component_curve_summary"]["curve"],
        )
        write_csv_rows(
            stage_dir / "beta_sweep.csv",
            [
                "beta",
                "eval_top_k",
                "best_epoch",
                "selected_budget",
                "selected_budget_key",
                "valid_accuracy",
                "valid_macro_f1",
                "valid_bot_f1",
                "valid_loss",
                "valid_routed_count",
                "valid_query_rate",
                "valid_net_gain",
                "valid_routed_wrong_precision",
                "valid_routed_wrong_coverage",
                "valid_conditional_fix_rate",
                "valid_gate_positive_rate",
                "valid_mean_gate_prob_routed",
                "valid_gate_precision",
                "valid_gate_recall",
                "test_accuracy",
                "test_macro_f1",
                "test_bot_f1",
                "test_loss",
                "test_routed_count",
                "test_query_rate",
                "test_net_gain",
                "test_routed_wrong_precision",
                "test_routed_wrong_coverage",
                "test_conditional_fix_rate",
                "test_gate_positive_rate",
                "test_mean_gate_prob_routed",
                "test_gate_precision",
                "test_gate_recall",
            ],
            beta_sweep,
        )
        write_csv_rows(
            stage_dir / "valid_budget_curve.csv",
            [
                "budget",
                "routed_count",
                "fix",
                "break",
                "net",
                "routed_fix",
                "routed_break",
                "routed_net",
                "wrong_node_fix_rate",
                "correct_node_break_rate",
                "accuracy",
                "macro_f1",
                "bot_f1",
                "delta_accuracy",
                "delta_macro_f1",
                "delta_bot_f1",
                "mean_base_confidence_routed",
                "mean_refiner_confidence_routed",
            ],
            valid_budget_curve,
        )
        write_csv_rows(
            stage_dir / "test_budget_curve.csv",
            [
                "budget",
                "routed_count",
                "fix",
                "break",
                "net",
                "routed_fix",
                "routed_break",
                "routed_net",
                "wrong_node_fix_rate",
                "correct_node_break_rate",
                "accuracy",
                "macro_f1",
                "bot_f1",
                "delta_accuracy",
                "delta_macro_f1",
                "delta_bot_f1",
                "mean_base_confidence_routed",
                "mean_refiner_confidence_routed",
            ],
            test_budget_curve,
        )
        write_text(
            stage_dir / "per_node_test.jsonl",
            "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in per_node_rows) + "\n",
        )
        save_stage_artifacts(
            stage_dir,
            {
                "outputs.pt": {
                    "logits": final_logits,
                    "prob": final_prob,
                    "pred": final_pred,
                    "labels": labels_t,
                    "base_pred": pred_gnn,
                    "routed_masks": routed_masks,
                    "eligible_refiner_masks": eligible_refiner_masks,
                    "router_prob": router_prob,
                    "router_score": router_score,
                    "oracle_advantage": oracle_advantage,
                    "gate_prob": gate_prob,
                    "mpe_gate_weights": mpe_gate_weights,
                    "mpe_selected_expert": mpe_selected_expert,
                    "mpe_gate_component_order": list(selector_component_order),
                    "selector_component_order": list(selector_component_order),
                    "selector_weights_test": selector_weights_test,
                    "selected_expert_test": selected_expert_test,
                    "selected_expert_name_test": selected_expert_name_test,
                    "neighbor_count_1hop": torch.tensor(
                        semantic_views.get("count_1hop", np.zeros(int(getattr(self, "graph_node_count", len(self.labels))), dtype=np.int64)),
                        dtype=torch.long,
                    ),
                    "neighbor_count_2hop": torch.tensor(
                        semantic_views.get("count_2hop", np.zeros(int(getattr(self, "graph_node_count", len(self.labels))), dtype=np.int64)),
                        dtype=torch.long,
                    ),
                },
                "checkpoint.pt": {
                    "router_model": final_run["router_state"],
                    "refiner_model": final_run["refiner_state"],
                    "fit_summary": final_run["fit_summary"],
                    "selected_beta": selected_beta,
                    "selected_budget": selected_budget,
                    "selected_eval_top_k": int(eval_top_k),
                    "selector_component_order": list(selector_component_order),
                    "selector_weights_test": selector_weights_test,
                    "selected_expert_test": selected_expert_test,
                    "selected_expert_name_test": selected_expert_name_test,
                },
                "beta_sweep": beta_sweep,
                "valid_budget_curve": valid_budget_curve,
                "test_budget_curve": test_budget_curve,
                **base_bundle,
            },
        )
        notes = [
            "# GLANCE joint router-refiner baseline",
            "",
            "This stage implements a GLANCE-style joint router+refiner contract under the current cached semantic embedding path.",
            "The public router is task-adapted for TwiBot20 reliability routing: base confidence is temperature-scaled, social features are direction-aware, and router supervision targets base-wrong reliability rather than paper-text advantage.",
            "The semantic expert alignment is intentionally deferred, and the public evaluation protocol is TwiBot20-adapted global-budget routing.",
            (
                "The base GNN and semantic expert remain frozen; only the routed-node refiner MLP is trained while the router is reused read-only from a prior joint stage."
                if joint_routing_protocol == "frozen_router_reuse"
                else "The base GNN and semantic expert remain frozen; only the router scorer MLP and the routed-node refiner MLP are trained."
            ),
            "The router is trained with base-wrong reliability BCE plus pairwise ranking. Oracle advantage is still recorded as a post-hoc diagnostic trace.",
            f"Training uses deterministic batch top-k with K_start={int(batch_size)} and K_end={int(eval_top_k)}. Final evaluation ranks the whole split by router score, selects budget on validation, and locks the selected budget ({selected_budget:.3f}) on test. Beta is selected on validation only from {{0.1, 0.2, 0.3}}.",
            "Use router_performance_summary.json/csv, router_epoch_curve.csv, and the budget-curve CSVs to inspect router quality directly.",
        ]
        write_text(stage_dir / "notes.md", "\n".join(notes) + "\n")
        return {
            "stage": "glance_joint_router_refine",
            "stage_dir": str(stage_dir),
            "metrics": metrics_payload,
        }

    @staticmethod
    def _budget_key(budget):
        return int(round(float(budget) * 100))

    def _build_counterfactual_budget_rows(
        self,
        budgets,
        risk_score,
        pred_gnn,
        pred_refiner,
        labels_np,
        eval_idx,
        prob_gnn,
        prob_refiner,
    ):
        risk_score = np.asarray(risk_score, dtype=np.float64)
        labels_np = np.asarray(labels_np, dtype=np.int64)
        pred_gnn_np = _labeled_prefix_pred(pred_gnn, labels_np.shape[0]).astype(np.int64)
        pred_ref_np = _labeled_prefix_pred(pred_refiner, labels_np.shape[0]).astype(np.int64)
        eval_idx = np.asarray(eval_idx, dtype=np.int64)
        prob_gnn_t = prob_gnn.detach().cpu().float() if torch.is_tensor(prob_gnn) else torch.tensor(prob_gnn, dtype=torch.float32)
        prob_ref_t = prob_refiner.detach().cpu().float() if torch.is_tensor(prob_refiner) else torch.tensor(prob_refiner, dtype=torch.float32)

        order = eval_idx[np.argsort(-risk_score[eval_idx])] if eval_idx.size else np.asarray([], dtype=np.int64)
        base_eval = _score_all(labels_np[eval_idx], pred_gnn_np[eval_idx]) if eval_idx.size else {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "bot_f1": 0.0,
            "count": 0,
        }
        base_wrong_count = int((pred_gnn_np[eval_idx] != labels_np[eval_idx]).sum()) if eval_idx.size else 0
        base_correct_count = int(eval_idx.size - base_wrong_count)
        rows = []
        payloads = {}
        for budget in parse_budget_list(budgets):
            k = min(max(int(eval_idx.size * float(budget)), 1), eval_idx.size) if eval_idx.size else 0
            routed = order[:k]
            routed_mask = np.zeros(labels_np.shape[0], dtype=bool)
            routed_mask[routed] = True
            final_pred = pred_gnn_np.copy()
            final_pred[routed] = pred_ref_np[routed]
            test_mask = np.zeros(labels_np.shape[0], dtype=bool)
            test_mask[eval_idx] = True
            delta = _delta_table(pred_gnn_np, final_pred, labels_np, test_mask)
            routed_delta = _delta_table(pred_gnn_np, final_pred, labels_np, routed_mask)
            final_eval = _score_all(labels_np[eval_idx], final_pred[eval_idx]) if eval_idx.size else dict(base_eval)
            row = {
                "budget": float(budget),
                "routed_count": int(k),
                "fix": int(delta["fix"]),
                "break": int(delta["broke"]),
                "net": int(delta["net"]),
                "routed_fix": int(routed_delta["fix"]),
                "routed_break": int(routed_delta["broke"]),
                "routed_net": int(routed_delta["net"]),
                "wrong_node_fix_rate": float(delta["fix"] / base_wrong_count) if base_wrong_count else 0.0,
                "correct_node_break_rate": float(delta["broke"] / base_correct_count) if base_correct_count else 0.0,
                "accuracy": float(final_eval["accuracy"]),
                "macro_f1": float(final_eval["macro_f1"]),
                "bot_f1": float(final_eval["bot_f1"]),
                "delta_accuracy": float(final_eval["accuracy"] - base_eval["accuracy"]),
                "delta_macro_f1": float(final_eval["macro_f1"] - base_eval["macro_f1"]),
                "delta_bot_f1": float(final_eval["bot_f1"] - base_eval["bot_f1"]),
                "mean_base_confidence_routed": float(prob_gnn_t[routed].max(dim=1).values.mean().item()) if k else 0.0,
                "mean_refiner_confidence_routed": float(prob_ref_t[routed].max(dim=1).values.mean().item()) if k else 0.0,
            }
            rows.append(row)
            payloads[str(self._budget_key(budget))] = {
                "metrics": final_eval,
                "delta_vs_base": {
                    "accuracy": row["delta_accuracy"],
                    "macro_f1": row["delta_macro_f1"],
                    "bot_f1": row["delta_bot_f1"],
                },
                "routed_node_ids": [int(item) for item in routed.tolist()],
                "final_pred": torch.tensor(final_pred, dtype=torch.long),
                "routed_mask": torch.tensor(routed_mask, dtype=torch.bool),
            }
        return rows, payloads

    def _load_glance_counterfactual_router_artifact(self):
        manifest = self._require_stage_json("glance_counterfactual_router", "manifest")
        metrics = self._require_stage_json("glance_counterfactual_router", "metrics")
        comparison = self._read_stage_json("glance_counterfactual_router", "comparison")
        risk_manifest = self._require_stage_json("glance_counterfactual_router", "risk_manifest")
        outputs = self._require_stage_tensor("glance_counterfactual_router", "outputs")
        router_scores = outputs.get("risk_score")
        if router_scores is None:
            raise MissingFrozenArtifactError(
                "glance_counterfactual_router outputs.pt must contain risk_score for budgeted refine."
            )

        def _read_budget_curve_rows(filename):
            budget_curve_path = self.experiment_root / "stages" / "glance_counterfactual_router" / filename
            if not budget_curve_path.exists():
                return []
            import csv

            with open(budget_curve_path, "r", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))

        return {
            "manifest": manifest,
            "metrics": metrics,
            "comparison": comparison,
            "risk_manifest": risk_manifest,
            "outputs": outputs,
            "risk_score": router_scores,
            "router_scores": router_scores,
            "valid_budget_curve": _read_budget_curve_rows("valid_budget_curve.csv"),
            "test_budget_curve": _read_budget_curve_rows("test_budget_curve.csv"),
        }

    def _run_glance_refiner_analysis(self, stage_dir, base_bundle):
        oracle_metrics = self._require_stage_json("glance_oracle_refine", "metrics")
        oracle_manifest = self._require_stage_json("glance_oracle_refine", "manifest")
        oracle_rows = self._require_stage_jsonl("glance_oracle_refine", "per_node_test")
        full_metrics = self._require_stage_json("glance_full_graph_refine", "metrics")
        full_manifest = self._require_stage_json("glance_full_graph_refine", "manifest")
        full_rows = self._require_stage_jsonl("glance_full_graph_refine", "per_node_test")
        budgeted_metrics = self._read_stage_json("glance_budgeted_refine", "metrics")
        budgeted_manifest = self._read_stage_json("glance_budgeted_refine", "manifest")
        budgeted_rows = self._require_stage_jsonl("glance_budgeted_refine", "per_node_test") if budgeted_metrics and budgeted_manifest else []

        oracle_analysis = self._build_refiner_analysis_summary(oracle_rows, "oracle_wrong_nodes")
        full_analysis = self._build_refiner_analysis_summary(full_rows, "full_graph_supervised_refiner")
        budgeted_analysis = self._build_refiner_analysis_summary(budgeted_rows, "router_routed_fixed_budget_refiner") if budgeted_rows else None
        comparison = {
            "contract": "glance_refiner_analysis_v1",
            "oracle_upper_bound": {
                "manifest": oracle_manifest,
                "metrics": oracle_metrics,
                "analysis": oracle_analysis,
            },
            "full_graph_lower_bound": {
                "manifest": full_manifest,
                "metrics": full_metrics,
                "analysis": full_analysis,
            },
            "budgeted_router_refine": (
                {
                    "manifest": budgeted_manifest,
                    "metrics": budgeted_metrics,
                    "analysis": budgeted_analysis,
                }
                if budgeted_metrics and budgeted_manifest and budgeted_analysis
                else None
            ),
            "comparison_summary": {
                "oracle_wrong_fix_rate": float(oracle_analysis["wrong_node_fix_rate"]),
                "full_graph_wrong_fix_rate": float(full_analysis["wrong_node_fix_rate"]),
                "full_graph_correct_break_rate": float(full_analysis["correct_node_break_rate"]),
                "oracle_vs_full_graph_fix_gap": float(oracle_analysis["wrong_node_fix_rate"] - full_analysis["wrong_node_fix_rate"]),
                "full_graph_net_gain": int(full_metrics.get("net_gain", 0)),
                "budgeted_wrong_fix_rate": float(budgeted_analysis["wrong_node_fix_rate"]) if budgeted_analysis else None,
                "budgeted_correct_break_rate": float(budgeted_analysis["correct_node_break_rate"]) if budgeted_analysis else None,
                "budgeted_net_gain": int(budgeted_analysis["net_gain"]) if budgeted_analysis else None,
                "budgeted_routed_wrong_coverage": float(budgeted_analysis["routed_wrong_coverage"]) if budgeted_analysis else None,
                "budgeted_routed_wrong_precision": float(budgeted_analysis["routed_wrong_precision"]) if budgeted_analysis else None,
                "budgeted_conditional_fix_rate_on_selected_wrong": (
                    float(budgeted_analysis["conditional_fix_rate_on_selected_wrong"]) if budgeted_analysis else None
                ),
                "router_ablation_interpretation": (
                    "oracle upper bound isolates what the correction refiner can do on true hard nodes; "
                    "full-graph lower bound shows the cost of removing node selection; "
                    "budgeted refine under the router-selected validation budget is the deployable operating point."
                ),
            },
        }
        write_json(stage_dir / "analysis_summary.json", comparison)
        notes = [
            "# GLANCE refiner analysis",
            "",
            "This stage compares the error-only oracle upper bound, the no-router full-graph lower bound, and the router-selected budgeted refine operating point when present.",
            "Use the oracle stage to estimate refiner headroom on true hard nodes.",
            "Use the full-graph stage to diagnose where a missing router causes correct nodes to be harmed.",
            "When present, glance_budgeted_refine is interpreted under the router's validation-selected fixed budget, not a test-selected optimum.",
            "The gap between oracle wrong-node fix rate and full-graph wrong-node fix rate is the main signal for router necessity.",
        ]
        write_text(stage_dir / "notes.md", "\n".join(notes) + "\n")
        save_stage_artifacts(
            stage_dir,
            {
                "comparison": comparison,
                **base_bundle,
            },
        )
        return {
            "stage": "glance_refiner_analysis",
            "stage_dir": str(stage_dir),
            "metrics": comparison["comparison_summary"],
        }

    def _run_glance_budgeted_refine(self, stage_dir, base_bundle):
        context = self.ensure_backbone_context()
        semantic = self._load_semantic_finetune_artifact()
        router_artifact = self._load_glance_counterfactual_router_artifact()
        gnn_outputs = context["gnn_outputs"]
        p_gnn = gnn_outputs.get("prob")
        logits_gnn = gnn_outputs.get("logits")
        pred_gnn = gnn_outputs.get("pred")
        z_gnn = gnn_outputs.get("node_repr")
        if any(item is None for item in (p_gnn, logits_gnn, pred_gnn, z_gnn)):
            raise MissingFrozenArtifactError(
                "glance_budgeted_refine requires frozen_g0 outputs with prob/logits/pred/node_repr."
            )

        p_gnn = p_gnn.detach().cpu().float()
        logits_gnn = logits_gnn.detach().cpu().float()
        pred_gnn = pred_gnn.detach().cpu().long()
        z_gnn = z_gnn.detach().cpu().float()
        labels_t = torch.tensor(self.labels, dtype=torch.long)
        labels_np = labels_t.numpy()
        train_idx = _idx_numpy(self.data["train_idx"])
        valid_idx = _idx_numpy(self.data["valid_idx"])
        test_idx = _idx_numpy(self.data["test_idx"])

        semantic_views = self._build_k_hop_semantic_views(
            semantic["embeddings"],
            self.data.get("edge_index"),
        )
        refiner_features = torch.cat(
            [z_gnn, semantic_views["ego"], semantic_views["hop1"], semantic_views["hop2"]],
            dim=1,
        )

        risk_scores = router_artifact["router_scores"].detach().cpu().numpy().astype(np.float32)
        budgets = self._router_budgets()
        train_order = train_idx[np.argsort(-risk_scores[train_idx])] if train_idx.size else np.asarray([], dtype=np.int64)
        valid_order = valid_idx[np.argsort(-risk_scores[valid_idx])] if valid_idx.size else np.asarray([], dtype=np.int64)
        test_order = test_idx[np.argsort(-risk_scores[test_idx])] if test_idx.size else np.asarray([], dtype=np.int64)
        base_test = _score_all(labels_np[test_idx], pred_gnn.numpy()[test_idx]) if test_idx.size else {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "bot_f1": 0.0,
            "count": 0,
        }
        base_wrong_count = int((pred_gnn.numpy()[test_idx] != labels_np[test_idx]).sum()) if test_idx.size else 0
        base_correct_count = int(test_idx.size - base_wrong_count)

        budget_rows = []
        budget_payload_json = {}
        analysis_by_budget = {}
        candidate_outputs_by_key = {}
        router_selected_budget = router_artifact["metrics"].get(
            "selected_budget_valid",
            router_artifact["metrics"].get("selected_budget"),
        )
        if router_selected_budget is None:
            raise MissingFrozenArtifactError(
                "glance_budgeted_refine requires glance_counterfactual_router to persist a validation-selected budget."
            )
        router_selected_key = str(self._budget_key(router_selected_budget))

        for budget in budgets:
            train_k = min(max(int(train_idx.size * float(budget)), 1), train_idx.size) if train_idx.size else 0
            valid_k = min(max(int(valid_idx.size * float(budget)), 1), valid_idx.size) if valid_idx.size else 0
            test_k = min(max(int(test_idx.size * float(budget)), 1), test_idx.size) if test_idx.size else 0
            train_routed = train_order[:train_k]
            valid_routed = valid_order[:valid_k]
            test_routed = test_order[:test_k]
            if train_routed.size == 0:
                continue

            model = GlanceRefinerMLP(
                input_dim=int(refiner_features.shape[1]),
                hidden_dim=128,
                activation=str(getattr(self.args, "activation", "leakyrelu")).lower(),
                dropout=0.1,
            )
            model, fit_summary = self._train_glance_refiner(
                model,
                refiner_features[train_routed],
                labels_t[train_routed],
                refiner_features[valid_routed],
                labels_t[valid_routed],
            )
            model.eval()
            with torch.no_grad():
                train_logits_ref = model(refiner_features[train_routed]).cpu() if train_routed.size else torch.empty((0, 2), dtype=torch.float32)
                valid_logits_ref = model(refiner_features[valid_routed]).cpu() if valid_routed.size else torch.empty((0, 2), dtype=torch.float32)
                test_logits_ref = model(refiner_features[test_routed]).cpu() if test_routed.size else torch.empty((0, 2), dtype=torch.float32)

            final_logits = logits_gnn.clone()
            final_prob = p_gnn.clone()
            final_pred = pred_gnn.clone()
            if train_routed.size:
                final_logits[train_routed] = train_logits_ref
                final_prob[train_routed] = torch.softmax(train_logits_ref, dim=1)
                final_pred[train_routed] = train_logits_ref.argmax(dim=1)
            if valid_routed.size:
                final_logits[valid_routed] = valid_logits_ref
                final_prob[valid_routed] = torch.softmax(valid_logits_ref, dim=1)
                final_pred[valid_routed] = valid_logits_ref.argmax(dim=1)
            if test_routed.size:
                final_logits[test_routed] = test_logits_ref
                final_prob[test_routed] = torch.softmax(test_logits_ref, dim=1)
                final_pred[test_routed] = test_logits_ref.argmax(dim=1)

            test_mask = np.zeros(labels_np.shape[0], dtype=bool)
            test_mask[test_idx] = True
            routed_test_mask = np.zeros(labels_np.shape[0], dtype=bool)
            routed_test_mask[test_routed] = True
            delta = _delta_table(pred_gnn.numpy(), final_pred.numpy(), labels_np, test_mask)
            routed_delta = _delta_table(pred_gnn.numpy(), final_pred.numpy(), labels_np, routed_test_mask)
            overall_test = _score_all(labels_np[test_idx], final_pred.numpy()[test_idx]) if test_idx.size else dict(base_test)
            row = {
                "budget": float(budget),
                "train_routed_count": int(train_routed.size),
                "valid_routed_count": int(valid_routed.size),
                "test_routed_count": int(test_routed.size),
                "fix": int(delta["fix"]),
                "break": int(delta["broke"]),
                "net": int(delta["net"]),
                "routed_fix": int(routed_delta["fix"]),
                "routed_break": int(routed_delta["broke"]),
                "routed_net": int(routed_delta["net"]),
                "wrong_node_fix_rate": float(delta["fix"] / base_wrong_count) if base_wrong_count else 0.0,
                "correct_node_break_rate": float(delta["broke"] / base_correct_count) if base_correct_count else 0.0,
                "accuracy": float(overall_test["accuracy"]),
                "macro_f1": float(overall_test["macro_f1"]),
                "bot_f1": float(overall_test["bot_f1"]),
                "delta_accuracy": float(overall_test["accuracy"] - base_test["accuracy"]),
                "delta_macro_f1": float(overall_test["macro_f1"] - base_test["macro_f1"]),
                "delta_bot_f1": float(overall_test["bot_f1"] - base_test["bot_f1"]),
            }
            budget_rows.append(row)
            budget_key = str(self._budget_key(budget))
            budget_payload_json[budget_key] = {
                "metrics": overall_test,
                "delta_vs_base": {
                    "accuracy": row["delta_accuracy"],
                    "macro_f1": row["delta_macro_f1"],
                    "bot_f1": row["delta_bot_f1"],
                },
                "routed_node_ids_train": [int(item) for item in train_routed.tolist()],
                "routed_node_ids_valid": [int(item) for item in valid_routed.tolist()],
                "routed_node_ids_test": [int(item) for item in test_routed.tolist()],
            }
            per_node_rows = []
            test_routed_np = routed_test_mask
            for node_idx in test_idx.tolist():
                per_node_rows.append({
                    "node_id": int(node_idx),
                    "base_pred": int(pred_gnn[node_idx].item()),
                    "final_pred": int(final_pred[node_idx].item()),
                    "label": int(labels_np[node_idx]),
                    "routed": bool(test_routed_np[node_idx]),
                    "was_wrong_base": bool(pred_gnn[node_idx].item() != labels_np[node_idx]),
                    "is_wrong_final": bool(final_pred[node_idx].item() != labels_np[node_idx]),
                    "neighbor_count_1hop": int(semantic_views["count_1hop"][node_idx]),
                    "neighbor_count_2hop": int(semantic_views["count_2hop"][node_idx]),
                })
            analysis_by_budget[budget_key] = self._build_refiner_analysis_summary(per_node_rows, f"budget_{budget_key}")
            candidate_outputs_by_key[budget_key] = {
                "budget": float(budget),
                "logits": final_logits,
                "prob": final_prob,
                "pred": final_pred,
                "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "fit_summary": fit_summary,
                "per_node_rows": per_node_rows,
                "analysis_summary": analysis_by_budget[budget_key],
                "row": row,
                "routed_masks": {
                    "train": torch.zeros(labels_t.numel(), dtype=torch.bool).scatter_(0, torch.tensor(train_routed, dtype=torch.long), True) if train_routed.size else torch.zeros(labels_t.numel(), dtype=torch.bool),
                    "valid": torch.zeros(labels_t.numel(), dtype=torch.bool).scatter_(0, torch.tensor(valid_routed, dtype=torch.long), True) if valid_routed.size else torch.zeros(labels_t.numel(), dtype=torch.bool),
                    "test": torch.tensor(test_routed_np, dtype=torch.bool),
                },
            }

        if not budget_rows:
            raise MissingFrozenArtifactError("glance_budgeted_refine could not build any budgeted routed subset.")

        if router_selected_key not in candidate_outputs_by_key:
            raise MissingFrozenArtifactError(
                "glance_budgeted_refine could not match the router-selected validation budget "
                f"{router_selected_budget} to a local budgeted refiner run."
            )
        selected_budget = float(router_selected_budget)
        selected_key = router_selected_key
        selected_outputs = candidate_outputs_by_key[selected_key]
        metrics_payload = {
            "contract": "glance_budgeted_refine_metrics_v1",
            "routing_mode": "router_routed_fixed_budget_refiner",
            "paper_faithful_glance_inspired": True,
            "not_deployable": False,
            "counterfactual_source_refiner": router_artifact["manifest"].get("counterfactual_source_refiner"),
            "selected_budget_source": "valid",
            "budget_source_stage": "glance_counterfactual_router",
            "selected_budget": float(selected_budget),
            "selected_budget_valid": float(selected_budget),
            "selected_budget_key": selected_key,
            "selected_budget_metrics": selected_outputs["row"],
            "selected_test_budget_metrics_under_valid_choice": selected_outputs["row"],
            "base_test": base_test,
            "overall_delta_vs_base_gnn": {
                "accuracy": float(selected_outputs["row"]["delta_accuracy"]),
                "macro_f1": float(selected_outputs["row"]["delta_macro_f1"]),
                "bot_f1": float(selected_outputs["row"]["delta_bot_f1"]),
            },
            "wrong_node_fix_rate": float(selected_outputs["row"]["wrong_node_fix_rate"]),
            "correct_node_break_rate": float(selected_outputs["row"]["correct_node_break_rate"]),
            "improved_count": int(selected_outputs["row"]["fix"]),
            "degraded_count": int(selected_outputs["row"]["break"]),
            "net_gain": int(selected_outputs["row"]["net"]),
            "base_wrong_count": base_wrong_count,
            "base_correct_count": base_correct_count,
            "budget_curve": budget_rows,
            "budget_curve_scope": "test_sweep_diagnostic_only",
            "fit_summary": selected_outputs["fit_summary"],
            "split_scope": {
                "train": int(train_idx.size),
                "valid": int(valid_idx.size),
                "test": int(test_idx.size),
            },
        }
        manifest = {
            "contract": "glance_budgeted_refine_v1",
            "status": "completed",
            "routing_mode": "router_routed_fixed_budget_refiner",
            "paper_faithful_glance_inspired": True,
            "diagnostic_lane_only": True,
            "not_deployable": False,
            "research_positioning": "selector_gated_correction_refiner_under_fixed_budget",
            "counterfactual_source_refiner": router_artifact["manifest"].get("counterfactual_source_refiner"),
            "router_manifest": router_artifact["manifest"],
            "budget_source_stage": "glance_counterfactual_router",
            "selected_budget_source": "valid",
            "semantic_manifest": semantic["manifest"],
            "frozen_g0_manifest": context["frozen_g0"]["manifest"],
            "selected_budget": float(selected_budget),
            "selected_budget_valid": float(selected_budget),
            "budgets": [float(item) for item in budgets],
            "refiner_architecture": {
                "type": "glance_budgeted_refine_mlp",
                "input": "[z_gnn || z_sem_0 || z_sem_1 || z_sem_2]",
                "hidden_dim": 128,
                "activation": str(getattr(self.args, "activation", "leakyrelu")).lower(),
                "dropout": 0.1,
                "output_dim": int(logits_gnn.shape[1]),
            },
        }
        self._write_stage_manifest(
            stage_dir,
            manifest,
            artifact_namespace="stages/joint_router_refinement",
            visibility="public",
            resolved_task="joint_router_refinement",
        )
        write_json(stage_dir / "metrics.json", metrics_payload)
        write_json(stage_dir / "budget_payloads.json", budget_payload_json)
        write_json(stage_dir / "analysis_summary.json", selected_outputs["analysis_summary"])
        write_csv_rows(
            stage_dir / "budget_curve.csv",
            [
                "budget",
                "train_routed_count",
                "valid_routed_count",
                "test_routed_count",
                "fix",
                "break",
                "net",
                "routed_fix",
                "routed_break",
                "routed_net",
                "wrong_node_fix_rate",
                "correct_node_break_rate",
                "accuracy",
                "macro_f1",
                "bot_f1",
                "delta_accuracy",
                "delta_macro_f1",
                "delta_bot_f1",
            ],
            budget_rows,
        )
        write_text(
            stage_dir / "per_node_test.jsonl",
            "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in selected_outputs["per_node_rows"]) + "\n",
        )
        save_stage_artifacts(
            stage_dir,
            {
                "outputs.pt": {
                    "logits": selected_outputs["logits"],
                    "prob": selected_outputs["prob"],
                    "pred": selected_outputs["pred"],
                    "labels": labels_t,
                    "base_pred": pred_gnn,
                    "routed_masks": selected_outputs["routed_masks"],
                    "neighbor_count_1hop": torch.tensor(semantic_views["count_1hop"], dtype=torch.long),
                    "neighbor_count_2hop": torch.tensor(semantic_views["count_2hop"], dtype=torch.long),
                },
                "checkpoint.pt": {
                    "model": selected_outputs["model_state"],
                    "fit_summary": selected_outputs["fit_summary"],
                },
                "router_budget_payloads": budget_payload_json,
                **base_bundle,
            },
        )
        notes = [
            "# GLANCE budgeted refine",
            "",
            "This stage is a diagnostic routed-set correction lane, not the main strict GLANCE baseline.",
            "It consumes routed node selections from glance_counterfactual_router and trains a routed-only correction refiner.",
            "For each budget, the refiner is trained on train_routed(B), selected on valid_routed(B), and applied only to test_routed(B).",
            "The final artifact is locked to the router's validation-selected budget; the per-budget curve is diagnostic only.",
            "Non-routed nodes strictly preserve the base GNN prediction.",
        ]
        write_text(stage_dir / "notes.md", "\n".join(notes) + "\n")
        return {
            "stage": "glance_budgeted_refine",
            "stage_dir": str(stage_dir),
            "metrics": metrics_payload,
        }

    def _run_glance_oracle_refine(self, stage_dir, base_bundle):
        context = self.ensure_backbone_context()
        semantic = self._load_semantic_finetune_artifact()
        active_edge_index, active_edge_type = self._active_graph_tensors()
        gnn_outputs = context["gnn_outputs"]
        p_gnn = gnn_outputs.get("prob")
        logits_gnn = gnn_outputs.get("logits")
        pred_gnn = gnn_outputs.get("pred")
        z_gnn = gnn_outputs.get("node_repr")
        if any(item is None for item in (p_gnn, logits_gnn, pred_gnn, z_gnn)):
            raise MissingFrozenArtifactError(
                "glance_oracle_refine requires frozen_g0 outputs with prob/logits/pred/node_repr."
            )

        p_gnn = p_gnn.detach().cpu().float()
        logits_gnn = logits_gnn.detach().cpu().float()
        pred_gnn = pred_gnn.detach().cpu().long()
        z_gnn = z_gnn.detach().cpu().float()
        labels_t = torch.tensor(self.labels, dtype=torch.long)

        routed_masks = self._oracle_wrong_node_masks(pred_gnn.numpy())
        semantic_views = self._build_k_hop_semantic_views(
            semantic["embeddings"],
            active_edge_index,
        )
        refiner_features = torch.cat(
            [z_gnn, semantic_views["ego"], semantic_views["hop1"], semantic_views["hop2"]],
            dim=1,
        )

        train_idx = np.flatnonzero(routed_masks["train"])
        valid_idx = np.flatnonzero(routed_masks["valid"])
        test_idx = np.flatnonzero(routed_masks["test"])
        if train_idx.size == 0:
            raise MissingFrozenArtifactError(
                "glance_oracle_refine found zero oracle-routed train nodes. "
                "This upper-bound stage requires at least one train error node."
            )

        model = GlanceRefinerMLP(
            input_dim=int(refiner_features.shape[1]),
            hidden_dim=128,
            activation=str(getattr(self.args, "activation", "leakyrelu")).lower(),
            dropout=0.1,
        )
        model, fit_summary = self._train_glance_refiner(
            model,
            refiner_features[train_idx],
            labels_t[train_idx],
            refiner_features[valid_idx],
            labels_t[valid_idx],
        )

        model.eval()
        with torch.no_grad():
            if train_idx.size > 0:
                train_logits_ref = model(refiner_features[train_idx]).cpu()
            else:
                train_logits_ref = torch.empty((0, 2), dtype=torch.float32)
            if valid_idx.size > 0:
                valid_logits_ref = model(refiner_features[valid_idx]).cpu()
            else:
                valid_logits_ref = torch.empty((0, 2), dtype=torch.float32)
            if test_idx.size > 0:
                test_logits_ref = model(refiner_features[test_idx]).cpu()
            else:
                test_logits_ref = torch.empty((0, 2), dtype=torch.float32)

        final_logits = logits_gnn.clone()
        final_prob = p_gnn.clone()
        final_pred = pred_gnn.clone()
        if train_idx.size > 0:
            final_logits[train_idx] = train_logits_ref
            final_prob[train_idx] = torch.softmax(train_logits_ref, dim=1)
            final_pred[train_idx] = train_logits_ref.argmax(dim=1)
        if valid_idx.size > 0:
            final_logits[valid_idx] = valid_logits_ref
            final_prob[valid_idx] = torch.softmax(valid_logits_ref, dim=1)
            final_pred[valid_idx] = valid_logits_ref.argmax(dim=1)
        if test_idx.size > 0:
            final_logits[test_idx] = test_logits_ref
            final_prob[test_idx] = torch.softmax(test_logits_ref, dim=1)
            final_pred[test_idx] = test_logits_ref.argmax(dim=1)

        test_delta = _delta_table(pred_gnn.numpy(), final_pred.numpy(), self.labels, self.test_mask)
        routed_test_ratio = float(test_idx.size / max(int(self.test_mask.sum()), 1))
        non_routed_test_mask = self.test_mask & ~routed_masks["test"]
        preservation = float(
            (final_pred.numpy()[non_routed_test_mask] == pred_gnn.numpy()[non_routed_test_mask]).mean()
        ) if non_routed_test_mask.any() else 1.0

        overall_test = _score_all(self.labels[self.test_mask], final_pred.numpy()[self.test_mask])
        routed_test = _score_all(self.labels[test_idx], final_pred.numpy()[test_idx]) if test_idx.size else {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "bot_f1": 0.0,
            "count": 0,
        }
        base_test = _score_all(self.labels[self.test_mask], pred_gnn.numpy()[self.test_mask])
        base_wrong_count = int(test_idx.size)
        base_correct_count = int(int(self.test_mask.sum()) - base_wrong_count)

        metrics_payload = {
            "contract": "glance_oracle_refine_metrics_v1",
            "routing_mode": "oracle_wrong_nodes",
            "upper_bound_only": True,
            "not_deployable": True,
            "overall_test": overall_test,
            "routed_test": routed_test,
            "base_test": base_test,
            "non_routed_test_preservation_accuracy": preservation,
            "routed_test_ratio": routed_test_ratio,
            "fixed_wrong_to_right": int(test_delta["fix"]),
            "fixed_right_to_wrong": int(test_delta["broke"]),
            "net_gain_on_routed": int(test_delta["net"]),
            "base_wrong_count": base_wrong_count,
            "base_correct_count": base_correct_count,
            "wrong_node_fix_rate": float(test_delta["fix"] / base_wrong_count) if base_wrong_count else 0.0,
            "correct_node_break_rate": 0.0,
            "overall_delta_vs_base_gnn": {
                "accuracy": float(overall_test["accuracy"] - base_test["accuracy"]),
                "macro_f1": float(overall_test["macro_f1"] - base_test["macro_f1"]),
                "bot_f1": float(overall_test["bot_f1"] - base_test["bot_f1"]),
            },
            "oracle_routed_counts": {
                "train": int(train_idx.size),
                "valid": int(valid_idx.size),
                "test": int(test_idx.size),
                "all": int(routed_masks["all"].sum()),
            },
            "fit_summary": fit_summary,
        }

        manifest = {
            "contract": "glance_oracle_refine_v1",
            "status": "completed",
            "routing_mode": "oracle_wrong_nodes",
            "upper_bound_only": True,
            "not_deployable": True,
            "research_positioning": "error_only_oracle_upper_bound_for_correction_refiner",
            "no_router_training": True,
            "no_llm_generation": True,
            "semantic_source": "semantic_finetune_roberta_finetuned",
            "semantic_stage_dir": str(semantic["dir"]),
            "semantic_manifest": semantic["manifest"],
            "frozen_g0_dir": str(context["frozen_g0"]["dir"]),
            "frozen_g0_manifest": context["frozen_g0"]["manifest"],
            "graph_source": context["graph_bundle"]["source"],
            "graph_source_root": context["graph_bundle"]["source_root"],
            "text_encoder_identity": str(semantic["manifest"].get("lm_model", semantic["manifest"].get("semantic_backbone", "roberta_finetuned"))),
            "refiner_architecture": {
                "type": "oracle_glance_refiner_mlp",
                "input": "[z_gnn || z_sem_0 || z_sem_1 || z_sem_2]",
                "hidden_dim": 128,
                "activation": str(getattr(self.args, "activation", "leakyrelu")).lower(),
                "dropout": 0.1,
                "output_dim": 2,
                "hard_switch": True,
            },
            "oracle_routed_counts": metrics_payload["oracle_routed_counts"],
        }
        self._write_stage_manifest(
            stage_dir,
            manifest,
            artifact_namespace="stages/local_conformal_diagnostic",
            visibility="public",
            resolved_task="local_conformal_diagnostic",
        )
        write_json(stage_dir / "metrics.json", metrics_payload)

        per_node_rows = []
        for node_idx in np.flatnonzero(self.test_mask):
            per_node_rows.append({
                "node_id": int(node_idx),
                "base_pred": int(pred_gnn[node_idx].item()),
                "final_pred": int(final_pred[node_idx].item()),
                "label": int(self.labels[node_idx]),
                "routed": bool(routed_masks["test"][node_idx]),
                "was_wrong_base": bool(pred_gnn[node_idx].item() != self.labels[node_idx]),
                "is_wrong_final": bool(final_pred[node_idx].item() != self.labels[node_idx]),
                "neighbor_count_1hop": int(semantic_views["count_1hop"][node_idx]),
                "neighbor_count_2hop": int(semantic_views["count_2hop"][node_idx]),
            })
        write_text(
            stage_dir / "per_node_test.jsonl",
            "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in per_node_rows) + "\n",
        )
        analysis_summary = self._build_refiner_analysis_summary(per_node_rows, "oracle_wrong_nodes")
        write_json(stage_dir / "analysis_summary.json", analysis_summary)

        save_stage_artifacts(
            stage_dir,
            {
                "outputs.pt": {
                    "logits": final_logits,
                    "prob": final_prob,
                    "pred": final_pred,
                    "labels": labels_t,
                    "base_pred": pred_gnn,
                    "routed_masks": {key: torch.tensor(value, dtype=torch.bool) for key, value in routed_masks.items()},
                    "neighbor_count_1hop": torch.tensor(semantic_views["count_1hop"], dtype=torch.long),
                    "neighbor_count_2hop": torch.tensor(semantic_views["count_2hop"], dtype=torch.long),
                },
                "checkpoint.pt": {
                    "model": model.state_dict(),
                    "fit_summary": fit_summary,
                },
                "oracle_summary": {
                    "routing_mode": "oracle_wrong_nodes",
                    "warning": "upper bound only; routed masks use ground-truth errors in every split",
                },
                "analysis_summary": analysis_summary,
                **base_bundle,
            },
        )

        notes = [
            "# GLANCE oracle upper bound",
            "",
            "This stage is an oracle upper-bound ablation.",
            "Routed nodes are defined by ground-truth base-GNN errors within each split.",
            "No router is trained and no generative LLM is queried.",
            "Routed-node refinement uses semantic_finetune roberta-f embeddings with ego/1-hop/2-hop mean pooling.",
        ]
        write_text(stage_dir / "notes.md", "\n".join(notes) + "\n")
        return {
            "stage": "glance_oracle_refine",
            "stage_dir": str(stage_dir),
            "metrics": metrics_payload,
        }

    def _run_glance_full_graph_refine(self, stage_dir, base_bundle):
        context = self.ensure_backbone_context()
        semantic = self._load_semantic_finetune_artifact()
        active_edge_index, active_edge_type = self._active_graph_tensors()
        gnn_outputs = context["gnn_outputs"]
        p_gnn = gnn_outputs.get("prob")
        logits_gnn = gnn_outputs.get("logits")
        pred_gnn = gnn_outputs.get("pred")
        z_gnn = gnn_outputs.get("node_repr")
        if any(item is None for item in (p_gnn, logits_gnn, pred_gnn, z_gnn)):
            raise MissingFrozenArtifactError(
                "glance_full_graph_refine requires frozen_g0 outputs with prob/logits/pred/node_repr."
            )

        p_gnn = p_gnn.detach().cpu().float()
        logits_gnn = logits_gnn.detach().cpu().float()
        pred_gnn = pred_gnn.detach().cpu().long()
        z_gnn = z_gnn.detach().cpu().float()
        labels_t = torch.tensor(self.labels, dtype=torch.long)

        semantic_views = self._build_k_hop_semantic_views(
            semantic["embeddings"],
            active_edge_index,
        )
        refiner_features = torch.cat(
            [z_gnn, semantic_views["ego"], semantic_views["hop1"], semantic_views["hop2"]],
            dim=1,
        )

        train_idx = _idx_numpy(self.data["train_idx"])
        valid_idx = _idx_numpy(self.data["valid_idx"])
        test_idx = _idx_numpy(self.data["test_idx"])
        if train_idx.size == 0:
            raise MissingFrozenArtifactError("glance_full_graph_refine requires a non-empty train_idx.")

        model = GlanceRefinerMLP(
            input_dim=int(refiner_features.shape[1]),
            hidden_dim=128,
            activation=str(getattr(self.args, "activation", "leakyrelu")).lower(),
            dropout=0.1,
        )
        model, fit_summary = self._train_glance_refiner(
            model,
            refiner_features[train_idx],
            labels_t[train_idx],
            refiner_features[valid_idx],
            labels_t[valid_idx],
        )

        model.eval()
        with torch.no_grad():
            final_logits = model(refiner_features).cpu()
            final_prob = torch.softmax(final_logits, dim=1)
            final_pred = final_logits.argmax(dim=1)

        test_delta = _delta_table(pred_gnn.numpy(), final_pred.numpy(), self.labels, self.test_mask)
        overall_test = _score_all(self.labels[self.test_mask], final_pred.numpy()[self.test_mask])
        base_test = _score_all(self.labels[self.test_mask], pred_gnn.numpy()[self.test_mask])
        lm_only_test = None
        overall_delta_vs_lm_only = None
        lm_only_pred = None
        if semantic.get("outputs") is not None:
            lm_outputs = semantic["outputs"]
            lm_prob = lm_outputs.get("prob_cal", lm_outputs.get("prob"))
            lm_pred_t = lm_outputs.get("pred")
            if lm_pred_t is None and lm_prob is not None:
                lm_pred_t = lm_prob.argmax(dim=1)
            if lm_pred_t is not None:
                lm_only_pred = lm_pred_t.detach().cpu().long() if torch.is_tensor(lm_pred_t) else torch.tensor(lm_pred_t, dtype=torch.long)
                lm_only_test = _score_all(self.labels[self.test_mask], lm_only_pred.numpy()[self.test_mask])
                overall_delta_vs_lm_only = {
                    "accuracy": float(overall_test["accuracy"] - lm_only_test["accuracy"]),
                    "macro_f1": float(overall_test["macro_f1"] - lm_only_test["macro_f1"]),
                    "bot_f1": float(overall_test["bot_f1"] - lm_only_test["bot_f1"]),
                }

        base_wrong_count = int((pred_gnn.numpy()[self.test_mask] != self.labels[self.test_mask]).sum())
        base_correct_count = int(int(self.test_mask.sum()) - base_wrong_count)

        metrics_payload = {
            "contract": "glance_full_graph_refine_metrics_v1",
            "routing_mode": "full_graph_supervised_refiner",
            "upper_bound_only": False,
            "not_deployable": False,
            "overall_test": overall_test,
            "base_test": base_test,
            "lm_only_test": lm_only_test,
            "overall_delta_vs_base_gnn": {
                "accuracy": float(overall_test["accuracy"] - base_test["accuracy"]),
                "macro_f1": float(overall_test["macro_f1"] - base_test["macro_f1"]),
                "bot_f1": float(overall_test["bot_f1"] - base_test["bot_f1"]),
            },
            "overall_delta_vs_lm_only": overall_delta_vs_lm_only,
            "improved_count": int(test_delta["fix"]),
            "degraded_count": int(test_delta["broke"]),
            "net_gain": int(test_delta["net"]),
            "count": int(test_delta["touched"]),
            "base_wrong_count": base_wrong_count,
            "base_correct_count": base_correct_count,
            "wrong_node_fix_rate": float(test_delta["fix"] / base_wrong_count) if base_wrong_count else 0.0,
            "correct_node_break_rate": float(test_delta["broke"] / base_correct_count) if base_correct_count else 0.0,
            "fit_summary": fit_summary,
            "split_scope": {
                "train": int(train_idx.size),
                "valid": int(valid_idx.size),
                "test": int(test_idx.size),
            },
        }

        manifest = {
            "contract": "glance_full_graph_refine_v1",
            "status": "completed",
            "routing_mode": "full_graph_supervised_refiner",
            "upper_bound_only": False,
            "not_deployable": False,
            "research_positioning": "no_router_lower_bound_router_ablation_for_refiner_diagnosis",
            "no_router_training": True,
            "no_llm_generation": True,
            "semantic_source": "semantic_finetune_roberta_finetuned",
            "semantic_stage_dir": str(semantic["dir"]),
            "semantic_manifest": semantic["manifest"],
            "frozen_g0_dir": str(context["frozen_g0"]["dir"]),
            "frozen_g0_manifest": context["frozen_g0"]["manifest"],
            "graph_source": context["graph_bundle"]["source"],
            "graph_source_root": context["graph_bundle"]["source_root"],
            "text_encoder_identity": str(semantic["manifest"].get("lm_model", semantic["manifest"].get("semantic_backbone", "roberta_finetuned"))),
            "refiner_architecture": {
                "type": "glance_full_graph_refiner_mlp",
                "input": "[z_gnn || z_sem_0 || z_sem_1 || z_sem_2]",
                "hidden_dim": 128,
                "activation": str(getattr(self.args, "activation", "leakyrelu")).lower(),
                "dropout": 0.1,
                "output_dim": 2,
                "full_graph": True,
            },
            "split_scope": metrics_payload["split_scope"],
            "lm_only_reference_available": bool(lm_only_test is not None),
            "seed_scope_note": "current local validation path is seed_1; multi-seed requires matching frozen_g0 per seed",
        }
        self._write_stage_manifest(
            stage_dir,
            manifest,
            artifact_namespace="stages/local_conflict_prune_diag",
            visibility="public",
            resolved_task="local_conflict_prune_diag",
        )
        write_json(stage_dir / "metrics.json", metrics_payload)

        per_node_rows = []
        for node_idx in np.flatnonzero(self.test_mask):
            row = {
                "node_id": int(node_idx),
                "base_pred": int(pred_gnn[node_idx].item()),
                "final_pred": int(final_pred[node_idx].item()),
                "label": int(self.labels[node_idx]),
                "routed": True,
                "neighbor_count_1hop": int(semantic_views["count_1hop"][node_idx]),
                "neighbor_count_2hop": int(semantic_views["count_2hop"][node_idx]),
                "was_wrong_base": bool(pred_gnn[node_idx].item() != self.labels[node_idx]),
                "is_wrong_final": bool(final_pred[node_idx].item() != self.labels[node_idx]),
            }
            if lm_only_pred is not None:
                row["lm_only_pred"] = int(lm_only_pred[node_idx].item())
            per_node_rows.append(row)
        write_text(
            stage_dir / "per_node_test.jsonl",
            "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in per_node_rows) + "\n",
        )
        analysis_summary = self._build_refiner_analysis_summary(per_node_rows, "full_graph_supervised_refiner")
        write_json(stage_dir / "analysis_summary.json", analysis_summary)

        save_stage_artifacts(
            stage_dir,
            {
                "outputs.pt": {
                    "logits": final_logits,
                    "prob": final_prob,
                    "pred": final_pred,
                    "labels": labels_t,
                    "base_pred": pred_gnn,
                    "neighbor_count_1hop": torch.tensor(semantic_views["count_1hop"], dtype=torch.long),
                    "neighbor_count_2hop": torch.tensor(semantic_views["count_2hop"], dtype=torch.long),
                },
                "checkpoint.pt": {
                    "model": model.state_dict(),
                    "fit_summary": fit_summary,
                },
                "refiner_summary": {
                    "scope": "full_graph_supervised_refiner",
                    "warning": "seed_1 local validation path uses shared LM embedding and per-seed frozen_g0 backbone",
                },
                "analysis_summary": analysis_summary,
                **base_bundle,
            },
        )

        notes = [
            "# GLANCE full-graph refiner",
            "",
            "This stage trains a full-graph supervised refiner on train_idx and selects the best epoch on valid_idx.",
            "The refiner input is [z_gnn || z_sem_0 || z_sem_1 || z_sem_2] with semantic embeddings from semantic_finetune or explicit --emb_path.",
            "No router is trained and no generative LLM is queried.",
            "Research positioning: no-router lower-bound / router ablation for diagnosing where the refiner helps or harms.",
            "Current local validation path is seed_1 only; multi-seed requires matching frozen_g0 artifacts per seed.",
        ]
        write_text(stage_dir / "notes.md", "\n".join(notes) + "\n")
        return {
            "stage": "glance_full_graph_refine",
            "stage_dir": str(stage_dir),
            "metrics": metrics_payload,
        }

    def _run_glance_counterfactual_router(self, stage_dir, base_bundle):
        context = self.ensure_backbone_context()
        semantic = self._load_semantic_finetune_artifact()
        refiner_artifact = self._load_glance_refiner_artifact()
        gnn_outputs = context["gnn_outputs"]
        p_gnn = gnn_outputs.get("prob")
        logits_gnn = gnn_outputs.get("logits")
        pred_gnn = gnn_outputs.get("pred")
        z_gnn = gnn_outputs.get("node_repr")
        if any(item is None for item in (p_gnn, logits_gnn, pred_gnn, z_gnn)):
            raise MissingFrozenArtifactError(
                "glance_counterfactual_router requires frozen_g0 outputs with prob/logits/pred/node_repr."
            )

        p_gnn = p_gnn.detach().cpu().float()
        logits_gnn = logits_gnn.detach().cpu().float()
        pred_gnn = pred_gnn.detach().cpu().long()
        z_gnn = z_gnn.detach().cpu().float()
        labels_t = torch.tensor(self.labels, dtype=torch.long)
        labels_np = labels_t.numpy()
        train_idx = _idx_numpy(self.data["train_idx"])
        valid_idx = _idx_numpy(self.data["valid_idx"])
        test_idx = _idx_numpy(self.data["test_idx"])
        semantic_views = self._build_k_hop_semantic_views(
            semantic["embeddings"],
            self.data.get("edge_index"),
        )
        if train_idx.size == 0:
            raise MissingFrozenArtifactError("glance_counterfactual_router requires a non-empty train_idx.")
        ref_outputs = refiner_artifact["outputs"]
        prob_ref = ref_outputs.get("prob")
        pred_ref = ref_outputs.get("pred")
        logits_ref = ref_outputs.get("logits")
        if prob_ref is None or pred_ref is None:
            raise MissingFrozenArtifactError(
                f"{refiner_artifact['stage_name']} outputs must include 'prob' and 'pred' for counterfactual routing."
            )
        prob_ref = prob_ref.detach().cpu().float() if torch.is_tensor(prob_ref) else torch.tensor(prob_ref, dtype=torch.float32)
        pred_ref = pred_ref.detach().cpu().long() if torch.is_tensor(pred_ref) else torch.tensor(pred_ref, dtype=torch.long)
        logits_ref = logits_ref.detach().cpu().float() if torch.is_tensor(logits_ref) else (
            torch.log(prob_ref.clamp_min(1e-8)) if logits_ref is None else torch.tensor(logits_ref, dtype=torch.float32)
        )

        gnn_loss = _node_cross_entropy_vector(p_gnn, labels_t)
        refiner_loss = _node_cross_entropy_vector(prob_ref, labels_t)
        gnn_correct = pred_gnn.eq(labels_t).float()
        refiner_correct = pred_ref.eq(labels_t).float()
        advantage = (gnn_loss - refiner_loss - float(getattr(self.args, "glance_llm_query_cost", 0.2))).numpy()

        budgets = self._router_budgets()
        router = GlanceForContextResidualRiskSelector(
            budgets=budgets,
            training_objective="glance_advantage",
            llm_query_cost=float(getattr(self.args, "glance_llm_query_cost", 0.2)),
        )
        router.fit(
            logits_gnn,
            p_gnn,
            labels_t,
            train_idx=train_idx,
            val_idx=valid_idx,
            edge_index=self.data.get("edge_index"),
            edge_type=self.data.get("edge_type"),
            node_repr=z_gnn,
            fit_idx=train_idx,
            gnn_loss=gnn_loss,
            llm_loss=refiner_loss,
            gnn_correct=gnn_correct,
            llm_correct=refiner_correct,
            training_objective="glance_advantage",
            llm_query_cost=float(getattr(self.args, "glance_llm_query_cost", 0.2)),
        )
        risk_manifest = router.build_manifest(
            logits_gnn,
            p_gnn,
            labels_t,
            val_idx=valid_idx,
            test_idx=test_idx,
            edge_index=self.data.get("edge_index"),
            edge_type=self.data.get("edge_type"),
            node_repr=z_gnn,
            q_probs=p_gnn,
            gnn_loss=gnn_loss,
            llm_loss=refiner_loss,
            gnn_correct=gnn_correct,
            llm_correct=refiner_correct,
            training_objective="glance_advantage",
            llm_query_cost=float(getattr(self.args, "glance_llm_query_cost", 0.2)),
        )
        risk_score = np.asarray(risk_manifest["risk_score"], dtype=np.float32)
        calibration_metadata = dict(risk_manifest.get("calibration_metadata", {}))
        split_audit = {
            "train_count": int(train_idx.size),
            "valid_count": int(valid_idx.size),
            "test_count": int(test_idx.size),
            "train_valid_overlap": int(np.intersect1d(train_idx, valid_idx).size),
            "train_test_overlap": int(np.intersect1d(train_idx, test_idx).size),
            "valid_test_overlap": int(np.intersect1d(valid_idx, test_idx).size),
        }
        calibration_metadata.update({
            "source": "glance_counterfactual_router",
            "paper_identity": "GLANCE-inspired counterfactual advantage router",
            "paper_faithful_glance_inspired": True,
            "official_code_verified": False,
            "counterfactual_outcome_used": True,
            "llm_counterfactual_outcome_used": True,
            "not_a_new_bot_classifier": True,
            "risk_score_semantics": "learned_router_score_proxy_for_counterfactual_advantage",
            "router_score_semantics": "deterministic_top_k_proxy_for_routing",
            "oracle_advantage_semantics": "counterfactual_reward_loss_gnn_minus_loss_refiner_minus_query_cost",
            "test_labels_used_for_training": False,
            "test_labels_used_for_threshold": False,
            "train_target": "1[loss_gnn - loss_refiner - cost > 0]",
            "refiner_input": "[z_gnn || z_sem_0 || z_sem_1 || z_sem_2]",
            "semantic_source": "semantic_finetune_roberta_finetuned",
            "split_audit": split_audit,
            "counterfactual_source_refiner": refiner_artifact["stage_name"],
        })
        risk_manifest["calibration_metadata"] = calibration_metadata

        valid_budget_rows, valid_budget_payloads = self._build_counterfactual_budget_rows(
            budgets=budgets,
            risk_score=risk_score,
            pred_gnn=pred_gnn.numpy(),
            pred_refiner=pred_ref.numpy(),
            labels_np=labels_np,
            eval_idx=valid_idx,
            prob_gnn=p_gnn,
            prob_refiner=prob_ref,
        )
        test_budget_rows, test_budget_payloads = self._build_counterfactual_budget_rows(
            budgets=budgets,
            risk_score=risk_score,
            pred_gnn=pred_gnn.numpy(),
            pred_refiner=pred_ref.numpy(),
            labels_np=labels_np,
            eval_idx=test_idx,
            prob_gnn=p_gnn,
            prob_refiner=prob_ref,
        )
        base_test = _score_all(labels_np[test_idx], pred_gnn.numpy()[test_idx])
        full_refiner_test = _score_all(labels_np[test_idx], pred_ref.numpy()[test_idx])
        full_delta = _delta_table(pred_gnn.numpy(), pred_ref.numpy(), labels_np, self.test_mask)
        base_wrong_count = int((pred_gnn.numpy()[test_idx] != labels_np[test_idx]).sum())
        base_correct_count = int(test_idx.size - base_wrong_count)
        selected_valid_row, best_key, valid_rows_by_key = self._select_best_budget_row(valid_budget_rows)
        test_rows_by_key = {str(self._budget_key(item["budget"])): item for item in test_budget_rows}
        selected_test_row = test_rows_by_key.get(best_key) if best_key is not None else None
        best_payload = test_budget_payloads.get(best_key, {}) if best_key is not None else {}
        best_final_pred = best_payload.get("final_pred", pred_gnn)
        best_routed_mask = best_payload.get("routed_mask", torch.zeros_like(pred_gnn, dtype=torch.bool))

        metrics_payload = {
            "contract": "glance_counterfactual_router_metrics_v1",
            "routing_mode": "glance_counterfactual_advantage",
            "paper_faithful_glance_inspired": True,
            "official_code_verified": False,
            "test_labels_used_for_training": False,
            "test_labels_used_for_threshold": False,
            "base_test": base_test,
            "full_refiner_test": full_refiner_test,
            "full_refiner_delta_vs_base": {
                "accuracy": float(full_refiner_test["accuracy"] - base_test["accuracy"]),
                "macro_f1": float(full_refiner_test["macro_f1"] - base_test["macro_f1"]),
                "bot_f1": float(full_refiner_test["bot_f1"] - base_test["bot_f1"]),
            },
            "full_refiner_fix_break": {
                "fix": int(full_delta["fix"]),
                "break": int(full_delta["broke"]),
                "net": int(full_delta["net"]),
                "wrong_node_fix_rate": float(full_delta["fix"] / base_wrong_count) if base_wrong_count else 0.0,
                "correct_node_break_rate": float(full_delta["broke"] / base_correct_count) if base_correct_count else 0.0,
            },
            "router_score_semantics": "learned_router_score_proxy_for_counterfactual_advantage",
            "oracle_advantage_semantics": "counterfactual_reward_loss_gnn_minus_loss_refiner_minus_query_cost",
            "selected_budget_source": "valid",
            "selected_budget": float(selected_valid_row["budget"]) if selected_valid_row else None,
            "selected_budget_valid": float(selected_valid_row["budget"]) if selected_valid_row else None,
            "selected_budget_key": best_key,
            "selected_valid_budget_metrics": selected_valid_row,
            "selected_test_budget_metrics_under_valid_choice": selected_test_row,
            "selected_budget_metrics": selected_test_row,
            "overall_delta_vs_base_gnn": selected_test_row and {
                "accuracy": float(selected_test_row["delta_accuracy"]),
                "macro_f1": float(selected_test_row["delta_macro_f1"]),
                "bot_f1": float(selected_test_row["delta_bot_f1"]),
            },
            "wrong_node_fix_rate": float(selected_test_row["wrong_node_fix_rate"]) if selected_test_row else 0.0,
            "correct_node_break_rate": float(selected_test_row["correct_node_break_rate"]) if selected_test_row else 0.0,
            "improved_count": int(selected_test_row["fix"]) if selected_test_row else 0,
            "degraded_count": int(selected_test_row["break"]) if selected_test_row else 0,
            "net_gain": int(selected_test_row["net"]) if selected_test_row else 0,
            "base_wrong_count": base_wrong_count,
            "base_correct_count": base_correct_count,
            "budget_curve": test_budget_rows,
            "budget_curve_legacy_scope": "test_sweep_diagnostic_only",
            "valid_budget_curve": valid_budget_rows,
            "test_budget_curve": test_budget_rows,
            "counterfactual_refiner_fit_summary": refiner_artifact["checkpoint"].get("fit_summary", {}),
            "router_training_summary": router.training_summary,
            "split_scope": {
                "train": int(train_idx.size),
                "valid": int(valid_idx.size),
                "test": int(test_idx.size),
            },
        }
        manifest = {
            "contract": "glance_counterfactual_router_v1",
            "status": "completed",
            "routing_mode": "glance_counterfactual_advantage",
            "paper_faithful_glance_inspired": True,
            "diagnostic_lane_only": True,
            "official_code_verified": False,
            "not_official_reproduction": True,
            "test_labels_used_for_training": False,
            "test_labels_used_for_threshold": False,
            "selected_budget_source": "valid",
            "counterfactual_outcome_used": True,
            "not_a_new_bot_classifier": True,
            "research_positioning": "paper_faithful_glance_inspired_counterfactual_router_for_refiner_utility",
            "no_llm_generation": True,
            "semantic_source": "semantic_finetune_roberta_finetuned",
            "semantic_stage_dir": str(semantic["dir"]),
            "semantic_manifest": semantic["manifest"],
            "frozen_g0_dir": str(context["frozen_g0"]["dir"]),
            "frozen_g0_manifest": context["frozen_g0"]["manifest"],
            "counterfactual_source_refiner": refiner_artifact["stage_name"],
            "counterfactual_source_refiner_manifest": refiner_artifact["manifest"],
            "router_training_target": "1[loss_gnn - loss_refiner - cost > 0]",
            "glance_llm_query_cost": float(getattr(self.args, "glance_llm_query_cost", 0.2)),
            "budgets": [float(item) for item in budgets],
            "selected_budget": float(selected_valid_row["budget"]) if selected_valid_row else None,
            "selected_budget_valid": float(selected_valid_row["budget"]) if selected_valid_row else None,
            "split_audit": split_audit,
            "refiner_architecture": {
                "type": str(refiner_artifact["manifest"].get("refiner_architecture", {}).get("type", "external_refiner")),
                "input": str(refiner_artifact["manifest"].get("refiner_architecture", {}).get("input", "external_refiner_input")),
                "hidden_dim": refiner_artifact["manifest"].get("refiner_architecture", {}).get("hidden_dim"),
                "activation": refiner_artifact["manifest"].get("refiner_architecture", {}).get("activation"),
                "dropout": refiner_artifact["manifest"].get("refiner_architecture", {}).get("dropout"),
                "output_dim": int(logits_gnn.shape[1]),
            },
        }
        self._write_stage_manifest(
            stage_dir,
            manifest,
            artifact_namespace="stages/glance_budgeted_refinement_internal",
            visibility="internal",
            resolved_task="glance_budgeted_refinement_internal",
        )
        write_json(stage_dir / "metrics.json", metrics_payload)
        write_json(stage_dir / "risk_manifest.json", risk_manifest)
        budget_payload_json = {}
        budget_payload_tensors = {}
        for budget_key, payload in test_budget_payloads.items():
            budget_payload_json[budget_key] = {
                "metrics": payload["metrics"],
                "delta_vs_base": payload["delta_vs_base"],
                "routed_node_ids": payload["routed_node_ids"],
            }
            budget_payload_tensors[f"budget_payload_{budget_key}.pt"] = {
                "final_pred": payload["final_pred"],
                "routed_mask": payload["routed_mask"],
                "routed_node_ids": torch.tensor(payload["routed_node_ids"], dtype=torch.long),
            }
        write_json(stage_dir / "budget_payloads.json", budget_payload_json)
        write_csv_rows(
            stage_dir / "budget_curve.csv",
            [
                "budget",
                "routed_count",
                "fix",
                "break",
                "net",
                "routed_fix",
                "routed_break",
                "routed_net",
                "wrong_node_fix_rate",
                "correct_node_break_rate",
                "accuracy",
                "macro_f1",
                "bot_f1",
                "delta_accuracy",
                "delta_macro_f1",
                "delta_bot_f1",
                "mean_base_confidence_routed",
                "mean_refiner_confidence_routed",
            ],
            test_budget_rows,
        )
        write_csv_rows(
            stage_dir / "valid_budget_curve.csv",
            [
                "budget",
                "routed_count",
                "fix",
                "break",
                "net",
                "routed_fix",
                "routed_break",
                "routed_net",
                "wrong_node_fix_rate",
                "correct_node_break_rate",
                "accuracy",
                "macro_f1",
                "bot_f1",
                "delta_accuracy",
                "delta_macro_f1",
                "delta_bot_f1",
                "mean_base_confidence_routed",
                "mean_refiner_confidence_routed",
            ],
            valid_budget_rows,
        )
        write_csv_rows(
            stage_dir / "test_budget_curve.csv",
            [
                "budget",
                "routed_count",
                "fix",
                "break",
                "net",
                "routed_fix",
                "routed_break",
                "routed_net",
                "wrong_node_fix_rate",
                "correct_node_break_rate",
                "accuracy",
                "macro_f1",
                "bot_f1",
                "delta_accuracy",
                "delta_macro_f1",
                "delta_bot_f1",
                "mean_base_confidence_routed",
                "mean_refiner_confidence_routed",
            ],
            test_budget_rows,
        )

        per_node_rows = []
        best_routed_np = best_routed_mask.detach().cpu().numpy().astype(bool)
        best_final_np = best_final_pred.detach().cpu().numpy().astype(np.int64)
        gnn_loss_np = gnn_loss.numpy()
        ref_loss_np = refiner_loss.numpy()
        for node_idx in test_idx.tolist():
            base_correct = bool(pred_gnn[node_idx].item() == int(labels_np[node_idx]))
            ref_correct = bool(pred_ref[node_idx].item() == int(labels_np[node_idx]))
            final_correct = bool(best_final_np[node_idx] == int(labels_np[node_idx]))
            per_node_rows.append({
                "node_id": int(node_idx),
                "base_pred": int(pred_gnn[node_idx].item()),
                "refiner_pred": int(pred_ref[node_idx].item()),
                "final_pred": int(best_final_np[node_idx]),
                "label": int(labels_np[node_idx]),
                "routed": bool(best_routed_np[node_idx]),
                "advantage_score": float(risk_score[node_idx]),
                "oracle_advantage": float(advantage[node_idx]),
                "base_loss": float(gnn_loss_np[node_idx]),
                "refiner_loss": float(ref_loss_np[node_idx]),
                "base_correct": base_correct,
                "refiner_correct": ref_correct,
                "final_correct": final_correct,
                "fix": bool((not base_correct) and final_correct),
                "break": bool(base_correct and (not final_correct)),
                "neighbor_count_1hop": int(semantic_views["count_1hop"][node_idx]),
                "neighbor_count_2hop": int(semantic_views["count_2hop"][node_idx]),
            })
        write_text(
            stage_dir / "per_node_test.jsonl",
            "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in per_node_rows) + "\n",
        )
        save_stage_artifacts(
            stage_dir,
            {
                "outputs.pt": {
                    "base_logits": logits_gnn,
                    "base_prob": p_gnn,
                    "base_pred": pred_gnn,
                    "refiner_logits": logits_ref,
                    "refiner_prob": prob_ref,
                    "refiner_pred": pred_ref,
                    "final_pred": best_final_pred,
                    "labels": labels_t,
                    "risk_score": torch.tensor(risk_score, dtype=torch.float32),
                    "router_score": torch.tensor(risk_score, dtype=torch.float32),
                    "selected_routed_mask": best_routed_mask,
                    "gnn_loss": gnn_loss,
                    "refiner_loss": refiner_loss,
                    "oracle_advantage": torch.tensor(advantage, dtype=torch.float32),
                    "neighbor_count_1hop": torch.tensor(semantic_views["count_1hop"], dtype=torch.long),
                    "neighbor_count_2hop": torch.tensor(semantic_views["count_2hop"], dtype=torch.long),
                },
                "checkpoint.pt": {
                    "counterfactual_source_refiner": refiner_artifact["stage_name"],
                    "counterfactual_source_refiner_fit_summary": refiner_artifact["checkpoint"].get("fit_summary", {}),
                    "router_state": router.state_dict_payload(),
                },
                "risk_manifest": risk_manifest,
                "valid_budget_curve": valid_budget_rows,
                "test_budget_curve": test_budget_rows,
                "budget_payloads": budget_payload_json,
                **budget_payload_tensors,
                **base_bundle,
            },
        )
        notes = [
            "# GLANCE counterfactual router",
            "",
            "This stage is a diagnostic utility-selector lane, not the main strict GLANCE baseline.",
            "It is a paper-faithful GLANCE-inspired adaptation for offline routed-set analysis, not an official reproduction.",
            "The learned router score is the proxy used for deterministic top-k routing.",
            "The oracle_advantage tensor is the post-hoc counterfactual reward trace: loss_gnn - loss_refiner - query_cost.",
            "The legacy risk_score field is retained as a compatibility alias of router_score.",
            "Budget sweeps are computed on both validation and test, but the final operating budget is selected on validation only and then locked for test reporting.",
            "No generative LLM is queried; semantic views come from cached/fine-tuned RoBERTa embeddings.",
        ]
        write_text(stage_dir / "notes.md", "\n".join(notes) + "\n")
        return {
            "stage": "glance_counterfactual_router",
            "stage_dir": str(stage_dir),
            "metrics": metrics_payload,
        }

    @staticmethod
    def _iter_glance_batches(idx, batch_size, shuffle=False, seed=0):
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive for GLANCE routing batches.")
        if shuffle:
            rng = np.random.default_rng(int(seed))
            idx = idx[rng.permutation(idx.size)]
        for start in range(0, idx.size, int(batch_size)):
            yield idx[start : start + int(batch_size)]

    @staticmethod
    def _glance_training_top_k(batch_size, epoch_index, decay_factor=0.5):
        batch_size = max(int(batch_size), 1)
        k_start = batch_size
        k_end = max(int(round(batch_size / 4.0)), 1)
        value = k_end + (k_start - k_end) * (float(decay_factor) ** int(epoch_index))
        return max(min(int(round(value)), batch_size), 1)

    @staticmethod
    def _glance_refiner_forward(refiner_model, batch_refiner, batch_base_logits=None):
        if isinstance(batch_refiner, dict):
            feature_kind = batch_refiner.get("feature_kind")
            if _is_prompt_expert_feature_kind(feature_kind) or _is_prompt_expert_semantic_view_mode(batch_refiner.get("semantic_view_mode")):
                outputs = refiner_model(
                    batch_refiner["z_gnn"],
                    batch_refiner["semantic_views"],
                    batch_refiner["structural_features"],
                    batch_refiner.get("graph_gate_features"),
                    batch_refiner.get("mpe_gate_features"),
                    batch_refiner.get("expert_presence_mask"),
                    batch_refiner.get("selector_context_views"),
                )
            else:
                raise ValueError(f"Unsupported refiner feature kind: {feature_kind}")
        else:
            outputs = refiner_model(batch_refiner)
        if isinstance(outputs, tuple):
            mpe_gate_weights = None
            mpe_selected_expert = None
            if len(outputs) == 5:
                refiner_logits, gate_logits, gate_prob, mpe_gate_weights, mpe_selected_expert = outputs
            elif len(outputs) == 4:
                refiner_logits, gate_logits, gate_prob, mpe_gate_weights = outputs
            elif len(outputs) == 3:
                refiner_logits, second, third = outputs
                if second is not None and second.dim() == 2 and third is not None and third.dim() == 1:
                    mpe_gate_weights = second
                    mpe_selected_expert = third.long()
                    gate_prob = None
                    gate_logits = None
                else:
                    gate_logits = second
                    gate_prob = third
            elif len(outputs) == 2:
                refiner_logits, second = outputs
                if second is not None and second.dim() == 2:
                    mpe_gate_weights = second
                    gate_prob = None
                    gate_logits = None
                else:
                    gate_prob = second
                    gate_logits = torch.logit(gate_prob.clamp_min(1e-6).clamp_max(1.0 - 1e-6))
            else:
                raise ValueError("Unsupported gated refiner output arity.")
        else:
            refiner_logits = outputs
            gate_logits = None
            gate_prob = None
            mpe_gate_weights = None
            mpe_selected_expert = None
        mixed_logits = refiner_logits
        if gate_prob is not None:
            if batch_base_logits is None:
                raise ValueError("Gated refiner requires batch_base_logits for mixed inference.")
            base_prob = torch.softmax(batch_base_logits, dim=1)
            ref_prob = torch.softmax(refiner_logits, dim=1)
            mixed_prob = (1.0 - gate_prob.unsqueeze(1)) * base_prob + gate_prob.unsqueeze(1) * ref_prob
            mixed_prob = mixed_prob.clamp_min(1e-8)
            mixed_prob = mixed_prob / mixed_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
            mixed_logits = torch.log(mixed_prob)
        return {
            "refiner_logits": refiner_logits,
            "mixed_logits": mixed_logits,
            "gate_logits": gate_logits,
            "gate_prob": gate_prob,
            "mpe_gate_weights": mpe_gate_weights,
            "mpe_selected_expert": mpe_selected_expert,
        }

    @staticmethod
    def _glance_refiner_sample_weights(base_wrong_target, utility_target, mode, base_wrong_weight=2.0, utility_weight=3.0):
        mode = str(mode or "off").lower()
        weights = torch.ones_like(base_wrong_target, dtype=torch.float32)
        if mode in {"base_wrong", "base_wrong_plus_utility"}:
            weights = weights + base_wrong_target.float() * max(float(base_wrong_weight) - 1.0, 0.0)
        if mode in {"utility_positive", "base_wrong_plus_utility"}:
            weights = weights + utility_target.float() * max(float(utility_weight) - 1.0, 0.0)
        return weights

    @staticmethod
    def _glance_gate_output_metrics(outputs, eval_idx):
        gate_prob = outputs.get("gate_prob") if isinstance(outputs, dict) else None
        oracle_advantage = outputs.get("oracle_advantage") if isinstance(outputs, dict) else None
        routed_mask = outputs.get("routed_mask") if isinstance(outputs, dict) else None
        if gate_prob is None or oracle_advantage is None or routed_mask is None:
            return {
                "gate_positive_rate": 0.0,
                "mean_gate_prob_routed": 0.0,
                "gate_precision": 0.0,
                "gate_recall": 0.0,
            }
        idx = torch.tensor(np.asarray(eval_idx, dtype=np.int64).reshape(-1), dtype=torch.long)
        if idx.numel() == 0:
            return {
                "gate_positive_rate": 0.0,
                "mean_gate_prob_routed": 0.0,
                "gate_precision": 0.0,
                "gate_recall": 0.0,
            }
        routed = routed_mask[idx].bool()
        if int(routed.sum().item()) == 0:
            return {
                "gate_positive_rate": 0.0,
                "mean_gate_prob_routed": 0.0,
                "gate_precision": 0.0,
                "gate_recall": 0.0,
            }
        gate_routed = gate_prob[idx][routed].detach().cpu().float()
        adv_routed = oracle_advantage[idx][routed].detach().cpu().float()
        gate_decision = gate_routed >= 0.5
        utility_positive = adv_routed > 0.0
        true_positive = gate_decision & utility_positive
        return {
            "gate_positive_rate": float(gate_decision.float().mean().item()),
            "mean_gate_prob_routed": float(gate_routed.mean().item()),
            "gate_precision": float(true_positive.sum().item() / max(int(gate_decision.sum().item()), 1)),
            "gate_recall": float(true_positive.sum().item() / max(int(utility_positive.sum().item()), 1)),
        }

    def _apply_glance_joint_policy(
        self,
        router_model,
        refiner_model,
        router_features,
        refiner_features,
        semantic_view_mode,
        base_logits,
        base_prob,
        base_pred,
        labels_t,
        beta,
        split_idx,
        batch_size,
        top_k,
    ):
        split_idx = np.asarray(split_idx, dtype=np.int64).reshape(-1)
        final_logits = base_logits.clone()
        final_prob = base_prob.clone()
        final_pred = base_pred.clone()
        eligible_refiner_mask = _full_graph_eligible_refiner_mask(refiner_features, base_pred.numel())
        selector_component_order = (
            list(refiner_features.get("selector_component_order", list(_GAUGLLM_SELECTOR_COMPONENT_ORDER)))
            if isinstance(refiner_features, dict)
            else list(_GAUGLLM_SELECTOR_COMPONENT_ORDER)
        )
        routed_mask = torch.zeros(base_pred.numel(), dtype=torch.bool)
        router_prob_all = torch.zeros(base_pred.numel(), dtype=torch.float32)
        router_score_all = torch.zeros(base_pred.numel(), dtype=torch.float32)
        oracle_advantage_all = torch.zeros(base_pred.numel(), dtype=torch.float32)
        gate_prob_all = torch.zeros(base_pred.numel(), dtype=torch.float32)
        mpe_gate_weights_all = torch.zeros((base_pred.numel(), len(selector_component_order)), dtype=torch.float32)
        mpe_selected_expert_all = torch.full((base_pred.numel(),), -1, dtype=torch.long)
        if split_idx.size == 0:
            return {
                "logits": final_logits,
                "prob": final_prob,
                "pred": final_pred,
                "routed_mask": routed_mask,
                "eligible_refiner_mask": eligible_refiner_mask,
                "router_prob": router_prob_all,
                "router_score": router_score_all,
                "oracle_advantage": oracle_advantage_all,
                "gate_prob": gate_prob_all,
                "mpe_gate_weights": mpe_gate_weights_all,
                "mpe_selected_expert": mpe_selected_expert_all,
                "selector_component_order": selector_component_order,
                "routed_count": 0,
                "eligible_count": 0,
            }

        router_model.eval()
        refiner_model.eval()
        router_device = next(router_model.parameters()).device
        refiner_device = next(refiner_model.parameters()).device
        with torch.no_grad():
            for batch_idx_np in self._iter_glance_batches(split_idx, batch_size, shuffle=False):
                batch_idx = torch.tensor(batch_idx_np, dtype=torch.long)
                batch_score, batch_prob = router_model(router_features[batch_idx].to(router_device))
                batch_score = batch_score.detach().cpu()
                batch_prob = batch_prob.detach().cpu()
                router_score_all[batch_idx] = batch_score
                router_prob_all[batch_idx] = batch_prob
                eligible_rel_cpu = torch.nonzero(eligible_refiner_mask[batch_idx], as_tuple=False).view(-1).long()
                eligible_idx = batch_idx[eligible_rel_cpu]
                if int(eligible_idx.numel()) > 0:
                    eligible_scores = batch_score[eligible_rel_cpu]
                    k = min(max(int(top_k), 1), int(eligible_idx.numel()))
                    routed_rel_local = torch.topk(eligible_scores, k=k, largest=True, sorted=False).indices
                    routed_rel_cpu = eligible_rel_cpu[routed_rel_local.detach().cpu()]
                    routed_idx = batch_idx[routed_rel_cpu]
                    routed_mask[routed_idx] = True
                    routed_forward = self._glance_refiner_forward(
                        refiner_model,
                        _refiner_feature_lookup(refiner_features, routed_idx, semantic_view_mode, refiner_device),
                        batch_base_logits=base_logits[routed_idx].to(refiner_device),
                    )
                    final_logits[routed_idx] = routed_forward["mixed_logits"].cpu()
                    final_prob[routed_idx] = torch.softmax(routed_forward["mixed_logits"].cpu(), dim=1)
                    final_pred[routed_idx] = routed_forward["mixed_logits"].cpu().argmax(dim=1)
                    if routed_forward.get("gate_prob") is not None:
                        gate_prob_all[routed_idx] = routed_forward["gate_prob"].detach().cpu()
                    if routed_forward.get("mpe_gate_weights") is not None:
                        mpe_gate_weights_all[routed_idx] = routed_forward["mpe_gate_weights"].detach().cpu()
                    if routed_forward.get("mpe_selected_expert") is not None:
                        mpe_selected_expert_all[routed_idx] = routed_forward["mpe_selected_expert"].detach().cpu().long()
                    eligible_forward = self._glance_refiner_forward(
                        refiner_model,
                        _refiner_feature_lookup(refiner_features, eligible_idx, semantic_view_mode, refiner_device),
                        batch_base_logits=base_logits[eligible_idx].to(refiner_device),
                    )
                    raw_full_ref_loss = F.cross_entropy(
                        eligible_forward["refiner_logits"].cpu(),
                        labels_t[eligible_idx].to(refiner_device).cpu(),
                        reduction="none",
                    )
                    base_loss = F.cross_entropy(
                        base_logits[eligible_idx],
                        labels_t[eligible_idx],
                        reduction="none",
                    )
                    oracle_advantage_all[eligible_idx] = base_loss.detach().cpu() - raw_full_ref_loss.detach().cpu() - float(beta)

        return {
            "logits": final_logits,
            "prob": final_prob,
            "pred": final_pred,
            "routed_mask": routed_mask,
            "eligible_refiner_mask": eligible_refiner_mask,
            "router_prob": router_prob_all,
            "router_score": router_score_all,
            "oracle_advantage": oracle_advantage_all,
            "gate_prob": gate_prob_all,
            "mpe_gate_weights": mpe_gate_weights_all,
            "mpe_selected_expert": mpe_selected_expert_all,
            "selector_component_order": selector_component_order,
            "routed_count": int(routed_mask[torch.tensor(split_idx, dtype=torch.long)].sum().item()),
            "eligible_count": int(eligible_refiner_mask[torch.tensor(split_idx, dtype=torch.long)].sum().item()),
        }

    def _apply_glance_global_budget_policy(
        self,
        router_model,
        refiner_model,
        router_features,
        refiner_features,
        semantic_view_mode,
        base_logits,
        base_prob,
        base_pred,
        labels_t,
        beta,
        split_idx,
        batch_size,
        budget,
    ):
        split_idx = np.asarray(split_idx, dtype=np.int64).reshape(-1)
        final_logits = base_logits.clone()
        final_prob = base_prob.clone()
        final_pred = base_pred.clone()
        eligible_refiner_mask = _full_graph_eligible_refiner_mask(refiner_features, base_pred.numel())
        selector_component_order = (
            list(refiner_features.get("selector_component_order", list(_GAUGLLM_SELECTOR_COMPONENT_ORDER)))
            if isinstance(refiner_features, dict)
            else list(_GAUGLLM_SELECTOR_COMPONENT_ORDER)
        )
        routed_mask = torch.zeros(base_pred.numel(), dtype=torch.bool)
        router_prob_all = torch.zeros(base_pred.numel(), dtype=torch.float32)
        router_score_all = torch.zeros(base_pred.numel(), dtype=torch.float32)
        oracle_advantage_all = torch.zeros(base_pred.numel(), dtype=torch.float32)
        gate_prob_all = torch.zeros(base_pred.numel(), dtype=torch.float32)
        mpe_gate_weights_all = torch.zeros((base_pred.numel(), len(selector_component_order)), dtype=torch.float32)
        mpe_selected_expert_all = torch.full((base_pred.numel(),), -1, dtype=torch.long)
        if split_idx.size == 0:
            return {
                "logits": final_logits,
                "prob": final_prob,
                "pred": final_pred,
                "routed_mask": routed_mask,
                "eligible_refiner_mask": eligible_refiner_mask,
                "router_prob": router_prob_all,
                "router_score": router_score_all,
                "oracle_advantage": oracle_advantage_all,
                "gate_prob": gate_prob_all,
                "mpe_gate_weights": mpe_gate_weights_all,
                "mpe_selected_expert": mpe_selected_expert_all,
                "selector_component_order": selector_component_order,
                "routed_count": 0,
                "eligible_count": 0,
            }

        router_model.eval()
        refiner_model.eval()
        router_device = next(router_model.parameters()).device
        refiner_device = next(refiner_model.parameters()).device
        split_scores = []
        split_probs = []
        with torch.no_grad():
            for batch_idx_np in self._iter_glance_batches(split_idx, batch_size, shuffle=False):
                batch_idx = torch.tensor(batch_idx_np, dtype=torch.long)
                batch_score, batch_prob = router_model(router_features[batch_idx].to(router_device))
                batch_score = batch_score.detach().cpu()
                batch_prob = batch_prob.detach().cpu()
                router_score_all[batch_idx] = batch_score
                router_prob_all[batch_idx] = batch_prob
                split_scores.append(batch_score)
                split_probs.append(batch_prob)
                eligible_rel_cpu = torch.nonzero(eligible_refiner_mask[batch_idx], as_tuple=False).view(-1).long()
                eligible_idx = batch_idx[eligible_rel_cpu]
                if int(eligible_idx.numel()) > 0:
                    eligible_forward = self._glance_refiner_forward(
                        refiner_model,
                        _refiner_feature_lookup(refiner_features, eligible_idx, semantic_view_mode, refiner_device),
                        batch_base_logits=base_logits[eligible_idx].to(refiner_device),
                    )
                    raw_full_ref_loss = F.cross_entropy(
                        eligible_forward["refiner_logits"].cpu(),
                        labels_t[eligible_idx].to(refiner_device).cpu(),
                        reduction="none",
                    )
                    base_loss = F.cross_entropy(base_logits[eligible_idx], labels_t[eligible_idx], reduction="none")
                    oracle_advantage_all[eligible_idx] = base_loss.detach().cpu() - raw_full_ref_loss.detach().cpu() - float(beta)

            eligible_split_idx = split_idx[eligible_refiner_mask[torch.tensor(split_idx, dtype=torch.long)].numpy()] if split_idx.size else np.asarray([], dtype=np.int64)
            order = eligible_split_idx[np.argsort(-router_score_all[eligible_split_idx].numpy())] if eligible_split_idx.size else np.asarray([], dtype=np.int64)
            k = min(max(int(round(split_idx.size * float(budget))), 1), int(eligible_split_idx.size)) if eligible_split_idx.size else 0
            routed = order[:k]
            if routed.size:
                routed_idx = torch.tensor(routed, dtype=torch.long)
                routed_mask[routed_idx] = True
                routed_forward = self._glance_refiner_forward(
                    refiner_model,
                    _refiner_feature_lookup(refiner_features, routed_idx, semantic_view_mode, refiner_device),
                    batch_base_logits=base_logits[routed_idx].to(refiner_device),
                )
                final_logits[routed_idx] = routed_forward["mixed_logits"].cpu()
                final_prob[routed_idx] = torch.softmax(routed_forward["mixed_logits"].cpu(), dim=1)
                final_pred[routed_idx] = routed_forward["mixed_logits"].cpu().argmax(dim=1)
                if routed_forward.get("gate_prob") is not None:
                    gate_prob_all[routed_idx] = routed_forward["gate_prob"].detach().cpu()
                if routed_forward.get("mpe_gate_weights") is not None:
                    mpe_gate_weights_all[routed_idx] = routed_forward["mpe_gate_weights"].detach().cpu()
                if routed_forward.get("mpe_selected_expert") is not None:
                    mpe_selected_expert_all[routed_idx] = routed_forward["mpe_selected_expert"].detach().cpu().long()

        return {
            "logits": final_logits,
            "prob": final_prob,
            "pred": final_pred,
            "routed_mask": routed_mask,
            "eligible_refiner_mask": eligible_refiner_mask,
            "router_prob": router_prob_all,
            "router_score": router_score_all,
            "oracle_advantage": oracle_advantage_all,
            "gate_prob": gate_prob_all,
            "mpe_gate_weights": mpe_gate_weights_all,
            "mpe_selected_expert": mpe_selected_expert_all,
            "selector_component_order": selector_component_order,
            "routed_count": int(routed_mask[torch.tensor(split_idx, dtype=torch.long)].sum().item()),
            "eligible_count": int(eligible_refiner_mask[torch.tensor(split_idx, dtype=torch.long)].sum().item()),
        }

    def _build_glance_joint_per_node_rows(
        self,
        split_idx,
        pred_gnn,
        final_pred,
        labels_np,
        routed_mask,
        router_prob,
        router_score,
        oracle_advantage,
        gate_prob,
        mpe_gate_weights,
        semantic_views,
        mode_name,
        route_prob_by_epoch=None,
        route_score_by_epoch=None,
        refiner_pred_by_epoch=None,
        base_pred_by_epoch=None,
        eligible_refiner_mask=None,
        mpe_selected_expert=None,
        selector_component_order=None,
    ):
        split_idx = np.asarray(split_idx, dtype=np.int64).reshape(-1)
        rows = []
        selector_component_order = list(selector_component_order or PromptExpertBundleRefinerMLP.MPE_COMPONENT_ORDER)
        final_pred_np = final_pred.detach().cpu().numpy().astype(np.int64)
        pred_gnn_np = pred_gnn.detach().cpu().numpy().astype(np.int64)
        routed_mask_np = routed_mask.detach().cpu().numpy().astype(bool)
        router_prob_np = router_prob.detach().cpu().numpy().astype(np.float32)
        router_score_np = router_score.detach().cpu().numpy().astype(np.float32) if router_score is not None else None
        oracle_advantage_np = oracle_advantage.detach().cpu().numpy().astype(np.float32) if oracle_advantage is not None else None
        gate_prob_np = gate_prob.detach().cpu().numpy().astype(np.float32) if gate_prob is not None else None
        mpe_gate_weights_np = (
            mpe_gate_weights.detach().cpu().numpy().astype(np.float32)
            if mpe_gate_weights is not None
            else None
        )
        eligible_refiner_mask_np = (
            eligible_refiner_mask.detach().cpu().numpy().astype(bool)
            if eligible_refiner_mask is not None
            else None
        )
        mpe_selected_expert_np = (
            mpe_selected_expert.detach().cpu().numpy().astype(np.int64)
            if mpe_selected_expert is not None
            else None
        )
        count_1hop = semantic_views.get("count_1hop")
        count_2hop = semantic_views.get("count_2hop")
        if count_1hop is None:
            count_1hop = semantic_views.get("count_following")
        if count_2hop is None:
            count_2hop = semantic_views.get("count_follower")

        def _selector_weight(node_idx, component_name):
            if mpe_gate_weights_np is None or component_name not in selector_component_order:
                return 0.0
            component_idx = selector_component_order.index(component_name)
            if component_idx < 0 or component_idx >= int(mpe_gate_weights_np.shape[1]):
                return 0.0
            return float(mpe_gate_weights_np[node_idx, component_idx])

        for node_idx in split_idx.tolist():
            selected_expert_idx = int(mpe_selected_expert_np[node_idx]) if mpe_selected_expert_np is not None else -1
            row = {
                "node_id": int(node_idx),
                "base_pred": int(pred_gnn_np[node_idx]),
                "final_pred": int(final_pred_np[node_idx]),
                "label": int(labels_np[node_idx]),
                "routed": bool(routed_mask_np[node_idx]),
                "eligible_refiner": bool(eligible_refiner_mask_np[node_idx]) if eligible_refiner_mask_np is not None else True,
                "router_prob": float(router_prob_np[node_idx]),
                "router_score": float(router_score_np[node_idx]) if router_score_np is not None else 0.0,
                "oracle_advantage": float(oracle_advantage_np[node_idx]) if oracle_advantage_np is not None else 0.0,
                "gate_prob": float(gate_prob_np[node_idx]) if gate_prob_np is not None else 0.0,
                "gate_decision": bool(gate_prob_np[node_idx] >= 0.5) if gate_prob_np is not None else False,
                "mpe_gate_graph_following": _selector_weight(node_idx, "graph_following"),
                "mpe_gate_graph_follower": _selector_weight(node_idx, "graph_follower"),
                "mpe_gate_tweet": _selector_weight(node_idx, "tweet"),
                "mpe_gate_conflict": _selector_weight(node_idx, "conflict"),
                "mpe_selected_expert_id": selected_expert_idx,
                "mpe_selected_expert": (
                    selector_component_order[selected_expert_idx]
                    if 0 <= selected_expert_idx < len(selector_component_order)
                    else ""
                ),
                "selector_weight_graph_following": _selector_weight(node_idx, "graph_following"),
                "selector_weight_graph_follower": _selector_weight(node_idx, "graph_follower"),
                "selector_weight_tweet": _selector_weight(node_idx, "tweet"),
                "selector_weight_conflict": _selector_weight(node_idx, "conflict"),
                "selected_expert": (
                    selector_component_order[selected_expert_idx]
                    if 0 <= selected_expert_idx < len(selector_component_order)
                    else ""
                ),
                "was_wrong_base": bool(pred_gnn_np[node_idx] != labels_np[node_idx]),
                "is_wrong_final": bool(final_pred_np[node_idx] != labels_np[node_idx]),
                "neighbor_count_1hop": int(count_1hop[node_idx]) if count_1hop is not None else 0,
                "neighbor_count_2hop": int(count_2hop[node_idx]) if count_2hop is not None else 0,
                "neighbor_count_following": int(semantic_views["count_following"][node_idx]) if "count_following" in semantic_views else 0,
                "neighbor_count_follower": int(semantic_views["count_follower"][node_idx]) if "count_follower" in semantic_views else 0,
                "has_following": int(semantic_views["has_following"][node_idx]) if "has_following" in semantic_views else 0,
                "has_follower": int(semantic_views["has_follower"][node_idx]) if "has_follower" in semantic_views else 0,
            }
            if route_prob_by_epoch is not None:
                row["router_prob_by_epoch"] = [float(v[node_idx]) for v in route_prob_by_epoch]
            if route_score_by_epoch is not None:
                row["router_score_by_epoch"] = [float(v[node_idx]) for v in route_score_by_epoch]
            if refiner_pred_by_epoch is not None:
                row["refiner_pred_by_epoch"] = [int(v[node_idx]) for v in refiner_pred_by_epoch]
            if base_pred_by_epoch is not None:
                row["base_pred_by_epoch"] = [int(v[node_idx]) for v in base_pred_by_epoch]
            rows.append(row)
        return {
            "rows": rows,
            "analysis": self._build_refiner_analysis_summary(rows, mode_name),
        }

    def _train_glance_joint_candidate(
        self,
        *,
        labels_t,
        labels_np,
        train_idx,
        valid_idx,
        test_idx,
        logits_gnn,
        p_gnn,
        pred_gnn,
        router_features,
        refiner_features,
        semantic_views,
        semantic_view_mode,
        activation,
        batch_size,
        max_epochs,
        patience,
        decay_factor,
        entropy_weight,
        router_weight,
        router_regression_weight,
        router_ranking_weight,
        router_calibration_weight,
        router_reliability_weight,
        learning_rate,
        weight_decay,
        beta,
        eval_top_k,
        refiner_explicit_gate=False,
        refiner_target_mode="predict",
        refiner_gate_target="base_wrong",
        refiner_weight_mode="off",
        refiner_base_wrong_weight=2.0,
        refiner_utility_weight=3.0,
        refiner_gate_weight=0.5,
        prompt_expert_fusion="projector_concat",
        routing_protocol="joint_train",
        frozen_router_bundle=None,
        selected_budget_override=None,
    ):
        candidate_seed = int(self.seed) * 100000 + int(round(float(beta) * 1000)) * 10 + int(eval_top_k)
        torch.manual_seed(candidate_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(candidate_seed)
        np.random.seed(candidate_seed % (2**32 - 1))
        routing_protocol = str(routing_protocol or "joint_train").lower()
        if routing_protocol == "frozen_router_reuse":
            if frozen_router_bundle is None:
                raise MissingFrozenArtifactError("frozen_router_reuse requires a loaded frozen_router_bundle.")
            router_model = frozen_router_bundle["router_model"]
            for parameter in router_model.parameters():
                parameter.requires_grad_(False)
        else:
            router_model = GlanceReliabilityRouterMLP(
                input_dim=int(router_features.shape[1]),
                hidden_dim=128,
                dropout=0.1,
            )
        refiner_feature_kind = refiner_features.get("feature_kind") if isinstance(refiner_features, dict) else None
        prompt_expert_fusion = str(prompt_expert_fusion or "projector_concat").lower()
        if _is_prompt_expert_feature_kind(refiner_feature_kind) or (
            isinstance(refiner_features, dict)
            and _is_prompt_expert_semantic_view_mode(refiner_features.get("semantic_view_mode"))
        ):
            component_dims = {
                name: int(refiner_features["semantic_views"][name].shape[1])
                for name in PromptExpertBundleRefinerMLP.COMPONENT_ORDER
            }
            refiner_model = PromptExpertBundleRefinerMLP(
                z_gnn_dim=int(refiner_features["z_gnn"].shape[1]),
                component_dims=component_dims,
                proj_dim=256,
                hidden_dim=128,
                structural_dim=int(refiner_features["structural_features"].shape[1]),
                graph_gate_dim=int(refiner_features["graph_gate_features"].shape[1]),
                activation=activation,
                dropout=0.1,
                explicit_gate=bool(refiner_explicit_gate),
                fusion_mode=prompt_expert_fusion,
                mpe_gate_dim=int(refiner_features.get("mpe_gate_features", torch.empty((0, 0))).shape[1])
                if "mpe_gate_features" in refiner_features
                else None,
            )
        elif bool(refiner_explicit_gate):
            refiner_model = GatedGlanceRefinerMLP(
                input_dim=int(refiner_features.shape[1]),
                hidden_dim=128,
                activation=activation,
                dropout=0.1,
            )
        else:
            refiner_model = GlanceRefinerMLP(
                input_dim=int(refiner_features.shape[1]),
                hidden_dim=128,
                activation=activation,
                dropout=0.1,
            )
        router_model.to(self.device)
        refiner_model.to(self.device)
        trainable_params = list(refiner_model.parameters())
        if routing_protocol != "frozen_router_reuse":
            trainable_params = list(router_model.parameters()) + trainable_params
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

        best_state = None
        best_epoch_summary = None
        best_score = (-1.0, float("inf"))
        wait = 0
        epoch_history = []
        epoch_component_curves = []
        train_wrong = (pred_gnn.numpy()[train_idx] != labels_np[train_idx]).astype(np.int64)
        valid_wrong = (pred_gnn.numpy()[valid_idx] != labels_np[valid_idx]).astype(np.int64)
        test_wrong = (pred_gnn.numpy()[test_idx] != labels_np[test_idx]).astype(np.int64)
        eligible_refiner_mask_all = _full_graph_eligible_refiner_mask(refiner_features, pred_gnn.numel())

        for epoch in range(max_epochs):
            if routing_protocol == "frozen_router_reuse":
                router_model.eval()
            else:
                router_model.train()
            refiner_model.train()
            train_loss_history = []
            train_pred_history = []
            train_route_history = []
            train_router_regression_history = []
            train_router_ranking_history = []
            train_router_selection_history = []
            train_router_reliability_history = []
            train_gate_history = []
            train_routed_weight_history = []
            train_adv_corr_history = []

            train_k = self._glance_training_top_k(batch_size, epoch, decay_factor=decay_factor)
            for batch_idx_np in self._iter_glance_batches(train_idx, batch_size, shuffle=True, seed=candidate_seed + epoch):
                batch_idx = torch.tensor(batch_idx_np, dtype=torch.long)
                batch_labels = labels_t[batch_idx].to(self.device)
                batch_router = router_features[batch_idx].to(self.device)
                batch_base_logits = logits_gnn[batch_idx].to(self.device)
                batch_base_prob = p_gnn[batch_idx].to(self.device)
                batch_base_pred = pred_gnn[batch_idx].to(self.device)
                if routing_protocol == "frozen_router_reuse":
                    with torch.no_grad():
                        batch_route_score, batch_route_prob = router_model(batch_router)
                else:
                    batch_route_score, batch_route_prob = router_model(batch_router)
                routed_mask = torch.zeros(batch_idx.numel(), dtype=torch.bool, device=self.device)
                batch_base_loss = F.cross_entropy(batch_base_logits, batch_labels, reduction="none")
                base_wrong_target = batch_base_pred.ne(batch_labels).float()
                raw_full_refiner_loss = batch_base_loss.detach().clone()
                utility_target = torch.zeros_like(base_wrong_target, dtype=torch.float32)
                oracle_advantage = torch.zeros_like(batch_base_loss, dtype=torch.float32)
                eligible_rel_cpu = torch.nonzero(eligible_refiner_mask_all[batch_idx], as_tuple=False).view(-1).long()
                eligible_rel = eligible_rel_cpu.to(self.device)
                routed_sample_weights = torch.ones((0,), dtype=torch.float32, device=self.device)
                if int(eligible_rel.numel()) > 0:
                    eligible_idx = batch_idx[eligible_rel_cpu]
                    eligible_forward = self._glance_refiner_forward(
                        refiner_model,
                        _refiner_feature_lookup(refiner_features, eligible_idx, semantic_view_mode, self.device),
                        batch_base_logits=batch_base_logits[eligible_rel],
                    )
                    eligible_raw_full_refiner_loss = F.cross_entropy(
                        eligible_forward["refiner_logits"],
                        batch_labels[eligible_rel],
                        reduction="none",
                    )
                    raw_full_refiner_loss[eligible_rel] = eligible_raw_full_refiner_loss.detach()
                    utility_target[eligible_rel] = (
                        batch_base_loss[eligible_rel].detach() - eligible_raw_full_refiner_loss.detach() - float(beta) > 0.0
                    ).float()
                    oracle_advantage[eligible_rel] = (
                        batch_base_loss[eligible_rel] - eligible_raw_full_refiner_loss.detach() - float(beta)
                    ).detach()

                    k = min(max(int(train_k), 1), int(eligible_rel.numel()))
                    eligible_scores = batch_route_score[eligible_rel]
                    routed_rel_local = torch.topk(eligible_scores, k=k, largest=True, sorted=False).indices
                    routed_rel = eligible_rel[routed_rel_local]
                    routed_rel_cpu = routed_rel.detach().cpu()
                    routed_mask[routed_rel] = True
                    routed_idx = batch_idx[routed_rel_cpu]
                    routed_labels = labels_t[routed_idx].to(self.device)
                    routed_forward = self._glance_refiner_forward(
                        refiner_model,
                        _refiner_feature_lookup(refiner_features, routed_idx, semantic_view_mode, self.device),
                        batch_base_logits=batch_base_logits[routed_rel],
                    )
                    routed_refiner_loss = F.cross_entropy(routed_forward["mixed_logits"], routed_labels, reduction="none")
                    routed_sample_weights = self._glance_refiner_sample_weights(
                        base_wrong_target[routed_rel],
                        utility_target[routed_rel],
                        refiner_weight_mode,
                        base_wrong_weight=refiner_base_wrong_weight,
                        utility_weight=refiner_utility_weight,
                    )
                else:
                    routed_rel = torch.zeros((0,), dtype=torch.long, device=self.device)
                    routed_labels = batch_labels[:0]
                    routed_forward = None
                    routed_refiner_loss = torch.zeros((0,), dtype=torch.float32, device=self.device)

                if refiner_target_mode == "keep_change":
                    gate_target_mode = str(refiner_gate_target or "base_wrong").lower()
                    if gate_target_mode == "utility_positive":
                        keep_change_target = utility_target[routed_rel].float()
                    elif gate_target_mode == "base_wrong":
                        keep_change_target = batch_base_pred[routed_rel].ne(routed_labels).float()
                    else:
                        raise ValueError(f"Unsupported joint_refiner_gate_target: {refiner_gate_target}")
                    refiner_gate_logits = routed_forward.get("gate_logits") if int(routed_rel.numel()) > 0 else None
                    if refiner_gate_logits is None:
                        if int(routed_rel.numel()) > 0:
                            raise ValueError("keep_change target mode requires a gated refiner.")
                        gate_loss = batch_base_loss.sum() * 0.0
                    else:
                        gate_loss = F.binary_cross_entropy_with_logits(
                            refiner_gate_logits,
                            keep_change_target,
                            weight=routed_sample_weights,
                        )
                    routed_refiner_loss = routed_refiner_loss * routed_sample_weights
                else:
                    gate_loss = batch_base_loss.sum() * 0.0
                    routed_refiner_loss = routed_refiner_loss * routed_sample_weights
                refiner_loss = batch_base_loss.detach().clone()
                refiner_loss[routed_mask] = routed_refiner_loss
                pred_loss = (
                    batch_base_loss[~routed_mask].sum()
                    + routed_refiner_loss.sum()
                ) / float(batch_labels.size(0))

                route_loss = batch_base_loss.sum() * 0.0
                router_regression_loss = batch_base_loss.sum() * 0.0
                if routing_protocol == "frozen_router_reuse":
                    router_ranking_loss = batch_base_loss.sum() * 0.0
                    router_selection_loss = batch_base_loss.sum() * 0.0
                else:
                    eligible_scores = batch_route_score[eligible_rel] if int(eligible_rel.numel()) > 0 else batch_route_score[:0]
                    eligible_targets = base_wrong_target[eligible_rel] if int(eligible_rel.numel()) > 0 else base_wrong_target[:0]
                    router_ranking_loss = _glance_pairwise_ranking_loss(eligible_scores, eligible_targets)
                    pos_count = int(eligible_targets.sum().detach().cpu().item())
                    neg_count = int(eligible_targets.numel() - pos_count)
                    if 0 < pos_count < int(eligible_targets.numel()):
                        pos_weight = torch.tensor(
                            float(neg_count / max(pos_count, 1)),
                            dtype=batch_route_score.dtype,
                            device=batch_route_score.device,
                        )
                        router_selection_loss = F.binary_cross_entropy_with_logits(
                            eligible_scores,
                            eligible_targets,
                            pos_weight=pos_weight,
                        )
                    else:
                        router_selection_loss = batch_route_score.sum() * 0.0
                utility_consistency_loss = batch_base_loss.sum() * 0.0
                total_loss = (
                    pred_loss
                    + float(refiner_gate_weight) * gate_loss
                    + float(router_ranking_weight) * router_ranking_loss
                    + float(router_calibration_weight) * router_selection_loss
                )

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

                train_loss_history.append(float(total_loss.detach().cpu().item()))
                train_pred_history.append(float(pred_loss.detach().cpu().item()))
                train_route_history.append(float(route_loss.detach().cpu().item()))
                train_router_regression_history.append(float(router_regression_loss.detach().cpu().item()))
                train_router_ranking_history.append(float(router_ranking_loss.detach().cpu().item()))
                train_router_selection_history.append(float(router_selection_loss.detach().cpu().item()))
                train_router_reliability_history.append(float(utility_consistency_loss.detach().cpu().item()))
                train_gate_history.append(float(gate_loss.detach().cpu().item()))
                train_routed_weight_history.append(float(routed_sample_weights.mean().detach().cpu().item()) if int(routed_sample_weights.numel()) > 0 else 0.0)
                train_adv_corr_history.append(
                    float(
                        _safe_score_corr(
                            batch_route_score[eligible_rel].detach().cpu().numpy() if int(eligible_rel.numel()) > 0 else np.asarray([], dtype=np.float32),
                            oracle_advantage[eligible_rel].detach().cpu().numpy() if int(eligible_rel.numel()) > 0 else np.asarray([], dtype=np.float32),
                        )
                        or 0.0
                    )
                )

            router_model.to("cpu")
            refiner_model.to("cpu")
            train_outputs = self._apply_glance_joint_policy(
                router_model,
                refiner_model,
                router_features,
                refiner_features,
                semantic_view_mode,
                logits_gnn,
                p_gnn,
                pred_gnn,
                labels_t,
                beta,
                train_idx,
                batch_size,
                eval_top_k,
            )
            valid_outputs = self._apply_glance_joint_policy(
                router_model,
                refiner_model,
                router_features,
                refiner_features,
                semantic_view_mode,
                logits_gnn,
                p_gnn,
                pred_gnn,
                labels_t,
                beta,
                valid_idx,
                batch_size,
                eval_top_k,
            )
            test_outputs = self._apply_glance_joint_policy(
                router_model,
                refiner_model,
                router_features,
                refiner_features,
                semantic_view_mode,
                logits_gnn,
                p_gnn,
                pred_gnn,
                labels_t,
                beta,
                test_idx,
                batch_size,
                eval_top_k,
            )
            valid_metrics = _classification_metrics_from_logits(
                valid_outputs["logits"],
                labels_t,
                torch.tensor(valid_idx, dtype=torch.long),
            )
            valid_metrics["routed_count"] = int(valid_outputs["routed_count"])
            valid_metrics["eligible_count"] = int(valid_outputs.get("eligible_count", 0))
            valid_metrics["query_rate"] = float(valid_outputs["routed_count"] / max(int(valid_idx.size), 1))
            train_metrics_epoch = _classification_metrics_from_logits(
                train_outputs["logits"],
                labels_t,
                torch.tensor(train_idx, dtype=torch.long),
            )
            train_metrics_epoch["routed_count"] = int(train_outputs["routed_count"])
            train_metrics_epoch["eligible_count"] = int(train_outputs.get("eligible_count", 0))
            train_metrics_epoch["query_rate"] = float(train_outputs["routed_count"] / max(int(train_idx.size), 1))
            train_router_diag_epoch = _safe_binary_score_metrics(train_wrong, train_outputs["router_score"][torch.tensor(train_idx, dtype=torch.long)].numpy())
            valid_router_diag_epoch = _safe_binary_score_metrics(valid_wrong, valid_outputs["router_score"][torch.tensor(valid_idx, dtype=torch.long)].numpy())
            test_router_diag_epoch = _safe_binary_score_metrics(test_wrong, test_outputs["router_score"][torch.tensor(test_idx, dtype=torch.long)].numpy())
            train_adv_diag_epoch = _safe_advantage_router_metrics(
                train_outputs["oracle_advantage"][torch.tensor(train_idx, dtype=torch.long)].numpy(),
                train_outputs["router_score"][torch.tensor(train_idx, dtype=torch.long)].numpy(),
                train_outputs["routed_mask"][torch.tensor(train_idx, dtype=torch.long)].numpy(),
            )
            valid_adv_diag_epoch = _safe_advantage_router_metrics(
                valid_outputs["oracle_advantage"][torch.tensor(valid_idx, dtype=torch.long)].numpy(),
                valid_outputs["router_score"][torch.tensor(valid_idx, dtype=torch.long)].numpy(),
                valid_outputs["routed_mask"][torch.tensor(valid_idx, dtype=torch.long)].numpy(),
            )
            test_adv_diag_epoch = _safe_advantage_router_metrics(
                test_outputs["oracle_advantage"][torch.tensor(test_idx, dtype=torch.long)].numpy(),
                test_outputs["router_score"][torch.tensor(test_idx, dtype=torch.long)].numpy(),
                test_outputs["routed_mask"][torch.tensor(test_idx, dtype=torch.long)].numpy(),
            )
            pred_gnn_labeled = _labeled_prefix_pred(pred_gnn.numpy(), labels_np.shape[0])
            train_pred_labeled = _labeled_prefix_pred(train_outputs["pred"].numpy(), labels_np.shape[0])
            valid_pred_labeled = _labeled_prefix_pred(valid_outputs["pred"].numpy(), labels_np.shape[0])
            test_pred_labeled = _labeled_prefix_pred(test_outputs["pred"].numpy(), labels_np.shape[0])
            train_delta_epoch = _delta_table(pred_gnn_labeled, train_pred_labeled, labels_np, _mask_from_idx(labels_np.shape[0], train_idx))
            valid_delta_epoch = _delta_table(pred_gnn_labeled, valid_pred_labeled, labels_np, _mask_from_idx(labels_np.shape[0], valid_idx))
            test_delta_epoch = _delta_table(pred_gnn_labeled, test_pred_labeled, labels_np, _mask_from_idx(labels_np.shape[0], test_idx))

            epoch_summary = {
                "epoch": int(epoch + 1),
                "beta": float(beta),
                "eval_top_k": int(eval_top_k),
                "train_top_k": int(train_k),
                "avg_total_loss": float(np.mean(train_loss_history)) if train_loss_history else 0.0,
                "avg_pred_loss": float(np.mean(train_pred_history)) if train_pred_history else 0.0,
                "avg_route_loss": float(np.mean(train_route_history)) if train_route_history else 0.0,
                "avg_router_regression_loss": float(np.mean(train_router_regression_history)) if train_router_regression_history else 0.0,
                "avg_router_ranking_loss": float(np.mean(train_router_ranking_history)) if train_router_ranking_history else 0.0,
                "avg_router_selection_loss": float(np.mean(train_router_selection_history)) if train_router_selection_history else 0.0,
                "avg_router_calibration_loss": float(np.mean(train_router_selection_history)) if train_router_selection_history else 0.0,
                "avg_router_reliability_loss": float(np.mean(train_router_reliability_history)) if train_router_reliability_history else 0.0,
                "avg_refiner_gate_loss": float(np.mean(train_gate_history)) if train_gate_history else 0.0,
                "avg_routed_sample_weight": float(np.mean(train_routed_weight_history)) if train_routed_weight_history else 0.0,
                "avg_router_advantage_corr": float(np.mean(train_adv_corr_history)) if train_adv_corr_history else 0.0,
                "train": train_metrics_epoch,
                "valid": valid_metrics,
                "router_diagnostics": {
                    "train": train_router_diag_epoch,
                    "valid": valid_router_diag_epoch,
                    "test": test_router_diag_epoch,
                },
                "advantage_router_diagnostics": {
                    "train": train_adv_diag_epoch,
                    "valid": valid_adv_diag_epoch,
                    "test": test_adv_diag_epoch,
                },
                "component_deltas": {
                    "train": train_delta_epoch,
                    "valid": valid_delta_epoch,
                    "test": test_delta_epoch,
                },
            }
            epoch_history.append(epoch_summary)
            epoch_component_curves.append({
                "epoch": int(epoch + 1),
                "beta": float(beta),
                "eval_top_k": int(eval_top_k),
                "train_top_k": int(train_k),
                "router": {
                    "train": train_router_diag_epoch,
                    "valid": valid_router_diag_epoch,
                    "test": test_router_diag_epoch,
                },
                "refiner": {
                    "train": {
                    "wrong_node_fix_rate": float(train_delta_epoch["fix"] / max(int((pred_gnn_labeled[train_idx] != labels_np[train_idx]).sum()), 1)),
                    "correct_node_break_rate": float(train_delta_epoch["broke"] / max(int(train_idx.size - (pred_gnn_labeled[train_idx] != labels_np[train_idx]).sum()), 1)),
                        "net_gain": int(train_delta_epoch["net"]),
                    },
                    "valid": {
                    "wrong_node_fix_rate": float(valid_delta_epoch["fix"] / max(int((pred_gnn_labeled[valid_idx] != labels_np[valid_idx]).sum()), 1)),
                    "correct_node_break_rate": float(valid_delta_epoch["broke"] / max(int(valid_idx.size - (pred_gnn_labeled[valid_idx] != labels_np[valid_idx]).sum()), 1)),
                        "net_gain": int(valid_delta_epoch["net"]),
                    },
                    "test": {
                    "wrong_node_fix_rate": float(test_delta_epoch["fix"] / max(int((pred_gnn_labeled[test_idx] != labels_np[test_idx]).sum()), 1)),
                    "correct_node_break_rate": float(test_delta_epoch["broke"] / max(int(test_idx.size - (pred_gnn_labeled[test_idx] != labels_np[test_idx]).sum()), 1)),
                        "net_gain": int(test_delta_epoch["net"]),
                    },
                },
            })
            if routing_protocol == "frozen_router_reuse":
                valid_budget_curve_epoch, _ = self._build_counterfactual_budget_rows(
                    budgets=[float(selected_budget_override)],
                    risk_score=valid_outputs["router_score"].numpy(),
                    pred_gnn=pred_gnn.numpy(),
                    pred_refiner=valid_outputs["pred"].numpy(),
                    labels_np=labels_np,
                    eval_idx=valid_idx,
                    prob_gnn=p_gnn,
                    prob_refiner=valid_outputs["prob"],
                )
                frozen_valid_row = valid_budget_curve_epoch[0] if valid_budget_curve_epoch else None
                current_score = (
                    float(frozen_valid_row["macro_f1"]) if frozen_valid_row else float(valid_metrics["macro_f1"]),
                    -float(frozen_valid_row["break"]) if frozen_valid_row else -float(valid_metrics["loss"]),
                )
                epoch_summary["frozen_router_valid_budget_metrics"] = frozen_valid_row
            else:
                current_score = (float(valid_metrics["macro_f1"]), -float(valid_metrics["loss"]))
            if current_score > best_score:
                best_score = current_score
                best_state = {
                    "router": {key: value.detach().cpu().clone() for key, value in router_model.state_dict().items()},
                    "refiner": {key: value.detach().cpu().clone() for key, value in refiner_model.state_dict().items()},
                }
                best_epoch_summary = epoch_summary
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

            router_model.to(self.device)
            refiner_model.to(self.device)

        if best_state is None or best_epoch_summary is None:
            raise MissingFrozenArtifactError("glance_joint_router_refine failed to produce a valid checkpoint.")

        router_model.load_state_dict(best_state["router"])
        refiner_model.load_state_dict(best_state["refiner"])
        router_model.to("cpu")
        refiner_model.to("cpu")
        if routing_protocol == "frozen_router_reuse":
            frozen_budget = float(selected_budget_override)
            topk_train_outputs = self._apply_glance_global_budget_policy(
                router_model,
                refiner_model,
                router_features,
                refiner_features,
                semantic_view_mode,
                logits_gnn,
                p_gnn,
                pred_gnn,
                labels_t,
                beta,
                train_idx,
                batch_size,
                frozen_budget,
            )
            topk_valid_outputs = self._apply_glance_global_budget_policy(
                router_model,
                refiner_model,
                router_features,
                refiner_features,
                semantic_view_mode,
                logits_gnn,
                p_gnn,
                pred_gnn,
                labels_t,
                beta,
                valid_idx,
                batch_size,
                frozen_budget,
            )
            topk_test_outputs = self._apply_glance_global_budget_policy(
                router_model,
                refiner_model,
                router_features,
                refiner_features,
                semantic_view_mode,
                logits_gnn,
                p_gnn,
                pred_gnn,
                labels_t,
                beta,
                test_idx,
                batch_size,
                frozen_budget,
            )
        else:
            topk_train_outputs = self._apply_glance_joint_policy(
                router_model,
                refiner_model,
                router_features,
                refiner_features,
                semantic_view_mode,
                logits_gnn,
                p_gnn,
                pred_gnn,
                labels_t,
                beta,
                train_idx,
                batch_size,
                eval_top_k,
            )
            topk_valid_outputs = self._apply_glance_joint_policy(
                router_model,
                refiner_model,
                router_features,
                refiner_features,
                semantic_view_mode,
                logits_gnn,
                p_gnn,
                pred_gnn,
                labels_t,
                beta,
                valid_idx,
                batch_size,
                eval_top_k,
            )
            topk_test_outputs = self._apply_glance_joint_policy(
                router_model,
                refiner_model,
                router_features,
                refiner_features,
                semantic_view_mode,
                logits_gnn,
                p_gnn,
                pred_gnn,
                labels_t,
                beta,
                test_idx,
                batch_size,
                eval_top_k,
            )

        topk_train_metrics = _classification_metrics_from_logits(
            topk_train_outputs["logits"],
            labels_t,
            torch.tensor(train_idx, dtype=torch.long),
        )
        topk_valid_metrics = _classification_metrics_from_logits(
            topk_valid_outputs["logits"],
            labels_t,
            torch.tensor(valid_idx, dtype=torch.long),
        )
        topk_test_metrics = _classification_metrics_from_logits(
            topk_test_outputs["logits"],
            labels_t,
            torch.tensor(test_idx, dtype=torch.long),
        )
        for split_metrics, split_outputs, split_idx in (
            (topk_train_metrics, topk_train_outputs, train_idx),
            (topk_valid_metrics, topk_valid_outputs, valid_idx),
            (topk_test_metrics, topk_test_outputs, test_idx),
        ):
            split_metrics["routed_count"] = int(split_outputs["routed_count"])
            split_metrics["eligible_count"] = int(split_outputs.get("eligible_count", 0))
            split_metrics["query_rate"] = float(split_outputs["routed_count"] / max(int(split_idx.size), 1))
            split_metrics.update(self._glance_gate_output_metrics(split_outputs, split_idx))

        topk_train_rows = self._build_glance_joint_per_node_rows(
            train_idx,
            pred_gnn,
            topk_train_outputs["pred"],
            labels_np,
            topk_train_outputs["routed_mask"],
            topk_train_outputs["router_prob"],
            topk_train_outputs["router_score"],
            topk_train_outputs["oracle_advantage"],
            topk_train_outputs.get("gate_prob"),
            topk_train_outputs.get("mpe_gate_weights"),
            semantic_views,
            "glance_joint_router_refine_train",
            eligible_refiner_mask=topk_train_outputs.get("eligible_refiner_mask"),
            mpe_selected_expert=topk_train_outputs.get("mpe_selected_expert"),
            selector_component_order=topk_train_outputs.get("selector_component_order"),
        )
        topk_valid_rows = self._build_glance_joint_per_node_rows(
            valid_idx,
            pred_gnn,
            topk_valid_outputs["pred"],
            labels_np,
            topk_valid_outputs["routed_mask"],
            topk_valid_outputs["router_prob"],
            topk_valid_outputs["router_score"],
            topk_valid_outputs["oracle_advantage"],
            topk_valid_outputs.get("gate_prob"),
            topk_valid_outputs.get("mpe_gate_weights"),
            semantic_views,
            "glance_joint_router_refine_valid",
            eligible_refiner_mask=topk_valid_outputs.get("eligible_refiner_mask"),
            mpe_selected_expert=topk_valid_outputs.get("mpe_selected_expert"),
            selector_component_order=topk_valid_outputs.get("selector_component_order"),
        )
        topk_test_rows = self._build_glance_joint_per_node_rows(
            test_idx,
            pred_gnn,
            topk_test_outputs["pred"],
            labels_np,
            topk_test_outputs["routed_mask"],
            topk_test_outputs["router_prob"],
            topk_test_outputs["router_score"],
            topk_test_outputs["oracle_advantage"],
            topk_test_outputs.get("gate_prob"),
            topk_test_outputs.get("mpe_gate_weights"),
            semantic_views,
            "glance_joint_router_refine_test",
            eligible_refiner_mask=topk_test_outputs.get("eligible_refiner_mask"),
            mpe_selected_expert=topk_test_outputs.get("mpe_selected_expert"),
            selector_component_order=topk_test_outputs.get("selector_component_order"),
        )
        train_wrong = (pred_gnn.numpy()[train_idx] != labels_np[train_idx]).astype(np.int64)
        valid_wrong = (pred_gnn.numpy()[valid_idx] != labels_np[valid_idx]).astype(np.int64)
        test_wrong = (pred_gnn.numpy()[test_idx] != labels_np[test_idx]).astype(np.int64)
        topk_train_router_diag = _safe_binary_score_metrics(train_wrong, topk_train_outputs["router_score"][torch.tensor(train_idx, dtype=torch.long)].numpy())
        topk_valid_router_diag = _safe_binary_score_metrics(valid_wrong, topk_valid_outputs["router_score"][torch.tensor(valid_idx, dtype=torch.long)].numpy())
        topk_test_router_diag = _safe_binary_score_metrics(test_wrong, topk_test_outputs["router_score"][torch.tensor(test_idx, dtype=torch.long)].numpy())
        topk_train_adv_diag = _safe_advantage_router_metrics(
            topk_train_outputs["oracle_advantage"][torch.tensor(train_idx, dtype=torch.long)].numpy(),
            topk_train_outputs["router_score"][torch.tensor(train_idx, dtype=torch.long)].numpy(),
            topk_train_outputs["routed_mask"][torch.tensor(train_idx, dtype=torch.long)].numpy(),
        )
        topk_valid_adv_diag = _safe_advantage_router_metrics(
            topk_valid_outputs["oracle_advantage"][torch.tensor(valid_idx, dtype=torch.long)].numpy(),
            topk_valid_outputs["router_score"][torch.tensor(valid_idx, dtype=torch.long)].numpy(),
            topk_valid_outputs["routed_mask"][torch.tensor(valid_idx, dtype=torch.long)].numpy(),
        )
        topk_test_adv_diag = _safe_advantage_router_metrics(
            topk_test_outputs["oracle_advantage"][torch.tensor(test_idx, dtype=torch.long)].numpy(),
            topk_test_outputs["router_score"][torch.tensor(test_idx, dtype=torch.long)].numpy(),
            topk_test_outputs["routed_mask"][torch.tensor(test_idx, dtype=torch.long)].numpy(),
        )
        component_curve_summary = {
            "best_epoch": int(best_epoch_summary["epoch"]),
            "curve": [
                {
                    "epoch": int(item["epoch"]),
                    "beta": float(item["beta"]),
                    "train_top_k": int(item["train_top_k"]),
                    "eval_top_k": int(item["eval_top_k"]),
                    "router_train_auroc": item["router"]["train"].get("auroc"),
                    "router_valid_auroc": item["router"]["valid"].get("auroc"),
                    "router_test_auroc": item["router"]["test"].get("auroc"),
                    "router_valid_adv_auroc": item.get("advantage_router_diagnostics", {}).get("valid", {}).get("auroc"),
                    "router_test_adv_auroc": item.get("advantage_router_diagnostics", {}).get("test", {}).get("auroc"),
                    "router_valid_auprc": item["router"]["valid"].get("auprc"),
                    "router_valid_adv_auprc": item.get("advantage_router_diagnostics", {}).get("valid", {}).get("auprc"),
                    "refiner_valid_fix_rate": item["refiner"]["valid"].get("wrong_node_fix_rate"),
                    "refiner_valid_break_rate": item["refiner"]["valid"].get("correct_node_break_rate"),
                    "refiner_test_fix_rate": item["refiner"]["test"].get("wrong_node_fix_rate"),
                    "refiner_test_break_rate": item["refiner"]["test"].get("correct_node_break_rate"),
                }
                for item in epoch_component_curves
            ],
        }

        budget_grid = [float(selected_budget_override)] if routing_protocol == "frozen_router_reuse" else self._router_budgets()
        valid_budget_curve, valid_budget_payloads = self._build_counterfactual_budget_rows(
            budgets=budget_grid,
            risk_score=topk_valid_outputs["router_score"].numpy(),
            pred_gnn=pred_gnn.numpy(),
            pred_refiner=topk_valid_outputs["pred"].numpy(),
            labels_np=labels_np,
            eval_idx=valid_idx,
            prob_gnn=p_gnn,
            prob_refiner=topk_valid_outputs["prob"],
        )
        test_budget_curve, test_budget_payloads = self._build_counterfactual_budget_rows(
            budgets=budget_grid,
            risk_score=topk_test_outputs["router_score"].numpy(),
            pred_gnn=pred_gnn.numpy(),
            pred_refiner=topk_test_outputs["pred"].numpy(),
            labels_np=labels_np,
            eval_idx=test_idx,
            prob_gnn=p_gnn,
            prob_refiner=topk_test_outputs["prob"],
        )
        selected_valid_row, selected_budget_key, _ = self._select_best_budget_row(valid_budget_curve)
        selected_test_row = {str(self._budget_key(item["budget"])): item for item in test_budget_curve}.get(selected_budget_key)

        fit_summary = {
            "best_epoch": int(best_epoch_summary["epoch"]),
            "best_valid": best_epoch_summary["valid"],
            "best_train": best_epoch_summary["train"],
            "epoch_history": epoch_history,
            "component_curve_summary": component_curve_summary,
            "router_diagnostics": {
                "train": topk_train_router_diag,
                "valid": topk_valid_router_diag,
                "test": topk_test_router_diag,
            },
            "advantage_router_diagnostics": {
                "train": topk_train_adv_diag,
                "valid": topk_valid_adv_diag,
                "test": topk_test_adv_diag,
            },
            "topk_reference": {
                "router_diagnostics": {
                    "train": topk_train_router_diag,
                    "valid": topk_valid_router_diag,
                    "test": topk_test_router_diag,
                },
                "advantage_router_diagnostics": {
                    "train": topk_train_adv_diag,
                    "valid": topk_valid_adv_diag,
                    "test": topk_test_adv_diag,
                },
            },
        }
        return {
            "beta": float(beta),
            "best_epoch": int(best_epoch_summary["epoch"]),
            "router_state": best_state["router"],
            "refiner_state": best_state["refiner"],
            "fit_summary": fit_summary,
            "train_outputs": topk_train_outputs,
            "valid_outputs": topk_valid_outputs,
            "test_outputs": topk_test_outputs,
            "selected_budget": float(selected_valid_row["budget"]) if selected_valid_row else float(self._router_budgets()[0]),
            "selected_budget_key": selected_budget_key,
            "selected_budget_metrics": {
                "train": topk_train_metrics,
                "valid": topk_valid_metrics,
                "test": topk_test_metrics,
            },
            "selected_valid_budget_metrics": selected_valid_row,
            "selected_test_budget_metrics_under_valid_choice": selected_test_row,
            "train_rows": topk_train_rows,
            "valid_rows": topk_valid_rows,
            "test_rows": topk_test_rows,
            "valid_budget_curve": valid_budget_curve,
            "test_budget_curve": test_budget_curve,
            "eval_top_k": int(eval_top_k),
            "routing_protocol": routing_protocol,
        }
